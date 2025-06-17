import cv2
import mediapipe as mp
import time
import math

# Setup video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Initialize Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Frame dimensions and plank settings
tmp_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
tmp_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width, height = int(tmp_w), int(tmp_h)
center = (width // 2, int(height * 0.8))
plank_length = int(width * 0.6)

# Physics parameters
g = 500  # gravity factor
score = 0.0
ball_pos = 0.0  # along plank, center=0
ball_vel = 0.0
achievements = {10: False, 30: False, 60: False}
achievement_text = ""
achievement_time = 0
prev_time = time.time()

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    # Pose detection for shoulder tilt
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    angle = 0.0
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        dx = rs.x - ls.x
        dy = rs.y - ls.y
        angle = math.atan2(dy, dx)

    # Physics update
    acc = g * math.sin(angle)
    ball_vel += acc * dt
    ball_pos += ball_vel * dt
    half_len = plank_length / 2

    # Check for failure
    if abs(ball_pos) > half_len:
        cv2.putText(frame, f"Game Over! Score: {int(score)}", (50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Circus Performer", frame)
        cv2.waitKey(2000)
        # reset game
        score = 0.0
        ball_pos = 0.0
        ball_vel = 0.0
        achievements = {k: False for k in achievements}
        achievement_text = ""
        prev_time = time.time()
        continue

    # Update score
    score += dt

    # Achievement check
    for thresh, done in achievements.items():
        if score >= thresh and not done:
            achievements[thresh] = True
            achievement_text = f"Achievement: {thresh}s balanced!"
            achievement_time = current_time

    # Draw plank
    x1 = int(center[0] - half_len * math.cos(angle))
    y1 = int(center[1] - half_len * math.sin(angle))
    x2 = int(center[0] + half_len * math.cos(angle))
    y2 = int(center[1] + half_len * math.sin(angle))
    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 5)

    # Draw ball
    bx = int(center[0] + ball_pos * math.cos(angle))
    by = int(center[1] + ball_pos * math.sin(angle))
    cv2.circle(frame, (bx, by), 15, (0, 0, 255), -1)

    # UI overlays
    cv2.putText(frame, f"Score: {int(score)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if achievement_text and (current_time - achievement_time) < 2:
        cv2.putText(frame, achievement_text, (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)

    cv2.imshow("Circus Performer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
