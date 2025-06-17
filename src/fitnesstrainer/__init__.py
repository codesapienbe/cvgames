import cv2
import mediapipe as mp
import time
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Squat counter variables
rep_count = 0
squat_down = False

# Angle calculation
def calculate_angle(a, b, c):
    # a, b, c are [x, y]
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    cos_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (math.hypot(*ba)*math.hypot(*bc) + 1e-6)
    angle = math.degrees(math.acos(max(min(cos_angle, 1.0), -1.0)))
    return angle

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    feedback = ""
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # Get coordinates
        hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        # Calculate angle
        angle = calculate_angle(hip, knee, ankle)
        # Draw landmarks
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Squat logic
        if angle < 70:
            feedback = "Go up"
            if not squat_down:
                squat_down = True
        if angle > 160:
            feedback = "Go down"
            if squat_down:
                rep_count += 1
                squat_down = False

        # Display angle
        cv2.putText(frame, f"Knee Angle: {int(angle)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display reps and feedback
    cv2.putText(frame, f"Reps: {rep_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if feedback:
        cv2.putText(frame, feedback, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Quit instruction
    cv2.putText(frame, "Press 'q' to quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("Fitness Trainer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
