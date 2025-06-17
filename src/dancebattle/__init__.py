import cv2
import mediapipe as mp
import time
import random

# Define simple dance moves
moves = ["hands_up", "hands_side", "touch_toe"]

# Function to check if a move is performed
def check_move(move, landmarks):
    try:
        lw = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        rw = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
        ls = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        la = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
    except Exception:
        return False
    # All coords normalized [0,1]
    if move == "hands_up":
        return lw.y < ls.y and rw.y < rs.y
    elif move == "hands_side":
        return abs(lw.y - ls.y) < 0.1 and abs(rw.y - rs.y) < 0.1 and lw.x < ls.x and rw.x > rs.x
    elif move == "touch_toe":
        return lw.y > lh.y and abs(lw.y - la.y) < 0.1 and rw.y > lh.y and abs(rw.y - la.y) < 0.1
    return False

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Create fullscreen window
cv2.namedWindow("Dance Battle", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Dance Battle", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

score = 0
current_move = random.choice(moves)
match_start = 0
matched = False

# Main loop (runs on import)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks
        if check_move(current_move, lm):
            if not matched:
                score += 1
                matched = True
                match_start = time.time()
        else:
            matched = False

    # After holding a successful match, pick next move
    if matched and (time.time() - match_start) > 2:
        current_move = random.choice(moves)
        matched = False
        match_start = time.time()

    # Display UI
    cv2.putText(frame, f"Move: {current_move}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Score: {score}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Dance Battle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
