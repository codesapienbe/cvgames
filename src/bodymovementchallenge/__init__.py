import cv2
import mediapipe as mp
import time
import random
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Screen dimensions
disp_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
disp_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Game settings
gap_width = int(disp_w * 0.3)
bar_height = 20
speed = 200  # pixels per second
spawn_interval = 2.0  # seconds
last_spawn = time.time()
obstacles = []  # each: {'y', 'gap_x'}
player_y = int(disp_h * 0.8)
score = 0
start_time = time.time()
prev_time = time.time()

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    now = time.time()
    dt = now - prev_time
    prev_time = now

    # Spawn obstacles
    if now - last_spawn > spawn_interval:
        gap_x = random.randint(gap_width//2, disp_w - gap_width//2)
        obstacles.append({'y': -bar_height, 'gap_x': gap_x})
        last_spawn = now

    # Move obstacles
    for obs in list(obstacles):
        obs['y'] += int(speed * dt)
        # Check pass
        if obs['y'] > disp_h:
            obstacles.remove(obs)
            score += 1

    # Draw obstacles
    for obs in obstacles:
        y = obs['y']
        gx = obs['gap_x']
        # left bar
        cv2.rectangle(frame, (0, y), (gx - gap_width//2, y + bar_height), (255, 255, 255), -1)
        # right bar
        cv2.rectangle(frame, (gx + gap_width//2, y), (disp_w, y + bar_height), (255, 255, 255), -1)

    # Detect player position via hips
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    player_x = disp_w // 2
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        player_x = int(((lh.x + rh.x)/2) * disp_w)
        # Draw skeleton
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Draw player indicator
    cv2.circle(frame, (player_x, player_y), 10, (0, 255, 0), -1)

    # Display score and time
    elapsed_time = now - start_time
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {elapsed_time:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Body Movement Challenge', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
