import cv2
import mediapipe as mp
import time
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Make window fullscreen
cv2.namedWindow("Drive Simulator", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Drive Simulator", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Screen dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Road parameters
lane_margin = int(width * 0.2)
line_color = (255, 255, 0)  # yellow road lines
dash_length = 30
dash_gap = 30
line_offset = 0
road_speed = 200  # pixels per second

# Car parameters
car_width, car_height = 50, 80
car_y = int(height * 0.75)
car_x = width // 2
prev_time = time.time()
steering_speed = 300  # pixels per second for steering
# Base hand size ratio to calibrate speed scaling
base_hand_ratio = None
# Obstacle cars, scoring, and levels
obstacles = []
spawn_interval = 2.0  # seconds between obstacle spawn
last_spawn = time.time()
score = 0
level = 1
score_threshold = 5

# Main loop runs on import
while True:
    ret, cam_frame = cap.read()
    if not ret:
        break
    # Time delta
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    # Prepare display canvas (black)
    display = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)  # dark grey background

    # Draw road background (white) on display
    left = lane_margin
    right = width - lane_margin
    cv2.line(display, (left, 0), (left, height), line_color, 2)
    cv2.line(display, (right, 0), (right, height), line_color, 2)
    line_offset = (line_offset + road_speed * dt) % (dash_length + dash_gap)
    cx = width // 2
    y = -int(line_offset)
    while y < height:
        cv2.line(display, (cx, y), (cx, y + dash_length), line_color, 2)
        y += dash_length + dash_gap

    # Spawn obstacles
    if current_time - last_spawn > spawn_interval:
        lane_width = (right - left) / 3.0
        lane_idx = random.randint(0, 2)
        obs_x = left + (lane_idx + 0.5) * lane_width
        obstacles.append({'x': obs_x, 'y': -car_height, 'speed': road_speed})
        last_spawn = current_time

    # Update and draw obstacles
    for obs in list(obstacles):
        obs['y'] += obs['speed'] * dt
        ox1 = int(obs['x'] - car_width/2)
        oy1 = int(obs['y'] - car_height/2)
        ox2 = int(obs['x'] + car_width/2)
        oy2 = int(obs['y'] + car_height/2)
        cv2.rectangle(display, (ox1, oy1), (ox2, oy2), (0, 0, 255), -1)  # red obstacles
        # Collision detection
        if abs(car_x - obs['x']) < car_width and abs(car_y - obs['y']) < car_height:
            # Game Over menu: retry or quit
            while True:
                cv2.putText(display, 'Game Over', (width//2-200, height//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                cv2.putText(display, 'Press R to Retry or Q to Quit', (width//2-300, height//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.imshow('Drive Simulator', display)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('r'):
                    # Reset game state
                    obstacles.clear()
                    spawn_interval = 2.0
                    last_spawn = time.time()
                    score = 0
                    level = 1
                    road_speed = 200
                    base_hand_ratio = None
                    car_x = width // 2
                    prev_time = time.time()
                    line_offset = 0
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)
            # Restart: exit obstacle loop and resume
        # Remove passed obstacles and increase score
        if obs['y'] - car_height/2 > height:
            obstacles.remove(obs)
            score += 1

    # Level progression
    if score >= level * score_threshold:
        level += 1
        spawn_interval = max(0.5, spawn_interval * 0.8)
        road_speed *= 1.2

    # Hand detection and steering wheel gesture logic
    detect_frame = cv2.flip(cam_frame, 1)
    rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    # Steering wheel gesture: two hands only
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        left_y = right_y = None
        ratios = []
        # Gather vertical positions and size ratios
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            wrist = hand_landmarks.landmark[0]
            wrist_y = wrist.y * height
            ys = [lm.y for lm in hand_landmarks.landmark]
            bbox_height = (max(ys) - min(ys)) * height
            ratios.append(bbox_height / height)
            if hand_label == 'Left':
                left_y = wrist_y
            else:
                right_y = wrist_y
        if left_y is not None and right_y is not None:
            # Calibrate base ratio on first detection
            avg_ratio = sum(ratios) / 2.0
            if base_hand_ratio is None:
                base_hand_ratio = avg_ratio
            # Speed scaling based on hand distance
            speed_factor = 1.0
            if avg_ratio > base_hand_ratio * 1.3:
                speed_factor = 1.5
            elif avg_ratio < base_hand_ratio * 0.7:
                speed_factor = 0.5
            current_speed = steering_speed * speed_factor
            # Sensitivity threshold and range
            threshold_y = height * 0.02
            sens_range = height * 0.3
            dy = left_y - right_y
            if abs(dy) > threshold_y:
                sensitivity = min((abs(dy) - threshold_y) / (sens_range - threshold_y), 1.0)
                steer_amount = current_speed * sensitivity * dt
                if dy < 0:
                    # Left hand higher -> steer right
                    car_x += steer_amount
                else:
                    # Right hand higher -> steer left
                    car_x -= steer_amount

    # Constrain car within road
    car_x = max(left + car_width//2, min(right - car_width//2, car_x))

    # Draw car as white rectangle on display
    x1 = int(car_x - car_width / 2)
    y1 = int(car_y - car_height / 2)
    x2 = int(car_x + car_width / 2)
    y2 = int(car_y + car_height / 2)
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 102, 204), -1)  # blue player car

    # Overlay webcam preview
    # Webcam preview
    preview = cv2.resize(detect_frame, (width//5, height//5))
    px, py = width - preview.shape[1] - 10, 10
    display[py:py+preview.shape[0], px:px+preview.shape[1]] = preview
    cv2.rectangle(display, (px, py), (px+preview.shape[1], py+preview.shape[0]), (0, 255, 0), 2)

    # UI overlay (white text)
    cv2.putText(display, "Drive Simulator", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,165,0), 2)  # orange title
    cv2.putText(display, f"Score: {score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)  # cyan score
    cv2.putText(display, f"Level: {level}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)  # green level
    cv2.putText(display, "Press 'q' to quit", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Drive Simulator", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
