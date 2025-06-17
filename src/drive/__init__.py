import cv2
import mediapipe as mp
import time
import numpy as np
import random
import os
import glob
import math

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

# Load resources directory and helper
resources_path = os.path.join(os.path.dirname(__file__), 'Resources')
wheel_img = None
obs_imgs = []
player_img = None
def overlay_img_alpha(bg, fg, pos):
    x, y = pos
    h, w = fg.shape[:2]
    if fg.shape[2] == 4:
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = fg[:, :, c] * alpha + bg[y:y+h, x:x+w, c] * (1 - alpha)
    else:
        bg[y:y+h, x:x+w] = fg

# Load steering wheel sprite
wheel_path = os.path.join(resources_path, 'steering_wheel.png')
if os.path.exists(wheel_path):
    wheel_img = cv2.imread(wheel_path, cv2.IMREAD_UNCHANGED)
# Load obstacle car sprites
for p in glob.glob(os.path.join(resources_path, 'obs_car*.png')):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is not None:
        obs_imgs.append(img)
# Load player car sprite
player_path = os.path.join(resources_path, 'player_car.png')
if os.path.exists(player_path):
    player_img = cv2.imread(player_path, cv2.IMREAD_UNCHANGED)

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
        obs = {'x': obs_x, 'y': -car_height, 'speed': road_speed}
        if obs_imgs:
            img = random.choice(obs_imgs)
            h_img, w_img = img.shape[:2]
            obs['img'] = img
            obs['w_img'] = w_img
            obs['h_img'] = h_img
        obstacles.append(obs)
        last_spawn = current_time

    # Update and draw obstacles
    for obs in list(obstacles):
        obs['y'] += obs['speed'] * dt
        if 'img' in obs:
            ox = int(obs['x'] - obs['w_img']/2)
            oy = int(obs['y'] - obs['h_img']/2)
            overlay_img_alpha(display, obs['img'], (ox, oy))
        else:
            ox1 = int(obs['x'] - car_width/2)
            oy1 = int(obs['y'] - car_height/2)
            ox2 = int(obs['x'] + car_width/2)
            oy2 = int(obs['y'] + car_height/2)
            cv2.rectangle(display, (ox1, oy1), (ox2, oy2), (0, 0, 255), -1)
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
    if player_img is not None:
        p_img = cv2.resize(player_img, (car_width, car_height))
        overlay_img_alpha(display, p_img, (x1, y1))
    else:
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 102, 204), -1)  # blue player car

    # Overlay steering wheel and hand positions (small, transparent, top-left)
    wheel_sz = width // 6
    ux, uy = 20, 20
    if wheel_img is not None:
        wheel = cv2.resize(wheel_img, (wheel_sz, wheel_sz), interpolation=cv2.INTER_AREA)
        # apply global transparency
        if wheel.shape[2] == 4:
            tmp = wheel.copy()
            alpha = (tmp[:, :, 3].astype(float) * 0.3).astype('uint8')
            tmp[:, :, 3] = alpha
            overlay_img_alpha(display, tmp, (ux, uy))
        else:
            roi = display[uy:uy+wheel_sz, ux:ux+wheel_sz].copy()
            resized = cv2.resize(wheel, (wheel_sz, wheel_sz), interpolation=cv2.INTER_AREA)
            roi[:] = resized
            display[uy:uy+wheel_sz, ux:ux+wheel_sz] = cv2.addWeighted(roi, 0.3, display[uy:uy+wheel_sz, ux:ux+wheel_sz], 0.7, 0)
    else:
        # fallback: transparent wheel with spokes
        roi = display[uy:uy+wheel_sz, ux:ux+wheel_sz].copy()
        center = (wheel_sz//2, wheel_sz//2)
        radius = wheel_sz//2 - 5
        cv2.circle(roi, center, radius, (200,200,200), 3)
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            x2 = center[0] + int(math.cos(angle) * radius)
            y2 = center[1] + int(math.sin(angle) * radius)
            cv2.line(roi, center, (x2, y2), (200,200,200), 2)
        display[uy:uy+wheel_sz, ux:ux+wheel_sz] = cv2.addWeighted(roi, 0.3, display[uy:uy+wheel_sz, ux:ux+wheel_sz], 0.7, 0)
    # draw hand positions on wheel area
    if results and results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            wpt = hl.landmark[0]
            hx = ux + int(wpt.x * wheel_sz)
            hy = uy + int(wpt.y * wheel_sz)
            cv2.circle(display, (hx, hy), 8, (0,255,0), -1)

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
