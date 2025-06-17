import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Screen dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Road parameters
lane_margin = int(width * 0.2)
line_color = (255, 255, 255)
dash_length = 30
dash_gap = 30
line_offset = 0
road_speed = 200  # pixels per second

# Car parameters
car_width, car_height = 50, 80
car_y = int(height * 0.75)
car_x = width // 2
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

    # Draw road background
    # Lane boundaries
    left = lane_margin
    right = width - lane_margin
    cv2.line(frame, (left, 0), (left, height), line_color, 2)
    cv2.line(frame, (right, 0), (right, height), line_color, 2)
    # Center dashed line
    line_offset = (line_offset + road_speed * dt) % (dash_length + dash_gap)
    cx = width // 2
    y = -int(line_offset)
    while y < height:
        cv2.line(frame, (cx, y), (cx, y + dash_length), line_color, 2)
        y += dash_length + dash_gap

    # Hand detection for steering
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        # Use wrist x-position for steering
        wrist = hand.landmark[0]
        car_x = int(wrist.x * width)
        # Constrain within road
        car_x = max(left + car_width//2, min(right - car_width//2, car_x))

    # Draw car as rectangle
    x1 = car_x - car_width // 2
    y1 = car_y - car_height // 2
    x2 = car_x + car_width // 2
    y2 = car_y + car_height // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)

    # UI overlay
    cv2.putText(frame, "Drive Simulator", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
    cv2.putText(frame, "Steer by moving your hand horizontally", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("Drive Simulator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
