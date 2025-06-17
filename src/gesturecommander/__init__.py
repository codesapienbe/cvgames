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

# Get frame dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ship initial state
ship_x, ship_y = width/2, height/2
vel = 0.0
acceleration = 200.0  # pixels per second^2
friction = 0.98
max_speed = 600.0
# Initial direction vector pointing to the right
dir_x, dir_y = 1.0, 0.0

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

    # Process hand for gesture and direction
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    thrust = False
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        # Count extended fingers (index, middle, ring, pinky)
        tips = [8, 12, 16, 20]
        count = 0
        for tip in tips:
            if hand.landmark[tip].y < hand.landmark[tip-2].y:
                count += 1
        # Open palm if 3 or more fingers extended
        if count >= 3:
            thrust = True
        # Update direction from wrist (0) to index tip (8)
        dx = hand.landmark[8].x - hand.landmark[0].x
        dy = hand.landmark[8].y - hand.landmark[0].y
        norm = np.hypot(dx, dy)
        if norm > 0.01:
            dir_x, dir_y = dx / norm, dy / norm
    # Update velocity
    if thrust:
        vel += acceleration * dt
    vel *= friction
    vel = max(0.0, min(vel, max_speed))
    # Update position
    ship_x += dir_x * vel * dt
    ship_y += dir_y * vel * dt
    # Wrap around edges
    if ship_x < 0: ship_x = width
    if ship_x > width: ship_x = 0
    if ship_y < 0: ship_y = height
    if ship_y > height: ship_y = 0

    # Draw ship (circle + heading line)
    cx, cy = int(ship_x), int(ship_y)
    cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 2)
    fx = int(cx + dir_x * 25)
    fy = int(cy + dir_y * 25)
    cv2.line(frame, (cx, cy), (fx, fy), (255, 255, 255), 2)

    # UI overlays
    cv2.putText(frame, "Open palm to thrust", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Gesture Commander", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
