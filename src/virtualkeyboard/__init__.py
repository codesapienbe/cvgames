import cv2
import mediapipe as mp
import math
import time

# Keyboard layout (QWERTY)
keys = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["Z","X","C","V","B","N","M"]
]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Key dimensions
rows = len(keys)
cols = max(len(r) for r in keys)
key_w = width // cols
key_h = height // (rows + 1)  # reserve one row for typed text

pinch_thresh = 0.05  # threshold in normalized coords
pressed = False
typed_text = ""

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    idx_x, idx_y = 0, 0
    pinch = False
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        x_norm = hand.landmark[8].x
        y_norm = hand.landmark[8].y
        idx_x = int(x_norm * width)
        idx_y = int(y_norm * height)
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        # Pinch detection (thumb and index)
        thumb = hand.landmark[4]
        dist = math.hypot(thumb.x - x_norm, thumb.y - y_norm)
        pinch = dist < pinch_thresh

    hovered_key = None
    # Draw keyboard
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x1 = j * key_w
            y1 = (i + 1) * key_h
            x2 = x1 + key_w
            y2 = y1 + key_h
            color = (100, 100, 100)
            if idx_x >= x1 and idx_x < x2 and idx_y >= y1 and idx_y < y2:
                hovered_key = key
                color = (200, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, key, (x1 + key_w//3, y1 + int(key_h*0.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Handle typing
    if hovered_key and pinch and not pressed:
        typed_text += hovered_key
        pressed = True
    if not pinch:
        pressed = False

    # Display typed text
    cv2.putText(frame, f"Typed: {typed_text}", (10, int(key_h*0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Virtual Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
