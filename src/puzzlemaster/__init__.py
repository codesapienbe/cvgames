import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Configuration
grid_size = 3
tile_size = 150
board_size = grid_size * tile_size

# Initialize grid (1 to n-1, 0 is empty)
grid = [[j + i * grid_size + 1 for j in range(grid_size)] for i in range(grid_size)]
grid[-1][-1] = 0

# Shuffle grid with random legal moves
def get_empty_pos():
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] == 0:
                return i, j

def shuffle_moves(n=100):
    moves = ['up', 'down', 'left', 'right']
    for _ in range(n):
        i, j = get_empty_pos()
        direction = random.choice(moves)
        if direction == 'up' and i < grid_size - 1:
            grid[i][j], grid[i+1][j] = grid[i+1][j], grid[i][j]
        if direction == 'down' and i > 0:
            grid[i][j], grid[i-1][j] = grid[i-1][j], grid[i][j]
        if direction == 'left' and j < grid_size - 1:
            grid[i][j], grid[i][j+1] = grid[i][j+1], grid[i][j]
        if direction == 'right' and j > 0:
            grid[i][j], grid[i][j-1] = grid[i][j-1], grid[i][j]

shuffle_moves()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Timing and gesture state
last_move_time = 0
cooldown = 0.5  # seconds
start_time = time.time()

# Capture setup
display = np.ones((board_size, board_size, 3), dtype=np.uint8) * 255
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    # Hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    now = time.time()

    # Gesture: one finger extended -> move
    gesture = None
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        # count extended fingers
        tips = [8, 12, 16, 20]
        count = 0
        for tip in tips:
            if hand.landmark[tip].y < hand.landmark[tip-2].y:
                count += 1
        if count == 1 and now - last_move_time > cooldown:
            # Determine direction
            dx = hand.landmark[8].x - hand.landmark[0].x
            dy = hand.landmark[8].y - hand.landmark[0].y
            if abs(dx) > abs(dy):
                gesture = 'right' if dx > 0 else 'left'
            else:
                gesture = 'down' if dy > 0 else 'up'
            # Move empty tile
            i, j = get_empty_pos()
            if gesture == 'up' and i < grid_size - 1:
                grid[i][j], grid[i+1][j] = grid[i+1][j], grid[i][j]
            if gesture == 'down' and i > 0:
                grid[i][j], grid[i-1][j] = grid[i-1][j], grid[i][j]
            if gesture == 'left' and j < grid_size - 1:
                grid[i][j], grid[i][j+1] = grid[i][j+1], grid[i][j]
            if gesture == 'right' and j > 0:
                grid[i][j], grid[i][j-1] = grid[i][j-1], grid[i][j]
            last_move_time = now

    # Draw board
    display.fill(255)
    for r in range(grid_size):
        for c in range(grid_size):
            x1 = c * tile_size
            y1 = r * tile_size
            x2 = x1 + tile_size
            y2 = y1 + tile_size
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 0), 2)
            val = grid[r][c]
            if val != 0:
                text = str(val)
                (w_text, h_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
                tx = x1 + (tile_size - w_text) // 2
                ty = y1 + (tile_size + h_text) // 2
                cv2.putText(display, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    # Timer
    elapsed = int(now - start_time)
    cv2.putText(display, f"Time: {elapsed}s", (10, board_size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Puzzle Master", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
