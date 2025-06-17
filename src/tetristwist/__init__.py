import cv2
import mediapipe as mp
import time
import random

# Tetris grid settings
grid_cols, grid_rows = 10, 20
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cell_size = min(width//grid_cols, height//grid_rows)
board_w = cell_size * grid_cols
board_h = cell_size * grid_rows
origin_x = (width - board_w)//2
origin_y = (height - board_h)//2

# Colors
bg_color = (0, 0, 0)
grid_color = (50, 50, 50)
static_color = (0, 255, 255)
piece_color = (0, 200, 100)

# Initialize static grid
static = [[0]*grid_cols for _ in range(grid_rows)]

# Define single I-block with two rotations
rotations = [ [(-2,0),(-1,0),(0,0),(1,0)], [(0,-2),(0,-1),(0,0),(0,1)] ]
rotation = 0
# Spawn piece
px, py = grid_cols//2, 0

# Hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
last_x, last_y = None, None

# Pinch detection
rotate_cooldown = False
pinch_thresh = 0.05

# Game timing
last_time = time.time()
fall_interval = 0.5  # seconds per row
score = 0

# Function to check if piece fits
def can_move(nx, ny, rot):
    for dx, dy in rotations[rot]:
        x, y = nx+dx, ny+dy
        if x < 0 or x >= grid_cols or y < 0 or y >= grid_rows: return False
        if static[y][x]: return False
    return True

# Place piece into static grid and clear lines
def freeze():
    global score
    for dx, dy in rotations[rotation]:
        x, y = px+dx, py+dy
        static[y][x] = 1
    # Clear full lines
    new_static = [row for row in static if not all(row)]
    cleared = grid_rows - len(new_static)
    for _ in range(cleared): new_static.insert(0, [0]*grid_cols)
    for r in range(grid_rows): static[r] = new_static[r]
    score += cleared

# Main loop
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    now = time.time()
    dt = now - last_time

    # Hand landmark for control
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        x_norm = hand.landmark[8].x
        y_norm = hand.landmark[8].y
        if last_x is not None:
            dx = x_norm - last_x
            if dx > 0.05 and can_move(px+1, py, rotation): px += 1
            if dx < -0.05 and can_move(px-1, py, rotation): px -= 1
        # Pinch to rotate
        thumb = hand.landmark[4]
        dist = ((thumb.x - hand.landmark[8].x)**2 + (thumb.y - hand.landmark[8].y)**2)**0.5
        if dist < pinch_thresh and not rotate_cooldown:
            new_rot = (rotation+1) % len(rotations)
            if can_move(px, py, new_rot): rotation = new_rot
            rotate_cooldown = True
        if dist > pinch_thresh: rotate_cooldown = False
        last_x, last_y = x_norm, y_norm

    # Automatic fall
    if dt >= fall_interval:
        last_time = now
        if can_move(px, py+1, rotation):
            py += 1
        else:
            # Freeze and spawn new
            freeze()
            # Check game over (spawn collision)
            px, py, rotation = grid_cols//2, 0, 0
            if not can_move(px, py, rotation):
                static = [[0]*grid_cols for _ in range(grid_rows)]
                score = 0
            continue

    # Draw background
    frame[:] = bg_color
    # Draw static blocks
    for r in range(grid_rows):
        for c in range(grid_cols):
            if static[r][c]:
                x1 = origin_x + c*cell_size
                y1 = origin_y + r*cell_size
                cv2.rectangle(frame, (x1,y1), (x1+cell_size,y1+cell_size), static_color, -1)
    # Draw current piece
    for dx, dy in rotations[rotation]:
        x, y = px+dx, py+dy
        if 0 <= y < grid_rows:
            x1 = origin_x + x*cell_size
            y1 = origin_y + y*cell_size
            cv2.rectangle(frame, (x1,y1), (x1+cell_size,y1+cell_size), piece_color, -1)

    # Draw grid lines
    for r in range(grid_rows+1):
        y = origin_y + r*cell_size
        cv2.line(frame, (origin_x,y), (origin_x+board_w, y), grid_color, 1)
    for c in range(grid_cols+1):
        x = origin_x + c*cell_size
        cv2.line(frame, (x,origin_y), (x,origin_y+board_h), grid_color, 1)

    # UI overlays
    cv2.putText(frame, f"Tetris Twist - Score: {score}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Move hand to shift, pinch to rotate, 'q' to quit", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.imshow("Tetris Twist", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Cleanup
cap.release()
cv2.destroyAllWindows()
