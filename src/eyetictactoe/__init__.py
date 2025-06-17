import cv2
import mediapipe as mp
import time
import random

# Initialize FaceMesh for nose tracking and blink detection
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Blink threshold (normalized)
BLINK_THRESHOLD = 0.02
prev_blink = False

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Game board: 0 empty, 1 X (player), -1 O (AI)
board = [[0]*3 for _ in range(3)]
game_over = False
winner_text = ""

# Check win condition
def check_win(bd, mark):
    # rows, cols
    for i in range(3):
        if all(bd[i][j]==mark for j in range(3)): return True
        if all(bd[j][i]==mark for j in range(3)): return True
    # diags
    if all(bd[i][i]==mark for i in range(3)): return True
    if all(bd[i][2-i]==mark for i in range(3)): return True
    return False

# Draw X/O in cell
def draw_mark(frame, mark, r, c, w, h):
    x = c*w + w//2
    y = r*h + h//2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'X' if mark==1 else 'O'
    cv2.putText(frame, text, (x-20, y+20), font, 2, (255,255,255), 3)

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    h, w = frame.shape[:2]

    # Process face mesh
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    nose_x = nose_y = None
    blink = False
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        # Nose tip
        nose = lm[1]
        nose_x, nose_y = int(nose.x*w), int(nose.y*h)
        # Blink detection (left eye)
        u = lm[159]; l = lm[145]
        if abs(u.y - l.y) < BLINK_THRESHOLD:
            blink = True

    # Draw 3x3 grid
    cell_w, cell_h = w//3, h//3
    for i in range(4):
        cv2.line(frame, (0, i*cell_h), (w, i*cell_h), (200,200,200), 2)
        cv2.line(frame, (i*cell_w, 0), (i*cell_w, h), (200,200,200), 2)

    # Highlight selected cell
    sel_r = sel_c = None
    if nose_x is not None:
        sel_c = min(2, nose_x // cell_w)
        sel_r = min(2, nose_y // cell_h)
        x1, y1 = sel_c*cell_w, sel_r*cell_h
        cv2.rectangle(frame,(x1,y1),(x1+cell_w,y1+cell_h),(0,255,0),3)

    # Handle player move on blink
    if blink and not prev_blink and not game_over and sel_r is not None:
        if board[sel_r][sel_c]==0:
            board[sel_r][sel_c] = 1
            # Check player win
            if check_win(board,1):
                game_over = True
                winner_text = "You win!"
            else:
                # AI random move
                empties = [(r,c) for r in range(3) for c in range(3) if board[r][c]==0]
                if empties:
                    r2,c2 = random.choice(empties)
                    board[r2][c2] = -1
                    if check_win(board,-1):
                        game_over = True
                        winner_text = "AI wins!"
                else:
                    game_over = True
                    winner_text = "Draw!"
    prev_blink = blink

    # Draw marks
    for r in range(3):
        for c in range(3):
            if board[r][c] != 0:
                draw_mark(frame, board[r][c], r, c, cell_w, cell_h)

    # If game over, display message
    if game_over:
        cv2.putText(frame, winner_text, (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
        cv2.putText(frame, "Press 'q' to quit", (w//4, h//2+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
    else:
        cv2.putText(frame, "Move head to select, blink to place", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("Eye Tic Tac Toe", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
