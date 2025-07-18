import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Tetris shapes
SHAPES = [
    np.array([[1, 1, 1, 1]]),
    np.array([[1, 1], [1, 1]]),
    np.array([[0, 1, 0], [1, 1, 1]]),
    np.array([[1, 1, 0], [0, 1, 1]]),
    np.array([[0, 1, 1], [1, 1, 0]]),
    np.array([[1, 0, 0], [1, 1, 1]]),
    np.array([[0, 0, 1], [1, 1, 1]])
]

COLORS = [
    (0, 255, 255), (255, 255, 0), (128, 0, 128),
    (0, 255, 0), (255, 0, 0), (255, 165, 0), (0, 0, 255)
]

class Tetris:
    def __init__(self, rows=20, cols=10):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.score = 0
        self.game_over = False
        self.spawn_piece()

    def spawn_piece(self):
        self.shape_idx = random.randint(0, len(SHAPES)-1)
        self.shape = SHAPES[self.shape_idx].copy()
        self.color = COLORS[self.shape_idx]
        self.pos = [0, self.cols // 2 - self.shape.shape[1] // 2]
        if self.collision(self.shape, self.pos):
            self.game_over = True

    def collision(self, shape, pos):
        for i in range(shape.shape[0]):
            for j in range(shape.shape[1]):
                if shape[i, j]:
                    x, y = pos[0] + i, pos[1] + j
                    if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
                        return True
                    if self.board[x, y]:
                        return True
        return False

    def lock_piece(self):
        for i in range(self.shape.shape[0]):
            for j in range(self.shape.shape[1]):
                if self.shape[i, j]:
                    x, y = self.pos[0] + i, self.pos[1] + j
                    self.board[x, y] = self.shape_idx + 1
        self.clear_lines()
        self.spawn_piece()

    def clear_lines(self):
        lines = 0
        for i in range(self.rows):
            if all(self.board[i, :] != 0):
                self.board[1:i+1] = self.board[0:i]
                self.board[0] = 0
                lines += 1
        self.score += lines * 100

    def move(self, dx):
        new_pos = [self.pos[0], self.pos[1] + dx]
        if not self.collision(self.shape, new_pos):
            self.pos = new_pos

    def drop(self):
        new_pos = [self.pos[0] + 1, self.pos[1]]
        if not self.collision(self.shape, new_pos):
            self.pos = new_pos
        else:
            self.lock_piece()

    def rotate(self):
        new_shape = np.rot90(self.shape)
        if not self.collision(new_shape, self.pos):
            self.shape = new_shape

    def draw(self, frame, cell_size=30, offset=(50, 50)):
        # Draw board
        for i in range(self.rows):
            for j in range(self.cols):
                color = (50, 50, 50) if self.board[i, j] == 0 else COLORS[self.board[i, j]-1]
                cv2.rectangle(frame, (offset[0]+j*cell_size, offset[1]+i*cell_size),
                              (offset[0]+(j+1)*cell_size, offset[1]+(i+1)*cell_size), color, -1)
                cv2.rectangle(frame, (offset[0]+j*cell_size, offset[1]+i*cell_size),
                              (offset[0]+(j+1)*cell_size, offset[1]+(i+1)*cell_size), (200, 200, 200), 1)
        # Draw current piece
        for i in range(self.shape.shape[0]):
            for j in range(self.shape.shape[1]):
                if self.shape[i, j]:
                    x = offset[0] + (self.pos[1]+j)*cell_size
                    y = offset[1] + (self.pos[0]+i)*cell_size
                    cv2.rectangle(frame, (x, y), (x+cell_size, y+cell_size), self.color, -1)
                    cv2.rectangle(frame, (x, y), (x+cell_size, y+cell_size), (255, 255, 255), 2)
        # Draw score
        cv2.putText(frame, f"Score: {self.score}", (offset[0], offset[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if self.game_over:
            cv2.putText(frame, "Game Over! Press r to restart or q to quit", (offset[0], offset[1]+self.rows*cell_size//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Hand: Left/Right=Move, Up=Rotate, Down=Drop", (offset[0], offset[1]+self.rows*cell_size+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def detect_gesture(hand_landmarks, prev_landmarks, threshold=0.1):
    if prev_landmarks is None:
        return None
    dx = hand_landmarks.landmark[8].x - prev_landmarks.landmark[8].x
    dy = hand_landmarks.landmark[8].y - prev_landmarks.landmark[8].y
    if abs(dx) > abs(dy):
        if dx > threshold:
            return 'right'
        elif dx < -threshold:
            return 'left'
    else:
        if dy < -threshold:
            return 'up'
        elif dy > threshold:
            return 'down'
    return None

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    game = Tetris()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Tetris Twist", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Tetris Twist", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    prev_landmarks = None
    last_gesture_time = 0
    gesture_cooldown = 0.3
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (600, 800))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        now = time.time()
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if prev_landmarks and now - last_gesture_time > gesture_cooldown and not game.game_over:
                gesture = detect_gesture(hand_landmarks, prev_landmarks)
                if gesture == 'left':
                    game.move(-1)
                    last_gesture_time = now
                elif gesture == 'right':
                    game.move(1)
                    last_gesture_time = now
                elif gesture == 'up':
                    game.rotate()
                    last_gesture_time = now
                elif gesture == 'down':
                    game.drop()
                    last_gesture_time = now
            prev_landmarks = hand_landmarks
        if not game.game_over:
            if now - last_gesture_time > 0.7:
                game.drop()
                last_gesture_time = now
        game.draw(frame)
        cv2.imshow("Tetris Twist", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            game = Tetris()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 