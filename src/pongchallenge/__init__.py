import cv2
import mediapipe as mp
import numpy as np
import time
import random

class PongGame:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.paddle_width = 20
        self.paddle_height = 100
        self.ball_radius = 15
        self.reset()

    def reset(self):
        self.paddle_y = self.height // 2 - self.paddle_height // 2
        self.ai_paddle_y = self.height // 2 - self.paddle_height // 2
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = random.choice([-6, 6])
        self.ball_dy = random.choice([-4, 4])
        self.score = 0
        self.ai_score = 0
        self.game_over = False

    def update(self, hand_y):
        # Move player paddle
        self.paddle_y = int(hand_y - self.paddle_height // 2)
        self.paddle_y = max(0, min(self.height - self.paddle_height, self.paddle_y))
        # Move AI paddle
        if self.ball_y > self.ai_paddle_y + self.paddle_height // 2:
            self.ai_paddle_y += 5
        elif self.ball_y < self.ai_paddle_y + self.paddle_height // 2:
            self.ai_paddle_y -= 5
        self.ai_paddle_y = max(0, min(self.height - self.paddle_height, self.ai_paddle_y))
        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        # Collisions
        if self.ball_y < self.ball_radius or self.ball_y > self.height - self.ball_radius:
            self.ball_dy *= -1
        # Player paddle collision
        if (self.ball_x - self.ball_radius < self.paddle_width and
            self.paddle_y < self.ball_y < self.paddle_y + self.paddle_height):
            self.ball_dx *= -1
            self.score += 1
        # AI paddle collision
        if (self.ball_x + self.ball_radius > self.width - self.paddle_width and
            self.ai_paddle_y < self.ball_y < self.ai_paddle_y + self.paddle_height):
            self.ball_dx *= -1
            self.ai_score += 1
        # Out of bounds
        if self.ball_x < 0 or self.ball_x > self.width:
            self.game_over = True

    def draw(self, frame):
        # Draw paddles
        cv2.rectangle(frame, (0, self.paddle_y), (self.paddle_width, self.paddle_y + self.paddle_height), (0, 255, 0), -1)
        cv2.rectangle(frame, (self.width - self.paddle_width, self.ai_paddle_y), (self.width, self.ai_paddle_y + self.paddle_height), (0, 0, 255), -1)
        # Draw ball
        cv2.circle(frame, (self.ball_x, self.ball_y), self.ball_radius, (255, 255, 255), -1)
        # Draw scores
        cv2.putText(frame, f"You: {self.score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"AI: {self.ai_score}", (self.width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if self.game_over:
            cv2.putText(frame, "Game Over! Press r to restart or q to quit", (self.width//2 - 250, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Move your hand up/down to control the paddle", (50, self.height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def get_hand_y(results, frame_height):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        y = int(hand.landmark[8].y * frame_height)
        return y
    return frame_height // 2

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    game = PongGame()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Pong Challenge", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pong Challenge", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (game.width, game.height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hand_y = get_hand_y(results, game.height)
        if not game.game_over:
            game.update(hand_y)
        game.draw(frame)
        cv2.imshow("Pong Challenge", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            game.reset()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 