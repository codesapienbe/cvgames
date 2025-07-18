import cv2
import mediapipe as mp
import numpy as np
import random
import time

class DriveSimulator:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.car_x = width // 2
        self.car_y = height - 100
        self.car_width = 60
        self.car_height = 100
        self.road_width = 400
        self.speed = 10
        self.score = 0
        self.game_over = False
        self.obstacles = []
        self.last_obstacle = time.time()
        self.obstacle_interval = 2.0

    def update(self, steer):
        self.car_x += int(steer * 20)
        self.car_x = max(self.width//2 - self.road_width//2 + self.car_width//2, min(self.width//2 + self.road_width//2 - self.car_width//2, self.car_x))
        # Move obstacles
        for obs in self.obstacles:
            obs['y'] += self.speed
        # Remove passed obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['y'] < self.height]
        # Add new obstacles
        if time.time() - self.last_obstacle > self.obstacle_interval:
            lane = random.randint(0, 2)
            obs_x = self.width//2 - self.road_width//2 + (lane + 0.5) * (self.road_width//3)
            self.obstacles.append({'x': int(obs_x), 'y': -50})
            self.last_obstacle = time.time()
        # Check collisions
        for obs in self.obstacles:
            if abs(self.car_x - obs['x']) < self.car_width and abs(self.car_y - obs['y']) < self.car_height:
                self.game_over = True
        if not self.game_over:
            self.score += 1

    def draw(self, frame):
        # Draw road
        cv2.rectangle(frame, (self.width//2 - self.road_width//2, 0), (self.width//2 + self.road_width//2, self.height), (50, 50, 50), -1)
        # Draw car
        cv2.rectangle(frame, (self.car_x - self.car_width//2, self.car_y - self.car_height//2), (self.car_x + self.car_width//2, self.car_y + self.car_height//2), (0, 255, 0), -1)
        # Draw obstacles
        for obs in self.obstacles:
            cv2.rectangle(frame, (obs['x'] - 30, obs['y'] - 30), (obs['x'] + 30, obs['y'] + 30), (0, 0, 255), -1)
        # Draw score
        cv2.putText(frame, f"Score: {self.score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if self.game_over:
            cv2.putText(frame, "Game Over! Press r to restart or q to quit", (100, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Steer with both hands like a wheel!", (50, self.height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def detect_steer(results):
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        y0 = results.multi_hand_landmarks[0].landmark[0].y
        y1 = results.multi_hand_landmarks[1].landmark[0].y
        steer = y0 - y1
        return np.clip(steer, -1, 1)
    return 0

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    game = DriveSimulator()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Drive Simulator", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Drive Simulator", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (game.width, game.height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        steer = detect_steer(results)
        if not game.game_over:
            game.update(steer)
        game.draw(frame)
        cv2.imshow("Drive Simulator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            game = DriveSimulator()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 