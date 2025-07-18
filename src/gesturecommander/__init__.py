import cv2
import mediapipe as mp
import numpy as np
import random
import time

class Spaceship:
    def __init__(self, width, height):
        self.x = width // 2
        self.y = height - 100
        self.width = width
        self.height = height
        self.score = 0
        self.game_over = False
        self.bullets = []
        self.aliens = []
        self.last_alien = time.time()
        self.last_shot = 0

    def move(self, dx):
        self.x = np.clip(self.x + dx, 50, self.width - 50)

    def shoot(self):
        now = time.time()
        if now - self.last_shot > 0.5:
            self.bullets.append({'x': self.x, 'y': self.y - 30})
            self.last_shot = now

    def update(self, gesture):
        if gesture == 'left':
            self.move(-20)
        elif gesture == 'right':
            self.move(20)
        elif gesture == 'shoot':
            self.shoot()
        # Move bullets
        for b in self.bullets:
            b['y'] -= 10
        self.bullets = [b for b in self.bullets if b['y'] > 0]
        # Spawn aliens
        if time.time() - self.last_alien > 1.5:
            self.aliens.append({'x': random.randint(50, self.width-50), 'y': 0})
            self.last_alien = time.time()
        # Move aliens
        for a in self.aliens:
            a['y'] += 5
        # Check collisions
        for b in self.bullets:
            for a in self.aliens:
                if abs(b['x'] - a['x']) < 20 and abs(b['y'] - a['y']) < 20:
                    self.score += 10
                    self.aliens.remove(a)
                    if b in self.bullets:
                        self.bullets.remove(b)
        # Game over if alien reaches bottom
        for a in self.aliens:
            if a['y'] > self.height - 50:
                self.game_over = True

    def draw(self, frame):
        # Draw spaceship
        cv2.rectangle(frame, (self.x-20, self.y-20), (self.x+20, self.y+20), (255,255,255), -1)
        # Draw bullets
        for b in self.bullets:
            cv2.circle(frame, (b['x'], b['y']), 5, (0,255,0), -1)
        # Draw aliens
        for a in self.aliens:
            cv2.circle(frame, (a['x'], a['y']), 20, (0,0,255), -1)
        # Draw score
        cv2.putText(frame, f"Score: {self.score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if self.game_over:
            cv2.putText(frame, "Game Over! Press r to restart or q to quit", (100, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            cv2.putText(frame, "Gestures: Left/Right=Move, Fist=Shoot", (50, self.height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def detect_gesture(hand_landmarks):
    if not hand_landmarks:
        return None
    # Left/right by index finger x movement, shoot by fist
    wrist = hand_landmarks.landmark[0]
    index = hand_landmarks.landmark[8]
    dx = index.x - wrist.x
    if dx > 0.1:
        return 'right'
    elif dx < -0.1:
        return 'left'
    # Fist (all fingers down)
    tips = [8, 12, 16, 20]
    if all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip-2].y for tip in tips):
        return 'shoot'
    return None

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    width, height = 800, 600
    game = Spaceship(width, height)
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Gesture Commander", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gesture Commander", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (width, height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        gesture = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)
        if not game.game_over:
            game.update(gesture)
        game.draw(frame)
        cv2.imshow("Gesture Commander", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            game = Spaceship(width, height)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 