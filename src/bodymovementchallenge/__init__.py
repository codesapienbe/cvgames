import cv2
import mediapipe as mp
import numpy as np
import random
import time

CHALLENGES = [
    "Raise both hands above your head!",
    "Touch your toes!",
    "Stand on one leg!",
    "Jump in place!",
    "Stretch your arms wide!"
]

class BodyMovementChallenge:
    def __init__(self):
        self.score = 0
        self.current_challenge = None
        self.challenge_start = 0
        self.challenge_duration = 5
        self.completed = False
        self.next_challenge()

    def next_challenge(self):
        self.current_challenge = random.choice(CHALLENGES)
        self.challenge_start = time.time()
        self.completed = False

    def check_pose(self, pose_landmarks):
        # Simple placeholder logic for demo
        if not pose_landmarks:
            return False
        # Example: if challenge is "Raise both hands above your head!"
        if "hands above" in self.current_challenge:
            left = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            right = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            head = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
            return left.y < head.y and right.y < head.y
        # Add more logic for other challenges as needed
        return random.random() < 0.1  # Randomly succeed for demo

    def update(self, pose_landmarks):
        if not self.completed and self.check_pose(pose_landmarks):
            self.score += 10
            self.completed = True

    def draw(self, frame):
        cv2.putText(frame, f"Challenge: {self.current_challenge}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Score: {self.score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if self.completed:
            cv2.putText(frame, "Success!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Try to complete the challenge!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press n for next challenge, q to quit", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    game = BodyMovementChallenge()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Body Movement Challenge", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Body Movement Challenge", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            game.update(results.pose_landmarks)
        game.draw(frame)
        cv2.imshow("Body Movement Challenge", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            game.next_challenge()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 