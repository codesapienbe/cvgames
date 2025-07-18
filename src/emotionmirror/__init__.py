import cv2
import mediapipe as mp
import numpy as np
import random
import time

EMOTIONS = ["happy", "sad", "surprised", "neutral"]

class EmotionMirror:
    def __init__(self):
        self.target_emotion = random.choice(EMOTIONS)
        self.last_change = time.time()
        self.score = 0
        self.feedback = ""

    def next_emotion(self):
        self.target_emotion = random.choice(EMOTIONS)
        self.last_change = time.time()
        self.feedback = ""

    def detect_emotion(self, face_landmarks):
        # Simple smile detection for demo
        if not face_landmarks:
            return "neutral"
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        left = face_landmarks.landmark[61]
        right = face_landmarks.landmark[291]
        lip_width = abs(right.x - left.x)
        lip_height = abs(upper_lip.y - lower_lip.y)
        if lip_height > 0 and lip_width / lip_height > 2.0:
            return "happy"
        # Add more rules for other emotions as needed
        return "neutral"

    def update(self, detected):
        if detected == self.target_emotion:
            self.score += 1
            self.feedback = "Great! You matched the emotion."
            self.next_emotion()
        else:
            self.feedback = f"Try to show: {self.target_emotion}"

    def draw(self, frame):
        cv2.putText(frame, f"Target: {self.target_emotion}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Score: {self.score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, self.feedback, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press n for next emotion, q to quit", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    game = EmotionMirror()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Emotion Mirror", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Emotion Mirror", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        detected = "neutral"
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            detected = game.detect_emotion(face_landmarks)
        game.update(detected)
        game.draw(frame)
        cv2.imshow("Emotion Mirror", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            game.next_emotion()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 