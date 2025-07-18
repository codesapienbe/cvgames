import cv2
import mediapipe as mp
import numpy as np
import time

STORY = [
    "You wake up in a mysterious room.",
    "A door appears in front of you.",
    "Blink to open the door.",
    "You see a bright light.",
    "Blink to walk toward the light.",
    "You find yourself outside, free!",
    "The End."
]

class EyeBlinkStory:
    def __init__(self):
        self.idx = 0
        self.last_blink = 0
        self.blink_cooldown = 1.0

    def next_line(self):
        if self.idx < len(STORY) - 1:
            self.idx += 1

    def detect_blink(self, face_landmarks):
        # For demo: use mouth open as blink (replace with real blink detection)
        if not face_landmarks:
            return False
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        return abs(upper_lip.y - lower_lip.y) > 0.05

    def draw(self, frame):
        cv2.putText(frame, STORY[self.idx], (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Blink to continue... (mouth open for demo)", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Press q to quit", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    game = EyeBlinkStory()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Eye Blink Story", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Eye Blink Story", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        blink = False
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            blink = game.detect_blink(face_landmarks)
        if blink and time.time() - game.last_blink > game.blink_cooldown:
            game.next_line()
            game.last_blink = time.time()
        game.draw(frame)
        cv2.imshow("Eye Blink Story", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 