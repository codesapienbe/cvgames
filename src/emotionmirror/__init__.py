import cv2
import mediapipe as mp
import numpy as np
import random
import time
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
import sys

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

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    game = EmotionMirror()
    cap = cv2.VideoCapture(0)
    WIDTH, HEIGHT = 960, 720
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Emotion Mirror (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    small_font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()
    with tracer.start_as_current_span("emotionmirror_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            detected = "neutral"
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                detected = game.detect_emotion(face_landmarks)
            prev_score = game.score
            game.update(detected)
            if game.score > prev_score:
                with tracer.start_as_current_span("emotion_matched"):
                    pass
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_n:
                        with tracer.start_as_current_span("next_emotion"):
                            game.next_emotion()
            # --- Pygame Rendering ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 