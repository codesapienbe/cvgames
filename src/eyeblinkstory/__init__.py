import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
import sys

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

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    game = EyeBlinkStory()
    cap = cv2.VideoCapture(0)
    WIDTH, HEIGHT = 960, 720
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Eye Blink Story (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    small_font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()
    with tracer.start_as_current_span("eyeblinkstory_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            blink = False
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                blink = game.detect_blink(face_landmarks)
            if blink and time.time() - game.last_blink > game.blink_cooldown:
                with tracer.start_as_current_span("blink_advance"):
                    game.next_line()
                game.last_blink = time.time()
            game.draw(frame)
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
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