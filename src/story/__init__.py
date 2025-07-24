import cv2
import mediapipe as mp
import time
import numpy as np
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

story = [
    "Welcome to the Eye Blink Story!",
    "Progress through the story by blinking.",
    "Once upon a time, there was a brave adventurer.",
    "He journeyed through shadowy forests.",
    "He crossed raging rivers and climbed steep mountains.",
    "A fearsome dragon blocked his path. Blink to face it!",
    "With courage and wit, he defeated the dragon.",
    "The kingdom was saved. The End!"
]

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Threshold for blink detection (distance between eyelids)
BLINK_THRESHOLD = 0.02

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Eye Blink Story (CV+Pygame)")
font = pygame.font.SysFont("Arial", 36)
clock = pygame.time.Clock()

def main():
    prev_blink = False
    blink_cooldown = False
    last_blink_time = 0
    story_index = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit(1)
    with tracer.start_as_current_span("story_session"):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_story"):
                            story_index = 0
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            blink = False
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                upper = face.landmark[159]
                lower = face.landmark[145]
                dist = abs(upper.y - lower.y)
                blink = dist < BLINK_THRESHOLD
            # On blink detected
            if blink and not prev_blink and not blink_cooldown:
                with tracer.start_as_current_span("blink_advance"):
                    story_index = min(story_index + 1, len(story) - 1)
                blink_cooldown = True
                last_blink_time = time.time()
            prev_blink = blink
            # Cooldown to avoid multiple triggers
            if blink_cooldown and time.time() - last_blink_time > 1:
                blink_cooldown = False
            # Draw background
            screen.fill((30, 30, 60))
            # Show webcam preview (small)
            frame_small = cv2.resize(frame, (160, 120))
            frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_small))
            screen.blit(surf, (WIDTH-170, HEIGHT-130))
            # Display current story text
            story_surf = font.render(story[story_index], True, (255, 255, 255))
            screen.blit(story_surf, (WIDTH//2 - story_surf.get_width()//2, 120))
            instr_surf = font.render("Blink to continue, 'q' to quit.", True, (200, 200, 200))
            screen.blit(instr_surf, (WIDTH//2 - instr_surf.get_width()//2, HEIGHT-60))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    import sys
    sys.exit()

if __name__ == "__main__":
    main()
