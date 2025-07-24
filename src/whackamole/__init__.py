import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import time
import random
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Whack-a-Mole (CV+Pygame)")
font = pygame.font.SysFont("Arial", 36)
clock = pygame.time.Clock()

MOLE_RADIUS = 40
MOLE_POSITIONS = [
    (200, 200), (450, 200), (700, 200),
    (200, 400), (450, 400), (700, 400)
]
MOLE_APPEAR_TIME = 1.2

class Mole:
    def __init__(self):
        self.position = random.choice(MOLE_POSITIONS)
        self.visible = True
        self.spawn_time = time.time()
    def respawn(self):
        self.position = random.choice(MOLE_POSITIONS)
        self.visible = True
        self.spawn_time = time.time()
    def hide(self):
        self.visible = False

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    mole = Mole()
    score = 0
    misses = 0
    running = True
    last_whack_time = 0
    with tracer.start_as_current_span("whackamole_session"):
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            mole = Mole()
                            score = 0
                            misses = 0
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            whack = False
            finger_pos = None
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                x = int(lm[8].x * WIDTH)
                y = int(lm[8].y * HEIGHT)
                finger_pos = (x, y)
                # Whack if finger is close to mole and cooldown passed
                if mole.visible and np.hypot(x - mole.position[0], y - mole.position[1]) < MOLE_RADIUS:
                    if time.time() - last_whack_time > 0.7:
                        with tracer.start_as_current_span("whack"):
                            score += 1
                            mole.hide()
                            last_whack_time = time.time()
            # Mole timing
            if mole.visible and time.time() - mole.spawn_time > MOLE_APPEAR_TIME:
                with tracer.start_as_current_span("miss"):
                    misses += 1
                    mole.hide()
            if not mole.visible:
                with tracer.start_as_current_span("mole_appear"):
                    mole.respawn()
            # Draw background
            screen.fill((60, 180, 60))
            # Draw mole
            if mole.visible:
                pygame.draw.circle(screen, (139,69,19), mole.position, MOLE_RADIUS)
                pygame.draw.circle(screen, (0,0,0), mole.position, MOLE_RADIUS, 3)
            # Draw finger cursor
            if finger_pos:
                pygame.draw.circle(screen, (0,255,255), finger_pos, 15)
            # Draw score and misses
            score_surf = font.render(f"Score: {score}", True, (0,0,0))
            miss_surf = font.render(f"Misses: {misses}", True, (200,0,0))
            screen.blit(score_surf, (20, 20))
            screen.blit(miss_surf, (20, 70))
            # Draw instructions
            instr1 = font.render("Point finger to whack the mole!", True, (100, 100, 100))
            instr2 = font.render("Press R to restart, Q to quit", True, (100, 100, 100))
            screen.blit(instr1, (20, HEIGHT-80))
            screen.blit(instr2, (20, HEIGHT-40))
            # Show webcam preview (small)
            frame_small = cv2.resize(frame, (120, 90))
            frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_small))
            screen.blit(surf, (WIDTH-130, HEIGHT-100))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
