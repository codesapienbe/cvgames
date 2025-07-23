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

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

WIDTH, HEIGHT = 960, 720
TARGET_RADIUS = 50
HAND_RADIUS = 30
TARGET_COLOR = (255, 0, 0)
HIT_COLOR = (0, 255, 0)
BG_COLOR = (30, 30, 30)

mp_hands = mp.solutions.hands

class Target:
    def __init__(self):
        self.x = random.randint(200, WIDTH - 200)
        self.y = random.randint(150, HEIGHT - 150)
        self.hit = False
        self.last_move = time.time()
    def move(self):
        self.x = random.randint(200, WIDTH - 200)
        self.y = random.randint(150, HEIGHT - 150)
        self.hit = False
        self.last_move = time.time()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boxing Trainer (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    target = Target()
    score = 0
    prev_hand_pos = None
    prev_time = time.time()
    with tracer.start_as_current_span("boxingtrainer_session"):
        running = True
        while running:
            # --- Computer Vision Input ---
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            hand_pos = None
            punch_detected = False
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                index_tip = hand.landmark[8]
                hx = int(index_tip.x * WIDTH)
                hy = int(index_tip.y * HEIGHT)
                hand_pos = (hx, hy)
                # Detect punch: fast movement toward target
                if prev_hand_pos is not None:
                    dx = hx - prev_hand_pos[0]
                    dy = hy - prev_hand_pos[1]
                    dt = time.time() - prev_time
                    speed = np.hypot(dx, dy) / (dt + 1e-5)
                    dist_to_target = np.hypot(hx - target.x, hy - target.y)
                    if speed > 1000 and dist_to_target < TARGET_RADIUS + HAND_RADIUS and not target.hit:
                        punch_detected = True
                        with tracer.start_as_current_span("punch"):
                            score += 1
                            target.hit = True
                            target.last_move = time.time()
                prev_hand_pos = (hx, hy)
                prev_time = time.time()
            # Move target after hit or every 2 seconds
            if target.hit and time.time() - target.last_move > 0.5:
                target.move()
            elif not target.hit and time.time() - target.last_move > 2.0:
                target.move()
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            screen.fill(BG_COLOR)
            # Draw target
            color = HIT_COLOR if target.hit else TARGET_COLOR
            pygame.draw.circle(screen, color, (target.x, target.y), TARGET_RADIUS)
            # Draw hand
            if hand_pos:
                pygame.draw.circle(screen, (0, 200, 255), hand_pos, HAND_RADIUS, 3)
            # Draw score
            score_text = font.render(f"Score: {score}", True, (255,255,255))
            screen.blit(score_text, (30, 30))
            instr = font.render("Punch the target with your hand!", True, (255,255,255))
            screen.blit(instr, (30, HEIGHT - 50))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 