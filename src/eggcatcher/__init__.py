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

WIDTH, HEIGHT = 800, 600
BASKET_WIDTH, BASKET_HEIGHT = 120, 40
EGG_RADIUS = 20
EGG_COLOR = (255, 255, 200)
BASKET_COLOR = (200, 100, 50)
BG_COLOR = (220, 240, 255)

mp_hands = mp.solutions.hands

class Egg:
    def __init__(self):
        self.x = random.randint(EGG_RADIUS, WIDTH - EGG_RADIUS)
        self.y = -EGG_RADIUS
        self.speed = random.uniform(3, 6)
    def update(self):
        self.y += self.speed
    def pos(self):
        return int(self.x), int(self.y)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Egg Catcher (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 32)
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    basket_x = WIDTH // 2
    basket_y = HEIGHT - 60
    eggs = [Egg()]
    score = 0
    misses = 0
    with tracer.start_as_current_span("eggcatcher_session"):
        running = True
        while running:
            # --- Computer Vision Input ---
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            hand_x = None
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                index_tip = hand.landmark[8]
                hand_x = int(index_tip.x * WIDTH)
            # --- Move basket with hand ---
            if hand_x is not None:
                basket_x = np.clip(hand_x, BASKET_WIDTH//2, WIDTH - BASKET_WIDTH//2)
            # --- Update eggs ---
            for egg in eggs:
                egg.update()
            # --- Check for catches or misses ---
            for egg in list(eggs):
                ex, ey = egg.pos()
                if basket_y - EGG_RADIUS < ey < basket_y + BASKET_HEIGHT//2 and abs(ex - basket_x) < BASKET_WIDTH//2:
                    with tracer.start_as_current_span("egg_caught"):
                        score += 1
                    eggs.remove(egg)
                    eggs.append(Egg())
                elif ey > HEIGHT + EGG_RADIUS:
                    with tracer.start_as_current_span("egg_missed"):
                        misses += 1
                    eggs.remove(egg)
                    eggs.append(Egg())
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            screen.fill(BG_COLOR)
            # Draw basket
            pygame.draw.rect(screen, BASKET_COLOR, (basket_x - BASKET_WIDTH//2, basket_y, BASKET_WIDTH, BASKET_HEIGHT), border_radius=12)
            # Draw eggs
            for egg in eggs:
                pygame.draw.circle(screen, EGG_COLOR, egg.pos(), EGG_RADIUS)
            # Draw score
            score_text = font.render(f"Score: {score}", True, (0,0,0))
            screen.blit(score_text, (30, 30))
            miss_text = font.render(f"Misses: {misses}", True, (200,0,0))
            screen.blit(miss_text, (30, 70))
            instr = font.render("Move your hand to catch eggs! Press 'q' to quit.", True, (0,0,0))
            screen.blit(instr, (30, HEIGHT - 50))
            pygame.display.flip()
            clock.tick(30)
            # --- Keyboard quit ---
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                running = False
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 