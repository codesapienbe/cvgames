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
BG_COLOR = (240, 240, 255)
BEAKER_COLOR = (180, 180, 255)
LIQUID_COLOR = (100, 200, 255)
HAND_COLOR = (255, 100, 100)

mp_hands = mp.solutions.hands

class Beaker:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 80
        self.height = 180
        self.liquid = 1.0  # 1.0 = full, 0.0 = empty
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
    def draw(self, surface):
        pygame.draw.rect(surface, BEAKER_COLOR, self.rect(), border_radius=10)
        # Draw liquid
        liquid_height = int(self.height * self.liquid)
        if liquid_height > 0:
            pygame.draw.rect(surface, LIQUID_COLOR, (self.x, self.y + self.height - liquid_height, self.width, liquid_height), border_radius=10)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chemistry Lab (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 32)
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    beaker1 = Beaker(200, HEIGHT//2 - 90)
    beaker2 = Beaker(WIDTH-280, HEIGHT//2 - 90)
    score = 0
    pouring = False
    last_pour_time = 0
    with tracer.start_as_current_span("chemistrylab_session"):
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
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                index_tip = hand.landmark[8]
                hx = int(index_tip.x * WIDTH)
                hy = int(index_tip.y * HEIGHT)
                hand_pos = (hx, hy)
                # If hand is over beaker1 and hand is low, start pouring
                if beaker1.rect().collidepoint(hx, hy) and hy > beaker1.y + beaker1.height - 40 and beaker1.liquid > 0.05:
                    pouring = True
                elif beaker2.rect().collidepoint(hx, hy) and pouring:
                    # Pour from beaker1 to beaker2
                    if time.time() - last_pour_time > 0.5 and beaker1.liquid > 0.05:
                        with tracer.start_as_current_span("pour"):
                            beaker1.liquid -= 0.1
                            beaker2.liquid = min(1.0, beaker2.liquid + 0.1)
                            score += 1
                            last_pour_time = time.time()
                else:
                    pouring = False
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            screen.fill(BG_COLOR)
            beaker1.draw(screen)
            beaker2.draw(screen)
            # Draw hand
            if hand_pos:
                pygame.draw.circle(screen, HAND_COLOR, hand_pos, 30, 4)
            # Draw score
            score_text = font.render(f"Score: {score}", True, (0,0,0))
            screen.blit(score_text, (30, 30))
            instr = font.render("Move your hand to pour from left to right beaker!", True, (0,0,0))
            screen.blit(instr, (30, HEIGHT - 50))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 