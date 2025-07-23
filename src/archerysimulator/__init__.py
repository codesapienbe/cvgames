import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

WIDTH, HEIGHT = 960, 720
TARGET_POS = (WIDTH - 200, HEIGHT // 2)
TARGET_RADIUS = 80
ARROW_COLOR = (120, 60, 0)
BOW_COLOR = (80, 80, 80)
BG_COLOR = (220, 240, 255)

mp_hands = mp.solutions.hands

class Arrow:
    def __init__(self, start, end, speed=30):
        self.x, self.y = start
        self.end = end
        self.speed = speed
        dx = end[0] - self.x
        dy = end[1] - self.y
        dist = np.hypot(dx, dy)
        self.vx = (dx / dist) * speed
        self.vy = (dy / dist) * speed
        self.active = True
    def update(self):
        if not self.active:
            return
        self.x += self.vx
        self.y += self.vy
        if (self.x > WIDTH or self.x < 0 or self.y > HEIGHT or self.y < 0):
            self.active = False
    def pos(self):
        return int(self.x), int(self.y)

def hand_distance(lm1, lm2):
    return np.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Archery Simulator (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 32)
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    arrows = []
    score = 0
    drawing_bow = False
    bow_start = (150, HEIGHT // 2)
    bow_end = bow_start
    last_shot_time = 0
    with tracer.start_as_current_span("archery_session"):
        running = True
        while running:
            # --- Computer Vision Input ---
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            hand_tip = None
            thumb_tip = None
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                hand_tip = hand.landmark[8]  # Index tip
                thumb_tip = hand.landmark[4] # Thumb tip
                # Map hand position to screen
                hx = int(hand_tip.x * WIDTH)
                hy = int(hand_tip.y * HEIGHT)
                tx = int(thumb_tip.x * WIDTH)
                ty = int(thumb_tip.y * HEIGHT)
                bow_end = (hx, hy)
                # If hand is pinched (index tip close to thumb tip), draw bow
                if hand_distance(hand_tip, thumb_tip) < 0.05:
                    drawing_bow = True
                else:
                    if drawing_bow and time.time() - last_shot_time > 0.5:
                        # Release arrow
                        with tracer.start_as_current_span("shoot_arrow"):
                            arrows.append(Arrow(bow_start, bow_end))
                            last_shot_time = time.time()
                    drawing_bow = False
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Update arrows ---
            for arrow in arrows:
                arrow.update()
            arrows = [a for a in arrows if a.active]
            # --- Collision detection ---
            for arrow in arrows:
                ax, ay = arrow.pos()
                dist = np.hypot(ax - TARGET_POS[0], ay - TARGET_POS[1])
                if dist < TARGET_RADIUS:
                    with tracer.start_as_current_span("hit_target"):
                        score += 10
                        arrow.active = False
            # --- Pygame Rendering ---
            screen.fill(BG_COLOR)
            # Draw target
            for r, color in zip([TARGET_RADIUS, 60, 40, 20], [(255,0,0),(255,255,255),(0,0,255),(255,255,0)]):
                pygame.draw.circle(screen, color, TARGET_POS, r)
            # Draw bow
            pygame.draw.line(screen, BOW_COLOR, bow_start, bow_end, 8)
            pygame.draw.circle(screen, (100, 60, 30), bow_start, 20)
            # Draw arrows
            for arrow in arrows:
                pygame.draw.line(screen, ARROW_COLOR, (int(bow_start[0]), int(bow_start[1])), arrow.pos(), 6)
                pygame.draw.circle(screen, (60, 30, 10), arrow.pos(), 8)
            # Draw score
            score_text = font.render(f"Score: {score}", True, (0,0,0))
            screen.blit(score_text, (30, 30))
            # Draw instructions
            instr = font.render("Pinch index and thumb to draw, release to shoot", True, (0,0,0))
            screen.blit(instr, (30, HEIGHT - 50))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 