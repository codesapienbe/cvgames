import cv2
import mediapipe as mp
import time
import numpy as np
import pygame
import sys
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

WIDTH, HEIGHT = 960, 720
BG_COLOR = (20, 20, 40)
SHIP_COLOR = (255, 255, 255)

mp_hands = mp.solutions.hands

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Gesture Commander (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 32)
    small_font = pygame.font.SysFont("Arial", 20)
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    ship_x, ship_y = WIDTH/2, HEIGHT/2
    vel = 0.0
    acceleration = 200.0
    friction = 0.98
    max_speed = 600.0
    dir_x, dir_y = 1.0, 0.0
    prev_time = time.time()
    with tracer.start_as_current_span("commander_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            thrust = False
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                # Count extended fingers (index, middle, ring, pinky)
                tips = [8, 12, 16, 20]
                count = 0
                for tip in tips:
                    if hand.landmark[tip].y < hand.landmark[tip-2].y:
                        count += 1
                # Open palm if 3 or more fingers extended
                if count >= 3:
                    thrust = True
                    with tracer.start_as_current_span("thrust"):
                        pass
                # Update direction from wrist (0) to index tip (8)
                dx = hand.landmark[8].x - hand.landmark[0].x
                dy = hand.landmark[8].y - hand.landmark[0].y
                norm = np.hypot(dx, dy)
                if norm > 0.01:
                    dir_x, dir_y = dx / norm, dy / norm
            # Update velocity
            if thrust:
                vel += acceleration * dt
            vel *= friction
            vel = max(0.0, min(vel, max_speed))
            # Update position
            ship_x += dir_x * vel * dt
            ship_y += dir_y * vel * dt
            # Wrap around edges
            if ship_x < 0: ship_x = WIDTH
            if ship_x > WIDTH: ship_x = 0
            if ship_y < 0: ship_y = HEIGHT
            if ship_y > HEIGHT: ship_y = 0
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            screen.fill(BG_COLOR)
            cx, cy = int(ship_x), int(ship_y)
            pygame.draw.circle(screen, SHIP_COLOR, (cx, cy), 15, 2)
            fx = int(cx + dir_x * 25)
            fy = int(cy + dir_y * 25)
            pygame.draw.line(screen, SHIP_COLOR, (cx, cy), (fx, fy), 2)
            # UI overlays
            instr1 = font.render("Open palm to thrust", True, (255,255,255))
            screen.blit(instr1, (10, 30))
            instr2 = small_font.render("Press 'q' to quit", True, (200,200,200))
            screen.blit(instr2, (10, HEIGHT - 40))
            pygame.display.flip()
            # --- Keyboard quit ---
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                running = False
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
