import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import sys
import time
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
WIDTH, HEIGHT = 600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Stack Blocks (CV+Pygame)")
font = pygame.font.SysFont("Arial", 36)
clock = pygame.time.Clock()

BLOCK_WIDTH = 120
BLOCK_HEIGHT = 40
BLOCK_COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

# Helper: detect fist (all fingers down)
def is_fist(hand_landmarks):
    if not hand_landmarks:
        return False
    tips = [8, 12, 16, 20]
    return all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip-2].y for tip in tips)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    stack = []  # list of (x, y, color)
    block_x = WIDTH // 2
    block_y = 80
    block_color = random.choice(BLOCK_COLORS)
    block_moving = True
    score = 0
    running = True
    last_drop_time = 0
    with tracer.start_as_current_span("stackblocks_session"):
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            stack.clear()
                            block_x = WIDTH // 2
                            block_y = 80
                            block_color = random.choice(BLOCK_COLORS)
                            block_moving = True
                            score = 0
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            # Move block horizontally with hand x
            if block_moving and results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                hand_x = int(lm[9].x * WIDTH)
                with tracer.start_as_current_span("move"):
                    block_x = min(max(BLOCK_WIDTH//2, hand_x), WIDTH-BLOCK_WIDTH//2)
                # Drop block with fist
                if is_fist(results.multi_hand_landmarks[0]):
                    if time.time() - last_drop_time > 1.0:
                        with tracer.start_as_current_span("drop"):
                            block_moving = False
                            last_drop_time = time.time()
            # Drop block with space key as fallback
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] and block_moving:
                if time.time() - last_drop_time > 1.0:
                    with tracer.start_as_current_span("drop"):
                        block_moving = False
                        last_drop_time = time.time()
            # Animate falling
            if not block_moving:
                block_y += 12
                # Check for stack or ground
                if not stack:
                    if block_y + BLOCK_HEIGHT//2 >= HEIGHT-20:
                        with tracer.start_as_current_span("stack"):
                            stack.append((block_x, HEIGHT-20-BLOCK_HEIGHT//2, block_color))
                            block_x = WIDTH // 2
                            block_y = 80
                            block_color = random.choice(BLOCK_COLORS)
                            block_moving = True
                            score += 1
                else:
                    top_x, top_y, _ = stack[-1]
                    if abs(block_x - top_x) < BLOCK_WIDTH and block_y + BLOCK_HEIGHT//2 >= top_y - BLOCK_HEIGHT//2:
                        with tracer.start_as_current_span("stack"):
                            stack.append((block_x, top_y-BLOCK_HEIGHT, block_color))
                            block_x = WIDTH // 2
                            block_y = 80
                            block_color = random.choice(BLOCK_COLORS)
                            block_moving = True
                            score += 1
                    elif block_y > HEIGHT:
                        # Missed
                        with tracer.start_as_current_span("miss"):
                            block_x = WIDTH // 2
                            block_y = 80
                            block_color = random.choice(BLOCK_COLORS)
                            block_moving = True
            # Draw background
            screen.fill((240, 240, 255))
            # Draw stack
            for x, y, color in stack:
                pygame.draw.rect(screen, color, (x-BLOCK_WIDTH//2, y-BLOCK_HEIGHT//2, BLOCK_WIDTH, BLOCK_HEIGHT))
            # Draw moving block
            if block_moving:
                pygame.draw.rect(screen, block_color, (block_x-BLOCK_WIDTH//2, block_y-BLOCK_HEIGHT//2, BLOCK_WIDTH, BLOCK_HEIGHT))
            # Draw score
            score_surf = font.render(f"Score: {score}", True, (0,0,0))
            screen.blit(score_surf, (20, 20))
            # Draw instructions
            instr1 = font.render("Move hand to move block", True, (100, 100, 100))
            instr2 = font.render("Make a fist to drop block", True, (100, 100, 100))
            instr3 = font.render("Press SPACE to drop (fallback)", True, (100, 100, 100))
            screen.blit(instr1, (20, HEIGHT-120))
            screen.blit(instr2, (20, HEIGHT-80))
            screen.blit(instr3, (20, HEIGHT-40))
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