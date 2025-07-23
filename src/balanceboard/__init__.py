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

WIDTH, HEIGHT = 800, 600
BOARD_WIDTH, BOARD_HEIGHT = 200, 20
BALL_RADIUS = 20
GRAVITY = 0.5
FRICTION = 0.98

mp_hands = mp.solutions.hands

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Balance Board (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 32)
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    board_x = WIDTH // 2 - BOARD_WIDTH // 2
    board_y = HEIGHT - 100
    ball_x = WIDTH // 2
    ball_y = board_y - BALL_RADIUS
    ball_vx = 0
    ball_vy = 0
    score = 0
    start_time = time.time()
    running = True
    with tracer.start_as_current_span("balanceboard_session"):
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
            # --- Move board with hand ---
            if hand_x is not None:
                board_x = np.clip(hand_x - BOARD_WIDTH // 2, 0, WIDTH - BOARD_WIDTH)
            # --- Ball physics ---
            ball_vy += GRAVITY
            ball_x += ball_vx
            ball_y += ball_vy
            ball_vx *= FRICTION
            # --- Collision with board ---
            if (board_y - BALL_RADIUS < ball_y < board_y + BOARD_HEIGHT and
                board_x < ball_x < board_x + BOARD_WIDTH):
                ball_y = board_y - BALL_RADIUS
                ball_vy = -abs(ball_vy) * 0.7
                # Ball bounces depending on where it hits the board
                offset = (ball_x - (board_x + BOARD_WIDTH // 2)) / (BOARD_WIDTH // 2)
                ball_vx += offset * 5
                with tracer.start_as_current_span("ball_bounce"):
                    pass
            # --- Collision with walls ---
            if ball_x < BALL_RADIUS:
                ball_x = BALL_RADIUS
                ball_vx = abs(ball_vx) * 0.7
            if ball_x > WIDTH - BALL_RADIUS:
                ball_x = WIDTH - BALL_RADIUS
                ball_vx = -abs(ball_vx) * 0.7
            # --- Ball falls off ---
            if ball_y > HEIGHT:
                with tracer.start_as_current_span("ball_fall"):
                    score = 0
                    ball_x = WIDTH // 2
                    ball_y = board_y - BALL_RADIUS
                    ball_vx = 0
                    ball_vy = 0
                    start_time = time.time()
            # --- Scoring ---
            score = int(time.time() - start_time)
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            screen.fill((200, 230, 255))
            pygame.draw.rect(screen, (100, 80, 60), (board_x, board_y, BOARD_WIDTH, BOARD_HEIGHT), border_radius=10)
            pygame.draw.circle(screen, (255, 100, 100), (int(ball_x), int(ball_y)), BALL_RADIUS)
            score_text = font.render(f"Score: {score}", True, (0,0,0))
            screen.blit(score_text, (30, 30))
            instr = font.render("Move your hand left/right to balance the ball!", True, (0,0,0))
            screen.blit(instr, (30, HEIGHT - 50))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
