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
pygame.display.set_caption("Virtual Pong (CV+Pygame)")
font = pygame.font.SysFont("Arial", 36)
clock = pygame.time.Clock()

PADDLE_WIDTH = 20
PADDLE_HEIGHT = 120
BALL_SIZE = 20
BALL_SPEED = 8

class Paddle:
    def __init__(self, x):
        self.x = x
        self.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
    def move(self, y):
        self.y = max(0, min(HEIGHT - PADDLE_HEIGHT, y - PADDLE_HEIGHT // 2))
    def draw(self, screen):
        pygame.draw.rect(screen, (255,255,255), (self.x, self.y, PADDLE_WIDTH, PADDLE_HEIGHT))

class Ball:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.vx = random.choice([-BALL_SPEED, BALL_SPEED])
        self.vy = random.choice([-BALL_SPEED, BALL_SPEED])
    def move(self):
        self.x += self.vx
        self.y += self.vy
        if self.y < 0 or self.y > HEIGHT - BALL_SIZE:
            self.vy *= -1
    def draw(self, screen):
        pygame.draw.ellipse(screen, (255,255,255), (self.x, self.y, BALL_SIZE, BALL_SIZE))
    def reset(self):
        self.__init__()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    player = Paddle(30)
    ai = Paddle(WIDTH - 50)
    ball = Ball()
    player_score = 0
    ai_score = 0
    running = True
    with tracer.start_as_current_span("virtualpong_session"):
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            player = Paddle(30)
                            ai = Paddle(WIDTH - 50)
                            ball = Ball()
                            player_score = 0
                            ai_score = 0
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            # Player paddle control
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                hand_y = int(lm[9].y * HEIGHT)
                with tracer.start_as_current_span("paddle_move"):
                    player.move(hand_y)
            # AI paddle follows ball
            ai.move(ball.y + BALL_SIZE//2)
            # Ball movement
            ball.move()
            # Collision with player paddle
            if (player.x < ball.x < player.x + PADDLE_WIDTH and
                player.y < ball.y + BALL_SIZE and ball.y < player.y + PADDLE_HEIGHT):
                with tracer.start_as_current_span("hit"):
                    ball.vx *= -1
                    ball.x = player.x + PADDLE_WIDTH
            # Collision with AI paddle
            if (ai.x < ball.x + BALL_SIZE < ai.x + PADDLE_WIDTH and
                ai.y < ball.y + BALL_SIZE and ball.y < ai.y + PADDLE_HEIGHT):
                ball.vx *= -1
                ball.x = ai.x - BALL_SIZE
            # Score
            if ball.x < 0:
                with tracer.start_as_current_span("miss"):
                    ai_score += 1
                    ball.reset()
            if ball.x > WIDTH:
                with tracer.start_as_current_span("score"):
                    player_score += 1
                    ball.reset()
            # Draw background
            screen.fill((0,0,0))
            # Draw paddles and ball
            player.draw(screen)
            ai.draw(screen)
            ball.draw(screen)
            # Draw scores
            score_surf = font.render(f"Player: {player_score}", True, (255,255,255))
            ai_score_surf = font.render(f"AI: {ai_score}", True, (255,255,255))
            screen.blit(score_surf, (30, 20))
            screen.blit(ai_score_surf, (WIDTH-180, 20))
            # Draw instructions
            instr1 = font.render("Move hand up/down to control paddle", True, (200, 200, 200))
            instr2 = font.render("Press R to restart, Q to quit", True, (200, 200, 200))
            screen.blit(instr1, (30, HEIGHT-80))
            screen.blit(instr2, (30, HEIGHT-40))
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