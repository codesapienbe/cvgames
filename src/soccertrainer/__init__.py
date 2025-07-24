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

# MediaPipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Soccer Trainer (CV+Pygame)")
font = pygame.font.SysFont("Arial", 32)
clock = pygame.time.Clock()

# Ball properties
def ball_rect(x, y):
    return pygame.Rect(x-20, y-20, 40, 40)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    ball_x = WIDTH // 2
    ball_y = HEIGHT - 60
    ball_speed = 0
    ball_vy = 0
    gravity = 0.7
    score = 0
    goal_x = random.randint(100, WIDTH-100)
    goal_y = 60
    running = True
    kicked = False
    last_kick_time = 0
    with tracer.start_as_current_span("soccertrainer_session"):
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            ball_x = WIDTH // 2
                            ball_y = HEIGHT - 60
                            ball_speed = 0
                            ball_vy = 0
                            score = 0
                            goal_x = random.randint(100, WIDTH-100)
                            kicked = False
                            last_kick_time = 0
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            # Detect kick (right ankle y moves up fast)
            kick_detected = False
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                right_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
                # If ankle is above knee (y is less), consider it a kick
                if right_ankle.y < right_knee.y and not kicked and time.time() - last_kick_time > 1.0:
                    kick_detected = True
                    kicked = True
                    last_kick_time = time.time()
                    with tracer.start_as_current_span("kick"):
                        ball_speed = random.randint(12, 16)
                        ball_vy = -random.randint(16, 20)
            if not results.pose_landmarks or (results.pose_landmarks and lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y > lm[mp_pose.PoseLandmark.RIGHT_KNEE].y):
                kicked = False
            # Ball movement
            if ball_speed > 0 or ball_vy != 0:
                ball_x += ball_speed
                ball_y += ball_vy
                ball_vy += gravity
                ball_speed *= 0.98
            # Check goal
            goal_rect = pygame.Rect(goal_x-50, goal_y-10, 100, 20)
            if ball_rect(ball_x, ball_y).colliderect(goal_rect):
                with tracer.start_as_current_span("goal"):
                    score += 1
                    ball_x = WIDTH // 2
                    ball_y = HEIGHT - 60
                    ball_speed = 0
                    ball_vy = 0
                    goal_x = random.randint(100, WIDTH-100)
            # Missed (ball out of bounds)
            if ball_y > HEIGHT or ball_x < 0 or ball_x > WIDTH:
                with tracer.start_as_current_span("miss"):
                    ball_x = WIDTH // 2
                    ball_y = HEIGHT - 60
                    ball_speed = 0
                    ball_vy = 0
            # Draw background
            screen.fill((60, 180, 60))
            # Draw goal
            pygame.draw.rect(screen, (255, 255, 255), goal_rect, 3)
            pygame.draw.rect(screen, (200, 200, 0), goal_rect)
            # Draw ball
            pygame.draw.ellipse(screen, (255, 255, 255), ball_rect(ball_x, ball_y))
            pygame.draw.circle(screen, (0, 0, 0), (int(ball_x), int(ball_y)), 20, 2)
            # Draw score
            score_surf = font.render(f"Score: {score}", True, (0,0,0))
            screen.blit(score_surf, (10, 10))
            # Show webcam preview (small)
            frame_small = cv2.resize(frame, (160, 120))
            frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_small))
            screen.blit(surf, (WIDTH-170, HEIGHT-130))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 