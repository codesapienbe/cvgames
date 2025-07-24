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
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ski Downhill (CV+Pygame)")
font = pygame.font.SysFont("Arial", 32)
clock = pygame.time.Clock()

# Skier properties
def skier_rect(x, y):
    return pygame.Rect(x-15, y-30, 30, 60)

# Game variables
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    skier_x = WIDTH // 2
    skier_y = HEIGHT - 80
    skier_speed = 5
    score = 0
    obstacles = []
    obstacle_timer = 0
    obstacle_interval = 1.2
    running = True
    crashed = False
    with tracer.start_as_current_span("skidownhill_session"):
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            skier_x = WIDTH // 2
                            skier_y = HEIGHT - 80
                            score = 0
                            obstacles.clear()
                            crashed = False
                            obstacle_timer = 0
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            # Use pose landmarks to steer (shoulders)
            steer = 0
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                center = (left_shoulder.x + right_shoulder.x) / 2
                if center < 0.45:
                    steer = -1
                elif center > 0.55:
                    steer = 1
                # Log steering
                if steer != 0:
                    with tracer.start_as_current_span("steer"):
                        pass
            if not crashed:
                skier_x += steer * skier_speed
                skier_x = max(30, min(WIDTH-30, skier_x))
            # Obstacles
            now = time.time()
            if not crashed and (now - obstacle_timer > obstacle_interval):
                with tracer.start_as_current_span("checkpoint"):
                    obstacles.append({'x': random.randint(40, WIDTH-40), 'y': -40})
                obstacle_timer = now
            for obs in obstacles:
                obs['y'] += 6
            obstacles = [o for o in obstacles if o['y'] < HEIGHT+40]
            # Collision
            skier = skier_rect(skier_x, skier_y)
            for obs in obstacles:
                if skier.colliderect(pygame.Rect(obs['x']-20, obs['y']-20, 40, 40)):
                    if not crashed:
                        with tracer.start_as_current_span("crash"):
                            crashed = True
            # Score
            if not crashed:
                for obs in obstacles:
                    if obs['y'] == skier_y:
                        score += 1
                        with tracer.start_as_current_span("score"):
                            pass
            # Draw background
            screen.fill((180, 220, 255))
            # Draw obstacles
            for obs in obstacles:
                pygame.draw.rect(screen, (100, 60, 20), (obs['x']-20, obs['y']-20, 40, 40))
            # Draw skier
            pygame.draw.ellipse(screen, (255, 255, 0), skier)
            pygame.draw.line(screen, (0,0,0), (skier_x, skier_y+20), (skier_x-15, skier_y+40), 4)
            pygame.draw.line(screen, (0,0,0), (skier_x, skier_y+20), (skier_x+15, skier_y+40), 4)
            # Draw score
            score_surf = font.render(f"Score: {score}", True, (0,0,0))
            screen.blit(score_surf, (10, 10))
            # Crash message
            if crashed:
                msg = font.render("Crashed! Press R to restart", True, (255,0,0))
                screen.blit(msg, (WIDTH//2 - msg.get_width()//2, HEIGHT//2))
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