import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import sys
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
import random

def main():
    # Set up OpenTelemetry tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    # Camera setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        sys.exit(1)
    height, width = frame.shape[:2]

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Runner Adventure (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    clock = pygame.time.Clock()

    # Game state
    player_y = height - 150
    player_x = 100
    player_width = 60
    player_height = 120
    velocity_y = 0
    gravity = 2
    jump_force = -35
    is_jumping = False
    is_ducking = False
    obstacles = []
    obstacle_speed = 15
    score = 0
    game_time = 45  # seconds
    start_time = time.time()
    last_obstacle_time = 0
    obstacle_interval = 1.5
    game_over = False
    with tracer.start_as_current_span("runneradventure_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            pose_landmarks = results.pose_landmarks
            now = time.time()
            # Detect jump (both hands above head) or duck (head low)
            jump_detected = False
            duck_detected = False
            if pose_landmarks:
                left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * height
                if left_wrist.y < nose.y and right_wrist.y < nose.y and not is_jumping:
                    jump_detected = True
                if nose.y * height > shoulder_y + 60:
                    duck_detected = True
            # Handle jump
            if not game_over and jump_detected and not is_jumping:
                with tracer.start_as_current_span("jump"):
                    velocity_y = jump_force
                    is_jumping = True
                    is_ducking = False
            # Handle duck
            if not game_over and duck_detected and not is_jumping:
                with tracer.start_as_current_span("duck"):
                    is_ducking = True
            else:
                is_ducking = False
            # Update player
            if is_jumping:
                player_y += velocity_y
                velocity_y += gravity
                if player_y >= height - 150:
                    player_y = height - 150
                    velocity_y = 0
                    is_jumping = False
            # Spawn obstacles
            if not game_over and now - last_obstacle_time > obstacle_interval:
                obstacle_height = random.choice([60, 120])
                obstacle_y = height - 150 if obstacle_height == 120 else height - 90
                obstacles.append([width, obstacle_y, 40, obstacle_height])
                last_obstacle_time = now
            # Move obstacles
            for obs in obstacles:
                obs[0] -= obstacle_speed
            obstacles = [obs for obs in obstacles if obs[0] + obs[2] > 0]
            # Check collisions
            for obs in obstacles:
                px, py, pw, ph = player_x, player_y, player_width, player_height if not is_ducking else player_height // 2
                ox, oy, ow, oh = obs
                if (px < ox + ow and px + pw > ox and py < oy + oh and py + ph > oy):
                    with tracer.start_as_current_span("obstacle_hit"):
                        game_over = True
            # Update score
            if not game_over:
                for obs in obstacles:
                    if obs[0] + obs[2] < player_x and not hasattr(obs, 'scored'):
                        score += 1
                        setattr(obs, 'scored', True)
            # Draw UI
            frame_disp = frame.copy()
            if pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_disp, pose_landmarks, mp_pose.POSE_CONNECTIONS)
            elapsed = int(now - start_time)
            remains = max(0, game_time - elapsed)
            cv2.putText(frame_disp, f"Score: {score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            cv2.putText(frame_disp, f"Time: {remains}s", (width-220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
            if game_over:
                cv2.putText(frame_disp, "Game Over!", (width//2-150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            # Draw player
            color = (0, 200, 255) if is_ducking else (0, 255, 0)
            cv2.rectangle(frame_disp, (player_x, int(player_y)), (player_x + player_width, int(player_y + (player_height // 2 if is_ducking else player_height))), color, -1)
            # Draw obstacles
            for obs in obstacles:
                ox, oy, ow, oh = obs
                cv2.rectangle(frame_disp, (int(ox), int(oy)), (int(ox + ow), int(oy + oh)), (255, 0, 0), -1)
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            player_y = height - 150
                            velocity_y = 0
                            is_jumping = False
                            is_ducking = False
                            obstacles = []
                            score = 0
                            start_time = time.time()
                            last_obstacle_time = 0
                            game_over = False
            # --- Pygame Rendering ---
            frame_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
            if remains <= 0:
                with tracer.start_as_current_span("game_over"):
                    running = False
        cap.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main() 