import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import sys
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

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
    pygame.display.set_caption("Jump Rope Simulator (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    clock = pygame.time.Clock()

    # Game state
    score = 0
    game_time = 30  # seconds
    start_time = time.time()
    last_jump_time = 0
    jump_cooldown = 0.7  # seconds between jumps
    jumping = False
    missed = False
    rope_y = height - 100
    rope_speed = 10
    rope_direction = 1
    with tracer.start_as_current_span("jumpropesimulator_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            pose_landmarks = results.pose_landmarks
            # Detect jump (use nose or mid-hip y position)
            jump_detected = False
            if pose_landmarks:
                # Use nose y for jump detection
                nose_y = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * height
                if nose_y < height // 2 and time.time() - last_jump_time > jump_cooldown:
                    jump_detected = True
                    last_jump_time = time.time()
                    with tracer.start_as_current_span("jump"):
                        score += 1
                    jumping = True
                    missed = False
                else:
                    jumping = False
            # Rope animation
            rope_y += rope_speed * rope_direction
            if rope_y > height - 50:
                rope_direction = -1
            elif rope_y < 100:
                rope_direction = 1
            # Miss detection: if rope is at bottom and not jumping
            if rope_direction == -1 and rope_y > height - 120 and not jumping and time.time() - last_jump_time > 0.2:
                if not missed:
                    with tracer.start_as_current_span("miss"):
                        score = max(0, score - 1)
                    missed = True
            # Draw UI
            frame_disp = frame.copy()
            if pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_disp, pose_landmarks, mp_pose.POSE_CONNECTIONS)
            elapsed = int(time.time() - start_time)
            remains = max(0, game_time - elapsed)
            cv2.putText(frame_disp, f"Score: {score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            cv2.putText(frame_disp, f"Time: {remains}s", (width-220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
            if jumping:
                cv2.putText(frame_disp, "Jump!", (width//2-100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
            elif missed:
                cv2.putText(frame_disp, "Missed!", (width//2-100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            # Draw rope
            cv2.line(frame_disp, (100, int(rope_y)), (width-100, int(rope_y)), (255, 200, 0), 10)
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            score = 0
                            start_time = time.time()
                            last_jump_time = 0
                            jumping = False
                            missed = False
                            rope_y = height - 100
                            rope_direction = 1
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