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
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Virtual Yoga Coach (CV+Pygame)")
font = pygame.font.SysFont("Arial", 36)
clock = pygame.time.Clock()

YOGA_POSES = [
    {"name": "T Pose", "desc": "Arms outstretched horizontally"},
    {"name": "Tree Pose", "desc": "Stand on one leg, other foot on inner thigh"},
    {"name": "Warrior", "desc": "One leg forward, arms outstretched"},
    {"name": "Chair", "desc": "Knees bent, arms up"},
]

# Helper: check if arms are outstretched (T Pose)
def is_t_pose(landmarks):
    if not landmarks:
        return False
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    # Check if wrists are horizontally aligned with shoulders
    return (abs(l_shoulder.y - l_wrist.y) < 0.07 and abs(r_shoulder.y - r_wrist.y) < 0.07)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    pose_idx = 0
    score = 0
    show_result = False
    result_text = ""
    last_pose_time = 0
    running = True
    with tracer.start_as_current_span("virtualyogacoach_session"):
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            pose_idx = 0
                            score = 0
                            show_result = False
                            result_text = ""
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            pose_detected = False
            correct = False
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if YOGA_POSES[pose_idx]["name"] == "T Pose":
                    pose_detected = is_t_pose(lm)
                # (Add more pose checks for other poses as needed)
            # Handle pose detection
            if pose_detected and not show_result:
                with tracer.start_as_current_span("pose_detected"):
                    with tracer.start_as_current_span("correct"):
                        score += 1
                        result_text = "Correct!"
                        show_result = True
                        last_pose_time = time.time()
            # Next pose after 2 seconds
            if show_result and time.time() - last_pose_time > 2:
                with tracer.start_as_current_span("next_pose"):
                    pose_idx = (pose_idx + 1) % len(YOGA_POSES)
                    show_result = False
                    result_text = ""
            # Draw background
            screen.fill((220, 240, 255))
            # Draw pose name and description
            pose_surf = font.render(f"Pose: {YOGA_POSES[pose_idx]['name']}", True, (0,0,0))
            desc_surf = font.render(YOGA_POSES[pose_idx]['desc'], True, (100,100,100))
            screen.blit(pose_surf, (WIDTH//2 - pose_surf.get_width()//2, 80))
            screen.blit(desc_surf, (WIDTH//2 - desc_surf.get_width()//2, 140))
            # Draw score
            score_surf = font.render(f"Score: {score}", True, (0,0,0))
            screen.blit(score_surf, (20, 20))
            # Draw result
            if show_result:
                result_surf = font.render(result_text, True, (0, 180, 0))
                screen.blit(result_surf, (WIDTH//2 - result_surf.get_width()//2, 300))
            # Draw instructions
            instr1 = font.render("Match the pose!", True, (100, 100, 100))
            instr2 = font.render("Press R to restart, Q to quit", True, (100, 100, 100))
            screen.blit(instr1, (20, HEIGHT-80))
            screen.blit(instr2, (20, HEIGHT-40))
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