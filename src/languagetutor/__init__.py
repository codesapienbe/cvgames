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
    pygame.display.set_caption("Language Tutor (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    clock = pygame.time.Clock()

    # Questions and answers (left hand = A, right hand = B)
    questions = [
        ("What is the capital of France?", "A) Paris", "B) London", "A"),
        ("Which is a fruit?", "A) Carrot", "B) Apple", "B"),
        ("What color is the sky?", "A) Blue", "B) Green", "A"),
        ("Which is an animal?", "A) Table", "B) Dog", "B"),
        ("What is 2+2?", "A) 4", "B) 5", "A"),
    ]
    random.shuffle(questions)
    q_index = 0
    score = 0
    game_time = 60  # seconds
    start_time = time.time()
    answer_given = False
    answer_result = None
    answer_time = 0
    with tracer.start_as_current_span("languagetutor_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            pose_landmarks = results.pose_landmarks
            left_hand_up = False
            right_hand_up = False
            if pose_landmarks:
                left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                if left_wrist.y < left_shoulder.y:
                    left_hand_up = True
                if right_wrist.y < right_shoulder.y:
                    right_hand_up = True
            # Handle answer
            if not answer_given and (left_hand_up or right_hand_up):
                correct = questions[q_index][3]
                if left_hand_up:
                    answer = "A"
                elif right_hand_up:
                    answer = "B"
                else:
                    answer = None
                if answer:
                    with tracer.start_as_current_span("answer"):
                        answer_given = True
                        answer_time = time.time()
                        if answer == correct:
                            answer_result = True
                            score += 1
                            with tracer.start_as_current_span("correct"):
                                pass
                        else:
                            answer_result = False
                            with tracer.start_as_current_span("incorrect"):
                                pass
            # Draw UI
            frame_disp = frame.copy()
            if pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_disp, pose_landmarks, mp_pose.POSE_CONNECTIONS)
            elapsed = int(time.time() - start_time)
            remains = max(0, game_time - elapsed)
            q, a1, a2, correct = questions[q_index]
            cv2.putText(frame_disp, f"Score: {score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            cv2.putText(frame_disp, f"Time: {remains}s", (width-220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
            cv2.putText(frame_disp, q, (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 2)
            cv2.putText(frame_disp, a1, (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,255), 2)
            cv2.putText(frame_disp, a2, (120, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,255), 2)
            cv2.putText(frame_disp, "Raise LEFT hand for A, RIGHT hand for B", (80, height-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            if answer_given:
                if answer_result:
                    cv2.putText(frame_disp, "Correct!", (width//2-100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                else:
                    cv2.putText(frame_disp, "Incorrect!", (width//2-120, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                if time.time() - answer_time > 2:
                    answer_given = False
                    answer_result = None
                    q_index = (q_index + 1) % len(questions)
                    with tracer.start_as_current_span("question_shown"):
                        pass
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
                            q_index = 0
                            answer_given = False
                            answer_result = None
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