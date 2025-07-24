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
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Spelling Bee (CV+Pygame)")
font = pygame.font.SysFont("Arial", 36)
clock = pygame.time.Clock()

WORDS = ["PYTHON", "COMPUTER", "VISION", "CAMERA", "PROGRAM", "SPELLING", "BEE", "KEYBOARD", "MONITOR", "MOUSE"]

# Helper: detect hand raised (y of wrist < y of shoulder)
def hand_raised(hand_landmarks):
    if not hand_landmarks:
        return False
    wrist = hand_landmarks.landmark[0]
    index = hand_landmarks.landmark[8]
    return wrist.y < index.y - 0.1

# Helper: detect mouth open (distance between upper and lower lip)
def mouth_open(face_landmarks):
    if not face_landmarks:
        return False
    lm = face_landmarks.landmark
    upper = lm[13]  # upper lip
    lower = lm[14]  # lower lip
    return abs(upper.y - lower.y) > 0.04

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    word = random.choice(WORDS)
    user_input = ""
    score = 0
    show_result = False
    result_text = ""
    last_submit_time = 0
    last_next_time = 0
    running = True
    with tracer.start_as_current_span("spellingbee_session"):
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            word = random.choice(WORDS)
                            user_input = ""
                            score = 0
                            show_result = False
                            result_text = ""
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)
            # Detect hand raise to submit answer
            submit = False
            if hand_results.multi_hand_landmarks:
                if hand_raised(hand_results.multi_hand_landmarks[0]):
                    if time.time() - last_submit_time > 1.5 and not show_result:
                        submit = True
                        last_submit_time = time.time()
            # Detect mouth open to go to next word
            next_word = False
            if face_results.multi_face_landmarks:
                if mouth_open(face_results.multi_face_landmarks[0]):
                    if time.time() - last_next_time > 1.5 and show_result:
                        next_word = True
                        last_next_time = time.time()
            # Handle answer submission
            if submit:
                with tracer.start_as_current_span("answer"):
                    if user_input.upper() == word:
                        with tracer.start_as_current_span("correct"):
                            score += 1
                            result_text = "Correct!"
                    else:
                        with tracer.start_as_current_span("incorrect"):
                            result_text = f"Incorrect! ({word})"
                    show_result = True
            # Handle next word
            if next_word:
                with tracer.start_as_current_span("next_word"):
                    word = random.choice(WORDS)
                    user_input = ""
                    show_result = False
                    result_text = ""
            # Handle keyboard input
            keys = pygame.key.get_pressed()
            for i in range(pygame.K_a, pygame.K_z+1):
                if keys[i]:
                    char = chr(i).upper()
                    if not show_result and (len(user_input) < len(word)):
                        user_input += char
                        time.sleep(0.1)
            if keys[pygame.K_BACKSPACE] and not show_result:
                user_input = user_input[:-1]
                time.sleep(0.1)
            # Draw background
            screen.fill((255, 255, 200))
            # Draw word (as blanks)
            blanks = " ".join([c if i < len(user_input) else "_" for i, c in enumerate(word)])
            word_surf = font.render(blanks, True, (0,0,0))
            screen.blit(word_surf, (WIDTH//2 - word_surf.get_width()//2, 120))
            # Draw user input
            input_surf = font.render(user_input, True, (0, 100, 200))
            screen.blit(input_surf, (WIDTH//2 - input_surf.get_width()//2, 200))
            # Draw instructions
            instr1 = font.render("Raise hand to submit", True, (100, 100, 100))
            instr2 = font.render("Open mouth for next word", True, (100, 100, 100))
            instr3 = font.render("Type with keyboard", True, (100, 100, 100))
            screen.blit(instr1, (50, HEIGHT-120))
            screen.blit(instr2, (50, HEIGHT-80))
            screen.blit(instr3, (50, HEIGHT-40))
            # Draw score
            score_surf = font.render(f"Score: {score}", True, (0,0,0))
            screen.blit(score_surf, (WIDTH-200, 20))
            # Draw result
            if show_result:
                result_surf = font.render(result_text, True, (0, 180, 0) if result_text.startswith("Correct") else (200, 0, 0))
                screen.blit(result_surf, (WIDTH//2 - result_surf.get_width()//2, 300))
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