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

def detect_swipe(hand_landmarks, prev_landmarks, width):
    if not hand_landmarks or not prev_landmarks:
        return None
    # Use index finger tip
    idx_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    prev_idx_tip = prev_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    dx = idx_tip.x - prev_idx_tip.x
    if abs(dx) > 0.15:
        return 'right' if dx > 0 else 'left'
    return None

def main():
    # Set up OpenTelemetry tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)

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
    pygame.display.set_caption("Magic Spells Casting (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    clock = pygame.time.Clock()

    # Spells: swipe left, swipe right, open palm, fist
    spells = [
        ("Swipe Left", "left"),
        ("Swipe Right", "right"),
        ("Open Palm", "open"),
        ("Fist", "fist"),
    ]
    random.shuffle(spells)
    spell_index = 0
    score = 0
    game_time = 45  # seconds
    start_time = time.time()
    prev_hand_landmarks = None
    spell_cast = False
    cast_result = None
    cast_time = 0
    with tracer.start_as_current_span("magicspellscasting_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            hand_landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None
            # Detect gesture
            gesture = None
            if hand_landmarks:
                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Detect swipe
                swipe = detect_swipe(hand_landmarks, prev_hand_landmarks, width)
                if swipe:
                    gesture = swipe
                else:
                    # Open palm: distance between thumb tip and pinky tip large
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    dist = np.linalg.norm(np.array([thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y]))
                    if dist > 0.4:
                        gesture = "open"
                    # Fist: distance between thumb tip and pinky tip small
                    elif dist < 0.15:
                        gesture = "fist"
            # Handle spell casting
            if not spell_cast and gesture:
                spell_name, spell_gesture = spells[spell_index]
                with tracer.start_as_current_span("spell_cast"):
                    spell_cast = True
                    cast_time = time.time()
                    if gesture == spell_gesture:
                        cast_result = True
                        score += 1
                        with tracer.start_as_current_span("correct"):
                            pass
                    else:
                        cast_result = False
                        with tracer.start_as_current_span("incorrect"):
                            pass
            # Draw UI
            frame_disp = frame.copy()
            elapsed = int(time.time() - start_time)
            remains = max(0, game_time - elapsed)
            cv2.putText(frame_disp, f"Score: {score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            cv2.putText(frame_disp, f"Time: {remains}s", (width-220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
            spell_name, spell_gesture = spells[spell_index]
            cv2.putText(frame_disp, f"Cast: {spell_name}", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 2)
            cv2.putText(frame_disp, "Swipe or gesture with your hand!", (80, height-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            if spell_cast:
                if cast_result:
                    cv2.putText(frame_disp, "Correct!", (width//2-100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                else:
                    cv2.putText(frame_disp, "Incorrect!", (width//2-120, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                if time.time() - cast_time > 2:
                    spell_cast = False
                    cast_result = None
                    spell_index = (spell_index + 1) % len(spells)
            prev_hand_landmarks = hand_landmarks
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
                            spell_index = 0
                            spell_cast = False
                            cast_result = None
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