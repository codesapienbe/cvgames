import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import sys
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

def detect_hand_gesture(hand_landmarks, mp_hands):
    if not hand_landmarks:
        return None
    # Open palm: distance between thumb tip and pinky tip large
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    dist = np.linalg.norm(np.array([thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y]))
    if dist > 0.4:
        return "open"
    elif dist < 0.15:
        return "fist"
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
    pygame.display.set_caption("Pet Simulator (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    clock = pygame.time.Clock()

    # Pet state
    hunger = 5
    happiness = 5
    max_state = 10
    last_feed_time = time.time()
    last_play_time = time.time()
    game_time = 60  # seconds
    start_time = time.time()
    action = None
    action_time = 0
    with tracer.start_as_current_span("petsimulator_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            hand_landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None
            gesture = detect_hand_gesture(hand_landmarks, mp_hands) if hand_landmarks else None
            # Handle pet interaction
            now = time.time()
            if gesture == "open" and now - last_feed_time > 2:
                with tracer.start_as_current_span("feed"):
                    hunger = min(max_state, hunger + 2)
                    action = "Fed!"
                    action_time = now
                last_feed_time = now
            elif gesture == "fist" and now - last_play_time > 2:
                with tracer.start_as_current_span("play"):
                    happiness = min(max_state, happiness + 2)
                    action = "Played!"
                    action_time = now
                last_play_time = now
            # Pet gets hungry/sad over time
            if now - last_feed_time > 5:
                hunger = max(0, hunger - 1)
                last_feed_time = now
            if now - last_play_time > 7:
                happiness = max(0, happiness - 1)
                last_play_time = now
            # Draw UI
            frame_disp = frame.copy()
            if hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_disp, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            elapsed = int(now - start_time)
            remains = max(0, game_time - elapsed)
            cv2.putText(frame_disp, f"Hunger: {hunger}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,200,255), 3)
            cv2.putText(frame_disp, f"Happiness: {happiness}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            cv2.putText(frame_disp, f"Time: {remains}s", (width-220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
            cv2.putText(frame_disp, "Show OPEN PALM to feed, FIST to play!", (80, height-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            if action and now - action_time < 2:
                cv2.putText(frame_disp, action, (width//2-100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)
            if hunger == 0:
                cv2.putText(frame_disp, "Pet is HUNGRY!", (width//2-180, height//2+100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
            if happiness == 0:
                cv2.putText(frame_disp, "Pet is SAD!", (width//2-180, height//2+160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            hunger = 5
                            happiness = 5
                            last_feed_time = time.time()
                            last_play_time = time.time()
                            start_time = time.time()
                            action = None
                            action_time = 0
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