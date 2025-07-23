import cv2
import mediapipe as mp
import time
import numpy as np
import pygame
import sys
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

WIDTH, HEIGHT = 800, 600
BG_COLOR = (30, 30, 40)

mp_hands = mp.solutions.hands
pygame.mixer.init()

# Generate a tone using pygame
SAMPLE_RATE = 44100

def play_tone(freq, duration=0.05):
    n_samples = int(SAMPLE_RATE * duration)
    buf = np.array([4096 * np.sin(2.0 * np.pi * freq * x / SAMPLE_RATE) for x in range(n_samples)]).astype(np.int16)
    sound = pygame.sndarray.make_sound(buf)
    sound.play()
    time.sleep(duration)
    sound.stop()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sound Conductor (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 32)
    small_font = pygame.font.SysFont("Arial", 20)
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    pitch = 0
    play_now = False
    with tracer.start_as_current_span("conductor_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            pitch = 0
            play_now = False
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                # Compute horizontal position of index fingertip
                x_norm = hand.landmark[8].x
                pitch = int(200 + x_norm * 800)  # map to 200-1000Hz
                # Count extended fingers
                tips = [8, 12, 16, 20]
                count = 0
                for tip in tips:
                    if hand.landmark[tip].y < hand.landmark[tip - 2].y:
                        count += 1
                # Open palm (>=4 fingers) plays tone
                if count >= 4:
                    play_now = True
            # Play tone if open palm
            if play_now and pitch > 0:
                with tracer.start_as_current_span("play_tone"):
                    play_tone(pitch)
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            screen.fill(BG_COLOR)
            freq_text = font.render(f"Frequency: {pitch} Hz", True, (0,255,0))
            screen.blit(freq_text, (10, 30))
            instr1 = font.render("Open palm to play", True, (255,255,0))
            screen.blit(instr1, (10, 70))
            instr2 = small_font.render("Press 'q' to quit", True, (255,255,255))
            screen.blit(instr2, (10, HEIGHT - 40))
            pygame.display.flip()
            # --- Keyboard quit ---
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                running = False
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
