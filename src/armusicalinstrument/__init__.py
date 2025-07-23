import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

WIDTH, HEIGHT = 800, 600
NUM_ZONES = 7
ZONE_HEIGHT = HEIGHT // NUM_ZONES
ZONE_COLORS = [(255, 200, 200), (255, 255, 200), (200, 255, 200), (200, 255, 255), (200, 200, 255), (255, 200, 255), (240, 240, 240)]
NOTE_NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

mp_hands = mp.solutions.hands

# Use pygame's built-in sound for simple tones
pygame.mixer.init()
FREQUENCIES = [261, 294, 329, 349, 392, 440, 493]  # C4, D4, E4, F4, G4, A4, B4

def play_tone(freq, duration=0.2):
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    buf = np.array([4096 * np.sin(2.0 * np.pi * freq * x / sample_rate) for x in range(n_samples)]).astype(np.int16)
    sound = pygame.sndarray.make_sound(buf)
    sound.play()
    time.sleep(duration)
    sound.stop()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Arm Musical Instrument (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 32)
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    last_zone = -1
    last_note_time = 0
    with tracer.start_as_current_span("arm_instrument_session"):
        running = True
        while running:
            # --- Computer Vision Input ---
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            hand_y = None
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                index_tip = hand.landmark[8]
                hand_y = int(index_tip.y * HEIGHT)
            # --- Detect zone and play note ---
            zone = -1
            if hand_y is not None:
                zone = min(hand_y // ZONE_HEIGHT, NUM_ZONES - 1)
                if zone != last_zone and time.time() - last_note_time > 0.3:
                    with tracer.start_as_current_span(f"play_note_{NOTE_NAMES[zone]}"):
                        play_tone(FREQUENCIES[zone])
                        last_note_time = time.time()
                    last_zone = zone
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            screen.fill((30, 30, 30))
            for i in range(NUM_ZONES):
                color = ZONE_COLORS[i]
                rect = pygame.Rect(0, i * ZONE_HEIGHT, WIDTH, ZONE_HEIGHT)
                pygame.draw.rect(screen, color, rect)
                note_text = font.render(NOTE_NAMES[i], True, (0,0,0))
                screen.blit(note_text, (20, i * ZONE_HEIGHT + ZONE_HEIGHT // 2 - note_text.get_height() // 2))
            # Highlight current zone
            if zone >= 0:
                pygame.draw.rect(screen, (255, 100, 100), (0, zone * ZONE_HEIGHT, WIDTH, ZONE_HEIGHT), 5)
            instr = font.render("Move your hand up/down to play notes!", True, (0,0,0))
            screen.blit(instr, (20, HEIGHT - 50))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 