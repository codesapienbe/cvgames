import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

INSTRUMENTS = ["Piano", "Violin", "Drums", "Flute"]

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

class SoundConductor:
    def __init__(self):
        self.tempo = 120
        self.volume = 50
        self.instrument = 0
        self.last_gesture = None
        self.last_gesture_time = 0
    def update(self, gesture):
        if gesture == "up":
            self.tempo = min(200, self.tempo + 5)
        elif gesture == "down":
            self.tempo = max(40, self.tempo - 5)
        elif gesture == "left":
            self.instrument = (self.instrument - 1) % len(INSTRUMENTS)
        elif gesture == "right":
            self.instrument = (self.instrument + 1) % len(INSTRUMENTS)
        elif gesture == "open":
            self.volume = min(100, self.volume + 5)
        elif gesture == "fist":
            self.volume = max(0, self.volume - 5)
    def draw(self, frame):
        cv2.putText(frame, f"Tempo: {self.tempo} BPM", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Volume: {self.volume}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Instrument: {INSTRUMENTS[self.instrument]}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, "Gestures: Up/Down=Tempo, Left/Right=Instrument, Open=Vol+, Fist=Vol-", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press q to quit", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def detect_gesture(hand_landmarks):
    if not hand_landmarks:
        return None
    wrist = hand_landmarks.landmark[0]
    index = hand_landmarks.landmark[8]
    dx = index.x - wrist.x
    dy = index.y - wrist.y
    if abs(dx) > abs(dy):
        if dx > 0.1:
            return "right"
        elif dx < -0.1:
            return "left"
    else:
        if dy < -0.1:
            return "up"
        elif dy > 0.1:
            return "down"
    tips = [8, 12, 16, 20]
    if all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y for tip in tips):
        return "open"
    if all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip-2].y for tip in tips):
        return "fist"
    return None

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    game = SoundConductor()
    cap = cv2.VideoCapture(0)
    WIDTH, HEIGHT = 800, 600
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sound Conductor (CV+Pygame)")
    clock = pygame.time.Clock()
    with tracer.start_as_current_span("soundconductor_session"):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            game.__init__()
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            gesture = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks)
                if gesture:
                    with tracer.start_as_current_span(f"gesture_{gesture}"):
                        game.update(gesture)
            game.draw(frame)
            # --- Pygame Rendering ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    import sys
    sys.exit()

if __name__ == "__main__":
    main() 