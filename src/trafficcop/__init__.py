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
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traffic Cop (CV+Pygame)")
font = pygame.font.SysFont("Arial", 36)
clock = pygame.time.Clock()

# Helper: detect open hand (all fingers up)
def is_open_hand(hand_landmarks):
    if not hand_landmarks:
        return False
    tips = [8, 12, 16, 20]
    return all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y for tip in tips)

# Helper: detect fist (all fingers down)
def is_fist(hand_landmarks):
    if not hand_landmarks:
        return False
    tips = [8, 12, 16, 20]
    return all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip-2].y for tip in tips)

class Car:
    def __init__(self, x, y, color, speed):
        self.x = x
        self.y = y
        self.color = color
        self.speed = speed
        self.passed = False
    def move(self, go):
        if go:
            self.y += self.speed
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x-20, self.y-10, 40, 20))

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    cars = []
    car_timer = 0
    car_interval = 1.5
    score = 0
    go = False
    running = True
    last_gesture = None
    with tracer.start_as_current_span("trafficcop_session"):
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            cars.clear()
                            car_timer = 0
                            score = 0
                            go = False
                            last_gesture = None
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            gesture = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                if is_open_hand(hand_landmarks):
                    gesture = "stop"
                elif is_fist(hand_landmarks):
                    gesture = "go"
            # Change light state on gesture
            if gesture and gesture != last_gesture:
                with tracer.start_as_current_span(f"gesture_{gesture}"):
                    if gesture == "go":
                        go = True
                        with tracer.start_as_current_span("light_green"):
                            pass
                    elif gesture == "stop":
                        go = False
                        with tracer.start_as_current_span("light_red"):
                            pass
                last_gesture = gesture
            # Spawn cars
            now = time.time()
            if now - car_timer > car_interval:
                x = random.randint(200, WIDTH-200)
                cars.append(Car(x, 0, random.choice([(255,0,0),(0,255,0),(0,0,255),(255,255,0)]), random.randint(4,8)))
                car_timer = now
            # Move cars
            for car in cars:
                car.move(go)
            # Check for cars passing the line
            for car in cars:
                if not car.passed and car.y > HEIGHT-100:
                    with tracer.start_as_current_span("car_pass"):
                        score += 1
                        car.passed = True
            # Remove cars out of screen
            cars = [car for car in cars if car.y < HEIGHT+20]
            # Draw background
            screen.fill((60, 60, 60))
            # Draw stop/go light
            pygame.draw.circle(screen, (0,255,0) if go else (255,0,0), (WIDTH//2, 80), 40)
            light_text = font.render("GO" if go else "STOP", True, (0,255,0) if go else (255,0,0))
            screen.blit(light_text, (WIDTH//2-50, 140))
            # Draw line
            pygame.draw.line(screen, (255,255,255), (0, HEIGHT-100), (WIDTH, HEIGHT-100), 5)
            # Draw cars
            for car in cars:
                car.draw(screen)
            # Draw score
            score_surf = font.render(f"Score: {score}", True, (255,255,255))
            screen.blit(score_surf, (20, 20))
            # Draw instructions
            instr1 = font.render("Open hand = STOP, Fist = GO", True, (200, 200, 200))
            instr2 = font.render("Press R to restart, Q to quit", True, (200, 200, 200))
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