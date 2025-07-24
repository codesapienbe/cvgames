import cv2
import mediapipe as mp
import numpy as np
import pygame
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
WIDTH, HEIGHT = 600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Virtual Calculator (CV+Pygame)")
font = pygame.font.SysFont("Arial", 36)
clock = pygame.time.Clock()

BUTTONS = [
    ["7", "8", "9", "/"],
    ["4", "5", "6", "*"],
    ["1", "2", "3", "-"],
    ["0", ".", "=", "+"],
    ["C"]
]

class Button:
    def __init__(self, x, y, w, h, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
    def draw(self, screen, font):
        pygame.draw.rect(screen, (200, 200, 200), self.rect)
        pygame.draw.rect(screen, (100, 100, 100), self.rect, 2)
        text = font.render(self.label, True, (0,0,0))
        screen.blit(text, (self.rect.x + self.rect.w//2 - text.get_width()//2, self.rect.y + self.rect.h//2 - text.get_height()//2))
    def is_pressed(self, x, y):
        return self.rect.collidepoint(x, y)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    # Layout buttons
    buttons = []
    margin = 20
    btn_w = (WIDTH - margin*2) // 4
    btn_h = 100
    for i, row in enumerate(BUTTONS):
        for j, label in enumerate(row):
            x = margin + j*btn_w
            y = 200 + i*btn_h
            buttons.append(Button(x, y, btn_w-10, btn_h-10, label))
    equation = ""
    result = ""
    last_press_time = 0
    press_cooldown = 0.7
    running = True
    with tracer.start_as_current_span("virtualcalculator_session"):
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            equation = ""
                            result = ""
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            now = time.time()
            press = None
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                x = int(lm[8].x * WIDTH)
                y = int(lm[8].y * HEIGHT)
                for btn in buttons:
                    if btn.is_pressed(x, y) and now - last_press_time > press_cooldown:
                        press = btn.label
                        last_press_time = now
                        with tracer.start_as_current_span("button_press"):
                            if press == "C":
                                with tracer.start_as_current_span("clear"):
                                    equation = ""
                                    result = ""
                            elif press == "=":
                                try:
                                    with tracer.start_as_current_span("calculation"):
                                        result = str(eval(equation))
                                except:
                                    result = "Error"
                            else:
                                equation += press
            # Draw background
            screen.fill((240, 240, 255))
            # Draw display
            disp = equation if not result else result
            disp_surf = font.render(disp, True, (0,0,0))
            screen.blit(disp_surf, (margin, 100))
            # Draw buttons
            for btn in buttons:
                btn.draw(screen, font)
            # Draw instructions
            instr1 = font.render("Point finger to press buttons", True, (100, 100, 100))
            instr2 = font.render("Press R to restart, Q to quit", True, (100, 100, 100))
            screen.blit(instr1, (20, HEIGHT-80))
            screen.blit(instr2, (20, HEIGHT-40))
            # Show webcam preview (small)
            frame_small = cv2.resize(frame, (120, 90))
            frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_small))
            screen.blit(surf, (WIDTH-130, HEIGHT-100))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 