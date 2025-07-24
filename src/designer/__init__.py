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

WIDTH, HEIGHT = 1200, 800
BG_COLOR = (245, 245, 255)
ELEMENT_COLORS = [(255, 200, 200), (200, 255, 200), (200, 200, 255)]

mp_hands = mp.solutions.hands

class Element:
    def __init__(self, kind, x, y):
        self.kind = kind  # 'image', 'text', 'video'
        self.x = x
        self.y = y
        self.width = 160
        self.height = 120
        self.selected = False
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
    def draw(self, surface, font):
        color = ELEMENT_COLORS[0] if self.kind == 'image' else ELEMENT_COLORS[1] if self.kind == 'text' else ELEMENT_COLORS[2]
        pygame.draw.rect(surface, color, self.rect(), border_radius=12)
        if self.selected:
            pygame.draw.rect(surface, (255, 100, 100), self.rect(), 4, border_radius=12)
        label = font.render(self.kind.capitalize(), True, (0,0,0))
        surface.blit(label, (self.x + 10, self.y + 10))
        if self.kind == 'image':
            pygame.draw.rect(surface, (180,180,180), (self.x+30, self.y+40, 100, 60))
        elif self.kind == 'video':
            pygame.draw.polygon(surface, (100,100,255), [(self.x+60, self.y+50), (self.x+100, self.y+70), (self.x+60, self.y+90)])
        elif self.kind == 'text':
            t = font.render("AaBb", True, (80,80,80))
            surface.blit(t, (self.x+40, self.y+60))

def hand_over_element(hand_pos, element):
    x, y = hand_pos
    return element.rect().collidepoint(x, y)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Website Designer (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 32)
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    elements = [Element('image', 50, 100), Element('text', 50, 250), Element('video', 50, 400)]
    dragging = None
    drag_offset = (0, 0)
    last_drag_time = 0
    with tracer.start_as_current_span("designer_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            hand_pos = None
            pinch = False
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                index_tip = hand.landmark[8]
                thumb_tip = hand.landmark[4]
                hx = int(index_tip.x * WIDTH)
                hy = int(index_tip.y * HEIGHT)
                hand_pos = (hx, hy)
                dist = np.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                pinch = dist < 0.05
            # Drag and drop logic
            if hand_pos:
                if pinch and dragging is None:
                    for el in elements[::-1]:
                        if hand_over_element(hand_pos, el):
                            el.selected = True
                            dragging = el
                            drag_offset = (hand_pos[0] - el.x, hand_pos[1] - el.y)
                            with tracer.start_as_current_span("element_selected"):
                                pass
                            break
                elif not pinch and dragging:
                    dragging.selected = False
                    with tracer.start_as_current_span("element_dropped"):
                        pass
                    dragging = None
                elif dragging and pinch:
                    dragging.x = hand_pos[0] - drag_offset[0]
                    dragging.y = hand_pos[1] - drag_offset[1]
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            screen.fill(BG_COLOR)
            for el in elements:
                el.draw(screen, font)
            instr = font.render("Pinch to grab and move elements! Press 'q' to quit.", True, (0,0,0))
            screen.blit(instr, (30, HEIGHT - 50))
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