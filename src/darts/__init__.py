import cv2
import mediapipe as mp
import numpy as np
import math
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

WIDTH, HEIGHT = 600, 600
BG_COLOR = (30, 30, 40)


def draw_dartboard(size=600):
    board = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size//2, size//2)
    for i in range(10, 0, -1):
        radius = int(size/2 * i/10)
        color = (0,0,0) if i%2 else (255,255,255)
        cv2.circle(board, center, radius, color, -1)
    cv2.circle(board, center, int(size/20), (0,0,255), -1)
    return board

def get_score(point, size=600):
    center = np.array([size/2, size/2])
    dist = np.linalg.norm(point - center)
    ring = int(dist / (size/2/10))
    return 10 - ring if 0 <= ring < 10 else 0

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    size = WIDTH
    board = draw_dartboard(size)
    score = 0
    throws = 0
    thrown = False
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Virtual Darts (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 32)
    small_font = pygame.font.SysFont("Arial", 20)
    clock = pygame.time.Clock()
    with tracer.start_as_current_span("darts_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)
            img = cv2.resize(frame, (size,size))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            display = board.copy()
            hand_pos = None
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    x = int(handLms.landmark[8].x * size)
                    y = int(handLms.landmark[8].y * size)
                    hand_pos = (x, y)
                    cv2.circle(display, (x,y), 10, (0,255,0), -1)
                    mp_draw.draw_landmarks(display, handLms, mp_hands.HAND_CONNECTIONS)
                    thumb = handLms.landmark[4]
                    index = handLms.landmark[8]
                    dist_thumb_index = math.hypot((thumb.x-index.x)*size, (thumb.y-index.y)*size)
                    if dist_thumb_index < 30 and not thrown:
                        with tracer.start_as_current_span("throw"):
                            score += get_score(np.array([x,y]), size)
                            throws += 1
                            thrown = True
                    if dist_thumb_index > 40:
                        thrown = False
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            # Convert OpenCV BGR to RGB for Pygame
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(display_rgb))
            screen.blit(surf, (0, 0))
            score_text = font.render(f"Score: {score}", True, (0,255,0))
            throws_text = font.render(f"Throws: {throws}", True, (0,255,0))
            screen.blit(score_text, (10, 10))
            screen.blit(throws_text, (10, 50))
            instr = small_font.render("Pinch thumb and index to throw. Press 'q' to quit.", True, (255,255,255))
            screen.blit(instr, (10, HEIGHT - 40))
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
