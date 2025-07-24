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
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Wireframe (CV+Pygame)")
font = pygame.font.SysFont("Arial", 36)
clock = pygame.time.Clock()

POINT_RADIUS = 18
DRAG_DIST = 30

# Initial wireframe shape (triangle)
wireframe_points = [
    [300, 200],
    [600, 200],
    [450, 400]
]

selected_idx = None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    global selected_idx
    running = True
    with tracer.start_as_current_span("wireframe_session"):
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            global wireframe_points
                            wireframe_points = [[300, 200], [600, 200], [450, 400]]
                            selected_idx = None
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            finger_pos = None
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                x = int(lm[8].x * WIDTH)
                y = int(lm[8].y * HEIGHT)
                finger_pos = (x, y)
                # Select or drag point
                if selected_idx is None:
                    for i, pt in enumerate(wireframe_points):
                        if np.hypot(x - pt[0], y - pt[1]) < DRAG_DIST:
                            with tracer.start_as_current_span("point_select"):
                                selected_idx = i
                                break
                else:
                    with tracer.start_as_current_span("drag"):
                        wireframe_points[selected_idx][0] = x
                        wireframe_points[selected_idx][1] = y
                    # Release if finger is far from point
                    if np.hypot(x - wireframe_points[selected_idx][0], y - wireframe_points[selected_idx][1]) > DRAG_DIST*2:
                        with tracer.start_as_current_span("release"):
                            selected_idx = None
            else:
                selected_idx = None
            # Draw background
            screen.fill((240, 240, 255))
            # Draw wireframe
            for i, pt in enumerate(wireframe_points):
                pygame.draw.circle(screen, (0, 100, 255), pt, POINT_RADIUS)
                pygame.draw.circle(screen, (0, 0, 0), pt, POINT_RADIUS, 2)
                next_pt = wireframe_points[(i+1)%len(wireframe_points)]
                pygame.draw.line(screen, (0,0,0), pt, next_pt, 4)
            # Draw finger cursor
            if finger_pos:
                pygame.draw.circle(screen, (0,255,255), finger_pos, 15)
            # Draw instructions
            instr1 = font.render("Drag points with your finger!", True, (100, 100, 100))
            instr2 = font.render("Press R to reset, Q to quit", True, (100, 100, 100))
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
