import cv2
import mediapipe as mp
import numpy as np
import time
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

COLORS = [
    (0, 0, 0), (128, 128, 128), (136, 0, 21), (237, 28, 36),
    (255, 127, 39), (255, 242, 0), (34, 177, 76), (0, 162, 232),
    (63, 72, 204), (163, 73, 164), (255, 255, 255)
]

def main():
    # Pygame setup
    pygame.init()
    width, height = 1280, 720
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Air Painter (CV+Pygame)")
    canvas = pygame.Surface((width, height))
    canvas.fill((255, 255, 255))
    font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()

    TOOLBAR_WIDTH = 80
    PALETTE_HEIGHT = 60
    COLOR_SWATCH_SIZE = 40
    PADDING = 10
    MARGIN = 5

    selected_color = (255, 0, 0)
    selected_tool = "brush"
    last_pos = None
    drawing = False
    erasing = False

    # OpenCV + MediaPipe setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    with tracer.start_as_current_span("airpainter_session"):
        running = True
        while running:
            # --- Computer Vision Input ---
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            index_tip = None
            thumb_tip = None
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                index_tip = hand.landmark[8]
                thumb_tip = hand.landmark[4]
                x, y = int(index_tip.x * width), int(index_tip.y * height)
                distance = np.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)
                if distance < 0.05:
                    if selected_tool == "brush":
                        with tracer.start_as_current_span("draw"):
                            pygame.draw.circle(canvas, selected_color, (x, y), 10)
                    elif selected_tool == "eraser":
                        with tracer.start_as_current_span("erase"):
                            pygame.draw.circle(canvas, (255, 255, 255), (x, y), 20)
                # Tool selection
                if x < TOOLBAR_WIDTH:
                    for i, tool in enumerate(["brush", "eraser", "clear"]):
                        y_tool = PADDING + i * (COLOR_SWATCH_SIZE + MARGIN)
                        if y_tool < y < y_tool + COLOR_SWATCH_SIZE:
                            with tracer.start_as_current_span(f"select_tool_{tool}"):
                                selected_tool = tool
                                if tool == "clear":
                                    canvas.fill((255, 255, 255))
                # Color selection
                elif y > height - PALETTE_HEIGHT:
                    for i, color in enumerate(COLORS):
                        x_color = PADDING + i * (COLOR_SWATCH_SIZE + MARGIN)
                        if x_color < x < x_color + COLOR_SWATCH_SIZE:
                            with tracer.start_as_current_span(f"select_color_{color}"):
                                selected_color = color
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            screen.fill((245, 245, 245))
            screen.blit(canvas, (0, 0))
            # Toolbar
            pygame.draw.rect(screen, (240, 240, 240), (0, 0, TOOLBAR_WIDTH, height))
            for i, tool in enumerate(["brush", "eraser", "clear"]):
                y_tool = PADDING + i * (COLOR_SWATCH_SIZE + MARGIN)
                color = (0, 0, 0) if tool != selected_tool else selected_color
                text = font.render(tool, True, color)
                screen.blit(text, (PADDING, y_tool))
            # Palette
            pygame.draw.rect(screen, (240, 240, 240), (0, height - PALETTE_HEIGHT, width, PALETTE_HEIGHT))
            for i, color in enumerate(COLORS):
                x_color = PADDING + i * (COLOR_SWATCH_SIZE + MARGIN)
                pygame.draw.rect(screen, color, (x_color, height - PALETTE_HEIGHT + PADDING, COLOR_SWATCH_SIZE, COLOR_SWATCH_SIZE))
                if color == selected_color:
                    pygame.draw.rect(screen, (0, 0, 0), (x_color, height - PALETTE_HEIGHT + PADDING, COLOR_SWATCH_SIZE, COLOR_SWATCH_SIZE), 3)
            pygame.display.flip()
            clock.tick(60)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
