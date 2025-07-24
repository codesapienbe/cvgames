import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
import numpy as np
import sys

WEBCAM_INDEX = 0
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_FPS = 60

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

cap = cv2.VideoCapture(WEBCAM_INDEX)
cap.set(3, SCREEN_WIDTH)
cap.set(4, SCREEN_HEIGHT)
cap.set(5, SCREEN_FPS)

success, img = cap.read()
if not success:
    print("Error: Could not read from camera.")
    sys.exit(1)

imgFront = cv2.imread("Resources/logo.png", cv2.IMREAD_UNCHANGED)
imgFront = cv2.resize(imgFront, (0, 0), None, 0.3, 0.3)

hf, wf, cf = imgFront.shape
hb, wb, cb = img.shape

fpsReader = cvzone.FPS()
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((wb, hb))
pygame.display.set_caption("Overlay Demo (CV+Pygame)")
font = pygame.font.SysFont("Arial", 24)
clock = pygame.time.Clock()

with tracer.start_as_current_span("overlay_session"):
    running = True
    while running:
        success, img = cap.read()
        if not success:
            break
        hands, img = detector.findHands(img)  # With Draw
        hand_found = False
        if hands:
            hand_found = True
        imgResult = cvzone.overlayPNG(img, imgFront, [0, hb - hf])
        _, imgResult = fpsReader.update(imgResult)
        # --- OpenTelemetry logging ---
        if hand_found:
            with tracer.start_as_current_span("hand_detected"):
                pass
        with tracer.start_as_current_span("overlay_shown"):
            pass
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        # --- Pygame Rendering ---
        frame_rgb = cv2.cvtColor(imgResult, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)
cap.release()
pygame.quit()
sys.exit()
