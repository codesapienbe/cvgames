import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
import numpy as np
import sys

IMAGE_PATH = os.getcwd() + os.path.sep + "Resources" + os.path.sep + "webpage.png"
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

detector = HandDetector(detectionCon=0.8)
startDist = None
scale = 0
cx, cy = 500, 500

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Resizer (CV+Pygame)")
font = pygame.font.SysFont("Arial", 24)
clock = pygame.time.Clock()

with tracer.start_as_current_span("resizer_session"):
    running = True
    while running:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        img1 = cv2.imread(IMAGE_PATH)
        zoomed = False
        if len(hands) == 2:
            if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and \
                    detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
                lmList1 = hands[0]["lmList"]
                lmList2 = hands[1]["lmList"]
                if startDist is None:
                    length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                    startDist = length
                length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                scale = int((length - startDist) // 2)
                cx, cy = info[4:]
                zoomed = True
                with tracer.start_as_current_span("zoom_gesture"):
                    pass
        else:
            startDist = None
        try:
            h1, w1, _ = img1.shape
            newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
            img1 = cv2.resize(img1, (newW, newH))
            img[cy - newH // 2:cy + newH // 2, cx - newW // 2:cx + newW // 2] = img1
        except:
            pass
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_s:
                    with tracer.start_as_current_span("save"):
                        cv2.imwrite('saved.png', img)
                elif event.key == pygame.K_r:
                    with tracer.start_as_current_span("reset"):
                        cv2.imwrite('saved.png', img)
                        img = cv2.imread('saved.png')
                elif event.key == pygame.K_z:
                    scale = 0
                elif event.key == pygame.K_x:
                    scale = -1
                elif event.key == pygame.K_c:
                    scale = 1
                elif event.key == pygame.K_v:
                    scale = 2
                elif event.key == pygame.K_b:
                    scale = 3
                elif event.key == pygame.K_n:
                    scale = 4
                elif event.key == pygame.K_m:
                    scale = 5
        # --- Pygame Rendering ---
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()
