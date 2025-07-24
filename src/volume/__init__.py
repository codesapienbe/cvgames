import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
import sys

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

wCam, hCam = 640, 480
WEBCAM_INDEX = 0
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_FPS = 60

cap = cv2.VideoCapture(WEBCAM_INDEX)
cap.set(3, SCREEN_WIDTH)
cap.set(4, SCREEN_HEIGHT)
cap.set(5, SCREEN_FPS)

pTime = 0

detector = htm.handDetector(detectionCon=0.7, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 0, 0)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Hand Volume Control (CV+Pygame)")
font = pygame.font.SysFont("Arial", 36)
clock = pygame.time.Clock()

with tracer.start_as_current_span("volume_session"):
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    with tracer.start_as_current_span("mute"):
                        volume.SetMute(True, None)
                elif event.key == pygame.K_u:
                    with tracer.start_as_current_span("unmute"):
                        volume.SetMute(False, None)
                elif event.key == pygame.K_r:
                    with tracer.start_as_current_span("reset"):
                        vol = 0
                        volume.SetMasterVolumeLevelScalar(vol / 100, None)
                elif event.key == pygame.K_f:
                    with tracer.start_as_current_span("full"):
                        vol = 100
                        volume.SetMasterVolumeLevelScalar(vol / 100, None)
        success, img = cap.read()
        if not success:
            break
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=True)
        if len(lmList) != 0:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
            if 250 < area < 1000:
                length, img, lineInfo = detector.findDistance(4, 8, img)
                volBar = np.interp(length, [50, 200], [400, 150])
                volPer = np.interp(length, [50, 200], [0, 100])
                smoothness = 10
                volPer = smoothness * round(volPer / smoothness)
                fingers = detector.fingersUp()
                if not fingers[4]:
                    with tracer.start_as_current_span("volume_change"):
                        volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    colorVol = (0, 255, 0)
                else:
                    colorVol = (255, 0, 0)
        # Drawings
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
        cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 3)
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        # --- Pygame Rendering ---
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)
cap.release()
pygame.quit()
sys.exit()
