import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

################################
wCam, hCam = 640, 480
################################
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
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 0, 0)

while True:
    success, img = cap.read()

    # Find Hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:

        # Filter based on size
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        # print(area)
        if 250 < area < 1000:

            # Find Distance between index and Thumb
            length, img, lineInfo = detector.findDistance(4, 8, img)
            # print(length)

            # Convert Volume
            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])

            # Reduce Resolution to make it smoother
            smoothness = 10
            volPer = smoothness * round(volPer / smoothness)

            # Check fingers up
            fingers = detector.fingersUp()
            # print(fingers)

            # If pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255, 0, 0)

    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, colorVol, 3)

    # Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('img.png', img)
    elif key == ord('m'):
        volume.SetMute(True, None)
    elif key == ord('u'):
        volume.SetMute(False, None)
    elif key == ord('+'):
        vol += 1
        volume.SetMasterVolumeLevelScalar(vol / 100, None)
    elif key == ord('-'):
        vol -= 1
        volume.SetMasterVolumeLevelScalar(vol / 100, None)
    elif key == ord('r'):
        vol = 0
        volume.SetMasterVolumeLevelScalar(vol / 100, None)
    elif key == ord('f'):
        vol = 100
        volume.SetMasterVolumeLevelScalar(vol / 100, None)
    elif key == ord('1'):
        vol = 1
        volume.SetMasterVolumeLevelScalar(vol / 100, None)
    elif key == ord('2'):
        vol = 2
        volume.SetMasterVolumeLevelScalar(vol / 100, None)
    elif key == ord('3'):
        vol = 3
        volume.SetMasterVolumeLevelScalar(vol / 100, None)
