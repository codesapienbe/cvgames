import cv2
import time
import PoseModule
import mediapipe as mp
import numpy as np

WEBCAM_INDEX = 0
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_FPS = 60

cap = cv2.VideoCapture(WEBCAM_INDEX)
cap.set(3, SCREEN_WIDTH)
cap.set(4, SCREEN_HEIGHT)
cap.set(5, SCREEN_FPS)

pTime = 0
detector = PoseModule.poseDetector()
while True:
    success, img = cap.read()
    img = cv2.resize(img, (960, 540))
    img = detector.findPose(img)
    lmList = detector.getPosition(img)

    length = lmList[19][2]
    per = np.interp(int(length), [157, 285], [100, 0])
    poseBar = np.interp(length, [157, 285], [150, 400])

    cv2.putText(img, f"{int(per)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
    cv2.rectangle(img, (50, int(poseBar)), (85, 400), (255, 0, 255), cv2.FILLED)
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (425, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
