import os

import cv2
from cvzone.HandTrackingModule import HandDetector

IMAGE_PATH = os.getcwd() + os.path.sep + "Resources" + os.path.sep + "webpage.png"
WEBCAM_INDEX = 0
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_FPS = 60

cap = cv2.VideoCapture(WEBCAM_INDEX)
cap.set(3, SCREEN_WIDTH)
cap.set(4, SCREEN_HEIGHT)
cap.set(5, SCREEN_FPS)

detector = HandDetector(detectionCon=0.8)
startDist = None
scale = 0
cx, cy = 500, 500
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    img1 = cv2.imread(IMAGE_PATH)

    if len(hands) == 2:
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and \
                detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            # print("Zoom Gesture")
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]
            # point 8 is the tip of the index finger
            if startDist is None:
                length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)

                startDist = length

            length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)

            scale = int((length - startDist) // 2)
            cx, cy = info[4:]
            print(scale)
    else:
        startDist = None

    try:
        h1, w1, _ = img1.shape
        newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
        img1 = cv2.resize(img1, (newW, newH))

        img[cy - newH // 2:cy + newH // 2, cx - newW // 2:cx + newW // 2] = img1
    except:
        pass

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break  # esc to quit
    elif key == ord('s'):
        cv2.imwrite('saved.png', img)
    elif key == ord('r'):
        cv2.imwrite('saved.png', img)
        img = cv2.imread('saved.png')
    elif key == ord('z'):
        scale = 0
    elif key == ord('x'):
        scale = -1
    elif key == ord('c'):
        scale = 1
    elif key == ord('v'):
        scale = 2
    elif key == ord('b'):
        scale = 3
    elif key == ord('n'):
        scale = 4
    elif key == ord('m'):
        scale = 5
    elif key == ord('q'):
        break
