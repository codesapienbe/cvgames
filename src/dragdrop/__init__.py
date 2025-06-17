import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import os
import argparse

# Create argument parser
parser = argparse.ArgumentParser(description='Hand tracking with draggable images')
parser.add_argument('--camera', type=int, default=0,
                    help='Camera index (default: 0)')

args = parser.parse_args()

cap = cv2.VideoCapture(args.camera)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(5, 60)

detector = HandDetector(detectionCon=0.8)


class DragImg():
    def __init__(self, path, posOrigin, imgType):

        self.posOrigin = posOrigin
        self.imgType = imgType
        self.path = path

        if self.imgType == 'png':
            self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        else:
            self.img = cv2.imread(self.path)

        # self.img = cv2.resize(self.img, (0,0),None,0.4,0.4)

        self.size = self.img.shape[:2]

    def update(self, cursor):
        ox, oy = self.posOrigin
        h, w = self.size

        # Check if in region
        if ox < cursor[0] < ox + w and oy < cursor[1] < oy + h:
            self.posOrigin = cursor[0] - w // 2, cursor[1] - h // 2


path = "Resources/ImagesPNG"
myList = os.listdir(path)
print(myList)

listImg = []
for x, pathImg in enumerate(myList):
    if 'png' in pathImg:
        imgType = 'png'
    else:
        imgType = 'jpg'
    listImg.append(DragImg(f'{path}/{pathImg}', [50 + x * 300, 50], imgType))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        # Check if clicked
        length, info, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
        print(length)
        if length < 60:
            cursor = lmList[8][:2]
            for imgObject in listImg:
                imgObject.update(cursor)

    try:

        for imgObject in listImg:

            # Draw for JPG image
            h, w = imgObject.size
            ox, oy = imgObject.posOrigin
            if imgObject.imgType == "png":
                # Draw for PNG Images
                img = cvzone.overlayPNG(img, imgObject.img, [ox, oy])
            else:
                img[oy:oy + h, ox:ox + w] = imgObject.img

    except:
        pass

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('c'):
        detector.clear()
    elif key == ord('s'):
        cv2.imwrite('Resources/Images/test.png', img)
    elif key == ord('q'):
        break
