import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import os

from screeninfo import get_monitors

primary_monitor = {}
for m in get_monitors():
    print("Connected monitors {}".format(m))
    if m.is_primary:
        primary_monitor = m
        break

cap = cv2.VideoCapture(0)
cap.set(3, primary_monitor.width)
cap.set(4, primary_monitor.height)

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

    def zoomIn(self, cursor):
        ox, oy = self.posOrigin
        h, w = self.size

        # Check if in region
        if ox < cursor[0] < ox + w and oy < cursor[1] < oy + h:
            self.size = h * 2, w * 2
        else:
            self.size = h * 2, w * 2


def resize(img, scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dsize = (width, height)
    return cv2.resize(img, dsize)


def text(value, pos, img):
    cv2.putText(img, value, (pos[0] + 25, pos[1] + 40), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 4)


path = "Resources"
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

    if not success:
        continue

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:

        landmarks = hands[0]['lmList']
        fingers = detector.fingersUp(hands[0])  # List of which fingers are up
        pointIndex = landmarks[8][0:2]
        middleIndex = landmarks[12][0:2]
        distance, _, img = detector.findDistance(pointIndex, middleIndex, img)

        if len(hands) == 1 and distance < 70:
            for imgObject in listImg:
                imgObject.update(pointIndex)
        elif len(hands) == 2:
            for imgIndex, imgObject in enumerate(listImg):
                listImg[imgIndex] = resize(img, 200)
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
    if (key == ord("c")):  # to clear the display calculator
        basketInfo = ""
    if key == ord('q'):  # to stop the program
        cv2.destroyAllWindows()
        cap.release()
        exit(-1)

cv2.destroyAllWindows()
cap.release()
