from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np
import argparse
from screeninfo import get_monitors

# Parse command line arguments
parser = argparse.ArgumentParser(description='Hand gesture-based catalog viewer')
parser.add_argument('--camera', type=int, default=0,
                    help='Camera index to use (default: 0)')
parser.add_argument('--width', type=int, default=1920,
                    help='Width of the window (default: 1920)')
parser.add_argument('--height', type=int, default=1080,
                    help='Height of the window (default: 1080)')
parser.add_argument("--resources", type=str, default="Resources",
                    help="Path to the resources folder (default: Resources)")
args = parser.parse_args()

# Parameters
width, height = 1920, 1080
if args.width and args.height:
    width, height = args.width, args.height
gestureThreshold = 300
if args.resources:
    path_folder = args.resources
else:
    path_folder = "Resources"

primary_monitor = {}
for m in get_monitors():
    print("Connected monitors {}".format(m))
    if m.is_primary:
        primary_monitor = m
        break

cap = cv2.VideoCapture(args.camera)
cap.set(3, primary_monitor.width)
cap.set(4, primary_monitor.height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
product_images = []
delay = 30
buttonPressed = False
counter = 0
select_mode = False
product_image_index = 0
button_delay_count = 0
basket_items = [[]]
product_index = -1
product_focused = False
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image

# Get list of presentation images
path_images = sorted(os.listdir(path_folder), key=len)

print(f"{len(path_images)} products imported..")

while cap.isOpened():
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    absolute_path_image = os.path.join(path_folder, path_images[product_image_index])
    imgCurrent = cv2.imread(absolute_path_image)

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:  # If hand is detected

        hand1 = hands[0]
        cx, cy = hand1["center"]
        lmList = hand1["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand1)  # List of which fingers are up

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        indexFinger = xVal, yVal

        if fingers == [1, 0, 0, 0, 0]:
            buttonPressed = True
            if product_image_index > 0:
                product_image_index -= 1
                basket_items = [[]]
                product_index = -1
                product_focused = False
        if fingers == [0, 0, 0, 0, 1]:
            buttonPressed = True
            if product_image_index < len(path_images) - 1:
                product_image_index += 1
                basket_items = [[]]
                product_index = -1
                product_focused = False

        if fingers == [0, 1, 0, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        if fingers == [0, 1, 1, 0, 0]:
            if product_focused is False:
                product_focused = True
                product_index += 1
                basket_items.append([])
            basket_items[product_index].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        else:
            product_focused = False

        if fingers == [0, 1, 1, 1, 0]:
            if basket_items:
                basket_items.pop(-1)
                product_index -= 1
                buttonPressed = True

    else:
        product_focused = False

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    for i, item in enumerate(basket_items):
        for j in range(len(item)):
            if j != 0:
                cv2.line(imgCurrent, item[j - 1], item[j], (0, 0, 200), 12)

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Slides", imgCurrent)
    # cv2.imshow("Image", img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
