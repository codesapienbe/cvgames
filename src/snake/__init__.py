import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import argparse
import time

# Set up argument parser
parser = argparse.ArgumentParser(description='Snake Game with Hand Tracking')
parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
args = parser.parse_args()

cap = cv2.VideoCapture(args.camera)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.75, maxHands=1)

# Load and resize background image
backgroundImg = cv2.imread("Resources/Background.png")
backgroundImg = cv2.resize(backgroundImg, (1280, 720))

class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = 0, 0  # previous head point

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False
        self.startTime = time.time()
        self.gameDuration = 120  # 2 minutes in seconds
        
        # Define snake colors for different growth stages
        self.snakeColors = [
            (0, 0, 255),    # Red (0-9 points)
            (0, 255, 0),    # Green (10-19 points)
            (255, 0, 0),    # Blue (20-29 points)
            (255, 255, 0),  # Yellow (30-39 points)
            (255, 0, 255),  # Magenta (40-49 points)
            (0, 255, 255),  # Cyan (50+ points)
        ]

    def getSnakeColor(self):
        colorIndex = min(self.score // 10, len(self.snakeColors) - 1)
        return self.snakeColors[colorIndex]

    def getRemainingTime(self):
        elapsedTime = time.time() - self.startTime
        remainingTime = max(0, self.gameDuration - elapsedTime)
        return int(remainingTime)

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead):
        remainingTime = self.getRemainingTime()
        
        if self.gameOver or remainingTime <= 0:
            if remainingTime <= 0:
                cvzone.putTextRect(imgMain, "Time's Up!", [300, 400],
                                   scale=7, thickness=5, offset=20)
            else:
                cvzone.putTextRect(imgMain, "Game Over", [300, 400],
                                   scale=7, thickness=5, offset=20)
            cvzone.putTextRect(imgMain, f'Your Score: {self.score}', [300, 550],
                               scale=7, thickness=5, offset=20)
        else:
            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            # Length Reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.allowedLength:
                        break

            # Check if snake ate the Food
            rx, ry = self.foodPoint
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
                    ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                self.randomFoodLocation()
                self.allowedLength += 50
                self.score += 1
                print(self.score)

            # Draw Snake
            if self.points:
                snakeColor = self.getSnakeColor()
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i - 1], self.points[i], snakeColor, 20)
                cv2.circle(imgMain, self.points[-1], 20, snakeColor, cv2.FILLED)

            # Draw Food
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood,
                                        (rx - self.wFood // 2, ry - self.hFood // 2))

            # Draw score and timer in the top right corner with pink color
            rightPadding = 20  # Padding from the right edge
            cvzone.putTextRect(imgMain, f'Score: {self.score}', 
                               [imgMain.shape[1] - 200, 80],
                               scale=1.5, thickness=2, offset=10, colorR=(255, 192, 203))  # Pink color
            cvzone.putTextRect(imgMain, f'Time: {remainingTime}s', 
                               [imgMain.shape[1] - 200, 120],
                               scale=1.5, thickness=2, offset=10, colorR=(255, 192, 203))  # Pink color

            # Check for Collision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (0, 255, 0), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

        return imgMain


game = SnakeGameClass("Resources/Donut.png")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    # Create a copy of the background image
    gameImg = backgroundImg.copy()

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        gameImg = game.update(gameImg, pointIndex)
    
    # Resize the webcam feed (2x smaller)
    webcamSize = (160, 120)  # Half the previous size
    webcamImg = cv2.resize(img, webcamSize)
    
    # Place the webcam feed in the bottom left corner
    y_offset = gameImg.shape[0] - webcamSize[1]
    x_offset = 0
    
    # Draw pink border around webcam frame
    borderThickness = 3
    cv2.rectangle(gameImg, 
                 (x_offset - borderThickness, y_offset - borderThickness),
                 (x_offset + webcamSize[0] + borderThickness, y_offset + webcamSize[1] + borderThickness),
                 (255, 192, 203),  # Pink color
                 borderThickness)
    
    # Place the webcam feed
    gameImg[y_offset:y_offset+webcamSize[1], x_offset:x_offset+webcamSize[0]] = webcamImg

    cv2.imshow("Snake Game", gameImg)
    key = cv2.waitKey(1)

    if key == ord('r'):
        game.gameOver = False
        game.startTime = time.time()  # Reset the timer when restarting
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
