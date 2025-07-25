import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
import sys
import os
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

screen_width = 1280
screen_height = 720

# Import the back button system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cvstore'))
from back_button import BackButton

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize back button
back_button = BackButton(screen_width, screen_height)

detector = HandDetector(detectionCon=0.75, maxHands=1)

# Load and resize background image
script_dir = os.path.dirname(os.path.abspath(__file__))
background_path = os.path.join(script_dir, "Resources", "Background.png")
backgroundImg = cv2.imread(background_path)
if backgroundImg is None:
    print(f"Error: Could not load background image from {background_path}")
    backgroundImg = np.zeros((720, 1280, 3), dtype=np.uint8)
    backgroundImg[:] = (50, 50, 50)
else:
    backgroundImg = cv2.resize(backgroundImg, (1280, 720))

class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = 0, 0
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        if self.imgFood is None:
            print(f"Error: Could not load food image from {pathFood}")
            self.imgFood = np.zeros((50, 50, 4), dtype=np.uint8)
            cv2.circle(self.imgFood, (25, 25), 20, (0, 255, 0, 255), -1)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()
        self.score = 0
        self.gameOver = False
        self.startTime = time.time()
        self.gameDuration = 120
        self.snakeColors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
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
    def update(self, imgMain, currentHead, tracer):
        remainingTime = self.getRemainingTime()
        if self.gameOver or remainingTime <= 0:
            if remainingTime <= 0:
                with tracer.start_as_current_span("game_over_time"):
                    cvzone.putTextRect(imgMain, "Time's Up!", [300, 400], scale=7, thickness=5, offset=20)
            else:
                with tracer.start_as_current_span("game_over"):
                    cvzone.putTextRect(imgMain, "Game Over", [300, 400], scale=7, thickness=5, offset=20)
            cvzone.putTextRect(imgMain, f'Your Score: {self.score}', [300, 550], scale=7, thickness=5, offset=20)
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
                with tracer.start_as_current_span("eat_food"):
                    self.randomFoodLocation()
                    self.allowedLength += 50
                    self.score += 1
            # Draw Snake
            if self.points:
                snakeColor = self.getSnakeColor()
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i - 1], self.points[i], snakeColor, 20)
                cv2.circle(imgMain, self.points[-1], 20, snakeColor, cv2.FILLED)
            # Draw Food
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))
            # Draw score and timer in the top right corner with pink color
            cvzone.putTextRect(imgMain, f'Score: {self.score}', [imgMain.shape[1] - 200, 80], scale=1.5, thickness=2, offset=10, colorR=(255, 192, 203))
            cvzone.putTextRect(imgMain, f'Time: {remainingTime}s', [imgMain.shape[1] - 200, 120], scale=1.5, thickness=2, offset=10, colorR=(255, 192, 203))
            # Check for Collision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            if len(pts) > 2:
                minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
                if minDist >= 0 and minDist < 20:
                    with tracer.start_as_current_span("crash"):
                        self.gameOver = True
        return imgMain

def main():
    food_path = os.path.join(script_dir, "Resources", "Donut.png")
    game = SnakeGameClass(food_path)
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Snake Game (CV+Pygame)")
    clock = pygame.time.Clock()
    running = True
    with tracer.start_as_current_span("snake_session"):
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            game.__init__(food_path)
            success, img = cap.read()
            # img = cv2.flip(img, 1)
            hands, img = detector.findHands(img, flipType=False)
            hand_position = None
            hand_landmarks = None
            if hands:
                hand_position = hands[0]["lmList"][9][:2]
                hand_landmarks = type('HandLandmarks', (), {
                    'landmark': [type('Landmark', (), {
                        'x': lm[0] / screen_width,
                        'y': lm[1] / screen_height
                    })() for lm in hands[0]["lmList"]]
                })()
            # Check if user wants to exit
            if back_button.handle_input(None, hand_landmarks, hand_position):
                with tracer.start_as_current_span("quit_to_appstore"):
                    print("User approved exit - returning to app store")
                running = False
                break
            # Create a copy of the background image
            gameImg = backgroundImg.copy()
            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]
                gameImg = game.update(gameImg, pointIndex, tracer)
            # Resize the webcam feed (2x smaller)
            webcamSize = (160, 120)
            webcamImg = cv2.resize(img, webcamSize)
            y_offset = gameImg.shape[0] - webcamSize[1]
            x_offset = 0
            borderThickness = 3
            cv2.rectangle(gameImg, (x_offset - borderThickness, y_offset - borderThickness),
                         (x_offset + webcamSize[0] + borderThickness, y_offset + webcamSize[1] + borderThickness),
                         (255, 192, 203), borderThickness)
            gameImg[y_offset:y_offset+webcamSize[1], x_offset:x_offset+webcamSize[0]] = webcamImg
            # Draw back button
            back_button.draw(gameImg, hand_position)
            # --- Pygame Rendering ---
            frame_rgb = cv2.cvtColor(gameImg, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
