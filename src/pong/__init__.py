import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import argparse
import sys
import os

# Import the back button system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cvstore'))
from back_button import BackButton

screen_width = 1280
screen_height = 720

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Pong Game with Hand Tracking')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    args = parser.parse_args()

    # Initialize camera with selected index
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {args.camera}")
        exit(-1)

    cap.set(3, 1280)

    # Initialize back button
    back_button = BackButton(screen_width, screen_height)
    cap.set(4, 720)

    # Importing all images with correct path
    resources_path = os.path.join(os.path.dirname(__file__), "Resources")
    imgBackground = cv2.imread(os.path.join(resources_path, "Background.png"))
    if imgBackground is None:
        print("Error: Could not load Background.png")
        exit(-1)

    imgGameOver = cv2.imread(os.path.join(resources_path, "gameOver.png"))
    if imgGameOver is None:
        print("Error: Could not load gameOver.png")
        exit(-1)

    imgBall = cv2.imread(os.path.join(resources_path, "Ball.png"), cv2.IMREAD_UNCHANGED)
    if imgBall is None:
        print("Error: Could not load Ball.png")
        exit(-1)

    imgBat1 = cv2.imread(os.path.join(resources_path, "bat1.png"), cv2.IMREAD_UNCHANGED)
    if imgBat1 is None:
        print("Error: Could not load bat1.png")
        exit(-1)

    imgBat2 = cv2.imread(os.path.join(resources_path, "bat2.png"), cv2.IMREAD_UNCHANGED)
    if imgBat2 is None:
        print("Error: Could not load bat2.png")
        exit(-1)

    # Hand Detector
    detector = HandDetector(detectionCon=1, maxHands=2)

    # Variables
    ballPos = [100, 100]
    speedX = 15
    speedY = 15
    gameOver = False
    score = [0, 0]

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        imgRaw = img.copy()

        # Find the hand and its landmarks
        hands, img = detector.findHands(img, flipType=False)  # with draw

        # Handle back button input
        hand_position = None
        hand_landmarks = None
        if hands:
            hand_position = hands[0]["lmList"][9][:2]  # Palm center
            # Convert to MediaPipe landmarks format for back button
            hand_landmarks = type('HandLandmarks', (), {
                'landmark': [type('Landmark', (), {
                    'x': lm[0] / screen_width,
                    'y': lm[1] / screen_height
                })() for lm in hands[0]["lmList"]]
            })()

        # Check if user wants to exit (key will be defined later in the loop)
        if hand_landmarks and back_button.handle_input(None, hand_landmarks, hand_position):
            print("User approved exit - returning to app store")
            cap.release()
            cv2.destroyAllWindows()
            return

        # Ensure both images have the same dimensions
        img = cv2.resize(img, (1280, 720))
        imgBackground = cv2.resize(imgBackground, (1280, 720))

        # Overlaying the background image
        img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

        # Check for hands
        if hands:
            for hand in hands:
                x, y, w, h = hand['bbox']
                h1, w1, _ = imgBat1.shape
                y1 = y - h1 // 2
                y1 = np.clip(y1, 20, 415)

                if hand['type'] == "Left":
                    img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                    if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                        speedX = -speedX
                        ballPos[0] += 30
                        score[0] += 1

                if hand['type'] == "Right":
                    img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                    if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                        speedX = -speedX
                        ballPos[0] -= 30
                        score[1] += 1

        # Game Over
        if ballPos[0] < 40 or ballPos[0] > 1200:
            gameOver = True

        if gameOver:
            img = imgGameOver
            cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX,
                        2.5, (200, 0, 200), 5)

        # If game not over move the ball
        else:
            # Move the Ball
            if ballPos[1] >= 500 or ballPos[1] <= 10:
                speedY = -speedY

            ballPos[0] += speedX
            ballPos[1] += speedY

            # Draw the ball
            img = cvzone.overlayPNG(img, imgBall, ballPos)

            cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
            cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

        img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))


        # Draw back button
        back_button.draw(img, hand_position)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('r'):
            ballPos = [100, 100]
            speedX = 15
            speedY = 15
            gameOver = False
            score = [0, 0]
            imgGameOver = cv2.imread(os.path.join(resources_path, "gameOver.png"))
        if key == ord('f'):
            speedX = 35
            speedY = 35
        if key == ord('s'):
            speedX = 15
            speedY = 15
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
