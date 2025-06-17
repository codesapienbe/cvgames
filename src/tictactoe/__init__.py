from random import random, randint

import cv2
import mediapipe as mp
import math
from screeninfo import get_monitors
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import cvzone
import time
import pygame
from deepface import DeepFace
import os
import sys
import argparse

# Initialize pygame mixer
pygame.mixer.init()

# Load sound effects
try:
    beep_sound = pygame.mixer.Sound("beep.mp3")
    win_sound = pygame.mixer.Sound("win.mp3")
    lose_sound = pygame.mixer.Sound("lose.mp3")
except:
    print("Warning: Sound files not found. Game will run without sound.")
    beep_sound = None
    win_sound = None
    lose_sound = None

# Load background image
try:
    background_img = cv2.imread("Resources/Background.png")
    if background_img is None:
        print("Warning: Background image not found. Using black background.")
        background_img = None
except:
    print("Warning: Could not load background image. Using black background.")
    background_img = None

class Button:

    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value
        # Define colors for X and O
        self.colors = {
            'X': (0, 255, 255),  # Cyan for X
            'O': (255, 165, 0),  # Orange for O
            ' ': (255, 255, 255) # White for empty
        }

    def click(self, img, value):
        color = self.colors.get(value, (255, 255, 255))
        self.border(color, img)
        self.text(value, img)

    def focused(self, x, y):
        return self.pos[0] < x < self.pos[0] + self.width and self.pos[1] < y < self.pos[1] + self.height

    def draw(self, img):
        color = self.colors.get(self.value, (255, 255, 255))
        self.border(color, img)
        self.text(self.value, img)

    def background(self, rgb, img):
        # for the background calculator
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), rgb, cv2.FILLED)

    def border(self, rgb, img):
        # for the border calculator
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), rgb, 3)

    def text(self, value, img):
        self.value = value
        color = self.colors.get(value, (255, 255, 255))
        # Adjust text position to be centered in the cell
        text_x = self.pos[0] + (self.width // 2) - 40
        text_y = self.pos[1] + (self.height // 2) + 40
        cv2.putText(img, self.value, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 6, color, 3)


class Game:

    def __init__(self):
        # All probable winning combinations
        self.solutions = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9], [1, 5, 9], [3, 5, 7]]
        self.default_values = [' ' for i in range(9)]
        # Storing the positions occupied by X and O
        self.player_selections = {'X': [], 'O': []}
        self.game_over = False
        self.winner = None

    def checkVictory(self, playerpos, curplayer):
        # Loop to check whether any winning combination is satisfied or not
        for i in self.solutions:
            if all(j in playerpos[curplayer] for j in i):
                # Return True if any winning combination is satisfied
                return True
        return False

    def checkDraw(self):
        return len(self.player_selections['X']) + len(self.player_selections['O']) == 9


def play_sound(sound):
    """Play a sound effect using pygame"""
    if sound:
        try:
            sound.play()
        except:
            pass  # Ignore any sound playback errors


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tic Tac Toe with MediaPipe Hand Tracking')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    args = parser.parse_args()

    # Set HD resolution
    GAME_WIDTH = 1280
    GAME_HEIGHT = 720

    # Initialize game
    game = Game()

    # List available cameras
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        print("Error: No cameras found!")
        return
    
    print(f"Available cameras: {available_cameras}")
    
    if args.camera not in available_cameras:
        print(f"Error: Camera {args.camera} is not available!")
        print(f"Please select one of the available cameras: {available_cameras}")
        return

    primary_monitor = {}
    for m in get_monitors():
        print("Connected monitors {}".format(m))
        if m.is_primary:
            primary_monitor = m
            break

    print(f"Attempting to open camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        print("Please check if:")
        print("1. The camera is properly connected")
        print("2. The camera is not being used by another application")
        print("3. You have the correct camera index")
        return

    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Successfully opened camera {args.camera}")
    print(f"Camera properties: {width}x{height} @ {fps}fps")

    cap.set(3, GAME_WIDTH)
    cap.set(4, GAME_HEIGHT)
    detector = HandDetector(detectionCon=0.75, maxHands=1)

    color = (255, 0, 255)
    counter = 0
    score = 0
    timeStart = time.time()
    totalTime = 120

    random_player_index = randint(0, 2) % 2
    next_player = " "

    if random_player_index == 0:
        next_player = "O"
    else:
        next_player = "X"

    # Calculate board size based on HD resolution
    board_size = int(min(GAME_WIDTH, GAME_HEIGHT) * 0.8)  # 80% of the smaller screen dimension
    cell_size = board_size // 3
    
    # Calculate starting position to center the board
    start_x = int((GAME_WIDTH - board_size) // 2)
    start_y = int((GAME_HEIGHT - board_size) // 2)

    # creating Button
    button_values = [[" ", " ", " "],
                     [" ", " ", " "],
                     [" ", " ", " "]]
    button_components = []

    for x in range(len(button_values)):
        for y in range(len(button_values[x])):
            pos_x = start_x + x * cell_size
            pos_y = start_y + y * cell_size
            button_components.append(Button((pos_x, pos_y), cell_size, cell_size, button_values[x][y]))

    # to avoid duplicated value inside calculator in event writing
    delay_counter = 0

    while cap.isOpened():
        success, img = cap.read()

        if img is None or img.size == 0:
            print("Error: Could not read frame from camera")
            break

        img = cv2.flip(img, 1)

        if time.time() - timeStart < totalTime and not game.game_over:
            # Create game board with background
            if background_img is not None:
                # Resize background image to match game dimensions
                game_board = cv2.resize(background_img, (GAME_WIDTH, GAME_HEIGHT))
            else:
                # Fallback to black background if image not available
                game_board = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
            
            # detection hands with improved confidence and drawing
            hands, img = detector.findHands(img, flipType=False, draw=True)
            
            # Draw game board background with semi-transparent overlay
            overlay = game_board.copy()
            cv2.rectangle(overlay, 
                         (int(start_x), int(start_y)), 
                         (int(start_x + board_size), int(start_y + board_size)), 
                         (255, 255, 255), 3)
            # Blend the overlay with the background
            cv2.addWeighted(overlay, 0.3, game_board, 0.7, 0, game_board)

            for button in button_components:
                button.draw(game_board)

            if hands:
                if len(hands) == 1:
                    landmarks = hands[0]["lmList"]
                    distance, _, img = detector.findDistance(landmarks[8][:2], landmarks[12][:2], img)
                    x, y = landmarks[8][:2]

                    # Draw hand tracking on game board
                    for lm in landmarks:
                        cv2.circle(game_board, (int(lm[0]), int(lm[1])), 3, (0, 255, 0), cv2.FILLED)
                    
                    # Draw distance line
                    cv2.line(game_board, 
                            (int(landmarks[8][0]), int(landmarks[8][1])),
                            (int(landmarks[12][0]), int(landmarks[12][1])),
                            (255, 255, 255), 2)

                    if distance < 65:
                        for button in button_components:
                            if button.focused(x, y) and delay_counter == 0:
                                if button.value == " " and next_player == "O":
                                    button.click(game_board, "X")
                                    game.player_selections['X'].append(button_components.index(button) + 1)
                                    if game.checkVictory(game.player_selections, 'X'):
                                        game.game_over = True
                                        game.winner = 'X'
                                        play_sound(win_sound)
                                    next_player = "X"
                                elif button.value == " " and next_player == "X":
                                    button.click(game_board, "O")
                                    game.player_selections['O'].append(button_components.index(button) + 1)
                                    if game.checkVictory(game.player_selections, 'O'):
                                        game.game_over = True
                                        game.winner = 'O'
                                        play_sound(win_sound)
                                    next_player = "O"
                                delay_counter = 1

                else:
                    cv2.putText(game_board, "Game paused.", (GAME_WIDTH // 2, GAME_HEIGHT),
                                cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 255), 10)

                # avoid duplicates
                if delay_counter != 0:
                    delay_counter += 1
                    if delay_counter > 10:
                        delay_counter = 0

            if counter:
                counter += 1
                color = (0, 255, 0)
                if counter == 3:
                    cx = randint(100, 1100)
                    cy = randint(100, 600)
                    color = (255, 0, 255)
                    score += 1
                    counter = 0

            # Game HUD - Make it more visible but smaller
            cvzone.putTextRect(game_board, f'Time: {int(totalTime - (time.time() - timeStart))}',
                               (GAME_WIDTH - 300, 50), scale=1.4, offset=10)
            cvzone.putTextRect(game_board, f'Score: {str(score).zfill(2)}', 
                               (GAME_WIDTH - 150, 50), scale=1.4, offset=10)

            # Add small camera view in bottom left corner - make it 2x smaller
            camera_view = img.copy()
            camera_view = cv2.resize(camera_view, (160, 120))  # 2x smaller than before
            # Add border around camera view
            cv2.rectangle(game_board, (10, GAME_HEIGHT - 130),
                         (170, GAME_HEIGHT - 10), (255, 255, 255), 2)
            # Place camera view in bottom left corner
            game_board[GAME_HEIGHT - 130:GAME_HEIGHT - 10,
                      10:170] = camera_view

            # Show the game board instead of the camera feed
            cv2.imshow("TicTacToe", game_board)

        else:
            if game.game_over:
                if game.winner:
                    cvzone.putTextRect(game_board, f'Player {game.winner} Wins!', (400, 400), scale=5, offset=30, thickness=7)
                else:
                    cvzone.putTextRect(game_board, 'Game Over - Draw!', (400, 400), scale=5, offset=30, thickness=7)
            else:
                cvzone.putTextRect(game_board, 'Time\'s Up!', (400, 400), scale=5, offset=30, thickness=7)
            cvzone.putTextRect(game_board, f'Your Score: {score}', (450, 500), scale=3, offset=20)
            cvzone.putTextRect(game_board, 'Press R to restart', (460, 575), scale=2, offset=10)
            cv2.imshow("TicTacToe", game_board)

        key = cv2.waitKey(1)
        if (key == ord("c")):  # to clear the display calculator
            equation = ""
        if key == ord('r'):  # to restart the game
            game = Game()
            timeStart = time.time()
            score = 0
            for button in button_components:
                button.click(game_board, " ")
        if key == ord('q'):  # to stop the program
            cap.release()
            cv2.destroyAllWindows()
            exit(-1)


if __name__ == "__main__":
    main()
