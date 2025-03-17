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

        # Defining Function to check Victory

    def checkVictory(self, playerpos, curplayer):

        # Loop to check whether any winning combination is satisfied or not
        for i in self.solutions:
            if all(j in playerpos[curplayer] for j in i):
                # Return True if any winning combination is satisfied
                return True
                # Return False if no combination is satisfied
        return False


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

    cap.set(3, primary_monitor.width)
    cap.set(4, primary_monitor.height)
    detector = HandDetector(detectionCon=0.8, maxHands=1)

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

    # Calculate board size based on screen dimensions
    board_size = int(min(primary_monitor.width, primary_monitor.height) * 0.8)  # 80% of the smaller screen dimension
    cell_size = board_size // 3
    
    # Calculate starting position to center the board
    start_x = int((primary_monitor.width - board_size) // 2)
    start_y = int((primary_monitor.height - board_size) // 2)

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

        if time.time() - timeStart < totalTime:
            # Apply strong blur to the background
            blurred = cv2.GaussianBlur(img, (99, 99), 30)  # High blur values for 90% blur effect
            img = cv2.addWeighted(img, 0.1, blurred, 0.9, 0)  # 90% blur, 10% original

            # detection hands
            hands, img = detector.findHands(img, flipType=False)

            # Draw game board background
            cv2.rectangle(img, 
                         (int(start_x), int(start_y)), 
                         (int(start_x + board_size), int(start_y + board_size)), 
                         (255, 255, 255), 3)

            for button in button_components:
                button.draw(img)

            if hands:
                if len(hands) == 1:
                    landmarks = hands[0]["lmList"]
                    distance, _, img = detector.findDistance(landmarks[8][:2], landmarks[12][:2], img)
                    x, y = landmarks[8][:2]

                    if distance < 65:
                        for button in button_components:
                            if button.focused(x, y) and delay_counter == 0:
                                if button.value == " " and next_player == "O":
                                    button.click(img, "X")
                                    next_player = "X"
                                elif button.value == " " and next_player == "X":
                                    button.click(img, "O")
                                    next_player = "O"
                                else:
                                    button.click(img, " ")
                                delay_counter = 1

                else:
                    cv2.putText(img, "Game paused.", (primary_monitor.width // 2, primary_monitor.height),
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

            # Game HUD
            cvzone.putTextRect(img, f'Time: {int(totalTime - (time.time() - timeStart))}',
                               (1000, 75), scale=3, offset=20)
            cvzone.putTextRect(img, f'Score: {str(score).zfill(2)}', (60, 75), scale=3, offset=20)

        else:
            cvzone.putTextRect(img, 'Game Over', (400, 400), scale=5, offset=30, thickness=7)
            cvzone.putTextRect(img, f'Your Score: {score}', (450, 500), scale=3, offset=20)
            cvzone.putTextRect(img, 'Press R to restart', (460, 575), scale=2, offset=10)

        cv2.imshow("TicTacToe", img)

        key = cv2.waitKey(1)
        if (key == ord("c")):  # to clear the display calculator
            equation = ""
        if key == ord('q'):  # to stop the program
            cap.release()
            cv2.destroyAllWindows()
            exit(-1)


if __name__ == "__main__":
    main()
