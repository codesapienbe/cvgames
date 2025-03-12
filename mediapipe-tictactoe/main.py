from random import random, randint

import cv2
import mediapipe as mp
import math
from screeninfo import get_monitors
from playsound import playsound
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import cvzone
import time


class Button:

    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def click(self, img, value):
        self.border((255, 255, 255), img)
        self.text(value, img)

    def focused(self, x, y):
        return self.pos[0] < x < self.pos[0] + self.width and self.pos[1] < y < self.pos[1] + self.height

    def draw(self, img):
        self.border((255, 255, 255), img)
        self.text(self.value, img)

    def background(self, rgb, img):
        # for the background calculator
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), rgb, cv2.FILLED)

    def border(self, rgb, img):
        # for the border calculator
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), rgb, 3)

    def text(self, value, img):
        self.value = value
        cv2.putText(img, self.value, (self.pos[0] + 80, self.pos[1] + 150), cv2.FONT_HERSHEY_PLAIN, 6, (255, 255, 255),
                    3)


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


def main():
    primary_monitor = {}
    for m in get_monitors():
        print("Connected monitors {}".format(m))
        if m.is_primary:
            primary_monitor = m
            break

    cap = cv2.VideoCapture(0)
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

    # MULTIPLAYER

    # creating Button
    button_values = [[" ", " ", " "],
                     [" ", " ", " "],
                     [" ", " ", " "]]
    button_components = []

    for x in range(len(button_values)):
        for y in range(len(button_values[x])):
            pos_x = x * 200 + 650  # starting from 650 pixel in the width
            pos_y = y * 200 + 100  # starting from 100 pixel in the height
            button_components.append(Button((pos_x, pos_y), 200, 200, button_values[x][y]))
    # to avoid duplicated value inside calculator in event writing
    delay_counter = 0

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1)

        if time.time() - timeStart < totalTime:

            # detection hands
            hands, img = detector.findHands(img, flipType=False)

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
                    # i did not add value into display calculator
                    # after passing 10 frames
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
