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

# Import the back button system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cvstore'))
from back_button import BackButton

# --- Add imports for OpenTelemetry and Pygame display ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

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


# --- Add OpenTelemetry setup after imports ---
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

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

    # Initialize back button
    back_button = BackButton(screen_width, screen_height)
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

    # Replace cv2.namedWindow/cv2.imshow/cv2.waitKey with Pygame display
    pygame.init()
    WIDTH, HEIGHT = GAME_WIDTH, GAME_HEIGHT
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TicTacToe (CV+Pygame)")
    clock = pygame.time.Clock()
    with tracer.start_as_current_span("tictactoe_session"):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            game = Game()
                            timeStart = time.time()
                            score = 0
                            for button in button_components:
                                button.click(game_board, " ")
            # ... existing game logic ...
            # At the end of each frame, render with Pygame:
            frame_rgb = cv2.cvtColor(game_board, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
