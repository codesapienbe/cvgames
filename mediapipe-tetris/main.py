from random import random, randint, choice
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

# Tetris pieces and their rotations
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # L
    [[1, 1, 1], [0, 0, 1]],  # J
    [[1, 1, 0], [0, 1, 1]],  # S
    [[0, 1, 1], [1, 1, 0]]   # Z
]

COLORS = [
    (0, 255, 255),   # Cyan
    (255, 255, 0),   # Yellow
    (128, 0, 128),   # Purple
    (255, 165, 0),   # Orange
    (0, 0, 255),     # Blue
    (0, 255, 0),     # Green
    (255, 0, 0)      # Red
]

class TetrisPiece:
    def __init__(self):
        self.shape_idx = randint(0, len(SHAPES) - 1)
        self.shape = SHAPES[self.shape_idx]
        self.color = COLORS[self.shape_idx]
        self.x = 3
        self.y = 0

    def rotate(self):
        # Rotate the piece 90 degrees clockwise
        rows = len(self.shape)
        cols = len(self.shape[0])
        rotated = [[0 for _ in range(rows)] for _ in range(cols)]
        for r in range(rows):
            for c in range(cols):
                rotated[c][rows-1-r] = self.shape[r][c]
        self.shape = rotated

class TetrisGame:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = [[0 for _ in range(width)] for _ in range(height)]
        self.current_piece = None
        self.game_over = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0

    def new_piece(self):
        self.current_piece = TetrisPiece()
        if self.check_collision():
            self.game_over = True

    def check_collision(self, offset_x=0, offset_y=0):
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = self.current_piece.x + x + offset_x
                    new_y = self.current_piece.y + y + offset_y
                    if (new_x < 0 or new_x >= self.width or 
                        new_y >= self.height or 
                        (new_y >= 0 and self.board[new_y][new_x])):
                        return True
        return False

    def lock_piece(self):
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    self.board[self.current_piece.y + y][self.current_piece.x + x] = self.current_piece.color

    def clear_lines(self):
        lines_to_clear = []
        for y in range(self.height):
            if all(self.board[y]):
                lines_to_clear.append(y)
        
        for y in lines_to_clear:
            del self.board[y]
            self.board.insert(0, [0 for _ in range(self.width)])
        
        cleared = len(lines_to_clear)
        if cleared:
            self.lines_cleared += cleared
            self.score += [100, 300, 500, 800][cleared - 1] * self.level
            self.level = self.lines_cleared // 10 + 1
            play_sound(beep_sound)

    def move(self, dx, dy):
        if not self.check_collision(offset_x=dx, offset_y=dy):
            self.current_piece.x += dx
            self.current_piece.y += dy
            return True
        return False

    def rotate(self):
        self.current_piece.rotate()
        if self.check_collision():
            for _ in range(3):  # Rotate back
                self.current_piece.rotate()

def play_sound(sound):
    """Play a sound effect using pygame"""
    if sound:
        try:
            sound.play()
        except:
            pass

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tetris with MediaPipe Hand Tracking')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    args = parser.parse_args()

    # Set HD resolution
    GAME_WIDTH = 1280
    GAME_HEIGHT = 720

    # Initialize game
    game = TetrisGame()
    game.new_piece()

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

    # Calculate board size and position
    board_width = int(GAME_WIDTH * 0.4)  # 40% of screen width
    board_height = int(GAME_HEIGHT * 0.8)  # 80% of screen height
    cell_size = min(board_width // game.width, board_height // game.height)
    
    # Adjust board size to fit cells perfectly
    board_width = cell_size * game.width
    board_height = cell_size * game.height
    
    # Center the board
    start_x = int((GAME_WIDTH - board_width) // 2)
    start_y = int((GAME_HEIGHT - board_height) // 2)

    last_move_time = time.time()
    move_delay = 0.5  # Initial delay between moves
    last_rotation_time = time.time()
    rotation_delay = 0.3  # Delay between rotations

    while cap.isOpened():
        success, img = cap.read()

        if img is None or img.size == 0:
            print("Error: Could not read frame from camera")
            break

        img = cv2.flip(img, 1)

        if not game.game_over:
            # Create game board with background
            if background_img is not None:
                game_board = cv2.resize(background_img, (GAME_WIDTH, GAME_HEIGHT))
            else:
                game_board = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
            
            # Draw game board background
            cv2.rectangle(game_board, 
                         (start_x, start_y), 
                         (start_x + board_width, start_y + board_height), 
                         (255, 255, 255), 3)

            # Draw the board
            for y in range(game.height):
                for x in range(game.width):
                    if game.board[y][x]:
                        cv2.rectangle(game_board,
                                    (start_x + x * cell_size, start_y + y * cell_size),
                                    (start_x + (x + 1) * cell_size, start_y + (y + 1) * cell_size),
                                    game.board[y][x], -1)

            # Draw current piece
            if game.current_piece:
                for y, row in enumerate(game.current_piece.shape):
                    for x, cell in enumerate(row):
                        if cell:
                            cv2.rectangle(game_board,
                                        (start_x + (game.current_piece.x + x) * cell_size,
                                         start_y + (game.current_piece.y + y) * cell_size),
                                        (start_x + (game.current_piece.x + x + 1) * cell_size,
                                         start_y + (game.current_piece.y + y + 1) * cell_size),
                                        game.current_piece.color, -1)

            # detection hands with improved confidence and drawing
            hands, img = detector.findHands(img, flipType=False, draw=True)
            
            current_time = time.time()

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
                        # Convert hand position to game coordinates
                        game_x = int((x - start_x) / cell_size)
                        game_y = int((y - start_y) / cell_size)

                        # Move piece based on hand position
                        if current_time - last_move_time > move_delay:
                            if game_x < game.current_piece.x:
                                game.move(-1, 0)
                            elif game_x > game.current_piece.x + len(game.current_piece.shape[0]):
                                game.move(1, 0)
                            last_move_time = current_time

                        # Rotate piece based on hand movement
                        if current_time - last_rotation_time > rotation_delay:
                            if y < start_y + board_height / 2:
                                game.rotate()
                            last_rotation_time = current_time

            # Auto move down
            if current_time - last_move_time > move_delay:
                if not game.move(0, 1):
                    game.lock_piece()
                    game.clear_lines()
                    game.new_piece()
                    if game.game_over:
                        play_sound(lose_sound)
                last_move_time = current_time

            # Game HUD
            cvzone.putTextRect(game_board, f'Score: {game.score}',
                               (GAME_WIDTH - 300, 50), scale=1.4, offset=10)
            cvzone.putTextRect(game_board, f'Level: {game.level}',
                               (GAME_WIDTH - 150, 50), scale=1.4, offset=10)
            cvzone.putTextRect(game_board, f'Lines: {game.lines_cleared}',
                               (GAME_WIDTH - 450, 50), scale=1.4, offset=10)

            # Add small camera view in bottom left corner
            camera_view = img.copy()
            camera_view = cv2.resize(camera_view, (160, 120))
            cv2.rectangle(game_board, (10, GAME_HEIGHT - 130),
                         (170, GAME_HEIGHT - 10), (255, 255, 255), 2)
            game_board[GAME_HEIGHT - 130:GAME_HEIGHT - 10,
                      10:170] = camera_view

            cv2.imshow("Tetris", game_board)

        else:
            # Game over screen
            cvzone.putTextRect(game_board, 'Game Over!', (400, 400), scale=5, offset=30, thickness=7)
            cvzone.putTextRect(game_board, f'Final Score: {game.score}', (450, 500), scale=3, offset=20)
            cvzone.putTextRect(game_board, 'Press R to restart', (460, 575), scale=2, offset=10)
            cv2.imshow("Tetris", game_board)

        key = cv2.waitKey(1)
        if key == ord('r'):  # restart game
            game = TetrisGame()
            game.new_piece()
        elif key == ord('q'):  # quit
            cap.release()
            cv2.destroyAllWindows()
            exit(-1)

if __name__ == "__main__":
    main() 