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
from collections import deque

# Import the back button system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cvstore'))
from back_button import BackButton

# Initialize pygame mixer
pygame.mixer.init()

# Load sound effects
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    beep_sound = pygame.mixer.Sound(os.path.join(script_dir, "beep.mp3"))
    win_sound = pygame.mixer.Sound(os.path.join(script_dir, "win.mp3"))
    lose_sound = pygame.mixer.Sound(os.path.join(script_dir, "lose.mp3"))
except Exception as e:
    print(f"Warning: Sound files not found: {e}. Game will run without sound.")
    beep_sound = None
    win_sound = None
    lose_sound = None

# Load background image
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    background_path = os.path.join(script_dir, "Resources", "Background.png")
    background_img = cv2.imread(background_path)
    if background_img is None:
        print(f"Warning: Background image not found at {background_path}. Using black background.")
        background_img = None
except Exception as e:
    print(f"Warning: Could not load background image: {e}. Using black background.")
    background_img = None

# --- Add imports for OpenTelemetry and Pygame display ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# --- Add OpenTelemetry setup after imports ---
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

class SimpleGestureController:
    def __init__(self, hold_threshold=65, move_delay=0.3, rotation_cooldown=0.8):
        self.hold_threshold = hold_threshold
        self.move_delay = move_delay
        self.rotation_cooldown = rotation_cooldown
        self.last_move_time = 0
        self.last_rotation_time = 0
        self.right_hand_was_together = False
        self.left_hand_was_together = False
        
    def can_move(self, current_time):
        """Check if enough time has passed for next move"""
        return current_time - self.last_move_time > self.move_delay
    
    def can_rotate(self, current_time):
        """Check if enough time has passed for next rotation"""
        return current_time - self.last_rotation_time > self.rotation_cooldown
    
    def update_move_time(self, current_time):
        """Update the last move time"""
        self.last_move_time = current_time
        
    def update_rotation_time(self, current_time):
        """Update the last rotation time"""
        self.last_rotation_time = current_time
        
    def should_move(self, right_hand_together, current_time):
        """Check if right hand should trigger movement (only on finger press)"""
        if right_hand_together and not self.right_hand_was_together and self.can_move(current_time):
            self.right_hand_was_together = True
            return True
        elif not right_hand_together:
            self.right_hand_was_together = False
        return False
        
    def should_rotate(self, left_hand_together, current_time):
        """Check if left hand should trigger rotation (only on finger press)"""
        if left_hand_together and not self.left_hand_was_together and self.can_rotate(current_time):
            self.left_hand_was_together = True
            return True
        elif not left_hand_together:
            self.left_hand_was_together = False
        return False

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
    (100, 200, 200),   # Soft cyan
    (200, 200, 100),   # Soft yellow
    (150, 100, 150),   # Soft purple
    (200, 150, 100),   # Soft orange
    (100, 100, 200),   # Soft blue
    (100, 200, 100),   # Soft green
    (200, 100, 100)    # Soft red
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

    # Initialize rotation gesture detector
    finger_controller = SimpleGestureController()

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
    detector = HandDetector(detectionCon=0.75, maxHands=2)
    
    # Initialize back button
    back_button = BackButton(GAME_WIDTH, GAME_HEIGHT)

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
    key = None  # Initialize key variable

    # Replace cv2.namedWindow/cv2.imshow/cv2.waitKey with Pygame display
    pygame.init()
    WIDTH, HEIGHT = GAME_WIDTH, GAME_HEIGHT
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tetris (CV+Pygame)")
    clock = pygame.time.Clock()
    with tracer.start_as_current_span("tetris_session"):
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
                            game = TetrisGame()
                            game.new_piece()
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