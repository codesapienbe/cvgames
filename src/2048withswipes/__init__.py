import cv2
import mediapipe as mp
import numpy as np
import random
import time
import pygame
import sys
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.score = 0
        self.game_over = False
        self.add_new_tile()
        self.add_new_tile()
    def add_new_tile(self):
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i][j] = 2 if random.random() < 0.9 else 4
    def move(self, direction):
        moved = False
        if direction == 'left':
            moved = self.move_left()
        elif direction == 'right':
            moved = self.move_right()
        elif direction == 'up':
            moved = self.move_up()
        elif direction == 'down':
            moved = self.move_down()
        if moved:
            self.add_new_tile()
            if self.is_game_over():
                self.game_over = True
        return moved
    def move_left(self):
        moved = False
        for i in range(self.size):
            row = self.grid[i].copy()
            new_row = self.merge_tiles(row)
            if not np.array_equal(row, new_row):
                self.grid[i] = new_row
                moved = True
        return moved
    def move_right(self):
        moved = False
        for i in range(self.size):
            row = self.grid[i][::-1].copy()
            new_row = self.merge_tiles(row)
            if not np.array_equal(row, new_row):
                self.grid[i] = new_row[::-1]
                moved = True
        return moved
    def move_up(self):
        moved = False
        for j in range(self.size):
            col = self.grid[:, j].copy()
            new_col = self.merge_tiles(col)
            if not np.array_equal(col, new_col):
                self.grid[:, j] = new_col
                moved = True
        return moved
    def move_down(self):
        moved = False
        for j in range(self.size):
            col = self.grid[:, j][::-1].copy()
            new_col = self.merge_tiles(col)
            if not np.array_equal(col, new_col):
                self.grid[:, j] = new_col[::-1]
                moved = True
        return moved
    def merge_tiles(self, tiles):
        tiles = tiles[tiles != 0]
        i = 0
        while i < len(tiles) - 1:
            if tiles[i] == tiles[i + 1]:
                tiles[i] *= 2
                self.score += tiles[i]
                tiles = np.delete(tiles, i + 1)
            i += 1
        return np.pad(tiles, (0, self.size - len(tiles)), 'constant')
    def is_game_over(self):
        if 0 in self.grid:
            return False
        for i in range(self.size):
            for j in range(self.size):
                current = self.grid[i][j]
                if j < self.size - 1 and self.grid[i][j + 1] == current:
                    return False
                if i < self.size - 1 and self.grid[i + 1][j] == current:
                    return False
        return True

def detect_swipe_gesture(hand_landmarks, prev_hand_landmarks, threshold=0.1):
    if prev_hand_landmarks is None:
        return None
    current_x = hand_landmarks.landmark[8].x
    current_y = hand_landmarks.landmark[8].y
    prev_x = prev_hand_landmarks.landmark[8].x
    prev_y = prev_hand_landmarks.landmark[8].y
    dx = current_x - prev_x
    dy = current_y - prev_y
    if abs(dx) > threshold or abs(dy) > threshold:
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
    return None

def main():
    # Initialize Pygame
    pygame.init()
    size = 4
    display_width = 800
    display_height = 600
    tile_size = display_width // size
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("2048 with Swipes (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    small_font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    game = Game2048(size)
    prev_hand_landmarks = None
    last_swipe_time = 0
    swipe_cooldown = 0.5  # seconds
    running = True

    colors = {
        0: (205, 205, 205),
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46)
    }

    with tracer.start_as_current_span("game_session"):
        while running:
            # --- Computer Vision Input ---
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            current_time = time.time()
            swipe = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                if prev_hand_landmarks and current_time - last_swipe_time > swipe_cooldown:
                    swipe = detect_swipe_gesture(hand_landmarks, prev_hand_landmarks)
                    if swipe and not game.game_over:
                        with tracer.start_as_current_span(f"swipe_{swipe}"):
                            moved = game.move(swipe)
                            if moved:
                                last_swipe_time = current_time
                prev_hand_landmarks = hand_landmarks
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            game = Game2048(size)
                    if event.key == pygame.K_q:
                        running = False
            # --- Pygame Rendering ---
            screen.fill((250, 248, 239))
            board_x = 50
            board_y = 100
            board_size = tile_size * size
            pygame.draw.rect(screen, (200, 200, 200), (board_x, board_y, board_size, board_size))
            for i in range(size):
                for j in range(size):
                    tile_x = board_x + j * tile_size
                    tile_y = board_y + i * tile_size
                    value = game.grid[i][j]
                    color = colors.get(value, (60, 58, 50))
                    pygame.draw.rect(screen, color, (tile_x + 5, tile_y + 5, tile_size - 10, tile_size - 10))
                    if value != 0:
                        text = font.render(str(value), True, (0, 0, 0))
                        text_rect = text.get_rect(center=(tile_x + tile_size // 2, tile_y + tile_size // 2))
                        screen.blit(text, text_rect)
            score_text = font.render(f"Score: {game.score}", True, (0, 0, 0))
            screen.blit(score_text, (50, 50))
            if game.game_over:
                over_text = font.render("Game Over!", True, (255, 0, 0))
                screen.blit(over_text, (display_width // 2 - 100, display_height // 2))
                restart_text = small_font.render("Press 'r' to restart or 'q' to quit", True, (0, 0, 0))
                screen.blit(restart_text, (display_width // 2 - 200, display_height // 2 + 50))
            else:
                info_text = small_font.render("Swipe with your hand to move tiles", True, (0, 0, 0))
                screen.blit(info_text, (50, display_height - 50))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 