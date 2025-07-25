import cv2
import mediapipe as mp
import numpy as np
import random
import time
import pygame
import sys
import logging
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Handle imports for both module and direct execution
try:
    # Try relative imports first (when run as module)
    from .config import ConfigManager
    from .util import ScreenManager
except ImportError:
    # Fallback to absolute imports (when run directly)
    try:
        from config import ConfigManager
        from util import ScreenManager
    except ImportError:
        # Create fallback classes if imports fail
        class ConfigManager:
            def __init__(self, db_path="settings.sqlite"):
                self.db_path = db_path
            def get_resolution(self):
                return (1920, 1080)
        
        class ScreenManager:
            def __init__(self, config_manager=None):
                self.pygame_init = False
                self.screen = None
                self.display_width = 1920
                self.display_height = 1080
                self.scale_factor = 1.0
                self.font_scale = 1.0
                self.config_manager = config_manager
            
            def initialize(self, title="Game Window", fullscreen=True):
                if not self.pygame_init:
                    pygame.init()
                    pygame.font.init()  # Ensure font module is initialized
                    self.pygame_init = True
                
                if self.config_manager:
                    self.display_width, self.display_height = self.config_manager.get_resolution()
                
                flags = pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE if fullscreen else pygame.DOUBLEBUF | pygame.HWSURFACE
                self.screen = pygame.display.set_mode((self.display_width, self.display_height), flags)
                pygame.display.set_caption(title)
                return self.screen
            
            def get_scaled_font(self, base_size, font_name="Arial"):
                return pygame.font.SysFont(font_name, base_size)
            
            def get_optimal_tile_size(self, grid_size, margin_ratio=0.1):
                available_width = self.display_width * (1 - margin_ratio)
                available_height = self.display_height * (1 - margin_ratio)
                return min(available_width // grid_size, available_height // grid_size)
            
            def center_rect(self, rect_width, rect_height):
                return (
                    (self.display_width - rect_width) // 2,
                    (self.display_height - rect_height) // 2
                )
            
            def cleanup(self):
                if self.pygame_init:
                    pygame.quit()
                    self.pygame_init = False

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "component": "2048_game", "message": "%(message)s"}',
    handlers=[logging.FileHandler('application.log')]
)
logger = logging.getLogger(__name__)

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
        logger.info("Game2048 initialized", extra={"grid_size": size, "score": self.score})
    
    def add_new_tile(self):
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i][j] = 2 if random.random() < 0.9 else 4
            logger.debug("Added new tile", extra={"position": (i, j), "value": self.grid[i][j]})
    
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
                logger.info("Game over reached", extra={"final_score": self.score})
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
    # Initialize pygame first
    pygame.init()
    pygame.font.init()  # Initialize font module
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Initialize screen manager with config
    screen_manager = ScreenManager(config_manager)
    screen = screen_manager.initialize("2048 with Swipes", fullscreen=True)
    
    # Get scaled fonts
    font = screen_manager.get_scaled_font(48)
    small_font = screen_manager.get_scaled_font(24)
    clock = pygame.time.Clock()

    # Initialize MediaPipe Hands
    hands = None
    mp_draw = None
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        mp_draw = mp.solutions.drawing_utils
        logger.info("MediaPipe Hands initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize MediaPipe", extra={"error": str(e)})
        print("Warning: MediaPipe not available. Hand gestures will be disabled.")

    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        print("Error: Could not open webcam.")
        screen_manager.cleanup()
        return

    game = Game2048(4)
    prev_hand_landmarks = None
    last_swipe_time = 0
    swipe_cooldown = 0.5  # seconds
    running = True

    # Calculate tile size based on screen dimensions for fullscreen experience
    size = 4
    # Use much smaller margins to maximize screen usage
    tile_size = screen_manager.get_optimal_tile_size(size, margin_ratio=0.05)
    board_size = tile_size * size
    
    # Calculate board position to fill most of the screen
    # Leave some space for score and controls
    score_height = 80
    controls_height = 60
    available_height = screen_manager.display_height - score_height - controls_height
    
    # Center the board horizontally and position it below the score
    board_x = (screen_manager.display_width - board_size) // 2
    board_y = score_height + (available_height - board_size) // 2
    
    # Ensure board doesn't exceed screen bounds
    if board_size > min(screen_manager.display_width, available_height):
        # Recalculate with smaller tiles if board is too large
        max_board_size = min(screen_manager.display_width, available_height) - 40  # 20px margin on each side
        tile_size = max_board_size // size
        board_size = tile_size * size
        board_x = (screen_manager.display_width - board_size) // 2
        board_y = score_height + (available_height - board_size) // 2

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
            
            if hands:
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
                                    logger.info("Swipe detected", extra={"direction": swipe, "score": game.score})
                    prev_hand_landmarks = hand_landmarks
            
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            game = Game2048(size)
                            logger.info("Game restarted")
                    elif event.key == pygame.K_q:
                        running = False
                    elif not game.game_over:
                        # Keyboard controls
                        if event.key in [pygame.K_LEFT, pygame.K_a]:
                            game.move('left')
                        elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                            game.move('right')
                        elif event.key in [pygame.K_UP, pygame.K_w]:
                            game.move('up')
                        elif event.key in [pygame.K_DOWN, pygame.K_s]:
                            game.move('down')
            
            # --- Pygame Rendering ---
            screen.fill((250, 248, 239))
            
            # Draw game board
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
            
            # Draw score and controls
            score_text = font.render(f"Score: {game.score}", True, (0, 0, 0))
            screen.blit(score_text, (50, 50))
            
            # Draw control hints
            controls_text = small_font.render("Swipe hand or use arrow keys/WASD | R=Restart | Q=Quit | ESC=Exit", True, (100, 100, 100))
            screen.blit(controls_text, (50, screen_manager.display_height - 50))
            
            if game.game_over:
                # Game over overlay
                overlay = pygame.Surface((screen_manager.display_width, screen_manager.display_height))
                overlay.set_alpha(150)
                overlay.fill((0, 0, 0))
                screen.blit(overlay, (0, 0))
                
                over_text = font.render("Game Over!", True, (255, 0, 0))
                over_rect = over_text.get_rect(center=(screen_manager.display_width // 2, screen_manager.display_height // 2 - 50))
                screen.blit(over_text, over_rect)
                
                final_score_text = font.render(f"Final Score: {game.score}", True, (255, 255, 255))
                final_score_rect = final_score_text.get_rect(center=(screen_manager.display_width // 2, screen_manager.display_height // 2))
                screen.blit(final_score_text, final_score_rect)
                
                restart_text = small_font.render("Press 'R' to restart or 'Q' to quit", True, (255, 255, 255))
                restart_rect = restart_text.get_rect(center=(screen_manager.display_width // 2, screen_manager.display_height // 2 + 50))
                screen.blit(restart_text, restart_rect)
            
            pygame.display.flip()
            clock.tick(30)
    
    logger.info("Game session ended", extra={"final_score": game.score})
    cap.release()
    screen_manager.cleanup()
    sys.exit()

if __name__ == "__main__":
    main() 