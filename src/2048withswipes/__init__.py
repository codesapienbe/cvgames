import cv2
import mediapipe as mp
import numpy as np
import random
import time

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
        # Remove zeros
        tiles = tiles[tiles != 0]
        # Merge adjacent equal tiles
        i = 0
        while i < len(tiles) - 1:
            if tiles[i] == tiles[i + 1]:
                tiles[i] *= 2
                self.score += tiles[i]
                tiles = np.delete(tiles, i + 1)
            i += 1
        # Pad with zeros
        return np.pad(tiles, (0, self.size - len(tiles)), 'constant')
    
    def is_game_over(self):
        # Check if there are empty cells
        if 0 in self.grid:
            return False
        
        # Check if any merges are possible
        for i in range(self.size):
            for j in range(self.size):
                current = self.grid[i][j]
                # Check right neighbor
                if j < self.size - 1 and self.grid[i][j + 1] == current:
                    return False
                # Check bottom neighbor
                if i < self.size - 1 and self.grid[i + 1][j] == current:
                    return False
        return True

def detect_swipe_gesture(hand_landmarks, prev_hand_landmarks, threshold=0.1):
    """Detect swipe gesture based on hand movement"""
    if prev_hand_landmarks is None:
        return None
    
    # Get index finger tip position
    current_x = hand_landmarks.landmark[8].x
    current_y = hand_landmarks.landmark[8].y
    prev_x = prev_hand_landmarks.landmark[8].x
    prev_y = prev_hand_landmarks.landmark[8].y
    
    # Calculate movement
    dx = current_x - prev_x
    dy = current_y - prev_y
    
    # Check if movement is significant
    if abs(dx) > threshold or abs(dy) > threshold:
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
    
    return None

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize game
    game = Game2048()
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create fullscreen window
    cv2.namedWindow("2048 with Swipes", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("2048 with Swipes", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Game variables
    prev_hand_landmarks = None
    last_swipe_time = 0
    swipe_cooldown = 0.5  # seconds
    display_width = 800
    display_height = 600
    tile_size = display_width // game.size
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (display_width, display_height))
        
        # Process hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        current_time = time.time()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detect swipe gesture
            if prev_hand_landmarks and current_time - last_swipe_time > swipe_cooldown:
                swipe = detect_swipe_gesture(hand_landmarks, prev_hand_landmarks)
                if swipe and not game.game_over:
                    game.move(swipe)
                    last_swipe_time = current_time
            
            prev_hand_landmarks = hand_landmarks
        
        # Draw game board
        board_x = 50
        board_y = 100
        board_size = tile_size * game.size
        
        # Draw background
        cv2.rectangle(frame, (board_x, board_y), 
                     (board_x + board_size, board_y + board_size), (200, 200, 200), -1)
        
        # Draw tiles
        for i in range(game.size):
            for j in range(game.size):
                tile_x = board_x + j * tile_size
                tile_y = board_y + i * tile_size
                value = game.grid[i][j]
                
                # Tile color based on value
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
                
                color = colors.get(value, (60, 58, 50))
                cv2.rectangle(frame, (tile_x + 5, tile_y + 5), 
                             (tile_x + tile_size - 5, tile_y + tile_size - 5), color, -1)
                
                if value != 0:
                    # Draw value text
                    text = str(value)
                    font_scale = 1.0 if value < 100 else 0.8 if value < 1000 else 0.6
                    thickness = 2 if value < 100 else 1
                    
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    text_x = tile_x + (tile_size - text_width) // 2
                    text_y = tile_y + (tile_size + text_height) // 2
                    
                    cv2.putText(frame, text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # Draw score and game status
        cv2.putText(frame, f"Score: {game.score}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        if game.game_over:
            cv2.putText(frame, "Game Over!", (display_width//2 - 100, display_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(frame, "Press 'r' to restart or 'q' to quit", 
                       (display_width//2 - 200, display_height//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            cv2.putText(frame, "Swipe with your hand to move tiles", (50, display_height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.imshow("2048 with Swipes", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            game = Game2048()
            game.game_over = False
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 