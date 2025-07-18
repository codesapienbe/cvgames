import cv2
import mediapipe as mp
import numpy as np
import random
import time

class Snake:
    def __init__(self, x, y):
        self.body = [(x, y)]
        self.direction = [1, 0]  # Start moving right
        self.grow = False
    
    def move(self, width, height):
        head = self.body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Wrap around screen
        new_head = (new_head[0] % width, new_head[1] % height)
        
        self.body.insert(0, new_head)
        
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
    
    def change_direction(self, new_direction):
        # Prevent 180-degree turns
        if (self.direction[0] != -new_direction[0] or 
            self.direction[1] != -new_direction[1]):
            self.direction = new_direction
    
    def check_collision(self):
        head = self.body[0]
        return head in self.body[1:]
    
    def eat_food(self, food_pos):
        if self.body[0] == food_pos:
            self.grow = True
            return True
        return False

class GestureSnake:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.snake = Snake(width // 2, height // 2)
        self.food = self.spawn_food()
        self.score = 0
        self.game_over = False
        self.speed = 10
        self.last_update = time.time()
        self.update_interval = 0.1  # seconds
    
    def spawn_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            food_pos = (x, y)
            if food_pos not in self.snake.body:
                return food_pos
    
    def update(self):
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.snake.move(self.width, self.height)
            
            # Check collision with self
            if self.snake.check_collision():
                self.game_over = True
                return
            
            # Check if snake ate food
            if self.snake.eat_food(self.food):
                self.score += 10
                self.food = self.spawn_food()
                # Increase speed
                self.update_interval = max(0.05, self.update_interval - 0.005)
            
            self.last_update = current_time
    
    def draw(self, frame):
        # Draw snake
        for i, segment in enumerate(self.snake.body):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)  # Head is brighter
            cv2.circle(frame, (segment[0] * 20 + 10, segment[1] * 20 + 10), 8, color, -1)
            cv2.circle(frame, (segment[0] * 20 + 10, segment[1] * 20 + 10), 8, (255, 255, 255), 2)
        
        # Draw food
        cv2.circle(frame, (self.food[0] * 20 + 10, self.food[1] * 20 + 10), 8, (0, 0, 255), -1)
        cv2.circle(frame, (self.food[0] * 20 + 10, self.food[1] * 20 + 10), 8, (255, 255, 255), 2)
        
        # Draw score
        cv2.putText(frame, f"Score: {self.score}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if self.game_over:
            cv2.putText(frame, "Game Over!", (self.width * 10 - 100, self.height * 10 // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(frame, f"Final Score: {self.score}", 
                       (self.width * 10 - 150, self.height * 10 // 2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'r' to restart or 'q' to quit", 
                       (self.width * 10 - 250, self.height * 10 // 2 + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Use hand gestures to control the snake", (10, self.height * 20 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def reset_game(self):
        self.snake = Snake(self.width // 2, self.height // 2)
        self.food = self.spawn_food()
        self.score = 0
        self.game_over = False
        self.update_interval = 0.1

def detect_hand_gesture(hand_landmarks):
    """Detect hand gesture and return direction for snake"""
    if not hand_landmarks:
        return None
    
    # Get hand landmarks
    wrist = hand_landmarks.landmark[0]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Get finger positions relative to wrist
    index_x = index_tip.x - wrist.x
    index_y = index_tip.y - wrist.y
    middle_x = middle_tip.x - wrist.x
    middle_y = middle_tip.y - wrist.y
    ring_x = ring_tip.x - wrist.x
    ring_y = ring_tip.y - wrist.y
    pinky_x = pinky_tip.x - wrist.x
    pinky_y = pinky_tip.y - wrist.y
    
    # Calculate hand center
    hand_center_x = (index_x + middle_x + ring_x + pinky_x) / 4
    hand_center_y = (index_y + middle_y + ring_y + pinky_y) / 4
    
    # Determine direction based on hand position relative to center
    threshold = 0.1
    
    if abs(hand_center_x) > abs(hand_center_y):
        if hand_center_x > threshold:
            return [1, 0]  # Right
        elif hand_center_x < -threshold:
            return [-1, 0]  # Left
    else:
        if hand_center_y > threshold:
            return [0, 1]  # Down
        elif hand_center_y < -threshold:
            return [0, -1]  # Up
    
    return None

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create fullscreen window
    cv2.namedWindow("Gesture Snake", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gesture Snake", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize game
    game_width = 40
    game_height = 30
    game = GestureSnake(game_width, game_height)
    
    # Create display frame
    display_width = game_width * 20
    display_height = game_height * 20
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (display_width, display_height))
        
        # Process hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detect gesture and control snake
            direction = detect_hand_gesture(hand_landmarks)
            if direction and not game.game_over:
                game.snake.change_direction(direction)
        
        # Update and draw game
        if not game.game_over:
            game.update()
        
        game.draw(frame)
        
        cv2.imshow("Gesture Snake", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            game.reset_game()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 