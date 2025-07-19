import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import sys
import os
import argparse

# Import the back button system
try:
    # Try relative import first
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cvstore'))
    from back_button import BackButton
except ImportError:
    # Fallback to absolute import
    import sys
    import os
    # Add the project root to the path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)
    sys.path.append(os.path.join(project_root, 'src', 'cvstore'))
    from back_button import BackButton

class Duck:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.reset()
        
    def reset(self):
        """Reset duck to starting position with random direction"""
        # Start from bottom of screen
        self.x = random.randint(50, self.screen_width - 50)
        self.y = self.screen_height + 50
        
        # Random direction (upward with slight horizontal movement)
        self.speed_x = random.uniform(-3, 3)
        self.speed_y = random.uniform(-8, -4)  # Negative for upward movement
        
        # Duck size
        self.width = 60
        self.height = 40
        
        # Animation
        self.wing_flap = 0
        self.wing_speed = 0.2
        
        # State
        self.alive = True
        self.hit = False
        
    def update(self):
        """Update duck position and animation"""
        if not self.alive:
            return
            
        # Update position
        self.x += self.speed_x
        self.y += self.speed_y
        
        # Update wing animation
        self.wing_flap += self.wing_speed
        
        # Check if duck is off screen
        if self.y < -50 or self.x < -50 or self.x > self.screen_width + 50:
            self.alive = False
            
    def draw(self, frame):
        """Draw the duck"""
        if not self.alive:
            return
            
        # Duck body (brown oval)
        cv2.ellipse(frame, 
                   (int(self.x), int(self.y)), 
                   (self.width // 2, self.height // 2), 
                   0, 0, 360, (139, 69, 19), -1)
        
        # Duck head (darker brown circle)
        head_x = int(self.x + self.width // 3)
        head_y = int(self.y - self.height // 4)
        cv2.circle(frame, (head_x, head_y), self.height // 3, (101, 67, 33), -1)
        
        # Duck beak (orange triangle)
        beak_x = head_x + self.height // 3
        beak_y = head_y
        beak_points = np.array([
            [beak_x, beak_y],
            [beak_x + 15, beak_y - 5],
            [beak_x + 15, beak_y + 5]
        ], np.int32)
        cv2.fillPoly(frame, [beak_points], (0, 165, 255))
        
        # Wings (flapping animation)
        wing_offset = int(10 * math.sin(self.wing_flap))
        wing_y = int(self.y + wing_offset)
        
        # Left wing
        left_wing_points = np.array([
            [self.x - self.width // 2, wing_y],
            [self.x - self.width // 2 - 20, wing_y - 15],
            [self.x - self.width // 2 - 10, wing_y + 10]
        ], np.int32)
        cv2.fillPoly(frame, [left_wing_points], (160, 82, 45))
        
        # Right wing
        right_wing_points = np.array([
            [self.x + self.width // 2, wing_y],
            [self.x + self.width // 2 + 20, wing_y - 15],
            [self.x + self.width // 2 + 10, wing_y + 10]
        ], np.int32)
        cv2.fillPoly(frame, [right_wing_points], (160, 82, 45))
        
        # Eye
        eye_x = head_x + 5
        eye_y = head_y - 5
        cv2.circle(frame, (eye_x, eye_y), 3, (0, 0, 0), -1)
        
    def check_hit(self, shot_x, shot_y):
        """Check if duck was hit by shot"""
        if not self.alive or self.hit:
            return False
            
        # Simple collision detection
        distance = math.sqrt((self.x - shot_x)**2 + (self.y - shot_y)**2)
        return distance < self.width // 2

class DuckHuntGame:
    def __init__(self, screen_width=1280, screen_height=720):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Game state
        self.score = 0
        self.ducks_shot = 0
        self.ducks_missed = 0
        self.game_time = 60  # 60 seconds
        self.time_remaining = self.game_time
        self.game_start_time = time.time()
        
        # Duck management
        self.ducks = []
        self.max_ducks = 3
        self.duck_spawn_timer = 0
        self.duck_spawn_interval = 2.0  # Spawn a new duck every 2 seconds
        
        # Shooting mechanics
        self.shot_cooldown = 0
        self.shot_cooldown_time = 0.5  # 0.5 seconds between shots
        self.last_shot_time = 0
        
        # Hand tracking
        self.hand_position = None
        self.gun_position = None
        self.aiming = False
        
        # Visual effects
        self.shots = []  # List of shot positions for visual effect
        self.hit_effects = []  # List of hit effects
        
        # Back button
        self.back_button = BackButton(screen_width, screen_height)
        
        # Colors
        self.colors = {
            'sky': (135, 206, 235),
            'grass': (34, 139, 34),
            'text': (255, 255, 255),
            'score': (255, 215, 0),
            'timer': (255, 69, 0),
            'shot': (255, 255, 0),
            'hit': (255, 0, 0)
        }
        
    def detect_gun_gesture(self, landmarks):
        """Detect if hand is in gun gesture (index finger pointing)"""
        if len(landmarks.landmark) < 21:
            return False, None
            
        # Get finger positions
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]
        
        # Get finger base positions
        thumb_base = landmarks.landmark[3]
        index_base = landmarks.landmark[6]
        middle_base = landmarks.landmark[10]
        ring_base = landmarks.landmark[14]
        pinky_base = landmarks.landmark[18]
        
        # Check if index finger is extended and others are closed
        index_extended = index_tip.y < index_base.y
        middle_closed = middle_tip.y > middle_base.y
        ring_closed = ring_tip.y > ring_base.y
        pinky_closed = pinky_tip.y > pinky_base.y
        
        # Thumb can be either extended or closed for gun gesture
        thumb_ok = True
        
        if index_extended and middle_closed and ring_closed and pinky_closed and thumb_ok:
            # Calculate gun position (index finger tip)
            gun_x = int(index_tip.x * self.screen_width)
            gun_y = int(index_tip.y * self.screen_height)
            return True, (gun_x, gun_y)
            
        return False, None
        
    def spawn_duck(self):
        """Spawn a new duck"""
        if len(self.ducks) < self.max_ducks:
            duck = Duck(self.screen_width, self.screen_height)
            self.ducks.append(duck)
            
    def shoot(self, shot_x, shot_y):
        """Handle shooting mechanics"""
        current_time = time.time()
        if current_time - self.last_shot_time < self.shot_cooldown_time:
            return
            
        self.last_shot_time = current_time
        
        # Add shot visual effect
        self.shots.append({
            'x': shot_x,
            'y': shot_y,
            'time': current_time,
            'duration': 0.3
        })
        
        # Check for hits
        for duck in self.ducks:
            if duck.check_hit(shot_x, shot_y):
                duck.hit = True
                duck.alive = False
                self.score += 10
                self.ducks_shot += 1
                
                # Add hit effect
                self.hit_effects.append({
                    'x': duck.x,
                    'y': duck.y,
                    'time': current_time,
                    'duration': 1.0
                })
                break
                
    def update(self):
        """Update game state"""
        current_time = time.time()
        
        # Update time remaining
        elapsed = current_time - self.game_start_time
        self.time_remaining = max(0, self.game_time - elapsed)
        
        # Spawn ducks
        if current_time - self.duck_spawn_timer > self.duck_spawn_interval:
            self.spawn_duck()
            self.duck_spawn_timer = current_time
            
        # Update ducks
        for duck in self.ducks:
            duck.update()
            
        # Remove dead ducks
        self.ducks = [duck for duck in self.ducks if duck.alive]
        
        # Update shot effects
        self.shots = [shot for shot in self.shots 
                     if current_time - shot['time'] < shot['duration']]
        
        # Update hit effects
        self.hit_effects = [effect for effect in self.hit_effects 
                           if current_time - effect['time'] < effect['duration']]
        
        # Count missed ducks
        for duck in self.ducks:
            if not duck.alive and not duck.hit:
                self.ducks_missed += 1
                
    def draw(self, frame):
        """Draw the game"""
        # Draw sky background
        frame[:] = self.colors['sky']
        
        # Draw grass at bottom
        grass_height = 100
        cv2.rectangle(frame, (0, self.screen_height - grass_height), 
                     (self.screen_width, self.screen_height), 
                     self.colors['grass'], -1)
        
        # Draw ducks
        for duck in self.ducks:
            duck.draw(frame)
            
        # Draw shot effects
        current_time = time.time()
        for shot in self.shots:
            alpha = 1.0 - (current_time - shot['time']) / shot['duration']
            if alpha > 0:
                color = tuple(int(c * alpha) for c in self.colors['shot'])
                cv2.circle(frame, (shot['x'], shot['y']), 10, color, -1)
                cv2.circle(frame, (shot['x'], shot['y']), 15, color, 2)
                
        # Draw hit effects
        for effect in self.hit_effects:
            alpha = 1.0 - (current_time - effect['time']) / effect['duration']
            if alpha > 0:
                color = tuple(int(c * alpha) for c in self.colors['hit'])
                size = int(30 * alpha)
                cv2.circle(frame, (int(effect['x']), int(effect['y'])), size, color, 3)
                
        # Draw gun cursor
        if self.gun_position:
            gun_x, gun_y = self.gun_position
            cv2.circle(frame, (gun_x, gun_y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (gun_x, gun_y), 8, (255, 0, 0), 2)
            
        # Draw UI
        self.draw_ui(frame)
        
        # Draw back button
        self.back_button.draw(frame, self.hand_position)
        
    def draw_ui(self, frame):
        """Draw user interface"""
        # Score
        score_text = f"Score: {self.score}"
        cv2.putText(frame, score_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['score'], 2)
        
        # Time remaining
        time_text = f"Time: {int(self.time_remaining)}s"
        cv2.putText(frame, time_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['timer'], 2)
        
        # Ducks shot
        ducks_text = f"Ducks Shot: {self.ducks_shot}"
        cv2.putText(frame, ducks_text, (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
        
        # Instructions
        if self.time_remaining > 0:
            instruction_text = "Point your index finger to aim and shoot!"
            cv2.putText(frame, instruction_text, 
                       (self.screen_width // 2 - 200, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Game over screen
        if self.time_remaining <= 0:
            self.draw_game_over(frame)
            
    def draw_game_over(self, frame):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Game over text
        game_over_text = "GAME OVER"
        text_size = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 3)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        text_y = self.screen_height // 2 - 50
        cv2.putText(frame, game_over_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        
        # Final score
        final_score_text = f"Final Score: {self.score}"
        score_size = cv2.getTextSize(final_score_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
        score_x = (self.screen_width - score_size[0]) // 2
        score_y = text_y + 80
        cv2.putText(frame, final_score_text, (score_x, score_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, self.colors['score'], 2)
        
        # Ducks shot
        ducks_text = f"Ducks Shot: {self.ducks_shot}"
        ducks_size = cv2.getTextSize(ducks_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        ducks_x = (self.screen_width - ducks_size[0]) // 2
        ducks_y = score_y + 60
        cv2.putText(frame, ducks_text, (ducks_x, ducks_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.colors['text'], 2)
        
        # Press any key to continue
        continue_text = "Press any key to return to App Store"
        continue_size = cv2.getTextSize(continue_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        continue_x = (self.screen_width - continue_size[0]) // 2
        continue_y = ducks_y + 60
        cv2.putText(frame, continue_text, (continue_x, continue_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
        
    def is_game_over(self):
        """Check if game is over"""
        return self.time_remaining <= 0

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Duck Hunt Game with Hand Tracking')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    args = parser.parse_args()

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {args.camera}")
        exit(-1)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize game
    game = DuckHuntGame(1280, 720)
    
    # Create window
    cv2.namedWindow('Duck Hunt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Duck Hunt', 1280, 720)

    print("🎯 Duck Hunt Game Started!")
    print("📋 Instructions:")
    print("   - Point your index finger to aim")
    print("   - Keep other fingers closed for gun gesture")
    print("   - Shoot ducks to score points")
    print("   - Press 'B' or use back button to exit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Resize frame
        frame = cv2.resize(frame, (1280, 720))

        # Process hand tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = game.hands.process(rgb_frame)

        # Handle hand tracking
        game.hand_position = None
        game.gun_position = None
        game.aiming = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand position for back button
                hand_x = int(hand_landmarks.landmark[9].x * 1280)
                hand_y = int(hand_landmarks.landmark[9].y * 720)
                game.hand_position = (hand_x, hand_y)

                # Check for gun gesture
                is_gun, gun_pos = game.detect_gun_gesture(hand_landmarks)
                if is_gun:
                    game.aiming = True
                    game.gun_position = gun_pos

                # Draw hand landmarks (optional)
                # game.mp_drawing.draw_landmarks(frame, hand_landmarks, game.mp_hands.HAND_CONNECTIONS)

        # Handle shooting
        if game.aiming and game.gun_position and not game.is_game_over():
            game.shoot(game.gun_position[0], game.gun_position[1])

        # Handle back button input
        key = cv2.waitKey(1) & 0xFF
        if game.back_button.handle_input(key, results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None, game.hand_position):
            print("User approved exit - returning to app store")
            break

        # Update game
        game.update()

        # Draw game
        game.draw(frame)

        # Show frame
        cv2.imshow('Duck Hunt', frame)

        # Handle keyboard input
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('r'):  # Restart game
            game = DuckHuntGame(1280, 720)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Thanks for playing Duck Hunt!")

if __name__ == "__main__":
    main()
