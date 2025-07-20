import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import platform
import sys
from typing import List, Tuple, Optional

class Fruit:
    def __init__(self, x: int, y: int, fruit_type: str, color: Tuple[int, int, int], hanging: bool = False):
        self.x = x
        self.y = y
        # Random speed between 1 and 10
        self.speed = random.uniform(1, 10)
        self.vx = random.uniform(-self.speed, self.speed) if not hanging else 0  # No horizontal movement for hanging fruits
        self.vy = random.uniform(self.speed * 0.5, self.speed) if not hanging else 0  # Downward velocity for dropping fruits
        self.gravity = 0.3 if not hanging else 0.1  # Less gravity for hanging fruits
        self.fruit_type = fruit_type
        self.color = color
        self.radius = 75  # Increased from 25 to 75 (3x larger)
        self.sliced = False
        self.slice_time = 0
        self.hanging = hanging
        self.string_length = random.randint(50, 100) if hanging else 0
        self.swing_angle = random.uniform(0, 2 * math.pi)  # Random swing direction
        self.swing_speed = random.uniform(0.02, 0.05)  # Swing speed
        
    def update(self):
        if self.hanging:
            # Hanging fruits swing back and forth
            self.swing_angle += self.swing_speed
            self.x += math.sin(self.swing_angle) * 0.5  # Gentle swinging motion
            self.vy += self.gravity
        else:
            # Regular dropping fruits
            self.x += self.vx
            self.y += self.vy
            self.vy += self.gravity
        
    def draw(self, frame):
        if self.hanging:
            # Draw string
            string_start_x = int(self.x)
            string_start_y = 0  # Top of screen
            string_end_x = int(self.x)
            string_end_y = int(self.y - self.radius - 10)
            
            # Draw string with slight curve
            for i in range(10):
                t = i / 10.0
                curve_x = string_start_x + math.sin(self.swing_angle * 0.5) * 5 * t
                curve_y = string_start_y + (string_end_y - string_start_y) * t
                cv2.circle(frame, (int(curve_x), int(curve_y)), 1, (139, 69, 19), -1)
            
            # Draw string attachment point
            cv2.circle(frame, (string_start_x, string_start_y), 3, (139, 69, 19), -1)
        
        if not self.sliced:
            # Draw fruit
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)
            
            # Add fruit details based on type
            if self.fruit_type == "apple":
                # Apple stem
                cv2.line(frame, (int(self.x), int(self.y - self.radius)), 
                        (int(self.x), int(self.y - self.radius - 24)), (139, 69, 19), 6)  # Increased size
                cv2.circle(frame, (int(self.x), int(self.y - self.radius - 24)), 6, (34, 139, 34), -1)  # Increased size
            elif self.fruit_type == "banana":
                # Banana curve
                cv2.ellipse(frame, (int(self.x), int(self.y)), (self.radius, self.radius//2), 0, 0, 180, (0, 200, 255), 6)  # Increased thickness
            elif self.fruit_type == "grape":
                # Grape cluster effect
                for i in range(5):  # More grapes for larger size
                    offset_x = random.randint(-15, 15)  # Increased offset
                    offset_y = random.randint(-15, 15)  # Increased offset
                    cv2.circle(frame, (int(self.x + offset_x), int(self.y + offset_y)), 24, self.color, -1)  # Increased size
            
            # Add highlight
            cv2.circle(frame, (int(self.x - 24), int(self.y - 24)), 9, (255, 255, 255), -1)  # Increased size
        else:
            # Draw sliced fruit (two halves) - increased size
            cv2.ellipse(frame, (int(self.x - 30), int(self.y)), (45, 24), 0, 0, 180, self.color, -1)  # Increased size
            cv2.ellipse(frame, (int(self.x + 30), int(self.y)), (45, 24), 0, 180, 360, self.color, -1)  # Increased size

class Bomb:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        # Random speed between 1 and 10
        self.speed = random.uniform(1, 10)
        self.vx = random.uniform(-self.speed, self.speed)
        self.vy = random.uniform(self.speed * 0.5, self.speed)  # Downward velocity
        self.gravity = 0.3
        self.radius = 60  # Increased from 20 to 60 (3x larger)
        self.exploded = False
        self.explosion_time = 0
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        
    def draw(self, frame):
        if not self.exploded:
            # Draw bomb body
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (50, 50, 50), -1)
            # Draw fuse
            cv2.line(frame, (int(self.x), int(self.y - self.radius)), 
                    (int(self.x), int(self.y - self.radius - 45)), (139, 69, 19), 9)  # Increased size
            # Draw fuse tip
            cv2.circle(frame, (int(self.x), int(self.y - self.radius - 45)), 15, (255, 165, 0), -1)  # Increased size
        else:
            # Draw explosion
            explosion_radius = int(90 + (time.time() - self.explosion_time) * 60)  # Increased explosion size
            cv2.circle(frame, (int(self.x), int(self.y)), explosion_radius, (0, 0, 255), 6)  # Increased thickness

class Sword:
    def __init__(self):
        self.length = 80
        self.width = 8
        self.handle_length = 20
        self.handle_width = 12
        
    def draw(self, frame, finger_pos: Tuple[int, int], hand_landmarks):
        if not finger_pos or not hand_landmarks:
            return
            
        # Get index finger tip and middle finger tip for sword direction
        index_tip = hand_landmarks.landmark[8]  # Index finger tip
        middle_tip = hand_landmarks.landmark[12]  # Middle finger tip
        wrist = hand_landmarks.landmark[0]  # Wrist
        
        h, w, _ = frame.shape
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        
        # Calculate sword direction (from wrist to index finger)
        sword_dx = index_x - wrist_x
        sword_dy = index_y - wrist_y
        
        # Normalize and extend
        length = math.sqrt(sword_dx**2 + sword_dy**2)
        if length > 0:
            sword_dx = (sword_dx / length) * self.length
            sword_dy = (sword_dy / length) * self.length
        
        # Sword tip position
        sword_tip_x = int(index_x + sword_dx)
        sword_tip_y = int(index_y + sword_dy)
        
        # Sword handle position (behind the hand)
        handle_x = int(wrist_x - sword_dx * 0.3)
        handle_y = int(wrist_y - sword_dy * 0.3)
        
        # Draw sword handle (brown)
        cv2.line(frame, (handle_x, handle_y), (index_x, index_y), (139, 69, 19), self.handle_width)
        
        # Draw sword blade (silver with gradient)
        for i in range(10):
            t = i / 10.0
            x1 = int(index_x + sword_dx * t)
            y1 = int(index_y + sword_dy * t)
            x2 = int(index_x + sword_dx * (t + 0.1))
            y2 = int(index_y + sword_dy * (t + 0.1))
            
            # Gradient from handle to tip
            color_intensity = int(200 - 100 * t)  # Brighter at handle, darker at tip
            cv2.line(frame, (x1, y1), (x2, y2), (color_intensity, color_intensity, color_intensity), 
                    max(1, int(self.width * (1 - t * 0.5))))
        
        # Draw sword tip
        cv2.circle(frame, (sword_tip_x, sword_tip_y), 3, (255, 255, 255), -1)
        
        # Draw sword guard (cross-guard)
        guard_length = 20
        guard_x1 = int(index_x - sword_dy * 0.3)
        guard_y1 = int(index_y + sword_dx * 0.3)
        guard_x2 = int(index_x + sword_dy * 0.3)
        guard_y2 = int(index_y - sword_dx * 0.3)
        cv2.line(frame, (guard_x1, guard_y1), (guard_x2, guard_y2), (192, 192, 192), 4)
        
        return (sword_tip_x, sword_tip_y)  # Return sword tip position for collision detection

class FruitNinja:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera with better error handling
        self.cap = self.initialize_camera()
        if self.cap is None:
            raise RuntimeError("Failed to initialize camera. Please check your webcam connection.")
        
        # Game state
        self.score = 0
        self.bombs_exploded = 0
        self.game_over = False
        self.fruits: List[Fruit] = []
        self.bombs: List[Bomb] = []
        self.swipe_trail: List[Tuple[int, int]] = []
        self.last_hand_pos: Optional[Tuple[int, int]] = None
        self.sword = Sword()
        
        # Fruit types and colors
        self.fruit_types = [
            ("apple", (0, 0, 255)),      # Red
            ("orange", (0, 165, 255)),   # Orange
            ("banana", (0, 255, 255)),   # Yellow
            ("grape", (128, 0, 128)),    # Purple
            ("strawberry", (0, 0, 128))  # Dark red
        ]
        
        # Spawn timers
        self.fruit_spawn_timer = 0
        self.bomb_spawn_timer = 0
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_thickness = 2
        
    def initialize_camera(self):
        """Initialize camera with fallback options and macOS-specific handling"""
        print("🔍 Initializing camera...")
        
        # Check if we're on macOS
        is_macos = platform.system() == "Darwin"
        if is_macos:
            print("🍎 Detected macOS - using specialized camera handling")
        
        # Try different camera backends and indices
        camera_configs = []
        
        if is_macos:
            # macOS-specific configurations
            camera_configs.extend([
                (0, cv2.CAP_AVFOUNDATION),  # AVFoundation backend
                (0, cv2.CAP_ANY),           # Auto-detect backend
                (1, cv2.CAP_AVFOUNDATION),  # Try second camera with AVFoundation
                (1, cv2.CAP_ANY),           # Try second camera with auto-detect
            ])
        else:
            # Generic configurations for other platforms
            camera_configs.extend([
                (0, cv2.CAP_ANY),
                (1, cv2.CAP_ANY),
                (-1, cv2.CAP_ANY),
            ])
        
        for camera_index, backend in camera_configs:
            backend_name = "AVFoundation" if backend == cv2.CAP_AVFOUNDATION else "Auto-detect"
            print(f"   Trying camera index: {camera_index} with backend: {backend_name}")
            
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                
                if cap.isOpened():
                    # Try to read a frame to confirm camera is working
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Camera initialized successfully (index: {camera_index}, backend: {backend_name})")
                        
                        # Set camera properties
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        # Try to enable autofocus (may not work on all cameras)
                        try:
                            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                        except:
                            pass  # Autofocus not supported
                        
                        return cap
                    else:
                        print(f"   Camera {camera_index} opened but failed to read frame")
                        cap.release()
                else:
                    print(f"   Camera {camera_index} failed to open")
            except Exception as e:
                print(f"   Error with camera {camera_index}: {e}")
                continue
        
        # If no camera found, try with minimal settings
        print("   Trying minimal camera settings...")
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print("✅ Camera initialized with minimal settings")
                    return cap
                else:
                    cap.release()
        except Exception as e:
            print(f"   Minimal settings failed: {e}")
        
        print("❌ No working camera found")
        
        # Provide platform-specific troubleshooting
        if is_macos:
            print("\n🍎 macOS Camera Troubleshooting:")
            print("1. Check System Preferences > Security & Privacy > Camera")
            print("   - Make sure your terminal/IDE has camera access")
            print("   - If using VS Code, check if it has camera permissions")
            print("2. Try running from Terminal.app instead of VS Code")
            print("3. Check if another app is using the camera")
            print("4. Try: sudo killall VDCAssistant")
            print("5. Restart your computer if issues persist")
        else:
            print("\n💡 General Camera Troubleshooting:")
            print("1. Make sure your webcam is connected")
            print("2. Check if another application is using the camera")
            print("3. Try restarting your computer")
            print("4. Check device manager for camera issues")
        
        return None
        
    def spawn_fruit(self):
        if time.time() - self.fruit_spawn_timer > 1.5:  # Spawn every 1.5 seconds
            x = random.randint(50, 1230)
            y = -50  # Start from above the screen
            fruit_type, color = random.choice(self.fruit_types)
            
            # 30% chance of hanging fruit
            hanging = random.random() < 0.3
            if hanging:
                y = random.randint(50, 200)  # Hanging fruits start from top area
            
            self.fruits.append(Fruit(x, y, fruit_type, color, hanging))
            self.fruit_spawn_timer = time.time()
            
    def spawn_bomb(self):
        if time.time() - self.bomb_spawn_timer > 3.0:  # Spawn every 3 seconds
            x = random.randint(50, 1230)
            y = -50  # Start from above the screen
            self.bombs.append(Bomb(x, y))
            self.bomb_spawn_timer = time.time()
            
    def check_swipe_collision(self, sword_tip_pos: Tuple[int, int]):
        if not sword_tip_pos:
            return
            
        # Check fruit collisions
        for fruit in self.fruits:
            if not fruit.sliced:
                distance = math.sqrt((sword_tip_pos[0] - fruit.x)**2 + (sword_tip_pos[1] - fruit.y)**2)
                if distance < fruit.radius + 15:  # Sword reach
                    fruit.sliced = True
                    fruit.slice_time = time.time()
                    self.score += 10
                    
        # Check bomb collisions
        for bomb in self.bombs:
            if not bomb.exploded:
                distance = math.sqrt((sword_tip_pos[0] - bomb.x)**2 + (sword_tip_pos[1] - bomb.y)**2)
                if distance < bomb.radius + 15:
                    bomb.exploded = True
                    bomb.explosion_time = time.time()
                    self.bombs_exploded += 1
                    self.score = max(0, self.score - 20)  # Lose 20 points
                    
                    if self.bombs_exploded >= 3:
                        self.game_over = True
                        
    def update_objects(self):
        # Update fruits
        for fruit in self.fruits[:]:
            fruit.update()
            # Remove fruits that are off screen or sliced for too long
            if fruit.y > 800 or (fruit.sliced and time.time() - fruit.slice_time > 1.0):
                self.fruits.remove(fruit)
                
        # Update bombs
        for bomb in self.bombs[:]:
            bomb.update()
            if bomb.y > 800 or (bomb.exploded and time.time() - bomb.explosion_time > 1.0):
                self.bombs.remove(bomb)
                
    def draw_ui(self, frame):
        # Draw score
        cv2.putText(frame, f"Score: {self.score}", (10, 50), 
                   self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        # Draw bombs exploded
        cv2.putText(frame, f"Bombs: {self.bombs_exploded}/3", (10, 100), 
                   self.font, self.font_scale, (0, 0, 255), self.font_thickness)
        
        # Draw sword trail
        if len(self.swipe_trail) > 1:
            for i in range(len(self.swipe_trail) - 1):
                cv2.line(frame, self.swipe_trail[i], self.swipe_trail[i+1], 
                        (0, 255, 0), 2)
                
        # Draw game over screen
        if self.game_over:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 720), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, "GAME OVER", (400, 300), 
                       self.font, 3, (0, 0, 255), 4)
            cv2.putText(frame, f"Final Score: {self.score}", (450, 400), 
                       self.font, 2, (255, 255, 255), 3)
            cv2.putText(frame, "Press 'q' to quit", (500, 500), 
                       self.font, 1, (255, 255, 255), 2)
                       
    def run(self):
        print("🎯 Fruit Ninja CV Game Started!")
        print("📋 Instructions:")
        print("   - Use your index finger as a sword")
        print("   - Slice fruits to score points")
        print("   - Avoid bombs (3 bombs = game over)")
        print("   - Some fruits hang from strings!")
        print("   - Press 'q' to quit")
        print()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Track hand and draw sword
            current_hand_pos = None
            sword_tip_pos = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get index finger tip for hand position
                    index_tip = hand_landmarks.landmark[8]
                    h, w, _ = frame.shape
                    x, y = int(index_tip.x * w), int(index_tip.y * h)
                    current_hand_pos = (x, y)
                    
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Draw sword and get sword tip position
                    sword_tip_pos = self.sword.draw(frame, current_hand_pos, hand_landmarks)
                    
            # Update swipe trail
            if sword_tip_pos:
                self.swipe_trail.append(sword_tip_pos)
                if len(self.swipe_trail) > 10:  # Keep last 10 positions
                    self.swipe_trail.pop(0)
                    
                # Check collisions with sword tip
                self.check_swipe_collision(sword_tip_pos)
                
                self.last_hand_pos = sword_tip_pos
            else:
                # Clear trail when hand is not detected
                self.swipe_trail.clear()
                
            if not self.game_over:
                # Spawn objects
                self.spawn_fruit()
                self.spawn_bomb()
                
                # Update game objects
                self.update_objects()
                
                # Draw objects
                for fruit in self.fruits:
                    fruit.draw(frame)
                for bomb in self.bombs:
                    bomb.draw(frame)
                    
            # Draw UI
            self.draw_ui(frame)
            
            # Show frame
            cv2.imshow('Fruit Ninja CV', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.game_over:
                # Restart game
                self.score = 0
                self.bombs_exploded = 0
                self.game_over = False
                self.fruits.clear()
                self.bombs.clear()
                self.swipe_trail.clear()
                
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"🎮 Game ended! Final score: {self.score}")

def main():
    try:
        game = FruitNinja()
        game.run()
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("💡 Troubleshooting tips:")
        print("   - Make sure your webcam is connected and not in use by another application")
        print("   - Check if your webcam permissions are enabled")
        print("   - Try closing other applications that might be using the camera")
        print("   - On macOS, ensure camera access is granted in System Preferences > Security & Privacy")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()