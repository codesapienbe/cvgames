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
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cvstore'))
    from back_button import BackButton
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)
    sys.path.append(os.path.join(project_root, 'src', 'cvstore'))
    from back_button import BackButton

class RetroColors:
    """NES Duck Hunt authentic color palette"""
    SKY_BLUE = (255, 206, 84)        # Classic sky blue
    GRASS_GREEN = (0, 168, 0)        # Bright green grass
    DUCK_BROWN = (136, 80, 0)        # Duck body brown
    DUCK_HEAD = (84, 50, 0)          # Darker head brown
    DUCK_WING = (168, 100, 0)        # Wing brown
    BEAK_ORANGE = (252, 188, 176)    # Duck beak
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (252, 216, 168)         # Score yellow
    RED = (164, 228, 252)            # Hit effects
    GREEN = (0, 255, 0)              # Ready indicator
    CROSSHAIR_RED = (0, 0, 168)      # Crosshair color

class Duck:
    def __init__(self, screen_width, screen_height, duck_type="normal"):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.duck_type = duck_type
        self.reset()
        
    def reset(self):
        """Reset duck to starting position with authentic flight pattern"""
        # Start positions vary like original game
        start_positions = [
            (0, self.screen_height - 200),           # Left side
            (self.screen_width, self.screen_height - 200),  # Right side
            (self.screen_width // 2, self.screen_height)     # Bottom center
        ]
        
        start_pos = random.choice(start_positions)
        self.x = start_pos[0]
        self.y = start_pos[1]
        
        # Authentic flight patterns
        if self.x <= 50:  # Coming from left
            self.speed_x = random.uniform(3, 6)
            self.speed_y = random.uniform(-4, -1)
        elif self.x >= self.screen_width - 50:  # Coming from right
            self.speed_x = random.uniform(-6, -3)
            self.speed_y = random.uniform(-4, -1)
        else:  # Coming from bottom
            self.speed_x = random.uniform(-2, 2)
            self.speed_y = random.uniform(-6, -3)
        
        # Duck properties
        self.width = 80
        self.height = 60
        
        # Animation
        self.wing_flap = 0
        self.wing_speed = 0.3
        self.direction = 1 if self.speed_x > 0 else -1
        
        # State
        self.alive = True
        self.hit = False
        self.falling = False
        self.fall_speed = 0
        self.hit_time = 0
        
    def update(self):
        """Update duck with authentic behavior"""
        if self.falling:
            # Falling animation when shot
            self.fall_speed += 0.5  # Gravity
            self.y += self.fall_speed
            if self.y > self.screen_height:
                self.alive = False
            return
            
        if not self.alive:
            return
        
        # Normal flight update
        self.x += self.speed_x
        self.y += self.speed_y
        
        # Wing flapping animation
        self.wing_flap += self.wing_speed
        
        # Slight bobbing motion (authentic duck flight)
        self.y += math.sin(time.time() * 2) * 0.5
        
        # Change direction occasionally (like original)
        if random.random() < 0.01:  # 1% chance per frame
            self.speed_x *= random.uniform(0.8, 1.2)
            self.speed_y *= random.uniform(0.8, 1.2)
            
        # Update direction for sprite flipping
        if self.speed_x != 0:
            self.direction = 1 if self.speed_x > 0 else -1
        
        # Check if duck escaped
        if (self.y < -50 or self.x < -100 or self.x > self.screen_width + 100):
            self.alive = False
            
    def draw(self, frame):
        """Draw authentic retro-style duck"""
        if not self.alive and not self.falling:
            return
            
        # Duck body (main brown oval)
        body_width = self.width // 2
        body_height = self.height // 3
        
        cv2.ellipse(frame, 
                   (int(self.x), int(self.y)), 
                   (body_width, body_height), 
                   0, 0, 360, RetroColors.DUCK_BROWN, -1)
        
        # Duck head (circular, positioned based on direction)
        head_offset_x = (self.width // 3) * self.direction
        head_x = int(self.x + head_offset_x)
        head_y = int(self.y - self.height // 4)
        head_radius = self.height // 3
        
        cv2.circle(frame, (head_x, head_y), head_radius, RetroColors.DUCK_HEAD, -1)
        
        # Duck beak (triangular, direction-aware)
        beak_length = 20
        beak_x = head_x + (head_radius * self.direction)
        beak_y = head_y
        
        if self.direction > 0:  # Facing right
            beak_points = np.array([
                [beak_x, beak_y],
                [beak_x + beak_length, beak_y - 8],
                [beak_x + beak_length, beak_y + 8]
            ], np.int32)
        else:  # Facing left
            beak_points = np.array([
                [beak_x, beak_y],
                [beak_x - beak_length, beak_y - 8],
                [beak_x - beak_length, beak_y + 8]
            ], np.int32)
            
        cv2.fillPoly(frame, [beak_points], RetroColors.BEAK_ORANGE)
        
        # Animated wings with proper flapping
        if not self.falling:
            wing_offset = int(15 * math.sin(self.wing_flap))
            wing_y = int(self.y + wing_offset // 2)
            
            # Wing shape based on direction
            wing_size = 25
            if self.direction > 0:  # Flying right
                # Left wing (back wing)
                left_wing_points = np.array([
                    [self.x - body_width, wing_y],
                    [self.x - body_width - wing_size, wing_y - wing_offset],
                    [self.x - body_width - wing_size + 10, wing_y + 15 - wing_offset]
                ], np.int32)
                cv2.fillPoly(frame, [left_wing_points], RetroColors.DUCK_WING)
                
                # Right wing (front wing)
                right_wing_points = np.array([
                    [self.x + body_width, wing_y],
                    [self.x + body_width + wing_size, wing_y - wing_offset],
                    [self.x + body_width + wing_size - 10, wing_y + 15 - wing_offset]
                ], np.int32)
                cv2.fillPoly(frame, [right_wing_points], RetroColors.DUCK_WING)
            else:  # Flying left
                # Right wing (back wing)
                right_wing_points = np.array([
                    [self.x + body_width, wing_y],
                    [self.x + body_width + wing_size, wing_y - wing_offset],
                    [self.x + body_width + wing_size - 10, wing_y + 15 - wing_offset]
                ], np.int32)
                cv2.fillPoly(frame, [right_wing_points], RetroColors.DUCK_WING)
                
                # Left wing (front wing)
                left_wing_points = np.array([
                    [self.x - body_width, wing_y],
                    [self.x - body_width - wing_size, wing_y - wing_offset],
                    [self.x - body_width - wing_size + 10, wing_y + 15 - wing_offset]
                ], np.int32)
                cv2.fillPoly(frame, [left_wing_points], RetroColors.DUCK_WING)
        
        # Duck eye
        eye_offset_x = (10) * self.direction
        eye_x = head_x + eye_offset_x
        eye_y = head_y - 8
        cv2.circle(frame, (eye_x, eye_y), 4, RetroColors.BLACK, -1)
        cv2.circle(frame, (eye_x + self.direction, eye_y - 1), 1, RetroColors.WHITE, -1)  # Eye highlight
        
    def check_hit(self, shot_x, shot_y):
        """Enhanced hit detection"""
        if not self.alive or self.hit or self.falling:
            return False
            
        # More precise hit detection based on duck body parts
        head_x = self.x + (self.width // 3) * self.direction
        head_y = self.y - self.height // 4
        
        # Check hit on body
        body_hit = math.sqrt((self.x - shot_x)**2 + (self.y - shot_y)**2) < self.width // 2
        
        # Check hit on head (more points)
        head_hit = math.sqrt((head_x - shot_x)**2 + (head_y - shot_y)**2) < self.height // 3
        
        if body_hit or head_hit:
            self.hit = True
            self.falling = True
            self.hit_time = time.time()
            self.fall_speed = 0
            return True
            
        return False

class RetroDuckHuntGame:
    def __init__(self, screen_width=1280, screen_height=720):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize MediaPipe Hands with better settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Game state
        self.round = 1
        self.score = 0
        self.shots_fired = 0
        self.shots_per_round = 3
        self.ducks_shot = 0
        self.ducks_missed = 0
        self.ducks_per_round = 10
        
        # Duck management
        self.ducks = []
        self.max_ducks_on_screen = 2
        self.duck_spawn_timer = 0
        self.duck_spawn_interval = 3.0
        self.ducks_spawned_this_round = 0
        
        # Enhanced shooting mechanics
        self.can_shoot = True
        self.shot_cooldown_time = 0.3
        self.last_shot_time = 0
        self.thumb_trigger_active = False
        self.trigger_confirmation_time = 0.1  # Require stable gesture
        self.trigger_start_time = 0
        
        # Hand tracking with gesture stability
        self.hand_position = None
        self.gun_position = None
        self.aiming = False
        self.gesture_stable_time = 0
        self.last_gesture_time = 0
        
        # Visual effects
        self.shots = []
        self.hit_effects = []
        self.muzzle_flashes = []
        
        # Back button
        self.back_button = BackButton(screen_width, screen_height)
        
        # Game timing
        self.game_start_time = time.time()
        self.round_start_time = time.time()
        
    def detect_enhanced_gun_gesture(self, landmarks):
        """Simplified and more reliable gun gesture detection"""
        if len(landmarks.landmark) < 21:
            return False, None, False
            
        # Get finger landmarks
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        thumb_mcp = landmarks.landmark[2]
        
        index_tip = landmarks.landmark[8]
        index_pip = landmarks.landmark[6]
        index_mcp = landmarks.landmark[5]
        
        middle_tip = landmarks.landmark[12]
        middle_pip = landmarks.landmark[10]
        middle_mcp = landmarks.landmark[9]
        
        ring_tip = landmarks.landmark[16]
        ring_pip = landmarks.landmark[14]
        ring_mcp = landmarks.landmark[13]
        
        pinky_tip = landmarks.landmark[20]
        pinky_pip = landmarks.landmark[18]
        pinky_mcp = landmarks.landmark[17]
        
        # SIMPLIFIED GESTURE DETECTION
        
        # 1. Check if index finger is extended (pointing)
        index_extended = index_tip.y < index_pip.y < index_mcp.y
        
        # 2. Check if other fingers are closed (simplified)
        middle_closed = middle_tip.y > middle_pip.y
        ring_closed = ring_tip.y > ring_pip.y
        pinky_closed = pinky_tip.y > pinky_pip.y
        
        # 3. SIMPLIFIED THUMB TRIGGER - just check if thumb is curled
        # Thumb is "triggered" if it's curled (tip below IP joint)
        thumb_triggered = thumb_tip.y > thumb_ip.y
        
        # 4. Gun gesture is valid if index extended and other fingers closed
        gun_gesture_valid = index_extended and middle_closed and ring_closed and pinky_closed
        
        if gun_gesture_valid:
            # Calculate target position (index finger tip)
            target_x = int(index_tip.x * self.screen_width)
            target_y = int(index_tip.y * self.screen_height)
            return True, (target_x, target_y), thumb_triggered
            
        return False, None, False
    
    def draw_hand_debug(self, frame, landmarks):
        """Draw hand landmarks for debugging (optional)"""
        # Draw key points for gun gesture
        key_points = [4, 8, 12, 16, 20]  # Thumb tip, index tip, middle tip, ring tip, pinky tip
        
        for point_id in key_points:
            if point_id < len(landmarks.landmark):
                point = landmarks.landmark[point_id]
                x = int(point.x * self.screen_width)
                y = int(point.y * self.screen_height)
                
                # Color code the points
                if point_id == 4:  # Thumb
                    color = (0, 255, 255)  # Yellow
                elif point_id == 8:  # Index
                    color = (0, 255, 0)    # Green
                else:  # Other fingers
                    color = (255, 0, 0)    # Red
                
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.putText(frame, str(point_id), (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    def shoot(self, shot_x, shot_y):
        """Enhanced shooting with authentic mechanics"""
        current_time = time.time()
        
        # Check cooldown and shot limit
        if (current_time - self.last_shot_time < self.shot_cooldown_time or 
            self.shots_fired >= self.shots_per_round):
            return
            
        self.last_shot_time = current_time
        self.shots_fired += 1
        self.can_shoot = self.shots_fired < self.shots_per_round
        
        # Add authentic muzzle flash
        self.muzzle_flashes.append({
            'x': shot_x,
            'y': shot_y,
            'time': current_time,
            'duration': 0.15
        })
        
        # Add shot visual effect
        self.shots.append({
            'x': shot_x,
            'y': shot_y,
            'time': current_time,
            'duration': 0.4
        })
        
        # Check for hits with authentic scoring
        hit_any_duck = False
        for duck in self.ducks:
            if duck.check_hit(shot_x, shot_y):
                hit_any_duck = True
                
                # Authentic Duck Hunt scoring
                base_score = 500  # Base score for hitting duck
                
                # Bonus for quick shots
                duck_alive_time = current_time - duck.hit_time if hasattr(duck, 'spawn_time') else 1
                if duck_alive_time < 2:
                    base_score += 500  # Quick shot bonus
                    
                self.score += base_score
                self.ducks_shot += 1
                
                # Add hit effect
                self.hit_effects.append({
                    'x': duck.x,
                    'y': duck.y,
                    'time': current_time,
                    'duration': 1.5,
                    'score': base_score
                })
                break
                
        if not hit_any_duck:
            self.ducks_missed += 1
            
    def spawn_duck(self):
        """Spawn duck with round-based logic"""
        if (len(self.ducks) < self.max_ducks_on_screen and 
            self.ducks_spawned_this_round < self.ducks_per_round):
            
            duck = Duck(self.screen_width, self.screen_height)
            duck.spawn_time = time.time()
            self.ducks.append(duck)
            self.ducks_spawned_this_round += 1
            
    def update(self):
        """Update game with round-based progression"""
        current_time = time.time()
        
        # Spawn ducks
        if current_time - self.duck_spawn_timer > self.duck_spawn_interval:
            self.spawn_duck()
            self.duck_spawn_timer = current_time
            
        # Update ducks
        for duck in self.ducks:
            duck.update()
            
        # Remove off-screen ducks
        initial_count = len(self.ducks)
        self.ducks = [duck for duck in self.ducks if duck.alive or duck.falling]
        
        # Count escaped ducks as missed
        escaped_count = initial_count - len(self.ducks) - sum(1 for d in self.ducks if d.falling)
        
        # Update visual effects
        self.shots = [shot for shot in self.shots 
                     if current_time - shot['time'] < shot['duration']]
        
        self.hit_effects = [effect for effect in self.hit_effects 
                           if current_time - effect['time'] < effect['duration']]
        
        self.muzzle_flashes = [flash for flash in self.muzzle_flashes 
                              if current_time - flash['time'] < flash['duration']]
        
        # Check round completion
        if (self.ducks_spawned_this_round >= self.ducks_per_round and 
            len(self.ducks) == 0):
            self.advance_round()
            
    def advance_round(self):
        """Advance to next round"""
        self.round += 1
        self.shots_fired = 0
        self.ducks_spawned_this_round = 0
        self.can_shoot = True
        
        # Increase difficulty
        if self.round % 3 == 0:  # Every 3 rounds
            self.max_ducks_on_screen = min(4, self.max_ducks_on_screen + 1)
            self.duck_spawn_interval = max(1.5, self.duck_spawn_interval - 0.2)
            
    def draw_authentic_background(self, frame):
        """Draw authentic Duck Hunt background"""
        # Sky gradient (top to bottom)
        for y in range(self.screen_height - 150):
            intensity = 1.0 - (y / (self.screen_height - 150)) * 0.3
            color = tuple(int(c * intensity) for c in RetroColors.SKY_BLUE)
            cv2.line(frame, (0, y), (self.screen_width, y), color, 1)
            
        # Grass area
        grass_height = 150
        cv2.rectangle(frame, (0, self.screen_height - grass_height), 
                     (self.screen_width, self.screen_height), 
                     RetroColors.GRASS_GREEN, -1)
        
        # Add grass texture
        for i in range(0, self.screen_width, 40):
            for j in range(3):
                grass_x = i + random.randint(-10, 10)
                grass_y = self.screen_height - random.randint(10, 50)
                cv2.line(frame, (grass_x, grass_y), 
                        (grass_x + random.randint(-5, 5), grass_y - random.randint(10, 25)), 
                        (0, 140, 0), 2)
        
    def draw_target_cursor(self, frame):
        """Draw target cursor instead of gun"""
        if self.gun_position:
            target_x, target_y = self.gun_position
            
            # Target dimensions
            outer_radius = 25
            inner_radius = 15
            center_radius = 5
            
            # Draw outer ring (red)
            cv2.circle(frame, (target_x, target_y), outer_radius, RetroColors.BLACK, 3)
            cv2.circle(frame, (target_x, target_y), outer_radius, RetroColors.RED, 2)
            
            # Draw inner ring (white)
            cv2.circle(frame, (target_x, target_y), inner_radius, RetroColors.BLACK, 3)
            cv2.circle(frame, (target_x, target_y), inner_radius, RetroColors.WHITE, 2)
            
            # Draw center dot (red)
            cv2.circle(frame, (target_x, target_y), center_radius, RetroColors.BLACK, 2)
            cv2.circle(frame, (target_x, target_y), center_radius, RetroColors.RED, 1)
            
            # Draw crosshair lines
            line_length = 35
            thickness = 2
            
            # Horizontal line
            cv2.line(frame, (target_x - line_length, target_y), 
                    (target_x + line_length, target_y), RetroColors.BLACK, thickness + 1)
            cv2.line(frame, (target_x - line_length, target_y), 
                    (target_x + line_length, target_y), RetroColors.WHITE, thickness)
            
            # Vertical line
            cv2.line(frame, (target_x, target_y - line_length), 
                    (target_x, target_y + line_length), RetroColors.BLACK, thickness + 1)
            cv2.line(frame, (target_x, target_y - line_length), 
                    (target_x, target_y + line_length), RetroColors.WHITE, thickness)
            
            # Draw hand position indicator (small dot where your hand is)
            cv2.circle(frame, (target_x, target_y), 3, RetroColors.WHITE, -1)
            cv2.circle(frame, (target_x, target_y), 2, RetroColors.BLACK, 1)
            
            # Add status indicator
            if self.aiming:
                if self.thumb_trigger_active:
                    status_text = "FIRING!"
                    status_color = RetroColors.RED
                    # Add firing effect
                    cv2.circle(frame, (target_x, target_y), outer_radius + 10, RetroColors.YELLOW, 3)
                else:
                    status_text = "READY"
                    status_color = RetroColors.GREEN
            else:
                status_text = "AIM"
                status_color = RetroColors.WHITE
            
            # Draw status text
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = target_x - text_size[0] // 2
            text_y = target_y - outer_radius - 20
            cv2.putText(frame, status_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, RetroColors.BLACK, 3)
            cv2.putText(frame, status_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
    def draw_retro_ui(self, frame):
        """Draw authentic retro-style UI"""
        # Score display (top left)
        score_text = f"SCORE: {self.score:06d}"
        cv2.putText(frame, score_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, RetroColors.WHITE, 3)
        cv2.putText(frame, score_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, RetroColors.YELLOW, 2)
        
        # Round display
        round_text = f"ROUND {self.round}"
        cv2.putText(frame, round_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, RetroColors.WHITE, 2)
        
        # Shots remaining
        shots_remaining = self.shots_per_round - self.shots_fired
        shots_text = f"SHOTS: {shots_remaining}"
        cv2.putText(frame, shots_text, (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, RetroColors.WHITE, 2)
        
        # Hit ratio
        total_ducks = self.ducks_shot + self.ducks_missed
        if total_ducks > 0:
            hit_ratio = int((self.ducks_shot / total_ducks) * 100)
            ratio_text = f"HIT RATIO: {hit_ratio}%"
            cv2.putText(frame, ratio_text, (20, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, RetroColors.WHITE, 2)
        
        # Instructions
        if self.aiming:
            if self.thumb_trigger_active:
                instruction = "FIRING!"
                color = RetroColors.RED
            else:
                instruction = "CURL THUMB TO SHOOT"
                color = RetroColors.WHITE
        else:
            instruction = "POINT INDEX FINGER TO AIM"
            color = RetroColors.WHITE
            
        cv2.putText(frame, instruction, 
                   (self.screen_width // 2 - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    def draw(self, frame):
        """Main draw function"""
        # Draw authentic background
        self.draw_authentic_background(frame)
        
        # Draw ducks
        for duck in self.ducks:
            duck.draw(frame)
            
        # Draw visual effects
        current_time = time.time()
        
        # Muzzle flashes
        for flash in self.muzzle_flashes:
            alpha = 1.0 - (current_time - flash['time']) / flash['duration']
            if alpha > 0:
                flash_size = int(30 * alpha)
                cv2.circle(frame, (flash['x'], flash['y']), 
                          flash_size, RetroColors.YELLOW, -1)
                cv2.circle(frame, (flash['x'], flash['y']), 
                          flash_size + 5, RetroColors.RED, 3)
        
        # Shot effects
        for shot in self.shots:
            alpha = 1.0 - (current_time - shot['time']) / shot['duration']
            if alpha > 0:
                shot_size = int(15 * alpha)
                cv2.circle(frame, (shot['x'], shot['y']), 
                          shot_size, RetroColors.YELLOW, 2)
                
        # Hit effects with score display
        for effect in self.hit_effects:
            alpha = 1.0 - (current_time - effect['time']) / effect['duration']
            if alpha > 0:
                # Hit explosion
                effect_size = int(40 * alpha)
                cv2.circle(frame, (int(effect['x']), int(effect['y'])), 
                          effect_size, RetroColors.RED, 3)
                
                # Score popup
                if 'score' in effect:
                    score_y = int(effect['y'] - 30 - (1 - alpha) * 50)
                    cv2.putText(frame, f"+{effect['score']}", 
                               (int(effect['x'] - 30), score_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, RetroColors.YELLOW, 2)
        
        # Draw target cursor
        self.draw_target_cursor(frame)
        
        # Draw UI
        self.draw_retro_ui(frame)
        
        # Draw back button
        self.back_button.draw(frame, self.hand_position)

def main():
    parser = argparse.ArgumentParser(description='Retro Duck Hunt Game with Enhanced Hand Tracking')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    args = parser.parse_args()

    print("🎯 Initializing Retro Duck Hunt...")
    
    # Initialize camera with fallback
    cap = None
    for camera_index in range(3):  # Try cameras 0, 1, 2
        print(f"📹 Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            # Test if we can actually read frames
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera {camera_index} working")
                break
            else:
                print(f"❌ Camera {camera_index} opened but can't read frames")
                cap.release()
                cap = None
        else:
            print(f"❌ Camera {camera_index} failed to open")
    
    if not cap or not cap.isOpened():
        print("❌ No working camera found")
        print("💡 Try running with --camera 1 or --camera 2")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize game
    game = RetroDuckHuntGame(1280, 720)
    
    # Debug mode flag
    show_hand_debug = False
    
    # Create window and set to full screen
    cv2.namedWindow('Retro Duck Hunt', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Retro Duck Hunt', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow('Retro Duck Hunt', 1280, 720)
    cv2.moveWindow('Retro Duck Hunt', 0, 0)

    print("🦆 RETRO DUCK HUNT - ENHANCED EDITION")
    print("=" * 50)
    print("📋 Controls:")
    print("   🎯 Point index finger to aim")
    print("   👍 Keep other fingers closed")
    print("   🔫 CURL THUMB to shoot!")
    print("   🔙 Use back button to exit")
    print("   🖥️ Press 'F' to toggle full screen")
    print("   🎮 Press 'R' to restart game")
    print("   🐛 Press 'H' to toggle hand debug")
    print("   ❌ Press 'Q' or 'ESC' to quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1280, 720))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = game.hands.process(rgb_frame)

        # Reset hand tracking
        game.hand_position = None
        game.gun_position = None
        game.aiming = False
        game.thumb_trigger_active = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand position for back button
                hand_x = int(hand_landmarks.landmark[9].x * 1280)
                hand_y = int(hand_landmarks.landmark[9].y * 720)
                game.hand_position = (hand_x, hand_y)

                # Enhanced gun gesture detection
                is_gun, gun_pos, thumb_trigger = game.detect_enhanced_gun_gesture(hand_landmarks)
                
                if is_gun:
                    game.aiming = True
                    game.gun_position = gun_pos
                    game.thumb_trigger_active = thumb_trigger
                    
                    # Shoot only when thumb is in trigger position
                    if thumb_trigger and game.can_shoot:
                        game.shoot(gun_pos[0], gun_pos[1])
                        print(f"🔫 SHOT FIRED at ({gun_pos[0]}, {gun_pos[1]})")
                else:
                    # Debug: show when gesture is not detected
                    if show_hand_debug:
                        print("❌ Gun gesture not detected - check finger positions")

        # Handle back button
        key = cv2.waitKey(1) & 0xFF
        if game.back_button.handle_input(key, results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None, game.hand_position):
            print("Returning to app store...")
            break

                # Update and draw game
        game.update()
        game.draw(frame)
        
        # Draw hand debug if enabled
        if show_hand_debug and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                game.draw_hand_debug(frame, hand_landmarks)
        
        cv2.imshow('Retro Duck Hunt', frame)
        
        # Ensure full screen mode is maintained
        try:
            current_prop = cv2.getWindowProperty('Retro Duck Hunt', cv2.WND_PROP_FULLSCREEN)
            if current_prop != cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty('Retro Duck Hunt', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.moveWindow('Retro Duck Hunt', 0, 0)
        except:
            pass

        # Keyboard controls
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):  # Restart
            game = RetroDuckHuntGame(1280, 720)
        elif key == ord('h'):  # Toggle hand debug
            show_hand_debug = not show_hand_debug
            print(f"🐛 Hand debug: {'ON' if show_hand_debug else 'OFF'}")
        elif key == ord('f'):  # Toggle full screen
            try:
                current_prop = cv2.getWindowProperty('Retro Duck Hunt', cv2.WND_PROP_FULLSCREEN)
                if current_prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty('Retro Duck Hunt', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty('Retro Duck Hunt', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.moveWindow('Retro Duck Hunt', 0, 0)
            except:
                pass

    cap.release()
    cv2.destroyAllWindows()
    print("🎮 Thanks for playing Retro Duck Hunt!")

if __name__ == "__main__":
    main()
