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
    
    # New duck colors for different types
    DUCK_GOLDEN = (0, 215, 255)      # Golden duck (highest score)
    DUCK_SILVER = (192, 192, 192)    # Silver duck (high score)
    DUCK_BLUE = (255, 0, 0)          # Blue duck (medium score)
    DUCK_GREEN = (0, 255, 0)         # Green duck (medium score)
    DUCK_RED = (0, 0, 255)           # Red duck (low score)
    DUCK_PURPLE = (128, 0, 128)      # Purple duck (bonus)

class Duck:
    def __init__(self, screen_width, screen_height, duck_type="normal", level=1):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.duck_type = duck_type
        self.level = level
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
        
        # Duck properties based on type
        self.set_duck_properties()
        
        # Level-based speed scaling
        speed_multiplier = 1.0 + (self.level - 1) * 0.3  # 1.0, 1.3, 1.6, 1.9...
        
        # Apply speed multiplier
        self.speed_x *= speed_multiplier
        self.speed_y *= speed_multiplier
        
        # Animation
        self.wing_flap = 0
        self.wing_speed = 0.3 * speed_multiplier
        self.direction = 1 if self.speed_x > 0 else -1
        
        # State
        self.alive = True
        self.hit = False
        self.falling = False
        self.fall_speed = 0
        self.hit_time = 0
        
    def set_duck_properties(self):
        """Set duck properties based on type"""
        if self.duck_type == "golden":
            self.width = 90
            self.height = 70
            self.speed_multiplier = 0.8
            self.score_value = 1500
            self.body_color = RetroColors.DUCK_GOLDEN
            self.head_color = RetroColors.DUCK_GOLDEN
        elif self.duck_type == "silver":
            self.width = 85
            self.height = 65
            self.speed_multiplier = 0.9
            self.score_value = 1000
            self.body_color = RetroColors.DUCK_SILVER
            self.head_color = RetroColors.DUCK_SILVER
        elif self.duck_type == "blue":
            self.width = 80
            self.height = 60
            self.speed_multiplier = 1.0
            self.score_value = 800
            self.body_color = RetroColors.DUCK_BLUE
            self.head_color = RetroColors.DUCK_BLUE
        elif self.duck_type == "green":
            self.width = 80
            self.height = 60
            self.speed_multiplier = 1.0
            self.score_value = 600
            self.body_color = RetroColors.DUCK_GREEN
            self.head_color = RetroColors.DUCK_GREEN
        elif self.duck_type == "red":
            self.width = 75
            self.height = 55
            self.speed_multiplier = 1.2
            self.score_value = 400
            self.body_color = RetroColors.DUCK_RED
            self.head_color = RetroColors.DUCK_RED
        elif self.duck_type == "purple":
            self.width = 70
            self.height = 50
            self.speed_multiplier = 1.5
            self.score_value = 2000  # Bonus duck!
            self.body_color = RetroColors.DUCK_PURPLE
            self.head_color = RetroColors.DUCK_PURPLE
        else:  # normal
            self.width = 80
            self.height = 60
            self.speed_multiplier = 1.0
            self.score_value = 500
            self.body_color = RetroColors.DUCK_BROWN
            self.head_color = RetroColors.DUCK_HEAD
        
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
        
        # Use the duck's assigned color
        body_color = self.body_color
        
        cv2.ellipse(frame, 
                   (int(self.x), int(self.y)), 
                   (body_width, body_height), 
                   0, 0, 360, body_color, -1)
        
        # Duck head (circular, positioned based on direction)
        head_offset_x = (self.width // 3) * self.direction
        head_x = int(self.x + head_offset_x)
        head_y = int(self.y - self.height // 4)
        head_radius = self.height // 3
        
        cv2.circle(frame, (head_x, head_y), head_radius, self.head_color, -1)
        
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
        
        # Game state
        self.round = 1
        self.level = 1
        self.score = 0
        self.shots_fired = 0
        self.ducks_shot = 0
        self.ducks_missed = 0
        self.ducks_spawned_this_round = 0
        self.ducks_escaped_this_round = 0
        self.total_ducks_this_round = 0
        self.game_completed = False
        self.game_won = False
        self.round_complete = False
        self.can_shoot = True
        self.last_shot_time = 0
        self.last_gesture_time = 0  # Add missing variable
        self.duck_spawn_timer = time.time()
        self.round_start_time = time.time()
        self.shot_cooldown_time = 0.3  # Cooldown between shots
        
        # Win condition
        self.win_threshold = 0.5  # 50% accuracy required
        
        # Duck management
        self.ducks = []
        self.duck_spawn_interval = 1.0
        self.ducks_per_round = 15
        self.max_ducks_on_screen = 4
        self.shots_per_round = 20
        
        # Visual effects
        self.shots = []
        self.hit_effects = []
        self.muzzle_flashes = []
        
        # Hand tracking
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Cursor stabilization
        self.smoothed_cursor_position = None
        self.last_cursor_position = None
        self.index_finger_detected = False
        self.last_index_position = None
        self.index_stability_threshold = 0.02
        self.cursor_stability_threshold = 5  # Pixels - movement below this is ignored
        self.cursor_smoothing_factor = 0.3  # Lower = smoother cursor
        
        # Aim stabilization
        self.aim_lock_active = False
        self.aim_lock_position = None
        self.aim_lock_start_time = 0
        self.aim_lock_duration = 0.5  # Lock aim for 0.5 seconds during shooting
        self.shooting_gesture_detected = False
        
        # Quarter-circle gesture detection
        self.thumb_positions = []
        self.max_thumb_history = 10
        
        # Background objects
        self.clouds = []
        self.trees = []
        self.bushes = []
        self.grass_animation_time = 0
        self.grass_animation_speed = 0.3  # Much slower grass animation
        
        # UI elements
        self.hand_position = None
        self.gun_position = None
        self.aiming = False
        self.thumb_trigger_active = False
        
        # Back button
        self.back_button = BackButton()
        
        # Initialize everything
        self.update_level_settings()
        self.initialize_background_objects()
        
    def update_level_settings(self):
        """Update game settings based on current level"""
        # MUCH MORE AGGRESSIVE SPAWNING - 5x more ducks!
        self.duck_spawn_interval = max(0.2, 1.0 - (self.level - 1) * 0.1)  # Much faster spawning (0.2-1.0 seconds)
        self.ducks_per_round = 15 + (self.level - 1) * 5  # Many more ducks per round (15, 20, 25, 30...)
        self.max_ducks_on_screen = min(12, 4 + (self.level - 1) * 2)  # More ducks on screen (4, 6, 8, 10, 12)
        self.shots_per_round = 20 + (self.level - 1) * 5  # More shots per round (20, 25, 30, 35...)
        
        print(f"🎯 LEVEL {self.level} - Ducks: {self.ducks_per_round}, Max on screen: {self.max_ducks_on_screen}, Shots: {self.shots_per_round}")
        
    def detect_enhanced_gun_gesture(self, landmarks):
        """Improved gun gesture detection with better index finger tracking and quarter-circle shooting"""
        if len(landmarks.landmark) < 21:
            return False, None, False
            
        # Get finger landmarks
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        thumb_mcp = landmarks.landmark[2]
        
        index_tip = landmarks.landmark[8]
        index_pip = landmarks.landmark[6]
        index_mcp = landmarks.landmark[5]
        index_dip = landmarks.landmark[7]
        
        middle_tip = landmarks.landmark[12]
        middle_pip = landmarks.landmark[10]
        middle_mcp = landmarks.landmark[9]
        
        ring_tip = landmarks.landmark[16]
        ring_pip = landmarks.landmark[14]
        ring_mcp = landmarks.landmark[13]
        
        pinky_tip = landmarks.landmark[20]
        pinky_pip = landmarks.landmark[18]
        pinky_mcp = landmarks.landmark[17]
        
        # IMPROVED INDEX FINGER DETECTION with multiple fallback methods
        index_extended = False
        
        # Method 1: Standard extension check
        if index_tip.y < index_pip.y < index_mcp.y:
            index_extended = True
        
        # Method 2: More lenient check with tolerance
        elif (index_tip.y < index_pip.y + self.index_stability_threshold and 
              index_pip.y < index_mcp.y + self.index_stability_threshold):
            index_extended = True
        
        # Method 3: Check if index is significantly more extended than other fingers
        elif (index_tip.y < middle_tip.y - 0.05 and 
              index_tip.y < ring_tip.y - 0.05 and 
              index_tip.y < pinky_tip.y - 0.05):
            index_extended = True
        
        # Method 4: Check if index is pointing forward (for side view)
        index_angle = math.atan2(index_tip.y - index_mcp.y, index_tip.x - index_mcp.x)
        if abs(index_angle) < 0.5:  # Roughly horizontal or pointing up
            index_extended = True
        
        # IMPROVED OTHER FINGER DETECTION (more lenient)
        middle_closed = middle_tip.y > middle_pip.y - 0.02
        ring_closed = ring_tip.y > ring_pip.y - 0.02
        pinky_closed = pinky_tip.y > pinky_pip.y - 0.02
        
        # QUARTER-CIRCLE SHOOTING GESTURE DETECTION
        current_time = time.time()
        current_thumb_pos = (thumb_tip.x, thumb_tip.y)
        
        # Store thumb position history
        self.thumb_positions.append({
            'pos': current_thumb_pos,
            'time': current_time
        })
        
        # Keep only recent positions
        if len(self.thumb_positions) > self.max_thumb_history:
            self.thumb_positions.pop(0)
        
        # Detect quarter-circle gesture
        shooting_gesture = self.detect_quarter_circle_gesture()
        
        # Gun gesture is valid if index extended and other fingers mostly closed
        gun_gesture_valid = index_extended and (middle_closed or ring_closed or pinky_closed)
        
        if gun_gesture_valid:
            # Calculate target position (index finger tip)
            target_x = int(index_tip.x * self.screen_width)
            target_y = int(index_tip.y * self.screen_height)
            
            # Apply cursor stabilization
            stabilized_position = self.stabilize_cursor_position((target_x, target_y))
            
            # Apply aim stabilization during shooting gesture
            final_position = self.handle_aim_stabilization(shooting_gesture, stabilized_position)
            
            # Update index finger tracking
            self.index_finger_detected = True
            self.last_index_position = final_position
            
            return True, final_position, shooting_gesture
            
        # Fallback: if we had a valid position recently, use it
        elif self.last_index_position and self.index_finger_detected:
            # Check if hand is still in view but gesture is unclear
            hand_center_x = int(landmarks.landmark[9].x * self.screen_width)
            hand_center_y = int(landmarks.landmark[9].y * self.screen_height)
            
            # If hand is still in reasonable position, maintain cursor
            if (0 < hand_center_x < self.screen_width and 
                0 < hand_center_y < self.screen_height):
                return True, self.last_index_position, False
        
        # Reset index finger detection if gesture is lost
        self.index_finger_detected = False
        return False, None, False
    
    def check_game_completion(self):
        """Check if the game is complete and determine win/lose condition"""
        if self.game_completed:
            return
            
        # Check if all ducks are gone (either shot or escaped) and we've spawned all ducks for this round
        if (len(self.ducks) == 0 and 
            self.ducks_spawned_this_round >= self.ducks_per_round and
            self.ducks_spawned_this_round > 0):  # Make sure we actually spawned ducks
            
            # Calculate win condition: must shoot at least 50% of ducks
            total_ducks = self.ducks_shot + self.ducks_escaped_this_round
            if total_ducks > 0:
                hit_percentage = self.ducks_shot / total_ducks
                
                if hit_percentage >= self.win_threshold:
                    self.game_won = True
                    print(f"🎉 YOU WIN! Hit {self.ducks_shot}/{total_ducks} ducks ({hit_percentage:.1%})")
                else:
                    self.game_won = False
                    print(f"💀 GAME OVER! Hit {self.ducks_shot}/{total_ducks} ducks ({hit_percentage:.1%}) - Need 50% to win")
                
                self.game_completed = True
                self.round_complete = True
                print(f"🏁 Game completed! Final score: {self.score}")
    
    def reset_game(self):
        """Reset the game for a new round"""
        self.round = 1
        self.level = 1
        self.score = 0
        self.shots_fired = 0
        self.ducks_shot = 0
        self.ducks_missed = 0
        self.ducks_spawned_this_round = 0
        self.ducks_escaped_this_round = 0
        self.total_ducks_this_round = 0
        self.game_completed = False
        self.game_won = False
        self.round_complete = False
        self.can_shoot = True
        self.last_shot_time = 0
        self.duck_spawn_timer = time.time()
        self.round_start_time = time.time()
        
        # Clear all ducks and effects
        self.ducks = []
        self.shots = []
        self.hit_effects = []
        self.muzzle_flashes = []
        
        # Reset cursor
        self.smoothed_cursor_position = None
        self.last_cursor_position = None
        self.index_finger_detected = False
        self.last_index_position = None
        
        # Reset aim stabilization
        self.aim_lock_active = False
        self.aim_lock_position = None
        self.aim_lock_start_time = 0
        self.shooting_gesture_detected = False
        
        # Clear thumb positions
        self.thumb_positions.clear()
        
        # Reinitialize background objects
        self.clouds = []
        self.trees = []
        self.bushes = []
        self.initialize_background_objects()
        
        # Update level settings
        self.update_level_settings()
        
        print("🔄 Game reset - starting new round!")
    
    def initialize_background_objects(self):
        """Initialize background objects for visual interest"""
        # Create clouds
        for i in range(5):
            cloud = {
                'x': random.randint(0, self.screen_width),
                'y': random.randint(50, 200),
                'size': random.randint(30, 80),
                'speed': random.uniform(0.5, 1.5),
                'opacity': random.uniform(0.3, 0.7)
            }
            self.clouds.append(cloud)
        
        # Create trees
        for i in range(8):
            tree = {
                'x': random.randint(50, self.screen_width - 50),
                'y': self.screen_height - 100,
                'height': random.randint(80, 150),
                'width': random.randint(40, 80),
                'type': random.choice(['pine', 'oak', 'maple'])
            }
            self.trees.append(tree)
        
        # Create bushes
        for i in range(12):
            bush = {
                'x': random.randint(20, self.screen_width - 20),
                'y': self.screen_height - 50,
                'size': random.randint(20, 40),
                'type': random.choice(['round', 'oval', 'spiky'])
            }
            self.bushes.append(bush)
    
    def update_background_objects(self):
        """Update background objects animation"""
        current_time = time.time()
        
        # Update grass animation time (much slower)
        self.grass_animation_time = current_time * self.grass_animation_speed
        
        # Update clouds
        for cloud in self.clouds:
            cloud['x'] += cloud['speed']
            if cloud['x'] > self.screen_width + 100:
                cloud['x'] = -100
                cloud['y'] = random.randint(50, 200)
    
    def handle_button_interactions(self, shooting_gesture, cursor_position):
        """Handle button interactions using shooting gesture"""
        if not shooting_gesture or not cursor_position:
            return False
            
        # Check replay button (center of screen)
        replay_button_center = (self.screen_width // 2, self.screen_height // 2 + 100)
        replay_button_size = 200
        
        # Check if cursor is over replay button
        if (abs(cursor_position[0] - replay_button_center[0]) < replay_button_size // 2 and
            abs(cursor_position[1] - replay_button_center[1]) < replay_button_size // 2):
            
            if self.game_completed:
                print("🔄 Replay button clicked!")
                self.reset_game()
                return True
        
        # Check back button (top-left corner)
        back_button_center = (100, 100)
        back_button_size = 150
        
        if (abs(cursor_position[0] - back_button_center[0]) < back_button_size // 2 and
            abs(cursor_position[1] - back_button_center[1]) < back_button_size // 2):
            
            print("🔙 Back button clicked!")
            return "back"
        
        return False
    
    def handle_aim_stabilization(self, shooting_gesture, cursor_position):
        """Lock aim position during shooting gesture to prevent unwanted movement"""
        current_time = time.time()
        
        if shooting_gesture and not self.shooting_gesture_detected:
            # Shooting gesture just started - lock the aim
            self.aim_lock_active = True
            self.aim_lock_position = cursor_position
            self.aim_lock_start_time = current_time
            self.shooting_gesture_detected = True
            print("🎯 Aim locked during shooting gesture")
        
        elif self.aim_lock_active:
            # Check if aim lock should expire
            if current_time - self.aim_lock_start_time > self.aim_lock_duration:
                self.aim_lock_active = False
                self.aim_lock_position = None
                self.shooting_gesture_detected = False
                print("🎯 Aim lock released")
            else:
                # Return locked position instead of current cursor position
                return self.aim_lock_position
        
        elif not shooting_gesture:
            # Reset shooting gesture detection when not shooting
            self.shooting_gesture_detected = False
        
        return cursor_position
    
    def stabilize_cursor_position(self, new_position):
        """Stabilize cursor position to reduce shaking when hand is stationary"""
        if new_position is None:
            return None
            
        current_time = time.time()
        
        # Initialize smoothed position if first time
        if self.smoothed_cursor_position is None:
            self.smoothed_cursor_position = new_position
            self.last_cursor_position = new_position
            return new_position
        
        # Calculate distance from last position
        if self.last_cursor_position:
            distance = math.sqrt((new_position[0] - self.last_cursor_position[0])**2 + 
                               (new_position[1] - self.last_cursor_position[1])**2)
        else:
            distance = 0
        
        # If movement is very small, maintain stability
        if distance < self.cursor_stability_threshold:
            # Keep cursor stable for small movements
            self.cursor_stable_time = current_time
            return self.smoothed_cursor_position
        
        # Apply smoothing for larger movements
        smoothed_x = (self.smoothed_cursor_position[0] * (1 - self.cursor_smoothing_factor) + 
                     new_position[0] * self.cursor_smoothing_factor)
        smoothed_y = (self.smoothed_cursor_position[1] * (1 - self.cursor_smoothing_factor) + 
                     new_position[1] * self.cursor_smoothing_factor)
        
        self.smoothed_cursor_position = (int(smoothed_x), int(smoothed_y))
        self.last_cursor_position = new_position
        
        return self.smoothed_cursor_position
    
    def detect_quarter_circle_gesture(self):
        """Detect thumb moving in quarter-circle from left to top (much more forgiving)"""
        if len(self.thumb_positions) < 3:  # Reduced requirement for easier detection
            return False
        
        current_time = time.time()
        
        # Get recent thumb positions
        recent_positions = self.thumb_positions[-3:]  # Use fewer positions for easier detection
        
        # Check if we have enough movement
        start_pos = recent_positions[0]['pos']
        end_pos = recent_positions[-1]['pos']
        
        # Calculate movement distance
        movement_distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Much more forgiving movement requirement (reduced from 0.1 to 0.05)
        if movement_distance < 0.05:
            return False
        
        # Check if movement is roughly quarter-circle (left to top)
        # Start should be more to the left (lower x) and lower (higher y)
        # End should be more to the right (higher x) and higher (lower y)
        x_movement = end_pos[0] - start_pos[0]  # Should be positive (rightward)
        y_movement = end_pos[1] - start_pos[1]  # Should be negative (upward)
        
        # Much more forgiving direction check (reduced requirements)
        if x_movement < 0.02 or y_movement > -0.02:  # Very minimal movement required
            return False
        
        # Check timing - much more forgiving (0.05 to 2.0 seconds)
        gesture_duration = recent_positions[-1]['time'] - recent_positions[0]['time']
        if 0.05 < gesture_duration < 2.0:  # Much wider time window
            return True
        
        return False
    
    def draw_game_completion_screen(self, frame):
        """Draw game completion screen with win/lose message and replay button"""
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Game result message
        if self.game_won:
            result_text = "🎉 YOU WIN! 🎉"
            result_color = RetroColors.GREEN
            subtitle = f"Hit {self.ducks_shot}/{self.total_ducks_this_round} ducks ({self.ducks_shot/self.total_ducks_this_round:.1%})"
        else:
            result_text = "💀 GAME OVER 💀"
            result_color = RetroColors.RED
            subtitle = f"Hit {self.ducks_shot}/{self.total_ducks_this_round} ducks ({self.ducks_shot/self.total_ducks_this_round:.1%}) - Need 50% to win"
        
        # Draw main result text
        text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        text_y = self.screen_height // 2 - 50
        
        cv2.putText(frame, result_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, result_color, 3)
        
        # Draw subtitle
        subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        subtitle_x = (self.screen_width - subtitle_size[0]) // 2
        subtitle_y = text_y + 60
        
        cv2.putText(frame, subtitle, (subtitle_x, subtitle_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, RetroColors.WHITE, 2)
        
        # Draw final score
        score_text = f"Final Score: {self.score}"
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        score_x = (self.screen_width - score_size[0]) // 2
        score_y = subtitle_y + 60
        
        cv2.putText(frame, score_text, (score_x, score_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, RetroColors.YELLOW, 2)
        
        # Draw replay button
        replay_button_center = (self.screen_width // 2, self.screen_height // 2 + 100)
        replay_button_size = 200
        
        # Button background
        cv2.rectangle(frame, 
                     (replay_button_center[0] - replay_button_size // 2, 
                      replay_button_center[1] - replay_button_size // 2),
                     (replay_button_center[0] + replay_button_size // 2, 
                      replay_button_center[1] + replay_button_size // 2),
                     RetroColors.GREEN, -1)
        
        # Button border
        cv2.rectangle(frame, 
                     (replay_button_center[0] - replay_button_size // 2, 
                      replay_button_center[1] - replay_button_size // 2),
                     (replay_button_center[0] + replay_button_size // 2, 
                      replay_button_center[1] + replay_button_size // 2),
                     RetroColors.WHITE, 3)
        
        # Button text
        button_text = "REPLAY"
        button_text_size = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        button_text_x = replay_button_center[0] - button_text_size[0] // 2
        button_text_y = replay_button_center[1] + button_text_size[1] // 2
        
        cv2.putText(frame, button_text, (button_text_x, button_text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, RetroColors.BLACK, 2)
        
        # Instructions
        instruction_text = "Point and shoot at REPLAY button to restart"
        instruction_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
        instruction_x = (self.screen_width - instruction_size[0]) // 2
        instruction_y = replay_button_center[1] + replay_button_size // 2 + 40
        
        cv2.putText(frame, instruction_text, (instruction_x, instruction_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, RetroColors.WHITE, 1)
    
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
                
                # Enhanced scoring based on duck type and level
                base_score = duck.score_value  # Use the score_value from the duck object
                
                # Level bonus (higher levels = more points)
                level_bonus = (self.level - 1) * 100
                base_score += level_bonus
                
                # Bonus for quick shots
                duck_alive_time = current_time - duck.hit_time if hasattr(duck, 'spawn_time') else 1
                if duck_alive_time < 2:
                    base_score += 500  # Quick shot bonus
                    
                self.score += base_score
                self.ducks_shot += 1
                
                # Add hit effect with special effects for different duck types
                effect_duration = 2.0
                effect_color = RetroColors.RED
                
                if duck.duck_type == "purple":
                    effect_duration = 3.0  # Longer effect for bonus ducks
                    effect_color = RetroColors.DUCK_PURPLE
                elif duck.duck_type == "golden":
                    effect_duration = 2.5
                    effect_color = RetroColors.DUCK_GOLDEN
                elif duck.duck_type == "silver":
                    effect_duration = 2.2
                    effect_color = RetroColors.DUCK_SILVER
                elif duck.duck_type == "blue":
                    effect_color = RetroColors.DUCK_BLUE
                elif duck.duck_type == "green":
                    effect_color = RetroColors.DUCK_GREEN
                elif duck.duck_type == "red":
                    effect_color = RetroColors.DUCK_RED
                
                self.hit_effects.append({
                    'x': duck.x,
                    'y': duck.y,
                    'time': current_time,
                    'duration': effect_duration,
                    'score': base_score,
                    'duck_type': duck.duck_type,
                    'effect_color': effect_color
                })
                break
                
        if not hit_any_duck:
            self.ducks_missed += 1
            
    def spawn_duck(self):
        """Spawn duck with level-based logic and different types"""
        if (len(self.ducks) < self.max_ducks_on_screen and 
            self.ducks_spawned_this_round < self.ducks_per_round):
            
            # Determine duck type based on level and random chance
            duck_type = "normal"
            rand_val = random.random()
            
            if self.level >= 5:
                # Purple ducks (rare bonus ducks)
                if rand_val < 0.05:  # 5% chance
                    duck_type = "purple"
                # Golden ducks
                elif rand_val < 0.15:  # 10% chance
                    duck_type = "golden"
                # Silver ducks
                elif rand_val < 0.30:  # 15% chance
                    duck_type = "silver"
                # Blue ducks
                elif rand_val < 0.50:  # 20% chance
                    duck_type = "blue"
                # Green ducks
                elif rand_val < 0.70:  # 20% chance
                    duck_type = "green"
                # Red ducks
                elif rand_val < 0.85:  # 15% chance
                    duck_type = "red"
                # Normal ducks
                else:  # 15% chance
                    duck_type = "normal"
            elif self.level >= 3:
                # Golden ducks
                if rand_val < 0.10:  # 10% chance
                    duck_type = "golden"
                # Silver ducks
                elif rand_val < 0.25:  # 15% chance
                    duck_type = "silver"
                # Blue ducks
                elif rand_val < 0.45:  # 20% chance
                    duck_type = "blue"
                # Green ducks
                elif rand_val < 0.65:  # 20% chance
                    duck_type = "green"
                # Red ducks
                elif rand_val < 0.80:  # 15% chance
                    duck_type = "red"
                # Normal ducks
                else:  # 20% chance
                    duck_type = "normal"
            elif self.level >= 2:
                # Blue ducks
                if rand_val < 0.20:  # 20% chance
                    duck_type = "blue"
                # Green ducks
                elif rand_val < 0.40:  # 20% chance
                    duck_type = "green"
                # Red ducks
                elif rand_val < 0.60:  # 20% chance
                    duck_type = "red"
                # Normal ducks
                else:  # 40% chance
                    duck_type = "normal"
            else:
                # Level 1: Include colored ducks from the start!
                if rand_val < 0.05:  # 5% chance for purple (rare)
                    duck_type = "purple"
                elif rand_val < 0.15:  # 10% chance for golden
                    duck_type = "golden"
                elif rand_val < 0.25:  # 10% chance for silver
                    duck_type = "silver"
                elif rand_val < 0.40:  # 15% chance for blue
                    duck_type = "blue"
                elif rand_val < 0.55:  # 15% chance for green
                    duck_type = "green"
                elif rand_val < 0.70:  # 15% chance for red
                    duck_type = "red"
                else:  # 30% chance for normal
                    duck_type = "normal"
            
            duck = Duck(self.screen_width, self.screen_height, duck_type, self.level)
            duck.spawn_time = time.time()
            self.ducks.append(duck)
            self.ducks_spawned_this_round += 1
            
            # Debug info for special ducks
            if duck_type != "normal":
                print(f"🦆 Spawned {duck_type.upper()} duck! (Score: {duck.score_value})")
            
    def update(self):
        """Update game with round-based progression"""
        current_time = time.time()
        
        # Check for game completion
        self.check_game_completion()
        
        # If game is completed, don't update further
        if self.game_completed:
            return
        
        # Spawn ducks
        if current_time - self.duck_spawn_timer > self.duck_spawn_interval:
            self.spawn_duck()
            self.duck_spawn_timer = current_time
            
        # Update ducks
        for duck in self.ducks:
            duck.update()
            
        # Remove off-screen ducks and count escaped
        initial_count = len(self.ducks)
        self.ducks = [duck for duck in self.ducks if duck.alive or duck.falling]
        
        # Count escaped ducks
        escaped_count = initial_count - len(self.ducks) - sum(1 for d in self.ducks if d.falling)
        self.ducks_escaped_this_round += escaped_count
        
        # Update total ducks count
        self.total_ducks_this_round = self.ducks_shot + self.ducks_escaped_this_round
        
        # Debug output for game completion tracking
        if self.ducks_spawned_this_round >= self.ducks_per_round and len(self.ducks) == 0:
            print(f"🔍 Game completion check: Spawned {self.ducks_spawned_this_round}/{self.ducks_per_round}, "
                  f"Shot {self.ducks_shot}, Escaped {self.ducks_escaped_this_round}, "
                  f"Ducks on screen: {len(self.ducks)}")
        
        # Update background objects
        self.update_background_objects()
        
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
        """Advance to next round with level progression"""
        self.round += 1
        self.shots_fired = 0
        self.ducks_spawned_this_round = 0
        self.ducks_escaped_this_round = 0
        self.can_shoot = True
        
        # Update level settings
        self.update_level_settings()
        
        # Show level up message
        if self.round % 3 == 1:  # Start of new level
            print(f"🎉 LEVEL UP! Now playing Level {self.level}")
            print(f"📊 New settings: {self.ducks_per_round} ducks, {self.max_ducks_on_screen} max on screen, {self.shots_per_round} shots")
            
    def draw_authentic_background(self, frame):
        """Draw authentic Duck Hunt background with enhanced objects"""
        # Sky gradient (top to bottom)
        for y in range(self.screen_height - 150):
            intensity = 1.0 - (y / (self.screen_height - 150)) * 0.3
            color = tuple(int(c * intensity) for c in RetroColors.SKY_BLUE)
            cv2.line(frame, (0, y), (self.screen_width, y), color, 1)
        
        # Draw clouds (behind everything else)
        for cloud in self.clouds:
            cloud_color = tuple(int(c * cloud['opacity']) for c in RetroColors.WHITE)
            cv2.circle(frame, (int(cloud['x']), int(cloud['y'])), cloud['size'], cloud_color, -1)
            cv2.circle(frame, (int(cloud['x'] - cloud['size']//2), int(cloud['y'])), cloud['size']//2, cloud_color, -1)
            cv2.circle(frame, (int(cloud['x'] + cloud['size']//2), int(cloud['y'])), cloud['size']//2, cloud_color, -1)
        
        # Grass area
        grass_height = 150
        cv2.rectangle(frame, (0, self.screen_height - grass_height), 
                     (self.screen_width, self.screen_height), 
                     RetroColors.GRASS_GREEN, -1)
        
        # Draw trees (behind grass)
        for tree in self.trees:
            # Tree trunk
            trunk_color = (50, 25, 0)  # Brown trunk
            cv2.rectangle(frame, 
                         (int(tree['x'] - tree['width']//4), int(tree['y'])),
                         (int(tree['x'] + tree['width']//4), int(tree['y'] + tree['height'])),
                         trunk_color, -1)
            
            # Tree foliage
            if tree['type'] == 'pine':
                # Triangular pine tree
                foliage_color = (0, 100, 0)  # Dark green
                points = np.array([
                    [tree['x'], tree['y'] - tree['height']],
                    [tree['x'] - tree['width']//2, tree['y']],
                    [tree['x'] + tree['width']//2, tree['y']]
                ], np.int32)
                cv2.fillPoly(frame, [points], foliage_color)
            else:
                # Round tree top
                foliage_color = (0, 120, 0)  # Medium green
                cv2.circle(frame, (int(tree['x']), int(tree['y'] - tree['height']//3)), 
                          tree['width']//2, foliage_color, -1)
        
        # Draw bushes
        for bush in self.bushes:
            bush_color = (0, 140, 0)  # Bush green
            if bush['type'] == 'round':
                cv2.circle(frame, (int(bush['x']), int(bush['y'])), bush['size'], bush_color, -1)
            elif bush['type'] == 'oval':
                cv2.ellipse(frame, (int(bush['x']), int(bush['y'])), 
                           (bush['size'], bush['size']//2), 0, 0, 360, bush_color, -1)
            else:  # spiky
                points = np.array([
                    [bush['x'], bush['y'] - bush['size']],
                    [bush['x'] - bush['size']//2, bush['y']],
                    [bush['x'] + bush['size']//2, bush['y']]
                ], np.int32)
                cv2.fillPoly(frame, [points], bush_color)
        
        # Add grass texture (MUCH SLOWER ANIMATION)
        grass_seed = int(self.grass_animation_time) % 1000  # Slower animation
        random.seed(grass_seed)
        
        for i in range(0, self.screen_width, 30):  # Less frequent grass blades
            for j in range(2):  # Fewer grass blades per position
                grass_x = i + random.randint(-15, 15)
                grass_y = self.screen_height - random.randint(5, 30)  # Shorter grass
                grass_height = random.randint(8, 15)  # Shorter grass blades
                
                # Draw grass blade
                cv2.line(frame, (grass_x, grass_y), 
                        (grass_x + random.randint(-3, 3), grass_y - grass_height), 
                        (0, 140, 0), 1)  # Thinner grass blades
        
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
                elif self.aim_lock_active:
                    status_text = "AIM LOCKED"
                    status_color = RetroColors.GREEN
                    # Add aim lock indicator
                    cv2.circle(frame, (target_x, target_y), outer_radius + 8, RetroColors.GREEN, 2)
                else:
                    status_text = "READY"
                    status_color = RetroColors.GREEN
                    
                    # Show shooting gesture hint
                    if len(self.thumb_positions) > 2:
                        # Draw quarter-circle hint (left to top)
                        hint_radius = 40
                        hint_center_x = target_x + hint_radius
                        hint_center_y = target_y + hint_radius
                        
                        # Draw quarter-circle arc (left to top)
                        cv2.ellipse(frame, (hint_center_x, hint_center_y), 
                                  (hint_radius, hint_radius), 0, 180, 270, 
                                  RetroColors.WHITE, 2)
                        
                        # Draw arrow indicating direction
                        arrow_start = (hint_center_x - hint_radius + 10, hint_center_y)
                        arrow_end = (hint_center_x - hint_radius + 25, hint_center_y - 15)
                        cv2.arrowedLine(frame, arrow_start, arrow_end, RetroColors.WHITE, 2)
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
        # Score display (top right)
        score_text = f"SCORE: {self.score:06d}"
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        score_x = self.screen_width - score_size[0] - 20
        cv2.putText(frame, score_text, (score_x, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, RetroColors.WHITE, 3)
        cv2.putText(frame, score_text, (score_x, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, RetroColors.YELLOW, 2)
        
        # Level and Round display (top right)
        level_text = f"LEVEL {self.level}"
        level_size = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        level_x = self.screen_width - level_size[0] - 20
        cv2.putText(frame, level_text, (level_x, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, RetroColors.WHITE, 2)
        
        round_text = f"ROUND {self.round}"
        round_size = cv2.getTextSize(round_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        round_x = self.screen_width - round_size[0] - 20
        cv2.putText(frame, round_text, (round_x, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, RetroColors.WHITE, 2)
        
        # Shots remaining (top right)
        shots_remaining = self.shots_per_round - self.shots_fired
        shots_text = f"SHOTS: {shots_remaining}"
        shots_size = cv2.getTextSize(shots_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        shots_x = self.screen_width - shots_size[0] - 20
        cv2.putText(frame, shots_text, (shots_x, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, RetroColors.WHITE, 2)
        
        # Hit ratio (top right)
        total_ducks = self.ducks_shot + self.ducks_missed
        if total_ducks > 0:
            hit_ratio = int((self.ducks_shot / total_ducks) * 100)
            ratio_text = f"HIT RATIO: {hit_ratio}%"
            ratio_size = cv2.getTextSize(ratio_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            ratio_x = self.screen_width - ratio_size[0] - 20
            cv2.putText(frame, ratio_text, (ratio_x, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, RetroColors.WHITE, 2)
        
        # Instructions (top center)
        if self.aiming:
            if self.thumb_trigger_active:
                instruction = "FIRING!"
                color = RetroColors.RED
            else:
                instruction = "MOVE THUMB LEFT→TOP TO SHOOT"
                color = RetroColors.WHITE
        else:
            instruction = "POINT INDEX FINGER TO AIM"
            color = RetroColors.WHITE
            
        cv2.putText(frame, instruction, 
                   (self.screen_width // 2 - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Progress towards win condition (top left)
        if self.total_ducks_this_round > 0:
            hit_percentage = self.ducks_shot / self.total_ducks_this_round
            progress_text = f"Progress: {self.ducks_shot}/{self.total_ducks_this_round} ({hit_percentage:.1%}) - Need 50% to win"
            
            # Color based on progress
            if hit_percentage >= 0.5:
                progress_color = RetroColors.GREEN
            elif hit_percentage >= 0.3:
                progress_color = RetroColors.YELLOW
            else:
                progress_color = RetroColors.RED
            
            cv2.putText(frame, progress_text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, progress_color, 1)
        
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
                # Hit explosion with special effects for different duck types
                effect_size = int(40 * alpha)
                effect_color = effect.get('effect_color', RetroColors.RED)
                effect_thickness = 5 if effect.get('duck_type') in ["purple", "golden"] else 3
                
                cv2.circle(frame, (int(effect['x']), int(effect['y'])), 
                          effect_size, effect_color, effect_thickness)
                
                # Extra sparkle effect for special ducks
                if effect.get('duck_type') in ["purple", "golden", "silver"]:
                    sparkle_size = int(20 * alpha)
                    cv2.circle(frame, (int(effect['x']), int(effect['y'])), 
                              sparkle_size, RetroColors.WHITE, 2)
                
                # Score popup with color coding
                if 'score' in effect:
                    score_y = int(effect['y'] - 30 - (1 - alpha) * 50)
                    score_color = effect_color if effect.get('duck_type') != "normal" else RetroColors.YELLOW
                    cv2.putText(frame, f"+{effect['score']}", 
                               (int(effect['x'] - 30), score_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)
        
        # Draw target cursor
        self.draw_target_cursor(frame)
        
        # Draw UI
        self.draw_retro_ui(frame)
        
        # Duck type legend (only show on first few levels)
        if self.level <= 3:
            legend_y = 120
            cv2.putText(frame, "DUCK TYPES:", (20, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, RetroColors.WHITE, 1)
            
            duck_types = [
                ("PURPLE", RetroColors.DUCK_PURPLE, 2000),
                ("GOLDEN", RetroColors.DUCK_GOLDEN, 1500),
                ("SILVER", RetroColors.DUCK_SILVER, 1000),
                ("BLUE", RetroColors.DUCK_BLUE, 800),
                ("GREEN", RetroColors.DUCK_GREEN, 600),
                ("RED", RetroColors.DUCK_RED, 400),
                ("BROWN", RetroColors.DUCK_BROWN, 500)
            ]
            
            for i, (duck_name, color, score) in enumerate(duck_types):
                legend_y += 25
                cv2.putText(frame, f"{duck_name}: {score}", (20, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw back button
        self.back_button.draw(frame, self.hand_position)

        # Draw game completion screen if game is completed
        if self.game_completed:
            self.draw_game_completion_screen(frame)

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
    print("   🔫 Move thumb in quarter-circle (left→top) to shoot!")
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

                # Enhanced gun gesture detection with improved stability
                is_gun, gun_pos, shooting_gesture = game.detect_enhanced_gun_gesture(hand_landmarks)
                
                if is_gun:
                    game.aiming = True
                    game.gun_position = gun_pos
                    game.thumb_trigger_active = shooting_gesture
                    game.last_gesture_time = time.time()  # Update gesture time
                    
                    # Handle button interactions if game is completed
                    if game.game_completed and shooting_gesture:
                        button_result = game.handle_button_interactions(shooting_gesture, gun_pos)
                        if button_result == "back":
                            print("Returning to app store...")
                            break
                        elif button_result:
                            # Game was reset, continue
                            continue
                    
                    # Shoot when quarter-circle gesture is detected (only if game is active)
                    elif shooting_gesture and game.can_shoot and not game.game_completed:
                        game.shoot(gun_pos[0], gun_pos[1])
                        print(f"🔫 SHOT FIRED at ({gun_pos[0]}, {gun_pos[1]}) - Quarter-circle gesture detected!")
                        
                        # Clear thumb position history after shooting to prevent multiple shots
                        game.thumb_positions.clear()
                else:
                    # Debug: show when gesture is not detected
                    if show_hand_debug:
                        print("❌ Gun gesture not detected - check finger positions")
        else:
            # No hand detected - maintain cursor for a short time if we had one recently
            if game.last_index_position and game.index_finger_detected:
                # Keep cursor visible for 0.5 seconds after hand is lost
                if time.time() - game.last_gesture_time < 0.5:
                    game.aiming = True
                    game.gun_position = game.last_index_position
                else:
                    game.index_finger_detected = False

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
