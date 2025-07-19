import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import sys
import os
import argparse
from enum import Enum

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

class WeaponType(Enum):
    PISTOL = 1
    SHOTGUN = 2
    CHAINGUN = 3

class DoomColors:
    """Classic Doom color palette"""
    DARK_RED = (0, 0, 139)
    BRIGHT_RED = (0, 0, 255)
    DARK_GRAY = (64, 64, 64)
    LIGHT_GRAY = (128, 128, 128)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    ORANGE = (0, 165, 255)
    BROWN = (42, 42, 165)
    BLUE = (255, 0, 0)
    MUZZLE_FLASH = (255, 255, 0)

class Enemy:
    def __init__(self, x, y, screen_width, screen_height, enemy_type="imp"):
        self.x = x
        self.y = y
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.enemy_type = enemy_type
        
        # Enemy properties
        self.width = 80
        self.height = 120
        self.health = 100
        self.max_health = 100
        self.alive = True
        
        # Movement
        self.speed = 1.5
        self.direction = random.uniform(0, 2 * math.pi)
        self.last_direction_change = time.time()
        
        # Animation
        self.animation_frame = 0
        self.animation_speed = 0.1
        
        # Combat
        self.last_attack_time = 0
        self.attack_cooldown = 2.0
        self.damage = 20
        
        # Visual effects
        self.hit_time = 0
        self.death_time = 0
        
    def update(self, player_x, player_y):
        """Update enemy AI and movement"""
        if not self.alive:
            return
            
        current_time = time.time()
        
        # Simple AI: move toward player occasionally
        if current_time - self.last_direction_change > 3.0:
            dx = player_x - self.x
            dy = player_y - self.y
            self.direction = math.atan2(dy, dx) + random.uniform(-0.5, 0.5)
            self.last_direction_change = current_time
            
        # Move
        self.x += math.cos(self.direction) * self.speed
        self.y += math.sin(self.direction) * self.speed
        
        # Keep on screen
        self.x = max(50, min(self.screen_width - 50, self.x))
        self.y = max(50, min(self.screen_height - 100, self.y))
        
        # Update animation
        self.animation_frame += self.animation_speed
        
    def draw(self, frame):
        """Draw the enemy"""
        if not self.alive:
            return
            
        current_time = time.time()
        
        # Hit flash effect
        hit_flash = current_time - self.hit_time < 0.2
        base_color = DoomColors.BRIGHT_RED if hit_flash else DoomColors.DARK_RED
        
        # Draw enemy body (simple demon-like shape)
        # Main body
        cv2.ellipse(frame, 
                   (int(self.x), int(self.y)), 
                   (self.width // 2, self.height // 2), 
                   0, 0, 360, base_color, -1)
        
        # Head
        head_y = int(self.y - self.height // 3)
        cv2.circle(frame, (int(self.x), head_y), self.width // 3, base_color, -1)
        
        # Eyes (glowing red)
        eye_offset = 15
        left_eye = (int(self.x - eye_offset), head_y - 10)
        right_eye = (int(self.x + eye_offset), head_y - 10)
        
        cv2.circle(frame, left_eye, 8, DoomColors.YELLOW, -1)
        cv2.circle(frame, right_eye, 8, DoomColors.YELLOW, -1)
        cv2.circle(frame, left_eye, 5, DoomColors.BRIGHT_RED, -1)
        cv2.circle(frame, right_eye, 5, DoomColors.BRIGHT_RED, -1)
        
        # Horns
        horn_points_left = np.array([
            [self.x - 25, head_y - 20],
            [self.x - 35, head_y - 40],
            [self.x - 20, head_y - 25]
        ], np.int32)
        
        horn_points_right = np.array([
            [self.x + 25, head_y - 20],
            [self.x + 35, head_y - 40],
            [self.x + 20, head_y - 25]
        ], np.int32)
        
        cv2.fillPoly(frame, [horn_points_left], DoomColors.BROWN)
        cv2.fillPoly(frame, [horn_points_right], DoomColors.BROWN)
        
        # Health bar
        health_ratio = self.health / self.max_health
        bar_width = self.width
        bar_height = 8
        bar_x = int(self.x - bar_width // 2)
        bar_y = int(self.y - self.height // 2 - 20)
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     DoomColors.DARK_GRAY, -1)
        
        # Health
        health_width = int(bar_width * health_ratio)
        health_color = DoomColors.GREEN if health_ratio > 0.5 else DoomColors.YELLOW if health_ratio > 0.25 else DoomColors.BRIGHT_RED
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + health_width, bar_y + bar_height), 
                     health_color, -1)
        
    def take_damage(self, damage):
        """Apply damage to enemy"""
        if not self.alive:
            return False
            
        self.health -= damage
        self.hit_time = time.time()
        
        if self.health <= 0:
            self.alive = False
            self.death_time = time.time()
            return True  # Enemy killed
        return False
        
    def check_hit(self, shot_x, shot_y):
        """Check if enemy was hit"""
        if not self.alive:
            return False
            
        distance = math.sqrt((self.x - shot_x)**2 + (self.y - shot_y)**2)
        return distance < max(self.width, self.height) // 2

class Weapon:
    def __init__(self, weapon_type):
        self.type = weapon_type
        self.setup_weapon_stats()
        
    def setup_weapon_stats(self):
        """Setup weapon-specific statistics"""
        if self.type == WeaponType.PISTOL:
            self.name = "PISTOL"
            self.max_ammo = 12
            self.current_ammo = 12
            self.damage = 25
            self.fire_rate = 0.3  # Seconds between shots
            self.reload_time = 1.5
            self.spread = 0.02  # Accuracy
            
        elif self.type == WeaponType.SHOTGUN:
            self.name = "SHOTGUN"
            self.max_ammo = 8
            self.current_ammo = 8
            self.damage = 80
            self.fire_rate = 0.8
            self.reload_time = 2.0
            self.spread = 0.08
            
        elif self.type == WeaponType.CHAINGUN:
            self.name = "CHAINGUN"
            self.max_ammo = 200
            self.current_ammo = 200
            self.damage = 15
            self.fire_rate = 0.1
            self.reload_time = 3.0
            self.spread = 0.05
            
        self.last_shot_time = 0
        self.reloading = False
        self.reload_start_time = 0
        
    def can_shoot(self):
        """Check if weapon can shoot"""
        current_time = time.time()
        return (self.current_ammo > 0 and 
                not self.reloading and
                current_time - self.last_shot_time >= self.fire_rate)
                
    def shoot(self):
        """Fire the weapon"""
        if not self.can_shoot():
            return False
            
        self.current_ammo -= 1
        self.last_shot_time = time.time()
        return True
        
    def start_reload(self):
        """Start reloading process"""
        if self.current_ammo < self.max_ammo and not self.reloading:
            self.reloading = True
            self.reload_start_time = time.time()
            
    def update(self):
        """Update weapon state"""
        if self.reloading:
            if time.time() - self.reload_start_time >= self.reload_time:
                self.current_ammo = self.max_ammo
                self.reloading = False
                
    def draw_weapon(self, frame, screen_width, screen_height, recoil_offset=0):
        """Draw weapon on screen"""
        weapon_bottom_y = screen_height - 50 + recoil_offset
        weapon_center_x = screen_width - 150
        
        if self.type == WeaponType.PISTOL:
            self.draw_pistol(frame, weapon_center_x, weapon_bottom_y)
        elif self.type == WeaponType.SHOTGUN:
            self.draw_shotgun(frame, weapon_center_x, weapon_bottom_y)
        elif self.type == WeaponType.CHAINGUN:
            self.draw_chaingun(frame, weapon_center_x, weapon_bottom_y)
            
    def draw_pistol(self, frame, x, y):
        """Draw pistol sprite"""
        # Pistol barrel
        barrel_points = np.array([
            [x - 10, y - 80],
            [x + 10, y - 80],
            [x + 15, y - 40],
            [x - 15, y - 40]
        ], np.int32)
        cv2.fillPoly(frame, [barrel_points], DoomColors.DARK_GRAY)
        
        # Pistol grip
        grip_points = np.array([
            [x - 15, y - 40],
            [x + 15, y - 40],
            [x + 20, y],
            [x - 20, y]
        ], np.int32)
        cv2.fillPoly(frame, [grip_points], DoomColors.BROWN)
        
        # Trigger guard
        cv2.ellipse(frame, (x, y - 25), (8, 15), 0, 0, 360, DoomColors.BLACK, 2)
        
    def draw_shotgun(self, frame, x, y):
        """Draw shotgun sprite"""
        # Double barrel
        cv2.rectangle(frame, (x - 8, y - 100), (x - 2, y - 30), DoomColors.DARK_GRAY, -1)
        cv2.rectangle(frame, (x + 2, y - 100), (x + 8, y - 30), DoomColors.DARK_GRAY, -1)
        
        # Stock
        stock_points = np.array([
            [x - 15, y - 30],
            [x + 15, y - 30],
            [x + 25, y],
            [x - 25, y]
        ], np.int32)
        cv2.fillPoly(frame, [stock_points], DoomColors.BROWN)
        
    def draw_chaingun(self, frame, x, y):
        """Draw chaingun sprite"""
        # Multiple barrels
        for i in range(6):
            angle = i * math.pi / 3
            barrel_x = x + int(15 * math.cos(angle))
            barrel_y = y - 60 + int(15 * math.sin(angle))
            cv2.circle(frame, (barrel_x, barrel_y), 4, DoomColors.DARK_GRAY, -1)
            
        # Main body
        cv2.rectangle(frame, (x - 25, y - 80), (x + 25, y - 20), DoomColors.LIGHT_GRAY, -1)
        
        # Handle
        cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y), DoomColors.BLACK, -1)

class DoomGame:
    def __init__(self, screen_width=1280, screen_height=720):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,  # Track both hands
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        
        # Game state
        self.score = 0
        self.health = 100
        self.max_health = 100
        self.armor = 0
        self.level = 1
        
        # Player position (for enemy AI)
        self.player_x = screen_width // 2
        self.player_y = screen_height // 2
        
        # Weapon system
        self.current_weapon = Weapon(WeaponType.PISTOL)
        self.available_weapons = [WeaponType.PISTOL, WeaponType.SHOTGUN, WeaponType.CHAINGUN]
        
        # Enemies
        self.enemies = []
        self.max_enemies = 5
        self.enemy_spawn_timer = 0
        self.enemy_spawn_interval = 3.0
        
        # Hand tracking
        self.left_hand = None
        self.right_hand = None
        self.weapon_hand = None  # Which hand holds the weapon
        
        # Gesture recognition
        self.current_gesture = None
        self.gesture_start_time = 0
        self.gesture_confirmation_time = 0.5
        
        # Visual effects
        self.shots = []
        self.explosions = []
        self.muzzle_flashes = []
        self.recoil_offset = 0
        self.recoil_decay = 0.8
        
        # Back button
        self.back_button = BackButton(screen_width, screen_height)
        
        # Game timing
        self.game_start_time = time.time()
        
    def detect_hand_gesture(self, landmarks, hand_label):
        """Advanced gesture recognition for both hands"""
        if len(landmarks.landmark) < 21:
            return "unknown"
            
        # Get all finger landmarks
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]
        
        thumb_pip = landmarks.landmark[3]
        index_pip = landmarks.landmark[6]
        middle_pip = landmarks.landmark[10]
        ring_pip = landmarks.landmark[14]
        pinky_pip = landmarks.landmark[18]
        
        # Check finger states
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y
        ring_extended = ring_tip.y < ring_pip.y
        pinky_extended = pinky_tip.y < pinky_pip.y
        thumb_extended = thumb_tip.x > thumb_pip.x if hand_label == "Right" else thumb_tip.x < thumb_pip.x
        
        # Gun gesture (index finger pointing)
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "gun"
            
        # Open palm (all fingers extended)
        if index_extended and middle_extended and ring_extended and pinky_extended:
            return "open_palm"
            
        # Closed fist (all fingers closed)
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "fist"
            
        # Thumbs up
        if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "thumbs_up"
            
        # Thumbs down
        if not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "thumbs_down"
            
        return "unknown"
        
    def check_reload_gesture(self):
        """Check if hands are positioned for reloading"""
        if not self.left_hand or not self.right_hand:
            return False
            
        left_wrist = self.left_hand.landmark[0]
        right_wrist = self.right_hand.landmark[0]
        
        # Check if one hand is above the other (reload gesture)
        vertical_distance = abs(left_wrist.y - right_wrist.y)
        horizontal_distance = abs(left_wrist.x - right_wrist.x)
        
        # One hand should be significantly above the other but horizontally close
        return vertical_distance > 0.15 and horizontal_distance < 0.2
        
    def spawn_enemy(self):
        """Spawn a new enemy"""
        if len(self.enemies) < self.max_enemies:
            # Spawn at random edge of screen
            edge = random.choice(['left', 'right', 'top'])
            if edge == 'left':
                x, y = 0, random.randint(100, self.screen_height - 200)
            elif edge == 'right':
                x, y = self.screen_width, random.randint(100, self.screen_height - 200)
            else:  # top
                x, y = random.randint(100, self.screen_width - 100), 0
                
            enemy = Enemy(x, y, self.screen_width, self.screen_height)
            self.enemies.append(enemy)
            
    def handle_shooting(self, gun_position):
        """Handle weapon firing"""
        if not self.current_weapon.can_shoot():
            return
            
        if self.current_weapon.shoot():
            gun_x, gun_y = gun_position
            
            # Add muzzle flash
            self.muzzle_flashes.append({
                'x': gun_x,
                'y': gun_y,
                'time': time.time(),
                'duration': 0.1
            })
            
            # Add recoil
            self.recoil_offset = 15
            
            # Create shot with weapon spread
            spread = self.current_weapon.spread
            shot_x = gun_x + random.uniform(-spread * 100, spread * 100)
            shot_y = gun_y + random.uniform(-spread * 100, spread * 100)
            
            # Add shot visual
            self.shots.append({
                'x': shot_x,
                'y': shot_y,
                'time': time.time(),
                'duration': 0.3
            })
            
            # Check hits on enemies
            for enemy in self.enemies:
                if enemy.check_hit(shot_x, shot_y):
                    killed = enemy.take_damage(self.current_weapon.damage)
                    if killed:
                        self.score += 100
                        # Add explosion effect
                        self.explosions.append({
                            'x': enemy.x,
                            'y': enemy.y,
                            'time': time.time(),
                            'duration': 1.0
                        })
                    break
                    
    def handle_melee(self, fist_position):
        """Handle melee attacks"""
        fist_x, fist_y = fist_position
        
        # Check for nearby enemies
        for enemy in self.enemies:
            distance = math.sqrt((enemy.x - fist_x)**2 + (enemy.y - fist_y)**2)
            if distance < 100:  # Melee range
                killed = enemy.take_damage(50)  # Melee damage
                if killed:
                    self.score += 50
                    
    def update(self):
        """Update game state"""
        current_time = time.time()
        
        # Update weapon
        self.current_weapon.update()
        
        # Update recoil
        self.recoil_offset *= self.recoil_decay
        if abs(self.recoil_offset) < 0.5:
            self.recoil_offset = 0
            
        # Spawn enemies
        if current_time - self.enemy_spawn_timer > self.enemy_spawn_interval:
            self.spawn_enemy()
            self.enemy_spawn_timer = current_time
            
        # Update enemies
        for enemy in self.enemies:
            enemy.update(self.player_x, self.player_y)
            
        # Remove dead enemies
        self.enemies = [enemy for enemy in self.enemies if enemy.alive]
        
        # Update visual effects
        self.shots = [shot for shot in self.shots 
                     if current_time - shot['time'] < shot['duration']]
        
        self.explosions = [exp for exp in self.explosions 
                          if current_time - exp['time'] < exp['duration']]
        
        self.muzzle_flashes = [flash for flash in self.muzzle_flashes 
                              if current_time - flash['time'] < flash['duration']]
        
    def draw_hud(self, frame):
        """Draw heads-up display"""
        # Health bar
        health_ratio = self.health / self.max_health
        health_width = 200
        health_height = 20
        health_x = 20
        health_y = self.screen_height - 100
        
        # Health background
        cv2.rectangle(frame, (health_x, health_y), 
                     (health_x + health_width, health_y + health_height), 
                     DoomColors.DARK_GRAY, -1)
        
        # Health bar
        current_health_width = int(health_width * health_ratio)
        health_color = DoomColors.GREEN if health_ratio > 0.5 else DoomColors.YELLOW if health_ratio > 0.25 else DoomColors.BRIGHT_RED
        cv2.rectangle(frame, (health_x, health_y), 
                     (health_x + current_health_width, health_y + health_height), 
                     health_color, -1)
        
        # Health text
        cv2.putText(frame, f"HEALTH: {self.health}", 
                   (health_x, health_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, DoomColors.WHITE, 2)
        
        # Ammo display
        ammo_x = 20
        ammo_y = health_y - 40
        
        if self.current_weapon.reloading:
            reload_progress = (time.time() - self.current_weapon.reload_start_time) / self.current_weapon.reload_time
            ammo_text = f"RELOADING... {int(reload_progress * 100)}%"
            color = DoomColors.YELLOW
        else:
            ammo_text = f"AMMO: {self.current_weapon.current_ammo}/{self.current_weapon.max_ammo}"
            color = DoomColors.WHITE
            
        cv2.putText(frame, ammo_text, (ammo_x, ammo_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Weapon name
        weapon_y = ammo_y - 30
        cv2.putText(frame, self.current_weapon.name, (ammo_x, weapon_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, DoomColors.ORANGE, 2)
        
        # Score
        score_text = f"SCORE: {self.score}"
        cv2.putText(frame, score_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, DoomColors.YELLOW, 2)
        
        # Enemy count
        enemy_text = f"ENEMIES: {len(self.enemies)}"
        cv2.putText(frame, enemy_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, DoomColors.WHITE, 2)
                   
    def draw_gesture_indicators(self, frame):
        """Draw gesture recognition indicators"""
        if self.left_hand and self.right_hand:
            if self.check_reload_gesture():
                # Show reload indicator
                cv2.putText(frame, "RELOADING!", 
                           (self.screen_width // 2 - 100, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, DoomColors.YELLOW, 3)
                           
        # Show current gestures
        gesture_y = 150
        if self.weapon_hand:
            weapon_gesture = self.detect_hand_gesture(self.weapon_hand, "Right")
            cv2.putText(frame, f"WEAPON HAND: {weapon_gesture.upper()}", 
                       (self.screen_width - 300, gesture_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, DoomColors.WHITE, 1)
        
    def draw_background(self, frame):
        """Draw Doom-style background"""
        # Dark gradient background
        for y in range(self.screen_height):
            intensity = 1.0 - (y / self.screen_height) * 0.7
            gray_value = int(30 * intensity)
            color = (gray_value, gray_value, gray_value)
            cv2.line(frame, (0, y), (self.screen_width, y), color, 1)
        
        # Add some texture lines
        for i in range(0, self.screen_width, 50):
            cv2.line(frame, (i, 0), (i, self.screen_height), DoomColors.DARK_GRAY, 1)
            
    def draw(self, frame):
        """Main drawing function"""
        # Draw background
        self.draw_background(frame)
        
        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(frame)
            
        # Draw visual effects
        current_time = time.time()
        
        # Draw explosions
        for explosion in self.explosions:
            alpha = 1.0 - (current_time - explosion['time']) / explosion['duration']
            if alpha > 0:
                explosion_size = int(60 * alpha)
                cv2.circle(frame, (int(explosion['x']), int(explosion['y'])), 
                          explosion_size, DoomColors.BRIGHT_RED, 3)
                cv2.circle(frame, (int(explosion['x']), int(explosion['y'])), 
                          explosion_size // 2, DoomColors.YELLOW, -1)
                          
        # Draw shots
        for shot in self.shots:
            alpha = 1.0 - (current_time - shot['time']) / shot['duration']
            if alpha > 0:
                shot_size = int(10 * alpha)
                cv2.circle(frame, (shot['x'], shot['y']), 
                          shot_size, DoomColors.MUZZLE_FLASH, -1)
                          
        # Draw muzzle flashes
        for flash in self.muzzle_flashes:
            alpha = 1.0 - (current_time - flash['time']) / flash['duration']
            if alpha > 0:
                flash_size = int(40 * alpha)
                cv2.circle(frame, (flash['x'], flash['y']), 
                          flash_size, DoomColors.MUZZLE_FLASH, -1)
        
        # Draw weapon
        self.current_weapon.draw_weapon(frame, self.screen_width, self.screen_height, int(self.recoil_offset))
        
        # Draw crosshair if aiming
        if self.weapon_hand:
            # Get gun position from index finger
            index_tip = self.weapon_hand.landmark[8]
            gun_x = int(index_tip.x * self.screen_width)
            gun_y = int(index_tip.y * self.screen_height)
            
            # Draw crosshair
            crosshair_size = 15
            cv2.line(frame, (gun_x - crosshair_size, gun_y), 
                    (gun_x + crosshair_size, gun_y), DoomColors.BRIGHT_RED, 2)
            cv2.line(frame, (gun_x, gun_y - crosshair_size), 
                    (gun_x, gun_y + crosshair_size), DoomColors.BRIGHT_RED, 2)
        
        # Draw HUD
        self.draw_hud(frame)
        
        # Draw gesture indicators
        self.draw_gesture_indicators(frame)
        
        # Draw back button
        self.back_button.draw(frame, (self.player_x, self.player_y))

def main():
    parser = argparse.ArgumentParser(description='Hand-Gesture Doom Game')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {args.camera}")
        exit(-1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    game = DoomGame(1280, 720)
    
    cv2.namedWindow('Hand Gesture Doom', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Gesture Doom', 1280, 720)

    print("👹 HAND GESTURE DOOM - RIP AND TEAR!")
    print("=" * 50)
    print("🎮 Controls:")
    print("   🔫 Point index finger to aim and shoot")
    print("   🔄 Place one hand above the other to reload")
    print("   👊 Make a fist for melee attacks")
    print("   👍👎 Thumbs up/down to switch weapons")
    print("   ✋ Open palm for movement")
    print("   🔙 Use back button to exit")
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
        game.left_hand = None
        game.right_hand = None
        game.weapon_hand = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                hand_label = handedness.classification[0].label
                
                if hand_label == "Left":
                    game.left_hand = hand_landmarks
                else:
                    game.right_hand = hand_landmarks
                    
                # Detect gestures
                gesture = game.detect_hand_gesture(hand_landmarks, hand_label)
                
                # Handle weapon gestures (prefer right hand)
                if gesture == "gun" and hand_label == "Right":
                    game.weapon_hand = hand_landmarks
                    
                    # Get gun position
                    index_tip = hand_landmarks.landmark[8]
                    gun_x = int(index_tip.x * 1280)
                    gun_y = int(index_tip.y * 720)
                    
                    # Shoot
                    game.handle_shooting((gun_x, gun_y))
                    
                elif gesture == "fist":
                    # Melee attack
                    wrist = hand_landmarks.landmark[0]
                    fist_x = int(wrist.x * 1280)
                    fist_y = int(wrist.y * 720)
                    game.handle_melee((fist_x, fist_y))
                    
                elif gesture == "thumbs_up":
                    # Next weapon
                    current_idx = game.available_weapons.index(game.current_weapon.type)
                    next_idx = (current_idx + 1) % len(game.available_weapons)
                    game.current_weapon = Weapon(game.available_weapons[next_idx])
                    
                elif gesture == "thumbs_down":
                    # Previous weapon
                    current_idx = game.available_weapons.index(game.current_weapon.type)
                    prev_idx = (current_idx - 1) % len(game.available_weapons)
                    game.current_weapon = Weapon(game.available_weapons[prev_idx])
        
        # Check for reload gesture
        if game.check_reload_gesture():
            game.current_weapon.start_reload()

        # Handle back button
        key = cv2.waitKey(1) & 0xFF
        hand_pos = (game.player_x, game.player_y) if results.multi_hand_landmarks else None
        if game.back_button.handle_input(key, results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None, hand_pos):
            print("Returning to app store...")
            break

        # Update and draw game
        game.update()
        game.draw(frame)

        cv2.imshow('Hand Gesture Doom', frame)

        # Keyboard controls
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):  # Restart
            game = DoomGame(1280, 720)

    cap.release()
    cv2.destroyAllWindows()
    print("👹 RIP AND TEAR COMPLETE! Thanks for playing!")

if __name__ == "__main__":
    main()
