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
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import threading
from collections import deque
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

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

class GameState(Enum):
    MAIN_MENU = 1
    PLAYING = 2
    GOAL_SCORED = 3
    GOAL_SAVED = 4
    GAME_OVER = 5
    SETTINGS = 6

class DifficultyLevel(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    LEGENDARY = 4

class ShotType(Enum):
    LOW_LEFT = 1
    LOW_CENTER = 2
    LOW_RIGHT = 3
    HIGH_LEFT = 4
    HIGH_CENTER = 5
    HIGH_RIGHT = 6

@dataclass
class GameConfig:
    """Game configuration settings"""
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    fullscreen: bool = True
    show_pose_landmarks: bool = False
    sensitivity: float = 0.8
    goalkeeper_skill: float = 0.7

class FootballColors:
    """Football game color palette"""
    # Field colors
    GRASS_GREEN = (0, 128, 0)
    DARK_GRASS = (0, 100, 0)
    FIELD_LINES = (255, 255, 255)
    
    # Goal colors
    GOAL_WHITE = (255, 255, 255)
    GOAL_SHADOW = (200, 200, 200)
    NET_COLOR = (240, 240, 240)
    
    # Player colors
    PLAYER_SHIRT = (0, 0, 255)  # Blue
    PLAYER_SHORTS = (0, 0, 180)
    PLAYER_SKIN = (220, 180, 140)
    
    # Goalkeeper colors
    GK_SHIRT = (255, 165, 0)  # Orange
    GK_SHORTS = (255, 140, 0)
    GK_GLOVES = (50, 50, 50)
    
    # Ball colors
    BALL_WHITE = (255, 255, 255)
    BALL_BLACK = (0, 0, 0)
    BALL_SHADOW = (100, 100, 100)
    
    # UI colors
    SCORE_YELLOW = (255, 255, 0)
    TEXT_WHITE = (255, 255, 255)
    TEXT_BLACK = (0, 0, 0)
    SUCCESS_GREEN = (0, 255, 0)
    DANGER_RED = (255, 0, 0)

class Vector2D:
    """2D Vector class for physics calculations"""
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector2D(self.x / mag, self.y / mag)
        return Vector2D(0, 0)
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Ball:
    """Football ball with realistic physics"""
    def __init__(self, x: float, y: float, screen_width: int, screen_height: int):
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(0, 0)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.radius = 20
        self.mass = 1.0
        self.gravity = 500  # Pixels per second squared
        self.air_resistance = 0.98
        self.bounce_damping = 0.7
        self.spinning = False
        self.spin_angle = 0
        self.trail = deque(maxlen=15)
        
        # Ball state
        self.is_moving = False
        self.on_ground = True
        self.goal_scored = False
        self.out_of_bounds = False
        
    def reset(self, x: float, y: float):
        """Reset ball to starting position"""
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(0, 0)
        self.is_moving = False
        self.on_ground = True
        self.goal_scored = False
        self.out_of_bounds = False
        self.spinning = False
        self.trail.clear()
    
    def kick(self, direction: Vector2D, power: float):
        """Apply kick force to ball"""
        max_power = 800  # Maximum kick velocity
        kick_force = direction.normalize() * (power * max_power)
        
        self.velocity = kick_force
        self.is_moving = True
        self.on_ground = False
        self.spinning = True
        
        # Add some randomness for realism
        self.velocity.x += random.uniform(-50, 50)
        self.velocity.y += random.uniform(-20, 20)
    
    def update(self, dt: float, ground_y: float):
        """Update ball physics"""
        if not self.is_moving:
            return
        
        # Add current position to trail
        self.trail.append((self.position.x, self.position.y))
        
        # Apply gravity
        self.velocity.y += self.gravity * dt
        
        # Apply air resistance
        self.velocity.x *= self.air_resistance
        self.velocity.y *= self.air_resistance
        
        # Update position
        self.position.x += self.velocity.x * dt
        self.position.y += self.velocity.y * dt
        
        # Ground collision
        if self.position.y + self.radius >= ground_y:
            self.position.y = ground_y - self.radius
            self.velocity.y *= -self.bounce_damping
            
            # Stop bouncing if velocity is too low
            if abs(self.velocity.y) < 50:
                self.velocity.y = 0
                self.on_ground = True
            
            # Friction on ground
            self.velocity.x *= 0.9
        
        # Side boundaries
        if self.position.x - self.radius <= 0 or self.position.x + self.radius >= self.screen_width:
            self.out_of_bounds = True
            self.is_moving = False
        
        # Top boundary
        if self.position.y - self.radius <= 0:
            self.out_of_bounds = True
            self.is_moving = False
        
        # Update spinning
        if self.spinning:
            self.spin_angle += self.velocity.magnitude() * dt * 0.01
        
        # Stop if velocity is very low
        if self.velocity.magnitude() < 30 and self.on_ground:
            self.is_moving = False
            self.spinning = False
    
    def check_goal_collision(self, goal_left: float, goal_right: float, goal_top: float, goal_bottom: float) -> bool:
        """Check if ball scored a goal"""
        if (goal_left <= self.position.x <= goal_right and 
            goal_top <= self.position.y <= goal_bottom):
            self.goal_scored = True
            self.is_moving = False
            return True
        return False
    
    def draw(self, frame):
        """Draw ball with realistic appearance"""
        x, y = int(self.position.x), int(self.position.y)
        
        # Draw trail
        trail_alpha = 1.0
        for i, trail_pos in enumerate(self.trail):
            alpha = trail_alpha * (i / len(self.trail)) * 0.3
            if alpha > 0:
                trail_color = tuple(int(c * alpha) for c in FootballColors.BALL_SHADOW)
                cv2.circle(frame, (int(trail_pos[0]), int(trail_pos[1])), 
                          max(3, int(self.radius * alpha)), trail_color, -1)
        
        # Draw ball shadow on ground
        shadow_y = frame.shape[0] - 50  # Ground level
        shadow_offset = max(0, (shadow_y - y) // 10)  # Shadow gets smaller with height
        cv2.ellipse(frame, (x, shadow_y), (self.radius - shadow_offset, self.radius // 3), 
                   0, 0, 360, FootballColors.BALL_SHADOW, -1)
        
        # Draw main ball
        cv2.circle(frame, (x, y), self.radius, FootballColors.BALL_WHITE, -1)
        
        # Draw soccer ball pattern
        # Hexagonal pattern (simplified)
        pattern_angle = self.spin_angle if self.spinning else 0
        for i in range(6):
            angle = (i * math.pi / 3) + pattern_angle
            pattern_x = x + int((self.radius - 5) * math.cos(angle))
            pattern_y = y + int((self.radius - 5) * math.sin(angle))
            cv2.circle(frame, (pattern_x, pattern_y), 3, FootballColors.BALL_BLACK, -1)
        
        # Central pentagon
        pentagon_points = []
        for i in range(5):
            angle = (i * 2 * math.pi / 5) + pattern_angle
            px = x + int(6 * math.cos(angle))
            py = y + int(6 * math.sin(angle))
            pentagon_points.append([px, py])
        
        pentagon_points = np.array(pentagon_points, np.int32)
        cv2.fillPoly(frame, [pentagon_points], FootballColors.BALL_BLACK)
        
        # Highlight for 3D effect
        highlight_x = x - self.radius // 3
        highlight_y = y - self.radius // 3
        cv2.circle(frame, (highlight_x, highlight_y), self.radius // 4, (255, 255, 255), -1)

class Goalkeeper:
    """AI Goalkeeper with realistic behavior"""
    def __init__(self, x: float, y: float, goal_width: float, goal_height: float, difficulty: DifficultyLevel):
        self.home_position = Vector2D(x, y)
        self.position = Vector2D(x, y)
        self.goal_width = goal_width
        self.goal_height = goal_height
        self.difficulty = difficulty
        
        # Goalkeeper properties
        self.width = 40
        self.height = 80
        self.dive_speed = 300
        self.reaction_time = 0.3  # Seconds
        self.reach = 120  # Goalkeeper reach
        
        # State
        self.state = "ready"  # ready, diving_left, diving_right, catching
        self.dive_start_time = 0
        self.dive_duration = 1.0
        self.dive_direction = Vector2D(0, 0)
        self.save_made = False
        
        self._setup_difficulty()
    
    def _setup_difficulty(self):
        """Setup goalkeeper skills based on difficulty"""
        if self.difficulty == DifficultyLevel.EASY:
            self.reaction_time = 0.8
            self.dive_speed = 200
            self.reach = 80
        elif self.difficulty == DifficultyLevel.MEDIUM:
            self.reaction_time = 0.5
            self.dive_speed = 300
            self.reach = 120
        elif self.difficulty == DifficultyLevel.HARD:
            self.reaction_time = 0.3
            self.dive_speed = 400
            self.reach = 140
        elif self.difficulty == DifficultyLevel.LEGENDARY:
            self.reaction_time = 0.2
            self.dive_speed = 500
            self.reach = 160
    
    def reset(self):
        """Reset goalkeeper to ready position"""
        self.position = Vector2D(self.home_position.x, self.home_position.y)
        self.state = "ready"
        self.save_made = False
    
    def react_to_shot(self, ball_position: Vector2D, ball_velocity: Vector2D):
        """Goalkeeper AI reaction to ball shot"""
        if self.state != "ready":
            return
        
        # Predict where ball will cross goal line
        if ball_velocity.x != 0:
            time_to_goal = (self.home_position.x - ball_position.x) / ball_velocity.x
            if time_to_goal > 0:
                predicted_y = ball_position.y + (ball_velocity.y * time_to_goal)
                
                # Add reaction time delay
                threading.Timer(self.reaction_time, self._start_dive, args=(predicted_y,)).start()
    
    def _start_dive(self, predicted_y: float):
        """Start diving towards predicted ball position"""
        if self.state != "ready":
            return
        
        goal_center = self.home_position.y
        goal_top = goal_center - self.goal_height // 2
        goal_bottom = goal_center + self.goal_height // 2
        
        # Clamp prediction to goal area
        predicted_y = max(goal_top, min(goal_bottom, predicted_y))
        
        # Determine dive direction
        if predicted_y < goal_center - 20:
            self.state = "diving_up"
            self.dive_direction = Vector2D(0, -1)
        elif predicted_y > goal_center + 20:
            self.state = "diving_down"
            self.dive_direction = Vector2D(0, 1)
        else:
            self.state = "catching"
            self.dive_direction = Vector2D(0, 0)
        
        self.dive_start_time = time.time()
    
    def update(self, dt: float):
        """Update goalkeeper state"""
        current_time = time.time()
        
        if self.state in ["diving_up", "diving_down", "catching"]:
            dive_progress = (current_time - self.dive_start_time) / self.dive_duration
            
            if dive_progress >= 1.0:
                # Dive complete
                self.state = "recovering"
                dive_progress = 1.0
            
            # Update position during dive
            dive_distance = self.dive_speed * dive_progress
            self.position.x = self.home_position.x
            self.position.y = self.home_position.y + (self.dive_direction.y * dive_distance)
        
        elif self.state == "recovering":
            # Return to home position
            recovery_speed = 200
            direction = (self.home_position - self.position).normalize()
            self.position = self.position + (direction * recovery_speed * dt)
            
            if self.position.distance_to(self.home_position) < 10:
                self.state = "ready"
                self.position = self.home_position
    
    def check_save(self, ball: Ball) -> bool:
        """Check if goalkeeper saves the ball"""
        if self.state not in ["diving_up", "diving_down", "catching"]:
            return False
        
        # Check if ball is within goalkeeper's reach
        distance = self.position.distance_to(ball.position)
        if distance <= self.reach:
            self.save_made = True
            return True
        
        return False
    
    def draw(self, frame):
        """Draw goalkeeper"""
        x, y = int(self.position.x), int(self.position.y)
        
        # Draw goalkeeper shadow
        cv2.ellipse(frame, (x, frame.shape[0] - 50), (self.width//2, 10), 
                   0, 0, 360, FootballColors.BALL_SHADOW, -1)
        
        # Body
        body_rect = (x - self.width//2, y - self.height//2, self.width, self.height//2)
        cv2.rectangle(frame, body_rect[:2], 
                     (body_rect[0] + body_rect[2], body_rect[1] + body_rect[3]), 
                     FootballColors.GK_SHIRT, -1)
        
        # Shorts
        shorts_rect = (x - self.width//3, y, self.width*2//3, self.height//3)
        cv2.rectangle(frame, shorts_rect[:2], 
                     (shorts_rect[0] + shorts_rect[2], shorts_rect[1] + shorts_rect[3]), 
                     FootballColors.GK_SHORTS, -1)
        
        # Head
        cv2.circle(frame, (x, y - self.height//2 - 15), 12, FootballColors.PLAYER_SKIN, -1)
        
        # Arms and gloves (position depends on dive state)
        if self.state == "diving_up":
            # Arms up
            cv2.line(frame, (x - 15, y - 20), (x - 30, y - 50), FootballColors.PLAYER_SKIN, 8)
            cv2.line(frame, (x + 15, y - 20), (x + 30, y - 50), FootballColors.PLAYER_SKIN, 8)
            cv2.circle(frame, (x - 30, y - 50), 8, FootballColors.GK_GLOVES, -1)
            cv2.circle(frame, (x + 30, y - 50), 8, FootballColors.GK_GLOVES, -1)
        elif self.state == "diving_down":
            # Arms down
            cv2.line(frame, (x - 15, y - 10), (x - 30, y + 20), FootballColors.PLAYER_SKIN, 8)
            cv2.line(frame, (x + 15, y - 10), (x + 30, y + 20), FootballColors.PLAYER_SKIN, 8)
            cv2.circle(frame, (x - 30, y + 20), 8, FootballColors.GK_GLOVES, -1)
            cv2.circle(frame, (x + 30, y + 20), 8, FootballColors.GK_GLOVES, -1)
        else:
            # Normal arm position
            cv2.line(frame, (x - 15, y - 20), (x - 25, y - 5), FootballColors.PLAYER_SKIN, 8)
            cv2.line(frame, (x + 15, y - 20), (x + 25, y - 5), FootballColors.PLAYER_SKIN, 8)
            cv2.circle(frame, (x - 25, y - 5), 8, FootballColors.GK_GLOVES, -1)
            cv2.circle(frame, (x + 25, y - 5), 8, FootballColors.GK_GLOVES, -1)
        
        # Legs
        cv2.line(frame, (x - 10, y + self.height//3), (x - 15, y + self.height//2 + 20), FootballColors.PLAYER_SKIN, 6)
        cv2.line(frame, (x + 10, y + self.height//3), (x + 15, y + self.height//2 + 20), FootballColors.PLAYER_SKIN, 6)

class PoseDetector:
    """Full body pose detection for football movements"""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Kick detection
        self.kick_threshold = 0.15  # Minimum leg movement for kick
        self.previous_pose = None
        self.kick_detected = False
        self.kick_power = 0.0
        self.kick_direction = Vector2D(0, 0)
        self.last_kick_time = 0
        self.kick_cooldown = 1.0  # Seconds between kicks
        
    def detect_kick(self, landmarks) -> Tuple[bool, float, Vector2D]:
        """Detect kicking motion and determine power and direction"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_kick_time < self.kick_cooldown:
            return False, 0.0, Vector2D(0, 0)
        
        if not landmarks or len(landmarks.landmark) < 33:
            return False, 0.0, Vector2D(0, 0)
        
        # Get leg landmarks
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        
        # Calculate leg velocities if we have previous pose
        if self.previous_pose:
            prev_right_ankle = self.previous_pose.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            prev_left_ankle = self.previous_pose.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # Calculate ankle movement
            right_ankle_movement = Vector2D(
                right_ankle.x - prev_right_ankle.x,
                right_ankle.y - prev_right_ankle.y
            )
            
            left_ankle_movement = Vector2D(
                left_ankle.x - prev_left_ankle.x,
                left_ankle.y - prev_left_ankle.y
            )
            
            # Check for significant forward movement (kicking motion)
            right_kick_power = max(0, -right_ankle_movement.x) * 10  # Forward is negative x
            left_kick_power = max(0, -left_ankle_movement.x) * 10
            
            # Determine which leg is kicking
            if right_kick_power > self.kick_threshold:
                self.last_kick_time = current_time
                kick_direction = Vector2D(-1, right_ankle_movement.y * 2)  # Forward kick
                return True, min(1.0, right_kick_power), kick_direction.normalize()
            
            elif left_kick_power > self.kick_threshold:
                self.last_kick_time = current_time
                kick_direction = Vector2D(-1, left_ankle_movement.y * 2)  # Forward kick
                return True, min(1.0, left_kick_power), kick_direction.normalize()
        
        # Store current pose for next frame
        self.previous_pose = landmarks
        
        return False, 0.0, Vector2D(0, 0)
    
    def get_body_center(self, landmarks) -> Optional[Vector2D]:
        """Get body center position"""
        if not landmarks or len(landmarks.landmark) < 33:
            return None
        
        # Use hip midpoint as body center
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        
        center_x = (right_hip.x + left_hip.x) / 2
        center_y = (right_hip.y + left_hip.y) / 2
        
        return Vector2D(center_x, center_y)

class FootballGame:
    """Main football penalty shootout game"""
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize MediaPipe Pose
        self.pose_detector = PoseDetector()
        
        # Game configuration
        self.config = GameConfig()
        
        # Game state
        self.state = GameState.MAIN_MENU
        self.score = {"player": 0, "goalkeeper": 0}
        self.attempts = 0
        self.max_attempts = 5
        self.round_number = 1
        
        # Field setup
        self.field_y = screen_height - 100  # Ground level
        self.goal_width = 300
        self.goal_height = 200
        self.goal_x = screen_width - 150
        self.goal_y = self.field_y - self.goal_height
        
        # Game objects
        self.ball = Ball(screen_width // 4, self.field_y - 30, screen_width, screen_height)
        self.goalkeeper = Goalkeeper(
            self.goal_x, self.goal_y + self.goal_height // 2, 
            self.goal_width, self.goal_height, self.config.difficulty
        )
        
        # Player tracking
        self.player_body_center = Vector2D(screen_width // 4, screen_height // 2)
        self.pose_landmarks = None
        
        # Visual effects
        self.celebration_effects = []
        self.screen_shake = 0
        self.goal_flash = 0
        
        # UI and controls
        self.back_button = BackButton(screen_width, screen_height)
        self.menu_selection = 0
        self.show_instructions = True
        
        # Game timing
        self.game_start_time = time.time()
        self.shot_start_time = 0
        
        # Statistics
        self.total_shots = 0
        self.goals_scored = 0
        self.saves_made = 0
        
    def reset_for_new_shot(self):
        """Reset game objects for a new penalty shot"""
        self.ball.reset(self.screen_width // 4, self.field_y - 30)
        self.goalkeeper.reset()
        self.shot_start_time = time.time()
        self.screen_shake = 0
        self.goal_flash = 0
    
    def handle_shot(self, power: float, direction: Vector2D):
        """Handle player shot"""
        if self.state != GameState.PLAYING or self.ball.is_moving:
            return
        
        self.total_shots += 1
        self.attempts += 1
        
        # Apply kick to ball
        self.ball.kick(direction, power)
        
        # Goalkeeper reacts to shot
        self.goalkeeper.react_to_shot(self.ball.position, self.ball.velocity)
        
        # Visual effects
        self.screen_shake = 5
    
    def check_goal_or_save(self):
        """Check if goal was scored or saved"""
        if not self.ball.is_moving:
            return
        
        # Check goal collision
        goal_left = self.goal_x - self.goal_width // 2
        goal_right = self.goal_x + self.goal_width // 2
        goal_top = self.goal_y
        goal_bottom = self.goal_y + self.goal_height
        
        if self.ball.check_goal_collision(goal_left, goal_right, goal_top, goal_bottom):
            # Check if goalkeeper saved it
            if self.goalkeeper.check_save(self.ball):
                self.state = GameState.GOAL_SAVED
                self.saves_made += 1
            else:
                self.state = GameState.GOAL_SCORED
                self.score["player"] += 1
                self.goals_scored += 1
                self.goal_flash = 1.0
                self.add_celebration_effect()
        
        elif self.ball.out_of_bounds:
            # Missed shot
            self.state = GameState.GOAL_SAVED  # Treat as save for simplicity
    
    def add_celebration_effect(self):
        """Add celebration particle effect"""
        for _ in range(50):
            effect = {
                'pos': Vector2D(
                    random.uniform(self.goal_x - 100, self.goal_x + 100),
                    random.uniform(self.goal_y, self.goal_y + 100)
                ),
                'vel': Vector2D(
                    random.uniform(-200, 200),
                    random.uniform(-300, -100)
                ),
                'life': 1.0,
                'color': random.choice([
                    FootballColors.SUCCESS_GREEN,
                    FootballColors.SCORE_YELLOW,
                    FootballColors.TEXT_WHITE
                ])
            }
            self.celebration_effects.append(effect)
    
    def update_celebration_effects(self, dt: float):
        """Update celebration particle effects"""
        for effect in self.celebration_effects[:]:
            # Update position
            effect['pos'] = effect['pos'] + (effect['vel'] * dt)
            
            # Apply gravity
            effect['vel'].y += 500 * dt
            
            # Reduce life
            effect['life'] -= dt * 2
            
            # Remove dead effects
            if effect['life'] <= 0:
                self.celebration_effects.remove(effect)
    
    def update_visual_effects(self, dt: float):
        """Update visual effects"""
        # Screen shake
        if self.screen_shake > 0:
            self.screen_shake -= dt * 10
            self.screen_shake = max(0, self.screen_shake)
        
        # Goal flash
        if self.goal_flash > 0:
            self.goal_flash -= dt * 2
            self.goal_flash = max(0, self.goal_flash)
        
        # Update celebration effects
        self.update_celebration_effects(dt)
    
    def check_game_end(self):
        """Check if penalty shootout is complete"""
        if self.attempts >= self.max_attempts:
            self.state = GameState.GAME_OVER
    
    def start_new_game(self):
        """Start a new penalty shootout"""
        self.state = GameState.PLAYING
        self.score = {"player": 0, "goalkeeper": 0}
        self.attempts = 0
        self.round_number = 1
        self.total_shots = 0
        self.goals_scored = 0
        self.saves_made = 0
        self.reset_for_new_shot()
    
    def update(self, dt: float):
        """Main game update loop"""
        # Update game objects
        if self.state == GameState.PLAYING:
            self.ball.update(dt, self.field_y)
            self.goalkeeper.update(dt)
            self.check_goal_or_save()
        
        # Update visual effects
        self.update_visual_effects(dt)
        
        # Check for game state transitions
        if self.state in [GameState.GOAL_SCORED, GameState.GOAL_SAVED]:
            # Auto-advance after a delay
            if not self.ball.is_moving and not self.goalkeeper.state == "diving":
                threading.Timer(2.0, self._advance_to_next_shot).start()
        
        self.check_game_end()
    
    def _advance_to_next_shot(self):
        """Advance to next penalty shot"""
        if self.attempts < self.max_attempts:
            self.state = GameState.PLAYING
            self.reset_for_new_shot()
        else:
            self.state = GameState.GAME_OVER
    
    def draw_field(self, frame):
        """Draw football field"""
        # Grass field
        cv2.rectangle(frame, (0, self.field_y - 200), 
                     (self.screen_width, self.screen_height), 
                     FootballColors.GRASS_GREEN, -1)
        
        # Field texture (grass lines)
        for y in range(self.field_y - 200, self.screen_height, 20):
            grass_intensity = 0.8 + 0.2 * random.random()
            grass_color = tuple(int(c * grass_intensity) for c in FootballColors.DARK_GRASS)
            cv2.line(frame, (0, y), (self.screen_width, y), grass_color, 2)
        
        # Penalty area
        penalty_area = (
            self.goal_x - self.goal_width,
            self.goal_y - 50,
            self.goal_width * 2,
            self.goal_height + 100
        )
        cv2.rectangle(frame, penalty_area[:2], 
                     (penalty_area[0] + penalty_area[2], penalty_area[1] + penalty_area[3]), 
                     FootballColors.FIELD_LINES, 3)
        
        # Penalty spot
        penalty_spot_x = self.screen_width // 2
        penalty_spot_y = self.field_y - 20
        cv2.circle(frame, (penalty_spot_x, penalty_spot_y), 5, FootballColors.FIELD_LINES, -1)
        
        # Center circle (partial view)
        cv2.circle(frame, (self.screen_width // 4, self.field_y), 100, FootballColors.FIELD_LINES, 3)
    
    def draw_goal(self, frame):
        """Draw football goal with net"""
        goal_left = self.goal_x - self.goal_width // 2
        goal_right = self.goal_x + self.goal_width // 2
        goal_top = self.goal_y
        goal_bottom = self.goal_y + self.goal_height
        
        # Goal posts
        cv2.rectangle(frame, (goal_left - 5, goal_top), 
                     (goal_left, goal_bottom), FootballColors.GOAL_WHITE, -1)
        cv2.rectangle(frame, (goal_right, goal_top), 
                     (goal_right + 5, goal_bottom), FootballColors.GOAL_WHITE, -1)
        cv2.rectangle(frame, (goal_left, goal_top - 5), 
                     (goal_right, goal_top), FootballColors.GOAL_WHITE, -1)
        
        # Net pattern
        net_spacing = 20
        for x in range(goal_left, goal_right, net_spacing):
            cv2.line(frame, (x, goal_top), (x, goal_bottom), FootballColors.NET_COLOR, 1)
        for y in range(goal_top, goal_bottom, net_spacing):
            cv2.line(frame, (goal_left, y), (goal_right, y), FootballColors.NET_COLOR, 1)
        
        # Goal flash effect
        if self.goal_flash > 0:
            flash_overlay = frame.copy()
            flash_color = tuple(int(c * self.goal_flash) for c in FootballColors.SUCCESS_GREEN)
            cv2.rectangle(flash_overlay, (goal_left - 20, goal_top - 20), 
                         (goal_right + 20, goal_bottom + 20), flash_color, -1)
            cv2.addWeighted(frame, 1.0 - self.goal_flash * 0.3, flash_overlay, self.goal_flash * 0.3, 0, frame)
    
    def draw_player_skeleton(self, frame):
        """Draw player pose skeleton"""
        if not self.pose_landmarks or not self.config.show_pose_landmarks:
            return
        
        # Draw pose landmarks
        self.pose_detector.mp_drawing.draw_landmarks(
            frame, self.pose_landmarks, self.pose_detector.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.pose_detector.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=self.pose_detector.mp_drawing.DrawingSpec(
                color=(255, 255, 0), thickness=2, circle_radius=2)
        )
    
    def draw_hud(self, frame):
        """Draw heads-up display"""
        if self.state not in [GameState.PLAYING, GameState.GOAL_SCORED, GameState.GOAL_SAVED]:
            return
        
        # Apply screen shake
        shake_x = int(random.uniform(-self.screen_shake, self.screen_shake))
        shake_y = int(random.uniform(-self.screen_shake, self.screen_shake))
        
        # Score display
        score_text = f"GOALS: {self.score['player']} / {self.max_attempts}"
        cv2.putText(frame, score_text, (50 + shake_x, 80 + shake_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, FootballColors.SCORE_YELLOW, 3)
        
        # Attempts counter
        attempts_text = f"SHOT {self.attempts + 1} OF {self.max_attempts}"
        if self.attempts < self.max_attempts:
            cv2.putText(frame, attempts_text, (50 + shake_x, 120 + shake_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, FootballColors.TEXT_WHITE, 2)
        
        # Round number
        round_text = f"ROUND {self.round_number}"
        cv2.putText(frame, round_text, (50 + shake_x, 50 + shake_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, FootballColors.TEXT_WHITE, 2)
        
        # Instructions
        if self.show_instructions and self.state == GameState.PLAYING and not self.ball.is_moving:
            instruction_text = "KICK THE BALL TO SHOOT!"
            inst_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            inst_x = (self.screen_width - inst_size[0]) // 2
            cv2.putText(frame, instruction_text, (inst_x + shake_x, 200 + shake_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, FootballColors.SCORE_YELLOW, 3)
        
        # Game state messages
        if self.state == GameState.GOAL_SCORED:
            goal_text = "GOAL!"
            goal_size = cv2.getTextSize(goal_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
            goal_x = (self.screen_width - goal_size[0]) // 2
            cv2.putText(frame, goal_text, (goal_x + shake_x, self.screen_height // 2 + shake_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, FootballColors.SUCCESS_GREEN, 5)
        
        elif self.state == GameState.GOAL_SAVED:
            save_text = "SAVED!" if self.goalkeeper.save_made else "MISSED!"
            save_color = FootballColors.DANGER_RED
            save_size = cv2.getTextSize(save_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
            save_x = (self.screen_width - save_size[0]) // 2
            cv2.putText(frame, save_text, (save_x + shake_x, self.screen_height // 2 + shake_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, save_color, 5)
        
        # Statistics
        if self.total_shots > 0:
            accuracy = (self.goals_scored / self.total_shots) * 100
            stats_text = f"ACCURACY: {accuracy:.1f}%"
            cv2.putText(frame, stats_text, (self.screen_width - 300 + shake_x, 50 + shake_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, FootballColors.TEXT_WHITE, 2)
    
    def draw_main_menu(self, frame):
        """Draw main menu"""
        # Background
        frame[:] = (20, 50, 20)  # Dark green
        
        # Title
        title_text = "PENALTY SHOOTOUT"
        title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        title_y = 200
        
        # Animated title
        glow_intensity = 0.5 + 0.5 * math.sin(time.time() * 2)
        glow_color = tuple(int(c * glow_intensity) for c in FootballColors.SCORE_YELLOW)
        
        cv2.putText(frame, title_text, (title_x + 4, title_y + 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3, glow_color, 6)
        cv2.putText(frame, title_text, (title_x, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3, FootballColors.TEXT_WHITE, 4)
        
        # Menu options
        menu_options = [
            "START GAME",
            "SETTINGS",
            "EXIT"
        ]
        
        for i, option in enumerate(menu_options):
            option_y = 400 + (i * 80)
            color = FootballColors.SCORE_YELLOW if i == self.menu_selection else FootballColors.TEXT_WHITE
            
            option_size = cv2.getTextSize(option, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            option_x = (self.screen_width - option_size[0]) // 2
            
            cv2.putText(frame, option, (option_x, option_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        # Instructions
        instructions = [
            "FULL BODY MOTION CONTROLS:",
            "• Stand back from camera for full body tracking",
            "• Kick with your leg to shoot the ball",
            "• Move your body to aim left/right",
            "• Try to score past the goalkeeper!"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_y = 700 + (i * 40)
            font_size = 1.2 if i == 0 else 1.0
            color = FootballColors.SCORE_YELLOW if i == 0 else FootballColors.TEXT_WHITE
            
            inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
            inst_x = (self.screen_width - inst_size[0]) // 2
            
            cv2.putText(frame, instruction, (inst_x, inst_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)
    
    def draw_game_over(self, frame):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Results
        final_score = self.score["player"]
        max_score = self.max_attempts
        
        if final_score >= max_score * 0.8:
            result_text = "EXCELLENT!"
            result_color = FootballColors.SUCCESS_GREEN
        elif final_score >= max_score * 0.6:
            result_text = "GOOD JOB!"
            result_color = FootballColors.SCORE_YELLOW
        else:
            result_text = "KEEP PRACTICING!"
            result_color = FootballColors.DANGER_RED
        
        # Draw result
        result_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 4)[0]
        result_x = (self.screen_width - result_size[0]) // 2
        result_y = self.screen_height // 2 - 100
        
        cv2.putText(frame, result_text, (result_x, result_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4, result_color, 4)
        
        # Final statistics
        stats_y = result_y + 100
        stats = [
            f"GOALS SCORED: {final_score}/{max_score}",
            f"SUCCESS RATE: {(final_score/max_score)*100:.1f}%",
            f"TOTAL SHOTS: {self.total_shots}",
            f"GOALKEEPER SAVES: {self.saves_made}"
        ]
        
        for i, stat in enumerate(stats):
            stat_size = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
            stat_x = (self.screen_width - stat_size[0]) // 2
            stat_y = stats_y + (i * 50)
            
            cv2.putText(frame, stat, (stat_x, stat_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, FootballColors.TEXT_WHITE, 2)
        
        # Continue prompt
        continue_text = "Press any key to return to menu"
        continue_size = cv2.getTextSize(continue_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        continue_x = (self.screen_width - continue_size[0]) // 2
        continue_y = stats_y + len(stats) * 50 + 80
        
        cv2.putText(frame, continue_text, (continue_x, continue_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, FootballColors.TEXT_WHITE, 2)
    
    def draw(self, frame):
        """Main drawing function"""
        if self.state == GameState.MAIN_MENU:
            self.draw_main_menu(frame)
        
        elif self.state in [GameState.PLAYING, GameState.GOAL_SCORED, GameState.GOAL_SAVED]:
            # Draw field
            self.draw_field(frame)
            
            # Draw goal
            self.draw_goal(frame)
            
            # Draw game objects
            self.goalkeeper.draw(frame)
            self.ball.draw(frame)
            
            # Draw pose skeleton
            self.draw_player_skeleton(frame)
            
            # Draw celebration effects
            for effect in self.celebration_effects:
                if effect['life'] > 0:
                    alpha = effect['life']
                    color = tuple(int(c * alpha) for c in effect['color'])
                    x, y = int(effect['pos'].x), int(effect['pos'].y)
                    cv2.circle(frame, (x, y), max(2, int(8 * alpha)), color, -1)
            
            # Draw HUD
            self.draw_hud(frame)
        
        elif self.state == GameState.GAME_OVER:
            self.draw_game_over(frame)
        
        # Draw back button
        hand_pos = None
        if self.pose_landmarks:
            body_center = self.pose_detector.get_body_center(self.pose_landmarks)
            if body_center:
                hand_pos = (int(body_center.x * self.screen_width), 
                           int(body_center.y * self.screen_height))
        
        self.back_button.draw(frame, hand_pos)

def main():
    """Main function to run the Football Game"""
    parser = argparse.ArgumentParser(description='Full-Body Football Penalty Shootout Game')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    parser.add_argument('--difficulty', type=str, default='medium', 
                       choices=['easy', 'medium', 'hard', 'legendary'],
                       help='Goalkeeper difficulty level')
    parser.add_argument('--show-pose', action='store_true', help='Show pose landmarks')
    parser.add_argument('--windowed', action='store_true', help='Run in windowed mode')
    args = parser.parse_args()

    # Set up OpenTelemetry tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {args.camera}")
        exit(-1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Initialize game
    game = FootballGame(1920, 1080)
    game.config.difficulty = DifficultyLevel[args.difficulty.upper()]
    game.config.show_pose_landmarks = args.show_pose
    game.config.fullscreen = not args.windowed

    # --- Pygame Setup ---
    pygame.init()
    WIDTH, HEIGHT = 1920, 1080
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Football Penalty Shootout - Full Body Control (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    clock = pygame.time.Clock()

    print("⚽ FOOTBALL PENALTY SHOOTOUT - FULL BODY EDITION ⚽")
    print("=" * 70)
    print("🎯 FULL BODY MOTION CONTROLS 🎯")
    print()
    print("📋 SETUP INSTRUCTIONS:")
    print("   📷 Position camera 6-8 feet away from you")
    print("   🧍 Stand where camera can see your full body")
    print("   ⚽ Face the camera, ball will be on your left")
    print()
    print("🦵 GAME CONTROLS:")
    print("   ⚽ KICK MOTION → Shoot the ball")
    print("   📍 BODY POSITION → Aim direction (left/right)")
    print("   💨 KICK SPEED → Shot power")
    print("   🔙 BACK BUTTON → Exit to app store")
    print()
    print("🥅 OBJECTIVE:")
    print("   • Score as many goals as possible")
    print("   • Beat the AI goalkeeper")
    print("   • Complete 5 penalty shots")
    print()
    print(f"🎮 DIFFICULTY: {args.difficulty.upper()}")
    print("=" * 70)

    last_time = time.time()
    with tracer.start_as_current_span("footandshoot_session"):
        running = True
        while running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1920, 1080))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = game.pose_detector.pose.process(rgb_frame)
            game.pose_landmarks = results.pose_landmarks
            if results.pose_landmarks:
                body_center = game.pose_detector.get_body_center(results.pose_landmarks)
                if body_center:
                    game.player_body_center = Vector2D(
                        body_center.x * game.screen_width,
                        body_center.y * game.screen_height
                    )
                if game.state == GameState.PLAYING:
                    kick_detected, power, direction = game.pose_detector.detect_kick(results.pose_landmarks)
                    if kick_detected:
                        with tracer.start_as_current_span("kick"):
                            game.handle_shot(power, direction)
            # Update game state
            game.update(dt)
            # Draw everything on the frame (OpenCV drawing)
            game.draw(frame)
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            game.start_new_game()
            # --- Pygame Rendering ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
