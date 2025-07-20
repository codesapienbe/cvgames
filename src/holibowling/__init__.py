import cv2
import mediapipe as mp
import numpy as np
import argparse
import sys
import os
import math
import random
import time
import logging
from enum import Enum
from collections import deque

# Import the back button system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cvstore'))
from back_button import BackButton

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Game constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
PLAYING_AREA_WIDTH = 400
PLAYING_AREA_HEIGHT = 600
FAULT_LINE_Y = SCREEN_HEIGHT - 300
BALL_RADIUS = 15
TARGET_BOX_SIZE = 80
STANDING_ZONE_HEIGHT = 200  # Area where player should stand

class GameState(Enum):
    SETUP = "setup"
    POSITIONING = "positioning"
    STANDING_READY = "standing_ready"
    AIMING = "aiming"
    THROWING = "throwing"
    BALL_IN_FLIGHT = "ball_in_flight"
    SCORING = "scoring"
    GAME_OVER = "game_over"

class BowlingScore:
    """Enhanced bowling scoring system with strikes and spares"""
    def __init__(self):
        self.frames = [[] for _ in range(10)]
        self.current_frame = 0
        self.current_roll = 0
        self.total_score = 0
        
    def add_roll(self, pins_hit):
        """Add a roll to the current frame"""
        try:
            if self.current_frame < 10:
                self.frames[self.current_frame].append(pins_hit)
                
                if len(self.frames[self.current_frame]) == 1:
                    # First roll
                    if pins_hit == 10:  # Strike
                        self.current_frame += 1
                    else:
                        self.current_roll = 1
                else:
                    # Second roll or third roll (10th frame)
                    if self.current_frame == 9:  # 10th frame
                        if len(self.frames[9]) < 3 and (self.frames[9][0] == 10 or sum(self.frames[9][:2]) == 10):
                            # Strike or spare in 10th frame - get third roll
                            pass
                        else:
                            self.current_frame += 1
                    else:
                        self.current_frame += 1
                        self.current_roll = 0
        except Exception as e:
            logger.error(f"Error in add_roll: {e}")
            # Fallback: just increment frame
            self.current_frame = min(self.current_frame + 1, 10)
                    
    def calculate_total_score(self):
        """Calculate total score with strikes and spares"""
        try:
            total = 0
            for i in range(10):
                if i < len(self.frames) and self.frames[i]:
                    frame_score = sum(self.frames[i])
                    
                    # Check for strike
                    if len(self.frames[i]) > 0 and self.frames[i][0] == 10:
                        # Add next two rolls
                        bonus = 0
                        if i + 1 < len(self.frames) and self.frames[i + 1]:
                            bonus += self.frames[i + 1][0]
                            if len(self.frames[i + 1]) > 1:
                                bonus += self.frames[i + 1][1]
                            elif i + 2 < len(self.frames) and self.frames[i + 2]:
                                bonus += self.frames[i + 2][0]
                        frame_score += bonus
                        
                    # Check for spare
                    elif len(self.frames[i]) > 1 and sum(self.frames[i][:2]) == 10:
                        # Add next roll
                        if i + 1 < len(self.frames) and self.frames[i + 1]:
                            frame_score += self.frames[i + 1][0]
                            
                    total += frame_score
                    
            return total
        except Exception as e:
            logger.error(f"Error calculating score: {e}")
            return sum(sum(frame) for frame in self.frames if frame)

class StandingBowlingGame:
    def __init__(self):
        self.state = GameState.SETUP
        self.score_system = BowlingScore()
        self.ball_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50]
        self.ball_velocity = [0, 0]
        self.ball_thrown = False
        self.ball_in_hand = True
        self.throw_power = 0
        self.max_throw_power = 25
        self.aim_angle = 0
        self.target_box_pos = [SCREEN_WIDTH // 2, 150]
        self.gesture_history = deque(maxlen=10)
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5
        self.error_message = ""
        self.error_timer = 0
        self.fault_line_violation = False
        self.player_position = [SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100]  # Player's standing position
        self.shoulder_angle = 0  # Real-time shoulder angle for aiming
        self.is_player_standing = False
        self.is_player_in_position = False
        
    def detect_player_position(self, pose_landmarks):
        """Detect if player is standing and in correct position using holistic landmarks"""
        try:
            if not pose_landmarks:
                return False, [0, 0]
                
            # Get key body landmarks for standing detection
            left_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            left_hip = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]
            left_ankle = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE]
            right_ankle = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE]
            
            # Calculate player position (center of shoulders)
            player_x = (left_shoulder.x + right_shoulder.x) / 2 * SCREEN_WIDTH
            player_y = (left_shoulder.y + right_shoulder.y) / 2 * SCREEN_HEIGHT
            
            # Check if player is standing (shoulders above hips, hips above ankles)
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            ankle_y = (left_ankle.y + right_ankle.y) / 2
            
            is_standing = (shoulder_y < hip_y < ankle_y) and (hip_y - shoulder_y > 0.1)
            
            # Check if player is in the standing zone
            standing_zone_top = SCREEN_HEIGHT - STANDING_ZONE_HEIGHT
            standing_zone_bottom = SCREEN_HEIGHT - 50
            in_position = (standing_zone_top <= player_y <= standing_zone_bottom)
            
            return is_standing, [player_x, player_y]
        except Exception as e:
            logger.error(f"Error detecting player position: {e}")
            return False, [0, 0]
    
    def calculate_shoulder_aiming_angle(self, pose_landmarks):
        """Calculate real-time aiming angle based on shoulder orientation"""
        try:
            if not pose_landmarks:
                return 0
                
            # Get shoulder positions
            left_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            
            # Calculate shoulder line angle
            shoulder_dx = right_shoulder.x - left_shoulder.x
            shoulder_dy = right_shoulder.y - left_shoulder.y
            
            # Calculate angle in degrees
            shoulder_angle = math.degrees(math.atan2(shoulder_dy, shoulder_dx))
            
            # Convert to aiming angle (0 = straight ahead, negative = left, positive = right)
            # Normalize to -45 to +45 degrees
            aiming_angle = np.clip(shoulder_angle * 2, -45, 45)
            
            return aiming_angle
        except Exception as e:
            logger.error(f"Error calculating shoulder angle: {e}")
            return 0
    
    def detect_throw_gesture(self, pose_landmarks, hand_landmarks):
        """Detect bowling throw gesture with proper motion tracking"""
        try:
            if not pose_landmarks or not hand_landmarks:
                return None
                
            # Get key landmarks for throw detection
            right_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
            right_wrist = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
            
            # Calculate arm angles
            arm_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Get hand position for ball tracking
            hand_x = right_wrist.x * SCREEN_WIDTH
            hand_y = right_wrist.y * SCREEN_HEIGHT
            
            # Track ball position to follow hand
            if self.ball_in_hand:
                self.ball_pos = [hand_x, hand_y]
            
            # Add to gesture history
            current_time = time.time()
            self.gesture_history.append((arm_angle, hand_x, hand_y, current_time))
            
            # Detect throw motion - simplified and more responsive
            if len(self.gesture_history) >= 3:
                recent_data = list(self.gesture_history)[-3:]
                angles = [data[0] for data in recent_data]
                hand_positions = [(data[1], data[2]) for data in recent_data]
                
                # Calculate motion characteristics
                angle_change = max(angles) - min(angles)
                hand_movement = math.sqrt(
                    (hand_positions[-1][0] - hand_positions[0][0])**2 + 
                    (hand_positions[-1][1] - hand_positions[0][1])**2
                )
                
                # Detect throw gesture based on arm motion
                if arm_angle < 120 and angle_change > 20 and hand_movement > 30:
                    return "throw"
                elif arm_angle > 140 and angle_change < 15:
                    return "windup"
                    
            return None
        except Exception as e:
            logger.error(f"Error in detect_throw_gesture: {e}")
            self.show_error("Please ensure your right arm is visible")
            return None
    
    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        try:
            a = np.array([point1.x, point1.y])
            b = np.array([point2.x, point2.y])
            c = np.array([point3.x, point3.y])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            
            return np.degrees(angle)
        except Exception as e:
            logger.error(f"Error calculating angle: {e}")
            return 90.0
    
    def check_fault_line_violation(self, hand_y):
        """Check if player crossed the fault line"""
        if hand_y < FAULT_LINE_Y:
            self.fault_line_violation = True
            return True
        return False
    
    def show_error(self, message):
        """Show error message to player"""
        self.error_message = message
        self.error_timer = 120  # Show for 2 seconds
    
    def update_ball_physics(self):
        """Update ball physics when thrown"""
        try:
            if not self.ball_thrown:
                return "in_hand"
            
            # Apply gravity
            self.ball_velocity[1] += 0.4
            
            # Update position
            self.ball_pos[0] += self.ball_velocity[0]
            self.ball_pos[1] += self.ball_velocity[1]
            
            # Check if ball hits target box
            target_left = self.target_box_pos[0] - TARGET_BOX_SIZE // 2
            target_right = self.target_box_pos[0] + TARGET_BOX_SIZE // 2
            target_top = self.target_box_pos[1] - TARGET_BOX_SIZE // 2
            target_bottom = self.target_box_pos[1] + TARGET_BOX_SIZE // 2
            
            if (target_left <= self.ball_pos[0] <= target_right and 
                target_top <= self.ball_pos[1] <= target_bottom):
                return "hit_target"
            
            # Check if ball is out of bounds
            if (self.ball_pos[1] > SCREEN_HEIGHT + 50 or 
                self.ball_pos[0] < 0 or 
                self.ball_pos[0] > SCREEN_WIDTH):
                return "ball_out"
                
            return "in_flight"
        except Exception as e:
            logger.error(f"Error in update_ball_physics: {e}")
            return "ball_out"
    
    def calculate_score(self):
        """Calculate score based on target hit"""
        try:
            if self.fault_line_violation:
                return 0  # Fault line violation = 0 points
            
            # Calculate distance from target center
            distance = math.sqrt(
                (self.ball_pos[0] - self.target_box_pos[0])**2 + 
                (self.ball_pos[1] - self.target_box_pos[1])**2
            )
            
            # Score based on accuracy
            if distance < TARGET_BOX_SIZE // 4:
                return 10  # Perfect hit = strike
            elif distance < TARGET_BOX_SIZE // 2:
                return 7   # Good hit
            elif distance < TARGET_BOX_SIZE:
                return 4   # Decent hit
            else:
                return 1   # Miss
        except Exception as e:
            logger.error(f"Error calculating score: {e}")
            return 0
    
    def reset_ball(self):
        """Reset ball to hand position"""
        try:
            self.ball_thrown = False
            self.ball_in_hand = True
            self.ball_velocity = [0, 0]
            self.throw_power = 0
            self.fault_line_violation = False
        except Exception as e:
            logger.error(f"Error resetting ball: {e}")
    
    def next_roll(self):
        """Move to next roll or frame"""
        try:
            pins_hit = self.calculate_score()
            self.score_system.add_roll(pins_hit)
            
            if self.score_system.current_frame >= 10:
                self.state = GameState.GAME_OVER
            else:
                if len(self.score_system.frames[self.score_system.current_frame]) == 1:
                    if self.score_system.frames[self.score_system.current_frame][0] == 10:
                        # Strike - move to next frame
                        self.score_system.current_frame += 1
                    else:
                        # Second roll in same frame
                        pass
                else:
                    # Frame complete
                    self.score_system.current_frame += 1
                
                self.state = GameState.POSITIONING
                self.reset_ball()
        except Exception as e:
            logger.error(f"Error in next_roll: {e}")
            self.state = GameState.POSITIONING
            self.reset_ball()

def draw_ar_playing_area(image):
    """Draw AR playing area with standing zone and fault line"""
    # Create overlay for AR effect
    overlay = image.copy()
    
    # Draw playing area boundaries
    area_left = (SCREEN_WIDTH - PLAYING_AREA_WIDTH) // 2
    area_right = (SCREEN_WIDTH + PLAYING_AREA_WIDTH) // 2
    area_top = 50
    area_bottom = SCREEN_HEIGHT - 50
    
    # Draw playing area background (semi-transparent)
    cv2.rectangle(overlay, (area_left, area_top), (area_right, area_bottom), (0, 100, 0), -1)
    cv2.rectangle(overlay, (area_left, area_top), (area_right, area_bottom), (255, 255, 255), 3)
    
    # Draw standing zone (where player should stand)
    standing_zone_top = SCREEN_HEIGHT - STANDING_ZONE_HEIGHT
    standing_zone_bottom = SCREEN_HEIGHT - 50
    cv2.rectangle(overlay, (area_left, standing_zone_top), (area_right, standing_zone_bottom), (0, 255, 0), 2)
    cv2.putText(overlay, "STANDING ZONE", (area_left + 50, standing_zone_top - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw fault line (critical boundary)
    cv2.line(overlay, (area_left, FAULT_LINE_Y), (area_right, FAULT_LINE_Y), (255, 0, 0), 5)
    cv2.putText(overlay, "FAULT LINE - DO NOT CROSS", 
               (area_left + 50, FAULT_LINE_Y - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Draw approach arrows
    for i in range(3):
        arrow_x = SCREEN_WIDTH // 2 + (i - 1) * 50
        arrow_y = standing_zone_bottom - 30
        cv2.arrowedLine(overlay, (arrow_x, arrow_y), (arrow_x, arrow_y - 40), 
                       (255, 255, 255), 3, tipLength=0.3)
    
    # Blend with original image for AR effect
    alpha = 0.4
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return image

def draw_ar_target_box(image, target_pos, size):
    """Draw AR target box"""
    # Create overlay for AR effect
    overlay = image.copy()
    
    # Draw target box
    left = int(target_pos[0] - size // 2)
    right = int(target_pos[0] + size // 2)
    top = int(target_pos[1] - size // 2)
    bottom = int(target_pos[1] + size // 2)
    
    # Draw box with different colors for scoring zones
    cv2.rectangle(overlay, (left, top), (right, bottom), (255, 255, 255), 3)  # Outer border
    
    # Inner scoring zones
    inner_size = size // 2
    inner_left = int(target_pos[0] - inner_size // 2)
    inner_right = int(target_pos[0] + inner_size // 2)
    inner_top = int(target_pos[1] - inner_size // 2)
    inner_bottom = int(target_pos[1] + inner_size // 2)
    
    cv2.rectangle(overlay, (inner_left, inner_top), (inner_right, inner_bottom), (0, 255, 0), 2)
    
    # Center bullseye
    center_size = size // 4
    center_left = int(target_pos[0] - center_size // 2)
    center_right = int(target_pos[0] + center_size // 2)
    center_top = int(target_pos[1] - center_size // 2)
    center_bottom = int(target_pos[1] + center_size // 2)
    
    cv2.rectangle(overlay, (center_left, center_top), (center_right, center_bottom), (0, 0, 255), -1)
    
    # Add target label
    cv2.putText(overlay, "TARGET", (left, top - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Blend with original image for AR effect
    alpha = 0.8
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return image

def draw_ar_ball(image, ball_pos, radius, in_hand=True):
    """Draw AR ball that follows hand or flies independently"""
    # Create overlay for AR effect
    overlay = image.copy()
    
    x, y = int(ball_pos[0]), int(ball_pos[1])
    
    if in_hand:
        # Ball in hand - more opaque
        cv2.circle(overlay, (x, y), radius, (0, 0, 150), -1)  # Dark blue
        cv2.circle(overlay, (x, y), radius, (255, 255, 255), 2)  # White outline
        
        # Draw finger holes
        hole_offset = radius // 3
        cv2.circle(overlay, (x - hole_offset, y - hole_offset), radius // 6, (0, 0, 0), -1)
        cv2.circle(overlay, (x + hole_offset, y - hole_offset), radius // 6, (0, 0, 0), -1)
        cv2.circle(overlay, (x, y + hole_offset), radius // 6, (0, 0, 0), -1)
        
        # Add "BALL" label
        cv2.putText(overlay, "BALL", (x - 20, y + radius + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        alpha = 0.9
    else:
        # Ball in flight - slightly transparent
        cv2.circle(overlay, (x, y), radius, (0, 0, 200), -1)  # Bright blue
        cv2.circle(overlay, (x, y), radius, (255, 255, 255), 2)  # White outline
        
        # Add motion blur effect
        cv2.circle(overlay, (x - 5, y - 5), radius // 2, (100, 100, 255), -1)
        
        alpha = 0.7
    
    # Blend with original image for AR effect
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return image

def draw_player_position_indicator(image, player_pos, is_standing, is_in_position):
    """Draw player position indicator"""
    x, y = int(player_pos[0]), int(player_pos[1])
    
    # Draw player position circle
    if is_standing and is_in_position:
        color = (0, 255, 0)  # Green - good position
        text = "PLAYER READY"
    elif is_standing:
        color = (0, 255, 255)  # Yellow - standing but wrong position
        text = "MOVE TO ZONE"
    else:
        color = (0, 0, 255)  # Red - not standing
        text = "STAND UP"
    
    cv2.circle(image, (x, y), 15, color, -1)
    cv2.circle(image, (x, y), 15, (255, 255, 255), 2)
    cv2.putText(image, text, (x - 50, y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Standing AR Bowling Game')
        parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        args = parser.parse_args()

        # Initialize camera
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            logger.error(f"Could not open camera with index {args.camera}")
            print("Error: Could not open camera. Please check your camera connection.")
            return

        cap.set(3, SCREEN_WIDTH)
        cap.set(4, SCREEN_HEIGHT)

        # Initialize back button
        back_button = BackButton(SCREEN_WIDTH, SCREEN_HEIGHT)

        # Initialize game
        game = StandingBowlingGame()

        # Initialize MediaPipe Holistic
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        ) as holistic:
            
            while cap.isOpened():
                try:
                    success, image = cap.read()
                    if not success:
                        logger.warning("Ignoring empty camera frame.")
                        continue

                    # Keep original image for AR overlay
                    image_raw = image.copy()

                    # Process with MediaPipe
                    image.flags.writeable = False
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    image.flags.writeable = True

                    # Handle back button
                    hand_position = None
                    hand_landmarks = None
                    try:
                        if results.left_hand_landmarks:  # Use left hand for back button
                            landmarks = results.left_hand_landmarks.landmark
                            if landmarks:
                                hand_position = [
                                    int(landmarks[9].x * SCREEN_WIDTH),
                                    int(landmarks[9].y * SCREEN_HEIGHT)
                                ]
                                hand_landmarks = type('HandLandmarks', (), {
                                    'landmark': [type('Landmark', (), {
                                        'x': lm.x,
                                        'y': lm.y
                                    })() for lm in landmarks]
                                })()

                        if hand_landmarks and back_button.handle_input(None, hand_landmarks, hand_position):
                            logger.info("User approved exit - returning to app store")
                            break
                    except Exception as e:
                        logger.error(f"Error handling back button: {e}")

                    # Detect player position and status
                    if results.pose_landmarks:
                        game.is_player_standing, game.player_position = game.detect_player_position(results.pose_landmarks)
                        game.shoulder_angle = game.calculate_shoulder_aiming_angle(results.pose_landmarks)
                        
                        # Check if player is in standing zone
                        standing_zone_top = SCREEN_HEIGHT - STANDING_ZONE_HEIGHT
                        standing_zone_bottom = SCREEN_HEIGHT - 50
                        game.is_player_in_position = (standing_zone_top <= game.player_position[1] <= standing_zone_bottom)

                    # Game logic based on state
                    try:
                        if game.state == GameState.SETUP:
                            game.state = GameState.POSITIONING
                            
                        elif game.state == GameState.POSITIONING:
                            # Guide player to standing position
                            if not game.is_player_standing:
                                cv2.putText(image, "Please STAND UP", 
                                           (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT - 100), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif not game.is_player_in_position:
                                cv2.putText(image, "Move to the STANDING ZONE", 
                                           (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT - 100), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            else:
                                game.state = GameState.STANDING_READY
                                
                        elif game.state == GameState.STANDING_READY:
                            # Player is standing and in position
                            cv2.putText(image, "READY TO BOWL - Raise your arm to start", 
                                       (SCREEN_WIDTH//2 - 250, SCREEN_HEIGHT - 100), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            # Check for windup gesture to start aiming
                            if results.pose_landmarks and results.right_hand_landmarks:
                                throw_gesture = game.detect_throw_gesture(
                                    results.pose_landmarks, 
                                    results.right_hand_landmarks
                                )
                                
                                if throw_gesture == "windup":
                                    game.state = GameState.AIMING
                                    
                        elif game.state == GameState.AIMING:
                            # Real-time aiming based on shoulder angle
                            game.aim_angle = game.shoulder_angle
                            
                            # Draw aiming line from player position
                            aim_start = (int(game.player_position[0]), int(game.player_position[1]))
                            aim_end = (
                                int(aim_start[0] + math.sin(math.radians(game.aim_angle)) * 300),
                                int(aim_start[1] - math.cos(math.radians(game.aim_angle)) * 300)
                            )
                            cv2.line(image, aim_start, aim_end, (0, 255, 0), 3)
                            
                            # Draw aim target
                            cv2.circle(image, aim_end, 5, (0, 255, 0), -1)
                            
                            # Check fault line violation
                            if results.pose_landmarks:
                                right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
                                hand_y = right_wrist.y * SCREEN_HEIGHT
                                game.check_fault_line_violation(hand_y)
                            
                            # Detect throw gesture
                            throw_gesture = game.detect_throw_gesture(
                                results.pose_landmarks, 
                                results.right_hand_landmarks
                            )
                            
                            # Show current gesture and aiming info
                            if throw_gesture:
                                cv2.putText(image, f"Gesture: {throw_gesture}", 
                                           (50, SCREEN_HEIGHT - 200), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
                            cv2.putText(image, f"Aim Angle: {game.aim_angle:.1f}°", 
                                       (50, SCREEN_HEIGHT - 250), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            if throw_gesture == "windup":
                                game.state = GameState.THROWING
                                
                        elif game.state == GameState.THROWING:
                            # Build throw power
                            game.throw_power = min(game.throw_power + 0.5, game.max_throw_power)
                            
                            # Update aim angle in real-time
                            game.aim_angle = game.shoulder_angle
                            
                            # Detect release gesture
                            throw_gesture = game.detect_throw_gesture(
                                results.pose_landmarks, 
                                results.right_hand_landmarks
                            )
                            
                            # Show current gesture and power
                            if throw_gesture:
                                cv2.putText(image, f"Gesture: {throw_gesture}", 
                                           (50, SCREEN_HEIGHT - 200), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
                            cv2.putText(image, f"Power: {int(game.throw_power)}", 
                                       (50, SCREEN_HEIGHT - 250), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            if throw_gesture == "throw" and game.throw_power > 3:
                                # Calculate ball velocity based on current aim angle and power
                                power_factor = game.throw_power / game.max_throw_power
                                game.ball_velocity[0] = math.sin(math.radians(game.aim_angle)) * power_factor * 15
                                game.ball_velocity[1] = -math.cos(math.radians(game.aim_angle)) * power_factor * 15
                                
                                game.ball_thrown = True
                                game.ball_in_hand = False
                                game.state = GameState.BALL_IN_FLIGHT
                                
                        elif game.state == GameState.BALL_IN_FLIGHT:
                            # Update ball physics
                            ball_status = game.update_ball_physics()
                            
                            if ball_status in ["hit_target", "ball_out"]:
                                game.state = GameState.SCORING
                                
                        elif game.state == GameState.SCORING:
                            # Calculate score and update
                            pins_hit = game.calculate_score()
                            
                            # Move to next roll
                            game.next_roll()
                            
                        elif game.state == GameState.GAME_OVER:
                            # Game over state
                            pass
                    except Exception as e:
                        logger.error(f"Error in game logic: {e}")
                        game.show_error("Game error - please restart")

                    # Draw AR elements
                    try:
                        # Draw playing area
                        image = draw_ar_playing_area(image)
                        
                        # Draw target box
                        image = draw_ar_target_box(image, game.target_box_pos, TARGET_BOX_SIZE)
                        
                        # Draw ball
                        image = draw_ar_ball(image, game.ball_pos, BALL_RADIUS, game.ball_in_hand)
                        
                        # Draw player position indicator
                        if results.pose_landmarks:
                            draw_player_position_indicator(image, game.player_position, 
                                                         game.is_player_standing, game.is_player_in_position)
                        
                    except Exception as e:
                        logger.error(f"Error drawing AR elements: {e}")
                    
                    # Draw UI elements
                    try:
                        # Draw power meter
                        if game.state == GameState.THROWING:
                            power_width = int((game.throw_power / game.max_throw_power) * 200)
                            cv2.rectangle(image, (50, SCREEN_HEIGHT - 100), (250, SCREEN_HEIGHT - 80), (100, 100, 100), -1)
                            cv2.rectangle(image, (50, SCREEN_HEIGHT - 100), (50 + power_width, SCREEN_HEIGHT - 80), (0, 255, 0), -1)
                            cv2.putText(image, f"Power: {int(game.throw_power)}", (50, SCREEN_HEIGHT - 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Draw score display
                        total_score = game.score_system.calculate_total_score()
                        cv2.putText(image, f"Total Score: {total_score}", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(image, f"Frame: {game.score_system.current_frame + 1}/10", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Draw fault line violation warning
                        if game.fault_line_violation:
                            cv2.putText(image, "FAULT LINE VIOLATION!", 
                                       (SCREEN_WIDTH//2 - 150, 200), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        
                        # Draw error message
                        if game.error_message and game.error_timer > 0:
                            cv2.putText(image, game.error_message, 
                                       (SCREEN_WIDTH//2 - 200, 250), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            game.error_timer -= 1
                            if game.error_timer <= 0:
                                game.error_message = ""
                        
                        # Draw instructions
                        if game.state == GameState.POSITIONING:
                            if not game.is_player_standing:
                                cv2.putText(image, "STAND UP to play", 
                                           (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT - 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            else:
                                cv2.putText(image, "Move to the green STANDING ZONE", 
                                           (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT - 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        elif game.state == GameState.STANDING_READY:
                            cv2.putText(image, "Raise your arm to start aiming", 
                                       (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        elif game.state == GameState.AIMING:
                            cv2.putText(image, "Rotate shoulders to aim, raise arm to throw", 
                                       (SCREEN_WIDTH//2 - 250, SCREEN_HEIGHT - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        elif game.state == GameState.THROWING:
                            cv2.putText(image, "Swing arm forward to release ball", 
                                       (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        elif game.state == GameState.GAME_OVER:
                            final_score = game.score_system.calculate_total_score()
                            cv2.putText(image, f"GAME OVER! Final Score: {final_score}", 
                                       (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT//2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            cv2.putText(image, "Press 'r' to restart or 'q' to quit", 
                                       (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT//2 + 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                    except Exception as e:
                        logger.error(f"Error drawing UI elements: {e}")
                    
                    # Draw MediaPipe landmarks (optional in debug mode)
                    try:
                        if args.debug:
                            if results.pose_landmarks:
                                mp_drawing.draw_landmarks(
                                    image,
                                    results.pose_landmarks,
                                    mp_holistic.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                                )
                            
                            if results.right_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    image,
                                    results.right_hand_landmarks,
                                    mp_holistic.HAND_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                                )
                    except Exception as e:
                        logger.error(f"Error drawing landmarks: {e}")

                    # Draw back button
                    try:
                        back_button.draw(image, hand_position)
                    except Exception as e:
                        logger.error(f"Error drawing back button: {e}")
                    
                    # Display
                    cv2.imshow('Standing AR Bowling Game', image)
                    
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        # Reset game
                        game = StandingBowlingGame()
                        
                except Exception as e:
                    logger.error(f"Error in main game loop: {e}")
                    # Continue running despite errors
                    continue

        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        print("A critical error occurred. Please restart the application.")
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main()