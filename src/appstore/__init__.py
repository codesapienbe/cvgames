import cv2
import mediapipe as mp
import sys
import os
import random
import string
import tempfile
import shutil
import json
import threading
import queue
import subprocess
import threading
import queue
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['cv2', 'mediapipe', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'mediapipe':
                import mediapipe
            elif package == 'numpy':
                import numpy
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nTo install missing packages, run:")
        print("pip install opencv-python mediapipe numpy")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_camera():
    """Check if camera is available"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not available")
            return False
        cap.release()
        print("✅ Camera is available")
        return True
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False

class GameCard:
    def __init__(self, name: str, description: str, rules: str, icon_path: str, module_path: str):
        self.name = name
        self.description = description
        self.rules = rules
        self.icon_path = icon_path
        self.module_path = module_path
        self.is_hovered = False
        self.is_selected = False
        self.hover_start_time = 0
        self.selection_progress = 0.0

class AppStore:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,  # Slightly lower for better performance
            min_tracking_confidence=0.4,   # Slightly lower for better performance
            model_complexity=0  # Use faster model
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Screen setup
        self.screen_width = 1920
        self.screen_height = 1080
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_height)
        # Performance optimizations
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Limit to 30 FPS for better performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        
        # App Store state
        self.games: List[GameCard] = []
        self.current_page = 0
        self.games_per_page = 10  # 5 cards per row × 2 rows
        self.interaction_enabled = True
        self.interaction_timer = 0
        self.interaction_timeout = 2.0  # 2 seconds
        self.last_hands_raised_time = 0
        
        # Animation and UX state
        self.animation_time = 0
        self.hover_animation_speed = 0.1
        self.card_animations = {}  # Store animation states for each card
        self.last_hand_position = (0, 0)
        self.hand_velocity = (0, 0)
        self.smooth_hand_position = (0, 0)
        
        # Navigation and gesture state
        self.hand_positions = []  # Store recent hand positions for swipe detection
        self.swipe_threshold = 100  # Minimum distance for swipe
        self.last_swipe_time = 0
        self.swipe_cooldown = 1.0  # Seconds between swipes
        self.slide_indicator_alpha = 0.0  # For slide gesture visual feedback
        self.last_slide_direction = ""
        self.show_slide_tutorial = False  # Don't show tutorial on startup
        self.tutorial_timer = 0
        self.tutorial_duration = 5.0  # Show tutorial for 5 seconds
        
        # Loading and game state
        self.loading_game = None  # Currently loading game
        self.loading_progress = 0.0  # Loading progress (0.0 to 1.0)
        self.loading_start_time = 0
        self.loading_duration = 3.0  # Loading takes 3 seconds
        self.game_process = None  # Subprocess for running game
        self.game_thread = None  # Thread running the game
        self.game_completed = False  # Thread-safe flag for game completion
        self.exiting_game = False  # Game exit state
        self.exit_progress = 0.0  # Exit progress
        self.games_disabled = False  # Disable game clicks after first click
        
        # UI constants
        self.card_width = 280  # Smaller cards to fit 5 in a row
        self.card_height = 380
        self.card_margin = 40
        self.zoom_factor = 1.15
        self.selection_time = 5.0  # Hold hand over card for 5 seconds to select
        
        # Info modal state
        self.show_info_modal = False
        self.info_button_rect = (50, self.screen_height - 120, 60, 60)
        
        # Navigation button regions
        self.left_nav_rect = (80, self.screen_height // 2 - 40, 80, 80)
        self.right_nav_rect = (self.screen_width - 160, self.screen_height // 2 - 40, 80, 80)
        
        # Glassmorphic colors
        self.colors = {
            'background': (15, 15, 25),
            'card_bg': (30, 30, 45, 180),  # RGBA for transparency
            'card_border': (80, 80, 120, 100),
            'card_hover': (60, 60, 90, 200),
            'card_ready': (40, 120, 60, 220),
            'text_primary': (255, 255, 255),
            'text_secondary': (200, 200, 220),
            'accent': (100, 150, 255),
            'success': (80, 200, 120),
            'warning': (255, 180, 80)
        }
        
        # Thread safety
        self.game_lock = threading.Lock()
        
        # Load all games
        self.load_games()
    

        
    def load_games(self):
        """Dynamically load all game modules from src directory"""
        # Use absolute path to ensure we're looking in the right place
        src_path = Path(__file__).parent.parent.absolute()
        print(f"🔍 Looking for games in: {src_path}")
        print(f"🔍 Current working directory: {Path.cwd()}")
        
        # List all directories first
        all_dirs = [d for d in src_path.iterdir() if d.is_dir()]
        print(f"📁 Found directories: {[d.name for d in all_dirs]}")
        
        game_dirs = [d for d in src_path.iterdir() if d.is_dir() and d.name not in ['appstore', 'cvstore', '__pycache__']]
        print(f"🎮 Game directories found: {[d.name for d in game_dirs]}")
        
        for game_dir in game_dirs:
            try:
                # Check if it's a valid game module
                init_file = game_dir / '__init__.py'
                readme_file = game_dir / 'README.md'
                player_file = game_dir / 'PLAYER.md'
                icon_file = game_dir / 'icon.png'
                
                print(f"🔍 Checking {game_dir.name}: init_file exists = {init_file.exists()}")
                
                if not init_file.exists():
                    print(f"❌ Skipping {game_dir.name}: no __init__.py")
                    continue
                
                # Load game information
                name = game_dir.name.replace('_', ' ').title()
                description = "A fun computer vision game!"
                rules = "Use hand gestures to play!"
                
                # Try to read README.md for description
                if readme_file.exists():
                    try:
                        with open(readme_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Extract first paragraph as description
                            lines = content.split('\n')
                            for line in lines:
                                if line.strip() and not line.startswith('#'):
                                    description = line.strip()[:100] + "..." if len(line) > 100 else line.strip()
                                    break
                    except:
                        pass
                
                # Try to read PLAYER.md for rules
                if player_file.exists():
                    try:
                        with open(player_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Extract first few lines as rules
                            lines = content.split('\n')
                            rules_lines = []
                            for line in lines[:5]:  # First 5 lines
                                if line.strip() and not line.startswith('#'):
                                    rules_lines.append(line.strip())
                            if rules_lines:
                                rules = ' '.join(rules_lines)[:150] + "..." if len(' '.join(rules_lines)) > 150 else ' '.join(rules_lines)
                    except:
                        pass
                
                # Handle icon - use existing icon.png or create placeholder
                icon_path = ""
                if icon_file.exists():
                    icon_path = str(icon_file)
                    print(f"✅ Using existing icon for: {name}")
                else:
                    # Create a placeholder icon path (will be handled by icon fetcher)
                    icon_path = str(icon_file)  # Use the expected path
                    print(f"⚠️  No icon found for: {name} - run fetch_game_icons.py to get icons")
                
                # Create game card
                game_card = GameCard(
                    name=name,
                    description=description,
                    rules=rules,
                    icon_path=icon_path,
                    module_path=str(game_dir)
                )
                
                self.games.append(game_card)
                print(f"✅ Added game: {name}")
                
            except Exception as e:
                print(f"❌ Error loading game {game_dir.name}: {e}")
                continue
        
        print(f"🎯 Loaded {len(self.games)} games total")
    
    def detect_hands_raised(self, hand_landmarks) -> bool:
        """Detect if both hands are raised (palm facing camera)"""
        if not hand_landmarks:
            return False
        
        # Check if any hand has palm facing camera (landmarks 5, 9, 13, 17 are knuckles)
        for landmarks in hand_landmarks:
            if len(landmarks.landmark) >= 21:
                # Check if palm is roughly facing camera (z-coordinate of palm center)
                palm_z = (landmarks.landmark[0].z + landmarks.landmark[5].z + 
                         landmarks.landmark[9].z + landmarks.landmark[13].z + 
                         landmarks.landmark[17].z) / 5
                
                # If palm is close to camera (negative z), hand is raised
                if palm_z < -0.1:
                    return True
        
        return False
    
    def detect_pause_gesture(self, hand_landmarks) -> bool:
        """Detect pause gesture: both hands raised and both closed"""
        if len(hand_landmarks) < 2:
            return False
        
        # Check if both hands are raised
        if not self.detect_hands_raised(hand_landmarks):
            return False
        
        # Check if both hands are closed (fist)
        closed_hands = 0
        for landmarks in hand_landmarks:
            if self.detect_fist(landmarks):
                closed_hands += 1
        
        return closed_hands >= 2
    
    def detect_resume_gesture(self, hand_landmarks) -> bool:
        """Detect resume gesture: both hands raised and both open"""
        if len(hand_landmarks) < 2:
            return False
        
        # Check if both hands are raised
        if not self.detect_hands_raised(hand_landmarks):
            return False
        
        # Check if both hands are open (open palm)
        open_hands = 0
        for landmarks in hand_landmarks:
            if self.detect_open_palm(landmarks):
                open_hands += 1
        
        return open_hands >= 2
    
    def detect_fist(self, landmarks) -> bool:
        """Detect if hand is closed (fist)"""
        if len(landmarks.landmark) < 21:
            return False
        
        # Check if all fingers are closed
        # Thumb: landmarks 4 vs 3
        # Index: landmarks 8 vs 6
        # Middle: landmarks 12 vs 10
        # Ring: landmarks 16 vs 14
        # Pinky: landmarks 20 vs 18
        
        thumb_closed = landmarks.landmark[4].y > landmarks.landmark[3].y
        index_closed = landmarks.landmark[8].y > landmarks.landmark[6].y
        middle_closed = landmarks.landmark[12].y > landmarks.landmark[10].y
        ring_closed = landmarks.landmark[16].y > landmarks.landmark[14].y
        pinky_closed = landmarks.landmark[20].y > landmarks.landmark[18].y
        
        return thumb_closed and index_closed and middle_closed and ring_closed and pinky_closed
    
    def detect_open_palm(self, landmarks) -> bool:
        """Detect if hand is open (palm facing camera)"""
        if len(landmarks.landmark) < 21:
            return False
        
        # Check if all fingers are extended
        thumb_open = landmarks.landmark[4].y < landmarks.landmark[3].y
        index_open = landmarks.landmark[8].y < landmarks.landmark[6].y
        middle_open = landmarks.landmark[12].y < landmarks.landmark[10].y
        ring_open = landmarks.landmark[16].y < landmarks.landmark[14].y
        pinky_open = landmarks.landmark[20].y < landmarks.landmark[18].y
        
        return thumb_open and index_open and middle_open and ring_open and pinky_open
    
    def detect_index_middle_pinch(self, landmarks) -> bool:
        """Detect if index and middle fingers are pinched together (click gesture)"""
        if len(landmarks.landmark) < 21:
            return False
        
        # Get index and middle finger tip positions
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        
        # Calculate distance between index and middle finger tips
        distance = ((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2) ** 0.5
        
        # Check if other fingers are extended (not closed)
        thumb_open = landmarks.landmark[4].y < landmarks.landmark[3].y
        ring_open = landmarks.landmark[16].y < landmarks.landmark[14].y
        pinky_open = landmarks.landmark[20].y < landmarks.landmark[18].y
        
        # Pinch threshold (adjust as needed)
        pinch_threshold = 0.05  # 5% of screen size
        
        return distance < pinch_threshold and thumb_open and ring_open and pinky_open
    
    def get_hand_position(self, landmarks) -> Tuple[int, int]:
        """Get hand position in screen coordinates"""
        if len(landmarks.landmark) < 21:
            return (0, 0)
        
        # Use palm center (landmark 9 - middle finger base)
        # Scale from processing frame (640x480) to screen coordinates (1920x1080)
        x = int(landmarks.landmark[9].x * self.screen_width)
        y = int(landmarks.landmark[9].y * self.screen_height)
        return (x, y)
    
    def update_smooth_hand_position(self, current_pos: Tuple[int, int]):
        """Update smooth hand position with velocity-based smoothing"""
        if self.last_hand_position != (0, 0):
            # Calculate velocity
            dx = current_pos[0] - self.last_hand_position[0]
            dy = current_pos[1] - self.last_hand_position[1]
            self.hand_velocity = (dx * 0.3, dy * 0.3)  # Smoothing factor
            
            # Apply smoothing
            smooth_x = int(self.smooth_hand_position[0] + self.hand_velocity[0])
            smooth_y = int(self.smooth_hand_position[1] + self.hand_velocity[1])
            self.smooth_hand_position = (smooth_x, smooth_y)
        else:
            self.smooth_hand_position = current_pos
        
        self.last_hand_position = current_pos
        
        # Update hand positions for swipe detection
        self.hand_positions.append(current_pos)
        if len(self.hand_positions) > 10:  # Keep last 10 positions
            self.hand_positions.pop(0)
    
    def detect_swipe_gesture(self) -> str:
        """Detect swipe gestures for navigation"""
        if len(self.hand_positions) < 5:
            return ""
        
        current_time = time.time()
        if current_time - self.last_swipe_time < self.swipe_cooldown:
            return ""
        
        # Calculate swipe direction
        start_pos = self.hand_positions[0]
        end_pos = self.hand_positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Check if swipe distance is sufficient
        distance = (dx**2 + dy**2)**0.5
        if distance < self.swipe_threshold:
            return ""
        
        # Determine swipe direction - prioritize horizontal swipes
        if abs(dx) > abs(dy) * 1.5:  # Horizontal swipe (more horizontal than vertical)
            if dx > 0:
                return "right"
            else:
                return "left"
        elif abs(dy) > abs(dx) * 1.5:  # Vertical swipe (more vertical than horizontal)
            if dy > 0:
                return "down"
            else:
                return "up"
        
        return ""
    
    def handle_swipe_navigation(self, swipe_direction: str):
        """Handle swipe navigation"""
        if swipe_direction == "left" and self.current_page > 0:
            self.current_page -= 1
            print("Swiped left - Previous page")
            self.slide_indicator_alpha = 1.0
            self.last_slide_direction = "left"
        elif swipe_direction == "right" and (self.current_page + 1) * self.games_per_page < len(self.games):
            self.current_page += 1
            print("Swiped right - Next page")
            self.slide_indicator_alpha = 1.0
            self.last_slide_direction = "right"
        
        self.last_swipe_time = time.time()
    
    def get_card_position(self, index: int) -> Tuple[int, int, int, int]:
        """Get position and size of a game card"""
        cards_per_row = 5  # 5 cards in a row
        row = index // cards_per_row
        col = index % cards_per_row
        
        # Calculate total width of all cards and margins
        total_cards_width = cards_per_row * self.card_width + (cards_per_row - 1) * self.card_margin
        start_x = (self.screen_width - total_cards_width) // 2
        
        x = start_x + col * (self.card_width + self.card_margin)
        y = 200 + row * (self.card_height + self.card_margin)  # Start below title
        
        return (x, y, self.card_width, self.card_height)
    
    def is_point_in_rect(self, point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
        """Check if point is inside rectangle"""
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    
    def draw_glassmorphic_card(self, frame, card: GameCard, x: int, y: int, width: int, height: int):
        """Draw a glassmorphic game card with modern design"""
        # Calculate zoom effect
        if card.is_hovered:
            zoom = self.zoom_factor
            # Adjust position to keep card centered when zoomed
            zoom_offset_x = int((width * zoom - width) / 2)
            zoom_offset_y = int((height * zoom - height) / 2)
            x -= zoom_offset_x
            y -= zoom_offset_y
            width = int(width * zoom)
            height = int(height * zoom)
        else:
            zoom = 1.0
        
        # Create glassmorphic effect with multiple layers
        # 1. Outer glow (if selected)
        if card.is_selected:
            glow_color = self.colors['success']
            cv2.rectangle(frame, (x-3, y-3), (x + width+3, y + height+3), glow_color, 6)
        
        # 2. Main card background with transparency effect
        if self.games_disabled:
            bg_color = (50, 50, 50)  # Grayed out when disabled
        elif card.is_selected:
            bg_color = self.colors['card_ready'][:3]  # Remove alpha for OpenCV
        elif card.is_hovered:
            bg_color = self.colors['card_hover'][:3]
        else:
            bg_color = self.colors['card_bg'][:3]
        
        # Create gradient-like effect with fewer layers for better performance
        for i in range(2):  # Reduced from 3 to 2 layers
            alpha = 0.3 - i * 0.15
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + width, y + height), bg_color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 3. Card border with glassmorphic effect
        border_color = self.colors['card_border'][:3]
        cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, 2)
        
        # 4. Inner highlight for glassmorphic look
        highlight_color = (255, 255, 255, 50)
        cv2.rectangle(frame, (x+2, y+2), (x + width-2, y + height//3), highlight_color[:3], -1)
        cv2.addWeighted(frame, 0.1, frame, 0.9, 0, frame)
        
        # 5. Rounded corners effect (simulated with small rectangles)
        corner_size = 15
        corner_color = bg_color
        # Top-left corner
        cv2.rectangle(frame, (x, y), (x + corner_size, y + corner_size), corner_color, -1)
        # Top-right corner
        cv2.rectangle(frame, (x + width - corner_size, y), (x + width, y + corner_size), corner_color, -1)
        # Bottom-left corner
        cv2.rectangle(frame, (x, y + height - corner_size), (x + corner_size, y + height), corner_color, -1)
        # Bottom-right corner
        cv2.rectangle(frame, (x + width - corner_size, y + height - corner_size), (x + width, y + height), corner_color, -1)
        
        # Game icon with glassmorphic effect
        icon_size = min(100, width // 3)
        icon_x = x + (width - icon_size) // 2
        icon_y = y + 30
        
        # Icon background with single layer for better performance
        overlay = frame.copy()
        cv2.rectangle(overlay, (icon_x, icon_y), (icon_x + icon_size, icon_y + icon_size), 
                     self.colors['accent'][:3], -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Icon border
        cv2.rectangle(frame, (icon_x, icon_y), (icon_x + icon_size, icon_y + icon_size), 
                     self.colors['text_primary'], 2)
        
        # Icon content (game controller symbol)
        controller_x = icon_x + icon_size // 2
        controller_y = icon_y + icon_size // 2
        cv2.circle(frame, (controller_x, controller_y), 15, self.colors['text_primary'], 2)
        cv2.circle(frame, (controller_x - 8, controller_y - 8), 3, self.colors['success'], -1)
        cv2.circle(frame, (controller_x + 8, controller_y - 8), 3, self.colors['warning'], -1)
        cv2.circle(frame, (controller_x, controller_y + 8), 3, self.colors['accent'], -1)
        
        # Game name with glassmorphic text effect
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9 if zoom > 1.0 else 0.7
        thickness = 2 if zoom > 1.0 else 1
        
        # Wrap text if too long
        words = card.name.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if text_width <= width - 30:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Draw text lines with shadow effect
        text_y = icon_y + icon_size + 40
        for i, line in enumerate(lines):
            if text_y + i * 35 > y + height - 100:  # Leave space for description
                break
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            text_x = x + (width - text_width) // 2
            
            # Text shadow
            cv2.putText(frame, line, (text_x + 2, text_y + i * 35 + 2), font, font_scale, 
                       (0, 0, 0), thickness + 1)
            # Main text
            cv2.putText(frame, line, (text_x, text_y + i * 35), font, font_scale, 
                       self.colors['text_primary'], thickness)
        
        # Game description with glassmorphic effect
        desc = card.description[:60] + "..." if len(card.description) > 60 else card.description
        desc_y = y + height - 60
        (desc_width, desc_height), _ = cv2.getTextSize(desc, font, 0.5, 1)
        desc_x = x + (width - desc_width) // 2
        
        # Description background
        desc_bg_y = desc_y - 15
        cv2.rectangle(frame, (desc_x - 10, desc_bg_y), (desc_x + desc_width + 10, desc_y + 10), 
                     self.colors['card_bg'][:3], -1)
        cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)
        
        # Description text
        cv2.putText(frame, desc, (desc_x, desc_y), font, 0.5, self.colors['text_secondary'], 1)
        
                # Selection indicator for simultaneous gesture
        if card.is_hovered and not self.games_disabled:
            center_x = x + width // 2
            center_y = y + height // 2
            
            # Draw "PINCH TO CLICK" indicator
            indicator_text = "PINCH TO CLICK"
            (text_width, text_height), _ = cv2.getTextSize(indicator_text, font, 0.5, 1)
            text_x = center_x - text_width // 2
            text_y = center_y + 50
            
            # Indicator background
            cv2.rectangle(frame, (text_x - 15, text_y - 15), (text_x + text_width + 15, text_y + 15), 
                         self.colors['accent'][:3], -1)
            cv2.addWeighted(frame, 0.4, frame, 0.6, 0, frame)
            
            # Indicator text
            cv2.putText(frame, indicator_text, (text_x, text_y), font, 0.5, self.colors['text_primary'], 1)
            
            # Draw pinch gesture icon
            hand_x = center_x
            hand_y = center_y - 20
            cv2.circle(frame, (hand_x, hand_y), 12, self.colors['text_primary'], 2)
            # Index and middle finger tips (pinched)
            cv2.circle(frame, (hand_x - 3, hand_y - 3), 3, (100, 255, 100), -1)  # Index (green)
            cv2.circle(frame, (hand_x + 3, hand_y - 3), 3, (100, 100, 255), -1)  # Middle (blue)
        elif card.is_hovered and self.games_disabled:
            center_x = x + width // 2
            center_y = y + height // 2
            
            # Draw "GAMES DISABLED" indicator
            indicator_text = "GAMES DISABLED"
            (text_width, text_height), _ = cv2.getTextSize(indicator_text, font, 0.5, 1)
            text_x = center_x - text_width // 2
            text_y = center_y + 50
            
            # Indicator background (red for disabled)
            cv2.rectangle(frame, (text_x - 15, text_y - 15), (text_x + text_width + 15, text_y + 15), 
                         (100, 50, 50), -1)
            cv2.addWeighted(frame, 0.4, frame, 0.6, 0, frame)
            
            # Indicator text
            cv2.putText(frame, indicator_text, (text_x, text_y), font, 0.5, (255, 200, 200), 1)
    
    def draw_info_modal(self, frame):
        """Draw the glassmorphic info modal with usage instructions"""
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Modal background with glassmorphic effect
        modal_width = 900
        modal_height = 650
        modal_x = (self.screen_width - modal_width) // 2
        modal_y = (self.screen_height - modal_height) // 2
        
        # Multiple layers for glassmorphic effect
        for i in range(3):
            alpha = 0.4 - i * 0.1
            overlay = frame.copy()
            cv2.rectangle(overlay, (modal_x, modal_y), (modal_x + modal_width, modal_y + modal_height), 
                         self.colors['card_bg'][:3], -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Modal border with glow
        cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_width, modal_y + modal_height), 
                     self.colors['accent'][:3], 3)
        
        # Glassmorphic title
        title = "CVGames App Store - How to Use"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (title_width, title_height), _ = cv2.getTextSize(title, font, 1.4, 2)
        title_x = modal_x + (modal_width - title_width) // 2
        
        # Title shadow
        cv2.putText(frame, title, (title_x + 2, modal_y + 52), font, 1.4, (0, 0, 0), 3)
        # Main title
        cv2.putText(frame, title, (title_x, modal_y + 50), font, 1.4, self.colors['text_primary'], 2)
        
        # Instructions
        instructions = [
            "🤲 HAND GESTURES:",
            "",
            "• Move your hand to navigate between game cards",
            "• Hover over a card to highlight it (zoom effect)",
            "• HOLD YOUR HAND over a game card for 5 seconds to select",
            "• Watch the progress bar fill up to completion",
            "",
            "⏸️ PAUSE INTERACTION:",
            "",
            "• Raise both hands AND close both fists to pause",
            "• Raise both hands AND open both palms to resume",
            "• Useful when you need to move around",
            "",
            "🎮 GAME CONTROLS:",
            "",
            "• Most games: Press 'q' to quit and return here",
            "• Some games have their own quit gestures",
            "• The app store will always be waiting when you exit",
            "",
            "💡 TIPS:",
            "",
            "• Keep your hand steady over the game card",
            "• Don't move your hand until selection is complete",
            "• Maintain good lighting for accurate tracking",
            "• Use your dominant hand for better control",
            "• Stay within camera view for reliable detection",
            "• The 5-second hold prevents accidental selections"
        ]
        
        # Draw instructions
        y_offset = modal_y + 100
        line_height = 30
        font_small = cv2.FONT_HERSHEY_SIMPLEX
        
        for i, line in enumerate(instructions):
            if y_offset + i * line_height > modal_y + modal_height - 50:
                break
            
            # Different colors for different sections
            if line.startswith("🤲") or line.startswith("⏸️") or line.startswith("🎮") or line.startswith("💡"):
                color = (255, 255, 0)  # Yellow for headers
                font_scale = 0.7
                thickness = 2
            elif line == "":
                color = (200, 200, 200)  # Gray for empty lines
                font_scale = 0.5
                thickness = 1
            else:
                color = (255, 255, 255)  # White for regular text
                font_scale = 0.6
                thickness = 1
            
            cv2.putText(frame, line, (modal_x + 30, y_offset + i * line_height), 
                       font_small, font_scale, color, thickness)
        
        # Close button
        close_text = "Press 'i' or close fist to close this info"
        (close_width, close_height), _ = cv2.getTextSize(close_text, font_small, 0.6, 1)
        close_x = modal_x + (modal_width - close_width) // 2
        cv2.putText(frame, close_text, (close_x, modal_y + modal_height - 20), 
                   font_small, 0.6, (150, 150, 150), 1)
    
    def draw_visual_cursor(self, frame, hand_x: int, hand_y: int):
        """Draw a smooth visual cursor at hand position"""
        if hand_x == 0 and hand_y == 0:
            return
        
        # Outer ring
        cv2.circle(frame, (hand_x, hand_y), 20, self.colors['accent'][:3], 2)
        
        # Inner circle
        cv2.circle(frame, (hand_x, hand_y), 8, self.colors['text_primary'], -1)
        
        # Center dot
        cv2.circle(frame, (hand_x, hand_y), 3, self.colors['background'], -1)
        
        # Pulse effect
        pulse_radius = int(15 + 5 * abs(np.sin(self.animation_time * 3)))
        cv2.circle(frame, (hand_x, hand_y), pulse_radius, self.colors['accent'][:3], 1)
    
    def draw_hand_trail(self, frame, positions: List[Tuple[int, int]]):
        """Draw a fading trail behind the hand"""
        if len(positions) < 2:
            return
        
        for i in range(len(positions) - 1):
            alpha = (i + 1) / len(positions) * 0.3
            overlay = frame.copy()
            cv2.line(overlay, positions[i], positions[i + 1], self.colors['accent'][:3], 2)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def draw_info_button(self, frame):
        """Draw the glassmorphic info button in the bottom left"""
        x, y, w, h = self.info_button_rect
        
        # Glassmorphic button background
        for i in range(2):
            alpha = 0.4 - i * 0.2
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), self.colors['accent'][:3], -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Button border
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['text_primary'], 2)
        
        # Info icon (question mark) with shadow
        cv2.putText(frame, "?", (x + 22, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(frame, "?", (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.colors['text_primary'], 2)
        
        # Label with background
        label_text = "INFO"
        (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_x = x + (w - label_width) // 2
        label_y = y + h + 25
        
        # Label background
        cv2.rectangle(frame, (label_x - 5, label_y - 15), (label_x + label_width + 5, label_y + 5), 
                     self.colors['card_bg'][:3], -1)
        cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)
        
        # Label text
        cv2.putText(frame, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_secondary'], 1)
    
    def draw_navigation_arrow(self, frame, direction: str, center: Tuple[int, int]):
        """Draw a glassmorphic navigation arrow"""
        x, y = center
        arrow_size = 40
        
        # Arrow background with glassmorphic effect
        for i in range(2):
            alpha = 0.3 - i * 0.1
            overlay = frame.copy()
            cv2.circle(overlay, (x, y), arrow_size, self.colors['accent'][:3], -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Arrow border
        cv2.circle(frame, (x, y), arrow_size, self.colors['text_primary'], 2)
        
        # Draw arrow based on direction
        if direction == "left":
            # Left arrow
            cv2.arrowedLine(frame, (x + 15, y), (x - 15, y), self.colors['text_primary'], 4, tipLength=0.4)
            # Additional arrow lines for thickness
            cv2.arrowedLine(frame, (x + 12, y - 3), (x - 12, y - 3), self.colors['text_primary'], 2, tipLength=0.4)
            cv2.arrowedLine(frame, (x + 12, y + 3), (x - 12, y + 3), self.colors['text_primary'], 2, tipLength=0.4)
        else:  # right
            # Right arrow
            cv2.arrowedLine(frame, (x - 15, y), (x + 15, y), self.colors['text_primary'], 4, tipLength=0.4)
            # Additional arrow lines for thickness
            cv2.arrowedLine(frame, (x - 12, y - 3), (x + 12, y - 3), self.colors['text_primary'], 2, tipLength=0.4)
            cv2.arrowedLine(frame, (x - 12, y + 3), (x + 12, y + 3), self.colors['text_primary'], 2, tipLength=0.4)
        
        # Pulse effect
        pulse_radius = int(arrow_size + 5 * abs(np.sin(self.animation_time * 2)))
        cv2.circle(frame, (x, y), pulse_radius, self.colors['accent'][:3], 1)
    
    def draw_navigation_button(self, frame, direction: str, rect: Tuple[int, int, int, int]):
        """Draw a clickable navigation button"""
        x, y, w, h = rect
        
        # Button background with single layer for better performance
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.colors['accent'][:3], -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Button border
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['text_primary'], 2)
        
        # Draw arrow based on direction
        center_x = x + w // 2
        center_y = y + h // 2
        
        if direction == "left":
            # Left arrow
            cv2.arrowedLine(frame, (center_x + 15, center_y), (center_x - 15, center_y), self.colors['text_primary'], 4, tipLength=0.4)
            # Additional arrow lines for thickness
            cv2.arrowedLine(frame, (center_x + 12, center_y - 3), (center_x - 12, center_y - 3), self.colors['text_primary'], 2, tipLength=0.4)
            cv2.arrowedLine(frame, (center_x + 12, center_y + 3), (center_x - 12, center_y + 3), self.colors['text_primary'], 2, tipLength=0.4)
        else:  # right
            # Right arrow
            cv2.arrowedLine(frame, (center_x - 15, center_y), (center_x + 15, center_y), self.colors['text_primary'], 4, tipLength=0.4)
            # Additional arrow lines for thickness
            cv2.arrowedLine(frame, (center_x - 12, center_y - 3), (center_x + 12, center_y - 3), self.colors['text_primary'], 2, tipLength=0.4)
            cv2.arrowedLine(frame, (center_x - 12, center_y + 3), (center_x + 12, center_y + 3), self.colors['text_primary'], 2, tipLength=0.4)
        
        # Pulse effect
        pulse_radius = int(w // 2 + 5 * abs(np.sin(self.animation_time * 2)))
        cv2.circle(frame, (center_x, center_y), pulse_radius, self.colors['accent'][:3], 1)
        
        # Click instruction text
        text = "CLICK"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, 0.5, 1)
        text_x = center_x - text_width // 2
        text_y = y + h + 20
        
        cv2.putText(frame, text, (text_x, text_y), font, 0.5, self.colors['text_secondary'], 1)
    
    def draw_slide_indicator(self, frame, direction: str):
        """Draw a slide gesture indicator"""
        if self.slide_indicator_alpha <= 0:
            return
        
        # Calculate indicator position and size
        indicator_width = 200
        indicator_height = 100
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        # Create overlay for slide indicator
        overlay = frame.copy()
        
        # Background rectangle
        cv2.rectangle(overlay, 
                     (center_x - indicator_width // 2, center_y - indicator_height // 2),
                     (center_x + indicator_width // 2, center_y + indicator_height // 2),
                     self.colors['accent'][:3], -1)
        
        # Border
        cv2.rectangle(overlay, 
                     (center_x - indicator_width // 2, center_y - indicator_height // 2),
                     (center_x + indicator_width // 2, center_y + indicator_height // 2),
                     self.colors['text_primary'], 3)
        
        # Draw slide arrow
        if direction == "left":
            # Left slide arrow
            arrow_start = (center_x + 30, center_y)
            arrow_end = (center_x - 30, center_y)
            cv2.arrowedLine(overlay, arrow_start, arrow_end, self.colors['text_primary'], 6, tipLength=0.5)
            
            # Additional arrow lines for thickness
            cv2.arrowedLine(overlay, (arrow_start[0], arrow_start[1] - 4), (arrow_end[0], arrow_end[1] - 4), 
                           self.colors['text_primary'], 3, tipLength=0.5)
            cv2.arrowedLine(overlay, (arrow_start[0], arrow_start[1] + 4), (arrow_end[0], arrow_end[1] + 4), 
                           self.colors['text_primary'], 3, tipLength=0.5)
            
            # Text
            cv2.putText(overlay, "SLIDE LEFT", (center_x - 60, center_y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_primary'], 2)
            
        elif direction == "right":
            # Right slide arrow
            arrow_start = (center_x - 30, center_y)
            arrow_end = (center_x + 30, center_y)
            cv2.arrowedLine(overlay, arrow_start, arrow_end, self.colors['text_primary'], 6, tipLength=0.5)
            
            # Additional arrow lines for thickness
            cv2.arrowedLine(overlay, (arrow_start[0], arrow_start[1] - 4), (arrow_end[0], arrow_end[1] - 4), 
                           self.colors['text_primary'], 3, tipLength=0.5)
            cv2.arrowedLine(overlay, (arrow_start[0], arrow_start[1] + 4), (arrow_end[0], arrow_end[1] + 4), 
                           self.colors['text_primary'], 3, tipLength=0.5)
            
            # Text
            cv2.putText(overlay, "SLIDE RIGHT", (center_x - 65, center_y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_primary'], 2)
        
        # Apply overlay with fade effect
        cv2.addWeighted(overlay, self.slide_indicator_alpha, frame, 1 - self.slide_indicator_alpha, 0, frame)
        
        # Fade out the indicator
        self.slide_indicator_alpha = max(0, self.slide_indicator_alpha - 0.05)
    
    def get_game_status(self) -> str:
        """Get current game status for display"""
        if not self.loading_game:
            return "Ready"
        
        if self.game_thread and self.game_thread.is_alive():
            if self.loading_progress >= 0.95:
                return "Running"
            else:
                return "Loading"
        else:
            return "Completed"
    
    def draw_loading_progress(self, frame):
        """Draw loading progress bar when launching a game"""
        if not self.loading_game:
            return
        
        # Create blurred background effect (70% blur)
        blurred_frame = frame.copy()
        blurred_frame = cv2.GaussianBlur(blurred_frame, (21, 21), 0)
        
        # Apply 70% blur overlay
        overlay = blurred_frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Loading title with status
        status = self.get_game_status()
        title = f"{status} {self.loading_game.name}..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        (title_width, title_height), _ = cv2.getTextSize(title, font, 2.0, 3)
        title_x = (self.screen_width - title_width) // 2
        title_y = self.screen_height // 2 - 100
        
        # Title shadow
        cv2.putText(frame, title, (title_x + 3, title_y + 3), font, 2.0, (0, 0, 0), 4)
        # Main title
        cv2.putText(frame, title, (title_x, title_y), font, 2.0, self.colors['text_primary'], 3)
        
        # Progress bar background
        bar_width = 600
        bar_height = 30
        bar_x = (self.screen_width - bar_width) // 2
        bar_y = self.screen_height // 2
        
        # Progress bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.colors['card_bg'][:3], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.colors['text_primary'], 2)
        
        # Progress bar fill
        fill_width = int(bar_width * self.loading_progress)
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         self.colors['success'], -1)
        
        # Progress percentage
        progress_text = f"{int(self.loading_progress * 100)}%"
        (progress_width, progress_height), _ = cv2.getTextSize(progress_text, font, 1.0, 2)
        progress_x = (self.screen_width - progress_width) // 2
        progress_y = bar_y + bar_height + 50
        
        cv2.putText(frame, progress_text, (progress_x, progress_y), font, 1.0, self.colors['text_primary'], 2)
        
        # Loading animation dots
        dots = "." * (int(self.animation_time * 2) % 4)
        dots_text = f"Loading{dots}"
        (dots_width, dots_height), _ = cv2.getTextSize(dots_text, font, 0.8, 2)
        dots_x = (self.screen_width - dots_width) // 2
        dots_y = progress_y + 40
        
        cv2.putText(frame, dots_text, (dots_x, dots_y), font, 0.8, self.colors['text_secondary'], 2)
    
    def draw_exit_progress(self, frame):
        """Draw exit progress bar when returning from game"""
        if not self.exiting_game:
            return
        
        # Create overlay for exit screen
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Exit title
        title = "Returning to App Store..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        (title_width, title_height), _ = cv2.getTextSize(title, font, 2.0, 3)
        title_x = (self.screen_width - title_width) // 2
        title_y = self.screen_height // 2 - 100
        
        # Title shadow
        cv2.putText(frame, title, (title_x + 3, title_y + 3), font, 2.0, (0, 0, 0), 4)
        # Main title
        cv2.putText(frame, title, (title_x, title_y), font, 2.0, self.colors['text_primary'], 3)
        
        # Progress bar background
        bar_width = 600
        bar_height = 30
        bar_x = (self.screen_width - bar_width) // 2
        bar_y = self.screen_height // 2
        
        # Progress bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.colors['card_bg'][:3], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.colors['text_primary'], 2)
        
        # Progress bar fill
        fill_width = int(bar_width * self.exit_progress)
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         self.colors['accent'], -1)
        
        # Progress percentage
        progress_text = f"{int(self.exit_progress * 100)}%"
        (progress_width, progress_height), _ = cv2.getTextSize(progress_text, font, 1.0, 2)
        progress_x = (self.screen_width - progress_width) // 2
        progress_y = bar_y + bar_height + 50
        
        cv2.putText(frame, progress_text, (progress_x, progress_y), font, 1.0, self.colors['text_primary'], 2)
    
    def draw_hand_dots(self, frame, landmarks):
        """Draw only dots on finger tips and palm center"""
        if len(landmarks.landmark) < 21:
            return
        
        # Define finger tip landmarks
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        palm_center = 9  # Middle finger base (palm center)
        
        # Draw dots on finger tips
        for tip_id in finger_tips:
            landmark = landmarks.landmark[tip_id]
            x = int(landmark.x * self.screen_width)
            y = int(landmark.y * self.screen_height)
            
            # Draw colored dot based on finger
            if tip_id == 4:  # Thumb
                color = (255, 100, 100)  # Red
                radius = 8
            elif tip_id == 8:  # Index
                color = (100, 255, 100)  # Green
                radius = 8
            elif tip_id == 12:  # Middle
                color = (100, 100, 255)  # Blue
                radius = 8
            elif tip_id == 16:  # Ring
                color = (255, 255, 100)  # Yellow
                radius = 8
            else:  # Pinky
                color = (255, 100, 255)  # Magenta
                radius = 8
            
            # Draw dot with glow effect
            cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), -1)  # White glow
            cv2.circle(frame, (x, y), radius, color, -1)  # Colored dot
            cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)  # White border
        
        # Draw palm center dot
        palm_landmark = landmarks.landmark[palm_center]
        palm_x = int(palm_landmark.x * self.screen_width)
        palm_y = int(palm_landmark.y * self.screen_height)
        
        # Palm center dot (larger, different color)
        cv2.circle(frame, (palm_x, palm_y), 12, (255, 255, 255), -1)  # White glow
        cv2.circle(frame, (palm_x, palm_y), 10, (200, 200, 200), -1)  # Gray center
        cv2.circle(frame, (palm_x, palm_y), 10, (255, 255, 255), 2)  # White border
    
    def draw_slide_tutorial(self, frame):
        """Draw slide gesture tutorial overlay"""
        if not self.show_slide_tutorial:
            return
        
        # Create overlay for tutorial
        overlay = frame.copy()
        
        # Semi-transparent background
        cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Tutorial content
        tutorial_text = [
            "CLICK GESTURES",
            "",
            "Pinch your INDEX and MIDDLE fingers together to click",
            "Keep other fingers (thumb, ring, pinky) extended",
            "",
            "Click on game cards to launch them",
            "Click on left/right buttons to navigate pages",
            "Click on INFO button for this tutorial",
            "",
            "Click anywhere to close this tutorial",
            "",
            f"Tutorial will close in {max(0, int(self.tutorial_duration - self.tutorial_timer))} seconds"
        ]
        
        # Draw tutorial text
        y_start = self.screen_height // 2 - 150
        for i, text in enumerate(tutorial_text):
            y = y_start + i * 40
            if text == "SLIDE GESTURES":
                # Title
                cv2.putText(frame, text, (self.screen_width // 2 - 150, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['accent'][:3], 3)
                cv2.putText(frame, text, (self.screen_width // 2 - 150, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['text_primary'], 2)
            elif text.startswith("Slide your hand"):
                # Instructions
                cv2.putText(frame, text, (self.screen_width // 2 - 200, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text_secondary'], 2)
            elif text.startswith("Tutorial will close"):
                # Timer
                cv2.putText(frame, text, (self.screen_width // 2 - 180, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_secondary'], 2)
            else:
                # Regular text
                cv2.putText(frame, text, (self.screen_width // 2 - 150, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_secondary'], 1)
        
        # Draw example pinch gesture
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2 + 100
        
        # Draw two circles representing index and middle finger tips
        cv2.circle(frame, (center_x - 30, center_y), 15, (100, 255, 100), -1)  # Index finger (green)
        cv2.circle(frame, (center_x + 30, center_y), 15, (100, 100, 255), -1)  # Middle finger (blue)
        
        # Draw arrow showing they come together
        cv2.arrowedLine(frame, (center_x - 20, center_y - 20), (center_x + 20, center_y - 20), 
                       self.colors['accent'][:3], 4, tipLength=0.3)
        
        cv2.putText(frame, "PINCH", (center_x - 40, center_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['accent'][:3], 2)
    
    def draw_page_dots(self, frame, total_pages: int, current_page: int):
        """Draw page indicator dots"""
        if total_pages <= 1:
            return
        
        # Calculate dots position
        dot_radius = 8
        dot_spacing = 25
        total_width = (total_pages - 1) * dot_spacing
        start_x = (self.screen_width - total_width) // 2
        y = self.screen_height - 80
        
        for i in range(total_pages):
            x = start_x + i * dot_spacing
            
            # Draw dot background with glassmorphic effect
            if i == current_page:
                # Active page dot
                cv2.circle(frame, (x, y), dot_radius + 2, self.colors['accent'][:3], -1)
                cv2.circle(frame, (x, y), dot_radius + 2, self.colors['text_primary'], 2)
                # Pulse effect for active dot
                pulse_radius = int(dot_radius + 3 + 2 * abs(np.sin(self.animation_time * 4)))
                cv2.circle(frame, (x, y), pulse_radius, self.colors['accent'][:3], 1)
            else:
                # Inactive page dot
                cv2.circle(frame, (x, y), dot_radius, self.colors['text_secondary'], -1)
                cv2.circle(frame, (x, y), dot_radius, self.colors['text_primary'], 1)
    
    def launch_game(self, game_card: GameCard):
        """Launch a game in a separate thread"""
        try:
            print(f"🚀 Launching {game_card.name}...")
            
            # Start loading state
            self.loading_game = game_card
            self.loading_progress = 0.0
            self.loading_start_time = time.time()
            
            # Create and start game thread
            game_thread = threading.Thread(
                target=self._run_game_in_thread,
                args=(game_card,),
                daemon=True
            )
            game_thread.start()
            
            # Store thread reference for monitoring
            self.game_thread = game_thread
            
        except Exception as e:
            print(f"❌ Error launching {game_card.name}: {e}")
            self.loading_game = None
            self.games_disabled = False
    
    def _run_game_in_thread(self, game_card: GameCard):
        """Run game in a separate thread"""
        try:
            # Change to the game directory
            game_dir = Path(game_card.module_path)
            init_file = game_dir / '__init__.py'
            
            if init_file.exists():
                print(f"🎮 Starting {game_card.name} in thread...")
                
                # Use subprocess to run the game
                self.game_process = subprocess.Popen([sys.executable, str(init_file)], 
                                                   cwd=str(game_dir))
                
                # Wait for the game to finish
                self.game_process.wait()
                
                print(f"✅ {game_card.name} finished")
                
                # Signal game completion (thread-safe)
                with self.game_lock:
                    self.game_completed = True
                
            else:
                print(f"❌ Game file not found: {init_file}")
                with self.game_lock:
                    self.game_completed = True
                
        except subprocess.TimeoutExpired:
            print(f"⚠️ {game_card.name} timed out")
            if self.game_process:
                self.game_process.kill()
            with self.game_lock:
                self.game_completed = True
        except Exception as e:
            print(f"❌ Error running {game_card.name}: {e}")
            with self.game_lock:
                self.game_completed = True
        finally:
            # Return to the original directory
            os.chdir(Path(__file__).parent.parent)
            # Re-enable game clicks when returning from game
            self.games_disabled = False
            print("✅ Games re-enabled for clicking")
    
    def run(self):
        """Main app store loop"""
        cv2.namedWindow('CVGames App Store', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('CVGames App Store', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Frame rate control
        fps_start_time = time.time()
        target_fps = 30
        frame_time = 1.0 / target_fps
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Update animation time (optimized for 30 FPS)
            self.animation_time += 0.033  # Approximately 30 FPS
            
            # Update tutorial timer
            if self.show_slide_tutorial:
                self.tutorial_timer += 0.016
                if self.tutorial_timer >= self.tutorial_duration:
                    self.show_slide_tutorial = False
            
            # Update loading progress
            if self.loading_game:
                elapsed_time = time.time() - self.loading_start_time
                self.loading_progress = min(elapsed_time / self.loading_duration, 1.0)
                
                # Check if game thread is still running
                if self.game_thread and not self.game_thread.is_alive():
                    # Thread finished but we haven't detected completion yet
                    with self.game_lock:
                        if not self.game_completed:
                            self.game_completed = True
                
                # Keep loading state until game actually starts
                if self.loading_progress >= 1.0 and self.game_thread and self.game_thread.is_alive():
                    # Game is running, keep loading state minimal
                    self.loading_progress = 0.95
            
            # Check for game completion (thread-safe)
            with self.game_lock:
                if self.game_completed:
                    # Start exit progress
                    self.exiting_game = True
                    self.exit_progress = 0.0
                    self.game_completed = False  # Reset flag
                    self.game_thread = None  # Clear thread reference
                    print("🎮 Game completed - starting exit sequence")
            
            # Update exit progress
            if self.exiting_game:
                self.exit_progress = min(self.exit_progress + 0.02, 1.0)  # 2% per frame
                if self.exit_progress >= 1.0:
                    self.exiting_game = False
                    self.exit_progress = 0.0
            
            # Create a black background instead of showing camera feed
            frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            
            # Optimized camera processing - read frame once
            ret, original_frame = self.cap.read()
            if ret:
                # Flip and resize for better performance
                original_frame = cv2.flip(original_frame, 1)
                # Resize to smaller size for processing (faster)
                process_frame = cv2.resize(original_frame, (640, 480))
                rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
            else:
                results = None
            
            # Reset hover states
            for card in self.games:
                card.is_hovered = False
            
            # Handle hand tracking and gestures
            if results.multi_hand_landmarks and self.interaction_enabled:
                # Check for pause gesture: both hands raised and both closed
                if len(results.multi_hand_landmarks) >= 2:
                    if self.detect_pause_gesture(results.multi_hand_landmarks):
                        current_time = time.time()
                        if current_time - self.last_hands_raised_time > self.interaction_timeout:
                            self.interaction_enabled = False
                            self.interaction_timer = current_time
                            self.last_hands_raised_time = current_time
                            print("Interaction paused - both hands raised and closed")
                
                # Check for resume gesture: both hands raised and both open
                if len(results.multi_hand_landmarks) >= 2:
                    if self.detect_resume_gesture(results.multi_hand_landmarks):
                        if not self.interaction_enabled:
                            self.interaction_enabled = True
                            print("Interaction resumed - both hands raised and open")
                
                # Process hand gestures for game selection and info button
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw only dots on finger tips instead of full hand landmarks
                    self.draw_hand_dots(frame, hand_landmarks)
                    
                    # Get hand position and update smooth tracking
                    hand_x, hand_y = self.get_hand_position(hand_landmarks)
                    self.update_smooth_hand_position((hand_x, hand_y))
                    
                    # Check for index-middle finger pinch (click gesture)
                    if self.detect_index_middle_pinch(hand_landmarks):
                        # Check if clicking on info button
                        if self.is_point_in_rect((hand_x, hand_y), self.info_button_rect):
                            self.show_slide_tutorial = True
                            self.tutorial_timer = 0
                            print("Slide tutorial opened via click")
                        
                        # Check if clicking on left navigation button
                        elif self.is_point_in_rect((hand_x, hand_y), self.left_nav_rect):
                            if self.current_page > 0:
                                self.current_page -= 1
                                print("Previous page via click")
                        
                        # Check if clicking on right navigation button
                        elif self.is_point_in_rect((hand_x, hand_y), self.right_nav_rect):
                            if (self.current_page + 1) * self.games_per_page < len(self.games):
                                self.current_page += 1
                                print("Next page via click")
                        
                        # Check if clicking on any game card (only if games are not disabled)
                        elif not self.games_disabled:
                            current_page_start = self.current_page * self.games_per_page
                            current_page_end = min(current_page_start + self.games_per_page, len(self.games))
                            
                            for i in range(current_page_start, current_page_end):
                                card_index = i - current_page_start
                                card = self.games[i]
                                card_rect = self.get_card_position(card_index)
                                
                                if self.is_point_in_rect((hand_x, hand_y), card_rect):
                                    print(f"Launching {card.name} via click")
                                    self.games_disabled = True  # Disable all game clicks
                                    self.launch_game(card)
                                    break
                    
                    # Check for tutorial close with pinch gesture
                    if self.show_slide_tutorial and self.detect_index_middle_pinch(hand_landmarks):
                        self.show_slide_tutorial = False
                        print("Tutorial closed via click")
                    
                    # Check for hover effects (no selection, just visual feedback)
                    current_page_start = self.current_page * self.games_per_page
                    current_page_end = min(current_page_start + self.games_per_page, len(self.games))
                    
                    for i in range(current_page_start, current_page_end):
                        card_index = i - current_page_start
                        card = self.games[i]
                        card_rect = self.get_card_position(card_index)
                        
                        if self.is_point_in_rect((hand_x, hand_y), card_rect):
                            card.is_hovered = True
                            break
                        else:
                            card.is_hovered = False
            
            # Check for interaction re-enable
            if not self.interaction_enabled:
                current_time = time.time()
                if current_time - self.interaction_timer > self.interaction_timeout:
                    # Check if hands are no longer raised
                    if results.multi_hand_landmarks:
                        hands_raised = self.detect_hands_raised(results.multi_hand_landmarks)
                        if not hands_raised:
                            self.interaction_enabled = True
                            print("Interaction enabled")
            
            # Detect swipe gestures for navigation
            if self.interaction_enabled and len(self.hand_positions) > 5:
                swipe_direction = self.detect_swipe_gesture()
                if swipe_direction:
                    self.handle_swipe_navigation(swipe_direction)
            
            # Draw glassmorphic background with optimized gradient
            for i in range(2):  # Reduced from 3 to 2 layers
                alpha = 0.4 - i * 0.2
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), 
                             self.colors['background'], -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw glassmorphic title
            title = "CVGames App Store"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (title_width, title_height), _ = cv2.getTextSize(title, font, 2.5, 3)
            title_x = (self.screen_width - title_width) // 2
            
            # Title shadow
            cv2.putText(frame, title, (title_x + 3, 83), font, 2.5, (0, 0, 0), 4)
            # Main title
            cv2.putText(frame, title, (title_x, 80), font, 2.5, self.colors['text_primary'], 3)
            
            # Draw glassmorphic subtitle
            subtitle = f"Found {len(self.games)} games - Use hand gestures to navigate"
            (subtitle_width, subtitle_height), _ = cv2.getTextSize(subtitle, font, 1.0, 2)
            subtitle_x = (self.screen_width - subtitle_width) // 2
            
            # Subtitle shadow
            cv2.putText(frame, subtitle, (subtitle_x + 2, 122), font, 1.0, (0, 0, 0), 3)
            # Main subtitle
            cv2.putText(frame, subtitle, (subtitle_x, 120), font, 1.0, self.colors['text_secondary'], 2)
            
            # Draw glassmorphic interaction status
            status_text = "Interaction: DISABLED" if not self.interaction_enabled else "Interaction: ENABLED"
            status_color = self.colors['warning'] if not self.interaction_enabled else self.colors['success']
            
            # Status background
            (status_width, status_height), _ = cv2.getTextSize(status_text, font, 1, 2)
            cv2.rectangle(frame, (45, self.screen_height - 55), (45 + status_width + 10, self.screen_height - 35), 
                         self.colors['card_bg'][:3], -1)
            cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)
            
            # Status text
            cv2.putText(frame, status_text, (50, self.screen_height - 50), font, 1, status_color, 2)
            
            # Draw info button
            self.draw_info_button(frame)
            
            # Draw glassmorphic instructions
            if self.games_disabled:
                if self.game_thread and self.game_thread.is_alive():
                    if self.loading_progress >= 0.95:
                        instructions = "Game is running... | Navigation buttons still work | Click INFO for tutorial"
                    else:
                        instructions = "Game is loading... | Navigation buttons still work | Click INFO for tutorial"
                else:
                    instructions = "Game is loading... | Navigation buttons still work | Click INFO for tutorial"
            else:
                instructions = "Pinch index & middle fingers to click | Click navigation buttons or game cards | Click INFO for tutorial"
            (inst_width, inst_height), _ = cv2.getTextSize(instructions, font, 0.7, 1)
            
            # Instructions background
            cv2.rectangle(frame, (45, self.screen_height - 25), (45 + inst_width + 10, self.screen_height - 5), 
                         self.colors['card_bg'][:3], -1)
            cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)
            
            # Instructions text
            cv2.putText(frame, instructions, (50, self.screen_height - 20), font, 0.7, self.colors['text_secondary'], 1)
            
            # Draw hand trail and cursor
            if len(self.hand_positions) > 1:
                self.draw_hand_trail(frame, self.hand_positions[-5:])  # Last 5 positions
            self.draw_visual_cursor(frame, self.smooth_hand_position[0], self.smooth_hand_position[1])
            
            # Draw slide indicator if active
            if self.slide_indicator_alpha > 0:
                self.draw_slide_indicator(frame, self.last_slide_direction)
            
            # Draw loading progress if active
            if self.loading_game:
                self.draw_loading_progress(frame)
            
            # Draw exit progress if active
            if self.exiting_game:
                self.draw_exit_progress(frame)
            
            # Draw slide tutorial if active
            if self.show_slide_tutorial:
                self.draw_slide_tutorial(frame)
            
            # Draw glassmorphic game cards for current page
            current_page_start = self.current_page * self.games_per_page
            current_page_end = min(current_page_start + self.games_per_page, len(self.games))
            
            for i in range(current_page_start, current_page_end):
                card_index = i - current_page_start
                card = self.games[i]
                card_rect = self.get_card_position(card_index)
                self.draw_glassmorphic_card(frame, card, *card_rect)
            
            # Draw glassmorphic navigation buttons (only when not loading)
            if not self.loading_game and not self.exiting_game:
                if self.current_page > 0:
                    self.draw_navigation_button(frame, "left", self.left_nav_rect)
                
                if (self.current_page + 1) * self.games_per_page < len(self.games):
                    self.draw_navigation_button(frame, "right", self.right_nav_rect)
            
            # Draw glassmorphic page indicator
            total_pages = (len(self.games) + self.games_per_page - 1) // self.games_per_page
            page_text = f"Page {self.current_page + 1} of {total_pages}"
            (page_width, page_height), _ = cv2.getTextSize(page_text, font, 0.9, 2)
            page_x = (self.screen_width - page_width) // 2
            page_y = self.screen_height - 50
            
            # Page indicator background
            cv2.rectangle(frame, (page_x - 15, page_y - 25), (page_x + page_width + 15, page_y + 10), 
                         self.colors['card_bg'][:3], -1)
            cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)
            
            # Page indicator text with shadow
            cv2.putText(frame, page_text, (page_x + 2, page_y + 2), font, 0.9, (0, 0, 0), 3)
            cv2.putText(frame, page_text, (page_x, page_y), font, 0.9, self.colors['text_primary'], 2)
            
            # Page dots indicator
            self.draw_page_dots(frame, total_pages, self.current_page)
            
            cv2.imshow('CVGames App Store', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🔄 Shutting down app store...")
                break
            elif key == ord('i'):  # Toggle tutorial
                self.show_slide_tutorial = not self.show_slide_tutorial
                if self.show_slide_tutorial:
                    self.tutorial_timer = 0
                print("Slide tutorial toggled")
            elif key == ord('t'):  # Toggle slide tutorial
                self.show_slide_tutorial = not self.show_slide_tutorial
                if self.show_slide_tutorial:
                    self.tutorial_timer = 0
                print("Slide tutorial toggled")
            elif key == ord('a') and self.current_page > 0:  # Previous page
                self.current_page -= 1
            elif key == ord('d') and (self.current_page + 1) * self.games_per_page < len(self.games):  # Next page
                self.current_page += 1
            
            # Frame rate limiting
            elapsed_time = time.time() - fps_start_time
            if elapsed_time < frame_time:
                time.sleep(frame_time - elapsed_time)
            fps_start_time = time.time()
        
        # Clean up any running game thread
        if self.game_thread and self.game_thread.is_alive():
            print("🔄 Cleaning up game thread...")
            if self.game_process:
                try:
                    # Try graceful termination first
                    self.game_process.terminate()
                    self.game_process.wait(timeout=3)
                    print("✅ Game terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    print("⚠️ Force killing game process...")
                    self.game_process.kill()
                    try:
                        self.game_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        print("❌ Could not terminate game process")
                except Exception as e:
                    print(f"❌ Error during game cleanup: {e}")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main launcher function"""
    print("🎮 CVGames App Store Launcher")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check camera
    if not check_camera():
        print("\n💡 Make sure your webcam is connected and not being used by another application.")
        return
    
    print("\n🚀 Starting CVGames App Store...")
    print("💡 Press 'q' to quit the app store")
    print("💡 Press 'i' to toggle tutorial")
    print("💡 Use index-middle finger pinch to click")
    print("💡 Click on game cards to launch them")
    print("💡 Click on navigation buttons to change pages")
    print("💡 Click on INFO button for tutorial")
    print("=" * 40)
    
    try:
        app_store = AppStore()
        app_store.run()
    except KeyboardInterrupt:
        print("\n👋 App store closed by user")
    except Exception as e:
        print(f"\n❌ Error starting app store: {e}")

if __name__ == "__main__":
    main() 