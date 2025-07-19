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
from enum import Enum

# Import the loading state module
try:
    from .loading_state import LoadingState
except ImportError:
    # If loading_state doesn't exist yet, we'll create it
    LoadingState = None

class AppState(Enum):
    """Application state enumeration"""
    IDLE = "IDLE"                    # App store is idle, ready for interaction
    GAME_SELECTING = "GAME_SELECTING"  # User is selecting a game (holding over card)
    GAME_LOADING = "GAME_LOADING"      # Game is loading/launching (no interaction allowed)
    GAME_START = "GAME_START"          # Game is starting/initializing
    GAME_RUNNING = "GAME_RUNNING"      # Game is currently running
    GAME_STOP = "GAME_STOP"            # Game is being stopped/terminated
    GAME_RETURNING = "GAME_RETURNING"  # Returning from game to app store
    STORE_START = "STORE_START"        # App store is starting/initializing
    STORE_STOP = "STORE_STOP"          # App store is shutting down
    PAUSED = "PAUSED"                 # Interaction is paused
    ERROR = "ERROR"                   # Error state
    SHUTTING_DOWN = "SHUTTING_DOWN"   # App is shutting down

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
        
        # Swipe hold tracking
        self.swipe_hold_start_time = 0
        self.swipe_hold_position = (0, 0)
        self.swipe_hold_progress = 0.0
        self.swipe_hold_direction = ""
        
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
        self.zoom_factor = 2.0  # Much larger zoom for clear visual feedback
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
        
        # Shutdown flag
        self.shutting_down = False
        
        # State machine management
        self.current_state = AppState.STORE_START
        self.state_lock = threading.Lock()
        self.state_change_time = time.time()
        self.state_duration = 0.0
        
        # State-specific data
        self.selected_game = None
        self.state_message = ""
        
        # Loading state management
        self.loading_state = None
        if LoadingState:
            self.loading_state = LoadingState(self.screen_width, self.screen_height)
        
        # Load all games
        self.load_games()
        
        # Transition to IDLE when ready
        self.set_state(AppState.IDLE, "App store ready")
    

        
    def load_games(self):
        """Dynamically load all game modules from src directory"""
        # Use absolute path to ensure we're looking in the right place
        src_path = Path(__file__).parent.parent.absolute()
        print(f"🔍 Looking for games in: {src_path}")
        print(f"🔍 Current working directory: {Path.cwd()}")
        
        # List all directories first
        all_dirs = [d for d in src_path.iterdir() if d.is_dir()]
        print(f"📁 Found directories: {[d.name for d in all_dirs]}")
        
        game_dirs = [d for d in src_path.iterdir() if d.is_dir() and d.name not in ['appstore', 'cvstore', 'loading_state', '__pycache__']]
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
    
    def set_state(self, new_state: AppState, message: str = ""):
        """Thread-safe state change with logging"""
        with self.state_lock:
            old_state = self.current_state
            self.current_state = new_state
            self.state_change_time = time.time()
            self.state_duration = 0.0
            self.state_message = message
            
            # Calculate duration of previous state
            if hasattr(self, '_last_state_change'):
                duration = time.time() - self._last_state_change
                print(f"🔄 State change: {old_state.value} → {new_state.value} (duration: {duration:.1f}s) - {message}")
            else:
                print(f"🔄 State change: {old_state.value} → {new_state.value} - {message}")
            
            self._last_state_change = time.time()
    
    def get_state(self) -> AppState:
        """Thread-safe state retrieval"""
        with self.state_lock:
            return self.current_state
    
    def get_state_info(self) -> Tuple[AppState, str, float]:
        """Thread-safe state info retrieval"""
        with self.state_lock:
            duration = time.time() - self.state_change_time
            return self.current_state, self.state_message, duration
    
    def is_state(self, state: AppState) -> bool:
        """Check if current state matches"""
        return self.get_state() == state
    
    def can_interact(self) -> bool:
        """Check if user can interact based on current state"""
        state = self.get_state()
        # Only allow interaction in these states
        return state in [AppState.IDLE, AppState.GAME_SELECTING, AppState.PAUSED]
    
    def update_state_duration(self):
        """Update state duration for display"""
        with self.state_lock:
            self.state_duration = time.time() - self.state_change_time
    
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
        """Get hand position in screen coordinates - use index finger tip"""
        if len(landmarks.landmark) < 21:
            return (0, 0)
        
        # Use index finger tip (landmark 8) instead of palm center
        # Scale from processing frame (640x480) to screen coordinates (1920x1080)
        x = int(landmarks.landmark[8].x * self.screen_width)
        y = int(landmarks.landmark[8].y * self.screen_height)
        return (x, y)
    
    def get_primary_hand_position(self, hand_landmarks_list) -> Tuple[int, int]:
        """Get position from the primary hand (right hand if both raised, otherwise the raised hand)"""
        if not hand_landmarks_list:
            return (0, 0)
        
        # If only one hand, use it
        if len(hand_landmarks_list) == 1:
            return self.get_hand_position(hand_landmarks_list[0])
        
        # If both hands are raised, prioritize the right hand (hand with higher x coordinate)
        # In mirrored view, right hand appears on the left side of the screen
        right_hand = None
        left_hand = None
        
        for landmarks in hand_landmarks_list:
            # Check if hand is raised (palm facing camera)
            if self.detect_hands_raised([landmarks]):
                # In mirrored view, right hand has lower x coordinate
                if right_hand is None or landmarks.landmark[8].x < right_hand.landmark[8].x:
                    right_hand = landmarks
                if left_hand is None or landmarks.landmark[8].x > left_hand.landmark[8].x:
                    left_hand = landmarks
        
        # Return right hand position if available, otherwise left hand
        if right_hand:
            return self.get_hand_position(right_hand)
        elif left_hand:
            return self.get_hand_position(left_hand)
        else:
            # If no hands are raised, use the first hand
            return self.get_hand_position(hand_landmarks_list[0])
    
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
        """Detect swipe gestures for navigation - requires 5-second hold at screen edges"""
        if len(self.hand_positions) < 2:
            return ""
        
        current_time = time.time()
        if current_time - self.last_swipe_time < self.swipe_cooldown:
            return ""
        
        # Get the current hand position (most recent)
        current_pos = self.hand_positions[-1]
        edge_threshold = 100  # Pixels from edge
        
        # Check if hand is at left edge
        if current_pos[0] < edge_threshold:
            # Start or continue left swipe hold
            if self.swipe_hold_direction != "left":
                self.swipe_hold_start_time = current_time
                self.swipe_hold_position = current_pos
                self.swipe_hold_direction = "left"
                self.swipe_hold_progress = 0.0
                print("🔄 Started left swipe hold")
            
            # Calculate hold progress
            elapsed_time = current_time - self.swipe_hold_start_time
            self.swipe_hold_progress = min(elapsed_time / self.selection_time, 1.0)
            
            # Check if hold is complete
            if self.swipe_hold_progress >= 1.0:
                self.swipe_hold_start_time = 0
                self.swipe_hold_progress = 0.0
                self.swipe_hold_direction = ""
                return "left"
        
        # Check if hand is at right edge
        elif current_pos[0] > self.screen_width - edge_threshold:
            # Start or continue right swipe hold
            if self.swipe_hold_direction != "right":
                self.swipe_hold_start_time = current_time
                self.swipe_hold_position = current_pos
                self.swipe_hold_direction = "right"
                self.swipe_hold_progress = 0.0
                print("🔄 Started right swipe hold")
            
            # Calculate hold progress
            elapsed_time = current_time - self.swipe_hold_start_time
            self.swipe_hold_progress = min(elapsed_time / self.selection_time, 1.0)
            
            # Check if hold is complete
            if self.swipe_hold_progress >= 1.0:
                self.swipe_hold_start_time = 0
                self.swipe_hold_progress = 0.0
                self.swipe_hold_direction = ""
                return "right"
        
        else:
            # Reset swipe hold if not at edges
            if self.swipe_hold_start_time > 0:
                self.swipe_hold_start_time = 0
                self.swipe_hold_progress = 0.0
                self.swipe_hold_direction = ""
        
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
        current_state = self.get_state()
        if current_state in [AppState.GAME_LOADING, AppState.GAME_START, AppState.GAME_RUNNING, AppState.GAME_STOP, AppState.GAME_RETURNING]:
            bg_color = (50, 50, 50)  # Grayed out when game is active
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
        corner_size = int(15 * zoom)
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
        icon_size = int(min(100, width // 3) * zoom)
        icon_x = x + (width - icon_size) // 2
        icon_y = y + int(30 * zoom)
        
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
        controller_radius = int(15 * zoom)
        cv2.circle(frame, (controller_x, controller_y), controller_radius, self.colors['text_primary'], 2)
        cv2.circle(frame, (controller_x - int(8 * zoom), controller_y - int(8 * zoom)), int(3 * zoom), self.colors['success'], -1)
        cv2.circle(frame, (controller_x + int(8 * zoom), controller_y - int(8 * zoom)), int(3 * zoom), self.colors['warning'], -1)
        cv2.circle(frame, (controller_x, controller_y + int(8 * zoom)), int(3 * zoom), self.colors['accent'], -1)
        
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
            if text_width <= width - int(30 * zoom):
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Draw text lines with shadow effect
        text_y = icon_y + icon_size + int(40 * zoom)
        line_height = int(35 * zoom)
        for i, line in enumerate(lines):
            if text_y + i * line_height > y + height - int(100 * zoom):  # Leave space for description
                break
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            text_x = x + (width - text_width) // 2
            
            # Text shadow
            cv2.putText(frame, line, (text_x + 2, text_y + i * line_height + 2), font, font_scale, 
                       (0, 0, 0), thickness + 1)
            # Main text
            cv2.putText(frame, line, (text_x, text_y + i * line_height), font, font_scale, 
                       self.colors['text_primary'], thickness)
        
        # Game description with glassmorphic effect
        desc = card.description[:60] + "..." if len(card.description) > 60 else card.description
        desc_y = y + height - int(60 * zoom)
        desc_font_scale = 0.5 * zoom
        desc_thickness = 1 if zoom <= 1.0 else 2
        (desc_width, desc_height), _ = cv2.getTextSize(desc, font, desc_font_scale, desc_thickness)
        desc_x = x + (width - desc_width) // 2
        
        # Description background
        desc_bg_y = desc_y - int(15 * zoom)
        cv2.rectangle(frame, (desc_x - int(10 * zoom), desc_bg_y), (desc_x + desc_width + int(10 * zoom), desc_y + int(10 * zoom)), 
                     self.colors['card_bg'][:3], -1)
        cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)
        
        # Description text
        cv2.putText(frame, desc, (desc_x, desc_y), font, desc_font_scale, self.colors['text_secondary'], desc_thickness)
        
        # Selection indicator for simultaneous gesture
        if card.is_hovered and self.can_interact():
            center_x = x + width // 2
            center_y = y + height // 2
            
            # Draw selection progress bar
            progress_bar_width = width - int(40 * zoom)
            progress_bar_height = int(8 * zoom)
            progress_bar_x = x + int(20 * zoom)
            progress_bar_y = y + height - int(30 * zoom)
            
            # Progress bar background
            cv2.rectangle(frame, (progress_bar_x, progress_bar_y), 
                         (progress_bar_x + progress_bar_width, progress_bar_y + progress_bar_height), 
                         (50, 50, 50), -1)
            cv2.rectangle(frame, (progress_bar_x, progress_bar_y), 
                         (progress_bar_x + progress_bar_width, progress_bar_y + progress_bar_height), 
                         (255, 255, 255), 1)
            
            # Progress bar fill
            fill_width = int(progress_bar_width * card.selection_progress)
            if fill_width > 0:
                # Color changes from yellow to green as progress increases
                if card.selection_progress < 0.5:
                    progress_color = (0, 255, 255)  # Yellow
                elif card.selection_progress < 0.8:
                    progress_color = (0, 200, 255)  # Orange
                else:
                    progress_color = (0, 255, 0)  # Green
                
                cv2.rectangle(frame, (progress_bar_x, progress_bar_y), 
                             (progress_bar_x + fill_width, progress_bar_y + progress_bar_height), 
                             progress_color, -1)
            
            # Selection status text
            if card.selection_progress < 0.3:
                status_text = "HOLD TO SELECT"
                status_color = (255, 255, 255)
            elif card.selection_progress < 0.7:
                status_text = "SELECTING..."
                status_color = (255, 255, 0)
            elif card.selection_progress < 1.0:
                status_text = "ALMOST READY!"
                status_color = (0, 255, 0)
            else:
                status_text = "LAUNCHING!"
                status_color = (0, 255, 0)
            
            # Status text background
            status_font_scale = 0.6 * zoom
            status_thickness = 1 if zoom <= 1.0 else 2
            (text_width, text_height), _ = cv2.getTextSize(status_text, font, status_font_scale, status_thickness)
            text_x = center_x - text_width // 2
            text_y = progress_bar_y - int(10 * zoom)
            
            # Status background
            cv2.rectangle(frame, (text_x - int(10 * zoom), text_y - int(20 * zoom)), (text_x + text_width + int(10 * zoom), text_y + int(5 * zoom)), 
                         (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)
            
            # Status text
            cv2.putText(frame, status_text, (text_x, text_y), font, status_font_scale, status_color, status_thickness)
            
            # Time remaining indicator
            if card.selection_progress < 1.0:
                remaining_time = self.selection_time - (time.time() - card.hover_start_time)
                if remaining_time > 0:
                    time_text = f"{remaining_time:.1f}s"
                    time_font_scale = 0.5 * zoom
                    time_thickness = 1 if zoom <= 1.0 else 2
                    (time_width, time_height), _ = cv2.getTextSize(time_text, font, time_font_scale, time_thickness)
                    time_x = center_x - time_width // 2
                    time_y = text_y - int(25 * zoom)
                    
                    # Time background
                    cv2.rectangle(frame, (time_x - int(5 * zoom), time_y - int(15 * zoom)), (time_x + time_width + int(5 * zoom), time_y + int(5 * zoom)), 
                                 (0, 0, 0), -1)
                    cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)
                    
                    # Time text
                    cv2.putText(frame, time_text, (time_x, time_y), font, time_font_scale, (200, 200, 200), time_thickness)
            
            # Draw selection ring around card
            ring_color = (0, 255, 255) if card.selection_progress < 0.5 else (0, 255, 0)
            ring_thickness = int(3 + 2 * card.selection_progress)
            cv2.rectangle(frame, (x - ring_thickness, y - ring_thickness), 
                         (x + width + ring_thickness, y + height + ring_thickness), 
                         ring_color, ring_thickness)
            
            # Pulse effect for selection
            pulse_radius = int(30 + 10 * abs(np.sin(self.animation_time * 6)))
            cv2.circle(frame, (center_x, center_y), pulse_radius, ring_color, 2)
            
        elif card.is_hovered and not self.can_interact():
            center_x = x + width // 2
            center_y = y + height // 2
            
            # Draw "GAMES DISABLED" indicator
            indicator_text = "GAMES DISABLED"
            indicator_font_scale = 0.5 * zoom
            indicator_thickness = 1 if zoom <= 1.0 else 2
            (text_width, text_height), _ = cv2.getTextSize(indicator_text, font, indicator_font_scale, indicator_thickness)
            text_x = center_x - text_width // 2
            text_y = center_y + int(50 * zoom)
            
            # Indicator background (red for disabled)
            cv2.rectangle(frame, (text_x - int(15 * zoom), text_y - int(15 * zoom)), (text_x + text_width + int(15 * zoom), text_y + int(15 * zoom)), 
                         (100, 50, 50), -1)
            cv2.addWeighted(frame, 0.4, frame, 0.6, 0, frame)
            
            # Indicator text
            cv2.putText(frame, indicator_text, (text_x, text_y), font, indicator_font_scale, (255, 200, 200), indicator_thickness)
    
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
        """Draw a highly visible cursor at hand position"""
        if hand_x == 0 and hand_y == 0:
            return
        
        # Make cursor much more visible with multiple layers
        # Outer glow ring (larger, more visible)
        cv2.circle(frame, (hand_x, hand_y), 25, (255, 255, 255), 3)  # White outer ring
        
        # Main cursor ring (accent color)
        cv2.circle(frame, (hand_x, hand_y), 20, self.colors['accent'][:3], 4)
        
        # Inner circle (white)
        cv2.circle(frame, (hand_x, hand_y), 12, (255, 255, 255), -1)
        
        # Center dot (accent color)
        cv2.circle(frame, (hand_x, hand_y), 6, self.colors['accent'][:3], -1)
        
        # Crosshair lines for better precision
        cv2.line(frame, (hand_x - 15, hand_y), (hand_x + 15, hand_y), (255, 255, 255), 2)
        cv2.line(frame, (hand_x, hand_y - 15), (hand_x, hand_y + 15), (255, 255, 255), 2)
        
        # Pulse effect (more visible)
        pulse_radius = int(20 + 8 * abs(np.sin(self.animation_time * 4)))
        cv2.circle(frame, (hand_x, hand_y), pulse_radius, self.colors['accent'][:3], 2)
    
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
        """Draw the glassmorphic info button in the bottom left with hold progress feedback"""
        x, y, w, h = self.info_button_rect
        
        # Get hold progress for info button
        hold_progress = 0.0
        if hasattr(self, 'info_button_hold_start') and self.info_button_hold_start > 0:
            elapsed_time = time.time() - self.info_button_hold_start
            hold_progress = min(elapsed_time / self.selection_time, 1.0)
        
        # Glassmorphic button background
        for i in range(2):
            alpha = 0.4 - i * 0.2
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), self.colors['accent'][:3], -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Button border with progress color
        border_color = self.colors['text_primary']
        if hold_progress > 0:
            # Change border color based on progress
            if hold_progress < 0.5:
                border_color = (0, 255, 255)  # Yellow
            elif hold_progress < 0.8:
                border_color = (0, 200, 255)  # Orange
            else:
                border_color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2)
        
        # Info icon (question mark) with shadow
        cv2.putText(frame, "?", (x + 22, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(frame, "?", (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.colors['text_primary'], 2)
        
        # Hold progress indicator
        if hold_progress > 0:
            # Progress ring around button
            center_x = x + w // 2
            center_y = y + h // 2
            ring_radius = int(w // 2 + 10)
            ring_thickness = int(3 + 2 * hold_progress)
            ring_color = border_color
            cv2.circle(frame, (center_x, center_y), ring_radius, ring_color, ring_thickness)
            
            # Progress text
            if hold_progress < 0.3:
                progress_text = "HOLD"
            elif hold_progress < 0.7:
                progress_text = "HOLDING..."
            elif hold_progress < 1.0:
                progress_text = "ALMOST..."
            else:
                progress_text = "READY!"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), _ = cv2.getTextSize(progress_text, font, 0.5, 1)
            text_x = center_x - text_width // 2
            text_y = y + h + 25
            
            # Text background
            cv2.rectangle(frame, (text_x - 5, text_y - 15), (text_x + text_width + 5, text_y + 5), 
                         (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)
            
            # Text
            cv2.putText(frame, progress_text, (text_x, text_y), font, 0.5, border_color, 1)
        else:
            # Default label
            label_text = "HOLD 5s"
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
        """Draw a clickable navigation button with hold progress feedback"""
        x, y, w, h = rect
        
        # Get hold progress for this button
        hold_progress = 0.0
        if direction == "left" and hasattr(self, 'left_nav_hold_start') and self.left_nav_hold_start > 0:
            elapsed_time = time.time() - self.left_nav_hold_start
            hold_progress = min(elapsed_time / self.selection_time, 1.0)
        elif direction == "right" and hasattr(self, 'right_nav_hold_start') and self.right_nav_hold_start > 0:
            elapsed_time = time.time() - self.right_nav_hold_start
            hold_progress = min(elapsed_time / self.selection_time, 1.0)
        
        # Button background with single layer for better performance
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.colors['accent'][:3], -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Button border
        border_color = self.colors['text_primary']
        if hold_progress > 0:
            # Change border color based on progress
            if hold_progress < 0.5:
                border_color = (0, 255, 255)  # Yellow
            elif hold_progress < 0.8:
                border_color = (0, 200, 255)  # Orange
            else:
                border_color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2)
        
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
        
        # Hold progress indicator
        if hold_progress > 0:
            # Progress ring around button
            ring_radius = int(w // 2 + 10)
            ring_thickness = int(3 + 2 * hold_progress)
            ring_color = border_color
            cv2.circle(frame, (center_x, center_y), ring_radius, ring_color, ring_thickness)
            
            # Progress text
            if hold_progress < 0.3:
                progress_text = "HOLD"
            elif hold_progress < 0.7:
                progress_text = "HOLDING..."
            elif hold_progress < 1.0:
                progress_text = "ALMOST..."
            else:
                progress_text = "READY!"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), _ = cv2.getTextSize(progress_text, font, 0.5, 1)
            text_x = center_x - text_width // 2
            text_y = y + h + 20
            
            # Text background
            cv2.rectangle(frame, (text_x - 5, text_y - 15), (text_x + text_width + 5, text_y + 5), 
                         (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)
            
            # Text
            cv2.putText(frame, progress_text, (text_x, text_y), font, 0.5, border_color, 1)
        else:
            # Default instruction text
            text = "HOLD 5s"
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
            "HAND GESTURE CONTROLS",
            "",
            "🎯 CURSOR POSITION:",
            "• Cursor follows your INDEX finger tip",
            "• Right hand is prioritized when both hands are raised",
            "• Single hand: uses that hand's index finger",
            "",
            "🎮 GAME SELECTION:",
            "• HOLD your hand over a game card for 5 seconds",
            "• Watch the progress bar fill up to completion",
            "• Keep your hand steady until selection is complete",
            "• The game will launch automatically when ready",
            "",
            "⬅️➡️ NAVIGATION:",
            "• HOLD your hand over LEFT/RIGHT navigation buttons for 5 seconds",
            "• Watch for progress feedback on the buttons",
            "• Keep your hand steady until navigation completes",
            "",
            "🔄 SWIPE GESTURE:",
            "• HOLD your hand at the LEFT EDGE of screen for 5 seconds to swipe left",
            "• HOLD your hand at the RIGHT EDGE of screen for 5 seconds to swipe right",
            "• Watch for progress feedback at the screen edges",
            "",
            "ℹ️ INFO BUTTON:",
            "• HOLD your hand over the INFO button for 5 seconds",
            "• This will open the tutorial (this screen)",
            "",
            "⏸️ PAUSE/RESUME:",
            "• Both hands raised + closed fists = pause interaction",
            "• Both hands raised + open palms = resume interaction",
            "",
            "HOLD anywhere on this screen for 5 seconds to close tutorial",
            "",
            f"Tutorial will close in {max(0, int(self.tutorial_duration - self.tutorial_timer))} seconds"
        ]
        
        # Draw tutorial text
        y_start = self.screen_height // 2 - 200
        for i, text in enumerate(tutorial_text):
            y = y_start + i * 35
            if text.startswith("HAND GESTURE CONTROLS"):
                # Main title
                cv2.putText(frame, text, (self.screen_width // 2 - 200, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['accent'][:3], 3)
                cv2.putText(frame, text, (self.screen_width // 2 - 200, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['text_primary'], 2)
            elif text.startswith("🎯") or text.startswith("👆") or text.startswith("🔄") or text.startswith("⏸️"):
                # Section headers
                cv2.putText(frame, text, (self.screen_width // 2 - 200, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['accent'][:3], 2)
            elif text.startswith("•"):
                # Bullet points
                cv2.putText(frame, text, (self.screen_width // 2 - 180, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_secondary'], 1)
            elif text.startswith("Tutorial will close"):
                # Timer
                cv2.putText(frame, text, (self.screen_width // 2 - 180, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_secondary'], 2)
            elif text == "":
                # Empty lines
                continue
            else:
                # Regular text
                cv2.putText(frame, text, (self.screen_width // 2 - 150, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_secondary'], 1)
        
        # Draw example hold-to-select gesture
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2 + 50
        
        # Draw a game card outline
        card_width = 120
        card_height = 80
        card_x = center_x - card_width // 2
        card_y = center_y - card_height // 2
        
        # Card background
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_width, card_y + card_height), 
                     self.colors['accent'][:3], 2)
        
        # Draw hand cursor over the card
        hand_x = center_x
        hand_y = center_y - 10
        
        # Hand cursor
        cv2.circle(frame, (hand_x, hand_y), 15, (255, 255, 255), 3)
        cv2.circle(frame, (hand_x, hand_y), 10, self.colors['accent'][:3], -1)
        
        # Progress bar on the card
        progress_width = card_width - 20
        progress_height = 6
        progress_x = card_x + 10
        progress_y = card_y + card_height - 15
        
        # Progress bar background
        cv2.rectangle(frame, (progress_x, progress_y), (progress_x + progress_width, progress_y + progress_height), 
                     (50, 50, 50), -1)
        
        # Progress bar fill (partially filled to show progress)
        fill_width = int(progress_width * 0.7)  # 70% filled
        cv2.rectangle(frame, (progress_x, progress_y), (progress_x + fill_width, progress_y + progress_height), 
                     (0, 255, 0), -1)
        
        cv2.putText(frame, "HOLD TO SELECT", (center_x - 60, center_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent'][:3], 2)
    
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
        """Launch a game using the LoadingState"""
        try:
            print(f"🚀 Launching {game_card.name} via LoadingState...")
            
            if self.loading_state:
                # Start loading state with callback
                self.loading_state.start_loading(
                    game_card.name, 
                    game_card.module_path,
                    on_complete=self._on_loading_complete
                )
                self.set_state(AppState.GAME_LOADING, f"Loading: {game_card.name}")
            else:
                # Fallback to old method if LoadingState not available
                print("⚠️ LoadingState not available, using fallback method")
                self._launch_game_fallback(game_card)
                
        except Exception as e:
            print(f"❌ Error launching {game_card.name}: {e}")
            self.set_state(AppState.ERROR, f"Failed to launch: {game_card.name}")
            self.set_state(AppState.IDLE, "Ready for selection")
    
    def _on_loading_complete(self):
        """Callback when loading is complete"""
        print("✅ LoadingState: Loading complete, transitioning to game running")
        self.set_state(AppState.GAME_RUNNING, f"Running: {self.selected_game.name if self.selected_game else 'Game'}")
    
    def _launch_game_fallback(self, game_card: GameCard):
        """Fallback game launching method"""
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
    
    def _run_game_in_thread(self, game_card: GameCard):
        """Run game in a separate thread"""
        try:
            # Change to the game directory
            game_dir = Path(game_card.module_path)
            init_file = game_dir / '__init__.py'
            
            if init_file.exists():
                print(f"🎮 Starting {game_card.name} in thread...")
                self.set_state(AppState.GAME_START, f"Starting: {game_card.name}")
                
                # Use subprocess to run the game
                self.game_process = subprocess.Popen([sys.executable, str(init_file)], 
                                                   cwd=str(game_dir))
                
                # Wait for the game to finish or for shutdown signal
                while self.game_process.poll() is None and not self.shutting_down:
                    time.sleep(0.1)  # Check every 100ms
                
                # If we're shutting down, terminate the game
                if self.shutting_down and self.game_process.poll() is None:
                    print(f"🔄 Terminating {game_card.name} due to app store shutdown...")
                    self.set_state(AppState.GAME_STOP, f"Stopping: {game_card.name}")
                    try:
                        self.game_process.terminate()
                        self.game_process.wait(timeout=2)
                    except:
                        self.game_process.kill()
                else:
                    print(f"✅ {game_card.name} finished")
                
                # Signal game completion (thread-safe)
                with self.game_lock:
                    self.game_completed = True
                
            else:
                print(f"❌ Game file not found: {init_file}")
                self.set_state(AppState.ERROR, f"Game file not found: {game_card.name}")
                with self.game_lock:
                    self.game_completed = True
                
        except subprocess.TimeoutExpired:
            print(f"⚠️ {game_card.name} timed out")
            self.set_state(AppState.ERROR, f"Game timed out: {game_card.name}")
            if self.game_process:
                self.game_process.kill()
            with self.game_lock:
                self.game_completed = True
        except Exception as e:
            print(f"❌ Error running {game_card.name}: {e}")
            self.set_state(AppState.ERROR, f"Error running: {game_card.name}")
            with self.game_lock:
                self.game_completed = True
        finally:
            # Return to the original directory
            os.chdir(Path(__file__).parent.parent)
            # Return to idle state when game ends (only if not shutting down)
            if not self.shutting_down:
                self.set_state(AppState.GAME_RETURNING, "Returning to app store")
                print("✅ Returning to app store")
    
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
            
            # Update state duration
            self.update_state_duration()
            
            # Update tutorial timer
            if self.show_slide_tutorial:
                self.tutorial_timer += 0.016
                if self.tutorial_timer >= self.tutorial_duration:
                    self.show_slide_tutorial = False
            
            # Update loading progress
            if self.loading_state and self.is_state(AppState.GAME_LOADING):
                # Use LoadingState for loading management
                loading_complete = self.loading_state.update(0.033)  # 30 FPS
                if loading_complete:
                    # Loading is complete, LoadingState will call the callback
                    pass
            elif self.loading_game:
                # Fallback loading logic
                elapsed_time = time.time() - self.loading_start_time
                self.loading_progress = min(elapsed_time / self.loading_duration, 1.0)
                
                # Check if game thread is still running
                if self.game_thread and not self.game_thread.is_alive():
                    # Thread finished but we haven't detected completion yet
                    with self.game_lock:
                        if not self.game_completed:
                            self.game_completed = True
                
                # State transitions based on loading progress
                current_state = self.get_state()
                if current_state == AppState.GAME_LOADING:
                    if self.loading_progress >= 1.0 and self.game_thread and self.game_thread.is_alive():
                        # Loading complete, transition to running
                        self.set_state(AppState.GAME_RUNNING, f"Running: {self.loading_game.name}")
                        self.loading_progress = 0.95  # Keep minimal loading indicator
                    elif self.loading_progress >= 1.0 and (not self.game_thread or not self.game_thread.is_alive()):
                        # Loading complete but thread not running - error state
                        self.set_state(AppState.ERROR, f"Failed to start: {self.loading_game.name}")
                        self.loading_game = None
            
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
                    self.set_state(AppState.IDLE, "Ready for selection")
            
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
            if results.multi_hand_landmarks and self.can_interact():
                # Check for pause gesture: both hands raised and both closed
                if len(results.multi_hand_landmarks) >= 2:
                    if self.detect_pause_gesture(results.multi_hand_landmarks):
                        current_time = time.time()
                        if current_time - self.last_hands_raised_time > self.interaction_timeout:
                            self.interaction_enabled = False
                            self.interaction_timer = current_time
                            self.last_hands_raised_time = current_time
                            self.set_state(AppState.PAUSED, "Interaction paused")
                            print("Interaction paused - both hands raised and closed")
                
                # Check for resume gesture: both hands raised and both open
                if len(results.multi_hand_landmarks) >= 2:
                    if self.detect_resume_gesture(results.multi_hand_landmarks):
                        if not self.interaction_enabled:
                            self.interaction_enabled = True
                            self.set_state(AppState.IDLE, "Interaction resumed")
                            print("Interaction resumed - both hands raised and open")
                
                # Get primary hand position (right hand if both raised, otherwise the raised hand)
                primary_hand_x, primary_hand_y = self.get_primary_hand_position(results.multi_hand_landmarks)
                self.update_smooth_hand_position((primary_hand_x, primary_hand_y))
                
                # Process hand gestures for game selection and info button
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw only dots on finger tips instead of full hand landmarks
                    self.draw_hand_dots(frame, hand_landmarks)
                    
                    # Get individual hand position for gesture detection
                    hand_x, hand_y = self.get_hand_position(hand_landmarks)
                    
                    # Check for info button hover and hold
                    if self.is_point_in_rect((primary_hand_x, primary_hand_y), self.info_button_rect):
                        # Start or continue info button hold
                        if not hasattr(self, 'info_button_hold_start'):
                            self.info_button_hold_start = time.time()
                            print("ℹ️ Started info button hold")
                        
                        # Calculate hold progress
                        elapsed_time = time.time() - self.info_button_hold_start
                        info_hold_progress = min(elapsed_time / self.selection_time, 1.0)
                        
                        # Check if hold is complete
                        if info_hold_progress >= 1.0:
                            self.show_slide_tutorial = True
                            self.tutorial_timer = 0
                            self.info_button_hold_start = 0
                            print("ℹ️ Info tutorial opened via hold")
                    else:
                        # Reset info button hold if not hovering
                        if hasattr(self, 'info_button_hold_start'):
                            self.info_button_hold_start = 0
                    
                    # Check for navigation button hover and hold
                    if self.is_point_in_rect((primary_hand_x, primary_hand_y), self.left_nav_rect):
                        # Start or continue left nav hold
                        if not hasattr(self, 'left_nav_hold_start'):
                            self.left_nav_hold_start = time.time()
                            print("⬅️ Started left navigation hold")
                        
                        # Calculate hold progress
                        elapsed_time = time.time() - self.left_nav_hold_start
                        left_hold_progress = min(elapsed_time / self.selection_time, 1.0)
                        
                        # Check if hold is complete
                        if left_hold_progress >= 1.0 and self.current_page > 0:
                            self.current_page -= 1
                            self.left_nav_hold_start = 0
                            print("⬅️ Previous page via hold")
                    else:
                        # Reset left nav hold if not hovering
                        if hasattr(self, 'left_nav_hold_start'):
                            self.left_nav_hold_start = 0
                    
                    if self.is_point_in_rect((primary_hand_x, primary_hand_y), self.right_nav_rect):
                        # Start or continue right nav hold
                        if not hasattr(self, 'right_nav_hold_start'):
                            self.right_nav_hold_start = time.time()
                            print("➡️ Started right navigation hold")
                        
                        # Calculate hold progress
                        elapsed_time = time.time() - self.right_nav_hold_start
                        right_hold_progress = min(elapsed_time / self.selection_time, 1.0)
                        
                        # Check if hold is complete
                        if right_hold_progress >= 1.0 and (self.current_page + 1) * self.games_per_page < len(self.games):
                            self.current_page += 1
                            self.right_nav_hold_start = 0
                            print("➡️ Next page via hold")
                    else:
                        # Reset right nav hold if not hovering
                        if hasattr(self, 'right_nav_hold_start'):
                            self.right_nav_hold_start = 0
                    
                    # Check for tutorial close with hold
                    if self.show_slide_tutorial and self.is_point_in_rect((primary_hand_x, primary_hand_y), (0, 0, self.screen_width, self.screen_height)):
                        # Start or continue tutorial close hold
                        if not hasattr(self, 'tutorial_close_hold_start'):
                            self.tutorial_close_hold_start = time.time()
                            print("❌ Started tutorial close hold")
                        
                        # Calculate hold progress
                        elapsed_time = time.time() - self.tutorial_close_hold_start
                        tutorial_hold_progress = min(elapsed_time / self.selection_time, 1.0)
                        
                        # Check if hold is complete
                        if tutorial_hold_progress >= 1.0:
                            self.show_slide_tutorial = False
                            self.tutorial_close_hold_start = 0
                            print("❌ Tutorial closed via hold")
                    else:
                        # Reset tutorial close hold if not hovering
                        if hasattr(self, 'tutorial_close_hold_start'):
                            self.tutorial_close_hold_start = 0
                
                # Check for hover effects using primary hand position
                current_page_start = self.current_page * self.games_per_page
                current_page_end = min(current_page_start + self.games_per_page, len(self.games))
                
                for i in range(current_page_start, current_page_end):
                    card_index = i - current_page_start
                    card = self.games[i]
                    card_rect = self.get_card_position(card_index)
                    
                    if self.is_point_in_rect((primary_hand_x, primary_hand_y), card_rect):
                        card.is_hovered = True
                        
                        # Only allow selection if we can interact
                        if not self.can_interact():
                            break
                        
                        # Start or continue selection timer
                        if card.hover_start_time == 0:
                            card.hover_start_time = time.time()
                            self.set_state(AppState.GAME_SELECTING, f"Selecting: {card.name}")
                            print(f"🎯 Started selection timer for: {card.name}")
                        
                        # Calculate selection progress
                        elapsed_time = time.time() - card.hover_start_time
                        card.selection_progress = min(elapsed_time / self.selection_time, 1.0)
                        
                        # Check if selection is complete
                        if card.selection_progress >= 1.0:
                            print(f"🎮 Selection complete for: {card.name} - launching game!")
                            self.selected_game = card
                            self.set_state(AppState.GAME_LOADING, f"Loading: {card.name}")
                            self.launch_game(card)
                            # Reset all selection states
                            for reset_card in self.games:
                                reset_card.hover_start_time = 0
                                reset_card.selection_progress = 0.0
                        break
                    else:
                        # Reset selection timer if not hovering
                        if card.hover_start_time > 0:
                            card.hover_start_time = 0
                            card.selection_progress = 0.0
                            if self.is_state(AppState.GAME_SELECTING):
                                self.set_state(AppState.IDLE, "Ready for selection")
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
            
            # Draw application status
            self.draw_application_status(frame)
            
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
            current_state = self.get_state()
            if current_state == AppState.GAME_LOADING:
                instructions = "Game is loading... | Please wait | No interaction available during loading"
            elif current_state == AppState.GAME_START:
                instructions = "Game is starting... | Please wait | No interaction available during startup"
            elif current_state == AppState.GAME_RUNNING:
                instructions = "Game is running... | Hold over buttons for 5 seconds to interact | Click INFO for tutorial"
            elif current_state == AppState.GAME_STOP:
                instructions = "Game is stopping... | Please wait | No interaction available during shutdown"
            elif current_state == AppState.GAME_RETURNING:
                instructions = "Returning to app store... | Please wait | No interaction available during return"
            else:
                instructions = "HOLD hand over game card for 5 seconds to select | HOLD over buttons for 5 seconds to interact | HOLD at screen edges for 5 seconds to swipe | Click INFO for tutorial"
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
            
            # Draw swipe hold progress
            self.draw_swipe_hold_progress(frame)
            
            # Draw loading progress if active
            if self.loading_game:
                self.draw_loading_progress(frame)
            
            # Draw dedicated loading screen if in loading state
            if self.loading_state and self.is_state(AppState.GAME_LOADING):
                self.loading_state.draw(frame)
            
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
                print("🔄 Shutting down app store and terminating any running games...")
                # Set flag to indicate we're shutting down
                self.shutting_down = True
                
                # Shutdown loading state if active
                if self.loading_state:
                    self.loading_state.shutdown()
                
                self.set_state(AppState.STORE_STOP, "Shutting down app store")
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
        
        # Enhanced cleanup: terminate any running game and close everything
        print("🔄 Performing complete cleanup...")
        
        # First, terminate any running game process
        if self.game_process:
            print("🔄 Terminating running game...")
            try:
                # Try graceful termination first
                self.game_process.terminate()
                print("⏳ Waiting for game to terminate gracefully...")
                self.game_process.wait(timeout=3)
                print("✅ Game terminated gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination fails
                print("⚠️ Force killing game process...")
                try:
                    self.game_process.kill()
                    self.game_process.wait(timeout=2)
                    print("✅ Game force killed")
                except subprocess.TimeoutExpired:
                    print("❌ Could not terminate game process - it may still be running")
                except Exception as e:
                    print(f"❌ Error force killing game: {e}")
            except Exception as e:
                print(f"❌ Error during game cleanup: {e}")
        
        # Kill any remaining game threads
        if self.game_thread and self.game_thread.is_alive():
            print("🔄 Cleaning up game thread...")
            # Note: Python threads can't be forcefully killed, but the process termination should handle this
        
        # Release camera and close windows
        print("🔄 Releasing camera and closing windows...")
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Force close any remaining OpenCV windows
        for i in range(10):  # Try multiple times to ensure all windows are closed
            cv2.waitKey(1)
        
        print("✅ App store shutdown complete")
    
    def draw_swipe_hold_progress(self, frame):
        """Draw swipe hold progress at screen edges"""
        if self.swipe_hold_progress <= 0:
            return
        
        # Determine which edge to show progress
        if self.swipe_hold_direction == "left":
            # Left edge progress
            edge_x = 50
            edge_y = self.screen_height // 2
            direction_text = "SWIPE LEFT"
        elif self.swipe_hold_direction == "right":
            # Right edge progress
            edge_x = self.screen_width - 50
            edge_y = self.screen_height // 2
            direction_text = "SWIPE RIGHT"
        else:
            return
        
        # Progress ring
        ring_radius = 40
        ring_thickness = int(5 + 3 * self.swipe_hold_progress)
        
        # Color based on progress
        if self.swipe_hold_progress < 0.5:
            ring_color = (0, 255, 255)  # Yellow
        elif self.swipe_hold_progress < 0.8:
            ring_color = (0, 200, 255)  # Orange
        else:
            ring_color = (0, 255, 0)  # Green
        
        # Draw progress ring
        cv2.circle(frame, (edge_x, edge_y), ring_radius, ring_color, ring_thickness)
        
        # Progress text
        if self.swipe_hold_progress < 0.3:
            progress_text = "HOLD"
        elif self.swipe_hold_progress < 0.7:
            progress_text = "HOLDING..."
        elif self.swipe_hold_progress < 1.0:
            progress_text = "ALMOST..."
        else:
            progress_text = "READY!"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(progress_text, font, 0.6, 1)
        text_x = edge_x - text_width // 2
        text_y = edge_y + ring_radius + 30
        
        # Text background
        cv2.rectangle(frame, (text_x - 10, text_y - 15), (text_x + text_width + 10, text_y + 5), 
                     (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)
        
        # Text
        cv2.putText(frame, progress_text, (text_x, text_y), font, 0.6, ring_color, 1)
        
        # Direction text
        (dir_width, dir_height), _ = cv2.getTextSize(direction_text, font, 0.5, 1)
        dir_x = edge_x - dir_width // 2
        dir_y = text_y + 25
        
        cv2.putText(frame, direction_text, (dir_x, dir_y), font, 0.5, (200, 200, 200), 1)

    def draw_application_status(self, frame):
        """Draw the current application status with clear state information"""
        current_state, state_message, state_duration = self.get_state_info()
        
        # Status background
        status_bg_width = 400
        status_bg_height = 80
        status_bg_x = self.screen_width - status_bg_width - 20
        status_bg_y = 20
        
        # Status background with glassmorphic effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (status_bg_x, status_bg_y), (status_bg_x + status_bg_width, status_bg_y + status_bg_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status border
        border_color = (255, 255, 255)
        if current_state == AppState.IDLE:
            border_color = (0, 255, 0)  # Green for idle
        elif current_state == AppState.GAME_SELECTING:
            border_color = (0, 255, 255)  # Yellow for selecting
        elif current_state == AppState.GAME_LOADING:
            border_color = (255, 165, 0)  # Orange for loading
        elif current_state == AppState.GAME_START:
            border_color = (255, 140, 0)  # Dark orange for starting
        elif current_state == AppState.GAME_RUNNING:
            border_color = (0, 255, 0)  # Green for running
        elif current_state == AppState.GAME_STOP:
            border_color = (255, 0, 0)  # Red for stopping
        elif current_state == AppState.GAME_RETURNING:
            border_color = (0, 191, 255)  # Blue for returning
        elif current_state == AppState.STORE_START:
            border_color = (0, 255, 255)  # Cyan for store starting
        elif current_state == AppState.STORE_STOP:
            border_color = (255, 0, 0)  # Red for store stopping
        elif current_state == AppState.PAUSED:
            border_color = (128, 128, 128)  # Gray for paused
        elif current_state == AppState.ERROR:
            border_color = (255, 0, 0)  # Red for error
        elif current_state == AppState.SHUTTING_DOWN:
            border_color = (255, 0, 0)  # Red for shutting down
        
        cv2.rectangle(frame, (status_bg_x, status_bg_y), (status_bg_x + status_bg_width, status_bg_y + status_bg_height), 
                     border_color, 2)
        
        # State text
        font = cv2.FONT_HERSHEY_SIMPLEX
        state_text = f"STATUS: {current_state.value}"
        (state_width, state_height), _ = cv2.getTextSize(state_text, font, 0.8, 2)
        state_x = status_bg_x + 10
        state_y = status_bg_y + 30
        
        cv2.putText(frame, state_text, (state_x, state_y), font, 0.8, border_color, 2)
        
        # Message text
        if state_message:
            message_text = state_message
            (message_width, message_height), _ = cv2.getTextSize(message_text, font, 0.6, 1)
            message_x = status_bg_x + 10
            message_y = status_bg_y + 55
            
            # Truncate message if too long
            if message_width > status_bg_width - 20:
                while message_width > status_bg_width - 20 and len(message_text) > 0:
                    message_text = message_text[:-1]
                    (message_width, message_height), _ = cv2.getTextSize(message_text + "...", font, 0.6, 1)
                message_text += "..."
            
            cv2.putText(frame, message_text, (message_x, message_y), font, 0.6, (255, 255, 255), 1)
        
        # Duration text
        duration_text = f"Duration: {state_duration:.1f}s"
        (duration_width, duration_height), _ = cv2.getTextSize(duration_text, font, 0.5, 1)
        duration_x = status_bg_x + status_bg_width - duration_width - 10
        duration_y = status_bg_y + 20
        
        cv2.putText(frame, duration_text, (duration_x, duration_y), font, 0.5, (200, 200, 200), 1)



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
    print("💡 Press 'q' to quit the app store and terminate any running games")
    print("💡 Press 'i' to toggle tutorial")
    print("💡 Use index-middle finger pinch to click")
    print("💡 Click on game cards to launch them")
    print("💡 Click on navigation buttons to change pages")
    print("💡 Click on INFO button for tutorial")
    print("=" * 40)
    
    app_store = None
    try:
        app_store = AppStore()
        app_store.run()
    except KeyboardInterrupt:
        print("\n👋 App store closed by user (Ctrl+C)")
        if app_store:
            app_store.shutting_down = True
    except Exception as e:
        print(f"\n❌ Error starting app store: {e}")
    finally:
        # Ensure cleanup happens even if there's an exception
        if app_store:
            print("🔄 Ensuring complete cleanup...")
            app_store.shutting_down = True
            # Force cleanup of any remaining processes
            if hasattr(app_store, 'game_process') and app_store.game_process:
                try:
                    app_store.game_process.terminate()
                    app_store.game_process.wait(timeout=1)
                except:
                    try:
                        app_store.game_process.kill()
                    except:
                        pass
            if hasattr(app_store, 'cap'):
                app_store.cap.release()
            cv2.destroyAllWindows()
            for i in range(5):
                cv2.waitKey(1)
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main() 