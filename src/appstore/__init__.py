import cv2
import mediapipe as mp
import json
import threading
import subprocess
import time
import platform
import os
from pathlib import Path
from typing import List, Tuple
from enum import Enum
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

class AppState(Enum):
    """Application state enumeration"""
    IDLE = "IDLE"                    # App store is idle, ready for interaction
    GAME_SELECTING = "GAME_SELECTING"  # User is selecting a game (holding over card)
    GAME_LOADING = "GAME_LOADING"      # Game is loading/launching (no interaction allowed)
    GAME_START = "GAME_START"          # Game is starting/initializing
    GAME_RUNNING = "GAME_RUNNING"      # Game is currently running
    GAME_STOP = "GAME_STOP"            # Game is being stopped/terminated
    GAME_RETURNING = "GAME_RETURNING"  # Returning from game to app store
    GAME_EXITING = "GAME_EXITING"      # Game is exiting with free movement period
    STORE_START = "STORE_START"        # App store is starting/initializing
    STORE_STOP = "STORE_STOP"          # App store is shutting down
    PAUSED = "PAUSED"                 # Interaction is paused
    ERROR = "ERROR"                   # Error state
    SHUTTING_DOWN = "SHUTTING_DOWN"   # App is shutting down

def detect_screen_resolution():
    """Detect the primary screen resolution"""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide the window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        return screen_width, screen_height
    except Exception as e:
        print(f"⚠️ Could not detect screen resolution: {e}")
        print("🖥️ Using default resolution: 1920x1080")
        return 1920, 1080

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

def detect_os_theme():
    """Detect the current OS theme (dark/light)"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        try:
            result = subprocess.run(['defaults', 'read', '-g', 'AppleInterfaceStyle'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                theme = result.stdout.strip()
                return "dark" if theme == "Dark" else "light"
        except:
            pass
        return "dark"
    
    elif system == "Windows":
        try:
            result = subprocess.run(['reg', 'query', 'HKCU', '/v', 'AppsUseLightTheme'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                if "0x1" in result.stdout:
                    return "light"
                else:
                    return "dark"
        except:
            pass
        return "light"
    
    elif system == "Linux":
        return "dark"
    
    return "dark"

def get_apple_style_colors(theme="dark"):
    """Get Apple-style color palette based on theme"""
    if theme == "dark":
        return {
            # Background colors
            'background': (8, 8, 12),
            'surface': (18, 18, 24),
            'surface_secondary': (28, 28, 36),
            
            # Card colors
            'card_bg': (24, 24, 32, 200),
            'card_border': (60, 60, 80, 120),
            'card_hover': (40, 40, 56, 220),
            'card_selected': (32, 80, 48, 240),
            'card_shadow': (0, 0, 0, 100),
            
            # Text colors
            'text_primary': (255, 255, 255),
            'text_secondary': (180, 180, 200),
            'text_tertiary': (120, 120, 140),
            
            # Accent colors
            'accent_primary': (0, 122, 255),
            'accent_secondary': (88, 86, 214),
            'accent_success': (52, 199, 89),
            'accent_warning': (255, 149, 0),
            'accent_error': (255, 59, 48),
            
            # Interactive elements
            'button_bg': (40, 40, 56, 180),
            'button_hover': (60, 60, 80, 200),
            'button_pressed': (20, 20, 28, 220),
            
            # Status colors
            'status_idle': (52, 199, 89),
            'status_loading': (255, 149, 0),
            'status_error': (255, 59, 48),
            'status_warning': (255, 204, 0),
        }
    else:  # Light theme
        return {
            # Background colors
            'background': (248, 248, 250),
            'surface': (255, 255, 255),
            'surface_secondary': (242, 242, 247),
            
            # Card colors
            'card_bg': (255, 255, 255, 240),
            'card_border': (200, 200, 220, 100),
            'card_hover': (245, 245, 250, 250),
            'card_selected': (240, 248, 240, 255),
            'card_shadow': (0, 0, 0, 60),
            
            # Text colors
            'text_primary': (0, 0, 0),
            'text_secondary': (60, 60, 67),
            'text_tertiary': (142, 142, 147),
            
            # Accent colors
            'accent_primary': (0, 122, 255),
            'accent_secondary': (88, 86, 214),
            'accent_success': (52, 199, 89),
            'accent_warning': (255, 149, 0),
            'accent_error': (255, 59, 48),
            
            # Interactive elements
            'button_bg': (245, 245, 250, 200),
            'button_hover': (235, 235, 240, 220),
            'button_pressed': (225, 225, 230, 240),
            
            # Status colors
            'status_idle': (52, 199, 89),
            'status_loading': (255, 149, 0),
            'status_error': (255, 59, 48),
            'status_warning': (255, 204, 0),
        }

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
        # Initialize MediaPipe Hands
        try:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe Hands initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing MediaPipe Hands: {e}")
            print("⚠️ Running in fallback mode without hand tracking")
            self.hands = None
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Screen setup
        self.screen_width, self.screen_height = detect_screen_resolution()
        print(f"🖥️ Detected screen resolution: {self.screen_width}x{self.screen_height}")
        
        # Initialize camera with fallback
        self.cap = None
        self.current_camera_index = 0
        self.select_camera(start_index=0)
        
        # App Store state
        self.games: List[GameCard] = []
        self.current_page = 0
        self.games_per_page = 10
        self.interaction_enabled = True
        self.interaction_timer = 0
        self.interaction_timeout = 2.0
        self.last_hands_raised_time = 0
        
        # Animation and UX state
        self.animation_time = 0
        self.hover_animation_speed = 0.1
        self.card_animations = {}
        self.last_hand_position = (0, 0)
        self.hand_velocity = (0, 0)
        self.smooth_hand_position = (0, 0)
        
        # Soft cursor state
        self.cursor_position = (0, 0)
        self.cursor_visible = False
        self.cursor_size = 20
        self.cursor_alpha = 0.8
        self.cursor_smooth_factor = 0.3
        self.cursor_interactive = False
        self.cursor_hover_target = None
        
        # Navigation and gesture state
        self.hand_positions = []
        self.swipe_threshold = 100
        self.last_swipe_time = 0
        self.swipe_cooldown = 1.0
        self.slide_indicator_alpha = 0.0
        self.last_slide_direction = ""
        self.show_slide_tutorial = False
        self.tutorial_timer = 0
        self.tutorial_duration = 5.0
        
        # Swipe hold tracking
        self.swipe_hold_start_time = 0
        self.swipe_hold_position = (0, 0)
        self.swipe_hold_progress = 0.0
        self.swipe_hold_direction = ""
        self.swipe_continuous_mode = False
        self.swipe_continuous_start_time = 0
        self.swipe_continuous_cooldown = 3.0
        self.swipe_continuous_direction = ""
        
        # Loading and game state
        self.loading_game = None
        self.loading_progress = 0.0
        self.loading_start_time = 0
        self.loading_duration = 3.0
        self.game_process = None
        self.game_thread = None
        self.game_completed = False
        self.exiting_game = False
        self.exit_progress = 0.0
        self.games_disabled = False
        
        # UI constants
        self.card_width = 280
        self.card_height = 380
        self.card_margin = 40
        self.zoom_factor = 1.2
        self.selection_time = 3.0
        
        # Info modal state
        self.show_info_modal = False
        self.info_button_rect = (50, self.screen_height - 120, 60, 60)
        
        # Navigation button regions
        self.left_nav_rect = (80, self.screen_height // 2 - 40, 80, 80)
        self.right_nav_rect = (self.screen_width - 160, self.screen_height // 2 - 40, 80, 80)
        
        # Detect OS theme and initialize colors
        self.current_theme = detect_os_theme()
        print(f"🎨 Detected OS theme: {self.current_theme}")
        self.colors = get_apple_style_colors(self.current_theme)
        print(f"✅ Apple-style color palette initialized for {self.current_theme} theme")
        
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
        
        # Game timeout management
        self.game_start_timeout = 10.0
        self.game_start_time = 0.0
        
        # Load demo games
        self.load_demo_games()
        
        # Pygame setup
        pygame.init()
        self.screen_width, self.screen_height = detect_screen_resolution()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("CVGames App Store (CV+Pygame)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 32)
        self.small_font = pygame.font.SysFont("Arial", 20)
        self.shutting_down = False
        
        # Set up OpenTelemetry tracing
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(__name__)
        span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Create the main window in full screen
        print("🖥️ App store window set to full screen mode")
        
        # Transition to IDLE when ready
        self.set_state(AppState.IDLE, "App store ready")
    
    def load_demo_games(self):
        """Load games from games.json file"""
        try:
            # Try to load from games.json file
            games_json_path = Path(__file__).parent.parent / "cvstore" / "games.json"
            if games_json_path.exists():
                with open(games_json_path, 'r', encoding='utf-8') as f:
                    games_data = json.load(f)
                
                for game_data in games_data['games']:
                    # Use the route field from games.json
                    module_name = game_data.get('route', game_data['title'].lower().replace(' ', '').replace('-', '').replace('_', ''))
                    
                    game_card = GameCard(
                        name=game_data['title'],
                        description=game_data['description'],
                        rules=f"Category: {game_data['category']}, Age: {game_data['age_range']}, Difficulty: {game_data['difficulty']}",
                        icon_path='',  # Will be implemented later
                        module_path=module_name
                    )
                    self.games.append(game_card)
                
                print(f"✅ Loaded {len(self.games)} games from games.json")
            else:
                print(f"⚠️ games.json not found at {games_json_path}, loading fallback games")
                self._load_fallback_games()
                
        except Exception as e:
            print(f"❌ Error loading games from games.json: {e}")
            print("⚠️ Loading fallback games")
            self._load_fallback_games()
    
    def _load_fallback_games(self):
        """Load fallback demo games if games.json is not available"""
        demo_games = [
            {
                'name': 'Hand Tracker',
                'description': 'Track your hand movements and gestures in real-time.',
                'rules': 'Move your hand around the screen to see tracking in action.',
                'icon_path': '',
                'module_path': 'hand_tracker'
            },
            {
                'name': 'Gesture Game',
                'description': 'Control objects using various hand gestures.',
                'rules': 'Use different hand poses to interact with game elements.',
                'icon_path': '',
                'module_path': 'gesture_game'
            },
            {
                'name': 'Air Piano',
                'description': 'Play piano notes by tapping in the air.',
                'rules': 'Tap different areas to play different musical notes.',
                'icon_path': '',
                'module_path': 'air_piano'
            },
            {
                'name': 'Virtual Pong',
                'description': 'Classic Pong game controlled with hand movements.',
                'rules': 'Move your hand up and down to control the paddle.',
                'icon_path': '',
                'module_path': 'virtual_pong'
            },
            {
                'name': 'Shape Matcher',
                'description': 'Match shapes using hand gestures and poses.',
                'rules': 'Create shapes with your hand to match the target.',
                'icon_path': '',
                'module_path': 'shape_matcher'
            },
            {
                'name': 'Color Catcher',
                'description': 'Catch falling colored objects with your hands.',
                'rules': 'Move your hands to catch objects of the right color.',
                'icon_path': '',
                'module_path': 'color_catcher'
            }
        ]
        
        for game_data in demo_games:
            game_card = GameCard(
                name=game_data['name'],
                description=game_data['description'],
                rules=game_data['rules'],
                icon_path=game_data['icon_path'],
                module_path=game_data['module_path']
            )
            self.games.append(game_card)
        
        print(f"✅ Loaded {len(self.games)} fallback games")
    
    def set_state(self, new_state: AppState, message: str = ""):
        """Thread-safe state change with logging"""
        with self.state_lock:
            old_state = self.current_state
            self.current_state = new_state
            self.state_change_time = time.time()
            self.state_duration = 0.0
            self.state_message = message
            
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
    
    def update_state_duration(self):
        """Update state duration for display"""
        with self.state_lock:
            self.state_duration = time.time() - self.state_change_time
    
    def can_interact(self) -> bool:
        """Check if user can interact based on current state"""
        state = self.get_state()
        return state in [AppState.IDLE, AppState.GAME_SELECTING, AppState.PAUSED]
    
    def get_hand_position(self, landmarks) -> Tuple[int, int]:
        """Get hand position in screen coordinates"""
        if len(landmarks.landmark) < 21:
            return (0, 0)
        
        x = int(landmarks.landmark[8].x * self.screen_width)
        y = int(landmarks.landmark[8].y * self.screen_height)
        return (x, y)
    
    def detect_hands_raised(self, hand_landmarks) -> bool:
        """Detect if hands are raised"""
        if not hand_landmarks:
            return False
        
        for landmarks in hand_landmarks:
            if len(landmarks.landmark) >= 21:
                palm_z = (landmarks.landmark[0].z + landmarks.landmark[5].z + 
                         landmarks.landmark[9].z + landmarks.landmark[13].z + 
                         landmarks.landmark[17].z) / 5
                
                if palm_z < -0.1:
                    return True
        
        return False
    
    def get_primary_hand_position(self, hand_landmarks_list) -> Tuple[int, int]:
        """Get position from the primary hand"""
        if not hand_landmarks_list:
            return (0, 0)
        
        if len(hand_landmarks_list) == 1:
            return self.get_hand_position(hand_landmarks_list[0])
        
        # Use the first hand for simplicity
        return self.get_hand_position(hand_landmarks_list[0])
    
    def detect_swipe_gesture(self) -> str:
        """Detect swipe gestures for navigation"""
        if len(self.hand_positions) < 2:
            return ""
        
        current_time = time.time()
        current_pos = self.hand_positions[-1]
        edge_threshold = 100
        
        # Check if hand is at left edge
        if current_pos[0] < edge_threshold:
            if self.swipe_hold_direction != "left":
                self.swipe_hold_start_time = current_time
                self.swipe_hold_direction = "left"
                self.swipe_hold_progress = 0.0
            
            elapsed_time = current_time - self.swipe_hold_start_time
            self.swipe_hold_progress = min(elapsed_time / self.selection_time, 1.0)
            
            if self.swipe_hold_progress >= 1.0:
                self.swipe_hold_start_time = 0
                self.swipe_hold_progress = 0.0
                self.swipe_hold_direction = ""
                return "left"
        
        # Check if hand is at right edge
        elif current_pos[0] > self.screen_width - edge_threshold:
            if self.swipe_hold_direction != "right":
                self.swipe_hold_start_time = current_time
                self.swipe_hold_direction = "right"
                self.swipe_hold_progress = 0.0
            
            elapsed_time = current_time - self.swipe_hold_start_time
            self.swipe_hold_progress = min(elapsed_time / self.selection_time, 1.0)
            
            if self.swipe_hold_progress >= 1.0:
                self.swipe_hold_start_time = 0
                self.swipe_hold_progress = 0.0
                self.swipe_hold_direction = ""
                return "right"
        else:
            # Reset if not at edges
            if self.swipe_hold_start_time > 0:
                self.swipe_hold_start_time = 0
                self.swipe_hold_progress = 0.0
                self.swipe_hold_direction = ""
        
        return ""
    
    def handle_swipe_navigation(self, swipe_direction: str):
        """Handle swipe navigation"""
        if swipe_direction == "left" and self.current_page > 0:
            self.current_page -= 1
            print("⬅️ Swiped left - Previous page")
        elif swipe_direction == "right" and (self.current_page + 1) * self.games_per_page < len(self.games):
            self.current_page += 1
            print("➡️ Swiped right - Next page")
        
        self.last_swipe_time = time.time()
    
    def get_card_position(self, index: int) -> Tuple[int, int, int, int]:
        """Get position and size of a game card"""
        cards_per_row = 5
        row = index // cards_per_row
        col = index % cards_per_row
        
        total_cards_width = cards_per_row * self.card_width + (cards_per_row - 1) * self.card_margin
        start_x = (self.screen_width - total_cards_width) // 2
        
        x = start_x + col * (self.card_width + self.card_margin)
        y = 200 + row * (self.card_height + self.card_margin)
        
        return (x, y, self.card_width, self.card_height)
    
    def is_point_in_rect(self, point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
        """Check if point is inside rectangle"""
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    
    def _draw_rounded_rectangle(self, frame, x: int, y: int, width: int, height: int, radius: int, color, thickness: int):
        """Draw a rounded rectangle"""
        # Main rectangle
        cv2.rectangle(frame, (x + radius, y), (x + width - radius, y + height), color, thickness)
        cv2.rectangle(frame, (x, y + radius), (x + width, y + height - radius), color, thickness)
        
        # Corner circles
        if thickness == -1:
            cv2.circle(frame, (x + radius, y + radius), radius, color, -1)
            cv2.circle(frame, (x + width - radius, y + radius), radius, color, -1)
            cv2.circle(frame, (x + radius, y + height - radius), radius, color, -1)
            cv2.circle(frame, (x + width - radius, y + height - radius), radius, color, -1)
        else:
            cv2.circle(frame, (x + radius, y + radius), radius, color, thickness)
            cv2.circle(frame, (x + width - radius, y + radius), radius, color, thickness)
            cv2.circle(frame, (x + radius, y + height - radius), radius, color, thickness)
            cv2.circle(frame, (x + width - radius, y + height - radius), radius, color, thickness)
    
    def _wrap_text(self, text: str, max_width: int, font, font_scale: float) -> List[str]:
        """Wrap text to fit within specified width"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            
            if text_size[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def draw_apple_style_card(self, surface, card: GameCard, x: int, y: int, width: int, height: int):
        # Zoom effect
        if getattr(card, 'is_hovered', False):
            zoom_offset = int((width * (self.zoom_factor - 1.0)) / 2)
            x -= zoom_offset
            y -= zoom_offset
            width = int(width * self.zoom_factor)
            height = int(height * self.zoom_factor)
        # Shadow
        shadow_offset = 8
        shadow_color = self.colors['card_shadow'][:3]
        pygame.draw.rect(surface, shadow_color, (x + shadow_offset, y + shadow_offset, width, height), border_radius=20)
        # Card background
        card_color = self.colors['card_hover'][:3] if getattr(card, 'is_hovered', False) else self.colors['card_bg'][:3]
        pygame.draw.rect(surface, card_color, (x, y, width, height), border_radius=20)
        # Card border
        border_color = self.colors['card_border'][:3]
        pygame.draw.rect(surface, border_color, (x, y, width, height), 2, border_radius=20)
        # Icon
        icon_size = min(width // 3, 80)
        icon_x = x + (width - icon_size) // 2
        icon_y = y + 20
        pygame.draw.rect(surface, self.colors['accent_primary'], (icon_x, icon_y, icon_size, icon_size))
        # Game initial
        initial = card.name[0].upper()
        font_scale = icon_size / 40
        icon_font = pygame.font.SysFont("Arial", int(36 * font_scale))
        initial_text = icon_font.render(initial, True, (255, 255, 255))
        text_x = icon_x + (icon_size - initial_text.get_width()) // 2
        text_y = icon_y + (icon_size - initial_text.get_height()) // 2
        surface.blit(initial_text, (text_x, text_y))
        # Game title
        title_y = icon_y + icon_size + 30
        title_lines = self._wrap_text(card.name, width - 40, self.font, 0.8)
        line_height = 25
        for i, line in enumerate(title_lines):
            line_surface = self.font.render(line, True, self.colors['text_primary'])
            text_x = x + (width - line_surface.get_width()) // 2
            surface.blit(line_surface, (text_x, title_y + i * line_height))
        # Description
        desc_y = title_y + len(title_lines) * line_height + 20
        desc_lines = self._wrap_text(card.description, width - 40, self.small_font, 0.8)
        for i, line in enumerate(desc_lines[:3]):
            desc_surface = self.small_font.render(line, True, self.colors['text_secondary'])
            text_x = x + (width - desc_surface.get_width()) // 2
            surface.blit(desc_surface, (text_x, desc_y + i * 18))
        # Selection progress bar
        if getattr(card, 'selection_progress', 0) > 0:
            bar_width = width - 40
            bar_height = 6
            bar_x = x + 20
            bar_y = y + height - 30
            pygame.draw.rect(surface, self.colors['surface_secondary'], (bar_x, bar_y, bar_width, bar_height))
            progress_width = int(bar_width * card.selection_progress)
            pygame.draw.rect(surface, self.colors['accent_success'], (bar_x, bar_y, progress_width, bar_height))
    
    def launch_game(self, game: GameCard):
        """Launch a game"""
        print(f"🎮 Launching game: {game.name}")
        self.set_state(AppState.GAME_LOADING, f"Loading {game.name}...")
        self.selected_game = game
        self.loading_start_time = time.time()
        
        def run_game():
            try:
                # Get the current directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                game_dir = os.path.join(current_dir, '..', game.module_path)
                
                # Check if game directory exists
                if not os.path.exists(game_dir):
                    print(f"❌ Game directory not found: {game_dir}")
                    self.set_state(AppState.ERROR, f"Game {game.name} not found")
                    time.sleep(3)
                    self.set_state(AppState.IDLE, "Ready for interaction")
                    self.selected_game = None
                    return
                
                # Check if __init__.py exists
                init_file = os.path.join(game_dir, '__init__.py')
                if not os.path.exists(init_file):
                    print(f"❌ Game file not found: {init_file}")
                    self.set_state(AppState.ERROR, f"Game {game.name} not found")
                    time.sleep(3)
                    self.set_state(AppState.IDLE, "Ready for interaction")
                    self.selected_game = None
                    return
                
                print(f"✅ Game {game.name} loaded successfully")
                self.set_state(AppState.GAME_RUNNING, f"{game.name} is running")
                
                # Launch the game as a subprocess
                import subprocess
                import sys
                
                # Change to the game directory
                original_dir = os.getcwd()
                os.chdir(game_dir)
                
                try:
                    # Launch the game
                    process = subprocess.Popen([sys.executable, '__init__.py'], 
                                            stdout=subprocess.PIPE, 
                                            stderr=subprocess.PIPE)
                    
                    # Wait for the game to finish
                    stdout, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        print(f"⚠️ Game {game.name} exited with code {process.returncode}")
                        if stderr:
                            print(f"Error: {stderr.decode()}")
                    
                finally:
                    # Restore original directory
                    os.chdir(original_dir)
                
                print(f"🔄 Game {game.name} finished")
                self.set_state(AppState.GAME_EXITING, f"Exiting {game.name}")
                
                # Brief exit period
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error launching game {game.name}: {e}")
                self.set_state(AppState.ERROR, f"Error launching {game.name}")
                time.sleep(3)
            
            finally:
                self.set_state(AppState.IDLE, "Ready for interaction")
                self.selected_game = None
        
        self.game_thread = threading.Thread(target=run_game, daemon=True)
        self.game_thread.start()
    
    def update_frame(self, surface):
        # Clear background
        surface.fill(self.colors['background'])
        # Draw title
        title = "CVGames App Store"
        title_text = self.font.render(title, True, self.colors['text_primary'])
        title_x = (self.screen_width - title_text.get_width()) // 2
        surface.blit(title_text, (title_x, 80))
        # Draw status
        state, message, duration = self.get_state_info()
        status_text = f"Status: {state.value}"
        if message:
            status_text += f" - {message}"
        status_surface = self.small_font.render(status_text, True, self.colors['text_secondary'])
        surface.blit(status_surface, (50, 130))
        # Draw games or loading screen
        if state in [AppState.IDLE, AppState.GAME_SELECTING]:
            self.draw_games(surface)
        if state == AppState.GAME_LOADING:
            self.draw_loading_screen(surface)
        self.draw_swipe_indicators(surface)
        self.draw_page_indicator(surface)
        self.draw_soft_cursor(surface)

    def draw_games(self, surface):
        start_index = self.current_page * self.games_per_page
        end_index = min(start_index + self.games_per_page, len(self.games))
        for i in range(start_index, end_index):
            card_index = i - start_index
            card = self.games[i]
            x, y, width, height = self.get_card_position(card_index)
            self.draw_apple_style_card(surface, card, x, y, width, height)

    def draw_loading_screen(self, surface):
        if not self.selected_game:
            return
        elapsed = time.time() - self.loading_start_time
        progress = min(elapsed / self.loading_duration, 1.0)
        loading_text = f"Loading {self.selected_game.name}..."
        loading_surface = self.font.render(loading_text, True, self.colors['text_primary'])
        text_x = (self.screen_width - loading_surface.get_width()) // 2
        text_y = self.screen_height // 2 - 50
        surface.blit(loading_surface, (text_x, text_y))
        # Progress bar
        bar_width = 400
        bar_height = 20
        bar_x = (self.screen_width - bar_width) // 2
        bar_y = text_y + 50
        pygame.draw.rect(surface, self.colors['surface_secondary'], (bar_x, bar_y, bar_width, bar_height))
        progress_width = int(bar_width * progress)
        pygame.draw.rect(surface, self.colors['accent_success'], (bar_x, bar_y, progress_width, bar_height))
    
    def draw_swipe_indicators(self, surface):
        if self.swipe_hold_progress > 0:
            edge_width = 20
            
            if self.swipe_hold_direction == "left":
                # Left edge indicator
                indicator_height = int(self.screen_height * self.swipe_hold_progress)
                y_start = (self.screen_height - indicator_height) // 2
                pygame.draw.rect(surface, self.colors['accent_primary'], (0, y_start, edge_width, indicator_height))
            
            elif self.swipe_hold_direction == "right":
                # Right edge indicator
                indicator_height = int(self.screen_height * self.swipe_hold_progress)
                y_start = (self.screen_height - indicator_height) // 2
                x_start = self.screen_width - edge_width
                pygame.draw.rect(surface, self.colors['accent_primary'], (x_start, y_start, edge_width, indicator_height))
    
    def draw_page_indicator(self, surface):
        total_pages = (len(self.games) + self.games_per_page - 1) // self.games_per_page
        if total_pages <= 1:
            return
        
        page_text = f"Page {self.current_page + 1} of {total_pages}"
        text_size = cv2.getTextSize(page_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        text_y = self.screen_height - 50
        
        page_surface = self.small_font.render(page_text, True, self.colors['text_secondary'])
        surface.blit(page_surface, (text_x, text_y))
    
    def draw_soft_cursor(self, surface):
        """Draw a soft, animated cursor at hand position"""
        if not self.cursor_visible:
            return
        
        x, y = self.cursor_position
        size = self.cursor_size
        
        # Determine cursor color based on interaction state
        if self.cursor_interactive:
            # Interactive cursor (when hovering over cards)
            base_color = self.colors['accent_success']
            size = int(size * 1.5)  # Larger when interactive
        else:
            # Normal cursor
            base_color = self.colors['accent_primary']
        
        # Create soft cursor with multiple layers
        for i in range(3):
            layer_size = size - i * 4
            if layer_size <= 0:
                break
            
            # Calculate alpha for each layer
            layer_alpha = self.cursor_alpha * (1.0 - i * 0.3)
            if layer_alpha <= 0:
                break
            
            # Create color with alpha
            color = tuple(int(c * layer_alpha) for c in base_color)
            
            # Draw soft circle
            pygame.draw.circle(surface, color, (x, y), layer_size)
        
        # Draw center dot
        center_color = tuple(int(c * self.cursor_alpha) for c in base_color)
        pygame.draw.circle(surface, center_color, (x, y), 3)
        
        # Draw subtle glow effect
        glow_size = size + 8
        glow_color = tuple(int(c * 0.2) for c in base_color)
        pygame.draw.circle(surface, glow_color, (x, y), glow_size, 2)
    
    def toggle_fullscreen(self):
        """Toggle between full screen and windowed mode"""
        try:
            print("🖥️ (Pygame) Fullscreen toggle requested")
        except Exception as e:
            print(f"⚠️ Error toggling full screen: {e}")
    
    def ensure_fullscreen(self):
        """Ensure the window is in full screen mode"""
        try:
            print("🖥️ (Pygame) Ensuring full screen mode")
        except Exception as e:
            print(f"⚠️ Error ensuring full screen: {e}")
    
    def process_hand_tracking(self, frame, results):
        """Process hand tracking results"""
        if not results.multi_hand_landmarks or not self.can_interact():
            # Reset all card hover states
            for card in self.games:
                card.is_hovered = False
                card.selection_progress = 0.0
            
            # Hide cursor when no hands detected
            self.cursor_visible = False
            self.cursor_interactive = False
            return
        
        # Get hand position
        hand_pos = self.get_primary_hand_position(results.multi_hand_landmarks)
        self.hand_positions.append(hand_pos)
        if len(self.hand_positions) > 10:
            self.hand_positions.pop(0)
        
        # Update cursor position with smoothing
        if self.cursor_visible:
            # Smooth cursor movement
            smooth_x = int(self.cursor_position[0] * (1 - self.cursor_smooth_factor) + 
                          hand_pos[0] * self.cursor_smooth_factor)
            smooth_y = int(self.cursor_position[1] * (1 - self.cursor_smooth_factor) + 
                          hand_pos[1] * self.cursor_smooth_factor)
            self.cursor_position = (smooth_x, smooth_y)
        else:
            # Initialize cursor position
            self.cursor_position = hand_pos
            self.cursor_visible = True
        
        # Check for swipe gestures
        swipe_direction = self.detect_swipe_gesture()
        if swipe_direction:
            self.handle_swipe_navigation(swipe_direction)
        
        # Check card interactions
        current_time = time.time()
        start_index = self.current_page * self.games_per_page
        end_index = min(start_index + self.games_per_page, len(self.games))
        
        # Reset cursor interaction state
        self.cursor_interactive = False
        
        for i in range(start_index, end_index):
            card_index = i - start_index
            card = self.games[i]
            x, y, width, height = self.get_card_position(card_index)
            card_rect = (x, y, width, height)
            
            if self.is_point_in_rect(hand_pos, card_rect):
                if not card.is_hovered:
                    card.is_hovered = True
                    card.hover_start_time = current_time
                    print(f"👋 Hovering over: {card.name}")
                
                # Set cursor to interactive state
                self.cursor_interactive = True
                
                # Update selection progress
                elapsed = current_time - card.hover_start_time
                card.selection_progress = min(elapsed / self.selection_time, 1.0)
                
                # Check if selection is complete
                if card.selection_progress >= 1.0:
                    print(f"✅ Selected: {card.name}")
                    self.launch_game(card)
                    break
            else:
                card.is_hovered = False
                card.selection_progress = 0.0
    
    def select_camera(self, start_index=0):
        """Select the first available non-OBS camera starting from start_index."""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None
        for camera_index in range(start_index, start_index + 3):  # Try 3 cameras from start_index
            print(f"📹 Trying camera index {camera_index % 3}...")
            cap = cv2.VideoCapture(camera_index % 3)
            if cap.isOpened():
                # Check camera name to skip OBS cameras
                camera_name = None
                try:
                    if hasattr(cap, 'getBackendName') and cap.getBackendName() == 'AVFoundation':
                        import subprocess
                        result = subprocess.run(['system_profiler', 'SPCameraDataType'], capture_output=True, text=True)
                        if result.returncode == 0:
                            lines = result.stdout.split('\n')
                            for i, line in enumerate(lines):
                                if f'Camera {camera_index % 3}' in line or f'Camera:' in line:
                                    for j in range(i, i+5):
                                        if 'Model ID:' in lines[j]:
                                            camera_name = lines[j].split(':', 1)[1].strip()
                                            break
                    elif hasattr(cap, 'getBackendName') and cap.getBackendName() == 'V4L2':
                        import subprocess
                        result = subprocess.run(['v4l2-ctl', '-d', f'/dev/video{camera_index % 3}', '--info'], capture_output=True, text=True)
                        if result.returncode == 0:
                            for line in result.stdout.split('\n'):
                                if 'Card type' in line:
                                    camera_name = line.split(':', 1)[1].strip()
                                    break
                except Exception as e:
                    pass
                if camera_name is None:
                    try:
                        camera_name = cap.get(cv2.CAP_PROP_DEVICE_DESCRIPTION)
                    except Exception:
                        camera_name = None
                if camera_name and 'OBS' in str(camera_name).upper():
                    print(f"❌ Skipping OBS camera at index {camera_index % 3} ({camera_name})")
                    cap.release()
                    continue
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Camera {camera_index % 3} working")
                    self.cap = cap
                    self.current_camera_index = camera_index % 3
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_height)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return
                else:
                    print(f"❌ Camera {camera_index % 3} opened but can't read frames")
                    cap.release()
            else:
                print(f"❌ Camera {camera_index % 3} failed to open")
        print("❌ No working camera found")
        raise RuntimeError("No working camera found")
    
    def run(self):
        """Main application loop"""
        print("🚀 Starting CVGames App Store...")
        print("🖥️ App will start in full screen mode")
        print("📋 Controls:")
        print("   - Move your hand to control the cursor")
        print("   - Hover over games to select them")
        print("   - Press 'F' to toggle full screen")
        print("   - Press 'A' for previous page, 'D' for next page")
        print("   - Press 'Q' or 'ESC' to quit")
        print("=" * 50)
        with self.tracer.start_as_current_span("appstore_session"):
            while not self.shutting_down:
                try:
                    # --- Computer Vision Input ---
                    ret, frame = self.cap.read()
                    if not ret:
                        print("❌ Failed to capture frame")
                        break
                    frame = cv2.flip(frame, 1)
                    frame = cv2.resize(frame, (self.screen_width, self.screen_height))
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.process_hand_tracking(frame, self.hands.process(rgb_frame))
                    # --- Pygame Event Handling ---
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.shutting_down = True
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                                self.shutting_down = True
                            elif event.key == pygame.K_f:
                                self.toggle_fullscreen()
                            elif event.key == pygame.K_a:
                                if self.current_page > 0:
                                    self.current_page -= 1
                            elif event.key == pygame.K_d:
                                if (self.current_page + 1) * self.games_per_page < len(self.games):
                                    self.current_page += 1
                            elif event.key == pygame.K_c:
                                print("🔄 Switching to next camera...")
                                self.select_camera(start_index=(self.current_camera_index + 1) % 3)
                    # --- Pygame Rendering ---
                    self.screen.fill((30, 30, 30))
                    self.update_frame(self.screen)  # update_frame should draw to the pygame surface
                    pygame.display.flip()
                    self.clock.tick(30)
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    break
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        self.shutting_down = True
        
        if self.cap:
            self.cap.release()
        
        print("✅ Cleanup complete")

def main():
    """Main function"""
    print("🎮 CVGames App Store - Hand Gesture Controlled Gaming Platform")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check camera
    if not check_camera():
        return
    
    try:
        # Create and run app store
        app = AppStore()
        app.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        print("👋 Thanks for using CVGames App Store!")

if __name__ == "__main__":
    main()
