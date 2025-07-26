import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import sys
import logging
import os
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Handle imports for both module and direct execution
try:
    # Try relative imports first (when run as module)
    from .config import ConfigManager
    from .util import ScreenManager
    from .game_logic import ColorPaintGame, GameState, ColoringImage
    from .converter import convert_image_to_coloring_page, convert_video_to_coloring_page
except ImportError:
    # Fallback to absolute imports (when run directly)
    try:
        from config import ConfigManager
        from util import ScreenManager
        from game_logic import ColorPaintGame, GameState, ColoringImage
        from converter import convert_image_to_coloring_page, convert_video_to_coloring_page
    except ImportError:
        # Create fallback classes if imports fail
        class ConfigManager:
            def __init__(self, db_path="airpainter_settings.sqlite"):
                self.db_path = db_path
            def get_resolution(self):
                return (1920, 1080)
            def get_default_brush_size(self):
                return 10
            def save_brush_size(self, brush_size):
                pass
        
        class ScreenManager:
            def __init__(self, config_manager=None):
                self.pygame_init = False
                self.screen = None
                self.display_width = 1920
                self.display_height = 1080
                self.scale_factor = 1.0
                self.font_scale = 1.0
                self.config_manager = config_manager
            
            def initialize(self, title="Air Painter", fullscreen=False):
                if not self.pygame_init:
                    pygame.init()
                    pygame.font.init()
                    self.pygame_init = True
                
                if self.config_manager:
                    self.display_width, self.display_height = self.config_manager.get_resolution()
                
                flags = pygame.DOUBLEBUF | pygame.HWSURFACE
                self.screen = pygame.display.set_mode((self.display_width, self.display_height), flags)
                pygame.display.set_caption(title)
                return self.screen
            
            def get_scaled_font(self, base_size, font_name="Arial"):
                return pygame.font.SysFont(font_name, base_size)
            
            def get_optimal_ui_sizes(self):
                return {
                    'toolbar_width': 200,
                    'palette_height': 120,
                    'color_swatch_size': 50,
                    'tool_button_height': 60,
                    'padding': 20,
                    'margin': 10,
                    'border_radius': 10,
                    'shadow_offset': 3,
                    'center_panel_width': 400,
                    'center_panel_height': 500
                }
            
            def get_optimal_brush_sizes(self):
                return [5, 10, 15, 20, 25, 30]
            
            def get_optimal_font_sizes(self):
                return {
                    'title_font': 28,
                    'tool_font': 16,
                    'info_font': 14,
                    'small_font': 12
                }
            
            def cleanup(self):
                if self.pygame_init:
                    pygame.quit()
                    self.pygame_init = False
        
        # Fallback game logic classes
        class GameState(Enum):
            SELECTING_IMAGE = "selecting_image"
            PAINTING = "painting"
            COMPLETED = "completed"
        
        class ColoringImage:
            def __init__(self, name, filepath, thumbnail_path, difficulty, category, description):
                self.name = name
                self.filepath = filepath
                self.thumbnail_path = thumbnail_path
                self.difficulty = difficulty
                self.category = category
                self.description = description
        
        class ColorPaintGame:
            def __init__(self, data_dir="data"):
                self.current_state = GameState.SELECTING_IMAGE
                self.selected_image = None
                self.available_images = []
                self.current_painting_surface = None
                self.original_image = None
                self.painted_areas = set()
                self.completion_percentage = 0.0
                self.images_dir = "data/images"
                self.thumbnails_dir = "data/thumbnails"
            
            def select_image(self, image):
                pass
            
            def paint_area(self, x, y, color, brush_size=5):
                return False
            
            def get_completion_percentage(self):
                return 0.0
            
            def get_current_image(self):
                return None
            
            def get_original_image(self):
                return None
            
            def get_available_images(self):
                return []
            
            def get_current_state(self):
                return GameState.SELECTING_IMAGE
            
            def reset_game(self):
                pass
            
            def create_thumbnail(self, filepath, thumbnail_path):
                pass
            
            def is_painting_active(self):
                return False
            
            def start_painting(self, x, y, color, brush_size=5):
                return False
            
            def continue_painting(self, x, y, color, brush_size=5):
                return False
            
            def stop_painting(self):
                pass
            
            def undo_last_paint(self):
                return False

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "component": "airpainter", "message": "%(message)s"}',
    handlers=[logging.FileHandler("application.log")]
)
logger = logging.getLogger(__name__)

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

class Tool(Enum):
    BRUSH = "🖌️"
    ERASER = "🧽"
    CLEAR = "🗑️"
    UNDO = "↶"
    SAVE = "💾"
    IMAGE_URL = "🌐"
    LOCAL_IMAGE = "📁"
    YOUTUBE_FRAME = "📺"

@dataclass
class Color:
    name: str
    rgb: Tuple[int, int, int]
    hex_code: str

class AirPainterUI:
    def __init__(self):
        # Initialize configuration and screen management
        self.config_manager = ConfigManager()
        self.screen_manager = ScreenManager(self.config_manager)
        
        # Initialize display - always maximized
        self.screen = self.screen_manager.initialize("Air Painter - Color by Number Game", fullscreen=True)
        self.width = self.screen_manager.display_width
        self.height = self.screen_manager.display_height
        
        # Get optimal sizes
        self.ui_sizes = self.screen_manager.get_optimal_ui_sizes()
        self.font_sizes = self.screen_manager.get_optimal_font_sizes()
        self.brush_sizes = self.screen_manager.get_optimal_brush_sizes()
        
        # Initialize game logic
        self.game = ColorPaintGame()
        
        # Setup components
        self.setup_canvas()
        self.setup_colors()
        self.setup_fonts()
        self.setup_state()
        self.setup_camera()
        
        logger.info("Air Painter UI initialized", extra={
            "resolution": f"{self.width}x{self.height}",
            "ui_sizes": self.ui_sizes,
            "brush_sizes": self.brush_sizes,
            "game_state": self.game.get_current_state().value
        })
        
        # Show loading screen
        self.show_loading_screen()
    
    def show_loading_screen(self):
        """Show a cute loading screen for 30 seconds"""
        import time
        import math
        
        logger.info("Starting loading screen")
        
        # Loading screen colors
        loading_colors = {
            'background': (240, 248, 255),  # Alice Blue
            'primary': (100, 149, 237),     # Cornflower Blue
            'secondary': (255, 182, 193),   # Light Pink
            'accent': (255, 218, 185),      # Peach
            'text': (70, 130, 180)          # Steel Blue
        }
        
        # Loading messages
        loading_messages = [
            "🎨 Preparing your magical paintbrush...",
            "🌈 Loading beautiful colors...",
            "🎯 Setting up hand tracking...",
            "✨ Creating your canvas...",
            "🎪 Getting ready for fun...",
            "🌟 Almost there...",
            "🎭 Preparing the stage...",
            "🎪 Let's start painting!"
        ]
        
        start_time = time.time()
        duration = 30  # 30 seconds
        message_index = 0
        last_message_time = start_time
        
        # Animation variables
        rotation_angle = 0
        bounce_offset = 0
        color_shift = 0
        
        while time.time() - start_time < duration:
            current_time = time.time()
            elapsed = current_time - start_time
            progress = elapsed / duration
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
            
            # Clear screen
            self.screen.fill(loading_colors['background'])
            
            # Update animation variables
            rotation_angle += 3
            bounce_offset = math.sin(current_time * 4) * 10
            color_shift = (color_shift + 1) % 360
            
            # Draw main loading circle
            center_x = self.width // 2
            center_y = self.height // 2 - 50
            
            # Animated rainbow circle
            for i in range(12):
                angle = rotation_angle + i * 30
                radius = 80 + bounce_offset
                x = center_x + math.cos(math.radians(angle)) * radius
                y = center_y + math.sin(math.radians(angle)) * radius
                
                # Rainbow colors
                hue = (color_shift + i * 30) % 360
                if hue < 60:
                    color = (255, int(255 * hue / 60), 0)
                elif hue < 120:
                    color = (int(255 * (120 - hue) / 60), 255, 0)
                elif hue < 180:
                    color = (0, 255, int(255 * (hue - 120) / 60))
                elif hue < 240:
                    color = (0, int(255 * (240 - hue) / 60), 255)
                elif hue < 300:
                    color = (int(255 * (hue - 240) / 60), 0, 255)
                else:
                    color = (255, 0, int(255 * (360 - hue) / 60))
                
                pygame.draw.circle(self.screen, color, (int(x), int(y)), 8)
            
            # Draw bouncing paintbrush
            brush_x = center_x
            brush_y = center_y + 120 + bounce_offset
            
            # Paintbrush handle
            pygame.draw.rect(self.screen, (139, 69, 19), (brush_x - 4, brush_y - 20, 8, 40))
            # Paintbrush tip
            pygame.draw.polygon(self.screen, (255, 255, 255), [
                (brush_x - 8, brush_y - 20),
                (brush_x + 8, brush_y - 20),
                (brush_x + 4, brush_y - 35),
                (brush_x - 4, brush_y - 35)
            ])
            # Paintbrush bristles
            for i in range(5):
                bristle_x = brush_x - 6 + i * 3
                pygame.draw.line(self.screen, (200, 200, 200), 
                               (bristle_x, brush_y - 20), 
                               (bristle_x - 1, brush_y - 35), 2)
            
            # Draw progress bar
            bar_width = 400
            bar_height = 20
            bar_x = (self.width - bar_width) // 2
            bar_y = center_y + 200
            
            # Background bar
            pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height), border_radius=10)
            
            # Progress bar
            progress_width = int(bar_width * progress)
            progress_color = (
                int(100 + 155 * progress),
                int(149 + 106 * progress),
                int(237 + 18 * progress)
            )
            pygame.draw.rect(self.screen, progress_color, (bar_x, bar_y, progress_width, bar_height), border_radius=10)
            
            # Progress text
            progress_text = f"{int(progress * 100)}%"
            progress_font = self.screen_manager.get_scaled_font(24, "Arial")
            progress_surface = progress_font.render(progress_text, True, loading_colors['text'])
            progress_rect = progress_surface.get_rect(center=(self.width // 2, bar_y + bar_height + 20))
            self.screen.blit(progress_surface, progress_rect)
            
            # Update loading message every 3.75 seconds
            if current_time - last_message_time > 3.75:
                message_index = (message_index + 1) % len(loading_messages)
                last_message_time = current_time
            
            # Draw loading message
            message = loading_messages[message_index]
            message_font = self.screen_manager.get_scaled_font(20, "Arial")
            message_surface = message_font.render(message, True, loading_colors['text'])
            message_rect = message_surface.get_rect(center=(self.width // 2, bar_y + 60))
            self.screen.blit(message_surface, message_rect)
            
            # Draw floating paint drops
            for i in range(5):
                drop_x = 100 + i * 150 + math.sin(current_time * 2 + i) * 20
                drop_y = 100 + math.cos(current_time * 1.5 + i) * 30
                drop_color = (
                    int(255 * (0.5 + 0.5 * math.sin(current_time + i))),
                    int(200 * (0.5 + 0.5 * math.cos(current_time + i))),
                    int(255 * (0.5 + 0.5 * math.sin(current_time * 0.7 + i)))
                )
                pygame.draw.circle(self.screen, drop_color, (int(drop_x), int(drop_y)), 15)
                pygame.draw.circle(self.screen, (255, 255, 255), (int(drop_x - 3), int(drop_y - 3)), 3)
            
            # Draw title
            title_font = self.screen_manager.get_scaled_font(36, "Arial")
            title_surface = title_font.render("🎨 Air Painter", True, loading_colors['text'])
            title_rect = title_surface.get_rect(center=(self.width // 2, 80))
            self.screen.blit(title_surface, title_rect)
            
            # Draw subtitle
            subtitle_font = self.screen_manager.get_scaled_font(18, "Arial")
            subtitle_surface = subtitle_font.render("Color by Number Game", True, loading_colors['text'])
            subtitle_rect = subtitle_surface.get_rect(center=(self.width // 2, 120))
            self.screen.blit(subtitle_surface, subtitle_rect)
            
            # Update display
            pygame.display.flip()
            
            # Cap at 60 FPS
            pygame.time.Clock().tick(60)
        
        logger.info("Loading screen completed")
        
    def setup_canvas(self):
        """Initialize drawing canvas"""
        self.canvas = pygame.Surface((self.width, self.height))
        self.canvas.fill((255, 255, 255))
        
    def setup_colors(self):
        """Define modern color palette with names"""
        self.colors = [
            Color("Black", (0, 0, 0), "#000000"),
            Color("Gray", (128, 128, 128), "#808080"),
            Color("Dark Red", (136, 0, 21), "#880015"),
            Color("Red", (237, 28, 36), "#ED1C24"),
            Color("Orange", (255, 127, 39), "#FF7F27"),
            Color("Yellow", (255, 242, 0), "#FFF200"),
            Color("Green", (34, 177, 76), "#22B14C"),
            Color("Blue", (0, 162, 232), "#00A2E8"),
            Color("Purple", (63, 72, 204), "#3F48CC"),
            Color("Pink", (163, 73, 164), "#A349A4"),
            Color("White", (255, 255, 255), "#FFFFFF")
        ]
        
        # UI Colors
        self.ui_colors = {
            'background': (248, 249, 250),
            'toolbar_bg': (255, 255, 255),
            'toolbar_border': (222, 226, 230),
            'palette_bg': (255, 255, 255),
            'palette_border': (222, 226, 230),
            'selected_bg': (13, 110, 253),
            'selected_text': (255, 255, 255),
            'text_primary': (33, 37, 41),
            'text_secondary': (108, 117, 125),
            'success': (25, 135, 84),
            'warning': (255, 193, 7),
            'error': (220, 53, 69),
            'cursor': (255, 0, 0),
            'cursor_outline': (255, 255, 255),
            'gesture_indicator': (0, 255, 0),
            'hand_trail': (100, 100, 255, 50)
        }
        
    def setup_fonts(self):
        """Initialize fonts with proper scaling"""
        self.title_font = self.screen_manager.get_scaled_font(self.font_sizes['title_font'], "Arial")
        self.tool_font = self.screen_manager.get_scaled_font(self.font_sizes['tool_font'], "Arial")
        self.info_font = self.screen_manager.get_scaled_font(self.font_sizes['info_font'], "Arial")
        self.small_font = self.screen_manager.get_scaled_font(self.font_sizes['small_font'], "Arial")
        
        self.clock = pygame.time.Clock()
        
    def setup_state(self):
        """Initialize application state"""
        self.selected_color = self.colors[3]  # Red
        self.selected_tool = Tool.BRUSH
        self.drawing_history = []
        self.current_brush_size = 1  # Index into brush_sizes
        self.last_pos = None
        self.drawing = False
        self.show_help = False
        self.camera_active = True
        
        # Real-time indicators
        self.cursor_pos = None
        self.hand_trail = []
        self.hand_landmarks = None  # For ghost hand effect
        self.gesture_active = False
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.1  # Seconds
        self.hand_confidence = 0.0
        self.pinch_strength = 0.0
        
        # New painting gesture detection
        self.painting_gesture_start_time = None
        self.painting_gesture_threshold = 2.0  # 2 seconds to activate painting
        self.painting_gesture_active = False
        self.last_hand_position = None
        self.hand_forward_threshold = 50  # Pixels forward movement threshold
        
        # Selection timing system
        self.selection_hold_time = 3.0  # Seconds to hold for selection
        self.hover_start_time = {}  # Track hover start times for each UI element
        self.current_hover_element = None
        self.hover_progress = 0.0  # 0.0 to 1.0 progress for selection
        
        # URL processing state
        self.showing_url_input = False
        self.url_input_text = ""
        self.url_input_cursor_pos = 0
        self.url_processing = False
        self.url_processing_progress = 0.0
        self.url_processing_message = ""
        
        # Two-hand circular menu system
        self.show_circular_menus = False
        self.left_hand_pos = None
        self.right_hand_pos = None
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.circular_menu_radius = 150
        self.menu_button_radius = 30
        self.menu_spacing = 20
        
        # Load default brush size from config
        default_brush_size = self.config_manager.get_default_brush_size()
        for i, size in enumerate(self.brush_sizes):
            if size >= default_brush_size:
                self.current_brush_size = i
                break
        
    def setup_camera(self):
        """Initialize camera and MediaPipe"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Could not open webcam")
                self.camera_active = False
                return
                
            # Configure camera for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # MediaPipe setup
            try:
                mp_hands = mp.solutions.hands
                self.hands = mp_hands.Hands(
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                    max_num_hands=2  # Support for two hands
                )
                logger.info("Camera and MediaPipe initialized successfully with two-hand support")
            except Exception as mp_error:
                logger.warning(f"MediaPipe hands import issue: {mp_error}")
                self.hands = None
                self.camera_active = False
            
        except Exception as e:
            logger.error(f"Camera setup failed: {e}")
            self.camera_active = False
            
    def draw_rounded_rect(self, surface, color, rect, radius):
        """Draw rounded rectangle with shadow effect"""
        x, y, w, h = rect
        
        # Shadow
        shadow_rect = (x + self.ui_sizes['shadow_offset'], y + self.ui_sizes['shadow_offset'], w, h)
        pygame.draw.rect(surface, (0, 0, 0, 30), shadow_rect, border_radius=radius)
        
        # Main rectangle
        pygame.draw.rect(surface, color, rect, border_radius=radius)
        
    def draw_cursor_indicator(self):
        """Draw real-time cursor indicator"""
        if not self.cursor_pos:
            return
            
        x, y = self.cursor_pos
        
        # Don't draw cursor in UI areas
        center_x = (self.width - self.ui_sizes['center_panel_width']) // 2
        center_y = (self.height - self.ui_sizes['center_panel_height']) // 2
        if (center_x <= x <= center_x + self.ui_sizes['center_panel_width'] and 
            center_y <= y <= center_y + self.ui_sizes['center_panel_height']) or y > self.height - self.ui_sizes['palette_height']:
            return
            
        # Cursor size based on tool
        if self.selected_tool == Tool.BRUSH:
            cursor_size = self.brush_sizes[self.current_brush_size]
            cursor_color = self.selected_color.rgb
        elif self.selected_tool == Tool.ERASER:
            cursor_size = 25
            cursor_color = (200, 200, 200)
        else:
            cursor_size = 15
            cursor_color = self.ui_colors['cursor']
            
        # Draw cursor outline
        pygame.draw.circle(self.screen, self.ui_colors['cursor_outline'], (x, y), cursor_size + 2)
        
        # Draw cursor fill
        pygame.draw.circle(self.screen, cursor_color, (x, y), cursor_size)
        
        # Draw crosshair for precision
        pygame.draw.line(self.screen, self.ui_colors['cursor_outline'], (x - 10, y), (x + 10, y), 2)
        pygame.draw.line(self.screen, self.ui_colors['cursor_outline'], (x, y - 10), (x, y + 10), 2)
        
    def draw_hand_trail(self):
        """Draw hand movement trail for better tracking feedback"""
        if len(self.hand_trail) < 2:
            return
            
        # Draw trail with fading effect
        for i in range(len(self.hand_trail) - 1):
            alpha = int(255 * (i / len(self.hand_trail)))
            color = (*self.ui_colors['hand_trail'][:3], alpha)
            
            # Create surface for alpha blending
            trail_surface = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(trail_surface, color, (2, 2), 2)
            
            pos = self.hand_trail[i]
            self.screen.blit(trail_surface, (pos[0] - 2, pos[1] - 2))
    
    def draw_ghost_hand(self):
        """Draw ghost hand effect for better UX"""
        if not self.cursor_pos or not hasattr(self, 'hand_landmarks') or not self.hand_landmarks:
            return
            
        x, y = self.cursor_pos
        
        # Don't draw ghost hand in UI areas
        status_height = 25
        right_panel_x = self.width - 420
        center_x = (self.width - self.ui_sizes['center_panel_width']) // 2
        center_y = (self.height - self.ui_sizes['center_panel_height']) // 2
        
        if (y < status_height or 
            (center_x <= x <= center_x + self.ui_sizes['center_panel_width'] and 
             center_y <= y <= center_y + self.ui_sizes['center_panel_height']) or 
            x > right_panel_x):
            return
        
        # Draw semi-transparent hand outline
        if len(self.hand_landmarks) >= 21:
            # Create ghost hand surface
            ghost_surface = pygame.Surface((100, 100), pygame.SRCALPHA)
            
            # Draw hand outline points
            hand_points = []
            for landmark in self.hand_landmarks:
                # Scale landmarks to ghost hand size
                px = int((landmark[0] - x + 50) * 0.3)
                py = int((landmark[1] - y + 50) * 0.3)
                hand_points.append((px, py))
            
            # Draw hand outline
            if len(hand_points) >= 5:
                # Palm
                pygame.draw.polygon(ghost_surface, (255, 255, 255, 80), hand_points[:5])
                
                # Fingers
                finger_indices = [
                    [5, 6, 7, 8],    # Thumb
                    [9, 10, 11, 12], # Index
                    [13, 14, 15, 16], # Middle
                    [17, 18, 19, 20], # Ring
                    [0, 1, 2, 3, 4]   # Pinky
                ]
                
                for finger in finger_indices:
                    if all(i < len(hand_points) for i in finger):
                        finger_points = [hand_points[i] for i in finger]
                        pygame.draw.lines(ghost_surface, (255, 255, 255, 100), False, finger_points, 2)
            
            # Draw ghost hand at cursor position
            self.screen.blit(ghost_surface, (x - 50, y - 50))
            
            # Draw pulsing effect
            import time
            pulse_size = int(10 + 5 * np.sin(time.time() * 3))
            pygame.draw.circle(self.screen, (255, 255, 255, 60), (x, y), pulse_size, 2)
    
    def draw_camera_feedback(self):
        """Draw small camera view in bottom-left corner for user feedback"""
        if not self.camera_active or not self.cap:
            return
            
        # Get camera frame
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Resize frame for small preview
        preview_width = 160
        preview_height = 120
        frame_resized = cv2.resize(frame, (preview_width, preview_height))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to Pygame surface
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        
        # Position in bottom-left corner
        margin = 20
        x = margin
        y = self.height - preview_height - margin
        
        # Draw background border
        border_rect = (x - 5, y - 5, preview_width + 10, preview_height + 10)
        pygame.draw.rect(self.screen, self.ui_colors['toolbar_border'], border_rect, 3, 5)
        
        # Draw camera preview
        self.screen.blit(frame_surface, (x, y))
        
        # Draw "CAMERA" label
        label_text = self.info_font.render("CAMERA", True, self.ui_colors['text_primary'])
        self.screen.blit(label_text, (x, y - 20))
            
    def draw_gesture_indicator(self):
        """Draw gesture feedback indicator for new painting gesture"""
        if not self.cursor_pos:
            return
            
        x, y = self.cursor_pos
        current_time = time.time()
        
        # Don't draw gesture indicator in UI areas
        center_x = (self.width - self.ui_sizes['center_panel_width']) // 2
        center_y = (self.height - self.ui_sizes['center_panel_height']) // 2
        
        # Check if cursor is in UI areas (status bar top, center panel, or right panel)
        status_height = 25
        right_panel_x = self.width - 420  # 400 width + 20 margin
        
        if (y < status_height or  # Status bar area
            (center_x <= x <= center_x + self.ui_sizes['center_panel_width'] and 
             center_y <= y <= center_y + self.ui_sizes['center_panel_height']) or  # Center panel
            x > right_panel_x):  # Right panel
            return
        
        # Show painting gesture progress
        if self.painting_gesture_start_time is not None:
            gesture_duration = time.time() - self.painting_gesture_start_time
            progress = min(gesture_duration / self.painting_gesture_threshold, 1.0)
            
            # Draw progress circle
            progress_radius = int(30 + 20 * progress)
            progress_color = self.ui_colors['success'] if progress >= 1.0 else self.ui_colors['gesture_indicator']
            
            # Animated pulse when gesture is active
            if self.painting_gesture_active:
                pulse_size = int(progress_radius + 10 * np.sin(current_time * 8))
                pygame.draw.circle(self.screen, progress_color, (x, y), pulse_size, 3)
            else:
                pygame.draw.circle(self.screen, progress_color, (x, y), progress_radius, 3)
            
            # Draw progress text
            if progress < 1.0:
                progress_text = f"{int(progress * 100)}%"
                text_surface = self.info_font.render(progress_text, True, progress_color)
                text_rect = text_surface.get_rect(center=(x, y + 50))
                self.screen.blit(text_surface, text_rect)
        else:
            # Show default cursor indicator
            pygame.draw.circle(self.screen, self.ui_colors['gesture_indicator'], (x, y), 15, 2)
            
    def draw_confidence_indicator(self):
        """Draw hand detection confidence indicator"""
        if not self.camera_active:
            return
            
        # Position in top-right corner
        indicator_x = self.width - 150
        indicator_y = 20
        
        # Background
        bg_rect = (indicator_x, indicator_y, 130, 60)
        self.draw_rounded_rect(self.screen, self.ui_colors['toolbar_bg'], bg_rect, self.ui_sizes['border_radius'])
        
        # Title
        title_text = self.info_font.render("HAND TRACKING", True, self.ui_colors['text_primary'])
        self.screen.blit(title_text, (indicator_x + 5, indicator_y + 5))
        
        # Confidence bar
        bar_rect = (indicator_x + 5, indicator_y + 25, 120, 8)
        pygame.draw.rect(self.screen, self.ui_colors['text_secondary'], bar_rect, border_radius=4)
        
        # Fill based on confidence
        fill_width = int(120 * self.hand_confidence)
        if fill_width > 0:
            fill_color = self.ui_colors['success'] if self.hand_confidence > 0.7 else self.ui_colors['warning']
            fill_rect = (indicator_x + 5, indicator_y + 25, fill_width, 8)
            pygame.draw.rect(self.screen, fill_color, fill_rect, border_radius=4)
            
        # Confidence text
        conf_text = self.info_font.render(f"{int(self.hand_confidence * 100)}%", True, self.ui_colors['text_secondary'])
        self.screen.blit(conf_text, (indicator_x + 5, indicator_y + 35))
        
    def draw_toolbar(self):
        """Draw the main toolbar - only show when circular menus are not active"""
        if not self.show_circular_menus:
            if self.game.get_current_state() == GameState.SELECTING_IMAGE:
                # Check if there are any available images
                available_images = self.game.get_available_images()
                if not available_images:
                    # Show initial selection screen
                    self.draw_initial_selection_screen()
                else:
                    # Draw image selection grid in center with adaptive sizing
                    panel_width = min(self.ui_sizes['center_panel_width'], self.width - 100)
                    panel_height = min(self.ui_sizes['center_panel_height'], self.height - 100)
                    center_x = (self.width - panel_width) // 2
                    center_y = (self.height - panel_height) // 2
                    
                    # Ensure panel doesn't go outside screen bounds
                    center_x = max(50, min(center_x, self.width - panel_width - 50))
                    center_y = max(50, min(center_y, self.height - panel_height - 50))
                    
                    self.draw_image_selection_buttons(center_x, center_y, panel_width, panel_height)
            else:
                # Draw tools and colors on the right side
                self.draw_right_panel()
    
    def draw_initial_selection_screen(self):
        """Draw initial selection screen with 3 options"""
        # Calculate adaptive panel dimensions
        panel_width = min(self.ui_sizes['center_panel_width'], self.width - 100)
        panel_height = min(self.ui_sizes['center_panel_height'], self.height - 100)
        
        # Calculate center position
        center_x = (self.width - panel_width) // 2
        center_y = (self.height - panel_height) // 2
        
        # Ensure panel doesn't go outside screen bounds
        center_x = max(50, min(center_x, self.width - panel_width - 50))
        center_y = max(50, min(center_y, self.height - panel_height - 50))
        
        # Panel background
        panel_rect = (center_x, center_y, panel_width, panel_height)
        self.draw_rounded_rect(self.screen, self.ui_colors['toolbar_bg'], panel_rect, self.ui_sizes['border_radius'])
        
        # Title
        title_text = self.title_font.render("CHOOSE YOUR IMAGE", True, self.ui_colors['text_primary'])
        title_rect = title_text.get_rect(center=(center_x + self.ui_sizes['center_panel_width'] // 2, center_y + 40))
        self.screen.blit(title_text, title_rect)
        
        # Three options
        options = [
            (Tool.LOCAL_IMAGE, "📁 Local Drive", "Choose an image from your computer"),
            (Tool.IMAGE_URL, "🌐 Image URL", "Download an image from the internet"),
            (Tool.YOUTUBE_FRAME, "📺 YouTube", "Extract frames from YouTube video (Coming Soon)")
        ]
        
        # Adaptive button sizes based on panel size
        button_width = min(300, panel_width - 40)
        button_height = min(80, int(panel_height * 0.15))
        spacing = min(20, int(panel_height * 0.05))
        start_y = center_y + int(panel_height * 0.2)
        
        for i, (tool, title, description) in enumerate(options):
            y = start_y + i * (button_height + spacing)
            x = center_x + (self.ui_sizes['center_panel_width'] - button_width) // 2
            
            button_rect = (x, y, button_width, button_height)
            
            # Check if cursor is hovering over this button
            element_id = f"initial_option_{tool.value}"
            is_hovering = self.is_cursor_hovering(button_rect)
            
            if is_hovering:
                self.update_hover_progress(element_id)
                self.draw_selection_indicator(button_rect, self.hover_progress)
            else:
                self.reset_hover_progress(element_id)
            
            # Button background (disabled for YouTube for now)
            if tool == Tool.YOUTUBE_FRAME:
                bg_color = self.ui_colors['text_secondary']  # Grayed out
                text_color = self.ui_colors['text_secondary']
            else:
                bg_color = self.ui_colors['toolbar_border']
                text_color = self.ui_colors['text_primary']
            
            self.draw_rounded_rect(self.screen, bg_color, button_rect, self.ui_sizes['border_radius'])
            
            # Button title
            title_text = self.tool_font.render(title, True, text_color)
            title_rect = title_text.get_rect(center=(x + button_width // 2, y + 25))
            self.screen.blit(title_text, title_rect)
            
            # Button description
            desc_text = self.info_font.render(description, True, text_color)
            desc_rect = desc_text.get_rect(center=(x + button_width // 2, y + 55))
            self.screen.blit(desc_text, desc_rect)
    
    def draw_image_selection_buttons(self, center_x, center_y, panel_width=None, panel_height=None):
        """Draw image selection buttons in centered toolbar"""
        available_images = self.game.get_available_images()
        
        # Use provided panel dimensions or calculate adaptive ones
        if panel_width is None:
            panel_width = min(self.ui_sizes['center_panel_width'], self.width - 100)
        if panel_height is None:
            panel_height = min(self.ui_sizes['center_panel_height'], self.height - 100)
        
        # Grid layout for images
        grid_width = 2
        grid_height = (len(available_images) + grid_width - 1) // grid_width
        
        # Adaptive image sizes based on panel size
        image_size = min(120, int(panel_width * 0.3))
        spacing = min(20, int(panel_width * 0.05))
        start_x = center_x + (panel_width - grid_width * (image_size + spacing)) // 2
        start_y = center_y + int(panel_height * 0.2)
        
        for i, image in enumerate(available_images):
            row = i // grid_width
            col = i % grid_width
            
            x = start_x + col * (image_size + spacing)
            y = start_y + row * (image_size + spacing + 30)
            
            # Image button background
            button_rect = (x, y, image_size, image_size)
            
            # Check if cursor is hovering over this button
            element_id = f"image_{i}"
            is_hovering = self.is_cursor_hovering(button_rect)
            
            if is_hovering:
                # Update hover progress
                self.update_hover_progress(element_id)
                # Draw selection indicator
                self.draw_selection_indicator(button_rect, self.hover_progress)
            else:
                # Reset hover progress if not hovering
                self.reset_hover_progress(element_id)
            
            # Button background
            bg_color = self.ui_colors['toolbar_bg']
            self.draw_rounded_rect(self.screen, bg_color, button_rect, self.ui_sizes['border_radius'])
            
            # Try to load and display thumbnail
            try:
                if os.path.exists(image.thumbnail_path):
                    thumbnail = pygame.image.load(image.thumbnail_path)
                    thumbnail = pygame.transform.scale(thumbnail, (image_size - 20, image_size - 20))
                    self.screen.blit(thumbnail, (x + 10, y + 10))
            except Exception as e:
                logger.warning(f"Could not load thumbnail: {e}")
            
            # Image name
            name_text = self.tool_font.render(image.name.replace('_', ' ').title(), True, self.ui_colors['text_primary'])
            name_rect = name_text.get_rect(center=(x + image_size // 2, y + image_size + 15))
            self.screen.blit(name_text, name_rect)
            
            # Difficulty
            diff_text = self.info_font.render(image.difficulty.upper(), True, self.ui_colors['text_secondary'])
            diff_rect = diff_text.get_rect(center=(x + image_size // 2, y + image_size + 35))
            self.screen.blit(diff_text, diff_rect)
    
    def draw_right_panel(self):
        """Draw tools and colors panel on the right side - Horizontal layout"""
        panel_width = 400  # Wider panel for horizontal layout
        panel_x = self.width - panel_width - 20
        panel_y = 20
        panel_height = 200  # Fixed height for horizontal layout
        
        # Panel background
        panel_rect = (panel_x, panel_y, panel_width, panel_height)
        self.draw_rounded_rect(self.screen, self.ui_colors['toolbar_bg'], panel_rect, self.ui_sizes['border_radius'])
        
        # Draw tools section (left side)
        self.draw_tools_section(panel_x, panel_y)
        
        # Draw colors section (right side)
        self.draw_colors_section(panel_x + panel_width // 2, panel_y)
    
    def draw_tools_section(self, panel_x, panel_y):
        """Draw tools section in the right panel - Horizontal layout with better spacing"""
        # Title
        title_text = self.title_font.render("TOOLS", True, self.ui_colors['text_primary'])
        title_rect = title_text.get_rect(center=(panel_x + 100, panel_y + 30))
        self.screen.blit(title_text, title_rect)
        
        # Tools (only essential ones for kids)
        tools = [Tool.BRUSH, Tool.ERASER, Tool.CLEAR, Tool.UNDO, Tool.SAVE]
        
        button_size = 60  # Larger buttons
        spacing = 25  # More spacing between buttons
        start_x = panel_x + 30  # More left margin
        start_y = panel_y + 70  # More top margin
        
        # Horizontal layout - 2 rows of tools
        for i, tool in enumerate(tools):
            row = i // 3  # 3 tools per row
            col = i % 3
            
            x = start_x + col * (button_size + spacing)
            y = start_y + row * (button_size + spacing)
            
            # Button background with better contrast
            button_rect = (x, y, button_size, button_size)
            bg_color = self.ui_colors['selected_bg'] if self.selected_tool == tool else self.ui_colors['toolbar_bg']
            self.draw_rounded_rect(self.screen, bg_color, button_rect, self.ui_sizes['border_radius'])
            
            # Border for better visibility
            border_color = self.ui_colors['selected_bg'] if self.selected_tool == tool else self.ui_colors['toolbar_border']
            pygame.draw.rect(self.screen, border_color, button_rect, 2, self.ui_sizes['border_radius'])
            
            # Check if cursor is hovering over this button
            element_id = f"tool_{tool.name.lower()}"
            is_hovering = self.is_cursor_hovering(button_rect)
            
            if is_hovering:
                self.update_hover_progress(element_id)
                self.draw_selection_indicator(button_rect, self.hover_progress)
            else:
                self.reset_hover_progress(element_id)
            
            # Tool icon with larger font and better contrast
            icon_text = self.title_font.render(self.get_tool_icon(tool), True, self.ui_colors['selected_text'] if self.selected_tool == tool else self.ui_colors['text_primary'])
            icon_rect = icon_text.get_rect(center=(x + button_size // 2, y + button_size // 2))
            self.screen.blit(icon_text, icon_rect)
    
    def draw_colors_section(self, panel_x, panel_y):
        """Draw colors section in the right panel - Horizontal layout with better spacing"""
        # Title
        title_text = self.title_font.render("COLORS", True, self.ui_colors['text_primary'])
        title_rect = title_text.get_rect(center=(panel_x + 100, panel_y + 30))
        self.screen.blit(title_text, title_rect)
        
        # Color swatches - horizontal layout with more spacing
        swatch_size = 45  # Larger swatches
        spacing = 20  # More spacing between swatches
        start_x = panel_x + 30  # More left margin
        start_y = panel_y + 70  # More top margin
        
        for i, color in enumerate(self.colors):
            row = i // 4  # 4 colors per row
            col = i % 4
            
            x = start_x + col * (swatch_size + spacing)
            y = start_y + row * (swatch_size + spacing)
            
            # Color swatch with border
            swatch_rect = (x, y, swatch_size, swatch_size)
            self.draw_rounded_rect(self.screen, color.rgb, swatch_rect, self.ui_sizes['border_radius'])
            
            # Border for better visibility
            border_color = self.ui_colors['selected_bg'] if self.selected_color == color else self.ui_colors['toolbar_border']
            pygame.draw.rect(self.screen, border_color, swatch_rect, 3, self.ui_sizes['border_radius'])
            
            # Check if cursor is hovering over this swatch
            element_id = f"color_{i}"
            is_hovering = self.is_cursor_hovering(swatch_rect)
            
            if is_hovering:
                self.update_hover_progress(element_id)
                self.draw_selection_indicator(swatch_rect, self.hover_progress)
            else:
                self.reset_hover_progress(element_id)
    
    def get_tool_icon(self, tool):
        """Get icon symbol for tool"""
        return tool.value
    
    def draw_tool_buttons(self, center_x, center_y):
        """Draw tool buttons in centered toolbar (for image selection)"""
        tools = [Tool.IMAGE_URL, Tool.LOCAL_IMAGE]
        
        # Grid layout for tools
        grid_width = 2
        grid_height = (len(tools) + grid_width - 1) // grid_width
        
        button_width = 150
        button_height = 80
        spacing = 20
        start_x = center_x + (self.ui_sizes['center_panel_width'] - grid_width * (button_width + spacing)) // 2
        start_y = center_y + 80
        
        for i, tool in enumerate(tools):
            row = i // grid_width
            col = i % grid_width
            
            x = start_x + col * (button_width + spacing)
            y = start_y + row * (button_height + spacing)
            
            button_rect = (x, y, button_width, button_height)
            
            # Check if cursor is hovering over this button
            element_id = f"tool_{tool.name.lower()}"
            is_hovering = self.is_cursor_hovering(button_rect)
            
            if is_hovering:
                # Update hover progress
                self.update_hover_progress(element_id)
                # Draw selection indicator
                self.draw_selection_indicator(button_rect, self.hover_progress)
            else:
                # Reset hover progress if not hovering
                self.reset_hover_progress(element_id)
            
            # Button background
            bg_color = self.ui_colors['selected_bg'] if tool == self.selected_tool else self.ui_colors['toolbar_bg']
            self.draw_rounded_rect(self.screen, bg_color, button_rect, self.ui_sizes['border_radius'])
            
            # Button text
            text_color = self.ui_colors['selected_text'] if tool == self.selected_tool else self.ui_colors['text_primary']
            
            # Tool display name (just the emoji)
            display_name = tool.value
            
            tool_text = self.tool_font.render(display_name, True, text_color)
            text_rect = tool_text.get_rect(center=(x + button_width // 2, y + button_height // 2))
            self.screen.blit(tool_text, text_rect)
                
    def draw_palette(self):
        """Draw modern color palette"""
        palette_rect = (0, self.height - self.ui_sizes['palette_height'], self.width, self.ui_sizes['palette_height'])
        self.draw_rounded_rect(self.screen, self.ui_colors['palette_bg'], palette_rect, self.ui_sizes['border_radius'])
        
        # Title
        title_text = self.title_font.render("COLORS", True, self.ui_colors['text_primary'])
        self.screen.blit(title_text, (self.ui_sizes['padding'], self.height - self.ui_sizes['palette_height'] + self.ui_sizes['padding']))
        
        # Color swatches
        x_offset = 120
        for i, color in enumerate(self.colors):
            swatch_rect = (x_offset + i * (self.ui_sizes['color_swatch_size'] + self.ui_sizes['margin']), 
                          self.height - self.ui_sizes['palette_height'] + self.ui_sizes['padding'] + 30,
                          self.ui_sizes['color_swatch_size'], self.ui_sizes['color_swatch_size'])
            
            # Color swatch with border
            border_color = self.ui_colors['selected_bg'] if color == self.selected_color else self.ui_colors['palette_border']
            pygame.draw.rect(self.screen, border_color, swatch_rect, border_radius=self.ui_sizes['border_radius'])
            
            # Inner color
            inner_rect = (swatch_rect[0] + 2, swatch_rect[1] + 2, 
                         swatch_rect[2] - 4, swatch_rect[3] - 4)
            pygame.draw.rect(self.screen, color.rgb, inner_rect, border_radius=self.ui_sizes['border_radius'] - 2)
            
            # Color name
            name_text = self.info_font.render(color.name, True, self.ui_colors['text_secondary'])
            name_rect = name_text.get_rect(center=(swatch_rect[0] + swatch_rect[2] // 2, 
                                                  swatch_rect[1] + swatch_rect[3] + 15))
            self.screen.blit(name_text, name_rect)
            
    def draw_status_bar(self):
        """Draw status bar with current tool and color info - Horizontal layout"""
        status_height = 25  # Smaller height
        status_rect = (0, 0, self.width, status_height)  # Top of screen
        pygame.draw.rect(self.screen, self.ui_colors['background'], status_rect)
        
        # Status text based on game state
        if self.game.get_current_state() == GameState.SELECTING_IMAGE:
            status_text = f"Images: {len(self.game.get_available_images())} | Hold 3s to select"
        elif self.game.get_current_state() == GameState.PAINTING:
            completion = self.game.get_completion_percentage()
            status_text = f"{self.selected_tool.value} | {self.selected_color.name} | {completion:.1f}%"
            if self.selected_tool == Tool.BRUSH:
                status_text += f" | Size: {self.brush_sizes[self.current_brush_size]}"
        elif self.game.get_current_state() == GameState.COMPLETED:
            status_text = "🎉 Complete! Press R for new image"
        else:
            status_text = "Ready to paint!"
            
        text_surface = self.info_font.render(status_text, True, self.ui_colors['text_secondary'])
        text_rect = text_surface.get_rect(center=(self.width // 2, status_height // 2))
        self.screen.blit(text_surface, text_rect)
        
    def draw_help_overlay(self):
        """Draw help overlay with instructions"""
        if not self.show_help:
            return
            
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Help box
        help_rect = (self.width // 4, self.height // 4, self.width // 2, self.height // 2)
        self.draw_rounded_rect(self.screen, self.ui_colors['toolbar_bg'], help_rect, self.ui_sizes['border_radius'])
        
        # Help content
        help_lines = [
            "AIR PAINTER CONTROLS",
            "",
            "👆 Point with index finger",
            "⏱️ Hold cursor for 3 seconds to select",
            "🖱️ Select tools from center panel",
            "🎨 Choose colors from bottom panel",
            "🌐 Import images/videos from URLs",
            "⌨️ Press H for help",
            "⌨️ Press ESC to exit",
            "⌨️ +/- to adjust brush size",
            "⌨️ Ctrl+S to save artwork",
            "",
            "Press H again to close"
        ]
        
        y_offset = help_rect[1] + self.ui_sizes['padding']
        for line in help_lines:
            if line.startswith("👆") or line.startswith("⏱️") or line.startswith("🖱️") or line.startswith("🎨") or line.startswith("🌐") or line.startswith("⌨️"):
                text_surface = self.tool_font.render(line, True, self.ui_colors['text_primary'])
            elif line == "AIR PAINTER CONTROLS":
                text_surface = self.title_font.render(line, True, self.ui_colors['selected_bg'])
            else:
                text_surface = self.info_font.render(line, True, self.ui_colors['text_secondary'])
                
            text_rect = text_surface.get_rect(center=(help_rect[0] + help_rect[2] // 2, y_offset))
            self.screen.blit(text_surface, text_rect)
            y_offset += 25
    
    def is_cursor_hovering(self, rect):
        """Check if cursor is hovering over a rectangle"""
        if not self.cursor_pos:
            return False
        x, y = self.cursor_pos
        return (rect[0] <= x <= rect[0] + rect[2] and 
                rect[1] <= y <= rect[1] + rect[3])
    
    def update_hover_progress(self, element_id):
        """Update hover progress for an element"""
        current_time = time.time()
        
        if element_id not in self.hover_start_time:
            self.hover_start_time[element_id] = current_time
            self.current_hover_element = element_id
        
        if self.current_hover_element == element_id:
            elapsed_time = current_time - self.hover_start_time[element_id]
            self.hover_progress = min(1.0, elapsed_time / self.selection_hold_time)
        else:
            # Different element, reset progress
            self.hover_start_time[element_id] = current_time
            self.current_hover_element = element_id
            self.hover_progress = 0.0
    
    def reset_hover_progress(self, element_id):
        """Reset hover progress for an element"""
        if element_id in self.hover_start_time:
            del self.hover_start_time[element_id]
        if self.current_hover_element == element_id:
            self.current_hover_element = None
            self.hover_progress = 0.0
    
    def draw_selection_indicator(self, rect, progress):
        """Draw selection progress indicator around a rectangle"""
        if progress <= 0:
            return
            
        # Calculate indicator properties
        indicator_width = 4
        indicator_color = self.ui_colors['selected_bg']
        
        # Draw progress ring around the rectangle
        x, y, w, h = rect
        center_x = x + w // 2
        center_y = y + h // 2
        radius = max(w, h) // 2 + 10
        
        # Draw progress arc
        if progress > 0:
            # Calculate arc angles
            start_angle = -90  # Start from top
            end_angle = start_angle + (360 * progress)
            
            # Draw multiple arcs for better visibility
            for i in range(3):
                current_radius = radius + i * 2
                pygame.draw.arc(self.screen, indicator_color, 
                              (center_x - current_radius, center_y - current_radius, 
                               current_radius * 2, current_radius * 2),
                              np.radians(start_angle), np.radians(end_angle), indicator_width)
        
        # Draw countdown text
        if progress < 1.0:
            remaining_time = self.selection_hold_time * (1.0 - progress)
            countdown_text = f"{remaining_time:.1f}s"
            text_surface = self.info_font.render(countdown_text, True, indicator_color)
            text_rect = text_surface.get_rect(center=(center_x, y - 20))
            self.screen.blit(text_surface, text_rect)
    
    def check_selection_complete(self, element_id):
        """Check if selection is complete for an element"""
        if element_id in self.hover_start_time:
            elapsed_time = time.time() - self.hover_start_time[element_id]
            if elapsed_time >= self.selection_hold_time:
                # Selection complete, reset progress
                self.reset_hover_progress(element_id)
                return True
        return False
    
    def draw_url_input_dialog(self):
        """Draw URL input dialog"""
        if not self.showing_url_input:
            return
            
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Dialog box
        dialog_width = 600
        dialog_height = 300
        dialog_x = (self.width - dialog_width) // 2
        dialog_y = (self.height - dialog_height) // 2
        dialog_rect = (dialog_x, dialog_y, dialog_width, dialog_height)
        
        self.draw_rounded_rect(self.screen, self.ui_colors['toolbar_bg'], dialog_rect, self.ui_sizes['border_radius'])
        
        # Title
        title_text = self.title_font.render("Import from URL", True, self.ui_colors['text_primary'])
        title_rect = title_text.get_rect(center=(dialog_x + dialog_width // 2, dialog_y + 30))
        self.screen.blit(title_text, title_rect)
        
        if self.url_processing:
            # Show processing progress
            progress_text = self.tool_font.render(self.url_processing_message, True, self.ui_colors['text_primary'])
            progress_rect = progress_text.get_rect(center=(dialog_x + dialog_width // 2, dialog_y + 100))
            self.screen.blit(progress_text, progress_rect)
            
            # Progress bar
            bar_rect = (dialog_x + 50, dialog_y + 150, dialog_width - 100, 20)
            pygame.draw.rect(self.screen, self.ui_colors['text_secondary'], bar_rect, border_radius=10)
            
            fill_width = int((dialog_width - 100) * self.url_processing_progress)
            if fill_width > 0:
                fill_rect = (dialog_x + 50, dialog_y + 150, fill_width, 20)
                pygame.draw.rect(self.screen, self.ui_colors['success'], fill_rect, border_radius=10)
            
            # Progress percentage
            percent_text = self.info_font.render(f"{int(self.url_processing_progress * 100)}%", True, self.ui_colors['text_secondary'])
            percent_rect = percent_text.get_rect(center=(dialog_x + dialog_width // 2, dialog_y + 180))
            self.screen.blit(percent_text, percent_rect)
            
        else:
            # URL input field
            input_label = self.tool_font.render("Enter URL:", True, self.ui_colors['text_primary'])
            self.screen.blit(input_label, (dialog_x + 30, dialog_y + 80))
            
            # Input box
            input_rect = (dialog_x + 30, dialog_y + 110, dialog_width - 60, 40)
            pygame.draw.rect(self.screen, self.ui_colors['background'], input_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.ui_colors['text_secondary'], input_rect, 2, border_radius=5)
            
            # Input text
            display_text = self.url_input_text
            if len(display_text) > 50:
                display_text = "..." + display_text[-47:]
            
            text_surface = self.tool_font.render(display_text, True, self.ui_colors['text_primary'])
            self.screen.blit(text_surface, (dialog_x + 40, dialog_y + 120))
            
            # Cursor
            if time.time() % 1 > 0.5:  # Blinking cursor
                cursor_x = dialog_x + 40 + self.tool_font.size(display_text[:self.url_input_cursor_pos])[0]
                pygame.draw.line(self.screen, self.ui_colors['text_primary'], 
                               (cursor_x, dialog_y + 120), (cursor_x, dialog_y + 140), 2)
            
            # Instructions
            instructions = [
                "• For images: Enter direct image URL (jpg, png, etc.)",
                "• For videos: Enter YouTube video URL",
                "• Press ENTER to process, ESC to cancel"
            ]
            
            y_offset = dialog_y + 180
            for instruction in instructions:
                inst_text = self.info_font.render(instruction, True, self.ui_colors['text_secondary'])
                self.screen.blit(inst_text, (dialog_x + 30, y_offset))
                y_offset += 20
            
    def handle_hand_gestures(self):
        """Process hand gestures for drawing and UI interaction with two-hand support"""
        if not self.camera_active:
            return None, None, None
            
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.hands is None:
            return None, None, None
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            self.hand_confidence = 0.0
            self.cursor_pos = None
            self.show_circular_menus = False
            return None, None, None
        
        # Handle two hands for circular menus
        if len(results.multi_hand_landmarks) >= 2:
            # Get both hands
            left_hand = results.multi_hand_landmarks[0]
            right_hand = results.multi_hand_landmarks[1]
            
            # Check if both hands have closed fists (all fingers closed)
            left_fist = self.is_hand_closed(left_hand)
            right_fist = self.is_hand_closed(right_hand)
            
            if left_fist and right_fist:
                # Show circular menus
                self.show_circular_menus = True
                
                # Convert hand positions to screen coordinates
                left_wrist = left_hand.landmark[0]
                right_wrist = right_hand.landmark[0]
                
                self.left_hand_pos = (int(left_wrist.x * self.width), int(left_wrist.y * self.height))
                self.right_hand_pos = (int(right_wrist.x * self.width), int(right_wrist.y * self.height))
                
                # Convert landmarks to list format
                self.left_hand_landmarks = []
                for landmark in left_hand.landmark:
                    self.left_hand_landmarks.append([landmark.x, landmark.y, landmark.z])
                
                self.right_hand_landmarks = []
                for landmark in right_hand.landmark:
                    self.right_hand_landmarks.append([landmark.x, landmark.y, landmark.z])
                
                # Use right hand for cursor (for painting)
                index_tip = right_hand.landmark[8]
                x, y = int(index_tip.x * self.width), int(index_tip.y * self.height)
                self.cursor_pos = (x, y)
                
                # Update hand trail
                self.hand_trail.append((x, y))
                if len(self.hand_trail) > 20:
                    self.hand_trail.pop(0)
                
                # Update confidence
                self.hand_confidence = 0.9
                
                # Hand position for painting
                hand_position = [x, y, right_hand.landmark[0].z]
                
                return (x, y), self.right_hand_landmarks, hand_position
            else:
                self.show_circular_menus = False
        
        # Single hand mode (original logic)
        hand = results.multi_hand_landmarks[0]
        index_tip = hand.landmark[8]
        thumb_tip = hand.landmark[4]
        
        # Convert to screen coordinates
        x, y = int(index_tip.x * self.width), int(index_tip.y * self.height)
        
        # Update cursor position
        self.cursor_pos = (x, y)
        
        # Update hand trail
        self.hand_trail.append((x, y))
        if len(self.hand_trail) > 20:  # Limit trail length
            self.hand_trail.pop(0)
            
        # Calculate pinch distance and strength (for backward compatibility)
        distance = np.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)
        self.pinch_strength = max(0, 1 - distance / 0.1)  # Normalize to 0-1
        
        # Update confidence (simplified)
        self.hand_confidence = 0.9  # MediaPipe doesn't provide direct confidence
        
        # Convert landmarks to list format for new painting gesture
        hand_landmarks = []
        for landmark in hand.landmark:
            hand_landmarks.append([landmark.x, landmark.y, landmark.z])
        
        # Hand position (center of hand)
        hand_position = [x, y, hand.landmark[0].z]  # Using wrist Z coordinate
        
        return (x, y), hand_landmarks, hand_position
    
    def is_hand_closed(self, hand_landmarks):
        """Check if hand is closed (fist gesture)"""
        if not hand_landmarks:
            return False
        
        # Get finger tip landmarks
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Get finger base landmarks (MCP joints)
        thumb_base = hand_landmarks.landmark[3]
        index_base = hand_landmarks.landmark[5]
        middle_base = hand_landmarks.landmark[9]
        ring_base = hand_landmarks.landmark[13]
        pinky_base = hand_landmarks.landmark[17]
        
        # Check if all fingers are closed (tip is below base)
        thumb_closed = thumb_tip.y > thumb_base.y
        index_closed = index_tip.y > index_base.y
        middle_closed = middle_tip.y > middle_base.y
        ring_closed = ring_tip.y > ring_base.y
        pinky_closed = pinky_tip.y > pinky_base.y
        
        return thumb_closed and index_closed and middle_closed and ring_closed and pinky_closed
    
    def draw_circular_menus(self):
        """Draw circular menus for tools and colors at fixed positions"""
        if not self.show_circular_menus:
            return
        
        # Fixed menu positions in center of screen
        screen_center_x = self.width // 2
        screen_center_y = self.height // 2
        
        # Left menu (tools) - slightly left of center
        left_menu_x = screen_center_x - 200
        left_menu_y = screen_center_y
        
        # Right menu (colors) - slightly right of center
        right_menu_x = screen_center_x + 200
        right_menu_y = screen_center_y
        
        # Draw left hand tools menu
        self.draw_circular_tools_menu((left_menu_x, left_menu_y))
        
        # Draw right hand colors menu
        self.draw_circular_colors_menu((right_menu_x, right_menu_y))
    
    def draw_circular_tools_menu(self, center_pos):
        """Draw circular tools menu with zoom effect and circular motion detection"""
        center_x, center_y = center_pos
        
        # Tools to display
        tools = [Tool.BRUSH, Tool.ERASER, Tool.CLEAR, Tool.UNDO, Tool.SAVE]
        
        # Calculate button positions in a circle
        num_tools = len(tools)
        angle_step = 2 * np.pi / num_tools
        
        # Get hand position for circular motion detection
        hand_pos = self.right_hand_pos if self.right_hand_pos else self.cursor_pos
        
        for i, tool in enumerate(tools):
            angle = i * angle_step
            button_x = center_x + int(self.circular_menu_radius * np.cos(angle))
            button_y = center_y + int(self.circular_menu_radius * np.sin(angle))
            
            # Calculate distance from hand to button
            if hand_pos:
                hand_x, hand_y = hand_pos
                distance = np.sqrt((hand_x - button_x) ** 2 + (hand_y - button_y) ** 2)
                
                # Zoom effect: button grows when hand is close
                zoom_factor = 1.0
                if distance <= self.menu_button_radius * 2:
                    zoom_factor = min(1.4, 1.0 + (self.menu_button_radius * 2 - distance) / (self.menu_button_radius * 2) * 0.4)
                
                # Apply zoom to button size
                current_radius = int(self.menu_button_radius * zoom_factor)
                
                # Check if hand is hovering over this button
                if distance <= current_radius:
                    # Highlight button
                    pygame.draw.circle(self.screen, self.ui_colors['selected_bg'], (button_x, button_y), current_radius)
                    
                    # Update selection progress
                    element_id = f"circular_tool_{tool.name.lower()}"
                    self.update_hover_progress(element_id)
                    
                    # Draw selection indicator
                    progress = self.hover_progress
                    if progress > 0:
                        indicator_radius = int(current_radius + 5 * progress)
                        pygame.draw.circle(self.screen, self.ui_colors['success'], (button_x, button_y), indicator_radius, 3)
                    
                    # Check if selection is complete
                    if self.check_selection_complete(element_id):
                        self.selected_tool = tool
                        logger.info(f"Tool selected via circular menu: {tool.name}")
                else:
                    self.reset_hover_progress(f"circular_tool_{tool.name.lower()}")
                    current_radius = self.menu_button_radius
            else:
                current_radius = self.menu_button_radius
            
            # Draw button border
            pygame.draw.circle(self.screen, self.ui_colors['toolbar_border'], (button_x, button_y), current_radius, 2)
            
            # Draw tool text (since emojis don't show properly)
            tool_text = tool.name.upper()
            text_surface = self.info_font.render(tool_text, True, self.ui_colors['text_primary'])
            text_rect = text_surface.get_rect(center=(button_x, button_y))
            self.screen.blit(text_surface, text_rect)
    
    def draw_circular_colors_menu(self, center_pos):
        """Draw circular colors menu with zoom effect and circular motion detection"""
        center_x, center_y = center_pos
        
        # Calculate button positions in a circle
        num_colors = len(self.colors)
        angle_step = 2 * np.pi / num_colors
        
        # Get hand position for circular motion detection
        hand_pos = self.right_hand_pos if self.right_hand_pos else self.cursor_pos
        
        for i, color in enumerate(self.colors):
            angle = i * angle_step
            button_x = center_x + int(self.circular_menu_radius * np.cos(angle))
            button_y = center_y + int(self.circular_menu_radius * np.sin(angle))
            
            # Calculate distance from hand to button
            if hand_pos:
                hand_x, hand_y = hand_pos
                distance = np.sqrt((hand_x - button_x) ** 2 + (hand_y - button_y) ** 2)
                
                # Zoom effect: button grows when hand is close
                zoom_factor = 1.0
                if distance <= self.menu_button_radius * 2:
                    zoom_factor = min(1.4, 1.0 + (self.menu_button_radius * 2 - distance) / (self.menu_button_radius * 2) * 0.4)
                
                # Apply zoom to button size
                current_radius = int(self.menu_button_radius * zoom_factor)
                
                # Check if hand is hovering over this button
                if distance <= current_radius:
                    # Highlight button
                    pygame.draw.circle(self.screen, self.ui_colors['selected_bg'], (button_x, button_y), current_radius, 3)
                    
                    # Update selection progress
                    element_id = f"circular_color_{i}"
                    self.update_hover_progress(element_id)
                    
                    # Draw selection indicator
                    progress = self.hover_progress
                    if progress > 0:
                        indicator_radius = int(current_radius + 5 * progress)
                        pygame.draw.circle(self.screen, self.ui_colors['success'], (button_x, button_y), indicator_radius, 3)
                    
                    # Check if selection is complete
                    if self.check_selection_complete(element_id):
                        self.selected_color = color
                        logger.info(f"Color selected via circular menu: {color.name}")
                else:
                    self.reset_hover_progress(f"circular_color_{i}")
                    current_radius = self.menu_button_radius
            else:
                current_radius = self.menu_button_radius
            
            # Draw color button
            pygame.draw.circle(self.screen, color.rgb, (button_x, button_y), current_radius)
            
            # Draw border
            border_color = self.ui_colors['selected_bg'] if self.selected_color == color else self.ui_colors['toolbar_border']
            pygame.draw.circle(self.screen, border_color, (button_x, button_y), current_radius, 2)
        
    def handle_tool_selection(self, x: int, y: int):
        """Handle tool selection from centered toolbar"""
        # Calculate center position
        center_x = (self.width - self.ui_sizes['center_panel_width']) // 2
        center_y = (self.height - self.ui_sizes['center_panel_height']) // 2
        
        # Check if cursor is within toolbar bounds
        if (center_x <= x <= center_x + self.ui_sizes['center_panel_width'] and 
            center_y <= y <= center_y + self.ui_sizes['center_panel_height']):
            
            tools = [Tool.BRUSH, Tool.ERASER, Tool.CLEAR, Tool.UNDO, Tool.SAVE, Tool.IMAGE_URL, Tool.LOCAL_IMAGE]
            
            # Grid layout for tools
            grid_width = 3
            button_width = 100
            button_height = 60
            spacing = 15
            start_x = center_x + (self.ui_sizes['center_panel_width'] - grid_width * (button_width + spacing)) // 2
            start_y = center_y + 80
            
            for i, tool in enumerate(tools):
                row = i // grid_width
                col = i % grid_width
                
                button_x = start_x + col * (button_width + spacing)
                button_y = start_y + row * (button_height + spacing)
                button_rect = (button_x, button_y, button_width, button_height)
                
                if (button_rect[0] <= x <= button_rect[0] + button_rect[2] and 
                    button_rect[1] <= y <= button_rect[1] + button_rect[3]):
                    
                    element_id = f"tool_{tool.value}"
                    
                    # Check if selection is complete
                    if self.check_selection_complete(element_id):
                        if tool == Tool.CLEAR:
                            # Clear the current painting (reset to white)
                            if self.game.get_current_image():
                                self.game.current_painting_surface.fill((255, 255, 255))
                                self.game.painted_areas.clear()
                                self.game.completion_percentage = 0.0
                                self.game.paint_history.clear()
                                logger.info("Painting cleared")
                        elif tool == Tool.UNDO:
                            # Undo last painting action
                            success = self.game.undo_last_paint()
                            if success:
                                logger.info("Undo performed via toolbar")
                            else:
                                logger.info("No actions to undo")
                        elif tool == Tool.SAVE:
                            self.save_artwork()
                        elif tool == Tool.IMAGE_URL:
                            self.show_url_input_dialog("image")
                            self.selected_tool = Tool.BRUSH  # Reset to brush after showing dialog
                        elif tool == Tool.LOCAL_IMAGE:
                            self.show_url_input_dialog("video")
                            self.selected_tool = Tool.BRUSH  # Reset to brush after showing dialog
                        else:
                            self.selected_tool = tool
                            logger.info(f"Tool selected: {tool.value}")
                        return True
        return False
        
    def handle_image_selection(self, x: int, y: int, pinch_distance: float):
        """Handle image selection from the centered grid"""
        available_images = self.game.get_available_images()
        
        # Calculate center position
        center_x = (self.width - self.ui_sizes['center_panel_width']) // 2
        center_y = (self.height - self.ui_sizes['center_panel_height']) // 2
        
        # Check if cursor is within toolbar bounds
        if (center_x <= x <= center_x + self.ui_sizes['center_panel_width'] and 
            center_y <= y <= center_y + self.ui_sizes['center_panel_height']):
            
            if not available_images:
                # Handle initial selection screen
                self.handle_initial_selection(x, y)
            else:
                # Handle image grid selection
                self.handle_image_grid_selection(x, y)
        return False
    
    def handle_initial_selection(self, x: int, y: int):
        """Handle selection from initial screen"""
        center_x = (self.width - self.ui_sizes['center_panel_width']) // 2
        center_y = (self.height - self.ui_sizes['center_panel_height']) // 2
        
        # Three options positions
        options = [Tool.LOCAL_IMAGE, Tool.IMAGE_URL, Tool.YOUTUBE_FRAME]
        button_width = 300
        button_height = 80
        spacing = 20
        start_y = center_y + 80
        
        for i, tool in enumerate(options):
            y_pos = start_y + i * (button_height + spacing)
            x_pos = center_x + (self.ui_sizes['center_panel_width'] - button_width) // 2
            
            button_rect = (x_pos, y_pos, button_width, button_height)
            
            # Check if cursor is in this button and selection is complete
            element_id = f"initial_option_{tool.name.lower()}"
            if self.is_cursor_hovering(button_rect) and self.check_selection_complete(element_id):
                if tool == Tool.LOCAL_IMAGE:
                    # For now, just show URL input for local image
                    self.show_url_input_dialog("image")
                elif tool == Tool.IMAGE_URL:
                    self.show_url_input_dialog("image")
                elif tool == Tool.YOUTUBE_FRAME:
                    # YouTube is disabled for now
                    logger.info("YouTube frame selection is not yet implemented")
                break
    
    def handle_image_grid_selection(self, x: int, y: int):
        """Handle selection from image grid"""
        available_images = self.game.get_available_images()
        if not available_images:
            return False
            
        # Calculate center position
        center_x = (self.width - self.ui_sizes['center_panel_width']) // 2
        center_y = (self.height - self.ui_sizes['center_panel_height']) // 2
        
        # Grid layout for images
        grid_width = 2
        image_size = 120
        spacing = 20
        start_x = center_x + (self.ui_sizes['center_panel_width'] - grid_width * (image_size + spacing)) // 2
        start_y = center_y + 80
        
        for i, image in enumerate(available_images):
            row = i // grid_width
            col = i % grid_width
            
            image_x = start_x + col * (image_size + spacing)
            image_y = start_y + row * (image_size + spacing + 30)
            image_rect = (image_x, image_y, image_size, image_size)
            
            # Check if cursor is within image bounds
            if (image_rect[0] <= x <= image_rect[0] + image_rect[2] and 
                image_rect[1] <= y <= image_rect[1] + image_rect[3]):
                
                element_id = f"image_{i}"
                
                # Check if selection is complete
                if self.check_selection_complete(element_id):
                    # Select the image
                    self.game.select_image(image)
                    logger.info(f"Image selected: {image.name}")
                    return True
        return False
    
    def handle_painting(self, x: int, y: int, hand_landmarks, hand_position):
        """Handle painting with new gesture: index finger forward + hand forward + 2 second hold"""
        
        # Check if index finger is pointing forward (gesture detection)
        if hand_landmarks and len(hand_landmarks) >= 21:
            # Get index finger tip (landmark 8) and middle finger tip (landmark 12)
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]
            wrist = hand_landmarks[0]
            
            # Check if index finger is extended forward (higher than middle finger)
            index_forward = index_tip[1] < middle_tip[1]  # Y coordinate is inverted
            
            # Check if hand has moved forward
            hand_forward = False
            if self.last_hand_position:
                forward_movement = self.last_hand_position[2] - hand_position[2]  # Z coordinate
                hand_forward = forward_movement > self.hand_forward_threshold
            
            # Update last hand position
            self.last_hand_position = hand_position
            
            # Check if painting gesture is active
            if index_forward and hand_forward:
                if self.painting_gesture_start_time is None:
                    # Start the gesture timer
                    self.painting_gesture_start_time = time.time()
                    logger.info("Painting gesture started")
                
                # Check if gesture has been held for 2 seconds
                gesture_duration = time.time() - self.painting_gesture_start_time
                if gesture_duration >= self.painting_gesture_threshold:
                    # Activate painting
                    if not self.painting_gesture_active:
                        self.painting_gesture_active = True
                        logger.info("Painting gesture activated")
                    
                    # Perform painting
                    if self.selected_tool == Tool.BRUSH:
                        if not self.game.is_painting_active():
                            success = self.game.start_painting(x, y, self.selected_color.rgb, self.brush_sizes[self.current_brush_size])
                        else:
                            success = self.game.continue_painting(x, y, self.selected_color.rgb, self.brush_sizes[self.current_brush_size])
                        
                        if success:
                            logger.info("Area painted", extra={"color": self.selected_color.name, "position": (x, y)})
                    elif self.selected_tool == Tool.ERASER:
                        if not self.game.is_painting_active():
                            success = self.game.start_painting(x, y, (255, 255, 255), 25)
                        else:
                            success = self.game.continue_painting(x, y, (255, 255, 255), 25)
                        
                        if success:
                            logger.info("Area erased", extra={"position": (x, y)})
                else:
                    # Show gesture progress
                    progress = gesture_duration / self.painting_gesture_threshold
                    logger.debug(f"Painting gesture progress: {progress:.2f}")
            else:
                # Reset gesture if conditions are not met
                if self.painting_gesture_start_time is not None:
                    self.painting_gesture_start_time = None
                    logger.info("Painting gesture reset")
                
                # Stop painting if gesture is broken
                if self.painting_gesture_active:
                    self.painting_gesture_active = False
                    if self.game.is_painting_active():
                        self.game.stop_painting()
                        logger.info("Painting stopped - gesture broken")
        else:
            # No hand detected, stop painting
            if self.painting_gesture_active:
                self.painting_gesture_active = False
                if self.game.is_painting_active():
                    self.game.stop_painting()
                    logger.info("Painting stopped - no hand detected")
            
            # Reset gesture state
            self.painting_gesture_start_time = None
    
    def handle_color_selection(self, x: int, y: int):
        """Handle color selection from palette"""
        if y > self.height - self.ui_sizes['palette_height']:
            x_offset = 120
            swatch_y = self.height - self.ui_sizes['palette_height'] + self.ui_sizes['padding'] + 30
            
            for i, color in enumerate(self.colors):
                swatch_rect = (x_offset + i * (self.ui_sizes['color_swatch_size'] + self.ui_sizes['margin']), 
                              swatch_y, self.ui_sizes['color_swatch_size'], self.ui_sizes['color_swatch_size'])
                
                if (swatch_rect[0] <= x <= swatch_rect[0] + swatch_rect[2] and 
                    swatch_rect[1] <= y <= swatch_rect[1] + swatch_rect[3]):
                    self.selected_color = color
                    logger.info(f"Color selected: {color.name}")
                    return True
        return False
        
    def draw_on_canvas(self, x: int, y: int, distance: float):
        """Draw on canvas based on current tool and gesture"""
        if distance < 0.05:  # Pinch detected
            self.gesture_active = True
            self.last_gesture_time = time.time()
            
            if self.selected_tool == Tool.BRUSH:
                brush_size = self.brush_sizes[self.current_brush_size]
                pygame.draw.circle(self.canvas, self.selected_color.rgb, (x, y), brush_size)
                
                # Draw line to previous position for smooth strokes
                if self.last_pos:
                    pygame.draw.line(self.canvas, self.selected_color.rgb, self.last_pos, (x, y), brush_size * 2)
                    
            elif self.selected_tool == Tool.ERASER:
                eraser_size = 25
                pygame.draw.circle(self.canvas, (255, 255, 255), (x, y), eraser_size)
                
                if self.last_pos:
                    pygame.draw.line(self.canvas, (255, 255, 255), self.last_pos, (x, y), eraser_size * 2)
                    
            self.last_pos = (x, y)
            self.drawing = True
            
        else:
            # Check if gesture should be deactivated (with cooldown)
            if time.time() - self.last_gesture_time > self.gesture_cooldown:
                self.gesture_active = False
            self.drawing = False
            self.last_pos = None
            
    def save_canvas_state(self):
        """Save current canvas state for undo functionality"""
        if len(self.drawing_history) > 10:  # Limit history
            self.drawing_history.pop(0)
        self.drawing_history.append(self.canvas.copy())
        
    def draw_image_selection_screen(self):
        """Draw the image selection screen"""
        # Background
        self.screen.fill(self.ui_colors['background'])
        
        # Title
        title_text = self.title_font.render("Choose an Image to Color", True, self.ui_colors['text_primary'])
        title_rect = title_text.get_rect(center=(self.width // 2, 100))
        self.screen.blit(title_text, title_rect)
        
        # Instructions
        instructions = self.info_font.render("Hold cursor on image for 3 seconds to select", True, self.ui_colors['text_secondary'])
        inst_rect = instructions.get_rect(center=(self.width // 2, 150))
        self.screen.blit(instructions, inst_rect)
        
        # Draw centered toolbar with images
        self.draw_toolbar()
    
    def draw_image_grid(self, images):
        """Draw a grid of available images"""
        grid_width = 3
        grid_height = (len(images) + grid_width - 1) // grid_width
        
        image_size = 200
        spacing = 50
        start_x = (self.width - self.ui_sizes['toolbar_width']) // 2 - (grid_width * (image_size + spacing)) // 2
        start_y = 200
        
        for i, image in enumerate(images):
            row = i // grid_width
            col = i % grid_width
            
            x = start_x + col * (image_size + spacing)
            y = start_y + row * (image_size + spacing + 30)
            
            # Image background
            image_rect = (x, y, image_size, image_size)
            self.draw_rounded_rect(self.screen, self.ui_colors['toolbar_bg'], image_rect, self.ui_sizes['border_radius'])
            
            # Try to load and display thumbnail
            try:
                if os.path.exists(image.thumbnail_path):
                    thumbnail = pygame.image.load(image.thumbnail_path)
                    thumbnail = pygame.transform.scale(thumbnail, (image_size - 20, image_size - 20))
                    self.screen.blit(thumbnail, (x + 10, y + 10))
            except Exception as e:
                logger.warning(f"Could not load thumbnail: {e}")
            
            # Image name
            name_text = self.tool_font.render(image.name.replace('_', ' ').title(), True, self.ui_colors['text_primary'])
            name_rect = name_text.get_rect(center=(x + image_size // 2, y + image_size + 15))
            self.screen.blit(name_text, name_rect)
            
            # Difficulty
            diff_text = self.info_font.render(image.difficulty.upper(), True, self.ui_colors['text_secondary'])
            diff_rect = diff_text.get_rect(center=(x + image_size // 2, y + image_size + 35))
            self.screen.blit(diff_text, diff_rect)
    
    def draw_painting_screen(self):
        """Draw the painting screen with the selected image"""
        # Draw the current painting surface
        current_image = self.game.get_current_image()
        if current_image:
            # Scale image to fit screen
            scaled_image = self.scale_image_to_fit(current_image)
            self.screen.blit(scaled_image, (0, 0))
        
        # Draw centered toolbar with tools
        self.draw_toolbar()
    
    def scale_image_to_fit(self, image):
        """Scale image to fit the available screen space"""
        if not image:
            return None
            
        available_width = self.width
        available_height = self.height - self.ui_sizes['palette_height']
        
        # Calculate scale to fit
        scale_x = available_width / image.get_width()
        scale_y = available_height / image.get_height()
        scale = min(scale_x, scale_y)
        
        new_width = int(image.get_width() * scale)
        new_height = int(image.get_height() * scale)
        
        return pygame.transform.scale(image, (new_width, new_height))
    
    def save_artwork(self):
        """Save artwork to file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"airpainter_artwork_{timestamp}.png"
            
            # Save the current painting surface
            current_image = self.game.get_current_image()
            if current_image:
                pygame.image.save(current_image, filename)
                logger.info("Artwork saved", extra={"filename": filename})
            else:
                logger.warning("No artwork to save")
        except Exception as e:
            logger.error("Failed to save artwork", extra={"error": str(e)})
    
    def show_url_input_dialog(self, url_type: str):
        """Show URL input dialog for image or video conversion"""
        self.showing_url_input = True
        self.url_input_text = ""
        self.url_input_cursor_pos = 0
        self.url_processing = False
        self.url_processing_progress = 0.0
        self.url_processing_message = f"Enter {url_type} URL..."
        
        logger.info(f"URL input dialog opened for {url_type}")
    
    def process_image_url(self, url: str):
        """Process image URL to create coloring page"""
        try:
            self.url_processing = True
            self.url_processing_message = "Downloading image..."
            self.url_processing_progress = 0.1
            
            # Create temporary file for downloaded image
            import tempfile
            import urllib.request
            import os
            
            temp_dir = tempfile.gettempdir()
            temp_image_path = os.path.join(temp_dir, "airpainter_temp_image.jpg")
            
            # Download image
            urllib.request.urlretrieve(url, temp_image_path)
            self.url_processing_progress = 0.3
            self.url_processing_message = "Converting to coloring page..."
            
            # Convert to coloring page
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"url_image_{timestamp}.png"
            output_path = os.path.join(self.game.images_dir, output_filename)
            
            convert_image_to_coloring_page(temp_image_path, output_path)
            self.url_processing_progress = 0.8
            self.url_processing_message = "Creating thumbnail..."
            
            # Create thumbnail
            thumbnail_path = os.path.join(self.game.thumbnails_dir, f"url_image_{timestamp}_thumb.png")
            self.game.create_thumbnail(output_path, thumbnail_path)
            
            # Add to game
            image = ColoringImage(
                name=f"url_image_{timestamp}",
                filepath=output_path,
                thumbnail_path=thumbnail_path,
                difficulty="medium",
                category="url_import",
                description="Imported from URL"
            )
            self.game.available_images.append(image)
            
            # Clean up temp file
            os.remove(temp_image_path)
            
            self.url_processing_progress = 1.0
            self.url_processing_message = "Image imported successfully!"
            
            logger.info("Image URL processed successfully", extra={"url": url, "output": output_path})
            
            # Reset dialog after a short delay
            import threading
            def reset_dialog():
                time.sleep(2)
                self.showing_url_input = False
                self.url_processing = False
                self.url_processing_message = ""
            
            threading.Thread(target=reset_dialog, daemon=True).start()
            
        except Exception as e:
            self.url_processing_message = f"Error: {str(e)}"
            logger.error("Failed to process image URL", extra={"error": str(e), "url": url})
            
            # Reset dialog after error
            import threading
            def reset_dialog():
                time.sleep(3)
                self.showing_url_input = False
                self.url_processing = False
                self.url_processing_message = ""
            
            threading.Thread(target=reset_dialog, daemon=True).start()
    
    def process_video_url(self, url: str):
        """Process video URL to create coloring pages"""
        try:
            self.url_processing = True
            self.url_processing_message = "Downloading video..."
            self.url_processing_progress = 0.1
            
            # Convert video to coloring pages
            output_files = convert_video_to_coloring_page(url, self.game.images_dir)
            
            self.url_processing_progress = 0.5
            self.url_processing_message = "Creating thumbnails..."
            
            # Create thumbnails for all generated images
            for filepath in output_files:
                filename = os.path.basename(filepath)
                name = os.path.splitext(filename)[0]
                thumbnail_path = os.path.join(self.game.thumbnails_dir, f"{name}_thumb.png")
                self.game.create_thumbnail(filepath, thumbnail_path)
                
                # Add to game
                image = ColoringImage(
                    name=name,
                    filepath=filepath,
                    thumbnail_path=thumbnail_path,
                    difficulty="medium",
                    category="video_import",
                    description="Imported from video"
                )
                self.game.available_images.append(image)
            
            self.url_processing_progress = 1.0
            self.url_processing_message = f"Video processed! {len(output_files)} images created."
            
            logger.info("Video URL processed successfully", extra={"url": url, "output_count": len(output_files)})
            
            # Reset dialog after a short delay
            import threading
            def reset_dialog():
                time.sleep(3)
                self.showing_url_input = False
                self.url_processing = False
                self.url_processing_message = ""
            
            threading.Thread(target=reset_dialog, daemon=True).start()
            
        except Exception as e:
            self.url_processing_message = f"Error: {str(e)}"
            logger.error("Failed to process video URL", extra={"error": str(e), "url": url})
            
            # Reset dialog after error
            import threading
            def reset_dialog():
                time.sleep(3)
                self.showing_url_input = False
                self.url_processing = False
                self.url_processing_message = ""
            
            threading.Thread(target=reset_dialog, daemon=True).start()
        
    def handle_keyboard_events(self, event):
        """Handle keyboard shortcuts"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return False  # Exit with Q key
            elif event.key == pygame.K_h:
                self.show_help = not self.show_help
                logger.info(f"Help overlay {'shown' if self.show_help else 'hidden'}")
            elif event.key == pygame.K_ESCAPE:
                return False  # Exit
            elif event.key == pygame.K_s and event.mod & pygame.KMOD_CTRL:
                # Save canvas
                self.save_artwork()
            elif event.key == pygame.K_z and event.mod & pygame.KMOD_CTRL:
                # Undo last painting action
                success = self.game.undo_last_paint()
                if success:
                    logger.info("Undo performed via keyboard")
                else:
                    logger.info("No actions to undo")
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                # Increase brush size
                self.current_brush_size = min(len(self.brush_sizes) - 1, self.current_brush_size + 1)
                # Save brush size preference
                self.config_manager.save_brush_size(self.brush_sizes[self.current_brush_size])
                logger.info(f"Brush size increased to {self.brush_sizes[self.current_brush_size]}")
            elif event.key == pygame.K_MINUS:
                # Decrease brush size
                self.current_brush_size = max(0, self.current_brush_size - 1)
                # Save brush size preference
                self.config_manager.save_brush_size(self.brush_sizes[self.current_brush_size])
                logger.info(f"Brush size decreased to {self.brush_sizes[self.current_brush_size]}")
            elif event.key == pygame.K_r:
                # Reset game to image selection
                self.game.reset_game()
                logger.info("Game reset to image selection")
            elif self.showing_url_input:
                # Handle URL input dialog
                if event.key == pygame.K_RETURN:
                    if self.url_input_text.strip():
                        if "youtube.com" in self.url_input_text or "youtu.be" in self.url_input_text:
                            self.process_video_url(self.url_input_text.strip())
                        else:
                            self.process_image_url(self.url_input_text.strip())
                elif event.key == pygame.K_ESCAPE:
                    self.showing_url_input = False
                    self.url_input_text = ""
                    self.url_processing = False
                elif event.key == pygame.K_BACKSPACE:
                    if self.url_input_cursor_pos > 0:
                        self.url_input_text = self.url_input_text[:self.url_input_cursor_pos-1] + self.url_input_text[self.url_input_cursor_pos:]
                        self.url_input_cursor_pos -= 1
                elif event.key == pygame.K_LEFT:
                    self.url_input_cursor_pos = max(0, self.url_input_cursor_pos - 1)
                elif event.key == pygame.K_RIGHT:
                    self.url_input_cursor_pos = min(len(self.url_input_text), self.url_input_cursor_pos + 1)
                elif event.unicode.isprintable():
                    # Insert character at cursor position
                    self.url_input_text = (self.url_input_text[:self.url_input_cursor_pos] + 
                                         event.unicode + 
                                         self.url_input_text[self.url_input_cursor_pos:])
                    self.url_input_cursor_pos += 1
        return True
        
    def run(self):
        """Main application loop"""
        logger.info("Starting Air Painter application")
        running = True
        
        # Auto-start with first available image (remove selection menu)
        available_images = self.game.get_available_images()
        if available_images:
            self.game.select_image(available_images[0])
            logger.info(f"Auto-selected first image: {available_images[0].name}")
        
        with tracer.start_as_current_span("airpainter_session"):
            while running:
                # Handle hand gestures with new gesture detection
                hand_pos, hand_landmarks, hand_position = self.handle_hand_gestures()
                
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if not self.handle_keyboard_events(event):
                            running = False
                            
                # Handle hand interactions
                if hand_pos and hand_landmarks and hand_position:
                    x, y = hand_pos
                    
                    # Handle interactions based on game state
                    if self.game.get_current_state() == GameState.SELECTING_IMAGE:
                        # For backward compatibility, use old pinch detection
                        pinch_distance = np.sqrt((hand_landmarks[8][0] - hand_landmarks[4][0]) ** 2 + 
                                               (hand_landmarks[8][1] - hand_landmarks[4][1]) ** 2)
                        self.handle_image_selection(x, y, pinch_distance)
                    else:
                        # Tool selection
                        if not self.handle_tool_selection(x, y):
                            # Color selection
                            if not self.handle_color_selection(x, y):
                                # Painting with new gesture detection - Horizontal layout
                                center_x = (self.width - self.ui_sizes['center_panel_width']) // 2
                                center_y = (self.height - self.ui_sizes['center_panel_height']) // 2
                                status_height = 25
                                right_panel_x = self.width - 420  # 400 width + 20 margin
                                
                                # Only paint if not in UI areas
                                if not (y < status_height or  # Status bar area
                                       (center_x <= x <= center_x + self.ui_sizes['center_panel_width'] and 
                                        center_y <= y <= center_y + self.ui_sizes['center_panel_height']) or  # Center panel
                                       x > right_panel_x):  # Right panel
                                    self.handle_painting(x, y, hand_landmarks, hand_position)
                                
                # Save canvas state periodically
                if self.drawing and len(self.drawing_history) == 0:
                    self.save_canvas_state()
                    
                # Render
                self.screen.fill(self.ui_colors['background'])
                self.screen.blit(self.canvas, (0, 0))
                
                # Draw based on game state
                if self.game.get_current_state() == GameState.SELECTING_IMAGE:
                    self.draw_image_selection_screen()
                else:
                    self.draw_painting_screen()
                
                # Draw real-time indicators
                self.draw_hand_trail()
                self.draw_ghost_hand()  # New ghost hand effect
                self.draw_gesture_indicator()
                self.draw_cursor_indicator()
                self.draw_confidence_indicator()
                self.draw_camera_feedback()  # New camera feedback
                
                # Draw UI elements
                self.draw_toolbar()
                if not self.show_circular_menus:
                    self.draw_palette()
                    self.draw_status_bar()
                self.draw_circular_menus()  # New circular menus
                self.draw_help_overlay()
                self.draw_url_input_dialog()
                
                pygame.display.flip()
                self.clock.tick(60)
                
        # Cleanup
        if self.camera_active:
            self.cap.release()
        self.screen_manager.cleanup()
        logger.info("Air Painter application closed")
        sys.exit()

def main():
    """Main entry point"""
    try:
        app = AirPainterUI()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
