import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import sys
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "component": "airpainter", "message": "%(message)s"}',
    handlers=[logging.FileHandler("application.log")]
)
logger = logging.getLogger(__name__)

class Tool(Enum):
    BRUSH = "brush"
    ERASER = "eraser"
    CLEAR = "clear"
    UNDO = "undo"

@dataclass
class Color:
    name: str
    rgb: Tuple[int, int, int]
    hex_code: str

class AirPainterUI:
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.setup_pygame()
        self.setup_colors()
        self.setup_ui_constants()
        self.setup_state()
        self.setup_camera()
        
    def setup_pygame(self):
        """Initialize Pygame with modern settings"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Air Painter - Modern CV Drawing App")
        self.canvas = pygame.Surface((self.width, self.height))
        self.canvas.fill((255, 255, 255))
        
        # Modern fonts
        self.title_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.tool_font = pygame.font.SysFont("Arial", 16, bold=True)
        self.info_font = pygame.font.SysFont("Arial", 14)
        
        self.clock = pygame.time.Clock()
        
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
        
    def setup_ui_constants(self):
        """Define UI layout constants"""
        self.TOOLBAR_WIDTH = 120
        self.PALETTE_HEIGHT = 80
        self.COLOR_SWATCH_SIZE = 45
        self.TOOL_BUTTON_HEIGHT = 50
        self.PADDING = 15
        self.MARGIN = 8
        self.BORDER_RADIUS = 8
        self.SHADOW_OFFSET = 2
        
    def setup_state(self):
        """Initialize application state"""
        self.selected_color = self.colors[3]  # Red
        self.selected_tool = Tool.BRUSH
        self.drawing_history = []
        self.brush_sizes = [5, 10, 15, 20]
        self.current_brush_size = 1  # Index into brush_sizes
        self.last_pos = None
        self.drawing = False
        self.show_help = False
        self.camera_active = True
        
        # Real-time indicators
        self.cursor_pos = None
        self.hand_trail = []
        self.gesture_active = False
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.1  # Seconds
        self.hand_confidence = 0.0
        self.pinch_strength = 0.0
        
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
                # Try different import approaches for MediaPipe
                if hasattr(mp.solutions, 'hands'):
                    mp_hands = mp.solutions.hands
                else:
                    # Alternative import method
                    mp_hands = getattr(mp.solutions, 'hands')
                    
                self.hands = mp_hands.Hands(
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                    max_num_hands=1
                )
                logger.info("Camera and MediaPipe initialized successfully")
            except Exception as mp_error:
                logger.warning(f"MediaPipe hands import issue: {mp_error}")
                # Fallback: create a dummy hands object
                self.hands = None
                self.camera_active = False
            
        except Exception as e:
            logger.error(f"Camera setup failed: {e}")
            self.camera_active = False
            
    def draw_rounded_rect(self, surface, color, rect, radius):
        """Draw rounded rectangle with shadow effect"""
        x, y, w, h = rect
        
        # Shadow
        shadow_rect = (x + self.SHADOW_OFFSET, y + self.SHADOW_OFFSET, w, h)
        pygame.draw.rect(surface, (0, 0, 0, 30), shadow_rect, border_radius=radius)
        
        # Main rectangle
        pygame.draw.rect(surface, color, rect, border_radius=radius)
        
    def draw_cursor_indicator(self):
        """Draw real-time cursor indicator"""
        if not self.cursor_pos:
            return
            
        x, y = self.cursor_pos
        
        # Don't draw cursor in UI areas
        if x < self.TOOLBAR_WIDTH or y > self.height - self.PALETTE_HEIGHT:
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
            
    def draw_gesture_indicator(self):
        """Draw gesture feedback indicator"""
        if not self.gesture_active or not self.cursor_pos:
            return
            
        x, y = self.cursor_pos
        current_time = time.time()
        
        # Animated gesture indicator
        pulse_size = int(20 + 10 * np.sin(current_time * 10))
        
        # Draw pulsing circle
        pygame.draw.circle(self.screen, self.ui_colors['gesture_indicator'], (x, y), pulse_size, 3)
        
        # Draw pinch strength indicator
        if self.pinch_strength > 0:
            strength_radius = int(30 * self.pinch_strength)
            pygame.draw.circle(self.screen, self.ui_colors['gesture_indicator'], (x, y), strength_radius, 2)
            
    def draw_confidence_indicator(self):
        """Draw hand detection confidence indicator"""
        if not self.camera_active:
            return
            
        # Position in top-right corner
        indicator_x = self.width - 150
        indicator_y = 20
        
        # Background
        bg_rect = (indicator_x, indicator_y, 130, 60)
        self.draw_rounded_rect(self.screen, self.ui_colors['toolbar_bg'], bg_rect, self.BORDER_RADIUS)
        
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
        """Draw modern toolbar with visual feedback"""
        # Toolbar background with shadow
        toolbar_rect = (0, 0, self.TOOLBAR_WIDTH, self.height)
        self.draw_rounded_rect(self.screen, self.ui_colors['toolbar_bg'], toolbar_rect, self.BORDER_RADIUS)
        
        # Title
        title_text = self.title_font.render("TOOLS", True, self.ui_colors['text_primary'])
        self.screen.blit(title_text, (self.PADDING, self.PADDING))
        
        # Tool buttons
        tools = [Tool.BRUSH, Tool.ERASER, Tool.CLEAR, Tool.UNDO]
        y_offset = 80
        
        for i, tool in enumerate(tools):
            button_rect = (self.PADDING, y_offset + i * (self.TOOL_BUTTON_HEIGHT + self.MARGIN), 
                          self.TOOLBAR_WIDTH - 2 * self.PADDING, self.TOOL_BUTTON_HEIGHT)
            
            # Button background
            bg_color = self.ui_colors['selected_bg'] if tool == self.selected_tool else self.ui_colors['toolbar_bg']
            self.draw_rounded_rect(self.screen, bg_color, button_rect, self.BORDER_RADIUS)
            
            # Button text
            text_color = self.ui_colors['selected_text'] if tool == self.selected_tool else self.ui_colors['text_primary']
            tool_text = self.tool_font.render(tool.value.upper(), True, text_color)
            text_rect = tool_text.get_rect(center=(
                button_rect[0] + button_rect[2] // 2,
                button_rect[1] + button_rect[3] // 2
            ))
            self.screen.blit(tool_text, text_rect)
            
            # Brush size indicator for brush tool
            if tool == Tool.BRUSH and tool == self.selected_tool:
                size_text = self.info_font.render(f"Size: {self.brush_sizes[self.current_brush_size]}", True, text_color)
                self.screen.blit(size_text, (self.PADDING, button_rect[1] + button_rect[3] + 5))
                
    def draw_palette(self):
        """Draw modern color palette"""
        palette_rect = (0, self.height - self.PALETTE_HEIGHT, self.width, self.PALETTE_HEIGHT)
        self.draw_rounded_rect(self.screen, self.ui_colors['palette_bg'], palette_rect, self.BORDER_RADIUS)
        
        # Title
        title_text = self.title_font.render("COLORS", True, self.ui_colors['text_primary'])
        self.screen.blit(title_text, (self.PADDING, self.height - self.PALETTE_HEIGHT + self.PADDING))
        
        # Color swatches
        x_offset = 120
        for i, color in enumerate(self.colors):
            swatch_rect = (x_offset + i * (self.COLOR_SWATCH_SIZE + self.MARGIN), 
                          self.height - self.PALETTE_HEIGHT + self.PADDING + 30,
                          self.COLOR_SWATCH_SIZE, self.COLOR_SWATCH_SIZE)
            
            # Color swatch with border
            border_color = self.ui_colors['selected_bg'] if color == self.selected_color else self.ui_colors['palette_border']
            pygame.draw.rect(self.screen, border_color, swatch_rect, border_radius=self.BORDER_RADIUS)
            
            # Inner color
            inner_rect = (swatch_rect[0] + 2, swatch_rect[1] + 2, 
                         swatch_rect[2] - 4, swatch_rect[3] - 4)
            pygame.draw.rect(self.screen, color.rgb, inner_rect, border_radius=self.BORDER_RADIUS - 2)
            
            # Color name
            name_text = self.info_font.render(color.name, True, self.ui_colors['text_secondary'])
            name_rect = name_text.get_rect(center=(swatch_rect[0] + swatch_rect[2] // 2, 
                                                  swatch_rect[1] + swatch_rect[3] + 15))
            self.screen.blit(name_text, name_rect)
            
    def draw_status_bar(self):
        """Draw status bar with current tool and color info"""
        status_height = 30
        status_rect = (self.TOOLBAR_WIDTH, self.height - status_height, 
                      self.width - self.TOOLBAR_WIDTH, status_height)
        pygame.draw.rect(self.screen, self.ui_colors['background'], status_rect)
        
        # Status text
        status_text = f"Tool: {self.selected_tool.value.title()} | Color: {self.selected_color.name}"
        if self.selected_tool == Tool.BRUSH:
            status_text += f" | Size: {self.brush_sizes[self.current_brush_size]}"
            
        text_surface = self.info_font.render(status_text, True, self.ui_colors['text_secondary'])
        self.screen.blit(text_surface, (self.TOOLBAR_WIDTH + self.PADDING, self.height - status_height + 8))
        
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
        self.draw_rounded_rect(self.screen, self.ui_colors['toolbar_bg'], help_rect, self.BORDER_RADIUS)
        
        # Help content
        help_lines = [
            "AIR PAINTER CONTROLS",
            "",
            "👆 Point with index finger",
            "🤏 Pinch to draw/erase",
            "🖱️ Select tools from left panel",
            "🎨 Choose colors from bottom panel",
            "⌨️ Press H for help",
            "⌨️ Press ESC to exit",
            "⌨️ +/- to adjust brush size",
            "",
            "Press H again to close"
        ]
        
        y_offset = help_rect[1] + self.PADDING
        for line in help_lines:
            if line.startswith("👆") or line.startswith("🤏") or line.startswith("🖱️") or line.startswith("🎨") or line.startswith("⌨️"):
                text_surface = self.tool_font.render(line, True, self.ui_colors['text_primary'])
            elif line == "AIR PAINTER CONTROLS":
                text_surface = self.title_font.render(line, True, self.ui_colors['selected_bg'])
            else:
                text_surface = self.info_font.render(line, True, self.ui_colors['text_secondary'])
                
            text_rect = text_surface.get_rect(center=(help_rect[0] + help_rect[2] // 2, y_offset))
            self.screen.blit(text_surface, text_rect)
            y_offset += 25
            
    def handle_hand_gestures(self):
        """Process hand gestures for drawing and UI interaction"""
        if not self.camera_active:
            return None, 0.0
            
        ret, frame = self.cap.read()
        if not ret:
            return None, 0.0
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.hands is None:
            return None, 0.0
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            self.hand_confidence = 0.0
            self.cursor_pos = None
            return None, 0.0
            
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
            
        # Calculate pinch distance and strength
        distance = np.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)
        self.pinch_strength = max(0, 1 - distance / 0.1)  # Normalize to 0-1
        
        # Update confidence (simplified)
        self.hand_confidence = 0.9  # MediaPipe doesn't provide direct confidence
        
        return (x, y), distance
        
    def handle_tool_selection(self, x: int, y: int):
        """Handle tool selection from toolbar"""
        if x < self.TOOLBAR_WIDTH:
            tools = [Tool.BRUSH, Tool.ERASER, Tool.CLEAR, Tool.UNDO]
            y_offset = 80
            
            for i, tool in enumerate(tools):
                button_rect = (self.PADDING, y_offset + i * (self.TOOL_BUTTON_HEIGHT + self.MARGIN), 
                              self.TOOLBAR_WIDTH - 2 * self.PADDING, self.TOOL_BUTTON_HEIGHT)
                
                if (button_rect[0] <= x <= button_rect[0] + button_rect[2] and 
                    button_rect[1] <= y <= button_rect[1] + button_rect[3]):
                    
                    if tool == Tool.CLEAR:
                        self.canvas.fill((255, 255, 255))
                        self.drawing_history.clear()
                        logger.info("Canvas cleared")
                    elif tool == Tool.UNDO and self.drawing_history:
                        # Simple undo - restore previous canvas state
                        if len(self.drawing_history) > 1:
                            self.drawing_history.pop()
                            self.canvas = self.drawing_history[-1].copy()
                        logger.info("Undo performed")
                    else:
                        self.selected_tool = tool
                        logger.info(f"Tool selected: {tool.value}")
                    return True
        return False
        
    def handle_color_selection(self, x: int, y: int):
        """Handle color selection from palette"""
        if y > self.height - self.PALETTE_HEIGHT:
            x_offset = 120
            swatch_y = self.height - self.PALETTE_HEIGHT + self.PADDING + 30
            
            for i, color in enumerate(self.colors):
                swatch_rect = (x_offset + i * (self.COLOR_SWATCH_SIZE + self.MARGIN), 
                              swatch_y, self.COLOR_SWATCH_SIZE, self.COLOR_SWATCH_SIZE)
                
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
        
    def handle_keyboard_events(self, event):
        """Handle keyboard shortcuts"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                self.show_help = not self.show_help
                logger.info(f"Help overlay {'shown' if self.show_help else 'hidden'}")
            elif event.key == pygame.K_ESCAPE:
                return False  # Exit
            elif event.key == pygame.K_s and event.mod & pygame.KMOD_CTRL:
                # Save canvas (placeholder)
                logger.info("Save canvas requested")
            elif event.key == pygame.K_z and event.mod & pygame.KMOD_CTRL:
                # Undo
                if self.drawing_history:
                    self.drawing_history.pop()
                    if self.drawing_history:
                        self.canvas = self.drawing_history[-1].copy()
                    else:
                        self.canvas.fill((255, 255, 255))
                    logger.info("Undo performed via keyboard")
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                # Increase brush size
                self.current_brush_size = min(len(self.brush_sizes) - 1, self.current_brush_size + 1)
                logger.info(f"Brush size increased to {self.brush_sizes[self.current_brush_size]}")
            elif event.key == pygame.K_MINUS:
                # Decrease brush size
                self.current_brush_size = max(0, self.current_brush_size - 1)
                logger.info(f"Brush size decreased to {self.brush_sizes[self.current_brush_size]}")
        return True
        
    def run(self):
        """Main application loop"""
        logger.info("Starting Air Painter application")
        running = True
        
        while running:
            # Handle hand gestures
            hand_pos, pinch_distance = self.handle_hand_gestures()
            
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if not self.handle_keyboard_events(event):
                        running = False
                        
            # Handle hand interactions
            if hand_pos:
                x, y = hand_pos
                
                # Tool selection
                if not self.handle_tool_selection(x, y):
                    # Color selection
                    if not self.handle_color_selection(x, y):
                        # Drawing (only if not in UI areas)
                        if (x > self.TOOLBAR_WIDTH and y < self.height - self.PALETTE_HEIGHT):
                            self.draw_on_canvas(x, y, pinch_distance)
                            
            # Save canvas state periodically
            if self.drawing and len(self.drawing_history) == 0:
                self.save_canvas_state()
                
            # Render
            self.screen.fill(self.ui_colors['background'])
            self.screen.blit(self.canvas, (0, 0))
            
            # Draw real-time indicators
            self.draw_hand_trail()
            self.draw_gesture_indicator()
            self.draw_cursor_indicator()
            self.draw_confidence_indicator()
            
            # Draw UI elements
            self.draw_toolbar()
            self.draw_palette()
            self.draw_status_bar()
            self.draw_help_overlay()
            
            pygame.display.flip()
            self.clock.tick(60)
            
        # Cleanup
        if self.camera_active:
            self.cap.release()
        pygame.quit()
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
