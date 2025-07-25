import pygame
import sys
import os
import ctypes
import logging

logger = logging.getLogger(__name__)

class ScreenManager:
    """Utility class for managing screen resolution, DPI scaling, and fullscreen display"""
    
    def __init__(self, config_manager=None):
        self.pygame_init = False
        self.screen = None
        self.display_width = 0
        self.display_height = 0
        self.scale_factor = 1.0
        self.font_scale = 1.0
        self.config_manager = config_manager
        
    def initialize(self, title="Game Window", fullscreen=True, force_resolution=None):
        """Initialize pygame display with proper DPI scaling and window management"""
        if not self.pygame_init:
            pygame.init()
            pygame.font.init()  # Ensure font module is initialized
            self.pygame_init = True
        
        # Get DPI scaling factor
        self._get_dpi_scaling()
        
        # Determine resolution
        if force_resolution:
            self.display_width, self.display_height = force_resolution
            logger.info("Using forced resolution", extra={"resolution": f"{self.display_width}x{self.display_height}"})
        elif self.config_manager:
            self.display_width, self.display_height = self.config_manager.get_resolution()
            logger.info("Using configured resolution", extra={"resolution": f"{self.display_width}x{self.display_height}"})
        else:
            # Use current screen resolution
            info = pygame.display.Info()
            self.display_width = info.current_w
            self.display_height = info.current_h
            logger.info("Using current screen resolution", extra={"resolution": f"{self.display_width}x{self.display_height}"})
        
        # Set display mode with proper flags
        if fullscreen:
            flags = pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
        else:
            flags = pygame.DOUBLEBUF | pygame.HWSURFACE
            
        self.screen = pygame.display.set_mode((self.display_width, self.display_height), flags)
        pygame.display.set_caption(title)
        
        # Set window properties for better fullscreen experience
        if fullscreen:
            self._set_window_properties()
        
        logger.info("ScreenManager initialized", extra={
            "width": self.display_width,
            "height": self.display_height,
            "scale_factor": self.scale_factor,
            "font_scale": self.font_scale,
            "fullscreen": fullscreen
        })
        
        return self.screen
    
    def _get_dpi_scaling(self):
        """Get system DPI scaling factor"""
        try:
            if sys.platform == "win32":
                # Windows DPI awareness
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per Monitor DPI Aware
                self.scale_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
            elif sys.platform == "linux":
                # Linux DPI scaling (try to get from environment)
                dpi = os.environ.get('GDK_SCALE', '1')
                self.scale_factor = float(dpi)
            else:
                # macOS and others
                self.scale_factor = 1.0
                
            # Ensure scale factor is reasonable
            self.scale_factor = max(0.5, min(3.0, self.scale_factor))
            self.font_scale = self.scale_factor
            
        except Exception as e:
            logger.warning("Could not determine DPI scaling", extra={"error": str(e)})
            self.scale_factor = 1.0
            self.font_scale = 1.0
    
    def _set_window_properties(self):
        """Set window properties for better fullscreen experience"""
        try:
            if sys.platform == "win32":
                # Windows: Set window to topmost and fullscreen
                hwnd = pygame.display.get_wm_info()["window"]
                ctypes.windll.user32.SetWindowPos(
                    hwnd, 0, 0, 0, 0, 0,
                    0x0001 | 0x0002 | 0x0004 | 0x0040  # SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_SHOWWINDOW
                )
                ctypes.windll.user32.SetForegroundWindow(hwnd)
            elif sys.platform == "linux":
                # Linux: Use wmctrl for window management
                os.system("wmctrl -r 'Game Window' -b add,fullscreen")
                os.system("wmctrl -a 'Game Window'")
        except Exception as e:
            logger.warning("Could not set window properties", extra={"error": str(e)})
    
    def get_scaled_font(self, base_size, font_name="Arial"):
        """Get font with proper DPI scaling"""
        scaled_size = int(base_size * self.font_scale)
        return pygame.font.SysFont(font_name, scaled_size)
    
    def get_scaled_dimensions(self, base_width, base_height):
        """Get scaled dimensions based on DPI and screen size"""
        return (
            int(base_width * self.scale_factor),
            int(base_height * self.scale_factor)
        )
    
    def center_rect(self, rect_width, rect_height):
        """Get centered position for a rectangle"""
        return (
            (self.display_width - rect_width) // 2,
            (self.display_height - rect_height) // 2
        )
    
    def get_optimal_tile_size(self, grid_size, margin_ratio=0.1):
        """Calculate optimal tile size for a grid-based game"""
        available_width = self.display_width * (1 - margin_ratio)
        available_height = self.display_height * (1 - margin_ratio)
        return min(available_width // grid_size, available_height // grid_size)
    
    def cleanup(self):
        """Cleanup pygame resources"""
        if self.pygame_init:
            pygame.quit()
            self.pygame_init = False 