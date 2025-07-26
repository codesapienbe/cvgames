import pygame
import sys
import os
import ctypes
import logging

logger = logging.getLogger(__name__)

class ScreenManager:
    """Utility class for managing screen resolution, DPI scaling, and fullscreen display for Air Painter"""
    
    def __init__(self, config_manager=None):
        self.pygame_init = False
        self.screen = None
        self.display_width = 0
        self.display_height = 0
        self.scale_factor = 1.0
        self.font_scale = 1.0
        self.config_manager = config_manager
        
    def initialize(self, title="Air Painter", fullscreen=True, force_resolution=None):
        """Initialize pygame display with proper DPI scaling and window management - Always maximized"""
        if not self.pygame_init:
            pygame.init()
            pygame.font.init()  # Ensure font module is initialized
            self.pygame_init = True
        
        # Get DPI scaling factor
        self._get_dpi_scaling()
        
        # Always use 1920x1080 resolution
        self.display_width = 1920
        self.display_height = 1080
        logger.info("Using fixed 1920x1080 resolution", extra={"resolution": f"{self.display_width}x{self.display_height}"})
        
        # Always use maximized window (not fullscreen to allow taskbar access)
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.RESIZABLE
        
        self.screen = pygame.display.set_mode((self.display_width, self.display_height), flags)
        pygame.display.set_caption(title)
        
        # Maximize the window
        self._maximize_window()
        
        logger.info("ScreenManager initialized with maximized window", extra={
            "width": self.display_width,
            "height": self.display_height,
            "scale_factor": self.scale_factor,
            "font_scale": self.font_scale,
            "maximized": True
        })
        
        return self.screen
    
    def _get_dpi_scaling(self):
        """Get system DPI scaling factor with adaptive resolution detection"""
        try:
            if sys.platform == "win32":
                # Windows DPI awareness
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per Monitor DPI Aware
                dpi_scale = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
                
                # Get actual screen resolution and scale
                import ctypes.wintypes
                user32 = ctypes.windll.user32
                user32.SetProcessDPIAware()
                
                # Get system metrics for actual resolution
                SM_CXSCREEN = 0
                SM_CYSCREEN = 1
                actual_width = user32.GetSystemMetrics(SM_CXSCREEN)
                actual_height = user32.GetSystemMetrics(SM_CYSCREEN)
                
                # Get logical screen size (what Windows reports as available)
                SM_CXVIRTUALSCREEN = 78
                SM_CYVIRTUALSCREEN = 79
                logical_width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
                logical_height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
                
                # Calculate the actual system scale factor
                if logical_width > 0 and logical_height > 0:
                    # This gives us the actual system scale (e.g., 1.25 for 125%)
                    system_scale = min(actual_width / logical_width, actual_height / logical_height)
                else:
                    # Fallback to DPI scale
                    system_scale = dpi_scale
                
                # Use the system scale directly - this is what Windows is actually using
                self.scale_factor = system_scale
                
                logger.info(f"Windows scaling detected", extra={
                    "dpi_scale": dpi_scale,
                    "system_scale": system_scale,
                    "actual_resolution": f"{actual_width}x{actual_height}",
                    "logical_resolution": f"{logical_width}x{logical_height}",
                    "final_scale": self.scale_factor
                })
                
            elif sys.platform == "linux":
                # Linux DPI scaling (try to get from environment)
                dpi = os.environ.get('GDK_SCALE', '1')
                self.scale_factor = float(dpi)
                
                # Try to get from xrandr if available
                try:
                    import subprocess
                    result = subprocess.run(['xrandr', '--query'], capture_output=True, text=True)
                    if result.returncode == 0:
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if '*' in line:  # Current resolution
                                parts = line.split()
                                if len(parts) >= 4:
                                    # Extract scale from xrandr output
                                    import re
                                    scale_match = re.search(r'(\d+(?:\.\d+)?)x', line)
                                    if scale_match:
                                        detected_scale = float(scale_match.group(1))
                                        self.scale_factor = max(self.scale_factor, detected_scale)
                except:
                    pass
                    
            else:
                # macOS and others
                self.scale_factor = 1.0
                
            # Ensure scale factor is reasonable
            self.scale_factor = max(0.5, min(3.0, self.scale_factor))
            self.font_scale = self.scale_factor
            
            logger.info(f"Final DPI scaling: {self.scale_factor:.2f}x", extra={
                "platform": sys.platform,
                "display_resolution": f"{self.display_width}x{self.display_height}",
                "scale_factor": self.scale_factor
            })
            
        except Exception as e:
            logger.warning("Could not determine DPI scaling", extra={"error": str(e)})
            self.scale_factor = 1.0
            self.font_scale = 1.0
    
    def _maximize_window(self):
        """Maximize the window for better user experience"""
        try:
            if sys.platform == "win32":
                # Windows: Maximize window
                hwnd = pygame.display.get_wm_info()["window"]
                ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3
                ctypes.windll.user32.SetForegroundWindow(hwnd)
            elif sys.platform == "linux":
                # Linux: Use wmctrl for window management
                try:
                    import subprocess
                    hwnd = pygame.display.get_wm_info()["window"]
                    subprocess.run(['wmctrl', '-ir', str(hwnd), '-b', 'add,maximized_vert,maximized_horz'])
                except:
                    pass
        except Exception as e:
            logger.warning("Could not maximize window", extra={"error": str(e)})
    
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
                os.system("wmctrl -r 'Air Painter' -b add,fullscreen")
                os.system("wmctrl -a 'Air Painter'")
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
    
    def get_optimal_canvas_size(self, toolbar_width=120, palette_height=80, margin_ratio=0.05):
        """Calculate optimal canvas size for drawing area"""
        available_width = self.display_width - toolbar_width - (self.display_width * margin_ratio)
        available_height = self.display_height - palette_height - (self.display_height * margin_ratio)
        return int(available_width), int(available_height)
    
    def get_optimal_ui_sizes(self):
        """Calculate optimal UI element sizes based on screen resolution and DPI scaling"""
        # Base scale from screen resolution
        base_width = 1920  # Reference width
        resolution_scale = min(self.display_width / base_width, 1.5)  # Cap scaling at 1.5x
        
        # Combine with system scale factor (this accounts for Windows 125% scaling)
        total_scale = resolution_scale * self.scale_factor
        
        # Ensure reasonable bounds - be more conservative with high DPI displays
        total_scale = max(0.3, min(2.0, total_scale))
        
        # Calculate panel dimensions based on screen height to prevent overflow
        # Use more conservative limits for high DPI displays
        max_panel_height = int(self.display_height * 0.6)  # Max 60% of screen height
        max_panel_width = int(self.display_width * 0.7)   # Max 70% of screen width
        
        return {
            'toolbar_width': min(int(200 * total_scale), max_panel_width),
            'palette_height': min(int(120 * total_scale), int(max_panel_height * 0.2)),
            'color_swatch_size': int(50 * total_scale),
            'tool_button_height': int(60 * total_scale),
            'padding': int(20 * total_scale),
            'margin': int(10 * total_scale),
            'border_radius': int(10 * total_scale),
            'shadow_offset': int(3 * total_scale),
            'center_panel_width': min(int(400 * total_scale), max_panel_width),
            'center_panel_height': min(int(500 * total_scale), max_panel_height)
        }
    
    def get_optimal_brush_sizes(self):
        """Calculate optimal brush sizes based on screen resolution and DPI scaling"""
        # Base scale from screen resolution
        base_width = 1920  # Reference width
        resolution_scale = min(self.display_width / base_width, 1.5)  # Cap scaling at 1.5x
        
        # Combine with system scale factor (this accounts for Windows 125% scaling)
        total_scale = resolution_scale * self.scale_factor
        
        # Ensure reasonable bounds - be more conservative with high DPI displays
        total_scale = max(0.3, min(2.0, total_scale))
        
        return [int(size * total_scale) for size in [5, 10, 15, 20, 25, 30]]
    
    def get_optimal_font_sizes(self):
        """Calculate optimal font sizes based on screen resolution and DPI scaling"""
        # Base scale from screen resolution
        base_width = 1920  # Reference width
        resolution_scale = min(self.display_width / base_width, 1.5)  # Cap scaling at 1.5x
        
        # Combine with system scale factor (this accounts for Windows 125% scaling)
        total_scale = resolution_scale * self.scale_factor
        
        # Ensure reasonable bounds - be more conservative with high DPI displays
        total_scale = max(0.3, min(2.0, total_scale))
        
        return {
            'title_font': int(28 * total_scale),
            'tool_font': int(16 * total_scale),
            'info_font': int(14 * total_scale),
            'small_font': int(12 * total_scale)
        }
    
    def cleanup(self):
        """Cleanup pygame resources"""
        if self.pygame_init:
            pygame.quit()
            self.pygame_init = False 