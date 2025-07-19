import cv2
import time
import numpy as np
import threading
from typing import Optional, Callable

class ExitingState:
    """Separate exiting state that acts as a bridge between games and app store"""
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.animation_time = 0.0
        self.exiting_progress = 0.0
        self.exiting_start_time = time.time()
        self.exiting_duration = 2.0  # 2 seconds exiting time
        self.shutting_down = False
        
        # Game info
        self.game_name = ""
        self.on_complete_callback = None
        
    def start_exiting(self, game_name: str, on_complete: Callable = None):
        """Start the exiting process for a game"""
        self.game_name = game_name
        self.on_complete_callback = on_complete
        self.exiting_progress = 0.0
        self.exiting_start_time = time.time()
        self.shutting_down = False
        
        print(f"🔄 ExitingState: Starting exit transition for {game_name}")
        
    def update(self, delta_time: float = 0.033):
        """Update exiting state"""
        self.animation_time += delta_time
        
        # Update exiting progress
        elapsed_time = time.time() - self.exiting_start_time
        self.exiting_progress = min(elapsed_time / self.exiting_duration, 1.0)
        
        # Check if exiting is complete
        if self.exiting_progress >= 1.0:
            if self.on_complete_callback:
                self.on_complete_callback()
            return True
        
        return False
        
    def draw(self, frame):
        """Draw the exiting screen"""
        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Exiting title
        title = "RETURNING TO APP STORE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (title_width, title_height), _ = cv2.getTextSize(title, font, 1.8, 3)
        title_x = (self.screen_width - title_width) // 2
        title_y = self.screen_height // 2 - 100
        
        # Title shadow
        cv2.putText(frame, title, (title_x + 3, title_y + 3), font, 1.8, (0, 0, 0), 4)
        # Main title
        cv2.putText(frame, title, (title_x, title_y), font, 1.8, (100, 150, 255), 3)
        
        # Game name
        if self.game_name:
            game_name = f"Exiting: {self.game_name}"
            (game_width, game_height), _ = cv2.getTextSize(game_name, font, 1.2, 2)
            game_x = (self.screen_width - game_width) // 2
            game_y = title_y + 60
            
            cv2.putText(frame, game_name, (game_x, game_y), font, 1.2, (255, 255, 255), 2)
        
        # Free movement message
        remaining_time = max(0, self.exiting_duration - (time.time() - self.exiting_start_time))
        message = f"Free movement period - ready in {remaining_time:.1f}s"
        (message_width, message_height), _ = cv2.getTextSize(message, font, 0.9, 2)
        message_x = (self.screen_width - message_width) // 2
        message_y = game_y + 50
        
        cv2.putText(frame, message, (message_x, message_y), font, 0.9, (200, 200, 200), 2)
        
        # Progress indicator (circular)
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2 + 80
        radius = 60
        
        # Progress circle background
        cv2.circle(frame, (center_x, center_y), radius, (100, 100, 100), 8)
        
        # Progress circle fill
        if self.exiting_progress > 0:
            # Calculate angle based on progress
            angle = int(360 * self.exiting_progress)
            
            # Color changes from blue to green as progress increases
            if self.exiting_progress < 0.5:
                progress_color = (255, 100, 100)  # Blue
            elif self.exiting_progress < 0.8:
                progress_color = (100, 255, 100)  # Light green
            else:
                progress_color = (100, 200, 100)  # Green
            
            # Draw progress arc
            if angle > 0:
                # Draw filled circle for completed portion
                cv2.circle(frame, (center_x, center_y), radius, progress_color, 8)
        
        # Center text
        progress_text = f"{int(self.exiting_progress * 100)}%"
        (progress_width, progress_height), _ = cv2.getTextSize(progress_text, font, 0.8, 2)
        progress_x = center_x - progress_width // 2
        progress_y = center_y + progress_height // 2
        
        cv2.putText(frame, progress_text, (progress_x, progress_y), font, 0.8, (255, 255, 255), 2)
        
        # Animated dots
        dots = "." * (int(self.animation_time * 2) % 4)
        dots_text = f"Preparing{dots}"
        (dots_width, dots_height), _ = cv2.getTextSize(dots_text, font, 0.7, 1)
        dots_x = (self.screen_width - dots_width) // 2
        dots_y = center_y + radius + 60
        
        cv2.putText(frame, dots_text, (dots_x, dots_y), font, 0.7, (150, 150, 150), 1)
        
        # Hand movement indicator
        indicator_text = "Move your hand freely"
        (indicator_width, indicator_height), _ = cv2.getTextSize(indicator_text, font, 0.6, 1)
        indicator_x = (self.screen_width - indicator_width) // 2
        indicator_y = dots_y + 40
        
        cv2.putText(frame, indicator_text, (indicator_x, indicator_y), font, 0.6, (180, 180, 180), 1)
        
    def shutdown(self):
        """Shutdown the exiting state"""
        self.shutting_down = True 