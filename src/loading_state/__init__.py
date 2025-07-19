import cv2
import time
import numpy as np
import threading
import subprocess
import sys
from pathlib import Path
from typing import Optional, Callable
import os

class LoadingState:
    """Separate loading state that acts as a bridge between app store and games"""
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.animation_time = 0.0
        self.loading_progress = 0.0
        self.loading_start_time = time.time()
        self.loading_duration = 3.0  # 3 seconds loading time
        self.game_process = None
        self.game_thread = None
        self.game_completed = False
        self.game_lock = threading.Lock()
        self.shutting_down = False
        
        # Game info
        self.game_name = ""
        self.game_path = ""
        self.on_complete_callback = None
        
    def start_loading(self, game_name: str, game_path: str, on_complete: Callable = None):
        """Start the loading process for a game"""
        self.game_name = game_name
        self.game_path = game_path
        self.on_complete_callback = on_complete
        self.loading_progress = 0.0
        self.loading_start_time = time.time()
        self.game_completed = False
        self.shutting_down = False
        
        print(f"🎮 LoadingState: Starting loading for {game_name}")
        
        # Start game thread
        self.game_thread = threading.Thread(target=self._launch_game_thread, daemon=True)
        self.game_thread.start()
        
    def _launch_game_thread(self):
        """Launch the game in a separate thread"""
        try:
            # Wait a bit to simulate loading
            time.sleep(1.0)
            
            # Change to the game directory
            game_dir = Path(self.game_path)
            init_file = game_dir / '__init__.py'
            
            if init_file.exists():
                print(f"🎮 LoadingState: Launching {self.game_name}...")
                
                # Use subprocess to run the game
                self.game_process = subprocess.Popen([sys.executable, str(init_file)], 
                                                   cwd=str(game_dir))
                
                # Wait for the game to finish or for shutdown signal
                while self.game_process.poll() is None and not self.shutting_down:
                    time.sleep(0.1)  # Check every 100ms
                
                # If we're shutting down, terminate the game
                if self.shutting_down and self.game_process.poll() is None:
                    print(f"🔄 LoadingState: Terminating {self.game_name} due to shutdown...")
                    try:
                        self.game_process.terminate()
                        self.game_process.wait(timeout=2)
                    except:
                        self.game_process.kill()
                else:
                    print(f"✅ LoadingState: {self.game_name} finished")
                
                # Signal game completion
                with self.game_lock:
                    self.game_completed = True
                
            else:
                print(f"❌ LoadingState: Game file not found: {init_file}")
                with self.game_lock:
                    self.game_completed = True
                
        except Exception as e:
            print(f"❌ LoadingState: Error running {self.game_name}: {e}")
            with self.game_lock:
                self.game_completed = True
        finally:
            # Return to the original directory
            os.chdir(Path(__file__).parent.parent)
            
    def update(self, delta_time: float = 0.033):
        """Update loading state"""
        self.animation_time += delta_time
        
        # Update loading progress
        elapsed_time = time.time() - self.loading_start_time
        self.loading_progress = min(elapsed_time / self.loading_duration, 1.0)
        
        # Check if loading is complete
        if self.loading_progress >= 1.0:
            # Check if game thread is running
            if self.game_thread and self.game_thread.is_alive():
                # Game is running, we can transition
                if self.on_complete_callback:
                    self.on_complete_callback()
                return True
            else:
                # Thread finished but we haven't detected completion yet
                with self.game_lock:
                    if not self.game_completed:
                        self.game_completed = True
                
                # If game completed during loading, call callback
                if self.on_complete_callback:
                    self.on_complete_callback()
                return True
        
        return False
        
    def draw(self, frame):
        """Draw the loading screen"""
        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Loading title
        title = "GAME LOADING"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (title_width, title_height), _ = cv2.getTextSize(title, font, 2.0, 3)
        title_x = (self.screen_width - title_width) // 2
        title_y = self.screen_height // 2 - 100
        
        # Title shadow
        cv2.putText(frame, title, (title_x + 3, title_y + 3), font, 2.0, (0, 0, 0), 4)
        # Main title
        cv2.putText(frame, title, (title_x, title_y), font, 2.0, (255, 165, 0), 3)
        
        # Game name
        if self.game_name:
            game_name = self.game_name
            (game_width, game_height), _ = cv2.getTextSize(game_name, font, 1.5, 2)
            game_x = (self.screen_width - game_width) // 2
            game_y = title_y + 80
            
            cv2.putText(frame, game_name, (game_x, game_y), font, 1.5, (255, 255, 255), 2)
        
        # Loading progress bar
        bar_width = 600
        bar_height = 30
        bar_x = (self.screen_width - bar_width) // 2
        bar_y = self.screen_height // 2 + 50
        
        # Progress bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Progress bar fill
        fill_width = int(bar_width * self.loading_progress)
        if fill_width > 0:
            # Color changes from orange to green as progress increases
            if self.loading_progress < 0.5:
                progress_color = (0, 165, 255)  # Orange
            elif self.loading_progress < 0.8:
                progress_color = (0, 140, 255)  # Dark orange
            else:
                progress_color = (0, 255, 0)  # Green
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), progress_color, -1)
        
        # Progress percentage
        progress_text = f"{int(self.loading_progress * 100)}%"
        (progress_width, progress_height), _ = cv2.getTextSize(progress_text, font, 1.0, 2)
        progress_x = (self.screen_width - progress_width) // 2
        progress_y = bar_y + bar_height + 50
        
        cv2.putText(frame, progress_text, (progress_x, progress_y), font, 1.0, (255, 255, 255), 2)
        
        # Loading message
        if self.loading_progress < 0.3:
            message = "Initializing game..."
        elif self.loading_progress < 0.6:
            message = "Loading game resources..."
        elif self.loading_progress < 0.9:
            message = "Starting game engine..."
        else:
            message = "Almost ready..."
        
        (message_width, message_height), _ = cv2.getTextSize(message, font, 0.8, 1)
        message_x = (self.screen_width - message_width) // 2
        message_y = progress_y + 50
        
        cv2.putText(frame, message, (message_x, message_y), font, 0.8, (200, 200, 200), 1)
        
        # Timeout warning (show after 7 seconds)
        elapsed_time = time.time() - self.loading_start_time
        if elapsed_time > 7.0:
            timeout_warning = f"Loading taking longer than expected... ({elapsed_time:.1f}s)"
            (warning_width, warning_height), _ = cv2.getTextSize(timeout_warning, font, 0.6, 1)
            warning_x = (self.screen_width - warning_width) // 2
            warning_y = message_y + 40
            
            # Warning background
            cv2.rectangle(frame, (warning_x - 10, warning_y - 15), (warning_x + warning_width + 10, warning_y + 5), 
                         (50, 50, 50), -1)
            cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)
            
            # Warning text (yellow for timeout warning)
            cv2.putText(frame, timeout_warning, (warning_x, warning_y), font, 0.6, (0, 255, 255), 1)
        
        # Animated loading dots
        dots = "." * (int(self.animation_time * 2) % 4)
        dots_text = f"Loading{dots}"
        (dots_width, dots_height), _ = cv2.getTextSize(dots_text, font, 0.6, 1)
        dots_x = (self.screen_width - dots_width) // 2
        dots_y = message_y + 40
        
        cv2.putText(frame, dots_text, (dots_x, dots_y), font, 0.6, (150, 150, 150), 1)
        
    def shutdown(self):
        """Shutdown the loading state"""
        self.shutting_down = True
        if self.game_process and self.game_process.poll() is None:
            try:
                self.game_process.terminate()
                self.game_process.wait(timeout=2)
            except:
                self.game_process.kill() 