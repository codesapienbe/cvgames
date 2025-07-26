import sqlite3
import os
import logging
import pygame
import sys

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages Air Painter configuration including screen resolution preferences"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            # Use user-specific settings directory
            home_dir = os.path.expanduser("~")
            settings_dir = os.path.join(home_dir, ".cvgames", "airpainter")
            os.makedirs(settings_dir, exist_ok=True)
            self.db_path = os.path.join(settings_dir, "airpainter_settings.sqlite")
        else:
            self.db_path = db_path
            
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create configuration table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS airpainter_config (
                    id INTEGER PRIMARY KEY,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create resolution history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS resolution_history (
                    id INTEGER PRIMARY KEY,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    used_count INTEGER DEFAULT 1,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create brush settings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS brush_settings (
                    id INTEGER PRIMARY KEY,
                    brush_size INTEGER NOT NULL,
                    used_count INTEGER DEFAULT 1,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Air Painter configuration database initialized", extra={"db_path": self.db_path})
            
        except Exception as e:
            logger.error("Failed to initialize database", extra={"error": str(e)})
            raise
    
    def get_available_resolutions(self):
        """Get list of available screen resolutions - Fixed to 1920x1080"""
        # Fixed resolution for consistent experience
        logger.info("Using fixed resolution: 1920x1080")
        return [(1920, 1080)]
    
    def get_stored_resolution(self):
        """Get the stored resolution preference"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT value FROM airpainter_config WHERE key = "resolution"')
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                width, height = map(int, result[0].split('x'))
                logger.info("Retrieved stored resolution", extra={"resolution": f"{width}x{height}"})
                return width, height
            else:
                logger.info("No stored resolution found, will prompt user")
                return None
                
        except Exception as e:
            logger.error("Failed to get stored resolution", extra={"error": str(e)})
            return None
    
    def save_resolution(self, width, height):
        """Save the selected resolution to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save to airpainter_config
            cursor.execute('''
                INSERT OR REPLACE INTO airpainter_config (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', ('resolution', f"{width}x{height}"))
            
            # Update resolution history
            cursor.execute('''
                INSERT OR REPLACE INTO resolution_history (width, height, used_count, last_used)
                VALUES (?, ?, 
                    COALESCE((SELECT used_count + 1 FROM resolution_history 
                             WHERE width = ? AND height = ?), 1),
                    CURRENT_TIMESTAMP)
            ''', (width, height, width, height))
            
            conn.commit()
            conn.close()
            
            logger.info("Resolution saved to database", extra={"resolution": f"{width}x{height}"})
            
        except Exception as e:
            logger.error("Failed to save resolution", extra={"error": str(e)})
    
    def get_most_used_resolution(self):
        """Get the most frequently used resolution"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT width, height FROM resolution_history 
                ORDER BY used_count DESC, last_used DESC 
                LIMIT 1
            ''')
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                width, height = result
                logger.info("Retrieved most used resolution", extra={"resolution": f"{width}x{height}"})
                return width, height
            else:
                return None
                
        except Exception as e:
            logger.error("Failed to get most used resolution", extra={"error": str(e)})
            return None
    
    def get_default_brush_size(self):
        """Get the default brush size preference"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT value FROM airpainter_config WHERE key = "default_brush_size"')
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                brush_size = int(result[0])
                logger.info("Retrieved default brush size", extra={"brush_size": brush_size})
                return brush_size
            else:
                return 10  # Default brush size
                
        except Exception as e:
            logger.error("Failed to get default brush size", extra={"error": str(e)})
            return 10
    
    def save_brush_size(self, brush_size):
        """Save the brush size preference"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO airpainter_config (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', ('default_brush_size', str(brush_size)))
            
            # Update brush settings history
            cursor.execute('''
                INSERT OR REPLACE INTO brush_settings (brush_size, used_count, last_used)
                VALUES (?, 
                    COALESCE((SELECT used_count + 1 FROM brush_settings 
                             WHERE brush_size = ?), 1),
                    CURRENT_TIMESTAMP)
            ''', (brush_size, brush_size))
            
            conn.commit()
            conn.close()
            
            logger.info("Brush size saved to database", extra={"brush_size": brush_size})
            
        except Exception as e:
            logger.error("Failed to save brush size", extra={"error": str(e)})
    
    def show_resolution_selection(self, available_resolutions):
        """Show resolution selection dialog"""
        try:
            # Initialize pygame for the dialog
            pygame.init()
            
            # Create a temporary window for the dialog
            dialog_width = 600
            dialog_height = 400
            dialog_screen = pygame.display.set_mode((dialog_width, dialog_height))
            pygame.display.set_caption("Air Painter - Select Resolution")
            
            font = pygame.font.SysFont("Arial", 24)
            small_font = pygame.font.SysFont("Arial", 18)
            
            running = True
            selected_index = 0
            
            while running:
                dialog_screen.fill((50, 50, 50))
                
                # Title
                title = font.render("Select Air Painter Resolution", True, (255, 255, 255))
                dialog_screen.blit(title, (dialog_width // 2 - title.get_width() // 2, 20))
                
                # Instructions
                instructions = small_font.render("Use UP/DOWN arrows to select, ENTER to confirm", True, (200, 200, 200))
                dialog_screen.blit(instructions, (dialog_width // 2 - instructions.get_width() // 2, 50))
                
                # Resolution options
                y_offset = 100
                for i, (width, height) in enumerate(available_resolutions):
                    color = (255, 255, 0) if i == selected_index else (255, 255, 255)
                    text = f"{width} x {height}"
                    if i == 0:
                        text += " (Current Screen)"
                    
                    resolution_text = font.render(text, True, color)
                    dialog_screen.blit(resolution_text, (dialog_width // 2 - resolution_text.get_width() // 2, y_offset))
                    y_offset += 40
                
                pygame.display.flip()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return None
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            selected_index = max(0, selected_index - 1)
                        elif event.key == pygame.K_DOWN:
                            selected_index = min(len(available_resolutions) - 1, selected_index + 1)
                        elif event.key == pygame.K_RETURN:
                            selected_resolution = available_resolutions[selected_index]
                            pygame.quit()
                            return selected_resolution
                        elif event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return None
            
            pygame.quit()
            return None
            
        except Exception as e:
            logger.error("Failed to show resolution selection", extra={"error": str(e)})
            return None
    
    def get_resolution(self):
        """Get resolution - Fixed to 1920x1080"""
        # Fixed resolution for consistent experience
        logger.info("Using fixed resolution: 1920x1080")
        return (1920, 1080) 