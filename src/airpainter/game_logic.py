import os
import pygame
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class GameState(Enum):
    SELECTING_IMAGE = "selecting_image"
    PAINTING = "painting"
    COMPLETED = "completed"

@dataclass
class ColoringImage:
    """Represents a coloring image with its metadata"""
    name: str
    filepath: str
    thumbnail_path: str
    difficulty: str  # "easy", "medium", "hard"
    category: str    # "animals", "nature", "fantasy", etc.
    description: str

class ColorPaintGame:
    """Core game logic for opening images and painting on them"""
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            # Use user-specific data directory
            home_dir = os.path.expanduser("~")
            self.data_dir = os.path.join(home_dir, ".cvgames", "airpainter", "data")
        else:
            self.data_dir = data_dir
            
        self.images_dir = os.path.join(self.data_dir, "images")
        self.thumbnails_dir = os.path.join(self.data_dir, "thumbnails")
        
        # Ensure directories exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.thumbnails_dir, exist_ok=True)
        
        # Game state
        self.current_state = GameState.SELECTING_IMAGE
        self.selected_image: Optional[ColoringImage] = None
        self.available_images: List[ColoringImage] = []
        self.current_painting_surface: Optional[pygame.Surface] = None
        self.original_image: Optional[pygame.Surface] = None
        self.painted_areas: set = set()  # Track painted pixels
        self.completion_percentage = 0.0
        
        # Painting state
        self.is_painting = False
        self.last_paint_pos = None
        self.paint_history = []  # For undo functionality
        
        # Load available images
        self.load_available_images()
        
        # Create sample images if none exist
        if not self.available_images:
            self.create_sample_images()
            
        logger.info("Color Paint Game initialized", extra={
            "images_count": len(self.available_images),
            "data_dir": self.data_dir
        })
    
    def load_available_images(self):
        """Load all available coloring images from the data directory"""
        try:
            if not os.path.exists(self.images_dir):
                return
                
            for filename in os.listdir(self.images_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    name = os.path.splitext(filename)[0]
                    filepath = os.path.join(self.images_dir, filename)
                    thumbnail_path = os.path.join(self.thumbnails_dir, f"{name}_thumb.png")
                    
                    # Create thumbnail if it doesn't exist
                    if not os.path.exists(thumbnail_path):
                        self.create_thumbnail(filepath, thumbnail_path)
                    
                    # Determine difficulty based on filename or content
                    difficulty = self.analyze_difficulty(filepath)
                    category = self.get_category_from_name(name)
                    
                    image = ColoringImage(
                        name=name,
                        filepath=filepath,
                        thumbnail_path=thumbnail_path,
                        difficulty=difficulty,
                        category=category,
                        description=f"Color this {category} image"
                    )
                    self.available_images.append(image)
                    
            logger.info(f"Loaded {len(self.available_images)} coloring images")
            
        except Exception as e:
            logger.error("Failed to load available images", extra={"error": str(e)})
    
    def create_thumbnail(self, image_path: str, thumbnail_path: str, size: Tuple[int, int] = (200, 200)):
        """Create a thumbnail for the given image"""
        try:
            # Load image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return
                
            # Resize to thumbnail size
            thumbnail = cv2.resize(img, size)
            cv2.imwrite(thumbnail_path, thumbnail)
            
        except Exception as e:
            logger.error("Failed to create thumbnail", extra={"error": str(e), "image_path": image_path})
    
    def analyze_difficulty(self, image_path: str) -> str:
        """Analyze image complexity to determine difficulty"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return "medium"
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Count unique colors (simplified complexity measure)
            unique_colors = len(np.unique(gray))
            
            if unique_colors < 50:
                return "easy"
            elif unique_colors < 100:
                return "medium"
            else:
                return "hard"
                
        except Exception as e:
            logger.warning("Could not analyze difficulty", extra={"error": str(e)})
            return "medium"
    
    def get_category_from_name(self, name: str) -> str:
        """Extract category from image name"""
        name_lower = name.lower()
        
        if any(word in name_lower for word in ['cat', 'dog', 'bird', 'fish', 'animal']):
            return "animals"
        elif any(word in name_lower for word in ['tree', 'flower', 'nature', 'landscape']):
            return "nature"
        elif any(word in name_lower for word in ['dragon', 'unicorn', 'fantasy', 'magic']):
            return "fantasy"
        elif any(word in name_lower for word in ['car', 'plane', 'vehicle', 'transport']):
            return "vehicles"
        else:
            return "general"
    
    def create_sample_images(self):
        """Create sample coloring images for demonstration"""
        try:
            # Create a simple sample image (cat outline)
            sample_images = [
                ("simple_cat", self.create_cat_outline),
                ("simple_flower", self.create_flower_outline),
                ("simple_house", self.create_house_outline),
                ("simple_tree", self.create_tree_outline)
            ]
            
            for name, create_func in sample_images:
                image_surface = create_func()
                filepath = os.path.join(self.images_dir, f"{name}.png")
                pygame.image.save(image_surface, filepath)
                
                # Create thumbnail
                thumbnail_path = os.path.join(self.thumbnails_dir, f"{name}_thumb.png")
                self.create_thumbnail(filepath, thumbnail_path)
                
                # Add to available images
                image = ColoringImage(
                    name=name,
                    filepath=filepath,
                    thumbnail_path=thumbnail_path,
                    difficulty="easy",
                    category=self.get_category_from_name(name),
                    description=f"Color this {self.get_category_from_name(name)} image"
                )
                self.available_images.append(image)
                
            logger.info("Created sample images")
            
        except Exception as e:
            logger.error("Failed to create sample images", extra={"error": str(e)})
    
    def create_cat_outline(self) -> pygame.Surface:
        """Create a simple cat outline for coloring"""
        surface = pygame.Surface((400, 400))
        surface.fill((255, 255, 255))
        
        # Draw cat outline (simplified)
        points = [
            (200, 100),  # Head
            (150, 150),  # Left ear
            (250, 150),  # Right ear
            (200, 200),  # Body
            (180, 250),  # Left leg
            (220, 250),  # Right leg
            (200, 300),  # Tail
        ]
        
        # Draw outline
        for i in range(len(points) - 1):
            pygame.draw.line(surface, (0, 0, 0), points[i], points[i + 1], 3)
        
        # Draw face features
        pygame.draw.circle(surface, (0, 0, 0), (180, 120), 5)  # Left eye
        pygame.draw.circle(surface, (0, 0, 0), (220, 120), 5)  # Right eye
        pygame.draw.circle(surface, (0, 0, 0), (200, 140), 3)  # Nose
        
        return surface
    
    def create_flower_outline(self) -> pygame.Surface:
        """Create a simple flower outline for coloring"""
        surface = pygame.Surface((400, 400))
        surface.fill((255, 255, 255))
        
        # Draw flower petals
        center = (200, 200)
        for i in range(8):
            angle = i * 45
            x = center[0] + 80 * np.cos(np.radians(angle))
            y = center[1] + 80 * np.sin(np.radians(angle))
            pygame.draw.circle(surface, (0, 0, 0), (int(x), int(y)), 30, 2)
        
        # Draw center
        pygame.draw.circle(surface, (0, 0, 0), center, 20, 2)
        
        # Draw stem
        pygame.draw.line(surface, (0, 0, 0), (200, 220), (200, 350), 3)
        
        return surface
    
    def create_house_outline(self) -> pygame.Surface:
        """Create a simple house outline for coloring"""
        surface = pygame.Surface((400, 400))
        surface.fill((255, 255, 255))
        
        # Draw house body
        pygame.draw.rect(surface, (0, 0, 0), (150, 200, 100, 120), 3)
        
        # Draw roof
        points = [(150, 200), (200, 150), (250, 200)]
        pygame.draw.polygon(surface, (0, 0, 0), points, 3)
        
        # Draw door
        pygame.draw.rect(surface, (0, 0, 0), (180, 250, 40, 70), 2)
        
        # Draw windows
        pygame.draw.rect(surface, (0, 0, 0), (160, 220, 20, 20), 2)
        pygame.draw.rect(surface, (0, 0, 0), (220, 220, 20, 20), 2)
        
        return surface
    

    
    def create_tree_outline(self) -> pygame.Surface:
        """Create a simple tree outline for coloring"""
        surface = pygame.Surface((400, 400))
        surface.fill((255, 255, 255))
        
        # Draw trunk
        pygame.draw.rect(surface, (0, 0, 0), (180, 250, 40, 120), 3)
        
        # Draw leaves (circles)
        leaf_positions = [
            (200, 200), (160, 180), (240, 180),
            (140, 160), (260, 160), (200, 140),
            (180, 160), (220, 160)
        ]
        
        for pos in leaf_positions:
            pygame.draw.circle(surface, (0, 0, 0), pos, 25, 2)
        
        return surface
    
    def select_image(self, image: ColoringImage):
        """Select an image to paint on"""
        try:
            self.selected_image = image
            self.current_state = GameState.PAINTING
            
            # Load the image
            self.original_image = pygame.image.load(image.filepath)
            
            # Create painting surface (same size as original)
            self.current_painting_surface = pygame.Surface(self.original_image.get_size())
            self.current_painting_surface.fill((255, 255, 255))  # White background
            
            # Reset painting state
            self.painted_areas.clear()
            self.completion_percentage = 0.0
            self.is_painting = False
            self.last_paint_pos = None
            self.paint_history = []
            
            logger.info("Image selected for painting", extra={
                "image_name": image.name,
                "image_size": self.original_image.get_size()
            })
            
        except Exception as e:
            logger.error("Failed to select image", extra={"error": str(e), "image": image.name})
    
    def start_painting(self, x: int, y: int, color: Tuple[int, int, int], brush_size: int = 5):
        """Start painting at the specified position"""
        if not self.current_painting_surface or not self.original_image:
            return False
            
        self.is_painting = True
        self.last_paint_pos = (x, y)
        
        # Save current state for undo
        self.save_paint_state()
        
        return self.paint_area(x, y, color, brush_size)
    
    def continue_painting(self, x: int, y: int, color: Tuple[int, int, int], brush_size: int = 5):
        """Continue painting from last position to current position"""
        if not self.is_painting or not self.last_paint_pos:
            return self.start_painting(x, y, color, brush_size)
        
        # Paint line from last position to current position
        start_x, start_y = self.last_paint_pos
        end_x, end_y = x, y
        
        # Calculate intermediate points for smooth line
        distance = max(abs(end_x - start_x), abs(end_y - start_y))
        if distance > 0:
            for i in range(distance + 1):
                t = i / distance
                px = int(start_x + t * (end_x - start_x))
                py = int(start_y + t * (end_y - start_y))
                self.paint_area(px, py, color, brush_size)
        
        self.last_paint_pos = (x, y)
        return True
    
    def stop_painting(self):
        """Stop the current painting session"""
        self.is_painting = False
        self.last_paint_pos = None
    
    def paint_area(self, x: int, y: int, color: Tuple[int, int, int], brush_size: int = 5):
        """Paint an area of the image with the specified color"""
        if not self.current_painting_surface or not self.original_image:
            return False
            
        try:
            # Check if we're within image bounds
            if x < 0 or y < 0 or x >= self.original_image.get_width() or y >= self.original_image.get_height():
                return False
            
            # Paint a circular area around the point
            for dx in range(-brush_size, brush_size + 1):
                for dy in range(-brush_size, brush_size + 1):
                    if dx*dx + dy*dy <= brush_size*brush_size:  # Circular brush
                        px, py = x + dx, y + dy
                        if (0 <= px < self.original_image.get_width() and 
                            0 <= py < self.original_image.get_height()):
                            
                            # Check if this pixel is part of the outline (black pixels)
                            pixel_color = self.original_image.get_at((px, py))
                            if pixel_color[:3] == (0, 0, 0):  # Black outline
                                self.current_painting_surface.set_at((px, py), color)
                                self.painted_areas.add((px, py))
            
            # Update completion percentage
            self.update_completion_percentage()
            
            return True
            
        except Exception as e:
            logger.error("Failed to paint area", extra={"error": str(e)})
            return False
    
    def save_paint_state(self):
        """Save current painting state for undo functionality"""
        if self.current_painting_surface:
            # Limit history to prevent memory issues
            if len(self.paint_history) > 10:
                self.paint_history.pop(0)
            self.paint_history.append(self.current_painting_surface.copy())
    
    def undo_last_paint(self):
        """Undo the last painting action"""
        if self.paint_history:
            self.current_painting_surface = self.paint_history.pop()
            # Recalculate painted areas
            self.recalculate_painted_areas()
            self.update_completion_percentage()
            logger.info("Undo performed")
            return True
        return False
    
    def recalculate_painted_areas(self):
        """Recalculate painted areas from current painting surface"""
        if not self.current_painting_surface or not self.original_image:
            return
            
        self.painted_areas.clear()
        for x in range(self.original_image.get_width()):
            for y in range(self.original_image.get_height()):
                original_pixel = self.original_image.get_at((x, y))
                painted_pixel = self.current_painting_surface.get_at((x, y))
                
                # If original was black (outline) and painted pixel is not white
                if original_pixel[:3] == (0, 0, 0) and painted_pixel[:3] != (255, 255, 255):
                    self.painted_areas.add((x, y))
    
    def update_completion_percentage(self):
        """Update the completion percentage based on painted areas"""
        if not self.original_image:
            return
            
        try:
            # Count total paintable pixels (black outline pixels)
            total_paintable = 0
            for x in range(self.original_image.get_width()):
                for y in range(self.original_image.get_height()):
                    pixel_color = self.original_image.get_at((x, y))
                    if pixel_color[:3] == (0, 0, 0):  # Black outline
                        total_paintable += 1
            
            if total_paintable > 0:
                self.completion_percentage = (len(self.painted_areas) / total_paintable) * 100
                
                # Check if completed
                if self.completion_percentage >= 95:  # Allow 5% tolerance
                    self.current_state = GameState.COMPLETED
                    logger.info("Image painting completed", extra={"completion": self.completion_percentage})
                    
        except Exception as e:
            logger.error("Failed to update completion percentage", extra={"error": str(e)})
    
    def get_completion_percentage(self) -> float:
        """Get the current completion percentage"""
        return self.completion_percentage
    
    def get_current_image(self) -> Optional[pygame.Surface]:
        """Get the current painting surface"""
        return self.current_painting_surface
    
    def get_original_image(self) -> Optional[pygame.Surface]:
        """Get the original image"""
        return self.original_image
    
    def get_available_images(self) -> List[ColoringImage]:
        """Get list of available images"""
        return self.available_images
    
    def get_current_state(self) -> GameState:
        """Get current game state"""
        return self.current_state
    
    def is_painting_active(self) -> bool:
        """Check if currently painting"""
        return self.is_painting
    
    def reset_game(self):
        """Reset the game to image selection state"""
        self.current_state = GameState.SELECTING_IMAGE
        self.selected_image = None
        self.current_painting_surface = None
        self.original_image = None
        self.painted_areas.clear()
        self.completion_percentage = 0.0
        self.is_painting = False
        self.last_paint_pos = None
        self.paint_history.clear()
        
        logger.info("Game reset to selection state") 