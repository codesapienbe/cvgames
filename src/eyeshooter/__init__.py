import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
import sys

class Target:
    def __init__(self, x, y, size=30, speed=2):
        self.x = x
        self.y = y
        self.size = size
        self.speed = speed
        self.active = True
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def update(self, frame_width, frame_height):
        # Move target
        self.y += self.speed
        
        # Remove if off screen
        if self.y > frame_height + self.size:
            self.active = False
    
    def draw(self, frame):
        if self.active:
            cv2.circle(frame, (int(self.x), int(self.y)), self.size, self.color, -1)
            cv2.circle(frame, (int(self.x), int(self.y)), self.size, (255, 255, 255), 2)

class Bullet:
    def __init__(self, x, y, target_x, target_y):
        self.x = x
        self.y = y
        self.speed = 15
        
        # Calculate direction to target
        dx = target_x - x
        dy = target_y - y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            self.dx = (dx / distance) * self.speed
            self.dy = (dy / distance) * self.speed
        else:
            self.dx = 0
            self.dy = -self.speed
    
    def update(self):
        self.x += self.dx
        self.y += self.dy
    
    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), 5, (255, 255, 0), -1)
    
    def is_off_screen(self, frame_width, frame_height):
        return (self.x < 0 or self.x > frame_width or 
                self.y < 0 or self.y > frame_height)

class EyeShooter:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.targets = []
        self.bullets = []
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.last_shot_time = 0
        self.shot_cooldown = 0.5  # seconds
        self.target_spawn_timer = 0
        self.target_spawn_delay = 2.0  # seconds
        self.crosshair_x = frame_width // 2
        self.crosshair_y = frame_height // 2
    
    def spawn_target(self):
        x = random.randint(50, self.frame_width - 50)
        y = -30
        size = random.randint(20, 40)
        speed = random.uniform(1, 3)
        self.targets.append(Target(x, y, size, speed))
    
    def shoot(self, target_x, target_y):
        current_time = time.time()
        if current_time - self.last_shot_time >= self.shot_cooldown:
            # Create bullet from crosshair position
            bullet = Bullet(self.crosshair_x, self.crosshair_y, target_x, target_y)
            self.bullets.append(bullet)
            self.last_shot_time = current_time
    
    def update(self):
        current_time = time.time()
        
        # Spawn targets
        if current_time - self.target_spawn_timer >= self.target_spawn_delay:
            self.spawn_target()
            self.target_spawn_timer = current_time
        
        # Update targets
        for target in self.targets[:]:
            target.update(self.frame_width, self.frame_height)
            if not target.active:
                self.targets.remove(target)
                self.lives -= 1
                if self.lives <= 0:
                    self.game_over = True
        
        # Update bullets
        for bullet in self.bullets[:]:
            bullet.update()
            if bullet.is_off_screen(self.frame_width, self.frame_height):
                self.bullets.remove(bullet)
                continue
            
            # Check collision with targets
            for target in self.targets[:]:
                distance = math.sqrt((bullet.x - target.x)**2 + (bullet.y - target.y)**2)
                if distance < target.size:
                    self.targets.remove(target)
                    self.bullets.remove(bullet)
                    self.score += 10
                    break
    
    def draw(self, frame):
        # Draw crosshair
        crosshair_size = 20
        cv2.line(frame, (self.crosshair_x - crosshair_size, self.crosshair_y), 
                (self.crosshair_x + crosshair_size, self.crosshair_y), (0, 255, 0), 2)
        cv2.line(frame, (self.crosshair_x, self.crosshair_y - crosshair_size), 
                (self.crosshair_x, self.crosshair_y + crosshair_size), (0, 255, 0), 2)
        cv2.circle(frame, (self.crosshair_x, self.crosshair_y), 5, (0, 255, 0), -1)
        
        # Draw targets
        for target in self.targets:
            target.draw(frame)
        
        # Draw bullets
        for bullet in self.bullets:
            bullet.draw(frame)
        
        # Draw UI
        cv2.putText(frame, f"Score: {self.score}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Lives: {self.lives}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if self.game_over:
            cv2.putText(frame, "Game Over!", (self.frame_width//2 - 100, self.frame_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(frame, f"Final Score: {self.score}", 
                       (self.frame_width//2 - 100, self.frame_height//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'r' to restart or 'q' to quit", 
                       (self.frame_width//2 - 200, self.frame_height//2 + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Blink to shoot at targets", (10, self.frame_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def reset_game(self):
        self.targets = []
        self.bullets = []
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.last_shot_time = 0
        self.target_spawn_timer = 0

def detect_eye_gaze(face_landmarks, frame_width, frame_height):
    """Detect eye gaze direction and convert to screen position"""
    if not face_landmarks:
        return None
    
    # Get eye landmarks
    left_eye = face_landmarks.landmark[159]  # Left eye center
    right_eye = face_landmarks.landmark[386]  # Right eye center
    
    # Calculate average eye position
    eye_x = (left_eye.x + right_eye.x) / 2
    eye_y = (left_eye.y + right_eye.y) / 2
    
    # Convert to pixel coordinates
    pixel_x = int(eye_x * frame_width)
    pixel_y = int(eye_y * frame_height)
    
    return (pixel_x, pixel_y)

def detect_blink(face_landmarks):
    """Detect eye blink using eye aspect ratio"""
    if not face_landmarks:
        return False
    
    # Get eye landmarks for left eye
    left_eye_top = face_landmarks.landmark[386]  # Upper eyelid
    left_eye_bottom = face_landmarks.landmark[374]  # Lower eyelid
    left_eye_left = face_landmarks.landmark[263]  # Left corner
    left_eye_right = face_landmarks.landmark[362]  # Right corner
    
    # Calculate eye aspect ratio
    left_ear = (np.linalg.norm([left_eye_top.x - left_eye_bottom.x, 
                               left_eye_top.y - left_eye_bottom.y]) + 
                np.linalg.norm([left_eye_left.x - left_eye_right.x, 
                               left_eye_left.y - left_eye_right.y])) / (2 * 
                np.linalg.norm([left_eye_left.x - left_eye_right.x, 
                               left_eye_left.y - left_eye_right.y]))
    
    # Threshold for blink detection
    return left_ear < 0.2

def main():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Initialize game
    game = EyeShooter(frame_width, frame_height)
    # Blink detection variables
    last_blink_time = 0
    blink_cooldown = 0.3  # seconds
    # --- Pygame Setup ---
    pygame.init()
    screen = pygame.display.set_mode((frame_width, frame_height))
    pygame.display.set_caption("Eye Shooter (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    clock = pygame.time.Clock()
    # Set up OpenTelemetry tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)
    with tracer.start_as_current_span("eyeshooter_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            current_time = time.time()
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                # Detect eye gaze
                gaze_pos = detect_eye_gaze(face_landmarks, frame_width, frame_height)
                if gaze_pos:
                    game.crosshair_x, game.crosshair_y = gaze_pos
                # Detect blink
                if detect_blink(face_landmarks):
                    if current_time - last_blink_time >= blink_cooldown:
                        with tracer.start_as_current_span("shoot"):
                            game.shoot(game.crosshair_x, game.crosshair_y)
                        last_blink_time = current_time
            # Update and draw game
            if not game.game_over:
                game.update()
            game.draw(frame)
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            game.reset_game()
            # --- Pygame Rendering ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 