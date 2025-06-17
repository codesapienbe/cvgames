import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import time
from screeninfo import get_monitors
from cvzone.HandTrackingModule import HandDetector
import cvzone
import argparse

# Initialize pygame mixer
pygame.mixer.init()

# Load sound effects
try:
    shoot_sound = pygame.mixer.Sound("Resources/shoot.mp3")
    explosion_sound = pygame.mixer.Sound("Resources/explosion.mp3")
    game_over_sound = pygame.mixer.Sound("Resources/game_over.mp3")
except:
    print("Warning: Sound files not found. Game will run without sound.")
    shoot_sound = None
    explosion_sound = None
    game_over_sound = None

# Load background image
try:
    background_img = cv2.imread("Resources/Background.png")
    if background_img is None:
        print("Warning: Background image not found. Using black background.")
        background_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    else:
        background_img = cv2.resize(background_img, (1280, 720))
except:
    print("Warning: Could not load background image. Using black background.")
    background_img = np.zeros((720, 1280, 3), dtype=np.uint8)

class Bullet:
    def __init__(self, x, y, speed=7):
        self.x = x
        self.y = y
        self.speed = speed
        self.active = True

    def move(self):
        self.y -= self.speed

    def draw(self, img):
        # Make bullets more visible with a larger size and brighter color
        cv2.circle(img, (int(self.x), int(self.y)), 5, (0, 0, 255), -1)  # Red color
        cv2.circle(img, (int(self.x), int(self.y)), 3, (255, 255, 255), -1)  # White center

class Alien:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 40
        self.speed = 2
        self.direction = 1
        self.active = True
        # Random color for each alien
        self.color = (
            random.randint(0, 255),  # B
            random.randint(0, 255),  # G
            random.randint(0, 255)   # R
        )

    def move(self):
        self.x += self.speed * self.direction

    def draw(self, img):
        # Draw alien with its unique color
        cv2.rectangle(img, 
                     (int(self.x - self.width//2), int(self.y - self.height//2)),
                     (int(self.x + self.width//2), int(self.y + self.height//2)),
                     self.color, -1)
        # Add a border to make it more visible
        cv2.rectangle(img, 
                     (int(self.x - self.width//2), int(self.y - self.height//2)),
                     (int(self.x + self.width//2), int(self.y + self.height//2)),
                     (255, 255, 255), 2)  # White border

class SpaceInvadersGame:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.player_x = width // 2
        self.player_y = height - 100
        self.player_width = 60
        self.player_height = 40
        self.bullets = []
        self.aliens = []
        self.score = 0
        self.game_over = False
        self.level = 1
        self.spawn_timer = 0
        self.spawn_delay = 2.0  # Initial delay between alien spawns
        self.startTime = time.time()
        self.gameDuration = 120  # 2 minutes in seconds

    def getRemainingTime(self):
        elapsedTime = time.time() - self.startTime
        remainingTime = max(0, self.gameDuration - elapsedTime)
        return int(remainingTime)

    def spawn_alien(self):
        x = random.randint(50, self.width - 50)
        self.aliens.append(Alien(x, 50))

    def check_collision(self, bullet, alien):
        return (abs(bullet.x - alien.x) < alien.width//2 and
                abs(bullet.y - alien.y) < alien.height//2)

    def update(self):
        # Spawn aliens
        current_time = time.time()
        if current_time - self.spawn_timer > self.spawn_delay:
            self.spawn_alien()
            self.spawn_timer = current_time
            # Decrease spawn delay as level increases
            self.spawn_delay = max(0.5, 2.0 - (self.level - 1) * 0.2)

        # Update bullets
        for bullet in self.bullets[:]:
            bullet.move()
            if bullet.y < 0:
                self.bullets.remove(bullet)

        # Update aliens
        for alien in self.aliens[:]:
            alien.move()
            
            # Change direction if reaching screen edges
            if alien.x < 50 or alien.x > self.width - 50:
                alien.direction *= -1
                alien.y += 20

            # Check for collision with bullets
            for bullet in self.bullets[:]:
                if self.check_collision(bullet, alien):
                    self.bullets.remove(bullet)
                    self.aliens.remove(alien)
                    self.score += 100
                    if explosion_sound:
                        explosion_sound.play()
                    break

            # Check if aliens reached bottom
            if alien.y > self.height - 100:
                self.game_over = True
                if game_over_sound:
                    game_over_sound.play()

        # Level up every 500 points
        self.level = self.score // 500 + 1

    def draw(self, img):
        # Draw player as a triangle
        player_points = np.array([
            [int(self.player_x), int(self.player_y - self.player_height//2)],  # Top point
            [int(self.player_x - self.player_width//2), int(self.player_y + self.player_height//2)],  # Bottom left
            [int(self.player_x + self.player_width//2), int(self.player_y + self.player_height//2)]   # Bottom right
        ], np.int32)
        player_points = player_points.reshape((-1, 1, 2))
        cv2.fillPoly(img, [player_points], (0, 0, 255))  # Red color
        # Add a white border to make it more visible
        cv2.polylines(img, [player_points], True, (255, 255, 255), 2)

        # Draw bullets
        for bullet in self.bullets:
            bullet.draw(img)

        # Draw aliens
        for alien in self.aliens:
            alien.draw(img)

        remainingTime = self.getRemainingTime()
        
        if self.game_over or remainingTime <= 0:
            if remainingTime <= 0:
                cvzone.putTextRect(img, "Time's Up!", [300, 400],
                                   scale=7, thickness=5, offset=20)
            else:
                cvzone.putTextRect(img, "Game Over", [300, 400],
                                   scale=7, thickness=5, offset=20)
            cvzone.putTextRect(img, f'Your Score: {self.score}', [300, 550],
                               scale=7, thickness=5, offset=20)
        else:
            # Draw score and timer in the top right corner with pink color
            rightPadding = 20  # Padding from the right edge
            cvzone.putTextRect(img, f'Score: {self.score}', 
                               [img.shape[1] - 200, 80],
                               scale=1.5, thickness=2, offset=10, colorR=(255, 192, 203))  # Pink color
            cvzone.putTextRect(img, f'Time: {remainingTime}s', 
                               [img.shape[1] - 200, 120],
                               scale=1.5, thickness=2, offset=10, colorR=(255, 192, 203))  # Pink color
            cvzone.putTextRect(img, f'Level: {self.level}', 
                               [img.shape[1] - 200, 160],
                               scale=1.5, thickness=2, offset=10, colorR=(255, 192, 203))  # Pink color

def play_sound(sound):
    """Play a sound effect using pygame"""
    if sound:
        try:
            sound.play()
        except:
            pass

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Space Invaders with MediaPipe Hand Tracking')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    args = parser.parse_args()

    # Set HD resolution
    GAME_WIDTH = 1280
    GAME_HEIGHT = 720

    # Initialize game
    game = SpaceInvadersGame(GAME_WIDTH, GAME_HEIGHT)

    # List available cameras
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        print("Error: No cameras found!")
        return
    
    print(f"Available cameras: {available_cameras}")
    
    if args.camera not in available_cameras:
        print(f"Error: Camera {args.camera} is not available!")
        print(f"Please select one of the available cameras: {available_cameras}")
        return

    print(f"Attempting to open camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        print("Please check if:")
        print("1. The camera is properly connected")
        print("2. The camera is not being used by another application")
        print("3. You have the correct camera index")
        return

    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Successfully opened camera {args.camera}")
    print(f"Camera properties: {width}x{height} @ {fps}fps")

    cap.set(3, GAME_WIDTH)
    cap.set(4, GAME_HEIGHT)
    detector = HandDetector(detectionCon=0.75, maxHands=1)

    last_shoot_time = time.time()
    shoot_delay = 0.5  # Delay between shots

    while cap.isOpened():
        success, img = cap.read()

        if img is None or img.size == 0:
            print("Error: Could not read frame from camera")
            break

        img = cv2.flip(img, 1)

        if not game.game_over:
            # Create a copy of the background image
            gameImg = background_img.copy()

            # Detect hands
            hands, img = detector.findHands(img, draw=True)

            if hands:
                hand = hands[0]
                fingers = detector.fingersUp(hand)
                
                # Move player based on hand position
                if hand['lmList']:
                    index_finger = hand['lmList'][8]
                    game.player_x = int(index_finger[0])
                    # Keep player within screen bounds
                    game.player_x = max(game.player_width//2, min(game.player_x, GAME_WIDTH - game.player_width//2))

                # Shoot when thumb is up
                current_time = time.time()
                if fingers[0] and current_time - last_shoot_time > shoot_delay:
                    game.bullets.append(Bullet(game.player_x, game.player_y - game.player_height//2))
                    last_shoot_time = current_time
                    if shoot_sound:
                        shoot_sound.play()

            # Update and draw game
            game.update()
            game.draw(gameImg)

            # Resize the webcam feed (2x smaller)
            webcamSize = (160, 120)  # Half the previous size
            webcamImg = cv2.resize(img, webcamSize)
            
            # Place the webcam feed in the bottom left corner
            y_offset = gameImg.shape[0] - webcamSize[1]
            x_offset = 0
            
            # Draw pink border around webcam frame
            borderThickness = 3
            cv2.rectangle(gameImg, 
                         (x_offset - borderThickness, y_offset - borderThickness),
                         (x_offset + webcamSize[0] + borderThickness, y_offset + webcamSize[1] + borderThickness),
                         (255, 192, 203),  # Pink color
                         borderThickness)
            
            # Place the webcam feed
            gameImg[y_offset:y_offset+webcamSize[1], x_offset:x_offset+webcamSize[0]] = webcamImg

            cv2.imshow("Space Invaders", gameImg)
        else:
            cv2.imshow("Space Invaders", gameImg)

        key = cv2.waitKey(1)
        if key == ord('r'):
            # Reset game
            game = SpaceInvadersGame(GAME_WIDTH, GAME_HEIGHT)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 