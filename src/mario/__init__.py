import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from screeninfo import get_monitors
from cvzone.HandTrackingModule import HandDetector
import cvzone
import argparse

# Initialize pygame mixer
pygame.mixer.init()

# Load sound effects
try:
    jump_sound = pygame.mixer.Sound("Resources/jump.mp3")
    coin_sound = pygame.mixer.Sound("Resources/coin.mp3")
    game_over_sound = pygame.mixer.Sound("Resources/game_over.mp3")
except:
    print("Warning: Sound files not found. Game will run without sound.")
    jump_sound = None
    coin_sound = None
    game_over_sound = None

# Load game resources
try:
    mario_img = cv2.imread("Resources/mario.png", cv2.IMREAD_UNCHANGED)
    ground_img = cv2.imread("Resources/ground.png")
    brick_img = cv2.imread("Resources/brick.png")
    coin_img = cv2.imread("Resources/coin.png")
    enemy_img = cv2.imread("Resources/enemy.png")
    background_img = cv2.imread("Resources/background.png")
    
    # Resize images if needed
    mario_img = cv2.resize(mario_img, (40, 60))
    ground_img = cv2.resize(ground_img, (40, 40))
    brick_img = cv2.resize(brick_img, (40, 40))
    coin_img = cv2.resize(coin_img, (30, 30))
    enemy_img = cv2.resize(enemy_img, (40, 40))
    background_img = cv2.resize(background_img, (1280, 720))
except Exception as e:
    print(f"Warning: Could not load game resources: {e}")
    print("Using simple shapes instead of sprites.")
    mario_img = None
    ground_img = None
    brick_img = None
    coin_img = None
    enemy_img = None
    background_img = np.zeros((720, 1280, 3), dtype=np.uint8)

class Mario:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 60
        self.velocity_y = 0
        self.velocity_x = 0
        self.jumping = False
        self.facing_right = True
        self.lives = 3
        self.score = 0
        self.gravity = 0.8
        self.jump_force = -15
        self.speed = 5

    def move(self, direction):
        self.velocity_x = direction * self.speed
        self.facing_right = direction > 0

    def jump(self):
        if not self.jumping:
            self.velocity_y = self.jump_force
            self.jumping = True
            if jump_sound:
                jump_sound.play()

    def update(self, platforms):
        # Apply gravity
        self.velocity_y += self.gravity
        self.y += self.velocity_y

        # Apply horizontal movement
        self.x += self.velocity_x

        # Check platform collisions
        for platform in platforms:
            if self.check_collision(platform):
                if self.velocity_y > 0:  # Falling
                    self.y = platform.y - self.height
                    self.velocity_y = 0
                    self.jumping = False
                elif self.velocity_y < 0:  # Jumping
                    self.y = platform.y + platform.height
                    self.velocity_y = 0

        # Keep player in bounds
        self.x = max(0, min(self.x, 1280 - self.width))
        self.y = max(0, min(self.y, 720 - self.height))

    def check_collision(self, platform):
        return (self.x < platform.x + platform.width and
                self.x + self.width > platform.x and
                self.y < platform.y + platform.height and
                self.y + self.height > platform.y)

    def draw(self, img):
        if mario_img is not None:
            # Draw Mario sprite
            y1, y2 = int(self.y), int(self.y + self.height)
            x1, x2 = int(self.x), int(self.x + self.width)
            if 0 <= y1 < img.shape[0] and 0 <= y2 <= img.shape[0] and \
               0 <= x1 < img.shape[1] and 0 <= x2 <= img.shape[1]:
                # Extract alpha channel
                alpha = mario_img[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=-1)
                # Blend Mario with background
                img[y1:y2, x1:x2] = (1 - alpha) * img[y1:y2, x1:x2] + \
                                  alpha * mario_img[:, :, :3]
        else:
            # Draw simple rectangle if sprite not available
            cv2.rectangle(img, 
                         (int(self.x), int(self.y)),
                         (int(self.x + self.width), int(self.y + self.height)),
                         (255, 0, 0), -1)

class Platform:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self, img):
        if ground_img is not None:
            # Draw platform sprite
            y1, y2 = int(self.y), int(self.y + self.height)
            x1, x2 = int(self.x), int(self.x + self.width)
            if 0 <= y1 < img.shape[0] and 0 <= y2 <= img.shape[0] and \
               0 <= x1 < img.shape[1] and 0 <= x2 <= img.shape[1]:
                img[y1:y2, x1:x2] = ground_img
        else:
            # Draw simple rectangle if sprite not available
            cv2.rectangle(img,
                         (int(self.x), int(self.y)),
                         (int(self.x + self.width), int(self.y + self.height)),
                         (139, 69, 19), -1)

class Coin:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 30
        self.collected = False

    def draw(self, img):
        if not self.collected:
            if coin_img is not None:
                # Draw coin sprite
                y1, y2 = int(self.y), int(self.y + self.height)
                x1, x2 = int(self.x), int(self.x + self.width)
                if 0 <= y1 < img.shape[0] and 0 <= y2 <= img.shape[0] and \
                   0 <= x1 < img.shape[1] and 0 <= x2 <= img.shape[1]:
                    img[y1:y2, x1:x2] = coin_img
            else:
                # Draw simple circle if sprite not available
                cv2.circle(img,
                          (int(self.x + self.width//2), int(self.y + self.height//2)),
                          self.width//2, (255, 215, 0), -1)

class MarioGame:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.mario = Mario(100, height - 200)
        self.platforms = []
        self.coins = []
        self.game_over = False
        self.setup_level()

    def setup_level(self):
        # Create ground
        self.platforms.append(Platform(0, self.height - 40, self.width, 40))
        
        # Create some platforms
        platform_positions = [
            (300, self.height - 200, 200, 20),
            (600, self.height - 300, 200, 20),
            (900, self.height - 400, 200, 20),
        ]
        
        for x, y, w, h in platform_positions:
            self.platforms.append(Platform(x, y, w, h))
        
        # Add coins
        coin_positions = [
            (350, self.height - 250),
            (650, self.height - 350),
            (950, self.height - 450),
        ]
        
        for x, y in coin_positions:
            self.coins.append(Coin(x, y))

    def update(self):
        if not self.game_over:
            self.mario.update(self.platforms)
            
            # Check coin collection
            for coin in self.coins:
                if not coin.collected and self.mario.check_collision(coin):
                    coin.collected = True
                    self.mario.score += 100
                    if coin_sound:
                        coin_sound.play()

            # Check if Mario fell off
            if self.mario.y > self.height:
                self.mario.lives -= 1
                if self.mario.lives <= 0:
                    self.game_over = True
                    if game_over_sound:
                        game_over_sound.play()
                else:
                    # Reset Mario position
                    self.mario.x = 100
                    self.mario.y = self.height - 200
                    self.mario.velocity_y = 0
                    self.mario.jumping = False

    def draw(self, img):
        # Draw background
        if background_img is not None:
            img[:] = background_img
        else:
            img[:] = (135, 206, 235)  # Sky blue background

        # Draw platforms
        for platform in self.platforms:
            platform.draw(img)

        # Draw coins
        for coin in self.coins:
            coin.draw(img)

        # Draw Mario
        self.mario.draw(img)

        # Draw HUD
        cvzone.putTextRect(img, f'Score: {self.mario.score}', [20, 40],
                          scale=1.5, thickness=2, offset=10)
        cvzone.putTextRect(img, f'Lives: {self.mario.lives}', [20, 80],
                          scale=1.5, thickness=2, offset=10)

        if self.game_over:
            cvzone.putTextRect(img, "Game Over", [400, 300],
                             scale=3, thickness=5, offset=20)
            cvzone.putTextRect(img, f'Final Score: {self.mario.score}', [400, 400],
                             scale=3, thickness=5, offset=20)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Mario Game with MediaPipe Hand Tracking')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    args = parser.parse_args()

    # Set HD resolution
    GAME_WIDTH = 1280
    GAME_HEIGHT = 720

    # Initialize game
    game = MarioGame(GAME_WIDTH, GAME_HEIGHT)

    # Initialize hand detector
    detector = HandDetector(detectionCon=0.8, maxHands=2)

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

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, GAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GAME_HEIGHT)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Flip the image horizontally for a later selfie-view display
        img = cv2.flip(img, 1)

        # Find hands
        hands, img = detector.findHands(img, flipType=False)

        # Process hand gestures
        if hands:
            for hand in hands:
                # Get hand landmarks
                lmList = hand["lmList"]
                if len(lmList) != 0:
                    # Get hand type (left or right)
                    handType = hand["type"]
                    
                    # Get hand center and fingers up
                    center = hand["center"]
                    fingers = detector.fingersUp(hand)

                    # Control Mario based on hand gestures
                    if handType == "Left":
                        # Left hand controls horizontal movement
                        if center[0] < GAME_WIDTH // 3:
                            game.mario.move(-1)
                        elif center[0] > (GAME_WIDTH * 2) // 3:
                            game.mario.move(1)
                        else:
                            game.mario.move(0)
                    else:
                        # Right hand controls jumping
                        if fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                            game.mario.jump()

        # Update and draw game
        game.update()
        game.draw(img)

        # Display the game
        cv2.imshow("MediaPipe Mario", img)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()