import cv2
import mediapipe as mp
import time
import random
import numpy as np
import pygame
from cvzone.HandTrackingModule import HandDetector
import cvzone
import argparse
import sys
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Loading screen function is defined locally below

pygame.mixer.init()

try:
    jump_sound = pygame.mixer.Sound("Resources/jump.mp3")
    coin_sound = pygame.mixer.Sound("Resources/coin.mp3")
    game_over_sound = pygame.mixer.Sound("Resources/game_over.mp3")
except:
    jump_sound = None
    coin_sound = None
    game_over_sound = None

try:
    mario_img = cv2.imread("Resources/mario.png", cv2.IMREAD_UNCHANGED)
    ground_img = cv2.imread("Resources/ground.png")
    brick_img = cv2.imread("Resources/brick.png")
    coin_img = cv2.imread("Resources/coin.png")
    enemy_img = cv2.imread("Resources/enemy.png")
    background_img = cv2.imread("Resources/background.png")
    mario_img = cv2.resize(mario_img, (40, 60))
    ground_img = cv2.resize(ground_img, (40, 40))
    brick_img = cv2.resize(brick_img, (40, 40))
    coin_img = cv2.resize(coin_img, (30, 30))
    enemy_img = cv2.resize(enemy_img, (40, 40))
    background_img = cv2.resize(background_img, (1280, 720))
except Exception as e:
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
        self.velocity_y += self.gravity
        self.y += self.velocity_y
        self.x += self.velocity_x
        for platform in platforms:
            if self.check_collision(platform):
                if self.velocity_y > 0:
                    self.y = platform.y - self.height
                    self.velocity_y = 0
                    self.jumping = False
                elif self.velocity_y < 0:
                    self.y = platform.y + platform.height
                    self.velocity_y = 0
        self.x = max(0, min(self.x, 1280 - self.width))
        self.y = max(0, min(self.y, 720 - self.height))

    def check_collision(self, platform):
        return (self.x < platform.x + platform.width and
                self.x + self.width > platform.x and
                self.y < platform.y + platform.height and
                self.y + self.height > platform.y)

    def draw(self, img):
        if mario_img is not None:
            y1, y2 = int(self.y), int(self.y + self.height)
            x1, x2 = int(self.x), int(self.x + self.width)
            if 0 <= y1 < img.shape[0] and 0 <= y2 <= img.shape[0] and 0 <= x1 < img.shape[1] and 0 <= x2 <= img.shape[1]:
                alpha = mario_img[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=-1)
                img[y1:y2, x1:x2] = (1 - alpha) * img[y1:y2, x1:x2] + alpha * mario_img[:, :, :3]
        else:
            cv2.rectangle(img, (int(self.x), int(self.y)), (int(self.x + self.width), int(self.y + self.height)), (255, 0, 0), -1)

class Platform:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self, img):
        if ground_img is not None:
            y1, y2 = int(self.y), int(self.y + self.height)
            x1, x2 = int(self.x), int(self.x + self.width)
            if 0 <= y1 < img.shape[0] and 0 <= y2 <= img.shape[0] and 0 <= x1 < img.shape[1] and 0 <= x2 <= img.shape[1]:
                img[y1:y2, x1:x2] = ground_img
        else:
            cv2.rectangle(img, (int(self.x), int(self.y)), (int(self.x + self.width), int(self.y + self.height)), (139, 69, 19), -1)

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
                y1, y2 = int(self.y), int(self.y + self.height)
                x1, x2 = int(self.x), int(self.x + self.width)
                if 0 <= y1 < img.shape[0] and 0 <= y2 <= img.shape[0] and 0 <= x1 < img.shape[1] and 0 <= x2 <= img.shape[1]:
                    img[y1:y2, x1:x2] = coin_img
            else:
                cv2.circle(img, (int(self.x + self.width//2), int(self.y + self.height//2)), self.width//2, (255, 215, 0), -1)

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
        self.platforms.append(Platform(0, self.height - 40, self.width, 40))
        platform_positions = [
            (300, self.height - 200, 200, 20),
            (600, self.height - 300, 200, 20),
            (900, self.height - 400, 200, 20),
        ]
        for x, y, w, h in platform_positions:
            self.platforms.append(Platform(x, y, w, h))
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
            for coin in self.coins:
                if not coin.collected and self.mario.check_collision(coin):
                    coin.collected = True
                    self.mario.score += 100
                    if coin_sound:
                        coin_sound.play()
            if self.mario.y > self.height:
                self.mario.lives -= 1
                if self.mario.lives <= 0:
                    self.game_over = True
                    if game_over_sound:
                        game_over_sound.play()
                else:
                    self.mario.x = 100
                    self.mario.y = self.height - 200
                    self.mario.velocity_y = 0
                    self.mario.jumping = False

    def draw(self, img):
        if background_img is not None:
            img[:] = background_img
        else:
            img[:] = (135, 206, 235)
        for platform in self.platforms:
            platform.draw(img)
        for coin in self.coins:
            coin.draw(img)
        self.mario.draw(img)
        cvzone.putTextRect(img, f'Score: {self.mario.score}', [20, 40], scale=1.5, thickness=2, offset=10)
        cvzone.putTextRect(img, f'Lives: {self.mario.lives}', [20, 80], scale=1.5, thickness=2, offset=10)
        if self.game_over:
            cvzone.putTextRect(img, "Game Over", [400, 300], scale=3, thickness=5, offset=20)
            cvzone.putTextRect(img, f'Final Score: {self.mario.score}', [400, 400], scale=3, thickness=5, offset=20)

def show_loading_screen(width, height, duration=3):
    """Display a loading screen with a progress bar."""
    loading_screen = np.zeros((height, width, 3), dtype=np.uint8)
    bar_width = int(width * 0.8)
    bar_height = 30
    bar_x = (width - bar_width) // 2
    bar_y = height // 2
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        loading_screen.fill(0)
        cv2.putText(loading_screen, "Loading...", (width//2 - 100, bar_y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(loading_screen, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        progress_width = int(bar_width * progress)
        cv2.rectangle(loading_screen, (bar_x, bar_y),
                     (bar_x + progress_width, bar_y + bar_height),
                     (0, 255, 0), -1)
        percentage = int(progress * 100)
        cv2.putText(loading_screen, f"{percentage}%",
                    (width//2 - 30, bar_y + bar_height + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Loading", loading_screen)
        cv2.waitKey(1)
    cv2.destroyWindow("Loading")

def main():
    parser = argparse.ArgumentParser(description='Mario Game with MediaPipe Hand Tracking')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    GAME_WIDTH = 1280
    GAME_HEIGHT = 720
    # Show loading screen before initializing the game
    show_loading_screen(GAME_WIDTH, GAME_HEIGHT, duration=3)
    game = MarioGame(GAME_WIDTH, GAME_HEIGHT)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    if not available_cameras:
        return
    if args.camera not in available_cameras:
        return
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, GAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GAME_HEIGHT)
    # Set up OpenTelemetry tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)
    # --- Pygame Setup ---
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
    pygame.display.set_caption("MediaPipe Mario (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    clock = pygame.time.Clock()
    prev_jump = False
    with tracer.start_as_current_span("mariomaster_session"):
        running = True
        while running:
            success, img = cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            hands, img = detector.findHands(img, flipType=False)
            # Gesture detection: open hand for movement, fist for jump
            jump_flag = False
            left_open = False
            right_open = False
            for hand in hands:
                fingers = detector.fingersUp(hand)
                if sum(fingers) == 0:
                    jump_flag = True
                else:
                    if hand["type"] == "Left":
                        left_open = True
                    elif hand["type"] == "Right":
                        right_open = True
            # Jump on fist (edge detection)
            if jump_flag:
                if not prev_jump:
                    with tracer.start_as_current_span("jump"):
                        game.mario.jump()
                prev_jump = True
            else:
                prev_jump = False
                # Movement: only if not jumping
                if right_open and not left_open:
                    with tracer.start_as_current_span("move_right"):
                        game.mario.move(1)
                elif left_open and not right_open:
                    with tracer.start_as_current_span("move_left"):
                        game.mario.move(-1)
                else:
                    game.mario.move(0)
            game.update()
            # Coin collection and game over logging
            for coin in game.coins:
                if coin.collected:
                    with tracer.start_as_current_span("coin_collected"):
                        pass
            if game.game_over:
                with tracer.start_as_current_span("game_over"):
                    pass
            game.draw(img)
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            game = MarioGame(GAME_WIDTH, GAME_HEIGHT)
                            prev_jump = False
            # --- Pygame Rendering ---
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
