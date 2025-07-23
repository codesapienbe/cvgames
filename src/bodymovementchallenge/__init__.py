import cv2
import mediapipe as mp
import numpy as np
import random
import time
import pygame
import sys
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

CHALLENGES = [
    "Raise both hands above your head!",
    "Touch your toes!",
    "Stand on one leg!",
    "Jump in place!",
    "Stretch your arms wide!"
]

class BodyMovementChallenge:
    def __init__(self):
        self.score = 0
        self.current_challenge = None
        self.challenge_start = 0
        self.challenge_duration = 5
        self.completed = False
        self.next_challenge()

    def next_challenge(self):
        self.current_challenge = random.choice(CHALLENGES)
        self.challenge_start = time.time()
        self.completed = False
        with tracer.start_as_current_span("new_challenge"):
            pass

    def check_pose(self, pose_landmarks):
        if not pose_landmarks:
            return False
        if "hands above" in self.current_challenge:
            left = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            right = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            head = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
            return left.y < head.y and right.y < head.y
        return random.random() < 0.1

    def update(self, pose_landmarks):
        if not self.completed and self.check_pose(pose_landmarks):
            self.score += 10
            self.completed = True
            with tracer.start_as_current_span("challenge_completed"):
                pass

    def draw(self, surface, font, small_font):
        surface.fill((30, 30, 30))
        challenge_text = font.render(f"Challenge: {self.current_challenge}", True, (255, 255, 0))
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        surface.blit(score_text, (50, 50))
        surface.blit(challenge_text, (50, 100))
        if self.completed:
            success_text = font.render("Success!", True, (0, 255, 0))
            surface.blit(success_text, (50, 200))
        else:
            try_text = font.render("Try to complete the challenge!", True, (255, 255, 255))
            surface.blit(try_text, (50, 200))
        instr_text = small_font.render("Press n for next challenge, q to quit", True, (255, 255, 255))
        surface.blit(instr_text, (50, 400))


def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    game = BodyMovementChallenge()
    cap = cv2.VideoCapture(0)
    WIDTH, HEIGHT = 960, 720
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Body Movement Challenge (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    small_font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()
    with tracer.start_as_current_span("bodymovement_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            pose_landmarks = results.pose_landmarks if results.pose_landmarks else None
            if pose_landmarks:
                mp_draw.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
            game.update(pose_landmarks)
            # Convert OpenCV frame to Pygame surface for background (optional)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            screen.blit(frame_surface, (0, 0))
            # Draw overlay UI
            game.draw(screen, font, small_font)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_n:
                        game.next_challenge()
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 