import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

WIDTH, HEIGHT = 800, 600
BG_COLOR = (240, 230, 200)
PERFORMER_COLOR = (100, 100, 255)
TRICK_COLOR = (255, 200, 0)

mp_pose = mp.solutions.pose

class Performer:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT - 120
        self.radius = 40
        self.trick_active = False
        self.trick_time = 0
        self.score = 0
    def do_trick(self):
        self.trick_active = True
        self.trick_time = time.time()
        self.score += 1
        with tracer.start_as_current_span("trick_performed"):
            pass
    def update(self):
        if self.trick_active and time.time() - self.trick_time > 1.0:
            self.trick_active = False
    def draw(self, surface, font):
        # Draw performer (simple circle)
        pygame.draw.circle(surface, PERFORMER_COLOR, (self.x, self.y), self.radius)
        # Draw arms up if trick active
        if self.trick_active:
            pygame.draw.line(surface, TRICK_COLOR, (self.x, self.y), (self.x-60, self.y-80), 8)
            pygame.draw.line(surface, TRICK_COLOR, (self.x, self.y), (self.x+60, self.y-80), 8)
        else:
            pygame.draw.line(surface, (80,80,80), (self.x, self.y), (self.x-60, self.y-20), 8)
            pygame.draw.line(surface, (80,80,80), (self.x, self.y), (self.x+60, self.y-20), 8)
        # Draw score
        score_text = font.render(f"Score: {self.score}", True, (0,0,0))
        surface.blit(score_text, (30, 30))
        instr = font.render("Raise both hands to perform a trick!", True, (0,0,0))
        surface.blit(instr, (30, HEIGHT - 50))

def hands_above_head(pose_landmarks):
    if not pose_landmarks:
        return False
    left = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    head = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    return left.y < head.y and right.y < head.y

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Circus Performer (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 32)
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    performer = Performer()
    with tracer.start_as_current_span("circusperformer_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            pose_landmarks = results.pose_landmarks if results.pose_landmarks else None
            # Detect trick
            if hands_above_head(pose_landmarks) and not performer.trick_active:
                performer.do_trick()
            performer.update()
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            screen.fill(BG_COLOR)
            performer.draw(screen, font)
            pygame.display.flip()
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 