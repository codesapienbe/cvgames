import cv2
import mediapipe as mp
import time
import random
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

WIDTH, HEIGHT = 960, 720
BG_COLOR = (30, 30, 40)

moves = ["hands_up", "hands_side", "touch_toe"]

# Function to check if a move is performed
def check_move(move, landmarks):
    try:
        lw = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        rw = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
        ls = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        la = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
    except Exception:
        return False
    if move == "hands_up":
        return lw.y < ls.y and rw.y < rs.y
    elif move == "hands_side":
        return abs(lw.y - ls.y) < 0.1 and abs(rw.y - rs.y) < 0.1 and lw.x < ls.x and rw.x > rs.x
    elif move == "touch_toe":
        return lw.y > lh.y and abs(lw.y - la.y) < 0.1 and rw.y > lh.y and abs(rw.y - la.y) < 0.1
    return False

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dance Battle (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    small_font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()
    score = 0
    current_move = random.choice(moves)
    match_start = 0
    matched = False
    with tracer.start_as_current_span("dancebattle_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks
                if check_move(current_move, lm):
                    if not matched:
                        score += 1
                        matched = True
                        match_start = time.time()
                        with tracer.start_as_current_span("move_matched"):
                            pass
                else:
                    matched = False
            # After holding a successful match, pick next move
            if matched and (time.time() - match_start) > 2:
                current_move = random.choice(moves)
                matched = False
                match_start = time.time()
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            screen.fill(BG_COLOR)
            move_text = font.render(f"Move: {current_move}", True, (0,255,0))
            score_text = font.render(f"Score: {score}", True, (0,255,0))
            screen.blit(move_text, (10, 30))
            screen.blit(score_text, (10, 70))
            instr = small_font.render("Press 'q' to quit", True, (255,255,255))
            screen.blit(instr, (10, HEIGHT - 40))
            pygame.display.flip()
            # --- Keyboard quit ---
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                running = False
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 