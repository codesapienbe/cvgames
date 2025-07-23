import cv2
import mediapipe as mp
import time
import math
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
CENTER = (WIDTH // 2, int(HEIGHT * 0.8))
PLANK_LENGTH = int(WIDTH * 0.6)
G = 500
BALL_RADIUS = 15

mp_pose = mp.solutions.pose

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Circus Performer (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    small_font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    score = 0.0
    ball_pos = 0.0
    ball_vel = 0.0
    achievements = {10: False, 30: False, 60: False}
    achievement_text = ""
    achievement_time = 0
    prev_time = time.time()
    with tracer.start_as_current_span("circus_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            angle = 0.0
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                dx = rs.x - ls.x
                dy = rs.y - ls.y
                angle = math.atan2(dy, dx)
            # Physics update
            acc = G * math.sin(angle)
            ball_vel += acc * dt
            ball_pos += ball_vel * dt
            half_len = PLANK_LENGTH / 2
            # Check for failure
            if abs(ball_pos) > half_len:
                with tracer.start_as_current_span("game_over"):
                    pass
                gameover_text = font.render(f"Game Over! Score: {int(score)}", True, (255,0,0))
                screen.fill((0,0,0))
                screen.blit(gameover_text, (50, HEIGHT // 2))
                pygame.display.flip()
                pygame.time.wait(2000)
                # reset game
                score = 0.0
                ball_pos = 0.0
                ball_vel = 0.0
                achievements = {k: False for k in achievements}
                achievement_text = ""
                prev_time = time.time()
                continue
            # Update score
            score += dt
            # Achievement check
            for thresh, done in achievements.items():
                if score >= thresh and not done:
                    achievements[thresh] = True
                    achievement_text = f"Achievement: {thresh}s balanced!"
                    achievement_time = current_time
                    with tracer.start_as_current_span(f"achievement_{thresh}s"):
                        pass
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            screen.fill((30, 30, 30))
            # Draw plank
            x1 = int(CENTER[0] - half_len * math.cos(angle))
            y1 = int(CENTER[1] - half_len * math.sin(angle))
            x2 = int(CENTER[0] + half_len * math.cos(angle))
            y2 = int(CENTER[1] + half_len * math.sin(angle))
            pygame.draw.line(screen, (255,255,255), (x1, y1), (x2, y2), 5)
            # Draw ball
            bx = int(CENTER[0] + ball_pos * math.cos(angle))
            by = int(CENTER[1] + ball_pos * math.sin(angle))
            pygame.draw.circle(screen, (255,0,0), (bx, by), BALL_RADIUS)
            # UI overlays
            score_text = font.render(f"Score: {int(score)}", True, (0,255,0))
            screen.blit(score_text, (10, 30))
            instr_text = small_font.render("Press 'q' to quit", True, (255,255,255))
            screen.blit(instr_text, (10, HEIGHT - 40))
            if achievement_text and (current_time - achievement_time) < 2:
                ach_text = font.render(achievement_text, True, (255, 215, 0))
                screen.blit(ach_text, (50, 60))
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
