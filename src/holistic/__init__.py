import cv2
import mediapipe as mp
import numpy as np
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    # Set up OpenTelemetry tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)
    # For webcam input:
    cap = cv2.VideoCapture(0)
    ret, image = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        return
    height, width = image.shape[:2]
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("MediaPipe Holistic (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        with tracer.start_as_current_span("holistic_session"):
            running = True
            while running:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                image.flags.writeable = True
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                found = False
                # Draw landmark annotation on the image.
                if results.face_landmarks:
                    found = True
                    mp_drawing.draw_landmarks(
                        image,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                if results.pose_landmarks:
                    found = True
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # --- OpenTelemetry logging ---
                if found:
                    with tracer.start_as_current_span("landmarks_detected"):
                        pass
                else:
                    with tracer.start_as_current_span("no_landmarks"):
                        pass
                # --- Pygame Event Handling ---
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            running = False
                # --- Pygame Rendering ---
                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                clock.tick(30)
    cap.release()
    pygame.quit()
    print("Application closed")
    sys.exit()

if __name__ == "__main__":
    main()