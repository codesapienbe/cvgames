import cv2
import mediapipe as mp
import argparse
import sys
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
import numpy as np

def main():
    # Default camera and confidence
    camera_index = 0
    min_detection_confidence = 0.5
    # Set up OpenTelemetry tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        sys.exit(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("MediaPipe Face Mesh (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5) as face_mesh:
        with tracer.start_as_current_span("face_mesh_session"):
            running = True
            while running:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                image.flags.writeable = True
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                face_found = False
                if results.multi_face_landmarks:
                    face_found = True
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                # Add info text
                cv2.putText(image, f"Camera: {camera_index} | Press 'q' or ESC to exit", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if face_found:
                    with tracer.start_as_current_span("face_detected"):
                        pass
                else:
                    with tracer.start_as_current_span("no_face_detected"):
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