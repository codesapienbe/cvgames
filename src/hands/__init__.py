import cv2
import mediapipe as mp
import argparse
import sys
import signal
import pygame
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
import numpy as np

# Standard argument parsing
parser = argparse.ArgumentParser(description='MediaPipe Module')
parser.add_argument('--camera', type=int, default=0, 
                   help='Camera index to use (default: 0)')
parser.add_argument('--min_detection_confidence', type=float, default=0.5,
                   help='Minimum detection confidence (default: 0.5)')
args = parser.parse_args()

# Standard camera initialization
def init_camera(camera_index):
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            available_cameras = []
            for i in range(10):
                temp_cap = cv2.VideoCapture(i)
                if temp_cap.isOpened():
                    available_cameras.append(i)
                    temp_cap.release()
            
            if available_cameras:
                print(f"Available cameras: {available_cameras}")
                print(f"Run with: --camera [index]")
            else:
                print("No cameras found")
            sys.exit(1)
        return cap
    except Exception as e:
        print(f"Camera error: {e}")
        sys.exit(1)

# Standard cleanup function
def cleanup(cap):
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Application closed gracefully")

# Signal handler for Ctrl+C
def signal_handler(sig, frame):
    print("\nInterrupted by user")
    cleanup(cap)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --------------------------
# MODULE-SPECIFIC CODE BELOW
# --------------------------

# Initialize MediaPipe solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def main():
    parser = argparse.ArgumentParser(description='MediaPipe Module')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    parser.add_argument('--min_detection_confidence', type=float, default=0.5, help='Minimum detection confidence (default: 0.5)')
    args = parser.parse_args()

    # Set up OpenTelemetry tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)

    # Camera setup
    cap = init_camera(args.camera)
    print(f"Using camera index: {args.camera}")
    print("Press 'q' or ESC to exit")
    ret, image = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        sys.exit(1)
    height, width = image.shape[:2]
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("MediaPipe Hands (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    with mp_hands.Hands(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=0.5
    ) as hands:
        with tracer.start_as_current_span("hands_session"):
            running = True
            while running:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                image.flags.writeable = True
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                hand_found = False
                if results.multi_hand_landmarks:
                    hand_found = True
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                # --- Pygame Event Handling ---
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            running = False
                # --- OpenTelemetry logging ---
                if hand_found:
                    with tracer.start_as_current_span("hand_detected"):
                        pass
                else:
                    with tracer.start_as_current_span("no_hand_detected"):
                        pass
                # --- Pygame Rendering ---
                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                clock.tick(30)
    cleanup(cap)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

# --------------------------
# MODULE-SPECIFIC CODE ABOVE
# --------------------------
