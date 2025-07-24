import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import sys
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

def apply_filter(frame, filter_name):
    if filter_name == "grayscale":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_name == "sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(frame, kernel)
        sepia = np.clip(sepia, 0, 255)
        return sepia.astype(np.uint8)
    elif filter_name == "cartoon":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    return frame

def main():
    # Set up OpenTelemetry tracing
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)

    # MediaPipe setup
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)

    # Camera setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        sys.exit(1)
    height, width = frame.shape[:2]

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Selfie Fun (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 36)
    clock = pygame.time.Clock()

    # Game state
    filters = ["none", "grayscale", "sepia", "cartoon"]
    filter_index = 0
    last_filter_time = 0
    filter_cooldown = 1.0
    photo_taken = False
    photo_time = 0
    game_time = 60  # seconds
    start_time = time.time()
    with tracer.start_as_current_span("selfiefun_session"):
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_detection.process(rgb)
            hand_results = hands.process(rgb)
            now = time.time()
            # Detect hand gesture for filter switch (open palm) or photo (fist)
            gesture = None
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                dist = np.linalg.norm(np.array([thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y]))
                if dist > 0.4:
                    gesture = "open"
                elif dist < 0.15:
                    gesture = "fist"
            # Switch filter
            if gesture == "open" and now - last_filter_time > filter_cooldown:
                filter_index = (filter_index + 1) % len(filters)
                last_filter_time = now
                with tracer.start_as_current_span("filter_applied"):
                    pass
            # Take photo
            if gesture == "fist" and not photo_taken:
                photo_taken = True
                photo_time = now
                cv2.imwrite("selfie.png", frame)
                with tracer.start_as_current_span("photo_taken"):
                    pass
            if photo_taken and now - photo_time > 2:
                photo_taken = False
            # Apply filter
            filter_name = filters[filter_index]
            filtered = apply_filter(frame, filter_name)
            if filter_name == "grayscale":
                filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
            frame_disp = filtered.copy()
            # Draw face box
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x1 = int(bboxC.xmin * width)
                    y1 = int(bboxC.ymin * height)
                    w = int(bboxC.width * width)
                    h = int(bboxC.height * height)
                    cv2.rectangle(frame_disp, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            # Draw UI
            elapsed = int(now - start_time)
            remains = max(0, game_time - elapsed)
            cv2.putText(frame_disp, f"Filter: {filter_name}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
            cv2.putText(frame_disp, f"Time: {remains}s", (width-220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
            if photo_taken:
                cv2.putText(frame_disp, "Photo Taken!", (width//2-150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
            cv2.putText(frame_disp, "Open palm to switch filter, fist to take photo", (80, height-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        with tracer.start_as_current_span("restart_game"):
                            filter_index = 0
                            last_filter_time = 0
                            photo_taken = False
                            photo_time = 0
                            start_time = time.time()
            # --- Pygame Rendering ---
            frame_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
            if remains <= 0:
                with tracer.start_as_current_span("game_over"):
                    running = False
        cap.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
