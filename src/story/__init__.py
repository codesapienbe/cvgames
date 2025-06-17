import cv2
import mediapipe as mp
import time
import numpy as np

story = [
    "Welcome to the Eye Blink Story!",
    "Progress through the story by blinking.",
    "Once upon a time, there was a brave adventurer.",
    "He journeyed through shadowy forests.",
    "He crossed raging rivers and climbed steep mountains.",
    "A fearsome dragon blocked his path. Blink to face it!",
    "With courage and wit, he defeated the dragon.",
    "The kingdom was saved. The End!"
]

# Threshold for blink detection (distance between eyelids)
BLINK_THRESHOLD = 0.02

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# State variables
prev_blink = False
blink_cooldown = False
last_blink_time = 0
story_index = 0

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Create fullscreen window
cv2.namedWindow("Eye Blink Story", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Eye Blink Story", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        blink = False
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            # Upper and lower eyelid landmarks
            upper = face.landmark[159]
            lower = face.landmark[145]
            dist = abs(upper.y - lower.y)
            blink = dist < BLINK_THRESHOLD

        # On blink detected
        if blink and not prev_blink and not blink_cooldown:
            story_index = min(story_index + 1, len(story) - 1)
            blink_cooldown = True
            last_blink_time = time.time()

        prev_blink = blink
        # Cooldown to avoid multiple triggers
        if blink_cooldown and time.time() - last_blink_time > 1:
            blink_cooldown = False

        # Display current story text
        cv2.putText(frame, story[story_index], (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Blink to continue, 'q' to quit.", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Eye Blink Story", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
