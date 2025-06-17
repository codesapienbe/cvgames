import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize FaceMesh for smile detection
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Game parameters
COUNTDOWN_TIME = 3
GAME_DURATION = 30
SMILE_RATIO_THRESHOLD = 1.7   # mouth width/height ratio
SMILE_COOLDOWN = 1.0          # seconds between counts

# State variables
start_time = None
countdown_start = time.time()
smile_count = 0
last_smile_time = 0
prev_smile = False

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    now = time.time()
    elapsed_since_countdown = now - countdown_start

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    smile_detected = False

    if elapsed_since_countdown >= COUNTDOWN_TIME:
        # Game has started
        if start_time is None:
            start_time = now
        game_elapsed = now - start_time
        # Check smile if within game duration
        if game_elapsed < GAME_DURATION and results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0]
            # mouth landmarks
            lm = mesh.landmark
            left = lm[61]; right = lm[291]; top = lm[13]; bottom = lm[14]
            mouth_w = (right.x - left.x) * width
            mouth_h = (bottom.y - top.y) * height
            if mouth_h > 0 and (mouth_w / mouth_h) > SMILE_RATIO_THRESHOLD:
                smile_detected = True
            # Count smile
            if smile_detected and not prev_smile and now - last_smile_time > SMILE_COOLDOWN:
                smile_count += 1
                last_smile_time = now
                prev_smile = True
            if not smile_detected:
                prev_smile = False
        # Draw time left and score
        if game_elapsed < GAME_DURATION:
            time_left = int(GAME_DURATION - game_elapsed)
            cv2.putText(frame, f"Time Left: {time_left}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, f"Smiles: {smile_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            # Game over
            cv2.putText(frame, f"Time's up! Total Smiles: {smile_count}", (10, height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Smile Detector", frame)
            cv2.waitKey(3000)
            break
    else:
        # Countdown phase
        n = COUNTDOWN_TIME - int(elapsed_since_countdown)
        cv2.putText(frame, f"Starting in {n}", (width//2 - 100, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)

    # UI instructions
    cv2.putText(frame, "Smile wide to count!", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
    cv2.imshow("Smile Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
