import cv2
import mediapipe as mp
import time
import datetime

# Initialize MediaPipe Face Mesh for smile detection
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Smile detection threshold parameters
SMILE_RATIO_THRESHOLD = 1.5  # mouth width to height ratio
COOLDOWN = 5  # seconds after capture before next

# State variables
countdown_start = None
captured = False
last_capture_time = 0

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    smile_detected = False
    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0]
        # mouth landmarks
        left = mesh.landmark[61]
        right = mesh.landmark[291]
        top = mesh.landmark[13]
        bottom = mesh.landmark[14]
        mouth_w = (right.x - left.x) * w
        mouth_h = (bottom.y - top.y) * h
        if mouth_h > 0 and (mouth_w / mouth_h) > SMILE_RATIO_THRESHOLD:
            smile_detected = True

    now = time.time()
    # Trigger countdown on smile if not in cooldown or countdown
    if smile_detected and not countdown_start and now - last_capture_time > COOLDOWN:
        countdown_start = now

    # Countdown handling
    if countdown_start:
        elapsed = now - countdown_start
        if elapsed < 3:
            # show countdown number
            n = 3 - int(elapsed)
            cv2.putText(frame, str(n), (w//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 10)
        else:
            # capture selfie
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"selfie_{timestamp}.png"
            cv2.imwrite(filename, frame)
            last_capture_time = now
            countdown_start = None
            captured = True

    # Show captured message briefly
    if captured:
        if now - last_capture_time < 2:
            cv2.putText(frame, f"Saved {filename}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            captured = False

    # UI
    cv2.putText(frame, "Smile to start selfie!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

    cv2.imshow("Selfie Fun", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()