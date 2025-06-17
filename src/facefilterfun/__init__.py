import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    # Apply pixelation filter to each detected face
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            # Clamp coordinates
            x, y = max(0, x), max(0, y)
            w, h = max(1, w), max(1, h)
            if x + w > width: w = width - x
            if y + h > height: h = height - y
            face = frame[y:y+h, x:x+w]
            # Pixelate
            small = cv2.resize(face, (w//15 or 1, h//15 or 1), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+w] = pixelated

    # UI
    cv2.putText(frame, "Face Filter Fun - Pixelate your face!", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("Face Filter Fun", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
