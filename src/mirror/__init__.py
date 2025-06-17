import cv2
import mediapipe as mp

# Initialize FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Map emotions to emoji
emoji_map = {"Happy": "😊", "Surprised": "😲", "Neutral": "😐"}

# Emotion detection based on mouth landmarks
def detect_emotion(landmarks, img_w, img_h):
    left = landmarks.landmark[61]
    right = landmarks.landmark[291]
    top = landmarks.landmark[13]
    bottom = landmarks.landmark[14]
    mouth_w = (right.x - left.x) * img_w
    mouth_h = (bottom.y - top.y) * img_h
    if mouth_w > mouth_h * 2.0:
        return "Happy"
    elif mouth_h > mouth_w * 0.6:
        return "Surprised"
    return "Neutral"

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    emotion = "Neutral"
    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0]
        emotion = detect_emotion(mesh, w, h)
        # Draw face mesh landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame, mesh, mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1))

    # Overlay UI
    cv2.putText(frame, f"Emotion: {emotion} {emoji_map[emotion]}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("Emotion Mirror", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
