import cv2
import mediapipe as mp
import time
import winsound

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Get frame dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Main loop runs on import
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pitch = 0
        play_tone = False
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            # Compute horizontal position of index fingertip
            x_norm = hand.landmark[8].x
            pitch = int(200 + x_norm * 800)  # map to 200-1000Hz
            # Count extended fingers
            tips = [8, 12, 16, 20]
            count = 0
            for tip in tips:
                if hand.landmark[tip].y < hand.landmark[tip - 2].y:
                    count += 1
            # Open palm (>=4 fingers) plays tone
            if count >= 4:
                play_tone = True

        # Play tone if open palm
        if play_tone and pitch > 0:
            winsound.Beep(pitch, 50)  # play 50ms tone

        # UI overlays
        cv2.putText(frame, f"Frequency: {pitch} Hz", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Open palm to play", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Sound Conductor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
