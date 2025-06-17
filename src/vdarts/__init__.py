import cv2
import mediapipe as mp
import numpy as np
import math

def draw_dartboard(size=600):
    board = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size//2, size//2)
    for i in range(10, 0, -1):
        radius = int(size/2 * i/10)
        color = (0,0,0) if i%2 else (255,255,255)
        cv2.circle(board, center, radius, color, -1)
    cv2.circle(board, center, int(size/20), (0,0,255), -1)
    return board

def get_score(point, size=600):
    center = np.array([size/2, size/2])
    dist = np.linalg.norm(point - center)
    ring = int(dist / (size/2/10))
    return 10 - ring if 0 <= ring < 10 else 0

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    size = 600
    board = draw_dartboard(size)
    score = 0
    throws = 0
    thrown = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("Virtual Darts", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Virtual Darts", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame,1)
        img = cv2.resize(frame, (size,size))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        display = board.copy()
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                x = int(handLms.landmark[8].x * size)
                y = int(handLms.landmark[8].y * size)
                cv2.circle(display, (x,y), 10, (0,255,0), -1)
                mp_draw.draw_landmarks(display, handLms, mp_hands.HAND_CONNECTIONS)

                thumb = handLms.landmark[4]
                index = handLms.landmark[8]
                dist_thumb_index = math.hypot((thumb.x-index.x)*size, (thumb.y-index.y)*size)
                if dist_thumb_index < 30 and not thrown:
                    score += get_score(np.array([x,y]), size)
                    throws += 1
                    thrown = True
                if dist_thumb_index > 40:
                    thrown = False

        cv2.putText(display, f"Score: {score}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(display, f"Throws: {throws}", (10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.imshow("Virtual Darts", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
