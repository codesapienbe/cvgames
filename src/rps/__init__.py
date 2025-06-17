import cv2
import mediapipe as mp
import random
import time

def classify_gesture(hand_landmarks):
    """Classify hand gesture as rock, paper, or scissors based on extended fingers."""
    tip_ids = [8, 12, 16, 20]
    count = 0
    for tip in tip_ids:
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[tip - 2].y
        if tip_y < pip_y:
            count += 1
    if count == 0:
        return "rock"
    elif count == 2:
        return "scissors"
    elif count == 4:
        return "paper"
    else:
        return None

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("Rock Paper Scissors AI", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Rock Paper Scissors AI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_gesture = None
    cooldown = False
    result_text = ""
    gesture_time = 0
    choices = ["rock", "paper", "scissors"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        gesture = None
        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]
            gesture = classify_gesture(handLms)
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        if gesture and gesture != prev_gesture and not cooldown:
            ai_choice = random.choice(choices)
            if gesture == ai_choice:
                result_text = f"Draw! Both {gesture}"
            elif (gesture == "rock" and ai_choice == "scissors") or \
                 (gesture == "paper" and ai_choice == "rock") or \
                 (gesture == "scissors" and ai_choice == "paper"):
                result_text = f"You win! {gesture} beats {ai_choice}"
            else:
                result_text = f"You lose! {ai_choice} beats {gesture}"
            prev_gesture = gesture
            cooldown = True
            gesture_time = time.time()

        if cooldown and time.time() - gesture_time > 2:
            cooldown = False
            prev_gesture = None
            result_text = ""

        cv2.putText(frame, f"Your: {gesture or '-'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{result_text or ''}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Rock Paper Scissors AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
