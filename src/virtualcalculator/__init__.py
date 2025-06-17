import cv2
import mediapipe as mp
import random
import time

# Gesture classification: count extended fingers (0–5)
def classify_number(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    count = 0
    for tip_id in tip_ids:
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[tip_id - 2]
        if tip_id == 4:
            # thumb: check horizontal extension
            if abs(tip.x - pip.x) > 0.03:
                count += 1
        else:
            # other fingers: tip above pip
            if tip.y < pip.y:
                count += 1
    return count

# Generate a new math problem

def generate_question():
    global a, b, op, answer, question_text
    a = random.randint(0, 5)
    b = random.randint(0, 5)
    op = random.choice(['+', '-'])
    if op == '-' and b > a:
        a, b = b, a
    answer = a + b if op == '+' else a - b
    question_text = f"{a} {op} {b} = ?"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Game state
score = 0
state = 'question'
feedback = ''
cooldown = 2.0
last_time = time.time()

generate_question()

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    now = time.time()

    # Process hand landmarks
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    number = None
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        number = classify_number(hand)
        # On question state and cooldown elapsed
        if state == 'question' and now - last_time > cooldown:
            last_time = now
            if number == answer:
                feedback = 'Correct!'
                score += 1
            else:
                feedback = f'Wrong! Ans: {answer}'
            state = 'feedback'

    # After feedback, generate next question
    if state == 'feedback' and now - last_time > 1.5:
        generate_question()
        feedback = ''
        state = 'question'

    # Display UI
    cv2.putText(frame, question_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if state == 'feedback':
        color = (0, 255, 0) if feedback.startswith('Correct') else (0, 0, 255)
        cv2.putText(frame, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Score: {score}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, "Show number gesture 0-5", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Virtual Calculator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
