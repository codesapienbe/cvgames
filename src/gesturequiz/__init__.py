import cv2
import mediapipe as mp
import time

# Questions set (multiple choice 4 options)
questions = [
    {"q": "Capital of France?", "opts": ["Paris", "London", "Berlin", "Rome"], "ans": 1},
    {"q": "2 + 3 = ?",       "opts": ["4", "5", "6", "7"],          "ans": 2},
    {"q": "Color of sky?",    "opts": ["Green", "Blue", "Red", "Yellow"], "ans": 2},
]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Finger count function (excluding thumb)
tip_ids = [8, 12, 16, 20]
def count_fingers(hand_landmarks):
    count = 0
    for tip in tip_ids:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
            count += 1
    return count

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Game state
score = 0
q_index = 0
state = 'question'
feedback = ''
last_time = time.time()
cooldown = 1.5  # seconds between input

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    now = time.time()

    # Draw question and options
    q = questions[q_index]
    cv2.putText(frame, f"Q{q_index+1}: {q['q']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    for i, opt in enumerate(q['opts'], start=1):
        cv2.putText(frame, f"{i}) {opt}", (10, 30 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

    number = 0
    if state == 'question' and now - last_time > cooldown:
        # Hand processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            number = count_fingers(hand)
            if 1 <= number <= len(q['opts']):
                # evaluate
                if number == q['ans']:
                    feedback = 'Correct!'
                    score += 1
                else:
                    correct = q['opts'][q['ans']-1]
                    feedback = f"Wrong! Ans: {correct}"
                state = 'feedback'
                last_time = now

    elif state == 'feedback' and now - last_time > cooldown:
        # Next question
        q_index = (q_index + 1) % len(questions)
        state = 'question'
        feedback = ''
        last_time = now

    # Display feedback or score
    if state == 'feedback':
        color = (0,255,0) if feedback.startswith('Correct') else (0,0,255)
        cv2.putText(frame, feedback, (10, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Score: {score}", (width-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    # Instructions
    cv2.putText(frame, "Show 1-4 fingers to answer, 'q' to quit", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

    cv2.imshow("Gesture Quiz", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
