import cv2
import mediapipe as mp
import random
import time

# Initialize webcam and set resolution
tape = cv2.VideoCapture(0)
if not tape.isOpened():
    print("Error: Could not open webcam.")
    exit(1)
# Optional: set resolution to 640x480
width, height = 640, 480
cap = tape
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Game variables
ball_radius = 10
ball_x, ball_y = width // 2, height // 2
ball_dx = random.choice([-7, 7])
ball_dy = -7
paddle_width, paddle_height = 100, 10
paddle_y = height - 30
paddle_x = width // 2
score = 0

# Main game loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Hand detection for paddle control
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        x_norm = hand.landmark[8].x
        paddle_x = int(x_norm * width)

    # Update ball position
    ball_x += ball_dx
    ball_y += ball_dy

    # Collisions with walls
    if ball_x <= ball_radius or ball_x >= width - ball_radius:
        ball_dx = -ball_dx
    if ball_y <= ball_radius:
        ball_dy = -ball_dy

    # Collision with paddle
    if ball_y >= paddle_y - ball_radius and (paddle_x - paddle_width//2) <= ball_x <= (paddle_x + paddle_width//2):
        ball_dy = -abs(ball_dy)
        score += 1

    # Check for miss (game over)
    if ball_y > height:
        cv2.putText(frame, f"Game Over! Score: {score}", (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Pong Challenge", frame)
        cv2.waitKey(1500)
        # Reset game
        ball_x, ball_y = width//2, height//2
        ball_dx = random.choice([-7, 7])
        ball_dy = -7
        score = 0
        continue

    # Draw ball and paddle
    cv2.circle(frame, (ball_x, ball_y), ball_radius, (255,255,255), -1)
    cv2.rectangle(frame, (paddle_x - paddle_width//2, paddle_y), (paddle_x + paddle_width//2, paddle_y + paddle_height), (255,255,255), -1)

    # UI overlays
    cv2.putText(frame, f"Score: {score}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Move hand to control paddle, 'q' to quit", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.imshow("Pong Challenge", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
