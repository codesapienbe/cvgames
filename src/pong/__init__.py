import cv2
import mediapipe as mp
import random

def main():
    width, height = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("Virtual Pong", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Virtual Pong", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    # Game variables
    ball_radius = 10
    ball_x, ball_y = width // 2, height // 2
    ball_dx = random.choice([-5, 5])
    ball_dy = random.choice([-5, 5])
    paddle_width, paddle_height = 100, 10
    paddle_y = height - 30
    paddle_x = width // 2
    score = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Detect hand and update paddle position
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            x_norm = hand.landmark[8].x
            paddle_x = int(width * x_norm)

        # Update ball
        ball_x += ball_dx
        ball_y += ball_dy

        # Collisions with walls
        if ball_x <= ball_radius or ball_x >= width - ball_radius:
            ball_dx = -ball_dx
        if ball_y <= ball_radius:
            ball_dy = -ball_dy

        # Collision with paddle
        if ball_y >= paddle_y - ball_radius and paddle_x - paddle_width // 2 <= ball_x <= paddle_x + paddle_width // 2:
            ball_dy = -ball_dy
            score += 1

        # Missed paddle: reset game
        if ball_y > height:
            cv2.putText(frame, f"Game Over! Score: {score}", (50, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Virtual Pong", frame)
            cv2.waitKey(1000)
            ball_x, ball_y = width // 2, height // 2
            ball_dx = random.choice([-5, 5])
            ball_dy = random.choice([-5, 5])
            score = 0
            continue

        # Draw elements
        cv2.circle(frame, (ball_x, ball_y), ball_radius, (255, 255, 255), -1)
        cv2.rectangle(frame, (paddle_x - paddle_width // 2, paddle_y),
                      (paddle_x + paddle_width // 2, paddle_y + paddle_height), (255, 255, 255), -1)
        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Virtual Pong", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
