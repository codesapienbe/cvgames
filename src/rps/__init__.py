import cv2
import mediapipe as mp
import random
import time
import tkinter as tk

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

    # Get screen resolution and set capture properties
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
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

    # Scoreboard variables
    wins = 0
    losses = 0
    draws = 0
    games_played = 0
    streak = 0  # positive for win streak, negative for loss streak
    max_win_streak = 0
    max_loss_streak = 0

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

            # Update scoreboard
            games_played += 1
            if result_text.startswith("Draw!"):
                draws += 1
                streak = 0
            elif result_text.startswith("You win!"):
                wins += 1
                if streak >= 0:
                    streak += 1
                else:
                    streak = 1
                if streak > max_win_streak:
                    max_win_streak = streak
            else:
                losses += 1
                if streak <= 0:
                    streak -= 1
                else:
                    streak = -1
                if abs(streak) > max_loss_streak:
                    max_loss_streak = abs(streak)

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

        # Display scoreboard and stats
        cv2.putText(frame, f"Wins: {wins}  Losses: {losses}  Draws: {draws}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        win_rate = int(wins / games_played * 100) if games_played > 0 else 0
        cv2.putText(frame, f"Games: {games_played}  Win Rate: {win_rate}%", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        streak_color = (0, 255, 0) if streak > 0 else (0, 0, 255) if streak < 0 else (255, 255, 255)
        cv2.putText(frame, f"Streak: {streak}", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, streak_color, 2)
        cv2.putText(frame, f"Max Win Streak: {max_win_streak}  Max Loss Streak: {max_loss_streak}", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

        cv2.imshow("Rock Paper Scissors AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
