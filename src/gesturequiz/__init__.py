import cv2
import mediapipe as mp
import numpy as np
import random
import time

class GestureQuiz:
    def __init__(self):
        self.questions = [
            {
                "question": "What is 2 + 2?",
                "options": ["3", "4", "5", "6"],
                "correct": 1
            },
            {
                "question": "Which planet is closest to the Sun?",
                "options": ["Venus", "Mercury", "Earth", "Mars"],
                "correct": 1
            },
            {
                "question": "What is the capital of France?",
                "options": ["London", "Berlin", "Paris", "Madrid"],
                "correct": 2
            },
            {
                "question": "How many fingers do you have on one hand?",
                "options": ["3", "4", "5", "6"],
                "correct": 2
            },
            {
                "question": "What color is the sky on a clear day?",
                "options": ["Red", "Green", "Blue", "Yellow"],
                "correct": 2
            }
        ]
        
        self.current_question = 0
        self.score = 0
        self.game_over = False
        self.selection_time = 0
        self.selection_threshold = 2.0  # seconds to hold gesture
        self.last_selection = None
        self.feedback = ""
        self.feedback_timer = 0
        self.feedback_duration = 2.0  # seconds to show feedback
        
    def get_current_question(self):
        if self.current_question < len(self.questions):
            return self.questions[self.current_question]
        return None
    
    def check_answer(self, selected_option):
        """Check if the selected answer is correct"""
        question = self.get_current_question()
        if question and selected_option == question["correct"]:
            self.score += 1
            self.feedback = "Correct! +1 point"
        else:
            self.feedback = f"Wrong! Correct answer was: {question['options'][question['correct']]}"
        
        self.feedback_timer = time.time()
        self.current_question += 1
        
        if self.current_question >= len(self.questions):
            self.game_over = True
    
    def reset_game(self):
        self.current_question = 0
        self.score = 0
        self.game_over = False
        self.selection_time = 0
        self.last_selection = None
        self.feedback = ""
        self.feedback_timer = 0

def detect_finger_count(hand_landmarks):
    """Detect number of extended fingers"""
    if not hand_landmarks:
        return None
    
    # Finger tip landmarks
    tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    
    fingers = []
    for tip_id in tip_ids:
        if tip_id == 4:  # Thumb
            # Check if thumb is extended (x position)
            if hand_landmarks.landmark[tip_id].x < hand_landmarks.landmark[tip_id - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # Check if finger is extended (y position)
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
    
    # Count extended fingers
    finger_count = sum(fingers)
    
    # Map finger count to option (1-4)
    if finger_count == 1:
        return 0  # Option A
    elif finger_count == 2:
        return 1  # Option B
    elif finger_count == 3:
        return 2  # Option C
    elif finger_count == 4:
        return 3  # Option D
    
    return None

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize quiz
    quiz = GestureQuiz()
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create fullscreen window
    cv2.namedWindow("Gesture Quiz", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gesture Quiz", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        current_time = time.time()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detect finger count
            selected_option = detect_finger_count(hand_landmarks)
            
            if selected_option is not None and not quiz.game_over:
                # Check if this is a new selection
                if selected_option != quiz.last_selection:
                    quiz.selection_time = current_time
                    quiz.last_selection = selected_option
                
                # Check if selection has been held long enough
                if current_time - quiz.selection_time >= quiz.selection_threshold:
                    quiz.check_answer(selected_option)
                    quiz.selection_time = current_time
                
                # Draw selection indicator
                option_letters = ["A", "B", "C", "D"]
                cv2.putText(frame, f"Selected: {option_letters[selected_option]}", 
                           (frame_width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw progress bar
                progress = min(1.0, (current_time - quiz.selection_time) / quiz.selection_threshold)
                bar_width = 200
                bar_height = 20
                bar_x = frame_width - bar_width - 20
                bar_y = 80
                
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             (100, 100, 100), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), 
                             (0, 255, 0), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             (255, 255, 255), 2)
        
        # Draw quiz interface
        if not quiz.game_over:
            question = quiz.get_current_question()
            if question:
                # Draw question
                cv2.putText(frame, f"Question {quiz.current_question + 1}/{len(quiz.questions)}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Score: {quiz.score}", (50, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Draw question text
                cv2.putText(frame, question["question"], (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Draw options
                option_letters = ["A", "B", "C", "D"]
                for i, (option, letter) in enumerate(zip(question["options"], option_letters)):
                    y_pos = 220 + i * 60
                    color = (0, 255, 0) if quiz.last_selection == i else (255, 255, 255)
                    cv2.putText(frame, f"{letter}. {option}", (50, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Draw instructions
                cv2.putText(frame, "Show finger count to select answer:", (50, frame_height - 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "1 finger = A, 2 fingers = B, 3 fingers = C, 4 fingers = D", 
                           (50, frame_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Hold gesture for 2 seconds to confirm", 
                           (50, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Draw results
            cv2.putText(frame, "Quiz Complete!", (frame_width//2 - 150, frame_height//2 - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(frame, f"Final Score: {quiz.score}/{len(quiz.questions)}", 
                       (frame_width//2 - 150, frame_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            percentage = (quiz.score / len(quiz.questions)) * 100
            if percentage >= 80:
                grade = "Excellent!"
                color = (0, 255, 0)
            elif percentage >= 60:
                grade = "Good!"
                color = (0, 255, 255)
            else:
                grade = "Try Again!"
                color = (0, 0, 255)
            
            cv2.putText(frame, grade, (frame_width//2 - 100, frame_height//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
            cv2.putText(frame, "Press 'r' to restart or 'q' to quit", 
                       (frame_width//2 - 200, frame_height//2 + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw feedback
        if quiz.feedback and current_time - quiz.feedback_timer < quiz.feedback_duration:
            cv2.putText(frame, quiz.feedback, (frame_width//2 - 150, frame_height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Gesture Quiz", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            quiz.reset_game()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 