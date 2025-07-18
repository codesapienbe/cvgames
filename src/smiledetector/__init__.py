import cv2
import mediapipe as mp
import numpy as np
import time
import random

class SmileDetector:
    def __init__(self):
        self.smile_score = 0
        self.max_smile_score = 100
        self.smile_threshold = 0.5
        self.game_mode = "calibration"  # calibration, game, results
        self.calibration_samples = []
        self.neutral_samples = []
        self.smile_samples = []
        self.current_emotion = "neutral"
        self.emotion_timer = 0
        self.emotion_duration = 3.0  # seconds to hold emotion
        self.score = 0
        self.rounds = 0
        self.max_rounds = 5
        self.target_emotion = "neutral"
        self.round_start_time = 0
        self.feedback = ""
        
    def calculate_smile_ratio(self, face_landmarks):
        """Calculate smile ratio using lip landmarks"""
        if not face_landmarks:
            return 0.0
        
        # Get lip landmarks
        upper_lip = face_landmarks.landmark[13]  # Upper lip
        lower_lip = face_landmarks.landmark[14]  # Lower lip
        left_corner = face_landmarks.landmark[61]  # Left lip corner
        right_corner = face_landmarks.landmark[291]  # Right lip corner
        
        # Calculate lip width and height
        lip_width = abs(right_corner.x - left_corner.x)
        lip_height = abs(upper_lip.y - lower_lip.y)
        
        # Calculate smile ratio (width/height ratio)
        if lip_height > 0:
            smile_ratio = lip_width / lip_height
        else:
            smile_ratio = 0.0
        
        return smile_ratio
    
    def detect_emotion(self, smile_ratio):
        """Detect emotion based on smile ratio"""
        if smile_ratio > self.smile_threshold:
            return "smile"
        else:
            return "neutral"
    
    def start_calibration(self):
        """Start calibration phase"""
        self.game_mode = "calibration"
        self.calibration_samples = []
        self.neutral_samples = []
        self.smile_samples = []
        self.feedback = "Please maintain a neutral expression for 3 seconds"
    
    def calibrate(self, smile_ratio):
        """Collect calibration samples"""
        if len(self.neutral_samples) < 30:  # 1 second at 30 fps
            self.neutral_samples.append(smile_ratio)
            self.feedback = f"Calibrating neutral: {len(self.neutral_samples)}/30"
        elif len(self.smile_samples) < 30:  # 1 second at 30 fps
            self.smile_samples.append(smile_ratio)
            self.feedback = f"Calibrating smile: {len(self.smile_samples)}/30"
        else:
            # Calculate thresholds from calibration data
            neutral_avg = np.mean(self.neutral_samples)
            smile_avg = np.mean(self.smile_samples)
            self.smile_threshold = (neutral_avg + smile_avg) / 2
            self.game_mode = "game"
            self.start_new_round()
    
    def start_new_round(self):
        """Start a new game round"""
        if self.rounds >= self.max_rounds:
            self.game_mode = "results"
            return
        
        self.rounds += 1
        self.target_emotion = random.choice(["neutral", "smile"])
        self.emotion_timer = 0
        self.round_start_time = time.time()
        self.feedback = f"Round {self.rounds}/{self.max_rounds}: Show {self.target_emotion}"
    
    def update_game(self, current_emotion):
        """Update game state"""
        if self.game_mode != "game":
            return
        
        current_time = time.time()
        
        if current_emotion == self.target_emotion:
            self.emotion_timer += 1/30  # Assuming 30 fps
            progress = min(1.0, self.emotion_timer / self.emotion_duration)
            self.feedback = f"Hold {self.target_emotion}: {progress:.1%}"
            
            if self.emotion_timer >= self.emotion_duration:
                self.score += 10
                self.feedback = f"Great! +10 points. Score: {self.score}"
                time.sleep(1)
                self.start_new_round()
        else:
            self.emotion_timer = 0
            self.feedback = f"Show {self.target_emotion}, not {current_emotion}"
    
    def draw_ui(self, frame):
        """Draw game UI"""
        height, width = frame.shape[:2]
        
        # Draw background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw game info
        y_offset = 30
        cv2.putText(frame, f"Mode: {self.game_mode.title()}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.game_mode == "calibration":
            cv2.putText(frame, self.feedback, (10, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif self.game_mode == "game":
            cv2.putText(frame, f"Score: {self.score}", (10, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, self.feedback, (10, y_offset + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif self.game_mode == "results":
            cv2.putText(frame, f"Final Score: {self.score}/{self.max_rounds * 10}", 
                       (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'r' to restart or 'q' to quit", 
                       (width//2 - 200, height//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw smile meter
        meter_width = 200
        meter_height = 20
        meter_x = width - meter_width - 20
        meter_y = 20
        
        # Background
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), 
                     (100, 100, 100), -1)
        
        # Smile level
        smile_width = int((self.smile_score / self.max_smile_score) * meter_width)
        color = (0, 255, 0) if self.smile_score > self.max_smile_score * 0.5 else (0, 255, 255)
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + smile_width, meter_y + meter_height), 
                     color, -1)
        
        # Border
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), 
                     (255, 255, 255), 2)
        
        cv2.putText(frame, f"Smile: {self.smile_score:.1f}", (meter_x, meter_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize smile detector
    detector = SmileDetector()
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create fullscreen window
    cv2.namedWindow("Smile Detector", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Smile Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Start calibration
    detector.start_calibration()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Draw face mesh
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            
            # Calculate smile ratio
            smile_ratio = detector.calculate_smile_ratio(face_landmarks)
            detector.smile_score = min(detector.max_smile_score, smile_ratio * 100)
            
            # Detect emotion
            current_emotion = detector.detect_emotion(smile_ratio)
            
            # Update game based on mode
            if detector.game_mode == "calibration":
                detector.calibrate(smile_ratio)
            elif detector.game_mode == "game":
                detector.update_game(current_emotion)
            
            # Draw emotion indicator
            emotion_color = (0, 255, 0) if current_emotion == "smile" else (0, 255, 255)
            cv2.putText(frame, f"Current: {current_emotion}", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        
        # Draw UI
        detector.draw_ui(frame)
        
        cv2.imshow("Smile Detector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector = SmileDetector()
            detector.start_calibration()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 