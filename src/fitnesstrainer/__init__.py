import cv2
import mediapipe as mp
import numpy as np
import time
import math

class Exercise:
    def __init__(self, name, target_reps, description):
        self.name = name
        self.target_reps = target_reps
        self.description = description
        self.current_reps = 0
        self.completed = False
        self.form_score = 0
        self.feedback = ""

class FitnessTrainer:
    def __init__(self):
        self.exercises = [
            Exercise("Jumping Jacks", 10, "Jump and spread arms and legs"),
            Exercise("Squats", 15, "Bend knees and lower body"),
            Exercise("Push-ups", 8, "Lower body to ground and push up"),
            Exercise("Arm Circles", 20, "Rotate arms in circular motion"),
            Exercise("Lunges", 12, "Step forward and lower back knee")
        ]
        
        self.current_exercise = 0
        self.workout_complete = False
        self.total_score = 0
        self.workout_start_time = time.time()
        
    def get_current_exercise(self):
        if self.current_exercise < len(self.exercises):
            return self.exercises[self.current_exercise]
        return None
    
    def next_exercise(self):
        self.current_exercise += 1
        if self.current_exercise >= len(self.exercises):
            self.workout_complete = True
    
    def update_exercise(self, pose_landmarks):
        """Update current exercise based on pose detection"""
        if self.workout_complete:
            return
        
        exercise = self.get_current_exercise()
        if not exercise:
            return
        
        if exercise.name == "Jumping Jacks":
            self.detect_jumping_jacks(pose_landmarks, exercise)
        elif exercise.name == "Squats":
            self.detect_squats(pose_landmarks, exercise)
        elif exercise.name == "Push-ups":
            self.detect_pushups(pose_landmarks, exercise)
        elif exercise.name == "Arm Circles":
            self.detect_arm_circles(pose_landmarks, exercise)
        elif exercise.name == "Lunges":
            self.detect_lunges(pose_landmarks, exercise)
    
    def detect_jumping_jacks(self, landmarks, exercise):
        """Detect jumping jack motion"""
        if not landmarks:
            return
        
        # Get key points
        left_shoulder = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        left_ankle = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
        
        # Calculate shoulder and hip widths
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hip_width = abs(left_hip.x - right_hip.x)
        
        # Check if arms are spread (shoulders wider than hips)
        if shoulder_width > hip_width * 1.2:
            # Check if legs are spread (ankles wider than hips)
            ankle_width = abs(left_ankle.x - right_ankle.x)
            if ankle_width > hip_width * 1.2:
                exercise.current_reps += 1
                exercise.feedback = "Great form! Keep it up!"
                if exercise.current_reps >= exercise.target_reps:
                    exercise.completed = True
                    self.total_score += 10
                    self.next_exercise()
            else:
                exercise.feedback = "Spread your legs wider"
        else:
            exercise.feedback = "Raise your arms higher"
    
    def detect_squats(self, landmarks, exercise):
        """Detect squat motion"""
        if not landmarks:
            return
        
        # Get key points
        left_hip = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
        
        # Calculate knee angle
        hip_to_knee = np.array([left_hip.x - left_knee.x, left_hip.y - left_knee.y])
        knee_to_ankle = np.array([left_knee.x - left_ankle.x, left_knee.y - left_ankle.y])
        
        # Calculate angle between vectors
        dot_product = np.dot(hip_to_knee, knee_to_ankle)
        hip_to_knee_norm = np.linalg.norm(hip_to_knee)
        knee_to_ankle_norm = np.linalg.norm(knee_to_ankle)
        
        if hip_to_knee_norm > 0 and knee_to_ankle_norm > 0:
            cos_angle = dot_product / (hip_to_knee_norm * knee_to_ankle_norm)
            angle = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
            
            # Detect squat (knee angle less than 120 degrees)
            if angle < 120:
                exercise.current_reps += 1
                exercise.feedback = "Good squat! Lower if possible"
                if exercise.current_reps >= exercise.target_reps:
                    exercise.completed = True
                    self.total_score += 10
                    self.next_exercise()
            else:
                exercise.feedback = "Bend your knees more"
    
    def detect_pushups(self, landmarks, exercise):
        """Detect push-up motion"""
        if not landmarks:
            return
        
        # Get key points
        left_shoulder = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        
        # Calculate arm angle
        shoulder_to_elbow = np.array([left_shoulder.x - left_elbow.x, left_shoulder.y - left_elbow.y])
        elbow_to_wrist = np.array([left_elbow.x - left_wrist.x, left_elbow.y - left_wrist.y])
        
        dot_product = np.dot(shoulder_to_elbow, elbow_to_wrist)
        shoulder_to_elbow_norm = np.linalg.norm(shoulder_to_elbow)
        elbow_to_wrist_norm = np.linalg.norm(elbow_to_wrist)
        
        if shoulder_to_elbow_norm > 0 and elbow_to_wrist_norm > 0:
            cos_angle = dot_product / (shoulder_to_elbow_norm * elbow_to_wrist_norm)
            angle = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
            
            # Detect push-up (arm angle less than 90 degrees)
            if angle < 90:
                exercise.current_reps += 1
                exercise.feedback = "Excellent push-up form!"
                if exercise.current_reps >= exercise.target_reps:
                    exercise.completed = True
                    self.total_score += 10
                    self.next_exercise()
            else:
                exercise.feedback = "Lower your body more"
    
    def detect_arm_circles(self, landmarks, exercise):
        """Detect arm circle motion"""
        if not landmarks:
            return
        
        # Get wrist positions
        left_wrist = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        
        # Simple detection: check if wrists are moving in circular pattern
        # This is a simplified version - in practice, you'd track movement over time
        wrist_height = (left_wrist.y + right_wrist.y) / 2
        
        # Count circles based on wrist height changes
        if hasattr(self, 'last_wrist_height'):
            if wrist_height < self.last_wrist_height - 0.1:  # Wrists moving up
                exercise.current_reps += 1
                exercise.feedback = "Keep the circles going!"
                if exercise.current_reps >= exercise.target_reps:
                    exercise.completed = True
                    self.total_score += 10
                    self.next_exercise()
        
        self.last_wrist_height = wrist_height
        exercise.feedback = "Make big arm circles"
    
    def detect_lunges(self, landmarks, exercise):
        """Detect lunge motion"""
        if not landmarks:
            return
        
        # Get key points
        left_hip = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
        right_hip = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
        
        # Calculate knee angles
        left_hip_to_knee = np.array([left_hip.x - left_knee.x, left_hip.y - left_knee.y])
        left_knee_to_ankle = np.array([left_knee.x - left_ankle.x, left_knee.y - left_ankle.y])
        
        right_hip_to_knee = np.array([right_hip.x - right_knee.x, right_hip.y - right_knee.y])
        right_knee_to_ankle = np.array([right_knee.x - right_ankle.x, right_knee.y - right_ankle.y])
        
        # Calculate angles
        left_angle = self.calculate_angle(left_hip_to_knee, left_knee_to_ankle)
        right_angle = self.calculate_angle(right_hip_to_knee, right_knee_to_ankle)
        
        # Detect lunge (one knee bent significantly)
        if left_angle < 100 or right_angle < 100:
            exercise.current_reps += 1
            exercise.feedback = "Great lunge! Keep alternating"
            if exercise.current_reps >= exercise.target_reps:
                exercise.completed = True
                self.total_score += 10
                self.next_exercise()
        else:
            exercise.feedback = "Step forward and bend your knee"
    
    def calculate_angle(self, v1, v2):
        """Calculate angle between two vectors"""
        dot_product = np.dot(v1, v2)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            cos_angle = dot_product / (v1_norm * v2_norm)
            return math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
        return 180
    
    def draw_ui(self, frame):
        """Draw fitness trainer UI"""
        height, width = frame.shape[:2]
        
        # Draw background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        if not self.workout_complete:
            exercise = self.get_current_exercise()
            if exercise:
                # Draw exercise info
                cv2.putText(frame, f"Exercise: {exercise.name}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Reps: {exercise.current_reps}/{exercise.target_reps}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Score: {self.total_score}", (20, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Draw description
                cv2.putText(frame, exercise.description, (width//2 - 200, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw feedback
                cv2.putText(frame, exercise.feedback, (width//2 - 200, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Draw progress bar
                progress = exercise.current_reps / exercise.target_reps
                bar_width = 400
                bar_height = 20
                bar_x = width//2 - bar_width//2
                bar_y = 120
                
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             (100, 100, 100), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), 
                             (0, 255, 0), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             (255, 255, 255), 2)
        else:
            # Draw workout complete screen
            workout_time = int(time.time() - self.workout_start_time)
            cv2.putText(frame, "Workout Complete!", (width//2 - 200, height//2 - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(frame, f"Final Score: {self.total_score}", 
                       (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {workout_time} seconds", 
                       (width//2 - 150, height//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'r' to restart or 'q' to quit", 
                       (width//2 - 200, height//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize fitness trainer
    trainer = FitnessTrainer()
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create fullscreen window
    cv2.namedWindow("Fitness Trainer", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Fitness Trainer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Update exercise detection
            trainer.update_exercise(results.pose_landmarks)
        
        # Draw UI
        trainer.draw_ui(frame)
        
        cv2.imshow("Fitness Trainer", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            trainer = FitnessTrainer()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 