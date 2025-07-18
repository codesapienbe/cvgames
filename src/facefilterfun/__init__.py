import cv2
import mediapipe as mp
import numpy as np
import random
import time

class FaceFilter:
    def __init__(self, name, effect_func):
        self.name = name
        self.effect_func = effect_func

class FaceFilterFun:
    def __init__(self):
        self.filters = [
            FaceFilter("Glasses", self.apply_glasses),
            FaceFilter("Hat", self.apply_hat),
            FaceFilter("Mustache", self.apply_mustache),
            FaceFilter("Rainbow", self.apply_rainbow),
            FaceFilter("Pixelate", self.apply_pixelate),
            FaceFilter("Blur", self.apply_blur),
            FaceFilter("Edge", self.apply_edge),
            FaceFilter("Cartoon", self.apply_cartoon)
        ]
        
        self.current_filter = 0
        self.filter_enabled = True
        self.last_switch_time = 0
        self.switch_cooldown = 2.0  # seconds
        self.score = 0
        self.game_time = 60  # seconds
        self.start_time = time.time()
        
    def apply_glasses(self, frame, face_landmarks):
        """Apply glasses filter"""
        if not face_landmarks:
            return frame
        
        # Get eye landmarks
        left_eye = face_landmarks.landmark[33]  # Left eye center
        right_eye = face_landmarks.landmark[263]  # Right eye center
        
        # Convert to pixel coordinates
        h, w = frame.shape[:2]
        left_eye_x = int(left_eye.x * w)
        left_eye_y = int(left_eye.y * h)
        right_eye_x = int(right_eye.x * w)
        right_eye_y = int(right_eye.y * h)
        
        # Calculate glasses dimensions
        eye_distance = abs(right_eye_x - left_eye_x)
        glasses_width = int(eye_distance * 2.5)
        glasses_height = int(glasses_width * 0.4)
        
        # Draw glasses
        glasses_color = (255, 0, 0)  # Blue glasses
        cv2.ellipse(frame, (left_eye_x, left_eye_y), (glasses_width//4, glasses_height//2), 
                   0, 0, 360, glasses_color, 3)
        cv2.ellipse(frame, (right_eye_x, right_eye_y), (glasses_width//4, glasses_height//2), 
                   0, 0, 360, glasses_color, 3)
        
        # Draw bridge
        cv2.line(frame, (left_eye_x + glasses_width//4, left_eye_y), 
                (right_eye_x - glasses_width//4, right_eye_y), glasses_color, 3)
        
        return frame
    
    def apply_hat(self, frame, face_landmarks):
        """Apply hat filter"""
        if not face_landmarks:
            return frame
        
        # Get forehead position
        nose = face_landmarks.landmark[1]  # Nose tip
        h, w = frame.shape[:2]
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)
        
        # Draw hat
        hat_color = (0, 255, 0)  # Green hat
        hat_width = 100
        hat_height = 60
        
        # Hat brim
        cv2.ellipse(frame, (nose_x, nose_y - 50), (hat_width//2, hat_height//3), 
                   0, 0, 180, hat_color, -1)
        
        # Hat top
        cv2.ellipse(frame, (nose_x, nose_y - 80), (hat_width//3, hat_height//2), 
                   0, 0, 360, hat_color, -1)
        
        return frame
    
    def apply_mustache(self, frame, face_landmarks):
        """Apply mustache filter"""
        if not face_landmarks:
            return frame
        
        # Get nose and mouth landmarks
        nose = face_landmarks.landmark[1]  # Nose tip
        upper_lip = face_landmarks.landmark[13]  # Upper lip
        
        h, w = frame.shape[:2]
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)
        lip_x = int(upper_lip.x * w)
        lip_y = int(upper_lip.y * h)
        
        # Draw mustache
        mustache_color = (0, 0, 255)  # Red mustache
        mustache_width = 80
        mustache_height = 20
        
        cv2.ellipse(frame, (lip_x, lip_y), (mustache_width//2, mustache_height//2), 
                   0, 0, 180, mustache_color, -1)
        
        return frame
    
    def apply_rainbow(self, frame, face_landmarks):
        """Apply rainbow effect to face"""
        if not face_landmarks:
            return frame
        
        # Create rainbow overlay
        h, w = frame.shape[:2]
        rainbow = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Generate rainbow colors
        for i in range(h):
            hue = int((i / h) * 180)  # 0-180 for OpenCV
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            rainbow[i, :] = color
        
        # Apply rainbow to face region
        face_region = self.get_face_region(face_landmarks, w, h)
        if face_region:
            x, y, width, height = face_region
            frame[y:y+height, x:x+width] = cv2.addWeighted(
                frame[y:y+height, x:x+width], 0.7, 
                rainbow[y:y+height, x:x+width], 0.3, 0
            )
        
        return frame
    
    def apply_pixelate(self, frame, face_landmarks):
        """Apply pixelation effect to face"""
        if not face_landmarks:
            return frame
        
        face_region = self.get_face_region(face_landmarks, frame.shape[1], frame.shape[0])
        if face_region:
            x, y, width, height = face_region
            
            # Extract face region
            face_roi = frame[y:y+height, x:x+width]
            
            # Pixelate
            small = cv2.resize(face_roi, (10, 10))
            pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Replace face region
            frame[y:y+height, x:x+width] = pixelated
        
        return frame
    
    def apply_blur(self, frame, face_landmarks):
        """Apply blur effect to face"""
        if not face_landmarks:
            return frame
        
        face_region = self.get_face_region(face_landmarks, frame.shape[1], frame.shape[0])
        if face_region:
            x, y, width, height = face_region
            
            # Extract face region
            face_roi = frame[y:y+height, x:x+width]
            
            # Apply blur
            blurred = cv2.GaussianBlur(face_roi, (25, 25), 0)
            
            # Replace face region
            frame[y:y+height, x:x+width] = blurred
        
        return frame
    
    def apply_edge(self, frame, face_landmarks):
        """Apply edge detection effect to face"""
        if not face_landmarks:
            return frame
        
        face_region = self.get_face_region(face_landmarks, frame.shape[1], frame.shape[0])
        if face_region:
            x, y, width, height = face_region
            
            # Extract face region
            face_roi = frame[y:y+height, x:x+width]
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Replace face region
            frame[y:y+height, x:x+width] = edges_colored
        
        return frame
    
    def apply_cartoon(self, frame, face_landmarks):
        """Apply cartoon effect to face"""
        if not face_landmarks:
            return frame
        
        face_region = self.get_face_region(face_landmarks, frame.shape[1], frame.shape[0])
        if face_region:
            x, y, width, height = face_region
            
            # Extract face region
            face_roi = frame[y:y+height, x:x+width]
            
            # Apply cartoon effect
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 9, 9)
            
            # Color quantization
            color = cv2.bilateralFilter(face_roi, 9, 300, 300)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            
            # Replace face region
            frame[y:y+height, x:x+width] = cartoon
        
        return frame
    
    def get_face_region(self, face_landmarks, width, height):
        """Get bounding box of face region"""
        if not face_landmarks:
            return None
        
        # Get face boundary landmarks
        x_coords = [landmark.x * width for landmark in face_landmarks.landmark]
        y_coords = [landmark.y * height for landmark in face_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 50
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def switch_filter(self):
        """Switch to next filter"""
        current_time = time.time()
        if current_time - self.last_switch_time >= self.switch_cooldown:
            self.current_filter = (self.current_filter + 1) % len(self.filters)
            self.last_switch_time = current_time
            self.score += 5
    
    def apply_current_filter(self, frame, face_landmarks):
        """Apply the current filter to the frame"""
        if self.filter_enabled and face_landmarks:
            filter_obj = self.filters[self.current_filter]
            frame = filter_obj.effect_func(frame, face_landmarks)
        return frame
    
    def draw_ui(self, frame):
        """Draw UI elements"""
        height, width = frame.shape[:2]
        
        # Draw background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Calculate remaining time
        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, self.game_time - elapsed_time)
        
        # Draw game info
        cv2.putText(frame, f"Filter: {self.filters[self.current_filter].name}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {self.score}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {int(remaining_time)}s", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(frame, "Wave your hand to change filters", (width//2 - 200, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Check if game is over
        if remaining_time <= 0:
            cv2.putText(frame, "Game Over!", (width//2 - 100, height//2 - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(frame, f"Final Score: {self.score}", 
                       (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'r' to restart or 'q' to quit", 
                       (width//2 - 200, height//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def detect_hand_wave(hand_landmarks, prev_hand_landmarks):
    """Detect hand wave gesture"""
    if not hand_landmarks or not prev_hand_landmarks:
        return False
    
    # Get wrist positions
    current_wrist = hand_landmarks.landmark[0]
    prev_wrist = prev_hand_landmarks.landmark[0]
    
    # Calculate horizontal movement
    movement = abs(current_wrist.x - prev_wrist.x)
    
    # Threshold for wave detection
    return movement > 0.1

def main():
    # Initialize MediaPipe Face Mesh and Hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize face filter fun
    filter_fun = FaceFilterFun()
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create fullscreen window
    cv2.namedWindow("Face Filter Fun", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Face Filter Fun", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Hand tracking variables
    prev_hand_landmarks = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face detection
        face_results = face_mesh.process(rgb)
        
        # Process hand detection
        hand_results = hands.process(rgb)
        
        # Apply face filter
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            frame = filter_fun.apply_current_filter(frame, face_landmarks)
        
        # Detect hand wave
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if prev_hand_landmarks:
                if detect_hand_wave(hand_landmarks, prev_hand_landmarks):
                    filter_fun.switch_filter()
            
            prev_hand_landmarks = hand_landmarks
        
        # Draw UI
        filter_fun.draw_ui(frame)
        
        cv2.imshow("Face Filter Fun", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            filter_fun = FaceFilterFun()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 