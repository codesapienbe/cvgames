import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui

class VirtualKeyboard:
    def __init__(self):
        self.text = ""
        self.cursor_position = 0
        self.selected_key = None
        self.selection_time = 0
        self.selection_threshold = 1.5  # seconds to hold selection
        self.last_key_press = 0
        self.key_cooldown = 0.5  # seconds between key presses
        
        # Keyboard layout
        self.keys = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', '<-', ' '],
            ['123', 'ABC', 'Done', 'Clear']
        ]
        
        self.key_width = 60
        self.key_height = 60
        self.key_spacing = 10
        self.keyboard_x = 50
        self.keyboard_y = 300
        
    def get_key_at_position(self, x, y):
        """Get the key at the given screen position"""
        for row_idx, row in enumerate(self.keys):
            for col_idx, key in enumerate(row):
                key_x = self.keyboard_x + col_idx * (self.key_width + self.key_spacing)
                key_y = self.keyboard_y + row_idx * (self.key_height + self.key_spacing)
                
                if (key_x <= x <= key_x + self.key_width and 
                    key_y <= y <= key_y + self.key_height):
                    return key, row_idx, col_idx
        return None, None, None
    
    def handle_key_press(self, key):
        """Handle key press action"""
        current_time = time.time()
        if current_time - self.last_key_press < self.key_cooldown:
            return
        
        if key == '<-':
            if self.text and self.cursor_position > 0:
                self.text = self.text[:self.cursor_position-1] + self.text[self.cursor_position:]
                self.cursor_position -= 1
        elif key == 'Clear':
            self.text = ""
            self.cursor_position = 0
        elif key == 'Done':
            # Simulate pressing Enter
            pyautogui.press('enter')
        elif key == '123':
            # Toggle to numbers (simplified)
            pass
        elif key == 'ABC':
            # Toggle to letters (simplified)
            pass
        elif key == ' ':
            self.text = self.text[:self.cursor_position] + ' ' + self.text[self.cursor_position:]
            self.cursor_position += 1
        elif len(key) == 1:
            # Regular character
            self.text = self.text[:self.cursor_position] + key + self.text[self.cursor_position:]
            self.cursor_position += 1
        
        self.last_key_press = current_time
    
    def draw(self, frame):
        """Draw the virtual keyboard"""
        height, width = frame.shape[:2]
        
        # Draw text area
        text_area_y = 100
        cv2.rectangle(frame, (50, text_area_y), (width - 50, text_area_y + 100), 
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (50, text_area_y), (width - 50, text_area_y + 100), 
                     (255, 255, 255), 2)
        
        # Draw text
        cv2.putText(frame, self.text, (60, text_area_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw cursor
        cursor_x = 60 + len(self.text[:self.cursor_position]) * 20
        cv2.line(frame, (cursor_x, text_area_y + 20), (cursor_x, text_area_y + 80), 
                (0, 255, 0), 3)
        
        # Draw keyboard
        for row_idx, row in enumerate(self.keys):
            for col_idx, key in enumerate(row):
                key_x = self.keyboard_x + col_idx * (self.key_width + self.key_spacing)
                key_y = self.keyboard_y + row_idx * (self.key_height + self.key_spacing)
                
                # Determine key color
                if self.selected_key == key:
                    color = (0, 255, 0)  # Green for selected
                elif key in ['<-', 'Clear', 'Done']:
                    color = (100, 100, 255)  # Blue for special keys
                elif key in ['123', 'ABC']:
                    color = (255, 100, 100)  # Red for mode keys
                else:
                    color = (100, 100, 100)  # Gray for regular keys
                
                # Draw key background
                cv2.rectangle(frame, (key_x, key_y), 
                             (key_x + self.key_width, key_y + self.key_height), 
                             color, -1)
                cv2.rectangle(frame, (key_x, key_y), 
                             (key_x + self.key_width, key_y + self.key_height), 
                             (255, 255, 255), 2)
                
                # Draw key text
                text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = key_x + (self.key_width - text_size[0]) // 2
                text_y = key_y + (self.key_height + text_size[1]) // 2
                cv2.putText(frame, key, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(frame, "Point your finger at keys to type", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Hold for 1.5 seconds to press a key", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def detect_finger_position(hand_landmarks, frame_width, frame_height):
    """Detect index finger position"""
    if not hand_landmarks:
        return None
    
    # Get index finger tip position
    index_tip = hand_landmarks.landmark[8]
    
    # Convert to pixel coordinates
    x = int(index_tip.x * frame_width)
    y = int(index_tip.y * frame_height)
    
    return (x, y)

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize virtual keyboard
    keyboard = VirtualKeyboard()
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create fullscreen window
    cv2.namedWindow("Virtual Keyboard", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Virtual Keyboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
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
            
            # Detect finger position
            finger_pos = detect_finger_position(hand_landmarks, frame_width, frame_height)
            
            if finger_pos:
                x, y = finger_pos
                
                # Draw finger cursor
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
                
                # Check if finger is over a key
                key, row_idx, col_idx = keyboard.get_key_at_position(x, y)
                
                if key:
                    # Check if this is a new selection
                    if key != keyboard.selected_key:
                        keyboard.selection_time = current_time
                        keyboard.selected_key = key
                    
                    # Check if selection has been held long enough
                    if current_time - keyboard.selection_time >= keyboard.selection_threshold:
                        keyboard.handle_key_press(key)
                        keyboard.selection_time = current_time
                    
                    # Draw selection progress
                    progress = min(1.0, (current_time - keyboard.selection_time) / keyboard.selection_threshold)
                    key_x = keyboard.keyboard_x + col_idx * (keyboard.key_width + keyboard.key_spacing)
                    key_y = keyboard.keyboard_y + row_idx * (keyboard.key_height + keyboard.key_spacing)
                    
                    # Draw progress ring
                    center_x = key_x + keyboard.key_width // 2
                    center_y = key_y + keyboard.key_height // 2
                    radius = 25
                    cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 2)
                    cv2.ellipse(frame, (center_x, center_y), (radius, radius), -90, 0, 360 * progress, (0, 255, 0), 3)
                else:
                    keyboard.selected_key = None
        
        # Draw keyboard
        keyboard.draw(frame)
        
        cv2.imshow("Virtual Keyboard", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            keyboard.text = ""
            keyboard.cursor_position = 0
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 