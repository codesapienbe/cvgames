import cv2
import mediapipe as mp
import numpy as np
import sys
import argparse
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView

class WebsiteDesigner(QMainWindow):
    def __init__(self, camera_index=0):
        super().__init__()
        self.setWindowTitle("Website Designer with Hand Tracking")
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # List available cameras
        available_cameras = []
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        if not available_cameras:
            print("Error: No cameras found!")
            sys.exit(1)
        
        print(f"Available cameras: {available_cameras}")
        
        if camera_index not in available_cameras:
            print(f"Error: Camera {camera_index} is not available!")
            print(f"Please select one of the available cameras: {available_cameras}")
            sys.exit(1)
        
        print(f"Attempting to open camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            print("Please check if:")
            print("1. The camera is properly connected")
            print("2. The camera is not being used by another application")
            print("3. You have the correct camera index")
            sys.exit(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Drawing variables
        self.drawing = False
        self.lines = []
        self.current_line = []
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left side - Camera and drawing
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Camera view
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        left_layout.addWidget(self.camera_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear Drawing")
        self.clear_button.clicked.connect(self.clear_drawing)
        controls_layout.addWidget(self.clear_button)
        left_layout.addLayout(controls_layout)
        
        layout.addWidget(left_widget)
        
        # Right side - HTML preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # HTML editor
        self.html_editor = QTextEdit()
        self.html_editor.setPlaceholderText("HTML will be generated here...")
        right_layout.addWidget(self.html_editor)
        
        # CSS editor
        self.css_editor = QTextEdit()
        self.css_editor.setPlaceholderText("CSS will be generated here...")
        right_layout.addWidget(self.css_editor)
        
        # Preview
        self.web_view = QWebEngineView()
        right_layout.addWidget(self.web_view)
        
        layout.addWidget(right_widget)
        
        # Set up timer for camera updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms = ~33fps
        
        # Initial HTML/CSS update
        self.update_preview()
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get index finger tip coordinates
                index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger.x * frame.shape[1])
                y = int(index_finger.y * frame.shape[0])
                
                # Get thumb tip coordinates for pinch detection
                thumb = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                thumb_x = int(thumb.x * frame.shape[1])
                thumb_y = int(thumb.y * frame.shape[0])
                
                # Calculate distance between thumb and index finger
                distance = np.sqrt((x - thumb_x)**2 + (y - thumb_y)**2)
                
                # Start/stop drawing based on pinch gesture
                if distance < 50:  # Pinch threshold
                    if not self.drawing:
                        self.drawing = True
                        self.current_line = [(x, y)]
                else:
                    if self.drawing:
                        self.drawing = False
                        if len(self.current_line) > 1:
                            self.lines.append(self.current_line)
                        self.current_line = []
                
                # Add points to current line while drawing
                if self.drawing:
                    self.current_line.append((x, y))
        
        # Draw all lines
        for line in self.lines:
            for i in range(len(line) - 1):
                cv2.line(frame, line[i], line[i + 1], (0, 255, 0), 2)
        
        # Draw current line
        if len(self.current_line) > 1:
            for i in range(len(self.current_line) - 1):
                cv2.line(frame, self.current_line[i], self.current_line[i + 1], (0, 255, 0), 2)
        
        # Convert frame to QImage and display
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_image))
        
        # Update HTML/CSS preview
        self.update_preview()
    
    def clear_drawing(self):
        self.lines = []
        self.current_line = []
        self.update_preview()
    
    def update_preview(self):
        # Generate HTML from lines
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
        """
        
        # Add CSS for lines
        for i, line in enumerate(self.lines):
            if len(line) > 1:
                points = [f"{x},{y}" for x, y in line]
                path = " ".join(points)
                html += f"""
                .line{i} {{
                    stroke: green;
                    stroke-width: 2;
                    fill: none;
                }}
                """
        
        html += """
            </style>
        </head>
        <body>
            <svg width="640" height="480" style="border: 1px solid black;">
        """
        
        # Add SVG paths for lines
        for i, line in enumerate(self.lines):
            if len(line) > 1:
                points = [f"{x},{y}" for x, y in line]
                path = " ".join(points)
                html += f'<path class="line{i}" d="M {path}"/>\n'
        
        html += """
            </svg>
        </body>
        </html>
        """
        
        # Update editors
        self.html_editor.setText(html)
        
        # Update preview
        self.web_view.setHtml(html)
    
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Website Designer with MediaPipe Hand Tracking')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = WebsiteDesigner(camera_index=args.camera)
    window.show()
    sys.exit(app.exec_()) 