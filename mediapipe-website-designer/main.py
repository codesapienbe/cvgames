import cv2
import mediapipe as mp
import numpy as np
import sys
import argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTextEdit, QFrame)
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtWebEngineWidgets import QWebEngineView
import math

class WebElement:
    def __init__(self, element_type, points, properties=None):
        self.element_type = element_type
        self.points = points
        self.properties = properties or {}
        
    def get_bounds(self):
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        return {
            'x': min(x_coords),
            'y': min(y_coords),
            'width': max(x_coords) - min(x_coords),
            'height': max(y_coords) - min(y_coords)
        }

class ComponentPalette(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(100)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Create component buttons
        self.container_btn = self.create_component_button("Container", "#4CAF50")
        self.button_btn = self.create_component_button("Button", "#2196F3")
        self.textfield_btn = self.create_component_button("Text Field", "#FF9800")
        
        layout.addWidget(self.container_btn)
        layout.addWidget(self.button_btn)
        layout.addWidget(self.textfield_btn)
        layout.addStretch()
        
        # Store button positions
        self.button_positions = {}
    
    def create_component_button(self, text, color):
        btn = QPushButton(text)
        btn.setFixedSize(120, 60)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
        """)
        return btn

    def get_button_at_position(self, x, y):
        # Convert global coordinates to local coordinates
        local_pos = self.mapFromGlobal(QPoint(x, y))
        
        # Check each button
        if self.container_btn.geometry().contains(local_pos):
            return "container"
        elif self.button_btn.geometry().contains(local_pos):
            return "button"
        elif self.textfield_btn.geometry().contains(local_pos):
            return "textfield"
        return None

class DesignCanvas(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(640, 480)
        self.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.elements = []
        self.drawing = False
        self.current_points = []
        self.hand_overlay = None
        self.hand_position = None
        self.current_gesture = None
        
        # Initialize painter for drawing
        self.painter = QPainter()
    
    def set_hand_overlay(self, image):
        self.hand_overlay = image
        self.update()
    
    def update_hand_position(self, x, y, gesture):
        self.hand_position = (x, y)
        self.current_gesture = gesture
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        self.painter.begin(self)
        
        # Draw existing elements
        for element in self.elements:
            self.draw_element(element)
        
        # Draw current drawing
        if self.drawing and self.current_points:
            self.draw_current_path()
        
        # Draw hand tracking overlay if available
        if self.hand_overlay:
            self.painter.drawImage(0, 0, self.hand_overlay)
        
        # Draw hand cursor if available
        if self.hand_position:
            x, y = self.hand_position
            self.draw_hand_cursor(x, y)
        
        self.painter.end()
    
    def draw_element(self, element):
        points = element.points
        bounds = element.get_bounds()
        
        # Set color based on element type
        if element.element_type == "container":
            color = QColor("#4CAF50")
        elif element.element_type == "button":
            color = QColor("#2196F3")
        else:  # textfield
            color = QColor("#FF9800")
        
        # Draw rectangle
        self.painter.setPen(QPen(color, 2))
        self.painter.setBrush(color.lighter(150))
        self.painter.drawRect(bounds['x'], bounds['y'], 
                            bounds['width'], bounds['height'])
        
        # Draw label
        self.painter.setPen(Qt.black)
        self.painter.drawText(bounds['x'] + 5, bounds['y'] + 20, 
                            element.element_type)
    
    def draw_current_path(self):
        if len(self.current_points) > 1:
            self.painter.setPen(QPen(Qt.green, 2))
            for i in range(len(self.current_points) - 1):
                self.painter.drawLine(
                    self.current_points[i][0], self.current_points[i][1],
                    self.current_points[i + 1][0], self.current_points[i + 1][1]
                )
    
    def draw_hand_cursor(self, x, y):
        # Draw a circle at hand position
        self.painter.setPen(QPen(QColor(0, 0, 255), 2))  # Blue color
        self.painter.setBrush(QColor(0, 0, 255, 150))  # Semi-transparent blue
        self.painter.drawEllipse(x - 10, y - 10, 20, 20)
        
        # Draw gesture indicator
        if self.current_gesture:
            color = {
                "container": QColor("#4CAF50"),
                "button": QColor("#2196F3"),
                "textfield": QColor("#FF9800")
            }.get(self.current_gesture, QColor(0, 0, 255))
            
            self.painter.setPen(QPen(color, 2))
            self.painter.setBrush(color.lighter(150))
            self.painter.drawText(x + 15, y + 5, self.current_gesture)

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
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            sys.exit(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left side - Design area
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Design canvas
        self.canvas = DesignCanvas()
        left_layout.addWidget(self.canvas)
        
        # Component palette
        self.palette = ComponentPalette()
        left_layout.addWidget(self.palette)
        
        # Connect palette buttons
        self.palette.container_btn.clicked.connect(lambda: self.select_component("container"))
        self.palette.button_btn.clicked.connect(lambda: self.select_component("button"))
        self.palette.textfield_btn.clicked.connect(lambda: self.select_component("textfield"))
        
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
    
    def select_component(self, component_type):
        self.canvas.selected_component = component_type
    
    def detect_gesture(self, hand_landmarks):
        # Get finger tip and pip landmarks
        finger_tips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        ]
        
        finger_pips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        ]
        
        # Check if fingers are extended
        fingers_extended = []
        for tip, pip in zip(finger_tips, finger_pips):
            # For thumb, check horizontal position
            if tip == finger_tips[0]:  # thumb
                fingers_extended.append(tip.x > pip.x)
            else:  # other fingers
                fingers_extended.append(tip.y < pip.y)
        
        # Detect gestures
        if all(fingers_extended):  # All fingers open
            return "container"
        elif not any(fingers_extended):  # All fingers closed
            return "button"
        elif fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[3:]):  # Index and middle fingers open
            return "textfield"
        return None
    
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
                
                # Detect gesture
                gesture = self.detect_gesture(hand_landmarks)
                
                # Update hand position in canvas
                self.canvas.update_hand_position(x, y, gesture)
                
                # Handle drawing
                if gesture:
                    if not self.canvas.drawing:
                        self.canvas.drawing = True
                        self.canvas.current_points = [(x, y)]
                    else:
                        self.canvas.current_points.append((x, y))
                else:
                    if self.canvas.drawing:
                        self.canvas.drawing = False
                        if len(self.canvas.current_points) > 1:
                            # Create element from drawn path
                            self.canvas.elements.append(WebElement(
                                gesture or "container",  # Default to container if no gesture
                                self.canvas.current_points
                            ))
                        self.canvas.current_points = []
        
        # Convert frame to QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Create a transparent overlay for the hand tracking
        overlay = QImage(width, height, QImage.Format_ARGB32)
        overlay.fill(Qt.transparent)
        painter = QPainter(overlay)
        painter.drawImage(0, 0, q_image)
        painter.end()
        
        # Update the canvas with the hand tracking overlay
        self.canvas.set_hand_overlay(overlay)
        
        # Update HTML/CSS preview
        self.update_preview()
    
    def update_preview(self):
        # Generate HTML
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    margin: 0;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                }
                .container {
                    border: 1px solid #4CAF50;
                    padding: 20px;
                    margin: 10px 0;
                    background-color: #E8F5E9;
                }
                .button {
                    background-color: #2196F3;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }
                .button:hover {
                    background-color: #1976D2;
                }
                .textbox {
                    width: 100%;
                    padding: 10px;
                    margin: 10px 0;
                    border: 1px solid #FF9800;
                    border-radius: 4px;
                    background-color: #FFF3E0;
                }
        """
        
        # Add element-specific CSS
        for i, element in enumerate(self.canvas.elements):
            bounds = element.get_bounds()
            css_class = f"element-{i}"
            html += f"""
                .{css_class} {{
                    position: absolute;
                    left: {bounds['x']}px;
                    top: {bounds['y']}px;
                    width: {bounds['width']}px;
                    height: {bounds['height']}px;
                }}
            """
        
        html += """
            </style>
        </head>
        <body>
        """
        
        # Add elements to HTML
        for i, element in enumerate(self.canvas.elements):
            css_class = f"element-{i}"
            if element.element_type == "button":
                html += f'<button class="button {css_class}">Button</button>\n'
            elif element.element_type == "textfield":
                html += f'<input type="text" class="textbox {css_class}" placeholder="Enter text...">\n'
            else:  # container
                html += f'<div class="container {css_class}">Container</div>\n'
        
        html += """
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