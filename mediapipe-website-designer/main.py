import cv2
import mediapipe as mp
import numpy as np
import sys
import argparse
from cvzone.HandTrackingModule import HandDetector
import cvzone

class WebElement:
    def __init__(self, element_type, x, y, width, height):
        self.element_type = element_type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = {
            "button": (255, 0, 0),      # Blue
            "textfield": (0, 255, 0),   # Green
            "label": (0, 0, 255),       # Red
            "div": (255, 255, 0)        # Yellow
        }.get(element_type, (255, 255, 255))

    def draw(self, img):
        # Draw the element with its color
        cv2.rectangle(img, 
                     (int(self.x - self.width//2), int(self.y - self.height//2)),
                     (int(self.x + self.width//2), int(self.y + self.height//2)),
                     self.color, -1)
        # Add a white border
        cv2.rectangle(img, 
                     (int(self.x - self.width//2), int(self.y - self.height//2)),
                     (int(self.x + self.width//2), int(self.y + self.height//2)),
                     (255, 255, 255), 2)
        # Add text label
        cvzone.putTextRect(img, self.element_type,
                          [int(self.x - self.width//2), int(self.y - self.height//2)],
                          scale=0.5, thickness=1, offset=5)

class WebsiteDesigner:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.elements = []
        self.selected_element = None
        self.dragging = False
        self.drag_start_pos = None
        
        # Component palette dimensions
        self.palette_height = 150
        self.palette_y = height - self.palette_height
        self.palette_width = width
        self.component_width = 120
        self.component_height = 80
        self.component_spacing = 40
        
        # Initialize component types
        self.component_types = ["button", "textfield", "label", "div"]
        self.component_positions = self.calculate_component_positions()
        self.hovered_component = None

    def calculate_component_positions(self):
        positions = {}
        # Calculate total width of all components
        total_width = len(self.component_types) * (self.component_width + self.component_spacing) - self.component_spacing
        # Calculate starting x position to center the components
        start_x = (self.width - total_width) // 2
        
        for i, comp_type in enumerate(self.component_types):
            x = start_x + i * (self.component_width + self.component_spacing)
            y = self.palette_y + self.palette_height//2
            positions[comp_type] = (x, y)
        return positions

    def draw_palette(self, img):
        # Draw palette background
        cv2.rectangle(img, 
                     (0, self.palette_y),
                     (self.width, self.height),
                     (50, 50, 50), -1)
        
        # Draw component buttons
        for comp_type, (x, y) in self.component_positions.items():
            # Calculate component size based on hover state
            width = self.component_width
            height = self.component_height
            if comp_type == self.hovered_component:
                width = int(width * 1.5)
                height = int(height * 1.5)
            
            # Draw component button with hover effect
            color = (150, 150, 150) if comp_type == self.hovered_component else (100, 100, 100)
            cv2.rectangle(img,
                         (x - width//2, y - height//2),
                         (x + width//2, y + height//2),
                         color, -1)
            # Add white border
            cv2.rectangle(img,
                         (x - width//2, y - height//2),
                         (x + width//2, y + height//2),
                         (255, 255, 255), 2)
            # Add text label
            cvzone.putTextRect(img, comp_type,
                             [x - width//2, y - height//2],
                             scale=0.7, thickness=2, offset=10)

    def draw(self, img):
        # Draw existing elements
        for element in self.elements:
            element.draw(img)
        
        # Draw palette
        self.draw_palette(img)
        
        # Draw selected element if dragging
        if self.dragging and self.selected_element:
            self.selected_element.draw(img)

    def get_component_at_position(self, x, y):
        self.hovered_component = None
        for comp_type, (comp_x, comp_y) in self.component_positions.items():
            # Use larger hit box for hover detection
            hover_width = int(self.component_width * 1.5)
            hover_height = int(self.component_height * 1.5)
            if (abs(x - comp_x) < hover_width//2 and
                abs(y - comp_y) < hover_height//2):
                self.hovered_component = comp_type
                # Use smaller hit box for actual selection
                if (abs(x - comp_x) < self.component_width//2 and
                    abs(y - comp_y) < self.component_height//2):
                    return comp_type
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Website Designer with MediaPipe Hand Tracking')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    args = parser.parse_args()

    # Set HD resolution
    GAME_WIDTH = 1280
    GAME_HEIGHT = 720

    # Initialize designer
    designer = WebsiteDesigner(GAME_WIDTH, GAME_HEIGHT)

    # List available cameras
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        print("Error: No cameras found!")
        return
    
    print(f"Available cameras: {available_cameras}")
    
    if args.camera not in available_cameras:
        print(f"Error: Camera {args.camera} is not available!")
        print(f"Please select one of the available cameras: {available_cameras}")
        return

    print(f"Attempting to open camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        print("Please check if:")
        print("1. The camera is properly connected")
        print("2. The camera is not being used by another application")
        print("3. You have the correct camera index")
        return

    # Initialize hand detector
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Resize frame to match game dimensions
        frame = cv2.resize(frame, (GAME_WIDTH, GAME_HEIGHT))

        # Find hands
        hands, frame = detector.findHands(frame, draw=False)

        if hands:
            hand = hands[0]  # Get the first hand detected
            fingers = detector.fingersUp(hand)
            
            # Get hand position
            x, y = hand['center']
            
            # Get thumb and index finger positions
            thumb_tip = hand['lmList'][4]  # Thumb tip
            index_tip = hand['lmList'][8]  # Index finger tip
            
            # Calculate distance between thumb and index finger
            distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
            
            # Draw points
            cv2.circle(frame, (thumb_tip[0], thumb_tip[1]), 5, (0, 255, 0), -1)  # Green for thumb
            cv2.circle(frame, (index_tip[0], index_tip[1]), 5, (255, 0, 0), -1)  # Blue for index
            
            # Check for pinch gesture (thumb and index finger close together)
            if distance < 30:  # Adjust this threshold as needed
                if not designer.dragging:
                    # Check if hand is over a component in the palette
                    component_type = designer.get_component_at_position(thumb_tip[0], thumb_tip[1])
                    if component_type:
                        # Start dragging a new component
                        designer.dragging = True
                        designer.selected_element = WebElement(
                            component_type,
                            thumb_tip[0],
                            thumb_tip[1],
                            100,  # Default width
                            50    # Default height
                        )
                        designer.drag_start_pos = (thumb_tip[0], thumb_tip[1])
            else:
                if designer.dragging:
                    if designer.selected_element:
                        # Update selected element position
                        dx = thumb_tip[0] - designer.drag_start_pos[0]
                        dy = thumb_tip[1] - designer.drag_start_pos[1]
                        designer.selected_element.x += dx
                        designer.selected_element.y += dy
                        designer.drag_start_pos = (thumb_tip[0], thumb_tip[1])
                        
                        # Add the element if it's in the design area
                        if designer.selected_element.y < designer.palette_y:
                            designer.elements.append(designer.selected_element)
                    
                    # Stop dragging
                    designer.dragging = False
                    designer.selected_element = None
                    designer.drag_start_pos = None

        # Draw the designer interface
        designer.draw(frame)

        # Display the frame
        cv2.imshow("Website Designer", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 