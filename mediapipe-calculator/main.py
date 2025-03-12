import cv2
import mediapipe as mp
import math
import argparse
from screeninfo import get_monitors
from playsound import playsound
from cvzone.HandTrackingModule import HandDetector
import numpy as np


class Label:
    def __init__(self, x, y, w, h, v, s):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.v = v
        self.s = s

    def draw(self, img):
        cv2.putText(img, self.v, (self.x + 25, self.y + 40), cv2.FONT_HERSHEY_PLAIN, self.s, (255, 255, 255), self.s)

    def resize(self, w, h, img):
        cv2.putText(img, self.v, (self.x + w, self.y + h), cv2.FONT_HERSHEY_PLAIN, self.s, (255, 255, 255), self.s)

    def color(self, rgb, img):
        cv2.putText(img, self.v, (self.x + 25, self.y + 40), cv2.FONT_HERSHEY_PLAIN, self.s, rgb, self.s)

    def text(self, value, img):
        cv2.putText(img, value, (self.x + 25, self.y + 40), cv2.FONT_HERSHEY_PLAIN, self.s, (255, 255, 255), self.s)

    def shrink(self, value, limit, img):
        cv2.putText(img, (self.v[:limit + 1][-1] + value), (self.x + 25, self.y + 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 255, 255),
                    2)


class Rectangle:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def draw(self, img):
        self.background((255, 255, 255), img)
        self.border((255, 255, 255), img)

    def background(self, rgb, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), rgb, cv2.FILLED)

    def border(self, rgb, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), rgb, 3)


class Button:

    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def click(self, img, x, y):
        if self.pos[0] < x < self.pos[0] + self.width and self.pos[1] < y < self.pos[1] + self.height:
            self.border((255, 255, 255), img)
            self.text(self.value, img)
            return True

        else:
            return False

    def draw(self, img):
        self.border((255, 255, 255), img)
        self.text(self.value, img)

    def background(self, rgb, img):
        # for the background calculator
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), rgb, cv2.FILLED)

    def border(self, rgb, img):
        # for the border calculator
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), rgb, 3)

    def text(self, value, img):
        self.value = value
        cv2.putText(img, self.value, (self.pos[0] + 25, self.pos[1] + 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255),
                    3)

    def text_hover(self, value, img):
        self.value = value
        cv2.putText(img, self.value, (self.pos[0] + 25, self.pos[1] + 50), cv2.FONT_HERSHEY_PLAIN, 6, (255, 255, 255),
                    3)

    def move(self, pos, img):
        self.pos = pos
        self.border((255, 255, 255), img)
        self.text(self.value, img)


def main(camera_index=0):
    primary_monitor = {}
    for m in get_monitors():
        print("Connected monitors {}".format(m))
        if m.is_primary:
            primary_monitor = m
            break

    cap = cv2.VideoCapture(camera_index)
    cap.set(3, primary_monitor.width)
    cap.set(4, primary_monitor.height)
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    
    # Calculator design parameters
    calc_width = 400
    calc_height = 500
    calc_x = (primary_monitor.width - calc_width) // 2  # Center horizontally
    calc_y = 100
    button_size = 80
    button_margin = 10
    display_height = 80
    
    # Button colors
    number_color = (50, 50, 50)
    operator_color = (0, 100, 150)
    special_color = (150, 50, 0)
    equals_color = (0, 150, 50)
    
    # creating Button
    button_values = [
        ["C", "DEL", "( )", "%"],
        ["7", "8", "9", "/"],
        ["4", "5", "6", "*"],
        ["1", "2", "3", "-"],
        ["0", ".", "=", "+"]
    ]
    
    button_components = []
    
    # Create buttons with proper spacing and colors
    for row in range(5):
        for col in range(4):
            pos_x = calc_x + col * (button_size + button_margin)
            pos_y = calc_y + display_height + row * (button_size + button_margin)
            
            # Assign different colors based on button type
            if button_values[row][col] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]:
                color = number_color
            elif button_values[row][col] in ["+", "-", "*", "/"]:
                color = operator_color
            elif button_values[row][col] == "=":
                color = equals_color
            else:
                color = special_color
                
            button = Button((pos_x, pos_y), button_size, button_size, button_values[row][col])
            button.color = color  # Store color with the button
            button_components.append(button)
    
    # to store the whole equation from the calculator
    global equation
    equation = ""
    
    # to avoid duplicated value inside calculator in event writing
    delay_counter = 0

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Failed to capture image from camera")
            break
            
        img = cv2.flip(img, 1)

        # Create a black background
        black_img = np.zeros_like(img)

        # Detect hands on the original image
        hand, img = detector.findHands(img, flipType=False)

        # If hands are detected, draw them on the black image
        if hand:
            # Draw hand landmarks on black image
            for h in hand:
                # Draw each landmark as a circle
                for lm in h["lmList"]:
                    x, y = int(lm[0]), int(lm[1])
                    cv2.circle(black_img, (x, y), 5, (0, 255, 0), cv2.FILLED)
                
                # Draw connections between landmarks for better visibility
                for connection in mp.solutions.hands.HAND_CONNECTIONS:
                    idx1, idx2 = connection
                    if idx1 < len(h["lmList"]) and idx2 < len(h["lmList"]):
                        pt1 = (int(h["lmList"][idx1][0]), int(h["lmList"][idx1][1]))
                        pt2 = (int(h["lmList"][idx2][0]), int(h["lmList"][idx2][1]))
                        cv2.line(black_img, pt1, pt2, (0, 255, 0), 2)

        # Draw calculator background
        cv2.rectangle(black_img, 
                     (calc_x - 10, calc_y - 10), 
                     (calc_x + calc_width + 10, calc_y + calc_height + 10), 
                     (30, 30, 30), 
                     cv2.FILLED)
        
        # Draw display background
        cv2.rectangle(black_img, 
                     (calc_x, calc_y), 
                     (calc_x + calc_width - 20, calc_y + display_height), 
                     (10, 10, 10), 
                     cv2.FILLED)
        cv2.rectangle(black_img, 
                     (calc_x, calc_y), 
                     (calc_x + calc_width - 20, calc_y + display_height), 
                     (100, 100, 100), 
                     2)

        # Display equation
        if len(equation) > 15:
            display_text = "..." + equation[-15:]
        else:
            display_text = equation
            
        cv2.putText(black_img, 
                   display_text, 
                   (calc_x + 10, calc_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, 
                   (255, 255, 255), 
                   2)

        # Draw buttons with custom colors
        for button in button_components:
            # Draw button background
            cv2.rectangle(black_img, 
                         button.pos, 
                         (button.pos[0] + button.width, button.pos[1] + button.height), 
                         button.color, 
                         cv2.FILLED)
            
            # Draw button border
            cv2.rectangle(black_img, 
                         button.pos, 
                         (button.pos[0] + button.width, button.pos[1] + button.height), 
                         (200, 200, 200), 
                         2)
            
            # Draw button text
            text_size = cv2.getTextSize(button.value, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = button.pos[0] + (button.width - text_size[0]) // 2
            text_y = button.pos[1] + (button.height + text_size[1]) // 2
            cv2.putText(black_img, 
                       button.value, 
                       (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (255, 255, 255), 
                       2)

        if hand:
            landmarks = hand[0]["lmList"]
            # Get index and middle finger points
            index_finger = landmarks[8][:2]
            middle_finger = landmarks[12][:2]
            
            # Calculate center point between fingers
            center_x = (index_finger[0] + middle_finger[0]) // 2
            center_y = (index_finger[1] + middle_finger[1]) // 2
            
            # Draw selection point
            cv2.circle(black_img, (center_x, center_y), 8, (0, 255, 255), cv2.FILLED)  # Yellow dot
            cv2.circle(black_img, (center_x, center_y), 3, (0, 0, 0), cv2.FILLED)  # Black center
            
            distance, _, _ = detector.findDistance(landmarks[8][:2], landmarks[12][:2], img)
            x, y = center_x, center_y  # Use center point for button detection

            if distance < 70:
                for button in button_components:
                    if button.click(black_img, x, y) and delay_counter == 0:
                        # Highlight the clicked button
                        cv2.rectangle(black_img, 
                                     button.pos, 
                                     (button.pos[0] + button.width, button.pos[1] + button.height), 
                                     (255, 255, 255), 
                                     cv2.FILLED)
                        
                        # Draw button text in black when highlighted
                        text_size = cv2.getTextSize(button.value, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                        text_x = button.pos[0] + (button.width - text_size[0]) // 2
                        text_y = button.pos[1] + (button.height + text_size[1]) // 2
                        cv2.putText(black_img, 
                                   button.value, 
                                   (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, 
                                   (0, 0, 0), 
                                   2)

                        if button.value == "=":
                            try:
                                equation = str(eval(equation))
                            except:
                                equation = "Error"
                        elif button.value == "C":
                            equation = ""
                        elif button.value == "DEL":
                            equation = equation[:-1] if equation else ""
                        elif button.value == "( )":
                            if not equation or equation[-1] in ['+', '-', '*', '/', '(']:
                                equation += '('
                            else:
                                equation += ')'
                        elif button.value == "%":
                            try:
                                current_value = float(equation)
                                equation = str(current_value / 100)
                            except:
                                equation = "Error"
                        elif (button.value in ["+", "-", "*", "/"]) and \
                             (equation.endswith("+") or equation.endswith("-") or 
                              equation.endswith("*") or equation.endswith("/")):
                            equation = equation[:-1] + button.value
                        else:
                            equation += button.value
                        delay_counter = 1

        # avoid duplicates
        if delay_counter != 0:
            delay_counter += 1
            if delay_counter > 10:
                delay_counter = 0

        # Add title and instructions
        cv2.putText(black_img, "Hand Calculator", (calc_x, calc_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(black_img, "Press 'q' to quit, 'c' to clear", 
                   (calc_x, calc_y + calc_height + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow("Hand Calculator", black_img)

        key = cv2.waitKey(1)
        if (key == ord("c")):  # to clear the display calculator
            equation = ""
        if key == ord('q'):  # to stop the program
            cv2.destroyAllWindows()
            cap.release()
            exit(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Camera device index (default: 0)')
    args = parser.parse_args()
    main(camera_index=args.camera)