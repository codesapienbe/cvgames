import cv2
import csv
from cvzone.HandTrackingModule import HandDetector
import time
import argparse
import numpy as np

# Set up argument parser
parser = argparse.ArgumentParser(description='Hand Tracking Quiz Game')
parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
args = parser.parse_args()

cap = cv2.VideoCapture(args.camera)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)

# Load background image
background = cv2.imread("Resources/Background.png")
if background is None:
    background = np.zeros((720, 1280, 3), dtype=np.uint8)  # Fallback to black if image not found
else:
    background = cv2.resize(background, (1280, 720))

class MCQ():
    def __init__(self, data):
        self.question = data[0]
        self.choice1 = data[1]
        self.choice2 = data[2]
        self.choice3 = data[3]
        self.choice4 = data[4]
        self.answer = int(data[5])
        self.userAns = None
        self.hover_start = None
        self.hover_box = None

    def create_glass_effect(self, img, x1, y1, x2, y2, alpha=0.5):
        # Create a semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), cv2.FILLED)
        
        # Add the overlay with transparency
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Add subtle white border for glass effect
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add subtle highlight
        cv2.line(img, (x1, y1), (x2, y1), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y2), (255, 255, 255), 2, cv2.LINE_AA)
        
        return img

    def update(self, cursor, bboxs, main_img):
        current_time = time.time()
        
        # Check which box is being hovered
        for x, bbox in enumerate(bboxs):
            x1, y1, x2, y2 = bbox
            if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:
                # Glass effect for hover
                overlay = main_img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), cv2.FILLED)
                cv2.addWeighted(overlay, 0.3, main_img, 0.7, 0, main_img)
                cv2.rectangle(main_img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
                
                if self.hover_box != x:
                    self.hover_start = current_time
                    self.hover_box = x
                elif current_time - self.hover_start > 1.0 and self.userAns is None:
                    self.userAns = x + 1
                    # Glass effect for selection
                    overlay = main_img.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.addWeighted(overlay, 0.4, main_img, 0.6, 0, main_img)
                elif self.userAns is None:
                    progress = min(1.0, (current_time - self.hover_start) / 1.0)
                    radius = 15
                    center = (x2 - radius - 5, y1 + radius + 5)
                    # Glass effect for progress circle
                    cv2.circle(main_img, center, radius, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.ellipse(main_img, center, (radius, radius), 
                              -90, 0, progress * 360, (0, 255, 255), 2, cv2.LINE_AA)
                return
        
        self.hover_box = None
        self.hover_start = None


# Import csv file data
pathCSV = "Resources/Questions.csv"
with open(pathCSV, newline='\n') as f:
    reader = csv.reader(f)
    dataAll = list(reader)[1:]

# Find the longest question and option lengths
max_question_len = max(len(row[0]) for row in dataAll)
max_option_len = max(len(str(item)) for row in dataAll for item in row[1:5])

# Create Object for each MCQ
mcqList = []
for q in dataAll:
    mcqList.append(MCQ(q))

print("Total MCQ Objects Created:", len(mcqList))

qNo = 0
qTotal = len(dataAll)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False, draw=False)

    # Draw only the points we need for interaction
    cursor_pos = None
    if hands:
        lmList = hands[0]['lmList']
        cursor_pos = (
            int(lmList[8][0]),
            int(lmList[8][1])
        )
        cv2.circle(img, (int(lmList[8][0]), int(lmList[8][1])), 5, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (int(lmList[12][0]), int(lmList[12][1])), 5, (0, 255, 0), cv2.FILLED)

    webcam_height = int(200 / 1.5)
    webcam_width = int(300 / 1.5)
    webcam = cv2.resize(img, (webcam_width, webcam_height))
    
    main_img = background.copy()
    main_img[720-webcam_height:720, 0:webcam_width] = webcam
    
    if qNo < qTotal:
        mcq = mcqList[qNo]

        # Question box with glass effect
        box_width = 600
        question_x = (1280 - box_width) // 2  # Center question box using same width as choices
        question_y = 100
        mcq.create_glass_effect(main_img, question_x-10, question_y-10, 
                              question_x + box_width, question_y + 60, alpha=0.3)
        # Draw question text without background with near-white color
        cv2.putText(main_img, mcq.question, (question_x + 30, question_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 255), 2, cv2.LINE_AA)

        # Options setup
        y_gap = 80  # Increased gap for better spacing
        start_x = (1280 - box_width) // 2  # This calculation is now shared with question box
        start_y = 200
        box_height = 60  # Height for all boxes

        # Define colors for glass effect
        normal_color = (30, 30, 30)
        hover_color = (255, 255, 0)
        select_color = (0, 255, 0)
        text_color = (200, 200, 255)  # Light blue-ish white for better contrast
        
        def draw_glass_box(img, text, pos, is_hover=False, is_selected=False):
            x, y = pos
            x1, y1 = x, y
            x2, y2 = x + box_width, y + box_height
            
            # Create glass effect
            mcq.create_glass_effect(img, x1, y1, x2, y2, alpha=0.3)
            
            # Add text without background
            text_x = x1 + 30
            text_y = y1 + (box_height // 2) + 10  # Adjusted for better vertical centering
            cv2.putText(img, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
            
            return img, (x1, y1, x2, y2)

        # Draw options with glass effect
        main_img, bbox1 = draw_glass_box(main_img, mcq.choice1, [start_x, start_y])
        main_img, bbox2 = draw_glass_box(main_img, mcq.choice2, [start_x, start_y + y_gap])
        main_img, bbox3 = draw_glass_box(main_img, mcq.choice3, [start_x, start_y + y_gap * 2])
        main_img, bbox4 = draw_glass_box(main_img, mcq.choice4, [start_x, start_y + y_gap * 3])

        if hands:
            cursor = cursor_pos
            mcq.update(cursor, [bbox1, bbox2, bbox3, bbox4], main_img)
            if mcq.userAns is not None:
                time.sleep(0.3)
                qNo += 1
    else:
        score = 0
        for mcq in mcqList:
            if mcq.answer == mcq.userAns:
                score += 1
        score = round((score / qTotal) * 100, 2)
        # Draw completion text without background
        cv2.putText(main_img, "Quiz Completed", (250, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(main_img, f'Your Score: {score}%', (700, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 255), 2, cv2.LINE_AA)

    # Progress bar with glass effect
    progress_width = 450
    progress_height = 22
    progress_x = 1280 - progress_width - 100
    progress_y = 720 - progress_height - 30
    
    # Glass effect for progress bar background
    mcq.create_glass_effect(main_img, progress_x, progress_y, 
                           progress_x + progress_width, progress_y + progress_height, alpha=0.3)
    
    # Progress fill with glass effect
    barValue = progress_x + (progress_width * qNo // qTotal)
    overlay = main_img.copy()
    cv2.rectangle(overlay, (progress_x, progress_y), 
                 (barValue, progress_y + progress_height), 
                 (0, 255, 255), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, main_img, 0.5, 0, main_img)
    
    # Progress text without background
    percentage = f'{round((qNo / qTotal) * 100)}%'
    cv2.putText(main_img, percentage, 
                (progress_x + progress_width + 10, progress_y + progress_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2, cv2.LINE_AA)

    # Draw cursor with neon effect
    if cursor_pos is not None:
        cursor_x = cursor_pos[0]
        cursor_y = cursor_pos[1]
        
        # Neon glow effect
        cursor_color = (0, 255, 255)  # Cyan
        glow_color = (0, 128, 128)    # Darker cyan for glow
        
        # Outer glow
        cv2.circle(main_img, (cursor_x, cursor_y), 10, glow_color, 2)
        cv2.line(main_img, (cursor_x - 25, cursor_y), (cursor_x + 25, cursor_y), glow_color, 4)
        cv2.line(main_img, (cursor_x, cursor_y - 25), (cursor_x, cursor_y + 25), glow_color, 4)
        
        # Inner bright lines
        cv2.line(main_img, (cursor_x - 20, cursor_y), (cursor_x + 20, cursor_y), cursor_color, 2)
        cv2.line(main_img, (cursor_x, cursor_y - 20), (cursor_x, cursor_y + 20), cursor_color, 2)
        cv2.circle(main_img, (cursor_x, cursor_y), 6, cursor_color, cv2.FILLED)

    cv2.imshow("Img", main_img)
    cv2.waitKey(1)
