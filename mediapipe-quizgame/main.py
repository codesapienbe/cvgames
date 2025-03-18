import cv2
import csv
from cvzone.HandTrackingModule import HandDetector
import cvzone
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
        self.hover_start = None  # Track when hovering started
        self.hover_box = None    # Track which box is being hovered

    def update(self, cursor, bboxs, main_img):
        current_time = time.time()
        
        # Check which box is being hovered
        for x, bbox in enumerate(bboxs):
            x1, y1, x2, y2 = bbox
            if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:
                # Draw hover effect
                cv2.rectangle(main_img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                
                # If hovering over a new box, reset timer
                if self.hover_box != x:
                    self.hover_start = current_time
                    self.hover_box = x
                # If hovering over same box for 1 second, select it
                elif current_time - self.hover_start > 1.0 and self.userAns is None:
                    self.userAns = x + 1
                    cv2.rectangle(main_img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
                # Draw progress circle while hovering
                elif self.userAns is None:
                    progress = min(1.0, (current_time - self.hover_start) / 1.0)
                    radius = 15
                    center = (x2 - radius - 5, y1 + radius + 5)
                    cv2.circle(main_img, center, radius, (255, 255, 255), 2)
                    cv2.ellipse(main_img, center, (radius, radius), 
                              -90, 0, progress * 360, (0, 255, 0), 3)
                return
        
        # If not hovering over any box, reset
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

        # Create neon effect for question box
        question_x = 440  # Centered horizontally
        question_y = 100
        # Draw question with neon effect
        main_img, bbox = cvzone.putTextRect(main_img, mcq.question, [question_x, question_y], 1.6, 1, 
                                          offset=15, border=0,  # Removed border
                                          colorR=(255, 0, 255),  # Magenta background
                                          colorT=(255, 255, 255))  # White text
        
        # Vertical layout for options with consistent box size
        box_width = 600  # Fixed width for all boxes
        y_gap = 70  # Gap between options
        
        # Calculate starting x position to center the boxes
        start_x = (1280 - box_width) // 2  # Center horizontally
        start_y = 200  # Start options below question
        
        # Define colors for different states
        normal_color = (50, 50, 50)      # Dark gray background
        hover_color = (255, 255, 0)      # Yellow for hover
        select_color = (0, 255, 0)       # Green for selected
        text_color = (255, 255, 255)     # White text
        border_color = (0, 255, 0)       # Neon green border
        
        # Add neon glow effect to boxes
        def draw_neon_box(img, text, pos, is_hover=False, is_selected=False):
            x, y = pos
            # First draw a fixed-width background rectangle
            box_height = 50  # Fixed height for the box
            x1 = x
            y1 = y
            x2 = x + box_width
            y2 = y + box_height
            
            # Draw background rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), normal_color, cv2.FILLED)
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1)
            
            # Calculate text position to center it in the box
            text_x = x1 + 20  # Add padding from left
            text_y = y1 + (box_height // 2)  # Vertically center
            
            # Draw text centered in the box without background (since we already drew it)
            img, _ = cvzone.putTextRect(img, text, [text_x, text_y], 1.6, 1,
                                      offset=0, border=0,  # No border or offset since we handle it manually
                                      colorR=normal_color,  # Same as background to blend
                                      colorT=text_color)
            
            # Add neon border effect
            if is_hover or is_selected:
                color = select_color if is_selected else hover_color
                # Draw outer glow
                cv2.rectangle(img, (x1-2, y1-2), (x2+2, y2+2), color, 2)
            
            return img, (x1, y1, x2, y2)  # Return the box coordinates for hover detection
        
        # Draw options vertically with neon effect
        main_img, bbox1 = draw_neon_box(main_img, mcq.choice1, [start_x, start_y])
        main_img, bbox2 = draw_neon_box(main_img, mcq.choice2, [start_x, start_y + y_gap])
        main_img, bbox3 = draw_neon_box(main_img, mcq.choice3, [start_x, start_y + y_gap * 2])
        main_img, bbox4 = draw_neon_box(main_img, mcq.choice4, [start_x, start_y + y_gap * 3])

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
        main_img, _ = cvzone.putTextRect(main_img, "Quiz Completed", [250, 300], 1, 2, 
                                       offset=20, border=0,  # Removed border
                                       colorR=(255, 0, 255),
                                       colorT=(255, 255, 255))
        main_img, _ = cvzone.putTextRect(main_img, f'Your Score: {score}%', [700, 300], 1, 2, 
                                       offset=20, border=0,  # Removed border
                                       colorR=(255, 0, 255),
                                       colorT=(255, 255, 255))

    # Draw Progress Bar with neon effect
    progress_width = 450  # Increased from 300 to 450 (1.5x)
    progress_height = 22  # Increased from 15 to 22 (1.5x)
    progress_x = 1280 - progress_width - 100  # Moved further left to prevent overlap
    progress_y = 720 - progress_height - 30  # Moved slightly higher
    
    # Calculate progress
    barValue = progress_x + (progress_width * qNo // qTotal)
    
    # Draw neon progress bar
    # Outer glow
    cv2.rectangle(main_img, (progress_x-2, progress_y-2), 
                 (progress_x + progress_width+2, progress_y + progress_height+2), 
                 (255, 0, 255), 3)  # Magenta glow
    
    # Inner bar
    cv2.rectangle(main_img, (progress_x, progress_y), 
                 (barValue, progress_y + progress_height), 
                 (0, 255, 0), cv2.FILLED)  # Neon green fill
    
    # Progress text with neon effect
    percentage = f'{round((qNo / qTotal) * 100)}%'
    main_img, _ = cvzone.putTextRect(main_img, percentage, 
                                   [progress_x + progress_width + 10, progress_y + progress_height//2], 
                                   1.2, 1, offset=5,  # Slightly larger text
                                   colorR=(255, 0, 255),
                                   colorT=(255, 255, 255))

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
