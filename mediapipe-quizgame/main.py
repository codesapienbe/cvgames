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
background = cv2.imread("Background.png")
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
        # Store cursor position (using full screen coordinates)
        cursor_pos = (
            int(lmList[8][0]),  # Use full X coordinate
            int(lmList[8][1])   # Use full Y coordinate
        )
        # Draw points on webcam feed
        cv2.circle(img, (int(lmList[8][0]), int(lmList[8][1])), 5, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (int(lmList[12][0]), int(lmList[12][1])), 5, (0, 255, 0), cv2.FILLED)

    # Resize webcam feed for bottom left corner (1.5 times smaller)
    webcam_height = int(200 / 1.5)
    webcam_width = int(300 / 1.5)
    webcam = cv2.resize(img, (webcam_width, webcam_height))
    
    # Create main display with background
    main_img = background.copy()
    
    # Place webcam feed in bottom left corner
    main_img[720-webcam_height:720, 0:webcam_width] = webcam
    
    # Place the quiz content on the main image
    if qNo < qTotal:
        mcq = mcqList[qNo]

        main_img, bbox = cvzone.putTextRect(main_img, mcq.question, [100, 100], 2, 2, offset=50, border=5)
        main_img, bbox1 = cvzone.putTextRect(main_img, mcq.choice1, [100, 250], 2, 2, offset=50, border=5)
        main_img, bbox2 = cvzone.putTextRect(main_img, mcq.choice2, [400, 250], 2, 2, offset=50, border=5)
        main_img, bbox3 = cvzone.putTextRect(main_img, mcq.choice3, [100, 400], 2, 2, offset=50, border=5)
        main_img, bbox4 = cvzone.putTextRect(main_img, mcq.choice4, [400, 400], 2, 2, offset=50, border=5)

        if hands:
            cursor = cursor_pos  # Use the full resolution cursor position
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
        main_img, _ = cvzone.putTextRect(main_img, "Quiz Completed", [250, 300], 2, 2, offset=50, border=5)
        main_img, _ = cvzone.putTextRect(main_img, f'Your Score: {score}%', [700, 300], 2, 2, offset=50, border=5)

    # Draw Progress Bar
    barValue = 150 + (950 // qTotal) * qNo
    cv2.rectangle(main_img, (150, 600), (barValue, 650), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(main_img, (150, 600), (1100, 650), (255, 0, 255), 5)
    main_img, _ = cvzone.putTextRect(main_img, f'{round((qNo / qTotal) * 100)}%', [1130, 635], 2, 2, offset=16)

    # Draw cursor on main display if hand is detected (draw last to be on top)
    if cursor_pos is not None:
        # Draw a larger, more visible cursor
        cursor_x = cursor_pos[0]
        cursor_y = cursor_pos[1]
        # Draw cursor shadow for better visibility
        shadow_color = (0, 0, 0)
        shadow_offset = 2
        # Shadow
        cv2.line(main_img, (cursor_x - 20 + shadow_offset, cursor_y + shadow_offset), 
                (cursor_x + 20 + shadow_offset, cursor_y + shadow_offset), shadow_color, 3)
        cv2.line(main_img, (cursor_x + shadow_offset, cursor_y - 20 + shadow_offset), 
                (cursor_x + shadow_offset, cursor_y + 20 + shadow_offset), shadow_color, 3)
        # Main cursor
        cursor_color = (0, 255, 255)  # Bright yellow
        cv2.line(main_img, (cursor_x - 20, cursor_y), (cursor_x + 20, cursor_y), cursor_color, 2)
        cv2.line(main_img, (cursor_x, cursor_y - 20), (cursor_x, cursor_y + 20), cursor_color, 2)
        cv2.circle(main_img, (cursor_x, cursor_y), 6, cursor_color, cv2.FILLED)
        cv2.circle(main_img, (cursor_x, cursor_y), 8, shadow_color, 1)

    cv2.imshow("Img", main_img)
    cv2.waitKey(1)
