import cv2
import mediapipe as mp
import numpy as np
import time

COLORS = [
    (0, 0, 0), (128, 128, 128), (136, 0, 21), (237, 28, 36),
    (255, 127, 39), (255, 242, 0), (34, 177, 76), (0, 162, 232),
    (63, 72, 204), (163, 73, 164), (255, 255, 255)
]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create fullscreen window
    cv2.namedWindow("Air Paint", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Air Paint", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Configurable UI dimensions and padding
    TOOLBAR_WIDTH = 80
    PALETTE_HEIGHT = 60
    COLOR_SWATCH_SIZE = 40
    PADDING = 10
    MARGIN = 5
    SELECTION_TIME = 3  # seconds for selection

    canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    selected_color = (255, 0, 0)
    selected_tool = "brush"
    selection_start_time = None
    current_hover_target = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame to correct the mirror effect
        frame = cv2.resize(frame, (1280, 720))
        background = np.full_like(frame, (245, 245, 245))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Draw fixed UI elements
        cv2.rectangle(background, (0, 0), (TOOLBAR_WIDTH, 720), (240, 240, 240), -1)
        tools = ["brush", "eraser", "clear"]
        for i, tool in enumerate(tools):
            y = PADDING + i * (COLOR_SWATCH_SIZE + MARGIN)
            cv2.putText(background, tool, (PADDING, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 0, 0) if tool != selected_tool else selected_color, 2)

        cv2.rectangle(background, (0, 720 - PALETTE_HEIGHT), (1280, 720), (240, 240, 240), -1)
        for i, color in enumerate(COLORS):
            x = PADDING + i * (COLOR_SWATCH_SIZE + MARGIN)
            cv2.rectangle(background, (x, 720 - PALETTE_HEIGHT + PADDING), 
                          (x + COLOR_SWATCH_SIZE, 720 - PADDING), color[::-1], -1)

        # Hand processing
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                # Get index finger and thumb positions
                index_tip = hand.landmark[8]
                thumb_tip = hand.landmark[4]
                x, y = int(index_tip.x * 1280), int(index_tip.y * 720)
                
                # Check if index and thumb are close enough to draw
                distance = np.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)
                if distance < 0.05:  # Threshold for drawing
                    if selected_tool == "brush":
                        cv2.circle(canvas, (x, y), 10, selected_color, -1)
                    elif selected_tool == "eraser":
                        cv2.circle(canvas, (x, y), 20, (255, 255, 255), -1)

                # Check UI interactions
                if y < 720 - PALETTE_HEIGHT and x < TOOLBAR_WIDTH:
                    current_target = "toolbar"
                elif y > 720 - PALETTE_HEIGHT:
                    current_target = "palette"
                else:
                    current_target = None
                
                # Selection timer logic
                current_time = time.time()
                if current_target == current_hover_target:
                    if selection_start_time is not None and (current_time - selection_start_time) > SELECTION_TIME:
                        if current_target == "toolbar":
                            if PADDING < y < PADDING + COLOR_SWATCH_SIZE: selected_tool = "brush"
                            elif PADDING + COLOR_SWATCH_SIZE + MARGIN < y < PADDING + 2 * COLOR_SWATCH_SIZE + MARGIN: selected_tool = "eraser"
                            elif PADDING + 2 * (COLOR_SWATCH_SIZE + MARGIN) < y < PADDING + 3 * COLOR_SWATCH_SIZE + 2 * MARGIN: canvas[:] = 255
                        elif current_target == "palette":
                            for i, color in enumerate(COLORS):
                                x_start = PADDING + i * (COLOR_SWATCH_SIZE + MARGIN)
                                if x_start < x < x_start + COLOR_SWATCH_SIZE:
                                    selected_color = color[::-1]
                        selection_start_time = None
                        current_hover_target = None
                else:
                    current_hover_target = current_target
                    selection_start_time = current_time

                # Draw selection progress
                if current_hover_target:
                    progress = min(1.0, (time.time() - selection_start_time) / SELECTION_TIME)
                    cv2.circle(background, (x, y), 20, (200, 200, 200), 2)
                    cv2.ellipse(background, (x, y), (20, 20), -90, 0, 360 * progress, (0, 255, 0), 2)

                # Draw only hand landmarks on light background
                mp_draw.draw_landmarks(background, hand, mp_hands.HAND_CONNECTIONS)

        # Combine canvas and UI
        display_frame = cv2.addWeighted(background, 0.7, canvas, 0.3, 0)
        
        # Draw current color indicator
        cv2.rectangle(display_frame, (PADDING, 720 - PALETTE_HEIGHT - COLOR_SWATCH_SIZE - MARGIN), 
                      (PADDING + COLOR_SWATCH_SIZE, 720 - PALETTE_HEIGHT - MARGIN), selected_color, -1)
        
        cv2.imshow("Air Paint", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
