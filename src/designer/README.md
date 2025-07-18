# 🎨 Website Designer

Draw website layouts using hand gestures and see the HTML/CSS preview in real-time.

## Category
Utility

## Age Range
All ages

## Difficulty
Medium

## Duration
10-30 min

## Requirements
- Python 3.8 or higher
- OpenCV
- MediaPipe
- PyQt5
- PyQtWebEngine

## Features
- Hand tracking using MediaPipe
- Draw lines using hand gestures (pinch to start/stop drawing)
- Real-time HTML/CSS preview using PyQt5's WebEngine
- Clear drawing functionality
- Split view with camera feed and preview

## How to Play
1. Run:
   ```bash
   python main.py
   ```
2. Play:
   - Pinch your thumb and index finger together to start drawing
   - Move your hand to draw lines
   - Release the pinch to stop drawing
   - Use the "Clear Drawing" button to reset the canvas
   - The right side of the window shows:
     - Generated HTML code
     - Generated CSS code
     - Live preview of your drawing
3. Quit: Close the application window.

## Notes
- Make sure you have good lighting for accurate hand tracking
- Keep your hand within the camera frame
- The drawing area is limited to the camera view (640x480 pixels)

---
This game is part of the CVGames collection. For more games and details, visit the main project documentation.
