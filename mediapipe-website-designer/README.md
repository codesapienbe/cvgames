# Website Designer with Hand Tracking

This application allows you to draw website layouts using hand gestures and see the HTML/CSS preview in real-time.

## Features

- Hand tracking using MediaPipe
- Draw lines using hand gestures (pinch to start/stop drawing)
- Real-time HTML/CSS preview using PyQt5's WebEngine
- Clear drawing functionality
- Split view with camera feed and preview

## Requirements

- Python 3.8 or higher
- OpenCV
- MediaPipe
- PyQt5
- PyQtWebEngine

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Use hand gestures to draw:
   - Pinch your thumb and index finger together to start drawing
   - Move your hand to draw lines
   - Release the pinch to stop drawing
   - Use the "Clear Drawing" button to reset the canvas

3. The right side of the window shows:
   - Generated HTML code
   - Generated CSS code
   - Live preview of your drawing

## How it Works

The application uses MediaPipe to track your hand movements and converts them into SVG paths. These paths are then rendered in real-time using HTML and CSS. The preview is updated continuously as you draw.

## Notes

- Make sure you have good lighting for accurate hand tracking
- Keep your hand within the camera frame
- The drawing area is limited to the camera view (640x480 pixels) 