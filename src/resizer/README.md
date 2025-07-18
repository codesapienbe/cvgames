# 🔍 Resizer

Resize images using hand gestures detected by camera.

## Category
Utility

## Age Range
All ages

## Difficulty
Easy

## Duration
10-30 min

## Requirements
- OpenCV
- MediaPipe
- Python 3.7+
- Webcam

## Features
- Image resizing
- Gesture-based controls
- Preset zoom levels

## How to Play
1. Run:
   ```bash
   python __init__.py
   ```
2. Play:
   - Use two hands with index and middle fingers extended (fingersUp [1,1,0,0,0]) to start zoom gesture.
   - Move hands closer or farther apart to scale the image under your cursor.
   - Press `s` to save the current view to `saved.png`.
   - Use keys `r` (reset scale), `z/x/c/v/b/n/m` for preset zoom levels.
3. Quit: Press `q` or `Esc` to exit.

---
This game is part of the CVGames collection. For more games and details, visit the main project documentation. 