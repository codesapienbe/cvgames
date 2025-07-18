# 😊 Smile Detector

Play mini-games by smiling or making expressions.

## Description
Advanced emotion detection system that can analyze facial expressions in real-time. Features both detection mode for live emotion analysis and training mode for capturing emotion samples. Includes debug visualization of face mesh landmarks.

## Category
Face Detection

## Age Range
3+

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
- Smile detection
- Mini-games
- Score tracking

## How to Play
1. Run:
   ```bash
   python __init__.py [--camera N] [--debug] [--train --emotion]
   ```
2. Play:
   - Detection mode: Live emotion analysis and FPS display.
   - Training mode: Use `--train` with flags like `--smile`, `--sad`, etc. to capture samples.
   - Debug: Add `--debug` to visualize face mesh landmarks.
3. Quit: Press `q`.

---
This game is part of the CVGames collection. For more games and details, visit the main project documentation. 