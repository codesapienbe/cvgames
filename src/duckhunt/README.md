# Duck Hunt

A classic arcade-style duck hunting game controlled by hand gestures using computer vision.

## Description

Duck Hunt brings the nostalgic arcade experience to life with modern computer vision technology. Players use their hand as a virtual gun to shoot ducks flying across the screen. The game features realistic duck animations, scoring system, and intuitive hand gesture controls.

## Features

- **Enhanced Hand Tracking**: Improved index finger detection with multiple fallback methods
- **Easy Quarter-Circle Shooting**: Forgiving thumb gesture (left→top) with only 20% motion required
- **Stabilized Cursor**: Reduced shaking when hand is stationary
- **Aim Stabilization**: Locks aim position during shooting gestures to prevent drift
- **Multiple Duck Types**: 7 different colored ducks with varying scores
- **High-Intensity Spawning**: 5x more ducks for action-packed gameplay (15-30+ per round)
- **Multiple Targets**: 4-12 ducks on screen simultaneously
- **Enhanced Background**: Moving clouds, trees, bushes, and natural grass animation
- **Progressive Difficulty**: More duck types and faster spawning at higher levels
- **Visual Effects**: Color-coded hit effects and score displays
- **Game Completion**: Win/lose conditions with 50% accuracy requirement
- **Gesture-Based UI**: All buttons use shooting gesture for interaction
- **Replay System**: Easy restart after game completion
- **Retro Graphics**: Authentic NES Duck Hunt style
- **Optimized UI Layout**: Scoreboard on top-right, progress on top-left

## Controls

- **Aim**: Point your index finger at the screen
- **Shoot**: Move thumb in quarter-circle motion (top to left)
- **Exit**: Press 'B' or use the back button gesture
- **Restart**: Press 'R' to restart the game

## Technical Details

- **Computer Vision**: OpenCV and MediaPipe for hand tracking
- **Gesture Recognition**: Enhanced index finger detection with multiple fallback methods
- **Shooting Gesture**: Quarter-circle thumb motion detection with trajectory analysis
- **Game Engine**: Custom game loop with collision detection
- **Graphics**: OpenCV drawing functions for all game elements

## Requirements

- OpenCV
- MediaPipe
- NumPy
- Webcam

## Game Mechanics

- Ducks spawn from the bottom of the screen
- Each duck moves in a random upward trajectory
- Hit ducks to score 10 points each
- Game lasts 60 seconds
- Multiple ducks can be on screen simultaneously

## Installation

The game is part of the CVGames collection and can be launched from the app store. 