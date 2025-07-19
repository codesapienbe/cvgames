# Duck Hunt

A classic arcade-style duck hunting game controlled by hand gestures using computer vision.

## Description

Duck Hunt brings the nostalgic arcade experience to life with modern computer vision technology. Players use their hand as a virtual gun to shoot ducks flying across the screen. The game features realistic duck animations, scoring system, and intuitive hand gesture controls.

## Features

- **Hand Gesture Control**: Point your index finger to aim and shoot
- **Animated Ducks**: Realistic duck movement with flapping wings
- **Scoring System**: Earn points for each duck hit
- **Timer**: 60-second time limit for each round
- **Visual Effects**: Shot animations and hit effects
- **Back Button**: Easy return to app store with gesture or keyboard control

## Controls

- **Aim**: Point your index finger at the screen
- **Shoot**: Keep index finger extended, other fingers closed
- **Exit**: Press 'B' or use the back button gesture
- **Restart**: Press 'R' to restart the game

## Technical Details

- **Computer Vision**: OpenCV and MediaPipe for hand tracking
- **Gesture Recognition**: Index finger pointing detection
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