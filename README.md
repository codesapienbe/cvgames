# CVGames - Computer Vision Games Collection

A comprehensive collection of computer vision games and applications that use hand gestures, facial expressions, and body movements for interaction. All games are designed to be accessible and fun for users of all ages (5-70 years).

## 🎮 Quick Start

### Launch the App Store (Recommended)
```bash
uv run appstore
```

This will start the gesture-controlled app store where you can browse and launch all games using simple hand gestures.

**Note**: Make sure you're in the project root directory when running this command.

### Manual Game Launch
```bash
cd src/[game_name]
python __init__.py
```

## 🏪 App Store Features

The CVGames App Store provides an intuitive, gesture-controlled interface for browsing and launching games:

- **🎯 Ultra-Simple UX**: Large, colorful game cards with zoom effects
- **🤲 Intuitive Gestures**: Hover, select, and pause interaction with hand movements
- **🔄 Dynamic Loading**: Automatically discovers all game modules
- **📱 Smart Interface**: Pagination, visual feedback, and status indicators

### App Store Controls
- **Move hand**: Navigate between game cards
- **Pinch fingers**: Click on game cards and navigation buttons
- **Raise both hands**: Pause interaction (useful for non-app movements)
- **Press 'q'**: Quit the app store

### Game Controls
- **Back Button**: Standardized BACK button in top-left corner of all games
- **Approval Dialog**: Confirmation required before returning to app store
- **Gesture Exit**: Pinch over BACK button or press 'B' key
- **5-second Timeout**: Dialog auto-closes if no action taken

## 🎯 Game Categories

### Hand Gesture Games
- **Finger Maze**: Navigate through mazes using finger tracking
- **Gesture Drawing**: Draw in the air with hand movements
- **Gesture Snake**: Control a snake game with hand gestures
- **Gesture Quiz**: Answer questions using hand signals
- **Rock Paper Scissors**: Play against the computer using hand gestures
- **Virtual Calculator**: Use hand gestures to perform calculations
- **Virtual Keyboard**: Type using hand gestures

### Face and Expression Games
- **Face Filter Fun**: Apply fun filters to your face
- **Face Puzzle**: Solve puzzles using facial expressions
- **Emotion Mirror**: Match target emotions with your expressions
- **Smile Detector**: Score points by smiling
- **Eye Blink Story**: Control story progression with eye blinks
- **Eye Shooter**: Shoot targets using eye gaze
- **Eye Tic Tac Toe**: Play tic-tac-toe with eye movements

### Body Movement Games
- **Fitness Trainer**: Follow exercise routines with pose tracking
- **Dance Battle**: Compete in dance challenges
- **Body Movement Challenge**: Complete physical challenges
- **Virtual Yoga Coach**: Practice yoga with pose guidance
- **Jump Rope Simulator**: Simulate jump rope with body movements
- **Soccer Trainer**: Practice soccer skills with body tracking

### Classic Games with CV
- **2048 with Swipes**: Play 2048 using hand swipes
- **Tetris**: Control Tetris with hand gestures
- **Pong**: Play Pong with hand-controlled paddles
- **Snake**: Control snake with hand movements
- **Tic Tac Toe**: Play using hand gestures
- **Memory Match**: Match cards using hand selection
- **Whack a Mole**: Hit moles using hand gestures

### Educational Games
- **Language Tutor**: Learn languages with gesture recognition
- **Spelling Bee**: Spell words using hand gestures
- **Chemistry Lab**: Virtual chemistry experiments
- **Quiz**: Answer questions using hand gestures
- **Story**: Interactive storytelling with gestures

### Creative and Fun
- **Air Painter**: Paint in the air with hand movements
- **Gesture Commander**: Control spaceships with complex gestures
- **Magic Spells Casting**: Cast spells with hand gestures
- **Pet Simulator**: Interact with virtual pets
- **Team Charades**: Play charades with gesture recognition

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Webcam
- Good lighting for accurate gesture recognition

### Dependencies
```bash
pip install opencv-python mediapipe numpy
```

### Optional Dependencies
Some games may require additional packages:
```bash
pip install pyautogui  # For virtual keyboard
pip install pygame     # For sound effects
```

## 🎮 How to Play

### General Controls
Most games follow these common control patterns:
- **Hand tracking**: Move your hand to control cursor/position
- **Finger gestures**: Use specific finger positions for actions
- **Facial expressions**: Use smiles, blinks, or other expressions
- **Body movements**: Use full body poses for fitness/exercise games

### Game-Specific Instructions
Each game includes:
- `README.md`: Game description and features
- `PLAYER.md`: Detailed instructions and controls
- `__init__.py`: The main game code

## 🏗️ Project Structure

```
cvgames/
├── launch_app_store.py          # App store launcher
├── src/
│   ├── app_store/              # Gesture-controlled app store
│   │   ├── __init__.py         # Main app store code
│   │   ├── README.md           # App store documentation
│   │   ├── PLAYER.md           # User guide
│   │   └── icon.svg            # App store icon
│   ├── [game_name]/            # Individual game modules
│   │   ├── __init__.py         # Game implementation
│   │   ├── README.md           # Game description
│   │   ├── PLAYER.md           # Game instructions
│   │   ├── icon.svg            # Game icon (optional)
│   │   └── Resources/          # Game assets
│   └── cvstore/                # Game metadata and utilities
├── docs/                       # Project documentation
└── README.md                   # This file
```

## 🎯 Design Principles

### Accessibility
- **Age-friendly**: Designed for users 5-70 years old
- **Simple gestures**: Easy to learn and remember
- **Visual feedback**: Clear indication of all actions
- **Error tolerance**: Forgiving gesture recognition

### User Experience
- **Fast learning curve**: Most games can be learned in minutes
- **Immediate feedback**: Real-time response to all interactions
- **Consistent controls**: Similar gesture patterns across games
- **Fullscreen experience**: Immersive gameplay

### Technical Excellence
- **Real-time performance**: 60 FPS target for smooth interaction
- **Robust gesture recognition**: Works in various lighting conditions
- **Modular architecture**: Easy to add new games
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🚀 Adding New Games

To add a new game to the collection:

1. Create a new directory in `src/`
2. Implement the game in `__init__.py`
3. Add `README.md` with game description
4. Add `PLAYER.md` with instructions
5. Optionally add `icon.svg` for the app store
6. The app store will automatically discover your game

### Game Template
```python
import cv2
import mediapipe as mp
import numpy as np

class GameName:
    def __init__(self):
        # Initialize camera and MediaPipe
        pass
    
    def run(self):
        # Main game loop
        pass

if __name__ == "__main__":
    game = GameName()
    game.run()
```

## 🤝 Contributing

We welcome contributions! Please:

1. Follow the existing code style and patterns
2. Include proper documentation (README.md, PLAYER.md)
3. Test your game thoroughly
4. Ensure it works for the target age range (5-70 years)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎉 Acknowledgments

- **MediaPipe**: For excellent hand, face, and pose tracking
- **OpenCV**: For computer vision capabilities
- **NumPy**: For numerical operations
- **The CVGames Community**: For feedback and contributions

## 🆘 Support

If you encounter issues:

1. Check the game's `PLAYER.md` for troubleshooting tips
2. Ensure your webcam is working and well-lit
3. Try the app store for easier navigation
4. Check that all dependencies are installed

---

**Enjoy exploring the world of computer vision games! 🎮✨**
