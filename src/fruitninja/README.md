# 🎯 Fruit Ninja CV

Slice flying fruit with your index finger as a sword, detected by camera using OpenCV and MediaPipe.

## Category
Hand Tracking & Gesture Recognition

## Age Range
6+

## Difficulty
Medium

## Duration
10-30 min

## Requirements
- OpenCV
- MediaPipe
- Python 3.7+
- Webcam

## 🎮 Game Features

### Core Gameplay
- **Index Finger Sword**: Your index finger acts as the sharpest edge of your sword
- **5 Different Fruits**: Each with unique colors and properties
  - 🍎 Red Apple
  - 🍊 Orange
  - 🍌 Yellow Banana
  - 🍇 Purple Grape
  - 🍓 Dark Red Strawberry
- **Bomb Avoidance**: Black bombs with fuses that explode when touched
- **Scoring System**: +10 points per fruit, -20 points per bomb
- **Game Over**: After 3 bombs exploded

### Visual Effects
- **Sword Trail**: Green trail showing your sword movement
- **Hand Tracking**: Real-time hand landmark visualization
- **Fruit Slicing**: Fruits split into two halves when sliced
- **Bomb Explosions**: Red expanding circles when bombs explode
- **Real-time UI**: Score display and bomb counter

### Controls
- **Hand Movement**: Control sword with hand gestures
- **'q' Key**: Quit game
- **'r' Key**: Restart game (when game over)

## 🚀 How to Play

### Quick Start
```bash
# Navigate to the fruitninja directory
cd src/fruitninja

# Run the game
python __init__.py
```

### Testing
```bash
# Run tests to verify components
python test_game.py
```

### Game Instructions
1. **Position yourself**: Stand in front of your webcam with good lighting
2. **Show your hand**: Extend your hand so the camera can see it clearly
3. **Use index finger**: Point with your index finger - this is your sword
4. **Slice fruits**: Move your hand to slice flying fruits for points
5. **Avoid bombs**: Don't touch the black bombs with fuses
6. **Survive**: Game ends after 3 bombs explode

## 🛠️ Technical Implementation

### Architecture
- **FruitNinja Class**: Main game controller
- **Fruit Class**: Individual fruit objects with physics
- **Bomb Class**: Bomb objects with explosion effects
- **MediaPipe Integration**: Hand tracking and landmark detection
- **OpenCV**: Video capture and rendering

### Key Components
- **Hand Tracking**: Uses MediaPipe Hands for precise finger detection
- **Collision Detection**: Distance-based collision system
- **Physics Engine**: Gravity and velocity simulation
- **Visual Effects**: Real-time rendering with OpenCV
- **Game State Management**: Score, lives, and game over conditions

### Performance Features
- **Optimized Rendering**: Efficient frame processing
- **Memory Management**: Automatic cleanup of off-screen objects
- **Smooth Tracking**: High-confidence hand detection
- **Responsive Controls**: Real-time gesture recognition

## 🎯 Game Rules

### Scoring
- **Fruit Sliced**: +10 points
- **Bomb Exploded**: -20 points
- **Goal**: Achieve the highest score possible

### Game Over Conditions
- **3 Bombs Exploded**: Automatic game over
- **Manual Quit**: Press 'q' to exit

### Spawn Rates
- **Fruits**: Every 1.5 seconds
- **Bombs**: Every 3.0 seconds

## 🔧 Troubleshooting

### Common Issues
- **Hand not detected**: Ensure good lighting and hand visibility
- **Poor tracking**: Keep hand steady, avoid rapid movements
- **Game lag**: Close other applications to free resources
- **Camera not working**: Check webcam permissions and connections

### Performance Tips
- **Lighting**: Use consistent, bright lighting
- **Background**: Use a plain, uncluttered background
- **Distance**: Position yourself 2-3 feet from camera
- **Hand Position**: Keep hand clearly visible in frame

## 📁 File Structure
```
fruitninja/
├── __init__.py          # Main game implementation
├── test_game.py         # Component testing script
├── PLAYER.md           # Detailed player instructions
├── README.md           # This file
└── icon.png            # Game icon
```

## 🤝 Contributing

This game is part of the CVGames collection. For more games and details, visit the main project documentation.

---
**Enjoy slicing fruits and achieving high scores!** 🍎🍊🍌🍇🍓 