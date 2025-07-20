# Holistic Bowling Game

A computer vision-based bowling game that uses your body movements and gestures to control the game.

## How to Play

### Setup
1. Run the game:
   ```bash
   python __init__.py
   ```

2. Stand in front of your camera with good lighting
3. Make sure your full upper body is visible

### Controls

#### Aiming
- **Tilt your shoulders** left or right to aim the ball
- The green aiming line shows your current aim direction
- Keep your shoulders level for straight shots

#### Bowling Motion
1. **Windup**: Raise your right arm back (arm angle > 150°)
2. **Power Build-up**: Hold the windup position to build power
3. **Release**: Swing your arm forward (arm angle < 90°) to release the ball

### Game Mechanics

- **10 Frames**: Standard bowling rules with 10 frames
- **2 Rolls per Frame**: You get 2 attempts per frame (unless you get a strike)
- **Pin Layout**: 10 pins arranged in a triangle formation
- **Physics**: Realistic ball physics with lane boundaries and pin collisions
- **Scoring**: Standard bowling scoring system

### Visual Feedback

- **Green Aiming Line**: Shows your current aim direction
- **Power Meter**: Displays power level during windup
- **Score Display**: Shows current score, frame, and roll
- **Pin Status**: White circles represent standing pins
- **Ball**: Red circle shows ball position and trajectory

### Tips for Better Performance

1. **Good Posture**: Stand with your feet shoulder-width apart
2. **Clear Movements**: Make deliberate, clear gestures
3. **Consistent Lighting**: Ensure good, consistent lighting
4. **Camera Distance**: Stand 6-8 feet from the camera
5. **Arm Visibility**: Keep your right arm clearly visible to the camera

### Controls

- **'r'**: Reset/restart the game
- **'q'**: Quit the game
- **Back Button**: Use hand gesture to return to app store

### Technical Requirements

- **Camera**: Webcam or USB camera
- **Lighting**: Good, consistent lighting
- **Space**: Room to move your arms freely
- **Distance**: 6-8 feet from camera for best detection

### Troubleshooting

- **Poor Detection**: Check lighting and camera positioning
- **Inconsistent Aiming**: Ensure shoulders are clearly visible
- **Ball Not Releasing**: Make sure arm swing motion is clear and deliberate
- **Game Not Responding**: Try restarting with 'r' key

Enjoy your holistic bowling experience! 