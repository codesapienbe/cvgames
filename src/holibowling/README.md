# Holistic Bowling Game - Technical Documentation

## Overview

The Holistic Bowling Game is a computer vision-based bowling simulation that uses MediaPipe Holistic for full-body pose detection and gesture recognition. Players control the game using natural body movements and arm gestures.

## Architecture

### Core Components

1. **MediaPipe Holistic Integration**
   - Full-body pose detection
   - Hand landmark tracking
   - Real-time gesture recognition

2. **Game State Machine**
   - `SETUP`: Initial game setup
   - `AIMING`: Player aiming phase
   - `BOWLING`: Ball release phase
   - `SCORING`: Ball physics and pin collision
   - `GAME_OVER`: End game state

3. **Physics Engine**
   - Ball trajectory calculation
   - Pin collision detection
   - Lane boundary handling
   - Realistic ball bouncing

### Gesture Recognition System

#### Aiming Gesture
- **Detection Method**: Shoulder tilt calculation
- **Landmarks Used**: Left and right shoulder positions
- **Algorithm**: 
  ```python
  shoulder_angle = atan2(right_shoulder.y - left_shoulder.y, 
                        right_shoulder.x - left_shoulder.x)
  aim_angle = clip(shoulder_angle * 2, -45, 45)
  ```

#### Bowling Gesture
- **Detection Method**: Arm angle calculation
- **Landmarks Used**: Right shoulder, elbow, and wrist
- **States**:
  - **Windup**: Arm angle > 150° (arm extended back)
  - **Swing**: 90° ≤ Arm angle ≤ 150° (arm in motion)
  - **Release**: Arm angle < 90° (arm forward)

### Physics Implementation

#### Ball Physics
```python
# Gravity effect
ball_velocity[1] += 0.5

# Position update
ball_pos[0] += ball_velocity[0]
ball_pos[1] += ball_velocity[1]

# Lane boundary collision
if ball_pos[0] <= lane_left + BALL_RADIUS or ball_pos[0] >= lane_right - BALL_RADIUS:
    ball_velocity[0] *= -0.8  # Bounce with energy loss
```

#### Pin Collision Detection
```python
for i, pin_pos in enumerate(pin_positions):
    if pins[i]:  # Only check active pins
        distance = sqrt((ball_pos[0] - pin_pos[0])**2 + 
                       (ball_pos[1] - pin_pos[1])**2)
        if distance < BALL_RADIUS + 10:
            pins[i] = False  # Knock down pin
```

### Pin Layout Algorithm

The pins are arranged in a standard bowling triangle formation:

```
       [7]
     [4] [8]
   [2] [5] [9]
 [1] [3] [6] [10]
```

```python
pin_layout = [
    [0, 0],      # Pin 1 (front)
    [-1, 1], [1, 1],  # Pins 2, 3 (second row)
    [-2, 2], [0, 2], [2, 2],  # Pins 4, 5, 6 (third row)
    [-3, 3], [-1, 3], [1, 3], [3, 3]  # Pins 7, 8, 9, 10 (fourth row)
]
```

## Technical Features

### Real-time Processing
- **Frame Rate**: Optimized for 30+ FPS
- **Latency**: Minimal input lag for responsive gameplay
- **Performance**: Efficient landmark processing

### Robust Detection
- **Confidence Thresholds**: Configurable detection confidence
- **Gesture Cooldown**: Prevents accidental gesture triggers
- **Error Handling**: Graceful degradation on poor detection

### Visual Feedback
- **Aiming Line**: Real-time aim direction visualization
- **Power Meter**: Visual power build-up indicator
- **Landmark Overlay**: Optional pose and hand landmark display
- **Game State Indicators**: Clear visual feedback for each game phase

## Dependencies

```python
import cv2                    # Computer vision and image processing
import mediapipe as mp        # Holistic pose detection
import numpy as np           # Numerical computations
import math                  # Mathematical functions
import random                # Random number generation
```

## Configuration

### Game Constants
```python
SCREEN_WIDTH = 1280          # Display width
SCREEN_HEIGHT = 720          # Display height
LANE_WIDTH = 200            # Bowling lane width
PIN_COUNT = 10              # Number of bowling pins
BALL_RADIUS = 15            # Ball size
```

### MediaPipe Settings
```python
min_detection_confidence=0.5    # Minimum confidence for detection
min_tracking_confidence=0.5     # Minimum confidence for tracking
model_complexity=1              # Model complexity (0, 1, or 2)
```

## Performance Optimization

### Memory Management
- Efficient landmark data structures
- Minimal image copying
- Optimized collision detection algorithms

### Processing Pipeline
1. **Image Capture**: Camera frame acquisition
2. **Preprocessing**: Image flip and color conversion
3. **Holistic Processing**: MediaPipe landmark detection
4. **Gesture Analysis**: Real-time gesture recognition
5. **Game Logic**: State machine and physics updates
6. **Rendering**: Visual elements and UI overlay

## Future Enhancements

### Planned Features
- **Multiplayer Support**: Two-player bowling matches
- **Advanced Physics**: Spin effects and oil patterns
- **Customization**: Ball and pin appearance options
- **Statistics**: Detailed scoring and performance tracking
- **Tutorial Mode**: Guided learning experience

### Technical Improvements
- **Machine Learning**: Enhanced gesture recognition
- **3D Visualization**: Depth-based gameplay
- **VR Integration**: Virtual reality support
- **Mobile Optimization**: Smartphone compatibility

## Troubleshooting

### Common Issues
1. **Poor Detection**: Check lighting and camera positioning
2. **High Latency**: Reduce model complexity or frame resolution
3. **Inconsistent Gestures**: Ensure clear, deliberate movements
4. **Performance Issues**: Close other applications using camera

### Debug Mode
Enable debug visualization by modifying the landmark drawing settings:
```python
# Show all landmarks for debugging
mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
```

## Contributing

This project follows the CVGames framework standards. When contributing:

1. Maintain consistent code style
2. Add comprehensive documentation
3. Include unit tests for new features
4. Follow the existing architecture patterns
5. Update both PLAYER.md and README.md files

## License

This project is part of the CVGames framework and follows the same licensing terms. 