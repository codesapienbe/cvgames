# Space Invaders with MediaPipe Hand Tracking

A classic Space Invaders game controlled by hand gestures using MediaPipe hand tracking.

## Features

- Control your spaceship using hand gestures
- Shoot at invading aliens
- Progressive difficulty with increasing levels
- Score tracking
- Sound effects (optional)
- HD resolution (1280x720)

## Requirements

- Python 3.8 or higher
- Webcam
- Required Python packages (install using `pip install -r requirements.txt`)

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the game:
   ```bash
   python main.py
   ```
2. If you have multiple cameras, you can specify which camera to use:
   ```bash
   python main.py --camera 1
   ```

## Controls

- Move your index finger left and right to control the spaceship
- Raise your thumb to shoot
- Press 'q' to quit the game

## Game Rules

- Shoot the green alien ships to score points
- Each alien is worth 100 points
- The game gets progressively harder as you level up
- If any alien reaches the bottom of the screen, it's game over
- Level up every 500 points

## Optional Resources

You can add the following files to the `Resources` directory for enhanced gameplay:
- `space_background.png`: A space-themed background image
- `shoot.mp3`: Sound effect for shooting
- `explosion.mp3`: Sound effect for alien explosions
- `game_over.mp3`: Sound effect for game over

## Troubleshooting

If you encounter any issues:
1. Make sure your webcam is properly connected and accessible
2. Check if all required packages are installed correctly
3. Ensure you have sufficient lighting for hand tracking
4. Try adjusting the camera index if the game doesn't detect your preferred camera 