# MediaPipe Mario Game

A Mario-style platformer game controlled by hand gestures using MediaPipe hand tracking.

## Requirements

- Python 3.8 or higher
- Webcam
- Required Python packages (install using `pip install -r requirements.txt`)

## Game Controls

- **Move Left**: Extend your left hand to the left
- **Move Right**: Extend your right hand to the right
- **Jump**: Raise both hands up
- **Attack**: Make a fist with either hand

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Game

```bash
python main.py
```

## Resources Required

The following resources need to be placed in the `Resources` directory:
- `mario.png`: Mario character sprite
- `ground.png`: Ground texture
- `brick.png`: Brick block texture
- `coin.png`: Coin sprite
- `enemy.png`: Enemy sprite
- `background.png`: Game background image
- `jump.mp3`: Jump sound effect
- `coin.mp3`: Coin collection sound
- `game_over.mp3`: Game over sound

## Game Features

- Hand gesture controls using MediaPipe
- Classic Mario-style platforming mechanics
- Collect coins and defeat enemies
- Multiple levels with increasing difficulty
- Score tracking and lives system 