# 🕹️ 2048 with Swipes

Play 2048 by swiping in the air with your hands.

## Category

Hand Tracking

## Age Range

8+

## Difficulty

Medium

## Duration

10-30 min

## Requirements

- OpenCV
- MediaPipe
- Python 3.7+
- Webcam

## Features

- Swipe detection with hand gestures
- Score system
- Resolution selection and storage
- Fullscreen support with DPI scaling
- Keyboard controls (arrow keys/WASD)

## How to Play

### Running the Game

```bash
# From the project root directory (recommended)
python -m src.2048withswipes

# Or navigate to the game directory and run
cd src/2048withswipes
python __main__.py
```

### First Launch

On first launch, you'll see a resolution selection dialog:

- Use UP/DOWN arrows to select your preferred resolution
- Press ENTER to confirm
- Your choice will be saved for future launches

### Game Controls

- **Hand Gestures**: Swipe your hand left/right/up/down to move tiles
- **Keyboard**: Use arrow keys or WASD for traditional input
- **Restart**: Press 'R' to restart the game
- **Quit**: Press 'Q' or ESC to exit

### Game Objective

- Combine tiles with the same number to reach 2048
- Each swipe moves all tiles in that direction
- When two tiles with the same number touch, they merge!
- Plan your moves ahead and try to keep high-value tiles in corners

## Configuration

The game automatically stores your resolution preference in `settings.sqlite`:

- **Stored Resolution**: Used on subsequent launches
- **Usage History**: Tracks most frequently used resolutions
- **Smart Fallbacks**: Falls back to 1920x1080 if no preference is set

## File Structure

```
src/2048withswipes/
├── __init__.py      # Main game logic
├── __main__.py      # Entry point for module execution
├── config.py        # Configuration management
├── util.py          # Screen management utilities
├── settings.sqlite  # SQLite database for user settings
├── icon.png         # Game icon
├── PLAYER.md        # Player documentation
└── README.md        # This file
```

## Technical Details

- **Screen Management**: Automatic DPI scaling support (125% zoom, etc.)
- **Fullscreen Mode**: Optimized for immersive gameplay
- **Responsive Design**: Adapts to any screen resolution
- **Structured Logging**: All events logged to `application.log`
- **Module Structure**: Self-contained with no external dependencies
- **Database Storage**: User preferences stored in SQLite `settings.sqlite`

---
This game is part of the CVGames collection. For more games and details, visit the main project documentation.
