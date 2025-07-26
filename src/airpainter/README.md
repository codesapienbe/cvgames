# 🎨 Air Painter - Color by Number Game

Paint beautiful images using hand gestures in this interactive color-by-number game for all ages (5-70).

## Category
Finger Tracking

## Age Range
4+

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
- **Color-by-Number Gameplay**: Paint pre-drawn outline images with beautiful colors
- **Image Library**: Choose from various categories (animals, nature, fantasy, vehicles)
- **Hand Gesture Controls**: Point and pinch to select and paint
- **Multiple Difficulty Levels**: Easy, medium, and hard images for all skill levels
- **Progress Tracking**: Real-time completion percentage and progress indicators
- **Enhanced UX/UI**: Modern interface with responsive design
- **Configuration Management**: Resolution selection and storage
- **DPI Scaling**: Automatic scaling for high-DPI displays
- **Color Palette**: 11 vibrant colors with descriptive names
- **Brush Size Control**: Multiple brush sizes with preference saving
- **Save Artwork**: Export completed paintings as PNG files
- **Undo Functionality**: Step-by-step undo with history
- **Real-time Feedback**: Visual indicators for hand tracking and gestures

## How to Play

### Running the Application

```bash
# From the project root directory (recommended)
python -m src.airpainter

# Or navigate to the game directory and run
cd src/airpainter
python __main__.py
```

### First Launch

On first launch, you'll see a resolution selection dialog:

- Use UP/DOWN arrows to select your preferred resolution
- Press ENTER to confirm
- Your choice will be saved for future launches

### Game Controls

- **Hand Gestures**: 
  - Point with index finger to move cursor
  - Pinch thumb and index finger to select images and paint
- **Image Selection**: Point and pinch on any image in the selection screen
- **Tool Selection**: Point and pinch on tools in the left panel
- **Color Selection**: Point and pinch on color swatches at the bottom
- **Keyboard Shortcuts**:
  - `H`: Toggle help overlay
  - `ESC`: Exit application
  - `R`: Reset to image selection
  - `+/-`: Adjust brush size
  - `Ctrl+S`: Save artwork
  - `Ctrl+Z`: Undo last action

### Game Features

- **Image Categories**: Animals, Nature, Fantasy, Vehicles, and more
- **Difficulty Levels**: Easy, Medium, Hard based on image complexity
- **Progress Tracking**: Real-time completion percentage
- **Color Palette**: 11 vibrant colors with descriptive names
- **Brush Sizes**: 6 different brush sizes (5-30 pixels)
- **Real-time Feedback**: 
  - Hand tracking confidence indicator
  - Gesture recognition feedback
  - Cursor position indicator
  - Hand movement trail
  - Completion progress

## Configuration

The application automatically stores your preferences in `~/.cvgames/airpainter/airpainter_settings.sqlite`:

- **Resolution Settings**: Stored and reused on subsequent launches
- **Brush Size Preferences**: Your preferred brush size is remembered
- **Usage History**: Tracks most frequently used settings
- **Smart Fallbacks**: Falls back to 1920x1080 if no preference is set

## File Structure

```
src/airpainter/
├── __init__.py              # Main game logic and UI
├── __main__.py              # Entry point for module execution
├── config.py                # Configuration management
├── util.py                  # Screen management utilities
├── game_logic.py            # Core game mechanics and image handling
├── icon.png                 # Game icon
├── PLAYER.md                # Player documentation
└── README.md                # This file

User Data (created automatically):
~/.cvgames/airpainter/
├── airpainter_settings.sqlite # SQLite database for user settings
└── data/                    # Image library directory
    ├── images/              # Original coloring images
    └── thumbnails/          # Generated thumbnails
```

## Technical Details

- **Game Engine**: Custom color-by-number painting system
- **Image Processing**: Automatic thumbnail generation and difficulty analysis
- **Screen Management**: Automatic DPI scaling support (125% zoom, etc.)
- **Responsive Design**: Adapts to any screen resolution
- **Structured Logging**: All events logged to `application.log`
- **OpenTelemetry Tracing**: Performance monitoring and debugging
- **Module Structure**: Self-contained with no external dependencies
- **Database Storage**: User preferences stored in SQLite `airpainter_settings.sqlite`
- **Fallback Support**: Graceful degradation when dependencies are missing

## Image Library

- **Sample Images**: Automatically generated if no images are found
- **Categories**: Animals, Nature, Fantasy, Vehicles, General
- **Difficulty Analysis**: Automatic complexity assessment
- **Thumbnail Generation**: Automatic thumbnail creation for image selection
- **Extensible**: Add your own images to the `~/.cvgames/airpainter/data/images/` directory

## Artwork Export

- **Format**: PNG files with timestamp
- **Naming**: `airpainter_artwork_YYYYMMDD_HHMMSS.png`
- **Location**: Saved in the current working directory
- **Quality**: Full resolution with transparency support

---
This application is part of the CVGames collection. For more games and details, visit the main project documentation. 