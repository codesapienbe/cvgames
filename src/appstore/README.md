# CVGames App Store

A gesture-controlled application launcher for the CVGames collection. This app store provides an intuitive, age-friendly interface (5-70 years) for browsing and launching computer vision games using hand gestures.

## Features

### 🎯 **Ultra-Simple UX**
- **Large, colorful game cards** with clear visual hierarchy
- **Zoom effects** when hovering over games (simulates mouse hover)
- **Progress rings** show selection progress
- **Minimal text** with icons and simple descriptions

### 🤲 **Intuitive Gesture Controls**
- **Hover**: Move hand over a game card to highlight it (zoom effect)
- **Select**: Close fist (all fingers to palm) to select a game (1 second hold)
- **Info Modal**: Open palm over INFO button to see usage instructions
- **Pause Interaction**: Raise both hands to temporarily disable gesture detection
- **Resume Interaction**: Lower hands after 2 seconds to re-enable

### 🔄 **Dynamic Game Loading**
- **Automatic discovery** of all game modules in the `src/` directory
- **Reads game metadata** from README.md and PLAYER.md files
- **Supports game icons** (icon.svg files)
- **No manual configuration** required

### 📱 **Smart Interface**
- **Pagination** for large game collections (6 games per page)
- **Visual feedback** for all interactions
- **Status indicators** show interaction state
- **Fullscreen immersive experience**

## How It Works

### Game Discovery
The app store automatically scans the `src/` directory for game modules. Each game module should have:
- `__init__.py` - Contains the game code
- `README.md` - Game description (first paragraph used)
- `PLAYER.md` - Game rules and instructions
- `icon.svg` - Game icon (optional)

### Gesture Recognition
- **Hand Position**: Uses MediaPipe to track hand landmarks
- **Hover Detection**: Calculates if hand is over a game card
- **Fist Detection**: Checks if all fingers are closed
- **Hands Raised**: Detects when both hands are raised to pause interaction

### Game Launching
- Games are launched in separate subprocesses
- App store waits for game completion
- Returns to app store when game exits
- Maintains fullscreen experience throughout

## Usage

### Starting the App Store
```bash
# Using uv (recommended)
uv run appstore

# Or directly
cd src/app_store
python __init__.py
```

### Controls
- **Move hand**: Navigate between game cards
- **Close fist**: Select and launch a game (after 1 second hold)
- **Open palm on INFO**: Show usage instructions
- **Raise both hands**: Pause interaction (useful for non-app movements)
- **Press 'q'**: Quit the app store
- **Press 'i'**: Toggle info modal
- **Press 'a'/'d'**: Navigate between pages (keyboard fallback)

### Visual Feedback
- **Dark blue cards**: Normal state
- **Blue cards with zoom**: Hover state
- **Green cards**: Ready to select (after 1 second hold)
- **Progress ring with percentage**: Shows selection progress
- **INFO button**: Bottom left with help instructions
- **Status text**: Shows interaction state (ENABLED/DISABLED)

## Technical Details

### Dependencies
- OpenCV (cv2) - Camera capture and UI rendering
- MediaPipe - Hand tracking and gesture recognition
- NumPy - Numerical operations
- Pathlib - File system operations
- Subprocess - Game launching

### Architecture
- **GameCard class**: Represents individual games with metadata
- **AppStore class**: Main application with gesture handling and UI
- **Dynamic loading**: Scans directory structure at runtime
- **Modular design**: Easy to extend with new features

### Performance
- **60 FPS** target for smooth interaction
- **Low latency** gesture recognition
- **Efficient rendering** with OpenCV
- **Memory efficient** game launching

## Accessibility Features

### Age-Friendly Design
- **Large touch targets** (300x400 pixel cards)
- **High contrast** colors and text
- **Simple gestures** that are easy to learn
- **Visual feedback** for all actions

### Learning Curve
- **Immediate feedback** on all interactions
- **Consistent gesture patterns** across all games
- **Clear visual hierarchy** with titles and descriptions
- **Intuitive navigation** with arrows and page indicators

## Future Enhancements

- **Voice commands** for game selection
- **Game categories** and filtering
- **Favorites system** for quick access
- **Game statistics** and play history
- **Multi-language support**
- **Custom themes** and personalization 