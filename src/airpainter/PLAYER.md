# PLAYER

## Quick Start

1. **Run the game:**
   ```bash
   # From project root (recommended)
   python -m src.airpainter
   
   # Or from the airpainter directory
   cd src/airpainter
   python __main__.py
   ```

2. **First time setup:**
   - Select your preferred resolution when prompted
   - Your choice will be saved for future launches

3. **Start painting:**
   - Point with your index finger to move the cursor
   - Pinch your thumb and index finger to select images and paint
   - Choose an image from the selection screen
   - Paint on the black outline areas with your chosen colors

## Controls

### Hand Gestures
- **Point**: Move index finger to control cursor position
- **Pinch**: Bring thumb and index finger together to select images and paint
- **Hold**: Keep pinch gesture to continue painting smoothly

### Keyboard Shortcuts
- `H`: Toggle help overlay
- `ESC`: Exit game
- `R`: Reset to image selection
- `+/-`: Increase/decrease brush size
- `Ctrl+S`: Save artwork as PNG file
- `Ctrl+Z`: Undo last painting action

### Tools
- **Brush**: Paint with selected color and size
- **Eraser**: Remove painted content (paint with white)
- **Clear**: Clear entire painting (reset to white)
- **Undo**: Step back through painting history
- **Save**: Export completed artwork to file
- **Image URL**: Import and convert image from URL to coloring page
- **Video URL**: Import YouTube video and extract frames as coloring pages

### Colors
- 11 colors available: Black, Gray, Dark Red, Red, Orange, Yellow, Green, Blue, Purple, Pink, White
- Point and pinch on color swatches to select

### Game Features
- **Image Selection**: Choose from various categories (animals, nature, fantasy, vehicles)
- **Difficulty Levels**: Easy, Medium, Hard based on image complexity
- **Progress Tracking**: Real-time completion percentage
- **Smooth Painting**: Continuous painting while holding pinch gesture
- **URL Import**: Import images and videos from URLs to create coloring pages
- **Auto Conversion**: Automatic edge detection and coloring page generation

## Tips

- **Image Selection**: Choose images based on difficulty level (Easy for beginners)
- **Smooth Painting**: Hold the pinch gesture to paint continuously
- **Brush Size**: Use +/- keys to adjust brush size for different areas
- **Precision**: Watch the cursor indicator for precise positioning
- **Progress**: Monitor completion percentage in the status bar
- **Saving**: Use Ctrl+S or the Save tool to export your completed artwork
- **Undo**: Use Ctrl+Z or the Undo tool to step back through painting actions
- **URL Import**: Use Image URL tool for direct image links, Video URL for YouTube videos
- **Auto Conversion**: Imported images are automatically converted to coloring pages with edge detection

## Troubleshooting

- **Hand not detected**: Ensure good lighting and hand is clearly visible
- **Cursor not moving**: Check that your index finger is pointing at the camera
- **Painting not working**: Make sure to pinch thumb and index finger together
- **Image not loading**: Check that images exist in the ~/.cvgames/airpainter/data/images directory
- **Poor performance**: Try reducing resolution in the settings dialog 