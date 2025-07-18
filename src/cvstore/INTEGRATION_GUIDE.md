# Back Button Integration Guide

This guide explains how to integrate the standardized back button system into any CVGames game.

## Quick Integration

### 1. Import the Back Button

Add these lines to your game's imports:

```python
import sys
import os

# Import the back button system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cvstore'))
from back_button import BackButton
```

### 2. Initialize the Back Button

In your main function, after setting up the camera and screen:

```python
# Initialize back button (adjust screen dimensions as needed)
back_button = BackButton(screen_width, screen_height)
```

### 3. Handle Input in Game Loop

In your main game loop, add this code to handle back button input:

```python
# Handle back button input
hand_position = None
hand_landmarks = None

# If using cvzone HandDetector:
if hands:
    hand_position = hands[0]["lmList"][9][:2]  # Palm center
    # Convert to MediaPipe landmarks format
    hand_landmarks = type('HandLandmarks', (), {
        'landmark': [type('Landmark', (), {
            'x': lm[0] / screen_width,
            'y': lm[1] / screen_height
        })() for lm in hands[0]["lmList"]]
    })()

# If using MediaPipe directly:
if results.multi_hand_landmarks:
    hand_landmarks = results.multi_hand_landmarks[0]
    hand_position = back_button.get_hand_position(hand_landmarks)

# Check if user wants to exit
if back_button.handle_input(key, hand_landmarks, hand_position):
    print("User approved exit - returning to app store")
    cap.release()
    cv2.destroyAllWindows()
    return  # or sys.exit(0)
```

### 4. Draw the Back Button

Add this line before showing your frame:

```python
# Draw back button
back_button.draw(frame, hand_position)

cv2.imshow("Your Game", frame)
```

## Complete Example

Here's a complete example of how to integrate the back button:

```python
import cv2
import numpy as np
import sys
import os

# Import the back button system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cvstore'))
from back_button import BackButton

def main():
    # Setup camera and screen
    cap = cv2.VideoCapture(0)
    screen_width = 1280
    screen_height = 720
    cap.set(3, screen_width)
    cap.set(4, screen_height)
    
    # Initialize back button
    back_button = BackButton(screen_width, screen_height)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Your game logic here
        # ...
        
        # Handle back button input
        hand_position = None
        hand_landmarks = None
        
        # Get hand data (example with cvzone)
        if hands:
            hand_position = hands[0]["lmList"][9][:2]
            hand_landmarks = type('HandLandmarks', (), {
                'landmark': [type('Landmark', (), {
                    'x': lm[0] / screen_width,
                    'y': lm[1] / screen_height
                })() for lm in hands[0]["lmList"]]
            })()
        
        # Check for exit
        if back_button.handle_input(key, hand_landmarks, hand_position):
            print("User approved exit - returning to app store")
            cap.release()
            cv2.destroyAllWindows()
            return
        
        # Draw back button
        back_button.draw(frame, hand_position)
        
        cv2.imshow("Your Game", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

## Features

### Back Button Controls

- **Keyboard**: Press 'B' to open approval dialog
- **Gesture**: Pinch index and middle fingers over the BACK button
- **Approval**: Pinch over YES/NO buttons in the dialog

### Approval Dialog

- **5-second timeout**: Dialog auto-closes if no action taken
- **Visual feedback**: Hover effects on buttons
- **Clear messaging**: Asks for confirmation before exiting

### Visual Design

- **Glassmorphic style**: Matches app store design
- **Hover effects**: Button highlights when hand is over it
- **Consistent positioning**: Top-left corner of screen
- **Professional appearance**: Clean, modern UI

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure the path to cvstore is correct
2. **Hand Detection**: Ensure hand landmarks are properly formatted
3. **Screen Dimensions**: Pass correct screen width/height to BackButton
4. **Gesture Recognition**: Check that pinch detection works in your environment

### Testing

1. **Keyboard Test**: Press 'B' to test approval dialog
2. **Gesture Test**: Try pinching over the BACK button
3. **Dialog Test**: Test YES/NO button interactions
4. **Timeout Test**: Wait 5 seconds to see auto-close

## Customization

### Button Position

You can customize the button position by modifying the `button_rect` in the BackButton class:

```python
# Default: (50, 50, 120, 60) - top-left corner
# Custom: (x, y, width, height)
back_button.button_rect = (100, 100, 150, 80)
```

### Colors

Modify the colors dictionary in the BackButton class:

```python
back_button.colors = {
    'button_bg': (100, 150, 255),      # Button background
    'button_hover': (120, 170, 255),   # Button hover
    'button_text': (255, 255, 255),    # Button text
    'dialog_bg': (30, 30, 45),         # Dialog background
    'yes_button': (80, 200, 120),      # Yes button
    'no_button': (200, 80, 80),        # No button
    # ... other colors
}
```

### Timeout

Change the approval timeout:

```python
back_button.approval_timeout = 10.0  # 10 seconds instead of 5
```

## Best Practices

1. **Always handle exit**: Don't let games run indefinitely
2. **Consistent positioning**: Keep back button in same location across games
3. **Clear feedback**: Make sure users know how to exit
4. **Graceful cleanup**: Properly release camera and destroy windows
5. **Error handling**: Handle cases where hand detection fails

## Support

If you encounter issues with the back button integration:

1. Check the console for error messages
2. Verify hand detection is working in your game
3. Test with keyboard input first ('B' key)
4. Ensure screen dimensions are correct
5. Check that the cvstore path is accessible 