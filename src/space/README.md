# 🛸 Space Invaders

Shoot at invading aliens using hand gestures detected by camera.

## Category
Classic Games

## Age Range
6+

## Difficulty
Medium

## Duration
10-30 min

## Requirements
- OpenCV
- MediaPipe
- Python 3.8 or higher
- Webcam
- Required Python packages (install using `pip install -r requirements.txt`)

## Features
- Control your spaceship using hand gestures
- Shoot at invading aliens
- Progressive difficulty with increasing levels
- Score tracking
- Sound effects (optional)
- HD resolution (1280x720)

## How to Play
1. Run:
   ```bash
   python main.py
   ```
2. Play:
   - Move your index finger left and right to control the spaceship
   - Raise your thumb to shoot
   - Shoot the green alien ships to score points
   - The game gets progressively harder as you level up
   - If any alien reaches the bottom of the screen, it's game over
   - Level up every 500 points
3. Quit: Press `q`.

## Optional Resources
You can add the following files to the `Resources` directory for enhanced gameplay:
- `space_background.png`: A space-themed background image
- `shoot.mp3`: Sound effect for shooting
- `explosion.mp3`: Sound effect for alien explosions
- `game_over.mp3`: Sound effect for game over

---
This game is part of the CVGames collection. For more games and details, visit the main project documentation. 