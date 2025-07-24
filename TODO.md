# Migration TODO for CVGames: Pygame + OpenCV/MediaPipe + OpenTelemetry

This file lists the remaining games and the required migration/implementation steps to ensure all games:
- Use **Pygame** for the main window, event loop, and rendering
- Use **OpenCV + MediaPipe** for computer vision input (webcam, hand/face/pose tracking, gesture detection)
- Add **OpenTelemetry (otel)** compatible logging for key game events
- All changes must be made in the game's `__init__.py` file
- Preserve all business logic and CV-based interaction

## Standard Implementation Steps
For each game:
1. Refactor the main loop and rendering to use Pygame for the window, event loop, and display.
2. Keep OpenCV+MediaPipe for webcam input and all CV-based interaction.
3. Add OpenTelemetry (otel) logging for key events (game start, move, gesture, game over, restart, quit, etc.).
4. Remove OpenCV window usage (`cv2.imshow`) and display everything in the Pygame window using surfarray and Pygame drawing.
5. Preserve all game logic and features.

---

## Migration Status

**All games in the previous migration list have been migrated or implemented.**

- Pygame is now used for rendering and the main loop in all games.
- OpenCV + MediaPipe (or cvzone/HandTrackingModule) is used for computer vision input.
- OpenTelemetry (otel) logging is present for key events.
- All code is in the respective `__init__.py` files.

**No games remain in the migration list.**

---

## Future Games / Additions

If new games or modules are added, list them here for migration/implementation:

- (Add new games here as needed)

---

## Notes
- Utility/transition modules (e.g., `cvstore`, `exiting_state`, `loading_state`) do **not** require migration.
- Always update the `__init__.py` file for each game.
- If a game is a duplicate or superseded by another (e.g., `drive` vs `drivesimulator`), only migrate the preferred one.

---

**This TODO should be kept up to date as you migrate or implement each game.**
