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

## Universal Loading Screen Integration

### 🎯 **Objective**

Create a universal, reusable loading screen system that can be integrated into all CVGames modules. The loading screen should provide a consistent, engaging user experience during application startup and system resource initialization.

### 📋 **Requirements**

#### **Core Features:**

1. **Universal Loading Screen Class** (`UniversalLoadingScreen`)
   - Configurable game title and branding
   - Customizable animation sequences
   - Progress tracking with visual feedback
   - Consistent timing (30 seconds total duration)
   - Platform-agnostic implementation

#### **Animation System:**

2. **Modular Animation Framework**
   - **Game-Specific Animations:** Each game can define its own animation sequence
   - **Default Animations:** Fallback animations for games without custom ones
   - **Animation Types:**
     - Rotating elements (like current hand tracking setup)
     - Pulsing effects
     - Color transitions
     - Text animations
     - Progress bar animations

#### **Configuration System:**

3. **Game Configuration Interface**

   ```python
   class GameLoadingConfig:
       game_name: str
       game_icon: str  # Emoji or text symbol
       primary_color: tuple
       secondary_color: tuple
       animation_sequence: list
       loading_messages: list
       estimated_duration: int  # seconds
   ```

#### **Integration Points:**

4. **Standard Integration Pattern**
   - All games should call `show_loading_screen()` at startup
   - Loading screen should be called before any heavy initialization
   - Consistent API across all modules
   - Error handling for failed loading

### 🔧 **Implementation Steps**

#### **Phase 1: Create Universal Loading Screen**

1. **Create `src/loading_screen.py`**
   - Extract current loading screen logic from `airpainter/__init__.py`
   - Make it configurable and reusable
   - Add animation framework
   - Add progress tracking system

2. **Loading Screen Features:**

   ```python
   class UniversalLoadingScreen:
       def __init__(self, config: GameLoadingConfig)
       def show_loading_screen(self)
       def update_progress(self, progress: float, message: str)
       def add_animation(self, animation_type: str, **params)
       def set_custom_animation(self, animation_func)
   ```

#### **Phase 2: Game-Specific Configurations**

3. **Create Default Configurations**
   - **Air Painter:** Current hand tracking setup animation
   - **2048:** Number tile animations
   - **Tetris:** Falling block animations
   - **Snake:** Snake movement animations
   - **Pong:** Ball bouncing animations
   - **Generic:** Default rotating/pulsing animations

4. **Animation Examples:**

   ```python
   # Air Painter Animation
   airpainter_config = GameLoadingConfig(
       game_name="Air Painter",
       game_icon="🎨",
       primary_color=(100, 150, 255),
       secondary_color=(255, 255, 255),
       animation_sequence=["hand_tracking_setup", "color_palette", "brush_loading"],
       loading_messages=[
           "Setting up Hand Tracking...",
           "Loading Color Palette...",
           "Initializing Brush Tools...",
           "Preparing Canvas...",
           "Ready to Paint!"
       ]
   )
   ```

#### **Phase 3: Integration into All Games**

5. **Update Each Game's `__init__.py`**
   - Import `UniversalLoadingScreen`
   - Add game-specific configuration
   - Replace any existing loading logic
   - Ensure consistent timing and user experience

6. **Games to Update:**
   - [ ] `airpainter` (already has loading screen - extract to universal)
   - [ ] `2048withswipes`
   - [ ] `tetris`
   - [ ] `snakemaster`
   - [ ] `pingpong`
   - [ ] `pongchallenge`
   - [ ] `virtualpong`
   - [ ] `tictactoe`
   - [ ] `memorymatch`
   - [ ] `quiz`
   - [ ] `spellingbee`
   - [ ] `gesturequiz`
   - [ ] `gesturesnake`
   - [ ] `gesturecommander`
   - [ ] `gesturedrawing`
   - [ ] `fruitninja`
   - [ ] `whackamole`
   - [ ] `duckhunt`
   - [ ] `archerysimulator`
   - [ ] `boxingtrainer`
   - [ ] `fitness`
   - [ ] `fitnesstrainer`
   - [ ] `virtualyogacoach`
   - [ ] `dance`
   - [ ] `dancebattle`
   - [ ] `circus`
   - [ ] `circusperformer`
   - [ ] `mariomaster`
   - [ ] `runneradventure`
   - [ ] `skidownhill`
   - [ ] `soccertrainer`
   - [ ] `holibowling`
   - [ ] `darts`
   - [ ] `shooter`
   - [ ] `eyeshotter`
   - [ ] `eyetictactoe`
   - [ ] `eyeblinkstory`
   - [ ] `face`
   - [ ] `facefilterfun`
   - [ ] `facepuzzle`
   - [ ] `smile`
   - [ ] `smiledetector`
   - [ ] `emotionmirror`
   - [ ] `selfie`
   - [ ] `selfiefun`
   - [ ] `mirror`
   - [ ] `story`
   - [ ] `teamcharades`
   - [ ] `languagetutor`
   - [ ] `virtualcalculator`
   - [ ] `virtualkeyboard`
   - [ ] `volume`
   - [ ] `soundconductor`
   - [ ] `armusicalinstrument`
   - [ ] `magicspellscasting`
   - [ ] `conductor`
   - [ ] `commander`
   - [ ] `designer`
   - [ ] `puzzlemaster`
   - [ ] `stackblocks`
   - [ ] `balanceboard`
   - [ ] `bodymovementchallenge`
   - [ ] `fingermaze`
   - [ ] `dragdrop`
   - [ ] `resizer`
   - [ ] `overlay`
   - [ ] `wireframe`
   - [ ] `pet simulator`
   - [ ] `chemistrylab`
   - [ ] `trafficcop`
   - [ ] `spaceinnovators`
   - [ ] `webshop`
   - [ ] `appstore`
   - [ ] `doomhands`
   - [ ] `drivemecrazy`
   - [ ] `footandshoot`
   - [ ] `hideandseek`
   - [ ] `jumpropesimulator`
   - [ ] `keyboard`
   - [ ] `mariomaster`
   - [ ] `rockpaperscissors`
   - [ ] `simonsays`
   - [ ] `tetristwist`
   - [ ] `virtualpong`

#### **Phase 4: Testing and Refinement**

7. **Quality Assurance**
   - Test loading screen on different screen resolutions
   - Ensure consistent timing across all games
   - Verify animations work smoothly
   - Test error handling and fallbacks
   - Performance optimization

8. **Documentation**
   - Create integration guide for new games
   - Document animation framework
   - Provide configuration examples
   - Add troubleshooting section

### 🎨 **Animation Framework Design**

#### **Built-in Animation Types:**

```python
ANIMATION_TYPES = {
    "rotating": "Rotating element animation",
    "pulsing": "Pulsing/breathing effect",
    "color_transition": "Smooth color transitions",
    "text_typing": "Typewriter text effect",
    "progress_bar": "Animated progress bar",
    "particle_system": "Particle effects",
    "game_specific": "Custom game animations"
}
```

#### **Custom Animation Interface:**

```python
def custom_animation(surface, progress, config):
    """Custom animation function for specific games"""
    # Game-specific animation logic
    pass
```

### 📊 **Success Criteria**

- [ ] All games have consistent loading experience
- [ ] Loading screen is configurable per game
- [ ] Animations are smooth and engaging
- [ ] 30-second duration is consistent
- [ ] Error handling works properly
- [ ] Performance is optimized
- [ ] Documentation is complete

### 🔄 **Migration Priority**

1. **High Priority:** Extract current airpainter loading screen
2. **Medium Priority:** Games with existing loading logic
3. **Low Priority:** Games without loading screens

### 📝 **Notes**

- Loading screen should not block the main thread
- Consider accessibility features (color blind friendly, etc.)
- Ensure loading screen works on different platforms
- Maintain consistent branding across all games

---

## Notes

- Utility/transition modules (e.g., `cvstore`, `exiting_state`, `loading_state`) do **not** require migration.
- Always update the `__init__.py` file for each game.
- If a game is a duplicate or superseded by another (e.g., `drive` vs `drivesimulator`), only migrate the preferred one.

---

**This TODO should be kept up to date as you migrate or implement each game.**
