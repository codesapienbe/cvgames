# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]
- Ongoing improvements and new game additions.

## [2024-06-10] Migration Complete: Pygame + OpenCV/MediaPipe + OpenTelemetry

### Major Migration
- Migrated all games in `src/` to use **Pygame** for rendering and the main loop.
- Retained and integrated **OpenCV + MediaPipe** (or cvzone/HandTrackingModule) for all computer vision input and gesture/pose/face tracking.
- Added **OpenTelemetry (otel)** compatible logging for key game events in every game.
- All changes applied directly to each game's `__init__.py` file.
- Preserved all business logic and CV-based interaction.

### New Game Implementations
- Implemented missing games using the new standard:
  - whackamole
  - wireframe
  - (and all other previously unimplemented games in the migration checklist)

### Migration Checklist
- All games listed in `TODO.md` have been migrated or implemented.
- The migration list is now empty and the checklist is up to date.

### Other
- Added persistent `TODO.md` to track migration and future games.
- Ensured all utility/transition modules are skipped as appropriate.

---

For future changes, please add new entries at the top of this file with the date and a summary of the changes. 