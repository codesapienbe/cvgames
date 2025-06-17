# PLAYER

1. Run:
   ```bash
   python __init__.py [--camera N] [--width W] [--height H] [--resources Resources]
   ```

2. Play:
   - Raise only your thumb (gesture [1,0,0,0,0]) above the green threshold line to view the previous product image.
   - Raise only your pinky (gesture [0,0,0,0,1]) above the threshold to view the next product image.
   - Index finger only (gesture [0,1,0,0,0]) to show a pointer and point at the product.
   - Index + Middle fingers (gesture [0,1,1,0,0]) to draw/mark on the product image.
   - Index + Middle + Ring fingers (gesture [0,1,1,1,0]) to undo/delete the last drawing.

3. Quit: Press `q` to exit the application. 