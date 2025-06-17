# MediaPipe Catalog

1. **Navigate Previous (Thumb Only)**
   - Raise only your thumb (other fingers down)
   - Gesture: [1, 0, 0, 0, 0]
   - Action: Moves to the previous product image

2. **Navigate Next (Pinky Only)**
   - Raise only your pinky finger (other fingers down)
   - Gesture: [0, 0, 0, 0, 1]
   - Action: Moves to the next product image

3. **Pointer Mode (Index Finger Only)**
   - Raise only your index finger
   - Gesture: [0, 1, 0, 0, 0]
   - Action: Shows a red circle pointer where you're pointing

4. **Drawing Mode (Index + Middle Fingers)**
   - Raise both index and middle fingers
   - Gesture: [0, 1, 1, 0, 0]
   - Action: Allows you to draw/mark on the product image

5. **Undo/Delete (Index + Middle + Ring Fingers)**
   - Raise index, middle, and ring fingers
   - Gesture: [0, 1, 1, 1, 0]
   - Action: Removes the last drawn item/marking

Additional notes:

- The gestures work when your hand is above the green threshold line shown in the video
- There's a small delay (30 frames) between gesture actions to prevent accidental triggers
- You can quit the application by pressing 'q' on your keyboard
- The small window in the top-right corner shows your camera feed for reference

The gesture format [x, x, x, x, x] represents the state of each finger from thumb to pinky, where 1 means raised and 0 means lowered.
