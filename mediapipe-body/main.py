import cv2
import mediapipe as mp
import numpy as np
import argparse
import sys

# Parse command-line arguments
parser = argparse.ArgumentParser(description='MediaPipe Body Pose Detection')
parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
parser.add_argument('--min_detection_confidence', type=float, default=0.5, help='Minimum detection confidence (default: 0.5)')
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192)  # gray
with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
        )

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# Try to open the specified camera
try:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        available_cameras = []
        # Try to find available cameras
        for i in range(10):  # Try the first 10 camera indices
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                available_cameras.append(i)
                temp_cap.release()
        
        if available_cameras:
            print(f"Available cameras: {available_cameras}")
            print(f"Please run again with: --camera [index]")
        else:
            print("No cameras found")
        sys.exit(1)
except Exception as e:
    print(f"Error accessing camera: {e}")
    sys.exit(1)

print(f"Using camera index: {args.camera}")
print("Press 'q' or 'ESC' to exit")

with mp_pose.Pose(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        # Check for exit keys
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q' key
            break

cap.release()
cv2.destroyAllWindows()
print("Application closed")
