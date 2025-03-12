#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
import argparse
import sys
import signal

# Parse command-line arguments
parser = argparse.ArgumentParser(description='MediaPipe Smile Detection')
parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
parser.add_argument('--threshold', type=float, default=0.12, help='Smile detection threshold (default: 0.12)')
parser.add_argument('--reset', type=float, default=0.09, help='Smile reset threshold (default: 0.09)')
parser.add_argument('--debug', action='store_true', help='Enable debug visualization')
args = parser.parse_args()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define important mouth landmark indices
# Main lip landmarks
UPPER_LIP_TOP = 13      # Upper lip top
UPPER_LIP_BOTTOM = 14   # Upper lip bottom
LOWER_LIP_TOP = 17      # Lower lip top
LOWER_LIP_BOTTOM = 18   # Lower lip bottom

# Additional landmarks for better smile detection
LEFT_MOUTH_CORNER = 61   # Left corner of mouth
RIGHT_MOUTH_CORNER = 291 # Right corner of mouth
LEFT_CHEEK = 206         # Left cheek
RIGHT_CHEEK = 426        # Right cheek
LEFT_EDGE = 78           # Left edge of mouth
RIGHT_EDGE = 308         # Right edge of mouth

# Mouth shape landmarks
UPPER_OUTER_LIP_RIGHT = 38
UPPER_OUTER_LIP_LEFT = 0
LOWER_OUTER_LIP_RIGHT = 41
LOWER_OUTER_LIP_LEFT = 40

# Smile counter initialization
smile_count = 0
smiling = False  # To prevent counting the same smile multiple times
smile_duration = 0  # Track how long the person has been smiling
smile_history = []  # Keep a history of recent smile ratios for smoothing

# Global variable for camera
cap = None

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\nInterrupted by user (Ctrl+C)")
    cleanup()
    sys.exit(0)

# Clean up function
def cleanup():
    global cap
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. You smiled {smile_count} times!")

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Function to calculate the Euclidean distance between two points
def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

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
print(f"Smile threshold: {args.threshold}, Reset threshold: {args.reset}")
print("Press 'q' or 'ESC' to exit, or Ctrl+C in terminal")

# Initialize MediaPipe Face Mesh
try:
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error reading from camera")
                break
                
            # Flip the image horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the image to RGB for processing
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # To improve performance
            image_rgb.flags.writeable = False
            results = face_mesh.process(image_rgb)
            image_rgb.flags.writeable = True
            
            # Get image dimensions
            h, w, _ = frame.shape
            
            # Helper function to get normalized landmark coordinates
            def get_landmark(index):
                point = face_landmarks.landmark[index]
                return int(point.x * w), int(point.y * h)
            
            # Check if face landmarks were detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the face mesh if in debug mode
                    if args.debug:
                        mp_drawing.draw_landmarks(
                            frame,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                    else:
                        # Draw only the mouth contour for cleaner visualization
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_LIPS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                    
                    # Extract lip positions
                    upper_lip_top = get_landmark(UPPER_LIP_TOP)
                    upper_lip_bottom = get_landmark(UPPER_LIP_BOTTOM)
                    lower_lip_top = get_landmark(LOWER_LIP_TOP)
                    lower_lip_bottom = get_landmark(LOWER_LIP_BOTTOM)
                    left_corner = get_landmark(LEFT_MOUTH_CORNER)
                    right_corner = get_landmark(RIGHT_MOUTH_CORNER)
                    left_edge = get_landmark(LEFT_EDGE)
                    right_edge = get_landmark(RIGHT_EDGE)
                    left_cheek = get_landmark(LEFT_CHEEK)
                    right_cheek = get_landmark(RIGHT_CHEEK)
                    
                    # Additional landmarks for shape analysis
                    upper_outer_lip_right = get_landmark(UPPER_OUTER_LIP_RIGHT)
                    upper_outer_lip_left = get_landmark(UPPER_OUTER_LIP_LEFT)
                    lower_outer_lip_right = get_landmark(LOWER_OUTER_LIP_RIGHT)
                    lower_outer_lip_left = get_landmark(LOWER_OUTER_LIP_LEFT)
                    
                    # Compute vertical lip distance and mouth width
                    lip_distance = distance(upper_lip_bottom, lower_lip_top)
                    mouth_width = distance(left_corner, right_corner)
                    
                    # Compute mouth curvature (higher value = more curved/smiling)
                    mouth_height = (distance(upper_lip_top, lower_lip_bottom) + 
                                   distance(upper_outer_lip_left, lower_outer_lip_left) + 
                                   distance(upper_outer_lip_right, lower_outer_lip_right)) / 3
                    
                    # Check for upward curve of mouth (smiling)
                    center_y = (upper_lip_bottom[1] + lower_lip_top[1]) / 2
                    corner_avg_y = (left_corner[1] + right_corner[1]) / 2
                    curve = center_y - corner_avg_y  # Positive if corners are higher (smile)
                    
                    # Calculate the smile ratio, avoiding division by zero
                    if mouth_width > 0:
                        # Combine multiple factors for better smile detection
                        smile_ratio = (lip_distance / mouth_width) + (curve / 50)
                    else:
                        smile_ratio = 0
                    
                    # Add current smile ratio to history for smoothing
                    smile_history.append(smile_ratio)
                    if len(smile_history) > 5:  # Keep only the most recent values
                        smile_history.pop(0)
                    
                    # Get smoothed smile ratio
                    smoothed_ratio = sum(smile_history) / len(smile_history)
                    
                    # Determine if smiling based on the threshold
                    if smoothed_ratio > args.threshold:
                        smile_duration += 1
                        if smile_duration > 3 and not smiling:  # Require sustained smile (reduced from 5 to 3 frames)
                            smile_count += 1
                            smiling = True
                            # Print debug info when smile detected
                            print(f"Smile detected! Ratio: {smoothed_ratio:.3f}")
                    elif smoothed_ratio < args.reset:
                        smile_duration = 0
                        smiling = False
                    
                    # Draw landmarks for the mouth with different colors
                    if args.debug:
                        # Draw all landmarks
                        cv2.circle(frame, upper_lip_top, 3, (255, 0, 0), -1)       # Blue
                        cv2.circle(frame, upper_lip_bottom, 3, (0, 255, 0), -1)    # Green
                        cv2.circle(frame, lower_lip_top, 3, (0, 255, 0), -1)       # Green
                        cv2.circle(frame, lower_lip_bottom, 3, (255, 0, 0), -1)    # Blue
                        cv2.circle(frame, left_corner, 5, (0, 0, 255), -1)         # Red
                        cv2.circle(frame, right_corner, 5, (0, 0, 255), -1)        # Red
                        cv2.circle(frame, left_edge, 3, (255, 255, 0), -1)         # Yellow
                        cv2.circle(frame, right_edge, 3, (255, 255, 0), -1)        # Yellow
                        cv2.circle(frame, left_cheek, 3, (0, 255, 255), -1)        # Cyan
                        cv2.circle(frame, right_cheek, 3, (0, 255, 255), -1)       # Cyan
                        
                        # Draw lines for visualization
                        cv2.line(frame, left_corner, right_corner, (255, 0, 255), 1)  # Mouth width
                        cv2.line(frame, upper_lip_bottom, lower_lip_top, (0, 255, 255), 2)  # Lip distance
                    else:
                        # Just highlight the main smile indicators
                        cv2.circle(frame, left_corner, 5, (0, 0, 255), -1)         # Red
                        cv2.circle(frame, right_corner, 5, (0, 0, 255), -1)        # Red
                        
                        # Draw mouth outline
                        cv2.line(frame, left_corner, right_corner, (255, 255, 255), 1)
                    
                    # Show the smile ratio on screen
                    ratio_text = f"Smile Ratio: {smoothed_ratio:.3f} (Threshold: {args.threshold:.3f})"
                    cv2.putText(frame, ratio_text, (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                    
                    # Show smile status with emoji
                    status = "Smiling! 😊" if smiling else "Not smiling 😐"
                    status_color = (0, 255, 0) if smiling else (0, 0, 255)  # Green if smiling, red otherwise
                    cv2.putText(frame, status, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
                    
                    # Visual indicator of smile state
                    indicator_size = 40
                    indicator_pos = (w - indicator_size - 20, 40)
                    indicator_color = (0, 255, 0) if smiling else (0, 0, 255)
                    cv2.circle(frame, indicator_pos, indicator_size, indicator_color, -1)
                    
                    # Add smile duration meter
                    if smile_duration > 0 and not smiling:
                        progress = min(smile_duration / 3, 1.0)  # 3 frames needed for a smile
                        meter_width = 100
                        meter_height = 10
                        meter_pos = (w - meter_width - 20, 100)
                        # Background (grey)
                        cv2.rectangle(frame, meter_pos, (meter_pos[0] + meter_width, meter_pos[1] + meter_height), 
                                     (100, 100, 100), -1)
                        # Progress (green)
                        progress_width = int(progress * meter_width)
                        cv2.rectangle(frame, meter_pos, 
                                     (meter_pos[0] + progress_width, meter_pos[1] + meter_height),
                                     (0, 255, 0), -1)
            
            # Display smile count on the screen
            cv2.putText(frame, f"Smile Count: {smile_count}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Add camera info and exit instructions
            cv2.putText(frame, f"Camera: {args.camera} | Press 'q' or ESC to exit", 
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show the output
            cv2.imshow('MediaPipe Smile Detector', frame)
            
            # Check for exit keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' key
                break
            elif key == ord('d'):  # Toggle debug mode with 'd' key
                args.debug = not args.debug
                print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
except KeyboardInterrupt:
    # This handles Ctrl+C within the with block
    pass
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Make sure we clean up properly no matter what happened
    cleanup()