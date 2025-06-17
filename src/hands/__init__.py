import cv2
import mediapipe as mp
import argparse
import sys
import signal

# Standard argument parsing
parser = argparse.ArgumentParser(description='MediaPipe Module')
parser.add_argument('--camera', type=int, default=0, 
                   help='Camera index to use (default: 0)')
parser.add_argument('--min_detection_confidence', type=float, default=0.5,
                   help='Minimum detection confidence (default: 0.5)')
args = parser.parse_args()

# Standard camera initialization
def init_camera(camera_index):
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            available_cameras = []
            for i in range(10):
                temp_cap = cv2.VideoCapture(i)
                if temp_cap.isOpened():
                    available_cameras.append(i)
                    temp_cap.release()
            
            if available_cameras:
                print(f"Available cameras: {available_cameras}")
                print(f"Run with: --camera [index]")
            else:
                print("No cameras found")
            sys.exit(1)
        return cap
    except Exception as e:
        print(f"Camera error: {e}")
        sys.exit(1)

# Standard cleanup function
def cleanup(cap):
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Application closed gracefully")

# Signal handler for Ctrl+C
def signal_handler(sig, frame):
    print("\nInterrupted by user")
    cleanup(cap)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --------------------------
# MODULE-SPECIFIC CODE BELOW
# --------------------------

# Initialize MediaPipe solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Camera setup
cap = init_camera(args.camera)
print(f"Using camera index: {args.camera}")
print("Press 'q' or ESC to exit")

with mp_hands.Hands(
    min_detection_confidence=args.min_detection_confidence,
    min_tracking_confidence=0.5
) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # MODULE-SPECIFIC PROCESSING
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # MODULE-SPECIFIC RENDERING
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        # Standard output
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        
        # Standard exit check
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):
            break

# --------------------------
# MODULE-SPECIFIC CODE ABOVE
# --------------------------

cleanup(cap)
