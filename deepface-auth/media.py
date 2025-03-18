import os
import base64
import random
import numpy as np
import cv2
import shutil
from deepface import DeepFace
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from filex import get_local_dirs
from dotenv import load_dotenv
import logging
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deepface.log"),
        logging.StreamHandler()
    ]
)

# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_CPU_ALLOW_GROWTH"] = "true"

logging.info("Initializing media utilities...")

base_dir, events_dir, identities_dir, checksums_dir, faces_dir, audios_dir, lookups_dir = get_local_dirs()
detector_backend = "retinaface" # Using RetinaFace for best face detection
logging.info(f"Using face detector backend: {detector_backend}")

model_name = "ArcFace" # Using ArcFace for best face recognition
logging.info(f"Using face recognition model: {model_name}")

# Default PostgreSQL connection string
pg_host="t2n-db"
pg_port=os.getenv("POSTGRES_PORT")
pg_connection_string = f"postgresql://postgres:postgres@${pg_host}:${pg_port}/postgres"

def detect_faces(img_path):
    logging.info(f"Detecting faces in image: {img_path}")
    try:
        result = DeepFace.extract_faces(img_path=img_path, detector_backend=detector_backend, enforce_detection=False, align=True)
        logging.info(f"Found {len(result)} faces")
        return result
    except Exception as e:
        logging.info(f"Error detecting face: {e}")
        return None


def verify_face(img_path1, img_path2):
    logging.info(f"Verifying faces between {img_path1} and {img_path2}")
    try:
        result = DeepFace.verify(img1_path=img_path1, img2_path=img_path2, detector_backend=detector_backend, model_name=model_name)
        logging.info(f"Verification result: {result}")
        return result
    except Exception as e:
        logging.info(f"Error verifying face: {e}")
        return None


def find_face(img_path, db_path):
    logging.info(f"Finding faces in {img_path} using database at {db_path}")
    try:
        result = DeepFace.find(img_path=img_path, db_path=db_path, detector_backend=detector_backend, enforce_detection=False, model_name=model_name)       
        logging.info(f"Found {len(result) if result else 0} matching faces")
        return result
    except Exception as e:
        logging.info(f"Error finding face: {e}")
        return None


def analyze_face(img_path):
    logging.info(f"Analyzing face in image: {img_path}")
    try:
        result = DeepFace.analyze(img_path=img_path, detector_backend=detector_backend)
        if result:
            logging.info(f"Analysis complete. Results: {result}")
            return result
        else:
            logging.info("No analysis results")
            return []
    except Exception as e:
        logging.info(f"Error analyzing face: {e}")
        return []


def zoom_face(image_base64, zoom_factor):
    logging.info(f"Zooming face with factor: {zoom_factor}")
    try:
        logging.info("Decoding base64 image")
        image = decode_image(image_base64)
        height, width = image.shape[:2]
        logging.info(f"Original image dimensions: {width}x{height}")
        
        center_x, center_y = width//2, height//2
        new_width = int(width/zoom_factor)
        new_height = int(height/zoom_factor)
        logging.info(f"New dimensions after zoom: {new_width}x{new_height}")
        
        x1 = center_x - new_width//2
        y1 = center_y - new_height//2
        x2 = x1 + new_width
        y2 = y1 + new_height
        logging.info(f"Cropping coordinates: ({x1},{y1}) to ({x2},{y2})")
        
        cropped = image[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        logging.info("Zoom operation complete")
        return zoomed
    except Exception as e:
        logging.info(f"Error zooming face: {e}")
        return None

def lookup_features(source_path, database_dir, age = None, gender = None, emotion = None, race = None):
    logging.info(f"Looking up faces with specific features in {source_path}")
    logging.info(f"Search criteria - Age: {age}, Gender: {gender}, Emotion: {emotion}, Race: {race}")
    
    found_faces = find_face(source_path, database_dir)
    if found_faces is None:
        logging.info("No faces found")
        return []
        
    analyzed_faces = []
    for found_face in found_faces:
        face_to_analyze = found_face['identity']
        logging.info(f"Analyzing face: {face_to_analyze}")
        analyzed_face = analyze_face(face_to_analyze)
        if analyzed_face:
            if age is not None and age[0] <= analyzed_face['age'] <= age[1]:
                logging.info(f"Found face matching age criteria: {analyzed_face['age']}")
                analyzed_faces.append(analyzed_face)
            if gender is not None and gender == analyzed_face['gender']:
                logging.info(f"Found face matching gender criteria: {analyzed_face['gender']}")
                analyzed_faces.append(analyzed_face)
            if emotion is not None and emotion == analyzed_face['dominant_emotion']:
                logging.info(f"Found face matching emotion criteria: {analyzed_face['dominant_emotion']}")
                analyzed_faces.append(analyzed_face)
            if race is not None and race == analyzed_face['dominant_race']:
                logging.info(f"Found face matching race criteria: {analyzed_face['dominant_race']}")
                analyzed_faces.append(analyzed_face)
    
    logging.info(f"Total matching faces found: {len(analyzed_faces)}")
    return analyzed_faces


def login_face(source_path, identities_dir):
    pass


def register_face(source_path, identity_name, identities_dir):
    logging.info(f"Registering face for identity: {identity_name}")
    logging.info(f"Source image: {source_path}")
    try:
        # Detect faces in the image
        logging.info("Detecting faces in image")
        faces = detect_faces(source_path)
        if not faces:
            logging.info("No faces detected in the image")
            return False
            
        # Create identity directory if it doesn't exist
        identity_dir = os.path.join(identities_dir, identity_name)
        logging.info(f"Creating identity directory: {identity_dir}")
        if not os.path.exists(identity_dir):
            os.makedirs(identity_dir)
            
        # Use original filename for target
        filename = os.path.basename(source_path)
        target_path = os.path.join(identity_dir, filename)
        logging.info(f"Target path: {target_path}")
        
        # Copy the image to identity directory
        logging.info("Copying image to identity directory")
        shutil.copy2(source_path, target_path)
        logging.info(f"Face image uploaded successfully to: {target_path}")
        return True
        
    except Exception as e:
        logging.info(f"Error uploading image: {e}")
        return False


def delete_face(identity_name, identities_dir):
    logging.info(f"Deleting face image: {identity_name}")
    try:
        absolute_path = os.path.join(identities_dir, identity_name)
        os.remove(absolute_path)
        logging.info(f"Face image deleted successfully: {absolute_path}")
    except Exception as e:
        logging.info(f"Error deleting face image: {e}")


def compare_images(frame1, frame2):
    logging.info("Comparing two frames")
    if frame2 is None:
        logging.info("Second frame is None, returning True")
        return True
    # Read the image file if frame1 is a filename
    if isinstance(frame1, str):
        logging.info(f"Reading first frame from file: {frame1}")
        frame1 = cv2.imread(frame1)
    # Read the image file if frame2 is a filename  
    if isinstance(frame2, str):
        logging.info(f"Reading second frame from file: {frame2}")
        frame2 = cv2.imread(frame2)
    logging.info("Converting frames to grayscale")
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    similarity = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
    logging.info(f"Frame similarity: {similarity}")
    return similarity < 0.8


def capture_face_frames(video_path, output_folder):
    logging.info(f"Capturing face frames from video: {video_path}")
    logging.info(f"Output folder: {output_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    logging.info("Created output directory")
    
    logging.info("Loading face cascade classifier")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_interval = int(fps * 2)  # 2 seconds interval
    frame_count = 0
    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"Video FPS: {fps}")
    logging.info(f"Total frames: {total_frames}")
    logging.info("Starting face detection from video frames...")
    last_saved_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate and display progress
        progress = (frame_count / total_frames) * 100
        logging.info(f"\rProgress: {progress:.1f}% (Frame {frame_count}/{total_frames})", end="")

        # Only process frames at 2 second intervals
        if frame_count % frames_per_interval == 0:
            logging.info(f"\nProcessing frame {frame_count}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                logging.info(f"Found {len(faces)} faces in frame {frame_count}")
                if last_saved_frame is None or compare_images(frame, last_saved_frame):
                    frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                    logging.info(f"Saving frame to: {frame_filename}")
                    cv2.imwrite(frame_filename, frame)
                    saved_count += 1
                    last_saved_frame = frame

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    
    logging.info(f"\nProcessing complete:")
    logging.info(f"Total frames processed: {frame_count}")
    logging.info(f"Total frames with faces saved: {saved_count}")

def has_face(img_path):
    logging.info(f"Processing image: {img_path}")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    try:
        image = cv2.imread(img_path)
        if image is None:
            logging.info(f"Failed to load image: {img_path}")
            return False
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use more conservative parameters for better accuracy
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Smaller scale factor for more thorough scanning
            minNeighbors=6,    # More neighbors required for higher confidence
            minSize=(40, 40),  # Larger minimum face size to reduce false positives
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            logging.info(f"Found {len(faces)} faces in {img_path}")
            return True
        else:
            logging.info(f"No faces found in {img_path}")
            return False
            
    except Exception as e:
        logging.info(f"Error processing {img_path}: {e}")
        return False

def decode_image(image_base64):
    logging.info("Decoding base64 image")
    img_data = base64.b64decode(image_base64)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    logging.info(f"Decoded image shape: {image.shape}")
    return image


def encode_image(image_path):
    logging.info(f"Encoding image to base64: {image_path}")
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
        logging.info("Image encoded successfully")
        return encoded_string.decode('utf-8')

def enhance_image(image_base64):
    logging.info("Enhancing image")
    try:
        logging.info("Decoding base64 image")
        image = decode_image(image_base64)
        
        # Upscale using bicubic interpolation
        logging.info("Upscaling image using bicubic interpolation")
        height, width = image.shape[:2]
        enhanced = cv2.resize(image, (width*4, height*4), interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening
        logging.info("Applying sharpening filter")
        kernel = np.array([[-1,-1,-1], 
                         [-1, 9,-1],
                         [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Enhance contrast
        logging.info("Enhancing contrast")
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        
        logging.info(f"Enhanced image shape: {enhanced.shape}")
        return enhanced
        
    except Exception as e:
        logging.info(f"Error enhancing image: {e}")
        return image # Return original image if enhancement fails


def open_images(folder_path):
    logging.info(f"Opening images from folder: {folder_path}")
    if not os.path.exists(folder_path):
        logging.info(f"Folder not found: {folder_path}")
        return []
        
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    try:
        logging.info("Reading directory contents")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                filepath = os.path.join(folder_path, filename)
                try:
                    img = Image.open(filepath)
                    images.append(img)
                except Exception as e:
                    logging.info(f"Error opening image {filename}: {e}")
                    continue
    except Exception as e:
        logging.info(f"Error reading folder {folder_path}: {e}")
        
    return images

