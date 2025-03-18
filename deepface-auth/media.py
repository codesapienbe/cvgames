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

load_dotenv()

# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_CPU_ALLOW_GROWTH"] = "true"

print("Initializing media utilities...")

base_dir, events_dir, identities_dir, checksums_dir, faces_dir, audios_dir, lookups_dir = get_local_dirs()
detector_backend = "retinaface" # Using RetinaFace for best face detection
print(f"Using face detector backend: {detector_backend}")

model_name = "ArcFace" # Using ArcFace for best face recognition
print(f"Using face recognition model: {model_name}")

# Default PostgreSQL connection string
pg_host="t2n-db"
pg_port=os.getenv("POSTGRES_PORT")
pg_connection_string = f"postgresql://postgres:postgres@${pg_host}:${pg_port}/postgres"

def detect_faces(img_path):
    print(f"Detecting faces in image: {img_path}")
    try:
        result = DeepFace.extract_faces(img_path=img_path, detector_backend=detector_backend, enforce_detection=False, align=True)
        print(f"Found {len(result)} faces")
        return result
    except Exception as e:
        print(f"Error detecting face: {e}")
        return None


def verify_face(img_path1, img_path2):
    print(f"Verifying faces between {img_path1} and {img_path2}")
    try:
        result = DeepFace.verify(img1_path=img_path1, img2_path=img_path2, detector_backend=detector_backend, model_name=model_name)
        print(f"Verification result: {result}")
        return result
    except Exception as e:
        print(f"Error verifying face: {e}")
        return None


def find_face(img_path, db_path):
    print(f"Finding faces in {img_path} using database at {db_path}")
    try:
        result = DeepFace.find(img_path=img_path, db_path=db_path, detector_backend=detector_backend, enforce_detection=False, model_name=model_name)       
        print(f"Found {len(result) if result else 0} matching faces")
        return result
    except Exception as e:
        print(f"Error finding face: {e}")
        return None


def analyze_face(img_path):
    print(f"Analyzing face in image: {img_path}")
    try:
        result = DeepFace.analyze(img_path=img_path, detector_backend=detector_backend)
        if result:
            print(f"Analysis complete. Results: {result}")
            return result
        else:
            print("No analysis results")
            return []
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return []


def zoom_face(image_base64, zoom_factor):
    print(f"Zooming face with factor: {zoom_factor}")
    try:
        print("Decoding base64 image")
        image = decode_image(image_base64)
        height, width = image.shape[:2]
        print(f"Original image dimensions: {width}x{height}")
        
        center_x, center_y = width//2, height//2
        new_width = int(width/zoom_factor)
        new_height = int(height/zoom_factor)
        print(f"New dimensions after zoom: {new_width}x{new_height}")
        
        x1 = center_x - new_width//2
        y1 = center_y - new_height//2
        x2 = x1 + new_width
        y2 = y1 + new_height
        print(f"Cropping coordinates: ({x1},{y1}) to ({x2},{y2})")
        
        cropped = image[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        print("Zoom operation complete")
        return zoomed
    except Exception as e:
        print(f"Error zooming face: {e}")
        return None


def lookup_faces(source_path, database_dir):
    print(f"Looking up faces from {source_path} in database: {database_dir}")
    found_faces = find_face(source_path, database_dir)
    if found_faces is None:
        print("No faces found")
        return []
    found_images = []
    for found_face in found_faces:
        print(f"Found matching face with {len(found_face['identity'])} images")
        found_images.extend(found_face['identity'])
    print(f"Total matching images found: {len(found_images)}")
    return found_images


def graph_faces(source_path, database_dir):
    print(f"Generating face connections graph for {source_path}")
    found_faces = find_face(source_path, database_dir)
    if found_faces is None:
        print("No faces found to graph")
        return
    print(found_faces)
    for found_face in found_faces:
        print("Creating graph visualization")
        # Create a graph
        G = nx.Graph()
        
        # Add nodes for each face
        for i, identity in enumerate(found_face['identity']):
            print(f"Adding node {i} for image: {identity}")
            G.add_node(i, image=identity)
            
        # Add edges between all faces
        for i in range(len(found_face['identity'])):
            for j in range(i+1, len(found_face['identity'])):
                print(f"Adding edge between nodes {i} and {j}")
                G.add_edge(i, j)
                
        print("Drawing graph")
        # Draw the graph
        plt.figure(figsize=(12,8))
        pos = nx.spring_layout(G, k=1.5) # Increase k to spread nodes further apart
        
        # Calculate node size based on figure size to ensure thumbnails fit
        fig = plt.gcf()
        fig_width, fig_height = fig.get_size_inches()
        node_size = min(fig_width, fig_height) * 100 # Scale node size with figure
        print(f"Using node size: {node_size}")
        
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=node_size, font_size=10, font_weight='bold')
        
        print("Adding face thumbnails")
        # Add face thumbnails as node images
        ax = plt.gca()
        thumbnail_size = 0.15 # Increase thumbnail size
        for node in G.nodes():
            print(f"Adding thumbnail for node {node}")
            img = plt.imread(G.nodes[node]['image'])
            # Center the thumbnail on the node position
            x, y = pos[node]
            ax_sub = ax.inset_axes([x - thumbnail_size/2, y - thumbnail_size/2, 
                                   thumbnail_size, thumbnail_size],
                                   transform=ax.transData)
            ax_sub.imshow(img)
            ax_sub.axis('off')
            
        # Adjust plot limits to ensure all thumbnails are visible
        plt.xlim(min(pos[node][0] for node in G.nodes()) - thumbnail_size,
                max(pos[node][0] for node in G.nodes()) + thumbnail_size)
        plt.ylim(min(pos[node][1] for node in G.nodes()) - thumbnail_size,
                max(pos[node][1] for node in G.nodes()) + thumbnail_size)
            
        plt.title("Face Connections Graph")
        plt.axis('off')
        plt.tight_layout()
        print("Displaying graph")
        plt.show()


def lookup_features(source_path, database_dir, age = None, gender = None, emotion = None, race = None):
    print(f"Looking up faces with specific features in {source_path}")
    print(f"Search criteria - Age: {age}, Gender: {gender}, Emotion: {emotion}, Race: {race}")
    
    found_faces = find_face(source_path, database_dir)
    if found_faces is None:
        print("No faces found")
        return []
        
    analyzed_faces = []
    for found_face in found_faces:
        face_to_analyze = found_face['identity']
        print(f"Analyzing face: {face_to_analyze}")
        analyzed_face = analyze_face(face_to_analyze)
        if analyzed_face:
            if age is not None and age[0] <= analyzed_face['age'] <= age[1]:
                print(f"Found face matching age criteria: {analyzed_face['age']}")
                analyzed_faces.append(analyzed_face)
            if gender is not None and gender == analyzed_face['gender']:
                print(f"Found face matching gender criteria: {analyzed_face['gender']}")
                analyzed_faces.append(analyzed_face)
            if emotion is not None and emotion == analyzed_face['dominant_emotion']:
                print(f"Found face matching emotion criteria: {analyzed_face['dominant_emotion']}")
                analyzed_faces.append(analyzed_face)
            if race is not None and race == analyzed_face['dominant_race']:
                print(f"Found face matching race criteria: {analyzed_face['dominant_race']}")
                analyzed_faces.append(analyzed_face)
    
    print(f"Total matching faces found: {len(analyzed_faces)}")
    return analyzed_faces


def login_face(source_path, identities_dir):
    print(f"Attempting face login with image: {source_path}")
    result = len(lookup_faces(source_path, identities_dir)) > 0
    print(f"Login {'successful' if result else 'failed'}")
    return result


def register_face(source_path, identity_name, identities_dir):
    print(f"Registering face for identity: {identity_name}")
    print(f"Source image: {source_path}")
    try:
        # Detect faces in the image
        print("Detecting faces in image")
        faces = detect_faces(source_path)
        if not faces:
            print("No faces detected in the image")
            return False
            
        # Create identity directory if it doesn't exist
        identity_dir = os.path.join(identities_dir, identity_name)
        print(f"Creating identity directory: {identity_dir}")
        if not os.path.exists(identity_dir):
            os.makedirs(identity_dir)
            
        # Use original filename for target
        filename = os.path.basename(source_path)
        target_path = os.path.join(identity_dir, filename)
        print(f"Target path: {target_path}")
        
        # Copy the image to identity directory
        print("Copying image to identity directory")
        shutil.copy2(source_path, target_path)
        print(f"Face image uploaded successfully to: {target_path}")
        return True
        
    except Exception as e:
        print(f"Error uploading image: {e}")
        return False


def delete_face(identity_name, identities_dir):
    print(f"Deleting face image: {identity_name}")
    try:
        absolute_path = os.path.join(identities_dir, identity_name)
        os.remove(absolute_path)
        print(f"Face image deleted successfully: {absolute_path}")
    except Exception as e:
        print(f"Error deleting face image: {e}")


def compare_images(frame1, frame2):
    print("Comparing two frames")
    if frame2 is None:
        print("Second frame is None, returning True")
        return True
    # Read the image file if frame1 is a filename
    if isinstance(frame1, str):
        print(f"Reading first frame from file: {frame1}")
        frame1 = cv2.imread(frame1)
    # Read the image file if frame2 is a filename  
    if isinstance(frame2, str):
        print(f"Reading second frame from file: {frame2}")
        frame2 = cv2.imread(frame2)
    print("Converting frames to grayscale")
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    similarity = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
    print(f"Frame similarity: {similarity}")
    return similarity < 0.8


def capture_face_frames(video_path, output_folder):
    print(f"Capturing face frames from video: {video_path}")
    print(f"Output folder: {output_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    print("Created output directory")
    
    print("Loading face cascade classifier")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_interval = int(fps * 2)  # 2 seconds interval
    frame_count = 0
    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print("Starting face detection from video frames...")
    last_saved_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate and display progress
        progress = (frame_count / total_frames) * 100
        print(f"\rProgress: {progress:.1f}% (Frame {frame_count}/{total_frames})", end="")

        # Only process frames at 2 second intervals
        if frame_count % frames_per_interval == 0:
            print(f"\nProcessing frame {frame_count}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                print(f"Found {len(faces)} faces in frame {frame_count}")
                if last_saved_frame is None or compare_images(frame, last_saved_frame):
                    frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                    print(f"Saving frame to: {frame_filename}")
                    cv2.imwrite(frame_filename, frame)
                    saved_count += 1
                    last_saved_frame = frame

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total frames with faces saved: {saved_count}")

def has_face(img_path):
    print(f"Processing image: {img_path}")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    try:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
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
            print(f"Found {len(faces)} faces in {img_path}")
            return True
        else:
            print(f"No faces found in {img_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def decode_image(image_base64):
    print("Decoding base64 image")
    img_data = base64.b64decode(image_base64)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(f"Decoded image shape: {image.shape}")
    return image


def encode_image(image_path):
    print(f"Encoding image to base64: {image_path}")
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
        print("Image encoded successfully")
        return encoded_string.decode('utf-8')

def enhance_image(image_base64):
    print("Enhancing image")
    try:
        print("Decoding base64 image")
        image = decode_image(image_base64)
        
        # Upscale using bicubic interpolation
        print("Upscaling image using bicubic interpolation")
        height, width = image.shape[:2]
        enhanced = cv2.resize(image, (width*4, height*4), interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening
        print("Applying sharpening filter")
        kernel = np.array([[-1,-1,-1], 
                         [-1, 9,-1],
                         [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Enhance contrast
        print("Enhancing contrast")
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        
        print(f"Enhanced image shape: {enhanced.shape}")
        return enhanced
        
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return image # Return original image if enhancement fails


def download_audio(source, target):
    print(f"Downloading audio from: {source}")
    print(f"Target path: {target}")
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',
            }],
            'outtmpl': target,
        }

        print("Starting download")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([source])
            
        print("Download complete")
        return target

    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None


def open_images(folder_path):
    print(f"Opening images from folder: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return []
        
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    try:
        print("Reading directory contents")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                filepath = os.path.join(folder_path, filename)
                try:
                    img = Image.open(filepath)
                    images.append(img)
                except Exception as e:
                    print(f"Error opening image {filename}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading folder {folder_path}: {e}")
        
    return images

