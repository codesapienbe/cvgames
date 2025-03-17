import base64
import os
import asyncio
from deepface import DeepFace
import psycopg2
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm 
import time
import logging
import argparse
import gc
import traceback
import json
import hashlib
import uuid
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deepface_processing.log"),
        logging.StreamHandler()
    ]
)

base_dir = os.getcwd()
events_dir = os.path.join(base_dir, "events")

base_model = "Facenet512"
base_detector = "retinaface"

# Database connection parameters
db_params = {
    "host": "localhost",
    "port": "5432",
    "database": "postgres",
    "user": "yilmaz"
}

def get_connection():
    """Get a database connection"""
    try:
        conn = psycopg2.connect(**db_params)
        return conn
    except Exception as e:
        logging.error(f"Error creating database connection: {str(e)}")
        raise

async def init_database(drop_table=False):
    """
    Initialize the database with required tables and extensions
    
    Args:
        drop_table: Whether to drop existing tables (default: False)
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Create vector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Check if table exists
        cursor.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'identities')")
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            if drop_table:
                logging.warning("Dropping existing identities table")
                cursor.execute("DROP TABLE IF EXISTS identities;")
                table_exists = False
            else:
                logging.info("Using existing identities table")
        
        # Create table if it doesn't exist
        if not table_exists:
            cursor.execute("""
                CREATE TABLE identities (
                    ID UUID PRIMARY KEY,
                    EVENT_CODE VARCHAR(100),
                    IMG_NAME VARCHAR(100),
                    EMBEDDING vector(512),
                    FACES_BASE64 TEXT,
                    CHECKSUM_SHA256 VARCHAR(64)
                );
            """)
            logging.info("Created new identities table")
        
        conn.commit()
        logging.info("Database initialized successfully")
    except Exception as e:
        if conn:
            conn.rollback()
        logging.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


async def extract_face_embeddings(img_path):
    try:
        objs = DeepFace.represent(
            img_path=img_path,
            model_name=base_model,
            detector_backend=base_detector,
            enforce_detection=False
        )
        
        result = []
        for obj in objs:
            result.append(obj.copy())
        
        gc.collect()
        # returns [ {'embedding': [...], 'facial_area': {'x': 100, 'y': 100, 'w': 100, 'h': 100}, 'landmarks': {'left_eye': (100, 100), 'right_eye': (100, 100), 'nose': (100, 100), 'mouth_left': (100, 100), 'mouth_right': (100, 100)}, 'region': {'x': 100, 'y': 100, 'w': 100, 'h': 100}, 'age': 20, 'gender': 'male', 'race': 'asian', 'emotion': 'happy', 'dominant_emotion': 'happy', 'dominant_race': 'asian', 'dominant_age': 20} ]
        return result
    except Exception as e:
        logging.error(f"Error extracting face embeddings from {img_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return []


async def capture_face_by_coordinates(img_path, coordinates):
    """
    Safely extract a face from an image using coordinates
    
    Args:
        img_path: Path to the image
        coordinates: Dictionary with x, y, w, h keys
        
    Returns:
        Face image or None if extraction fails
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Could not read image: {img_path}")
            return None
            
        # Get coordinates ensuring they're within image boundaries
        x = max(0, coordinates["x"])
        y = max(0, coordinates["y"])
        w = coordinates["w"]
        h = coordinates["h"]
        
        # Make sure coordinates don't exceed image dimensions
        height, width = img.shape[:2]
        if x + w > width:
            w = width - x
        if y + h > height:
            h = height - y
            
        if w <= 0 or h <= 0:
            logging.warning(f"Invalid face coordinates in {img_path}: {coordinates}")
            return None
            
        # Extract the face region
        face_img = img[y:y+h, x:x+w]
        if face_img.size == 0:
            logging.warning(f"Extracted empty face from {img_path} with coordinates {coordinates}")
            return None
            
        return face_img
    except Exception as e:
        logging.error(f"Error capturing face from {img_path}: {str(e)}")
        return None

async def extract_faces_by_coordinates(img_path, faces):
    # extract faces by face coordinates from img_path, using opencv
    result = []
    for face in faces:
        captured_face = await capture_face_by_coordinates(img_path, face["facial_area"])
        result.append(captured_face)
    return result


async def extract_faces_base64(img_path, objs):
    """
    Extract faces from image and convert to base64
    
    Args:
        img_path: Path to the image
        objs: List of objects with facial_area data from DeepFace.represent()
        
    Returns:
        Comma-separated string of base64-encoded face images
    """
    if not objs:
        logging.warning(f"No face objects provided for {img_path}")
        return ""
        
    try:
        # Debug log the objects structure
        logging.info(f"Processing {len(objs)} face objects for {img_path}")
        
        result = []
        for i, obj in enumerate(objs):
            try:
                if "facial_area" not in obj:
                    logging.warning(f"Missing facial_area in object {i} for {img_path}")
                    continue
                    
                facial_area = obj["facial_area"]
                logging.info(f"Processing face {i} with area: {facial_area}")
                
                # Capture the face from the image
                face_img = await capture_face_by_coordinates(img_path, facial_area)
                if face_img is None:
                    logging.warning(f"Failed to capture face {i} from {img_path}")
                    continue
                
                # Resize face to standard size
                resized_face = cv2.resize(face_img, (100, 100))
                
                # Encode to JPEG format in memory
                success, buffer = cv2.imencode('.jpg', resized_face)
                if not success:
                    logging.warning(f"Failed to encode face {i} from {img_path}")
                    continue
                    
                # Convert to base64
                face_base64 = base64.b64encode(buffer).decode("utf-8")
                result.append(face_base64)
                logging.info(f"Successfully encoded face {i} from {img_path}")
                
            except Exception as e:
                logging.error(f"Error processing face {i} from {img_path}: {str(e)}")
                continue
        
        # Join all encoded faces with commas
        if result:
            logging.info(f"Successfully encoded {len(result)} faces from {img_path}")
            return ",".join(result)
        else:
            logging.warning(f"No faces could be encoded from {img_path}")
            return ""
            
    except Exception as e:
        logging.error(f"Error in extract_faces_base64 for {img_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return ""

async def calculate_image_checksum(img_path):
    """
    Calculate SHA-256 checksum of image content
    
    Args:
        img_path: Path to the image
        
    Returns:
        str: Hex digest of SHA-256 hash of image content
    """
    try:
        with open(img_path, 'rb') as f:
            file_content = f.read()
            checksum = hashlib.sha256(file_content).hexdigest()
            logging.info(f"Generated content checksum for {os.path.basename(img_path)}: {checksum[:10]}...")
            return checksum
    except Exception as e:
        message = f"Error calculating content checksum for {img_path}: {str(e)}"
        logging.error(message)
        raise Exception(message)

async def image_already_processed(img_path, event_code):
    """
    Check if image has already been processed by looking for matching name and checksum
    
    Args:
        img_path: Path to the image
        event_code: Event code for the image
        
    Returns:
        bool: True if image is already in the database, False otherwise
    """
    conn = None
    cursor = None
    
    try:
        try:
            checksum = await calculate_image_checksum(img_path)
        except Exception as e:
            logging.error(f"Could not calculate checksum for {img_path}, assuming not processed: {str(e)}")
            return False
            
        img_name = os.path.basename(img_path)
        
        logging.info(f"Checking if image {img_name} with checksum {checksum[:10]}... exists in database")
        
        # Check database for this image
        conn = get_connection()
        cursor = conn.cursor()
        
        # Start with an exact match on checksum and name
        cursor.execute(
            """
            SELECT COUNT(*) FROM identities 
            WHERE img_name = %s AND checksum_sha256 = %s AND event_code = %s
            """, 
            (img_name, checksum, event_code)
        )
        
        count = cursor.fetchone()[0]
        
        if count > 0:
            logging.info(f"MATCH FOUND: {count} records for {img_name} with exact checksum match")
            return True
            
        # If that didn't match, try just by checksum as a fallback
        # This catches the same image with different names
        cursor.execute(
            "SELECT COUNT(*) FROM identities WHERE checksum_sha256 = %s",
            (checksum,)
        )
        
        count_by_checksum = cursor.fetchone()[0]
        if count_by_checksum > 0:
            logging.info(f"MATCH FOUND: {count_by_checksum} records with matching checksum {checksum[:10]}... (different names)")
            return True
            
        logging.info(f"No matching records found for image {img_name} with checksum {checksum[:10]}...")
        return False
        
    except Exception as e:
        logging.error(f"Error checking if image {img_path} is already processed: {str(e)}")
        logging.error(traceback.format_exc())
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

async def process_image(img_path, event_code):
    conn = None
    cursor = None
    
    logging.info(f"Processing image: {img_path}")
    face_count = 0

    if not os.path.exists(img_path):
        logging.error(f"Image not found: {img_path}")
        return -1
    
    # Check if image is already processed - using same checksum method
    if await image_already_processed(img_path, event_code):
        logging.info(f"Image {img_path} already processed, skipping")
        return 0
    
    if not await has_face(img_path):
        logging.info(f"No faces found in {img_path}")
        return 0

    try:
        # Extract face embeddings
        objs = await extract_face_embeddings(img_path)
        if not objs:
            logging.info(f"No faces found in {img_path} by DeepFace")
            return -1
            
        logging.info(f"DeepFace found {len(objs)} faces in {img_path}")
        
        conn = get_connection()
        cursor = conn.cursor()

        # Generate checksum once for this image based on content
        # Using same method as in image_already_processed
        try:
            checksum = await calculate_image_checksum(img_path)
        except Exception as e:
            logging.error(f"Error calculating checksum for {img_path}, cannot proceed: {str(e)}")
            return -1
            
        img_name = os.path.basename(img_path)
        
        # Log the checksum details for easier debugging
        logging.info(f"Storing image {img_name} with checksum {checksum[:10]}...")

        for i, obj in enumerate(objs):
            # Get specific embedding for this face
            embedding = obj["embedding"]
            
            # Get the facial area for this face
            if "facial_area" not in obj:
                logging.warning(f"Missing facial_area for face {i} in {img_path}, skipping")
                continue
                
            # Extract the specific face as base64
            face_img = await capture_face_by_coordinates(img_path, obj["facial_area"])
            if face_img is None:
                logging.warning(f"Failed to capture face {i} from {img_path}, skipping")
                continue
                
            # Resize and encode the face
            try:
                resized_face = cv2.resize(face_img, (100, 100))
                success, buffer = cv2.imencode('.jpg', resized_face)
                if not success:
                    logging.warning(f"Failed to encode face {i} from {img_path}, skipping base64")
                    face_base64 = ""
                else:
                    face_base64 = base64.b64encode(buffer).decode("utf-8")
            except Exception as e:
                logging.error(f"Error processing face image {i} from {img_path}: {str(e)}")
                face_base64 = ""
            
            # Generate a unique ID for this face
            face_id = uuid.uuid4()
            
            # Use parameterized query instead of string formatting for consistency
            try:
                cursor.execute(
                    """
                    INSERT INTO identities
                        (id, event_code, img_name, embedding, faces_base64, checksum_sha256)
                        VALUES
                        (%s, %s, %s, %s, %s, %s)
                    """,
                    (str(face_id), event_code, img_name, str(embedding), face_base64, checksum)
                )
                
                logging.info(f"Inserting face #{i+1} (ID: {face_id}) from {img_path} with checksum {checksum[:10]}...")
                face_count += 1
            except Exception as e:
                logging.error(f"Error inserting face #{i+1} from {img_path}: {str(e)}")
                continue
            
        conn.commit()
        logging.info(f"Committed changes for {img_path} with {face_count} faces")
        
        # Verify records were saved (using the same checksum method as validation)
        cursor.execute(
            "SELECT COUNT(*) FROM identities WHERE img_name = %s AND checksum_sha256 = %s",
            (img_name, checksum)
        )
        count = cursor.fetchone()[0]
        logging.info(f"Verification: Found {count} records for {img_name} with matching checksum after insertion")
        
    except Exception as e:
        if conn:
            conn.rollback()
        logging.error(f"Error processing {img_path}: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        
    return face_count


async def process_images_from_dir(source_dir="events", drop_table=False):
    """
    Process all images from a directory and save the embeddings to the database
    
    Args:
        source_dir: Directory containing images to process
        drop_table: Whether to drop existing tables (default: False)
    """
    # Initialize database without dropping the table by default
    try:
        await init_database(drop_table=drop_table)
    except Exception as e:
        logging.error(f"Failed to initialize database: {str(e)}")
        return 0
    
    # Find all image files
    image_files = []
    for dirpath, _, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_files.append(os.path.join(dirpath, filename))
    
    total_files = len(image_files)
    logging.info(f"Found {total_files} image files to process in {source_dir}")
    
    if total_files == 0:
        logging.warning(f"No image files found in {source_dir}")
        return 0
    
    # Process files sequentially but asynchronously
    total_faces = 0
    processed_images = 0
    skipped_images = 0
    event_code = source_dir.split("/")[-1]
    
    # Process sequentially
    for i, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        face_count = await process_image(img_path, event_code)
        if face_count == -1:  # Error occurred
            skipped_images += 1
            logging.info(f"Skipped image {i+1}/{total_files} - No faces detected: {img_path}")
            continue
        elif face_count == 0:  # Duplicate or no faces
            skipped_images += 1
            logging.info(f"Skipped image {i+1}/{total_files} - Already processed or no faces: {img_path}")
        else:
            total_faces += face_count
            processed_images += 1
            logging.info(f"Processed image {i+1}/{total_files} with {face_count} faces: {img_path}")
        
        # Force garbage collection after each image
        gc.collect()
    
    logging.info(f"Processing summary:")
    logging.info(f"- Total images: {total_files}")
    logging.info(f"- Successfully processed: {processed_images}")
    logging.info(f"- Skipped (duplicates or no faces): {skipped_images}")
    logging.info(f"- Total faces detected: {total_faces}")
    
    # Create index after all processing is complete if we processed images
    if processed_images > 0:
        conn = None
        cursor = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Check if index already exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 
                    FROM pg_indexes 
                    WHERE indexname LIKE '%identities%' AND indexdef LIKE '%hnsw%'
                )
            """)
            index_exists = cursor.fetchone()[0]
            
            if not index_exists:
                tic = time.time()
                logging.info("Creating HNSW index on embeddings...")
                cursor.execute(
                    "CREATE INDEX ON identities USING hnsw (embedding vector_l2_ops);"
                )
                conn.commit()
                toc = time.time()
                logging.info(f"Index created in {round(toc-tic, 2)} seconds")
            else:
                logging.info("HNSW index already exists on embeddings")
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Error creating index: {str(e)}")
            logging.error(traceback.format_exc())
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    return total_faces


async def show_target_image(target_path):
    """Display the target image"""
    logging.info(f"Loading target image: {target_path}")
    target_img = cv2.imread(target_path)

    if target_img is None:
        logging.error(f"Failed to load image: {target_path}")
        return

    plt.imshow(target_img[:, :, ::-1])
    plt.show()



async def has_face(img_path):
    """
    Detect if an image has faces using OpenCV - used for quick screening
    Note: This function is now primarily for quick screening, actual face
    count should come from DeepFace for accuracy.
    """
    print(f"Pre-screening image: {img_path}")
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
            return True
        else:
            return False
            
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
        return False

async def search_target_image(target_path, show_plots=False):
    """
    Search for matches to the target image in the database
    
    Args:
        target_path: Path to the target image
        show_plots: Whether to display plots of matches (default: False)
        
    Returns:
        dict: JSON-compatible dictionary with search results
    """
    logging.info(f"Searching for matches to target image: {target_path}")
    
    conn = None
    cursor = None
    result_dict = {
        "target_image": target_path,
        "matches": [],
        "status": "error",
        "message": ""
    }

    target_has_face = await has_face(target_path)
    if not target_has_face:
        msg = f"No faces detected in target image: {target_path}"
        logging.error(msg)
        result_dict["message"] = msg
        return result_dict
    
    try:
        # Get the embedding of the target image
        objs = DeepFace.represent(
            img_path=target_path,
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=False
        )

        if not objs:
            msg = f"No faces detected in target image: {target_path}"
            logging.error(msg)
            result_dict["message"] = msg
            return result_dict

        # Get the embedding of the target image
        target_embedding = objs[0]["embedding"]

        # Get a connection
        conn = get_connection()
        cursor = conn.cursor()
        
        # Search for the target image in the database
        cursor.execute(
            f"""
            SELECT * 
            FROM (
                SELECT i.id, i.img_name, i.embedding <-> '{str(target_embedding)}' AS distance
                FROM identities i
            ) a
            WHERE distance < 10
            ORDER BY distance ASC
            """
        )

        # Get the results
        results = cursor.fetchall()

        if not results:
            msg = "No matches found in the database"
            logging.info(msg)
            result_dict["status"] = "success"
            result_dict["message"] = msg
            return result_dict

        # Process the results
        logging.info(f"Found {len(results)} matches:")
        for i, result in enumerate(results):
            id, img_path, distance = result
            match_info = {
                "id": id,
                "image_path": img_path,
                "distance": float(distance)
            }
            result_dict["matches"].append(match_info)
            logging.info(f"Match #{i+1}: ID={id}, Image={img_path}, Distance={distance}")
            
            # Optionally display the matched images
            if show_plots:
                match_img = cv2.imread(img_path)
                if match_img is not None:
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.title("Target Image")
                    plt.imshow(cv2.imread(target_path)[:, :, ::-1])
                    plt.subplot(1, 2, 2)
                    plt.title(f"Match #{i+1} (Distance: {distance:.4f})")
                    plt.imshow(match_img[:, :, ::-1])
                    plt.show()
        
        result_dict["status"] = "success"
        result_dict["message"] = f"Found {len(results)} matches"
            
    except Exception as e:
        error_msg = f"Error searching for target image: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        result_dict["message"] = error_msg
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    return result_dict


# JSON logging for debugging face detection
async def log_to_json(data):
    """Log data to a JSON file for debugging"""
    log_file = "face_detection_log.json"
    
    # Create file if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([], f)
    
    # Read existing data
    try:
        with open(log_file, 'r') as f:
            existing_data = json.load(f)
    except:
        existing_data = []
    
    # Add new data with timestamp
    data['timestamp'] = datetime.now().isoformat()
    existing_data.append(data)
    
    # Write updated data
    with open(log_file, 'w') as f:
        json.dump(existing_data, f, indent=2)

async def main():
    args = parse_arguments()
    
    try:
        # Handle database initialization if requested
        if args.init:
            logging.info(f"Initializing database (drop table: {args.drop})")
            await init_database(drop_table=args.drop)
        
        # Process images and save to database
        if args.save:
            logging.info(f"Processing images from source directory: {args.source}")
            await process_images_from_dir(source_dir=args.source, drop_table=args.drop)
        
        # Handle target image operations
        if args.target:
            if os.path.exists(args.target):
                if args.search:
                    # Search directly
                    logging.info("Searching directly...")
                    results = await search_target_image(args.target, show_plots=args.show)
                    
                    # If not showing plots, print the JSON results
                    if not args.show:
                        print(json.dumps(results, indent=2))
                else:
                    # Just display the target image
                    await show_target_image(args.target)
            else:
                logging.error(f"Target image not found: {args.target}")
                
        # If no arguments provided, show usage
        if not (args.save or args.target or args.init):
            logging.info("No actions specified. Use --save to process images, --target to specify a target image, or --init to initialize the database.")
            logging.info("Run with --help for more information.")
            
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DeepFace with PGVector for face recognition and search")
    
    parser.add_argument("--save", action="store_true", 
                        help="Process images and save embeddings to database")
    
    parser.add_argument("--source", type=str, default="events", 
                        help="Source directory containing images to process (default: events)")
    
    parser.add_argument("--target", type=str, 
                        help="Target image to search for or display")
    
    parser.add_argument("--search", action="store_true", 
                        help="Search for the target image in the database")
                        
    parser.add_argument("--show", action="store_true",
                        help="Show plots when searching (default: False, returns JSON)")
    
    parser.add_argument("--init", action="store_true",
                        help="Initialize the database (doesn't drop tables by default)")
                        
    parser.add_argument("--drop", action="store_true",
                        help="Drop existing tables when initializing (use with --init or --save)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Run the async main
    asyncio.run(main())
