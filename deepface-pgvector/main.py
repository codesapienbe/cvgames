import os
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

def init_database():
    """Initialize the database with required tables and extensions"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("DROP TABLE IF EXISTS identities;")
        cursor.execute("""
            CREATE TABLE identities (
                ID UUID PRIMARY KEY,
                EVENT_CODE VARCHAR(100),
                IMG_NAME VARCHAR(100),
                embedding vector(128),
                CHECKSUM_SHA256 VARCHAR(64)
            );
        """)
        
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


def extract_face_embeddings(img_path):
    try:
        objs = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet",
            enforce_detection=False
        )
        
        result = []
        for obj in objs:
            result.append(obj.copy())
        
        gc.collect()
        
        return result
    except Exception as e:
        logging.error(f"Error extracting face embeddings from {img_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return []


def process_image(img_path, event_code):
    conn = None
    cursor = None
    
    img_has_face = has_face(img_path)
    if not img_has_face:
        logging.info(f"No faces found in {img_path}")
        return -1, []
    
    logging.info(f"Processing image: {img_path}")
    face_count = 0
    
    try:
        objs = extract_face_embeddings(img_path)
        if not objs:
            logging.info(f"No faces found in {img_path}")
            return face_count
            
        logging.info(f"Found {len(objs)} faces in {img_path}")
        
        conn = get_connection()
        cursor = conn.cursor()
        
        for i, obj in enumerate(objs):
            embedding = obj["embedding"]
            face_id = uuid.uuid4()
            checksum = hashlib.sha256(img_path.encode()).hexdigest()
            statement = f"""
                INSERT INTO identities
                    (id, event_code, img_name, embedding, checksum_sha256)
                    VALUES
                    ('{face_id}', '{event_code}', '{os.path.basename(img_path)}', '{str(embedding)}', '{checksum}')
            """
            
            cursor.execute(statement)
            logging.info(f"Inserting face #{face_id} from {img_path} into identities table")
            face_count += 1
            
        conn.commit()
        logging.info(f"Committed changes for {img_path} with {face_count} faces")
        
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


def process_images_from_dir(source_dir="events"):
    """
    Process all images from a directory and save the embeddings to the database
    """
    # Initialize database
    try:
        init_database()
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
    
    # Process files sequentially
    total_faces = 0
    next_idx = 0
    event_code = source_dir.split("/")[-1]
    
    for img_path in tqdm(image_files, desc="Processing images"):
        face_count = process_image(img_path, event_code)
        if face_count == -1: # No faces found in image, skip it
            continue

        total_faces += face_count
        # Force garbage collection after each image
        gc.collect()
    
    logging.info(f"Total of {total_faces} face embeddings inserted into the database")
    
    # Create index after all processing is complete
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        tic = time.time()
        logging.info("Creating HNSW index on embeddings...")
        cursor.execute(
            "CREATE INDEX ON identities USING hnsw (embedding vector_l2_ops);"
        )
        conn.commit()
        toc = time.time()
        logging.info(f"Index created in {round(toc-tic, 2)} seconds")
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


def show_target_image(target_path):
    """Display the target image"""
    logging.info(f"Loading target image: {target_path}")
    target_img = cv2.imread(target_path)

    if target_img is None:
        logging.error(f"Failed to load image: {target_path}")
        return

    plt.imshow(target_img[:, :, ::-1])
    plt.show()



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

def search_target_image(target_path, show_plots=False):
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

    target_has_face = has_face(target_path)
    if not target_has_face:
        msg = f"No faces detected in target image: {target_path}"
        logging.error(msg)
        result_dict["message"] = msg
        return result_dict
    
    try:
        # Get the embedding of the target image
        objs = DeepFace.represent(
            img_path=target_path,
            model_name="Facenet",
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
            LIMIT 10
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


def cleanup_connections():
    """No longer needed as we're not using a connection pool"""
    logging.info("Cleanup not needed - using individual connections")


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
                        help="Initialize the database")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        # Process images and save to database
        if args.save:
            logging.info(f"Processing images from source directory: {args.source}")
            process_images_from_dir(source_dir=args.source)
        
        # Handle target image operations
        if args.target:
            if os.path.exists(args.target):
                if args.search:
                    # Search for matches to target image
                    results = search_target_image(args.target, show_plots=args.show)
                    
                    # If not showing plots, print the JSON results
                    if not args.show:
                        print(json.dumps(results, indent=2))
                else:
                    # Just display the target image
                    show_target_image(args.target)
            else:
                logging.error(f"Target image not found: {args.target}")
                
        # If no arguments provided, show usage
        if not (args.save or args.target):
            logging.info("No actions specified. Use --save to process images or --target to specify a target image.")
            logging.info("Run with --help for more information.")

        if args.init:
            init_database()
            
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
