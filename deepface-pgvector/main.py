import os
from deepface import DeepFace
import psycopg2
from psycopg2 import pool
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm 
import time
import logging
import argparse
import concurrent.futures
from threading import Lock, Semaphore
import gc
import traceback
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deepface_processing.log"),
        logging.StreamHandler()
    ]
)

# Create a connection pool instead of a single connection
try:
    connection_pool = pool.ThreadedConnectionPool(
        minconn=5,
        maxconn=20,
        host="localhost",
        port="5432",
        database="postgres",
        user="yilmaz"
    )
    logging.info("Connection pool created successfully")
except Exception as e:
    logging.error(f"Error creating connection pool: {str(e)}")
    raise

# Create a semaphore to limit concurrent DeepFace operations
# This helps prevent segmentation faults by limiting concurrent access to computational resources
deepface_semaphore = Semaphore(2)  # Limit to 2 concurrent DeepFace operations

# Initialize database tables
def init_database():
    """Initialize the database with required tables and extensions"""
    conn = None
    cursor = None
    try:
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        
        # Create vector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Drop and recreate identities table
        cursor.execute("DROP TABLE IF EXISTS identities;")
        cursor.execute("""
            CREATE TABLE identities (
            ID INT PRIMARY KEY,
            IMG_NAME VARCHAR(100),
            embedding vector(128));
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
            connection_pool.putconn(conn)


def extract_face_embeddings(img_path):
    """
    Extract face embeddings from an image using DeepFace
    
    This function is separated to use a semaphore for limiting concurrent DeepFace operations
    """
    try:
        # Use a semaphore to limit concurrent DeepFace operations
        with deepface_semaphore:
            objs = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=False
            )
            
            # Create a copy of the results to avoid memory issues
            result = []
            for obj in objs:
                result.append(obj.copy())
            
            # Force garbage collection to free memory
            gc.collect()
            
            return result
    except Exception as e:
        logging.error(f"Error extracting face embeddings from {img_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return []


def process_image(img_path, start_idx):
    """
    Process a single image file, extract face embeddings and save to database
    
    Args:
        img_path: Path to the image file
        start_idx: Starting index for face IDs from this image
        
    Returns:
        tuple: (number of faces processed, list of face IDs)
    """
    conn = None
    cursor = None
    logging.info(f"Processing image: {img_path}")
    face_count = 0
    face_ids = []
    
    try:
        # Extract face embeddings with limited concurrency
        objs = extract_face_embeddings(img_path)
        
        if not objs:
            logging.info(f"No faces found in {img_path}")
            return face_count, face_ids
            
        logging.info(f"Found {len(objs)} faces in {img_path}")
        
        # Get a connection from the pool
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        
        # Process each face in the image
        for i, obj in enumerate(objs):
            embedding = obj["embedding"]
            face_id = start_idx + i
            face_ids.append(face_id)
            
            statement = f"""
                INSERT INTO identities
                    (id, img_name, embedding)
                    VALUES
                    ({face_id}, '{img_path}', '{str(embedding)}')
            """
            
            cursor.execute(statement)
            logging.info(f"Inserting face #{face_id} from {img_path} into identities table")
            face_count += 1
            
        # Commit after all faces from this image are processed
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
            connection_pool.putconn(conn)
        
    return face_count, face_ids


def process_images_from_dir(source_dir="events", max_workers=10):
    """
    Process all images in a directory using multi-threading
    
    Args:
        source_dir: Directory containing images to process
        max_workers: Maximum number of concurrent threads
        
    Returns:
        int: Total number of face embeddings inserted
    """
    # Initialize database structure
    try:
        init_database()
    except Exception as e:
        logging.error(f"Failed to initialize database: {str(e)}")
        return 0

    # Find all image files
    image_files = []
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_files.append(os.path.join(dirpath, filename))
    
    total_files = len(image_files)
    logging.info(f"Found {total_files} image files to process in {source_dir}")
    
    if total_files == 0:
        logging.warning(f"No image files found in {source_dir}")
        return 0
    
    # Adjust max_workers to be conservative
    max_workers = min(max_workers, 5)  # Limit to 5 workers maximum to avoid overloading
    max_batch_size = 3  # Limit batch size to prevent memory issues
    
    # Process files using multi-threading with limited concurrency
    total_faces = 0
    next_idx = 0
    
    with tqdm(total=total_files, desc="Processing images") as pbar:
        # Process files in small batches to control resource usage
        for i in range(0, total_files, max_batch_size):
            batch = image_files[i:i+max_batch_size]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
                # Start the processing tasks
                future_to_file = {
                    executor.submit(process_image, img_path, next_idx + idx): img_path 
                    for idx, img_path in enumerate(batch)
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    img_path = future_to_file[future]
                    try:
                        face_count, face_ids = future.result()
                        total_faces += face_count
                        if face_ids:
                            next_idx = max(next_idx, max(face_ids) + 1)
                    except Exception as exc:
                        logging.error(f'{img_path} generated an exception: {exc}')
                        logging.error(traceback.format_exc())
                    finally:
                        pbar.update(1)
            
            # Force garbage collection between batches
            gc.collect()
    
    logging.info(f"Total of {total_faces} face embeddings inserted into the database")

    # Create index after all processing is complete
    conn = None
    cursor = None
    try:
        conn = connection_pool.getconn()
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
            connection_pool.putconn(conn)
    
    return total_faces


def process_images_sequentially(source_dir="events"):
    """
    Process all images sequentially - no multithreading
    This is a safer alternative when multithreading causes segmentation faults
    """
    # Initialize database
    try:
        init_database()
    except Exception as e:
        logging.error(f"Failed to initialize database: {str(e)}")
        return 0
        
    # Find all image files
    image_files = []
    for dirpath, dirnames, filenames in os.walk(source_dir):
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
    
    for img_path in tqdm(image_files, desc="Processing images"):
        face_count, face_ids = process_image(img_path, next_idx)
        total_faces += face_count
        if face_ids:
            next_idx = max(next_idx, max(face_ids) + 1)
        # Force garbage collection after each image
        gc.collect()
    
    logging.info(f"Total of {total_faces} face embeddings inserted into the database")
    
    # Create index after all processing is complete
    conn = None
    cursor = None
    try:
        conn = connection_pool.getconn()
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
    finally:
        if cursor:
            cursor.close()
        if conn:
            connection_pool.putconn(conn)
    
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
    
    try:
        # Get the embedding of the target image
        with deepface_semaphore:
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

        # Get a connection from the pool
        conn = connection_pool.getconn()
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
            connection_pool.putconn(conn)
    
    return result_dict


def cleanup_connections():
    """Close all connections in the pool"""
    if 'connection_pool' in globals():
        connection_pool.closeall()
        logging.info("All database connections closed")


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

    parser.add_argument("--threads", type=int, default=3,
                        help="Number of concurrent threads for processing (default: 3)")
                        
    parser.add_argument("--sequential", action="store_true",
                        help="Process images sequentially (no multithreading)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        # Process images and save to database
        if args.save:
            logging.info(f"Processing images from source directory: {args.source}")
            
            if args.sequential:
                logging.info("Using sequential processing (no multithreading)")
                process_images_sequentially(source_dir=args.source)
            else:
                logging.info(f"Using multithreaded processing with {args.threads} threads")
                process_images_from_dir(source_dir=args.source, max_workers=args.threads)
        
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
            
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        # Make sure to clean up connections when done
        cleanup_connections()
