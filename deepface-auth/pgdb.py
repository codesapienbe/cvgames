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
import gc
import traceback
import json
import hashlib
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deepface.log"),
        logging.StreamHandler()
    ]
)

base_model = "ArcFace"
base_detector = "retinaface"

# Database connection parameters
db_params = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "postgres"),
    "user": os.getenv("POSTGRES_USER", "yilmaz"),
    "password": os.getenv("POSTGRES_PASSWORD", "P@ssw0rd")
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
    """Initialize the database with required tables"""
    conn = get_connection()
    cur = conn.cursor()
    
    if drop_table:
        cur.execute("DROP TABLE IF EXISTS face_embeddings")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id SERIAL PRIMARY KEY,
            event_code VARCHAR(255),
            image_path TEXT,
            face_path TEXT,
            embedding vector(2622),  # ArcFace produces 2622-dimensional embeddings
            checksum VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    cur.close()
    conn.close()

async def extract_face_embeddings(img_path):
    """Extract face embeddings from an image"""
    try:
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name=base_model,
            detector_backend=base_detector,
            enforce_detection=False,
            align=True,  # Enable alignment for better accuracy
            normalization="base"  # Use base normalization for ArcFace
        )
        if not embedding:
            logging.error("No embedding was extracted")
            return None
        return embedding[0]['embedding']
    except Exception as e:
        logging.error(f"Error extracting embeddings: {str(e)}")
        return None

async def capture_face_by_coordinates(img_path, coordinates):
    """Capture face from image using coordinates"""
    try:
        img = cv2.imread(img_path)
        # Handle both dictionary and tuple formats
        if isinstance(coordinates, dict):
            x, y, w, h = coordinates['x'], coordinates['y'], coordinates['w'], coordinates['h']
        else:
            x, y, w, h = coordinates
        face = img[y:y+h, x:x+w]
        return face
    except Exception as e:
        logging.error(f"Error capturing face: {str(e)}")
        return None

async def extract_faces_by_coordinates(img_path, faces):
    """Extract faces by face coordinates from img_path"""
    try:
        img = cv2.imread(img_path)
        extracted_faces = []
        for face in faces:
            x, y, w, h = face
            face_img = img[y:y+h, x:x+w]
            extracted_faces.append(face_img)
        return extracted_faces
    except Exception as e:
        logging.error(f"Error extracting faces: {str(e)}")
        return []

async def extract_faces_base64(img_path, objs):
    """Extract faces and convert to base64"""
    try:
        faces = await extract_faces_by_coordinates(img_path, objs)
        base64_faces = []
        for face in faces:
            _, buffer = cv2.imencode('.jpg', face)
            base64_face = base64.b64encode(buffer).decode('utf-8')
            base64_faces.append(base64_face)
        return base64_faces
    except Exception as e:
        logging.error(f"Error extracting base64 faces: {str(e)}")
        return []

async def calculate_image_checksum(img_path):
    """Calculate checksum of an image"""
    try:
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logging.error(f"Error calculating checksum: {str(e)}")
        return None

async def image_already_processed(img_path, event_code):
    """Check if image has already been processed"""
    try:
        checksum = await calculate_image_checksum(img_path)
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM face_embeddings WHERE checksum = %s AND event_code = %s",
            (checksum, event_code)
        )
        result = cur.fetchone()
        cur.close()
        conn.close()
        return result is not None
    except Exception as e:
        logging.error(f"Error checking processed image: {str(e)}")
        return False

async def process_image(img_path, event_code):
    """Process a single image and store face embeddings"""
    try:
        if await image_already_processed(img_path, event_code):
            logging.info(f"Image already processed: {img_path}")
            return

        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=base_detector,
            enforce_detection=False
        )

        if not faces:
            logging.info(f"No faces found in image: {img_path}")
            return

        embedding = await extract_face_embeddings(img_path)
        if not embedding:
            return

        checksum = await calculate_image_checksum(img_path)
        face_path = f"faces/{uuid.uuid4()}.jpg"
        face = await capture_face_by_coordinates(img_path, faces[0]['facial_area'])
        if face is not None:
            cv2.imwrite(face_path, face)

        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO face_embeddings (event_code, image_path, face_path, embedding, checksum)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (event_code, img_path, face_path, embedding, checksum)
        )
        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        logging.error(f"Error processing image {img_path}: {str(e)}")
        traceback.print_exc()

async def process_images_from_dir(source_dir="events", drop_table=False):
    """Process all images from a directory"""
    await init_database(drop_table)
    
    for event_dir in os.listdir(source_dir):
        event_path = os.path.join(source_dir, event_dir)
        if os.path.isdir(event_path):
            for img_name in os.listdir(event_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(event_path, img_name)
                    await process_image(img_path, event_dir)

async def search_target_image(target_path, show_plots=False):
    """Search for similar faces in the database"""
    try:
        target_embedding = await extract_face_embeddings(target_path)
        if not target_embedding:
            return []

        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, event_code, image_path, face_path, embedding, 
                   (embedding <=> %s) as distance
            FROM face_embeddings
            ORDER BY distance
            LIMIT 5
            """,
            (target_embedding,)
        )
        results = cur.fetchall()
        cur.close()
        conn.close()

        if show_plots:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 6, 1)
            plt.imshow(cv2.imread(target_path))
            plt.title("Target")
            
            for i, result in enumerate(results, 1):
                plt.subplot(1, 6, i+1)
                plt.imshow(cv2.imread(result[3]))
                plt.title(f"Match {i}\nDistance: {result[5]:.4f}")

            plt.tight_layout()
            plt.show()

        return results

    except Exception as e:
        logging.error(f"Error searching target image: {str(e)}")
        return []

async def has_face(img_path):
    """Check if an image contains a face"""
    try:
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=base_detector,
            enforce_detection=False
        )
        return len(faces) > 0
    except Exception as e:
        logging.error(f"Error checking for faces: {str(e)}")
        return False 