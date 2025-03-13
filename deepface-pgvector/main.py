import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from deepface import DeepFace
from sqlalchemy import create_engine, Column, Integer, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
import argparse
import sqlalchemy
import time  # Add time module for measuring execution time
import os
import numpy as np  # Add this import

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

Base = declarative_base()

class FaceEmbedding(Base):
    __tablename__ = 'face_embeddings'
    id = Column(Integer, primary_key=True)
    embedding = Column(Vector(512))  # ArcFace also uses 512 dimensions
    image_path = Column(sqlalchemy.String(255))

# Change the database connection URL from asyncpg to psycopg2
engine = create_engine('postgresql+psycopg2://yilmaz@localhost/postgres')

# Only create tables if they don't exist, otherwise update schema
inspector = inspect(engine)
if not inspector.has_table('face_embeddings'):
    Base.metadata.create_all(engine)
else:
    # Check if column exists and add it if not
    columns = [col['name'] for col in inspector.get_columns('face_embeddings')]
    if 'image_path' not in columns:
        # Use raw SQL for adding the column - more compatible across SQLAlchemy versions
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text(
                "ALTER TABLE face_embeddings ADD COLUMN image_path VARCHAR(255)"
            ))
            conn.commit()

Session = sessionmaker(bind=engine)
session = Session()

def save_face_embedding(image_path, detector_backend='mtcnn', model_name='Facenet512'):
    """Save face embeddings with optimized detection parameters"""
    print(f"Processing image: {image_path}")
    print(f"Using detector: {detector_backend}")
    print(f"Using model: {model_name}")
    
    # Try with requested model first
    try:
        results = DeepFace.represent(
            img_path=image_path, 
            model_name=model_name,
            enforce_detection=False,
            detector_backend=detector_backend
        )
    except Exception as e:
        print(f"Error with {model_name} model: {str(e)}")
        
        # If model fails, try Facenet512 as fallback
        if model_name != 'Facenet512':
            print(f"Falling back to Facenet512 model...")
            try:
                results = DeepFace.represent(
                    img_path=image_path,
                    model_name='Facenet512',
                    enforce_detection=False,
                    detector_backend=detector_backend
                )
                model_name = 'Facenet512'  # Update model name for later use
            except Exception as e2:
                print(f"Facenet512 model failed: {str(e2)}")
                # Final fallback - try OpenCV detector with Facenet512
                try:
                    results = DeepFace.represent(
                        img_path=image_path,
                        model_name='Facenet512',
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    model_name = 'Facenet512'
                    detector_backend = 'opencv'
                except Exception as e3:
                    print(f"All attempts failed. Final error: {str(e3)}")
                    return
        else:
            # Try OpenCV detector as fallback
            try:
                results = DeepFace.represent(
                    img_path=image_path,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                detector_backend = 'opencv'
            except Exception as e2:
                print(f"All attempts failed. Final error: {str(e2)}")
                return
    
    # Check if we got any results
    if not results:
        print(f"No faces detected in {image_path}")
        return
        
    # Count how many faces were found
    face_count = len(results)
    print(f"Found {face_count} faces in {image_path}")
    
    # Save each face embedding to the database
    for i, result in enumerate(results):
        embedding = result['embedding']
        face_info = f"{image_path} (face {i+1}/{face_count})"
        new_face = FaceEmbedding(embedding=embedding, image_path=face_info)
        
        # Measure database insertion time
        start_time = time.time()
        session.add(new_face)
        session.commit()
        end_time = time.time()
        
        # Calculate elapsed time in milliseconds
        elapsed_ms = (end_time - start_time) * 1000
        print(f"  Face {i+1}: Database insertion took {elapsed_ms:.2f} ms")
    
    print(f"Saved {face_count} face embeddings to database")

def find_similar_face(query_image_path, threshold=0.9, detector_backend='mtcnn', 
                      min_similarity=80.0, model_name='Facenet512'):
    """
    Find similar faces in the database with advanced matching parameters.
    
    Parameters:
        query_image_path (str): Path to the query image
        threshold (float): Base similarity threshold (0-1)
        detector_backend (str): Face detection backend
        min_similarity (float): Minimum similarity percentage (0-100) for matches
        model_name (str): Face recognition model to use
    """
    # Get embeddings for all faces in the query image
    print(f"Analyzing image: {query_image_path}")
    print(f"Using detector: {detector_backend}")
    print(f"Using model: {model_name}")
    print(f"Required minimum similarity: {min_similarity}%")
    
    # Try with requested model first
    try:
        results = DeepFace.represent(
            img_path=query_image_path,
            model_name=model_name,
            enforce_detection=False,
            detector_backend=detector_backend
        )
    except Exception as e:
        print(f"Error with {model_name} model: {str(e)}")
        
        # If model fails, try Facenet512 as fallback
        if model_name != 'Facenet512':
            print(f"Falling back to Facenet512 model...")
            try:
                results = DeepFace.represent(
                    img_path=query_image_path,
                    model_name='Facenet512',
                    enforce_detection=False,
                    detector_backend=detector_backend
                )
                model_name = 'Facenet512'  # Update model name for later use
            except Exception as e2:
                print(f"Facenet512 model failed: {str(e2)}")
                # Final fallback - try OpenCV detector with Facenet512
                try:
                    results = DeepFace.represent(
                        img_path=query_image_path,
                        model_name='Facenet512',
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    model_name = 'Facenet512'
                    detector_backend = 'opencv'
                except Exception as e3:
                    print(f"All attempts failed. Final error: {str(e3)}")
                    return
        else:
            # Try OpenCV detector as fallback
            try:
                results = DeepFace.represent(
                    img_path=query_image_path,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                detector_backend = 'opencv'
            except Exception as e2:
                print(f"All attempts failed. Final error: {str(e2)}")
                return
    
    if not results:
        print(f"No faces detected in {query_image_path}")
        return
        
    print(f"Found {len(results)} faces in query image")
    
    # Create an index on the embedding column if it doesn't exist
    try:
        with engine.connect() as conn:
            # Check if index exists
            exists = conn.execute(sqlalchemy.text(
                "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.relname = 'face_embeddings_embedding_idx'"
            )).scalar()
            
            if not exists:
                print("Creating vector index for faster similarity search...")
                conn.execute(sqlalchemy.text(
                    "CREATE INDEX face_embeddings_embedding_idx ON face_embeddings USING ivfflat "
                    "(embedding vector_l2_ops) WITH (lists = 100)"
                ))
                conn.commit()
                print("Vector index created successfully")
    except Exception as e:
        print(f"Warning: Could not create index: {str(e)}")
    
    # Find matches for each face
    for i, result in enumerate(results):
        query_embedding = np.array(result['embedding'])
        print(f"\nSearching for matches for face {i+1}/{len(results)}...")
        
        # Measure vector similarity search time
        start_time = time.time()
        
        # Get matches with distance information
        matches_query = session.query(
            FaceEmbedding.id, 
            FaceEmbedding.image_path,
            FaceEmbedding.embedding.l2_distance(query_embedding).label('distance')
        ).order_by('distance').limit(20)
        
        matches = matches_query.all()
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"  Vector similarity search took {elapsed_ms:.2f} ms")
        
        # Correct similarity calculation for Facenet512
        # Typical L2 distances: 0-20 for Facenet512, with smaller being better
        high_quality_matches = []
        
        for match in matches:
            # Convert distance to similarity score (0-100%)
            # For ArcFace, use a different similarity calculation
            # ArcFace typically has a smaller distance range (0-1.2 typically)
            if model_name == 'ArcFace':
                similarity = max(0, 100 * (1 - (match.distance / 1.2)))
            else:
                # For Facenet512 and others, use existing calculation
                similarity = max(0, 100 * (1 - (match.distance / 20.0)))
            
            if similarity >= min_similarity:
                match.similarity = similarity
                high_quality_matches.append(match)
        
        if high_quality_matches:
            print(f"Matches with ≥{min_similarity}% similarity for face {i+1}:")
            for j, match in enumerate(high_quality_matches):
                print(f"  Match {j+1}: ID {match.id} - {match.image_path}")
                print(f"     Similarity: {match.similarity:.2f}% (distance: {match.distance:.4f})")
        else:
            print(f"No matches found with ≥{min_similarity}% similarity for face {i+1}")
            if matches:
                best_match = matches[0]
                best_similarity = max(0, 100 * (1 - (best_match.distance / 20.0)))
                print("  Best available match (below similarity threshold):")
                print(f"     ID {best_match.id} - {best_match.image_path}")
                print(f"     Similarity: {best_similarity:.2f}% (distance: {best_match.distance:.4f})")
                print(f"     Required: {min_similarity}% similarity, missing: {min_similarity-best_similarity:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Face Embedding Management')
    parser.add_argument('--save', help='Path to image for saving embedding')
    parser.add_argument('--find', help='Path to image for finding similar face')
    parser.add_argument('--model', default='Facenet512',  # Changed default to Facenet512
                       choices=['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 
                                'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace'],
                       help='Face recognition model to use')
    parser.add_argument('--threshold', type=float, default=0.6, 
                        help='Base similarity threshold (0-1)')
    parser.add_argument('--similarity', type=float, default=80.0,
                        help='Minimum similarity percentage (0-100) for matches')
    parser.add_argument('--detector', default='mtcnn',
                        choices=['retinaface', 'opencv', 'mtcnn', 'ssd', 'dlib'],
                        help='Face detection backend')
    parser.add_argument('--show-all', action='store_true',
                        help='Show all matches regardless of similarity score')
    args = parser.parse_args()

    if args.save:
        save_face_embedding(args.save, detector_backend=args.detector, model_name=args.model)
    elif args.find:
        min_similarity = 0.0 if args.show_all else args.similarity
        find_similar_face(
            args.find, 
            threshold=args.threshold,
            detector_backend=args.detector,
            min_similarity=min_similarity,
            model_name=args.model
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

