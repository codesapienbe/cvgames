import asyncio
import os
from datetime import datetime
import shutil
import sys
import zipfile
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import uvicorn
from celery import Celery
from media import analyze_face, capture_face_frames, create_collage, create_video_with_audio, detect_faces, has_face, lookup_faces, verify_face
from zipfile import ZipFile
from dotenv import load_dotenv
from filex import get_local_dirs
from docker import DockerClient

load_dotenv()

docker_client = DockerClient(base_url='unix://var/run/docker.sock')

print("Initializing API...")
base_dir, events_dir, identities_dir, checksums_dir, faces_dir, audios_dir, lookups_dir = get_local_dirs()

# start docker container for REDIS and POSTGRES
redis_container = docker_client.containers.run("redis:latest", detach=True)
postgres_container = docker_client.containers.run("postgres:latest", detach=True)

# wait for containers to start
redis_container.wait()
postgres_container.wait()

# check if containers are running
print(f"Redis container status: {redis_container.status}")
print(f"Postgres container status: {postgres_container.status}")

if redis_container.status != "running" or postgres_container.status != "running":
    print("Containers are not running")
    sys.exit(1)

app = FastAPI(title="Deepface API", version="1.0.0")

REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
if not REDIS_PASSWORD:
    raise ValueError("REDIS_PASSWORD is not set")

REDIS_HOST = os.getenv("REDIS_HOST")
if not REDIS_HOST:
    raise ValueError("REDIS_HOST is not set")

REDIS_PORT = os.getenv("REDIS_PORT")
if not REDIS_PORT:
    raise ValueError("REDIS_PORT is not set")
connection_link = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
print(f"Redis connection link: {connection_link}")

# This is just a template url for the lookups, there is no actual download happening here.
DOWNLOAD_BASE = "https://aws-bucket.ehb.be/lookups"
print(f"Download base: {DOWNLOAD_BASE}")


celery_app = Celery(
    "tasks", broker=connection_link, 
    backend=connection_link, 
    broker_connection_retry_on_startup=True)

# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_CPU_ALLOW_GROWTH"] = "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@celery_app.task
def lookup_task(tmp_path, eventid, fname, lname, phone, eventname):
    try:
        target_absolute_path = os.path.join(events_dir, eventid)
        matches = lookup_faces(tmp_path, target_absolute_path)
        if not matches:
            return {"message": "No matches found", "status_code": 404}
        
        # Create a unique filename for the zip file
        zip_filename = f'{fname}_{lname}_{eventid}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        zip_file_path = os.path.join(lookups_dir, zip_filename)
        
        # Create the zip file
        with ZipFile(zip_file_path, 'w') as zip_obj:
            for i, match in enumerate(matches):
                filename = os.path.basename(match)
                zip_obj.write(match, filename)
        
        # Generate download URL
        download_url = f'{DOWNLOAD_BASE}/{zip_filename}'

        return {
            "message": f"Created zip file with {len(matches)} matching images",
            "download_url": download_url,
            "filename": zip_filename,
            "status_code": 200
        }
    except Exception as e:
        return {"error": str(e), "status_code": 500}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/face/lookup/", summary="Find matching faces")
async def lookup(
    source: UploadFile = File(...),
    eventid: str = Query(..., description="Event ID"),
    fname: str = Query(None, description="First name"),
    lname: str = Query(None, description="Last name"),
    phone: str = Query(None, description="Phone number"),
    eventname: str = Query(..., description="Event name"),
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = await source.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    task = lookup_task.delay(tmp_path, eventid, fname, lname, phone, eventname)
    
    return {"task_id": task.id}


@celery_app.task
def detect_task(tmp_path):
    try:
        detection_results = detect_faces(tmp_path)
        if not detection_results:
            return {"face_count": 0, "faces": []}
            
        return {
            "face_count": len(detection_results),
            "faces": [{
                "confidence": result["confidence"],
                "facial_area": result["facial_area"]
            } for result in detection_results]
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/face/detect/", summary="Detect faces in image")
async def detect(source: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = await source.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    task = detect_task.delay(tmp_path)
    
    return {"task_id": task.id}


@celery_app.task
def verify_task(src_path, tgt_path):
    try:
        verification_result = verify_face(src_path, tgt_path)
        return {
            "verified": verification_result["verified"],
            "distance": verification_result["distance"],
            "threshold": verification_result["threshold"],
        }
    finally:
        if os.path.exists(src_path):
            os.unlink(src_path)
        if os.path.exists(tgt_path):
            os.unlink(tgt_path)


@app.post("/face/verify/", summary="Compare two faces")
async def verify(
    source: UploadFile = File(...),
    target: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False) as src_tmp, \
         tempfile.NamedTemporaryFile(delete=False) as tgt_tmp:
        
        src_content = await source.read()
        src_tmp.write(src_content)
        src_path = src_tmp.name
        
        tgt_content = await target.read()
        tgt_tmp.write(tgt_content)
        tgt_path = tgt_tmp.name
    
    task = verify_task.delay(src_path, tgt_path)
    
    return {"task_id": task.id}


@celery_app.task
def analyze_task(tmp_path):
    try:
        analyzed_faces = analyze_face(tmp_path)
            
        if not analyzed_faces:
            print("Analysis failed")
            return {"message": "Analysis failed", "status_code": 404}

        if len(analyzed_faces) > 1:
            print("\nNote: Multiple faces detected. Here is the analysis for each face:")
        else:
            print("\nFace Analysis:")

        return [{
            "age": face["age"],
            "gender": face["gender"], 
            "dominant_emotion": face["dominant_emotion"],
            "dominant_race": face["dominant_race"]
        } for face in analyzed_faces]
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/face/analyze", summary="Analyze given image")
async def analyze(source: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = await source.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    task = analyze_task.delay(tmp_path)
    
    return {"task_id": task.id}


@celery_app.task
def count_task(tmp_path):
    try:
        detection_results = detect_faces(tmp_path)
        if not detection_results:
            return {"face_count": 0}
            
        return {
            "face_count": len(detection_results)
        }
    except Exception as e:
        return {
            "error": str(e)
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/face/count/", summary="Count faces in image")
async def count(source: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = await source.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    task = count_task.delay(tmp_path)
    
    return {"task_id": task.id}


@celery_app.task
def collage_video_task(tmp_path, target):
    try:
        target_absolute_path = os.path.join(events_dir, target)
        matches = lookup_faces(tmp_path, target_absolute_path)
    
        if not matches:
            return {"message": "No matches found", "status_code": 404}

        video_path = tempfile.mktemp(suffix=".mp4")
        create_video_with_audio(
            [matches], # Wrap matches in list since function expects list of results
            output_path=video_path,
            fps=30,
            video_width=2560,
            video_height=1440,
            audio_path=os.path.join(audios_dir, "tiktok2024.mp3"),
            durationInMs=60000
        )
        
        return {
            "file_path": video_path,
            "filename": f"video_collage_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4",
            "status_code": 200
        }
    except Exception as e:
        return {"error": str(e), "status_code": 500}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/video/collage", summary="Create a video collage from given image that is found in the given capsule with an additional audio, preferablly background music.")
async def collage_video(
    source: UploadFile = File(...),
    target: str = Query(..., description="Target capsule name")
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = await source.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    task = collage_video_task.delay(tmp_path, target)
    
    return {"task_id": task.id}


@celery_app.task
def collage_photo_task(tmp_path, target):
    try:
        target_absolute_path = os.path.join(events_dir, target)
        matches = lookup_faces(tmp_path, target_absolute_path)
    
        if not matches:
            return {"message": "No matches found", "status_code": 404}

        photo_path = tempfile.mktemp(suffix=".jpg")
        create_collage(
            [matches],
            collage_width=2560,
            collage_height=1440,
            output_path=photo_path
        )
        
        return {
            "file_path": photo_path,
            "filename": f"photo_collage_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg",
            "status_code": 200
        }
    except Exception as e:
        return {"error": str(e), "status_code": 500}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/photo/collage", summary="Create a photo collage from given image that is found in the given capsule.")
async def collage_photo(
    source: UploadFile = File(...),
    target: str = Query(..., description="Target capsule name")
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = await source.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    task = collage_photo_task.delay(tmp_path, target)
    
    return {"task_id": task.id}


@celery_app.task
def capture_video_task(video_path):
    try:
        output_dir = tempfile.mkdtemp()
        capture_face_frames(video_path, output_dir)
        
        # Create zip file in temp directory
        zip_path = os.path.join(tempfile.gettempdir(), f"face_frames_{datetime.now().timestamp()}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        shutil.rmtree(output_dir)
        
        return {
            "file_path": zip_path,
            "filename": os.path.basename(zip_path),
            "status_code": 200
        }
    except Exception as e:
        return {"error": str(e), "status_code": 500}
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


@app.post("/video/capture", summary="Extract frames containing faces from an uploaded video")
async def capture_video(
    video: UploadFile = File(...),
):
    # Create temp files with delete=False
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        content = await video.read()
        tmp_file.write(content)
        video_path = tmp_file.name

    task = capture_video_task.delay(video_path)
    
    return {"task_id": task.id}


@celery_app.task
def capture_photo_task(temp_files):
    try:
        output_dir = tempfile.mkdtemp()
        photos_with_faces = []

        for img_path in temp_files:
            if has_face(img_path):
                photos_with_faces.append(img_path)
        
        # Create zip file in temp directory
        zip_path = os.path.join(tempfile.gettempdir(), f"face_frames_{datetime.now().timestamp()}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for photo_path in photos_with_faces:
                if os.path.exists(photo_path):
                    arcname = os.path.basename(photo_path)
                    zipf.write(photo_path, arcname)

        # Cleanup temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        shutil.rmtree(output_dir)
        
        return {
            "file_path": zip_path,
            "filename": os.path.basename(zip_path),
            "status_code": 200
        }
    except Exception as e:
        # Cleanup temp files in case of error
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        return {"error": str(e), "status_code": 500}


@app.post("/photo/capture", summary="Captures faces from a given photo archive and returns a list of photos containing at least one face in a zip file.")
async def capture_photo(
    photos: list[UploadFile] = File(...),
):
    temp_files = []
    try:
        # Create temp file for each uploaded photo
        for photo in photos:
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(photo.filename)[1], delete=False) as tmp_file:
                content = await photo.read()
                tmp_file.write(content)
                temp_files.append(tmp_file.name)

        task = capture_photo_task.delay(temp_files)
        
        return {"task_id": task.id}
    except Exception as e:
        # Cleanup temp files in case of error
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@celery_app.task
def update_model_task(target_dir):
    """
    Task to preprocess a directory by indexing all images for face recognition.
    This helps speed up subsequent lookups.
    """
    try:
        print(f"Pre-processing directory: {target_dir}")
        # Create a dummy image to trigger indexing of the entire directory
        with tempfile.NamedTemporaryFile(suffix=".jpg") as dummy_file:
            # Just create an empty file to trigger the indexing process
            dummy_file.write(b'')
            dummy_file.flush()
            
            # This will index the entire directory
            lookup_faces(dummy_file.name, target_dir)
            
        return {"message": f"Successfully preprocessed directory: {target_dir}"}
    except Exception as e:
        return {"error": str(e)}


@celery_app.task
def photo_upload_task(temp_files, filenames, target):
    target_dir = os.path.join(events_dir, target)
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        for i, tmp_path in enumerate(temp_files):
            target_path = os.path.join(target_dir, filenames[i])
            shutil.move(tmp_path, target_path)

        # Start preprocessing the directory in the background
        update_model_task.delay(target_dir)
        
        return {"message": f"Uploaded {len(filenames)} files to {target}", "received_count": len(filenames)}
    
    except Exception as e:
        # Cleanup only files that weren't successfully moved
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return {"error": str(e), "status_code": 500}


@app.post("/photo/upload", summary="Upload multiple files to a capsule")
async def photo_upload_api(
    photos: list[UploadFile] = File(...),
    target: str = Query(..., description="Target capsule name")
):
    temp_files = []
    filenames = []
    
    try:
        for photo in photos:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                content = await photo.read()
                tmp_file.write(content)
                temp_files.append(tmp_file.name)
                filenames.append(photo.filename)

        task = photo_upload_task.delay(temp_files, filenames, target)
        
        return {"task_id": task.id}
    
    except Exception as e:
        # Cleanup temp files in case of error
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@celery_app.task
def audio_upload_task(temp_files, filenames):
    valid_audio_extensions = ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac']
    uploaded_files = []
    
    try:
        # Create audios directory if it doesn't exist
        os.makedirs(audios_dir, exist_ok=True)
        
        for i, temp_file in enumerate(temp_files):
            # Check if the file has a valid audio extension
            file_ext = os.path.splitext(filenames[i])[1].lower()
            
            if file_ext not in valid_audio_extensions:
                continue  # Skip invalid files
                
            # Move the file to the target directory
            target_path = os.path.join(audios_dir, filenames[i])
            shutil.move(temp_file, target_path)
            uploaded_files.append({
                "filename": filenames[i],
                "path": target_path
            })
        
        if not uploaded_files:
            return {
                "error": f"No valid audio files found. Supported formats: {', '.join(valid_audio_extensions)}",
                "status_code": 400
            }
        
        return {
            "message": f"Successfully uploaded {len(uploaded_files)} audio files",
            "files": uploaded_files
        }
    
    except Exception as e:
        # Clean up any temporary files that might still exist
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        return {"error": str(e), "status_code": 500}


@app.post("/audio/upload", summary="Upload multiple audio files to audios directory")
async def upload_audio_api(
    audios: list[UploadFile] = File(...),
):
    temp_files = []
    filenames = []
    
    try:
        for audio in audios:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                content = await audio.read()
                tmp_file.write(content)
                temp_files.append(tmp_file.name)
                filenames.append(audio.filename)
        
        task = audio_upload_task.delay(temp_files, filenames)
        
        return {"task_id": task.id}
    
    except Exception as e:
        # Clean up any temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/tasks/{task_id}", summary="Get task result by task ID")
async def get_task_result(task_id: str):
    """
    Retrieve the result of a task by its task ID.
    
    Parameters:
    - task_id: The ID of the task to retrieve the result for
    
    Returns:
    - Task status and result if available
    """
    try:
        # Get the AsyncResult object for the task
        task_result = celery_app.AsyncResult(task_id)
        
        # Check the task state
        if task_result.state == 'PENDING':
            response = {
                'status': 'PENDING',
                'message': 'Task is pending'
            }
        elif task_result.state == 'FAILURE':
            response = {
                'status': 'FAILURE',
                'message': str(task_result.result),
                'traceback': task_result.traceback
            }
        else:
            # Task completed successfully
            response = {
                'status': task_result.state,
                'result': task_result.result
            }
        
        return response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'status': 'error',
                'message': f'Error retrieving task result: {str(e)}'
            }
        )

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    print("Celery worker configured successfully!")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
