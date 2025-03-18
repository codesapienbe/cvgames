import asyncio
import logging
import os
from datetime import datetime
import shutil
import zipfile
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import uvicorn
from media import analyze_face, capture_face_frames, detect_faces, has_face, verify_face
from pgdb import init_database, process_image, search_target_image, process_images_from_dir
from dotenv import load_dotenv
from filex import get_local_dirs
from mqtt_client import MQTTClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deepface.log"),
        logging.StreamHandler()
    ]
)

logging.info("Initializing API...")
base_dir, events_dir, identities_dir, checksums_dir, faces_dir, audios_dir, lookups_dir = get_local_dirs()

# initialize database
init_database(
    drop_table=True
)

app = FastAPI(title="Deepface API", version="1.0.0")

# Initialize MQTT client
mqtt_client = MQTTClient()

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

@app.post("/face/detect/", summary="Detect faces in image")
async def detect(source: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = await source.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    task_id = mqtt_client.publish_task("detect", {"tmp_path": tmp_path})
    return {"task_id": task_id}

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
    
    task_id = mqtt_client.publish_task("verify", {
        "src_path": src_path,
        "tgt_path": tgt_path
    })
    return {"task_id": task_id}

@app.post("/face/analyze", summary="Analyze given image")
async def analyze(source: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = await source.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    task_id = mqtt_client.publish_task("analyze", {"tmp_path": tmp_path})
    return {"task_id": task_id}

@app.post("/face/count/", summary="Count faces in image")
async def count(source: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = await source.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    task_id = mqtt_client.publish_task("count", {"tmp_path": tmp_path})
    return {"task_id": task_id}

@app.post("/video/capture", summary="Extract frames containing faces from an uploaded video")
async def capture_video(
    video: UploadFile = File(...),
):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        content = await video.read()
        tmp_file.write(content)
        video_path = tmp_file.name

    task_id = mqtt_client.publish_task("capture_video", {"video_path": video_path})
    return {"task_id": task_id}

@app.post("/photo/capture", summary="Captures faces from a given photo archive and returns a list of photos containing at least one face in a zip file.")
async def capture_photo(
    photos: list[UploadFile] = File(...),
):
    temp_files = []
    try:
        for photo in photos:
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(photo.filename)[1], delete=False) as tmp_file:
                content = await photo.read()
                tmp_file.write(content)
                temp_files.append(tmp_file.name)

        task_id = mqtt_client.publish_task("capture_photo", {"temp_files": temp_files})
        return {"task_id": task_id}
    except Exception as e:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.post("/photo/upload", summary="Upload files to a pool")
async def photo_upload_api(
    photos: list[UploadFile] = File(...),
    target: str = Query(..., description="Target pool name")
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

        task_id = mqtt_client.publish_task("photo_upload", {
            "temp_files": temp_files,
            "filenames": filenames,
            "target": target
        })
        return {"task_id": task_id}
    
    except Exception as e:
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
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
        result = mqtt_client.get_task_result(task_id)
        if result is None:
            return {
                'status': 'PENDING',
                'message': 'Task is pending'
            }
        return {
            'status': 'COMPLETED',
            'result': result
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'status': 'error',
                'message': f'Error retrieving task result: {str(e)}'
            }
        )

@app.on_event("shutdown")
async def shutdown_event():
    mqtt_client.disconnect()

@app.post("/face/process/", summary="Process an image and store face embeddings")
async def process_face(
    source: UploadFile = File(...),
    event_code: str = Query(..., description="Event code for the image")
):
    """Process an image and store face embeddings in the database"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(source.filename)[1]) as temp_file:
            content = await source.read()
            temp_file.write(content)
            temp_file.flush()
            temp_path = temp_file.name

        await process_image(temp_path, event_code)
        os.unlink(temp_path)
        return JSONResponse({"message": "Image processed successfully"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/face/search/", summary="Search for similar faces in the database")
async def search_face(
    target: UploadFile = File(...),
    show_plots: bool = Query(False, description="Whether to show plots of matches")
):
    """Search for similar faces in the database"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(target.filename)[1]) as temp_file:
            content = await target.read()
            temp_file.write(content)
            temp_file.flush()
            temp_path = temp_file.name

        results = await search_target_image(temp_path, show_plots)
        os.unlink(temp_path)
        
        # Format results for JSON response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result[0],
                "event_code": result[1],
                "image_path": result[2],
                "face_path": result[3],
                "distance": float(result[5])
            })
        
        return JSONResponse({"matches": formatted_results})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/face/process_directory/", summary="Process all images from a directory")
async def process_directory(
    source_dir: str = Query(..., description="Source directory containing images"),
    drop_table: bool = Query(False, description="Whether to drop existing table before processing")
):
    """Process all images from a directory and store face embeddings"""
    try:
        await process_images_from_dir(source_dir, drop_table)
        return JSONResponse({"message": "Directory processed successfully"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

