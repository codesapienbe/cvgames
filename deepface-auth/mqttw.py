import paho.mqtt.client as mqtt
import json
import logging
import os
import tempfile
import shutil
from datetime import datetime
import zipfile
from dotenv import load_dotenv
from media import analyze_face, capture_face_frames, detect_faces, has_face, verify_face

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deepface.log"),
        logging.StreamHandler()
    ]
)

# MQTT connection parameters
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info("Connected to MQTT broker")
        # Subscribe to all task topics
        client.subscribe("deepface/tasks/#")
    else:
        logging.error(f"Failed to connect to MQTT broker with code: {rc}")

def process_detect_task(tmp_path):
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

def process_verify_task(src_path, tgt_path):
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

def process_analyze_task(tmp_path):
    try:
        analyzed_faces = analyze_face(tmp_path)
            
        if not analyzed_faces:
            logging.error("Analysis failed")
            return {"message": "Analysis failed", "status_code": 404}

        return [{
            "age": face["age"],
            "gender": face["gender"], 
            "dominant_emotion": face["dominant_emotion"],
            "dominant_race": face["dominant_race"]
        } for face in analyzed_faces]
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def process_count_task(tmp_path):
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

def process_capture_video_task(video_path):
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

def process_capture_photo_task(temp_files):
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

def on_message(client, userdata, msg):
    try:
        topic_parts = msg.topic.split('/')
        task_type = topic_parts[-1]
        message = json.loads(msg.payload.decode())
        task_id = message["task_id"]
        task_data = message["data"]

        logging.info(f"Processing task {task_id} of type {task_type}")

        # Process the task based on its type
        if task_type == "detect":
            result = process_detect_task(task_data["tmp_path"])
        elif task_type == "verify":
            result = process_verify_task(task_data["src_path"], task_data["tgt_path"])
        elif task_type == "analyze":
            result = process_analyze_task(task_data["tmp_path"])
        elif task_type == "count":
            result = process_count_task(task_data["tmp_path"])
        elif task_type == "capture_video":
            result = process_capture_video_task(task_data["video_path"])
        elif task_type == "capture_photo":
            result = process_capture_photo_task(task_data["temp_files"])
        else:
            result = {"error": f"Unknown task type: {task_type}"}

        # Publish the result
        result_topic = f"deepface/results/{task_id}"
        client.publish(result_topic, json.dumps(result))
        logging.info(f"Published result for task {task_id}")

    except Exception as e:
        logging.error(f"Error processing task: {e}")
        error_result = {"error": str(e)}
        result_topic = f"deepface/results/{task_id}"
        client.publish(result_topic, json.dumps(error_result))

def main():
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        logging.info(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.loop_forever()
    except Exception as e:
        logging.error(f"Failed to connect to MQTT broker: {e}")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main() 