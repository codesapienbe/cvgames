import paho.mqtt.client as mqtt
import json
import logging
import os
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

# MQTT connection parameters
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

class MQTTClient:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.task_results = {}
        self.connect()

    def connect(self):
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            logging.info(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        except Exception as e:
            logging.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("Connected to MQTT broker")
            # Subscribe to results topic
            self.client.subscribe("deepface/results/#")
        else:
            logging.error(f"Failed to connect to MQTT broker with code: {rc}")

    def on_message(self, client, userdata, msg):
        try:
            task_id = msg.topic.split('/')[-1]
            result = json.loads(msg.payload.decode())
            self.task_results[task_id] = result
            logging.info(f"Received result for task {task_id}")
        except Exception as e:
            logging.error(f"Error processing MQTT message: {e}")

    def publish_task(self, task_type, task_data):
        task_id = f"{task_type}_{os.urandom(4).hex()}"
        topic = f"deepface/tasks/{task_type}"
        
        message = {
            "task_id": task_id,
            "data": task_data
        }
        
        self.client.publish(topic, json.dumps(message))
        logging.info(f"Published task {task_id} to topic {topic}")
        return task_id

    def get_task_result(self, task_id):
        return self.task_results.get(task_id)

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        logging.info("Disconnected from MQTT broker") 