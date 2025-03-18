import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deepface.log"),
        logging.StreamHandler()
    ]
)

def get_local_dirs():
    
    base_dir = os.path.dirname(__file__)
    logging.info(f"Base directory set to: {base_dir}")

    events_dir = os.path.join(base_dir, "events")
    logging.info(f"Events directory set to: {events_dir}")

    identities_dir = os.path.join(base_dir, "identities")
    logging.info(f"Identities directory set to: {identities_dir}")

    checksums_dir = os.path.join(base_dir, "checksums")
    logging.info(f"Checksums directory set to: {checksums_dir}")

    faces_dir = os.path.join(base_dir, "faces")
    logging.info(f"Faces directory set to: {faces_dir}")

    audios_dir = os.path.join(base_dir, "audios")
    logging.info(f"Audios directory set to: {audios_dir}")

    lookups_dir = os.path.join(base_dir, "lookups")
    logging.info(f"Lookups directory set to: {lookups_dir}")

    return base_dir, events_dir, identities_dir, checksums_dir, faces_dir, audios_dir, lookups_dir