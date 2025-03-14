import os
import logging
from main import app

# Configure logging for the worker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [WORKER] %(message)s',
    handlers=[
        logging.FileHandler("celery_worker.log"),
        logging.StreamHandler()
    ]
)

# Configure worker concurrency based on available CPU cores
import multiprocessing
cpu_count = multiprocessing.cpu_count()
worker_concurrency = max(1, min(cpu_count - 1, 4))  # Use N-1 cores, but max 4

if __name__ == '__main__':
    logging.info(f"Starting Celery worker with concurrency={worker_concurrency}")
    app.worker_main(
        argv=[
            'worker',
            '--loglevel=INFO',
            f'--concurrency={worker_concurrency}',
            '-n=worker1@%h'
        ]
    ) 