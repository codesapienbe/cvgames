# DeepFace with PGVector

Face recognition and search system using DeepFace and PostgreSQL with pgvector.

## Requirements

- Python 3.8+
- PostgreSQL with pgvector extension
- Redis server (for Celery task queue)

## Installation

1. Install required Python packages:
   ```bash
   pip install deepface psycopg2-binary opencv-python matplotlib tqdm celery redis
   ```

2. Make sure PostgreSQL with pgvector extension is installed and running.

3. Make sure Redis is installed and running:
   ```bash
   # Check if Redis is running
   redis-cli ping
   
   # Start Redis if not running
   brew services start redis  # Mac
   # or
   sudo systemctl start redis  # Linux
   ```

## High-Performance Processing with Multiple Workers

For optimal performance when processing large image sets, we recommend using multiple Celery workers. The included script makes this easy:

```bash
# Make the script executable
chmod +x run_workers.sh

# Start 8 Celery workers (32 processing threads total)
./run_workers.sh
```

You can adjust the number of workers by editing the `NUM_WORKERS` variable in the script. By default, it starts 8 workers, each with 4 processing threads, for a total of 32 concurrent processing threads.

## Usage

### Process Images and Save Embeddings

```bash
# Process all images in the 'events' directory using Celery workers
python main.py --save --source events

# Process images without Celery (slower, sequential processing)
python main.py --save --source events --no-celery
```

### Search for a Target Image

```bash
# Search for a target image and show results as JSON
python main.py --target /path/to/target.jpg --search

# Search and display visual results
python main.py --target /path/to/target.jpg --search --show
```

### Initialize Database

```bash
python main.py --init
```

## Monitoring Celery Workers

Monitor the status of workers:
```bash
celery -A main status
```

View logs for a specific worker:
```bash
tail -f celery_worker1.log
```

Stop all workers:
```bash
pkill -f 'celery -A main worker'
```

## Performance Optimization

This system uses a multi-level approach to parallel processing:

1. **Multiple Celery worker processes** - Each worker process handles batches of images
2. **Thread-level parallelism** - Each worker uses 4 threads to process images within a batch
3. **Batch processing** - Images are processed in batches of 10 for efficient distribution

For the best performance on a multi-core system, use 8 or more worker processes.

## Troubleshooting

If you encounter errors, check:

1. PostgreSQL and Redis are running
2. The database has the pgvector extension installed
3. Celery workers are running (check with `celery -A main status`)
4. The logs for detailed error messages (`*.log` files)

## Advanced Configuration

Edit the `db_params` dictionary in `main.py` to configure your PostgreSQL connection.

# DeepFace with PGVector and Celery

This application uses DeepFace for face recognition, PostgreSQL with pgvector for vector storage, and Celery with Redis for distributed task processing.

## Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Redis (for Celery message broker)

## Installation

1. Install Python dependencies:
```
pip install deepface psycopg2-binary opencv-python matplotlib tqdm celery redis
```

2. Make sure Redis is running:
```
redis-cli ping
```
Should return `PONG`

3. Make sure PostgreSQL is running with pgvector extension installed

## Running with Celery

### Step 1: Start Celery Workers

In one terminal, start Celery workers:

```
cd deepface-pgvector
python celery_worker.py
```

Alternatively, you can use the standard Celery command directly:
```
celery -A celery_worker worker --loglevel=info
```

You can start multiple workers on different terminals to increase processing power.

### Step 2: Run the Main Application

In another terminal, run the application:

```
# Process images and save embeddings to database (using Celery)
python main.py --save --source ./events/YOUR_EVENT_FOLDER

# Search for matches to a target image (using Celery)
python main.py --target ./path/to/image.jpg --search
```

### Fallback to Sequential Processing

If Celery workers aren't available or you encounter problems, you can disable Celery:

```
# Without Celery (sequential processing)
python main.py --save --source ./events/YOUR_EVENT_FOLDER --no-celery

# Search without Celery
python main.py --target ./path/to/image.jpg --search --no-celery
```

## Troubleshooting Celery

If you encounter issues with Celery:

1. Check that Redis is running: `redis-cli ping`
2. Verify Celery workers are running: look for "Starting Celery worker" message
3. Try running with `--no-celery` flag to bypass Celery completely
4. Check the logs in `celery_worker.log` and `deepface_processing.log`
5. Increase timeout if tasks are taking too long (modify timeout in main.py)

Common issues:
- Timeout errors: Tasks are taking longer than the timeout period
- Connection errors: Redis may not be running or accessible
- Worker not available: Celery worker process may not be running

## Command Line Arguments

- `--save`: Process images and save embeddings to database
- `--source`: Source directory containing images to process (default: events)
- `--target`: Target image to search for or display
- `--search`: Search for the target image in the database
- `--show`: Show plots when searching (default: False, returns JSON)
- `--init`: Initialize the database
- `--no-celery`: Disable Celery task processing and use sequential processing

## Performance Considerations

- Using Celery allows for parallel processing of images, which can significantly improve performance
- Redis is used as the message broker and result backend for Celery
- Each image is processed as a separate task, allowing for better resource utilization
- Face detection and embedding extraction are computationally intensive operations
- The application automatically checks for Celery worker availability and falls back to sequential processing if needed

# 😎 Face Finder 🔍🤳

This is a cool computer program that can find faces in pictures and remember them! It's like teaching your computer to recognize your friends. ✨🧠💻

## 🌟 What This Program Can Do 🌟

- 👁️ Look at lots of pictures and find all the faces
- 🧠 Remember what each face looks like
- 🔄 Find pictures of the same person even if they look different (like smiling 😃 or frowning 😠)
- 🔎 Let you search for someone's face to see if they're in other pictures

## 🎮 How To Use It 🎮

You can talk to the program by typing special commands:

```
python main.py [options]
```

Here are the cool things you can tell it to do:

- `--save` 💾 : Tell the program to look at your pictures and remember the faces
- `--source FOLDER` 📁 : Tell it which folder has your pictures (if you don't say, it looks in a folder called "events")
- `--target PICTURE` 🎯 : Show it a picture of someone to find
- `--search` 🔍 : Ask it to look for the person in your picture
- `--show` 📺 : Ask it to show you the matches with pictures
- `--threads NUMBER` ⚡ : Make it work faster by using more brain power
- `--sequential` 🐢 : Make it work more carefully one picture at a time

### ✨ Examples ✨

1. To scan all your pictures and remember the faces:
   ```
   python main.py --save
   ```
   📸 → 💾

2. To scan pictures in your vacation folder:
   ```
   python main.py --save --source my_summer_vacation
   ```
   🏖️ → 💾

3. To find pictures of your friend from a photo:
   ```
   python main.py --target picture_of_my_friend.jpg --search
   ```
   👧 → 🔍

4. To find your friend AND see the matching pictures:
   ```
   python main.py --target picture_of_my_friend.jpg --search --show
   ```
   👧 → 🔍 → 📺

## 🧙‍♂️ How It Works 🪄

This program uses a special computer brain called "FaceNet" 🧠 that can look at a face and turn it into 128 numbers. These numbers describe the face - like how far apart the eyes are 👀 or how big the nose is 👃.

When the program remembers a face, it stores these numbers in a special box called a "database" 📦. Later, when you want to find someone, it compares their face numbers to all the saved face numbers and finds the closest matches! ✅

## 👁️ Face Detection Tools 🔍

The program can use different tools to find faces in pictures:

1. **RetinaFace** 🦅: This is super accurate but needs a powerful computer. 💪💻
2. **MTCNN** 🏆: This is pretty good and works well on most computers. 👍
3. **Dlib** 🚀: This is faster but might miss some faces. 💨
4. **OpenCV** ⚡: This is the fastest but not as good at finding all faces. 🏎️

## 🧠 Face Recognition Tools 🤔

After finding a face, the program uses these tools to remember it:

1. **FaceNet** 🌟: This is really good at remembering faces accurately. 💯
2. **ArcFace** 🏹: This is also very accurate. 🎯
3. **VGG-Face** 👤: This is not as good but works OK. 👌
4. **Facenet512** ⭐: This is a better version of FaceNet. 📈

## 🛠️ What You Need To Run This Program 🛠️

- A computer program called PostgreSQL 🐘 with something special added to it called pgvector 📊
- Some Python helper programs: 
  - 🐍 deepface
  - 🔌 psycopg2
  - 👁️ opencv-python
  - 📊 matplotlib
  - ⏱️ tqdm

Don't worry about all these fancy names - a grown-up can help you install them if you want to try the program! 👨‍👩‍👧‍👦 👩‍💻
