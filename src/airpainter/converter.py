import logging
from PIL import Image, ImageOps, ImageFilter
import os
import tempfile
import yt_dlp
import cv2
from tqdm import tqdm


logger = logging.getLogger(__name__)


# ---------- image → coloring page ---------- #
def convert_image_to_coloring_page(
        image_path: str,
        save_path: str
) -> str:
    """
    Turn a RGB image file into a high-contrast B/W outline drawing.
    Returns the path of the saved coloring page.
    """
    img = Image.open(image_path).convert("RGB")
    gray = ImageOps.grayscale(img)                       # remove colour
    edges = gray.filter(ImageFilter.FIND_EDGES)          # detect edges
    inverted = ImageOps.invert(edges)                    # white bg, dark lines
    threshold = 100
    is_black = lambda p: 0 if p < threshold else 255
    bw = inverted.point(is_black, mode="1")  # hard-threshold
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bw.save(save_path)
    return save_path

# ---------- YouTube video → many coloring pages ---------- #
def convert_video_to_coloring_page(
        youtube_url: str,
        save_dir: str = os.path.expanduser("~/.cvgames/airpainter/data/images/")
) -> list[str]:
    """
    1. Download the YouTube video.
    2. Grab one frame every 10 s.
    3. Convert each frame to a coloring page via convert_image_to_coloring_page().
    Returns a list of saved file paths.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. download video into /tmp --- #
    tmp_tpl = os.path.join(tempfile.gettempdir(), "airpainter_video.%(ext)s")
    ydl_opts = {"format": "best", "outtmpl": tmp_tpl, "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_path = ydl.prepare_filename(info)

    # --- 2. open video & compute timestamps --- #
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30          # fallback if missing
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    timestamps = range(0, int(duration), 10)       # 0 s, 10 s, 20 s, …

    # --- 3. iterate with progress bar --- #
    output_files = []
    for t in tqdm(timestamps, desc="Processing frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue                               # skip if read fails
        frame_path = os.path.join(tempfile.gettempdir(), f"airpainter_frame_{t}.png")
        cv2.imwrite(frame_path, frame)

        out_path = os.path.join(save_dir, f"coloring_page_{t}s.png")
        convert_image_to_coloring_page(frame_path, out_path)
        output_files.append(out_path)

    cap.release()
    os.remove(video_path)                          # tidy up

    return output_files
