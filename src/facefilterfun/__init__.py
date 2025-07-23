import cv2
import mediapipe as mp
import numpy as np
import os
import time
from PIL import Image
import io
import logging
import json
import subprocess
import platform
import urllib.request
import tempfile
import re
import webbrowser

# Structured logging setup for application.log
LOG_FILE = os.path.join(os.path.dirname(__file__), '../../application.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(message)s'
)

def log_structured(level, component, message, **kwargs):
    log_entry = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()),
        'level': level,
        'component': component,
        'message': message,
    }
    log_entry.update(kwargs)
    logging.log(getattr(logging, level), json.dumps(log_entry))

def is_imagemagick_installed():
    try:
        # Try to run 'magick -version' (newer) or 'convert -version' (older)
        result = subprocess.run(['magick', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return True
    except Exception:
        pass
    try:
        result = subprocess.run(['convert', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return True
    except Exception:
        pass
    return False

def get_latest_imagemagick_url():
    # Scrape the official download page for the latest Windows binary
    try:
        page_url = 'https://imagemagick.org/script/download.php#windows'
        with urllib.request.urlopen(page_url) as response:
            html = response.read().decode('utf-8')
        # Find the first .exe link for Windows 64-bit DLL
        match = re.search(r'href=["\'](https://imagemagick.org/download/binaries/ImageMagick-[^"\']+-x64-dll.exe)["\']', html)
        if match:
            return match.group(1)
    except Exception as e:
        log_structured('ERROR', 'facefilterfun', 'Failed to fetch latest ImageMagick URL', error=str(e))
    # Fallback to a known working version (may be outdated)
    return 'https://imagemagick.org/download/binaries/ImageMagick-7.1.1-32-Q16-HDRI-x64-dll.exe'


def install_imagemagick_windows():
    # Download and run the latest official ImageMagick installer for Windows
    IMAGEMAGICK_URL = get_latest_imagemagick_url()
    try:
        log_structured('INFO', 'facefilterfun', 'Downloading ImageMagick installer', url=IMAGEMAGICK_URL)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as tmp_file:
            urllib.request.urlretrieve(IMAGEMAGICK_URL, tmp_file.name)
            installer_path = tmp_file.name
        log_structured('INFO', 'facefilterfun', 'Running ImageMagick installer', installer=installer_path)
        # Run installer in silent mode
        subprocess.run([installer_path, '/silent'], check=True)
        log_structured('INFO', 'facefilterfun', 'ImageMagick installed successfully', installer=installer_path)
        os.remove(installer_path)
        return True
    except Exception as e:
        log_structured('ERROR', 'facefilterfun', 'Failed to auto-install ImageMagick', error=str(e), url=IMAGEMAGICK_URL)
        # Open the official download page in the user's browser
        webbrowser.open('https://imagemagick.org/script/download.php#windows')
        # Show manual install instructions
        blank = np.zeros((480, 800, 3), dtype=np.uint8)
        cv2.putText(blank, "ImageMagick auto-install failed.", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(blank, "A browser window has opened.", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(blank, "Download and install manually.", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(blank, "Then restart the game.", (60, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Face Filter Fun", blank)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
        import sys
        sys.exit(1)

def ensure_imagemagick():
    if platform.system() == 'Windows' and not is_imagemagick_installed():
        import ctypes
        MB_OKCANCEL = 1
        MB_ICONQUESTION = 0x20
        prompt = "ImageMagick is required for SVG mask conversion.\nDo you want to auto-install it now?"
        result = ctypes.windll.user32.MessageBoxW(0, prompt, "FaceFilterFun - Install Dependency", MB_OKCANCEL | MB_ICONQUESTION)
        if result == 1:  # OK
            if install_imagemagick_windows():
                # Add to PATH for current process
                magick_dir = r'C:\Program Files\ImageMagick-7.1.1-Q16-HDRI'
                os.environ['PATH'] = magick_dir + os.pathsep + os.environ['PATH']
                return True
            else:
                blank = np.zeros((480, 800, 3), dtype=np.uint8)
                cv2.putText(blank, "ImageMagick install failed.", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(blank, "See application.log for details.", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.imshow("Face Filter Fun", blank)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
                import sys
                sys.exit(1)
        else:
            log_structured('ERROR', 'facefilterfun', 'User declined ImageMagick install. Exiting.')
            blank = np.zeros((480, 800, 3), dtype=np.uint8)
            cv2.putText(blank, "ImageMagick is required.", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(blank, "Install manually and restart.", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Face Filter Fun", blank)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
            import sys
            sys.exit(1)
    elif not is_imagemagick_installed():
        log_structured('ERROR', 'facefilterfun', 'ImageMagick not found. Please install it and ensure it is in your PATH.')
        blank = np.zeros((480, 800, 3), dtype=np.uint8)
        cv2.putText(blank, "ImageMagick is required.", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(blank, "Install and add to PATH.", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Face Filter Fun", blank)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        import sys
        sys.exit(1)
    return True

# Pillow LANCZOS compatibility
try:
    from PIL import Image as _Image
    LANCZOS_RESAMPLING = getattr(_Image.Resampling, 'LANCZOS', None)
    if LANCZOS_RESAMPLING is None:
        raise AttributeError
except AttributeError:
    LANCZOS_RESAMPLING = getattr(Image, 'LANCZOS', None)
    if LANCZOS_RESAMPLING is None:
        LANCZOS_RESAMPLING = getattr(Image, 'BICUBIC', 0)

# README NOTE (for Windows users):
# To use SVG masks, you must install the native Cairo library (libcairo-2.dll).
# 1. Download Windows binaries from https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
# 2. Extract and add the bin/ directory (containing libcairo-2.dll) to your PATH.
# 3. Restart your terminal/IDE.
# 4. Ensure 'cairosvg' and 'cairocffi' are installed in your Python environment.
# If libcairo is missing, SVG masks will be skipped and errors will be logged to application.log.

def svg_to_png_wand(svg_path, png_path, width=512, height=512):
    try:
        from wand.image import Image as WandImage
        with WandImage(filename=svg_path, resolution=300) as img:
            img.format = 'png'
            img.resize(width, height)
            img.save(filename=png_path)
        log_structured('INFO', 'facefilterfun', f'Converted SVG to PNG with Wand', svg=svg_path, png=png_path, correlation_id=None)
        return True
    except Exception as e:
        log_structured('ERROR', 'facefilterfun', f'Failed to convert SVG to PNG with Wand', svg=svg_path, png=png_path, error=str(e), correlation_id=None)
        return False

def ensure_png_masks(mask_dir):
    """
    Convert all SVG masks in mask_dir to PNG if PNG is missing or SVG is newer.
    Returns a dict mapping mask base names to PNG file paths.
    """
    png_paths = {}
    try:
        for fname in os.listdir(mask_dir):
            if fname.lower().endswith('.svg'):
                base = os.path.splitext(fname)[0]
                svg_path = os.path.join(mask_dir, fname)
                png_path = os.path.join(mask_dir, base + '.png')
                convert = False
                if not os.path.exists(png_path):
                    convert = True
                else:
                    svg_mtime = os.path.getmtime(svg_path)
                    png_mtime = os.path.getmtime(png_path)
                    if svg_mtime > png_mtime:
                        convert = True
                if convert:
                    if not svg_to_png_wand(svg_path, png_path, 512, 512):
                        continue
                png_paths[base] = png_path
    except Exception as e:
        log_structured('ERROR', 'facefilterfun', 'Failed to process mask directory', mask_dir=mask_dir, error=str(e), correlation_id=None)
    return png_paths

class MaskResource:
    def __init__(self, path):
        self.path = path
        self.is_svg = self.path.lower().endswith('.svg')
        self.base_img = None  # Loaded as needed

    def get_mask(self, width, height):
        # Returns the mask as a BGRA (with alpha) numpy array at the desired size
        try:
            image = Image.open(self.path).convert('RGBA').resize((width, height), LANCZOS_RESAMPLING)
        except Exception as e:
            log_structured(
                'ERROR',
                'facefilterfun',
                f'Failed to load mask: {self.path}',
                error=str(e),
                correlation_id=None
            )
            image = Image.new('RGBA', (width, height), (0,0,0,0))
        return np.array(image)

class FaceFilter:
    def __init__(self, name, mask_resource, x_offset=0, y_offset=-0.4):
        self.name = name
        self.mask_resource = mask_resource
        self.x_offset = x_offset  # Mask X offset as a fraction of face width
        self.y_offset = y_offset  # Mask Y offset as a fraction of face height

    def apply(self, frame, face_landmarks, frame_shape):
        if not face_landmarks:
            return frame
        h, w = frame_shape
        # Get face bounds
        x_coords = [lm.x * w for lm in face_landmarks.landmark]
        y_coords = [lm.y * h for lm in face_landmarks.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        # Enlarge bounding box slightly
        face_width = x_max - x_min
        face_height = y_max - y_min
        mask_w = int(face_width * 1.2)
        mask_h = int(face_height * 1.3)
        x = int(x_min + self.x_offset * mask_w)
        y = int(y_min + self.y_offset * mask_h)
        # Get and resize mask
        mask_img = self.mask_resource.get_mask(mask_w, mask_h)
        # Overlay mask
        frame = overlay_image(frame, mask_img, x, y)
        return frame

def overlay_image(bg, fg, x, y):
    # Overlay a BGRA foreground (fg) onto bg at (x,y)
    h, w = bg.shape[:2]
    fg_h, fg_w = fg.shape[:2]
    if x < 0:  # Clamp to frame
        fg = fg[:, -x:]
        fg_w -= -x
        x = 0
    if y < 0:
        fg = fg[-y:, :]
        fg_h -= -y
        y = 0
    if x + fg_w > w:
        fg = fg[:, :w - x]
        fg_w = w - x
    if y + fg_h > h:
        fg = fg[:h - y, :]
        fg_h = h - y
    if fg_w <= 0 or fg_h <= 0:
        return bg
    alpha_mask = fg[:,:,3] / 255.0
    alpha_inv = 1.0 - alpha_mask
    for c in range(3):
        bg[y:y+fg_h, x:x+fg_w, c] = (
            alpha_mask * fg[:fg_h, :fg_w, c] + alpha_inv * bg[y:y+fg_h, x:x+fg_w, c]
        )
    return bg

def is_palm_open(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    landmarks = hand_landmarks.landmark
    return all(
        landmarks[tip].y < landmarks[pip].y
        for tip, pip in zip(tips_ids[1:], pip_ids[1:])  # Thumb excluded for robustness
    )

def both_hands_open(multi_hand_landmarks):
    return (
        multi_hand_landmarks
        and len(multi_hand_landmarks) >= 2
        and is_palm_open(multi_hand_landmarks[0])
        and is_palm_open(multi_hand_landmarks[1])
    )

class FaceFilterFun:
    def __init__(self, mask_dir='masks'):
        self.masks = []
        self.score = 0
        self.current_filter = 0
        self.filter_enabled = True
        self.switch_cooldown = 2.0  # seconds
        self.last_switch_time = 0
        self.game_time = 60
        self.start_time = time.time()
        self.mask_dir = mask_dir
        # Ensure all SVGs are converted to PNGs before loading masks
        self.png_mask_map = ensure_png_masks(self.mask_dir)
        self.load_masks()
        if not self.masks:
            log_structured('ERROR', 'facefilterfun', 'No masks loaded. Game cannot start.', mask_dir=self.mask_dir, correlation_id=None)

    def load_masks(self):
        # Edit this list to match your mask files and nice display names
        mask_files = [
            ('Elsa', 'elsa'),
            ('Olaf', 'olaf'),
            ('Mickey', 'mickey'),
            ('Monster High', 'monsterhigh'),
            # add more as needed
        ]
        self.masks = []
        for name, base in mask_files:
            png_path = self.png_mask_map.get(base)
            if png_path and os.path.exists(png_path):
                self.masks.append(FaceFilter(name, MaskResource(png_path)))
            else:
                # Fallback: try to load PNG directly if present
                fallback_png = os.path.join(self.mask_dir, base + '.png')
                if os.path.exists(fallback_png):
                    self.masks.append(FaceFilter(name, MaskResource(fallback_png)))
                else:
                    log_structured('WARN', 'facefilterfun', f'Mask PNG not found', mask=name, path=fallback_png, correlation_id=None)

    def apply_current_filter(self, frame, face_landmarks):
        if self.filter_enabled and face_landmarks and self.masks:
            f = self.masks[self.current_filter]
            frame = f.apply(frame, face_landmarks, frame.shape[:2])
        return frame

    def switch_filter(self):
        current_time = time.time()
        if current_time - self.last_switch_time > self.switch_cooldown:
            self.current_filter = (self.current_filter + 1) % len(self.masks)
            self.last_switch_time = current_time
            self.score += 5

    def draw_ui(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        elapsed = time.time() - self.start_time
        remains = max(0, self.game_time - elapsed)
        cv2.putText(frame, f"Mask: {self.masks[self.current_filter].name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Score: {self.score}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Time: {int(remains)}s", (w - 220, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, "Raise both hands with palms open to change mask", (w//2 - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,230,255), 2)
        if remains <= 0:
            cv2.putText(frame, "Game Over!", (w//2 - 150, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
            cv2.putText(frame, f"Final Score: {self.score}", (w//2 - 170, h//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'r' to restart or 'q' to quit", (w//2 - 320, h//2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

def main():
    solutions = mp.solutions
    mp_face_mesh = solutions.face_mesh
    mp_hands = solutions.hands
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2)
    mp_draw = solutions.drawing_utils
    filter_fun = FaceFilterFun()
    if not filter_fun.masks:
        # Show a user-friendly message in the OpenCV window and exit
        blank = np.zeros((480, 800, 3), dtype=np.uint8)
        cv2.putText(blank, "No masks available.", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        cv2.putText(blank, "Check application.log for details.", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Face Filter Fun", blank)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cv2.namedWindow("Face Filter Fun", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Face Filter Fun", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    last_hands_open = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)
        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        frame = filter_fun.apply_current_filter(frame, face_landmarks)
        if hand_results.multi_hand_landmarks:
            for hl in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            hands_open = both_hands_open(hand_results.multi_hand_landmarks)
            if hands_open and not last_hands_open:
                filter_fun.switch_filter()
            last_hands_open = hands_open
        else:
            last_hands_open = False
        filter_fun.draw_ui(frame)
        cv2.imshow("Face Filter Fun", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            filter_fun = FaceFilterFun()
            if not filter_fun.masks:
                blank = np.zeros((480, 800, 3), dtype=np.uint8)
                cv2.putText(blank, "No masks available.", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                cv2.putText(blank, "Check application.log for details.", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.imshow("Face Filter Fun", blank)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
                return
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ensure_imagemagick()
    main()
