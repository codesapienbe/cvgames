import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": ["cv2", "mediapipe", "numpy", "argparse", "signal"],
    "excludes": [],
    "include_files": [],
    "include_msvcr": True,
    "zip_include_packages": ["*"],
    "build_exe": "dist/EmotionDetector"
}

# GUI applications require a different base on Windows
base = None
if sys.platform == "win32":
    base = "Console"

setup(
    name="EmotionDetector",
    version="1.0",
    description="Emotion Detection using MediaPipe",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "main.py",
            base=base,
            target_name="EmotionDetector.exe",
            icon=None
        )
    ]
) 