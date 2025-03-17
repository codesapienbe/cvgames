#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
import argparse
import sys
import signal
import os
import json
import time
from typing import Tuple, List, Dict

# Add training data directory
TRAINING_DATA_DIR = "training_data"
if not os.path.exists(TRAINING_DATA_DIR):
    os.makedirs(TRAINING_DATA_DIR)

class FaceLandmarks:
    """Class to hold face landmark constants and helper methods"""
    # Face contour landmarks
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    # Mouth landmarks
    UPPER_LIP_TOP = 13
    UPPER_LIP_BOTTOM = 14
    LOWER_LIP_TOP = 17
    LOWER_LIP_BOTTOM = 18
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    LEFT_CHEEK = 206
    RIGHT_CHEEK = 426
    LEFT_EDGE = 78
    RIGHT_EDGE = 308
    UPPER_OUTER_LIP_RIGHT = 38
    UPPER_OUTER_LIP_LEFT = 0
    LOWER_OUTER_LIP_RIGHT = 41
    LOWER_OUTER_LIP_LEFT = 40
    
    # Additional mouth landmarks for better detection
    UPPER_LIP_MID = 0
    LOWER_LIP_MID = 17
    LEFT_MOUTH_INNER = 78
    RIGHT_MOUTH_INNER = 308
    UPPER_LIP_LEFT = 37
    UPPER_LIP_RIGHT = 267
    LOWER_LIP_LEFT = 84
    LOWER_LIP_RIGHT = 314

    # Eye landmarks
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_LEFT = 33
    LEFT_EYE_RIGHT = 133
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_LEFT = 362
    RIGHT_EYE_RIGHT = 263
    
    # Additional eye landmarks
    LEFT_EYE_TOP_2 = 158
    LEFT_EYE_BOTTOM_2 = 144
    RIGHT_EYE_TOP_2 = 385
    RIGHT_EYE_BOTTOM_2 = 373
    LEFT_EYE_IRIS = 468
    RIGHT_EYE_IRIS = 473
    LEFT_EYE_PUPIL = 474
    RIGHT_EYE_PUPIL = 475

    # Eyebrow landmarks
    LEFT_EYEBROW_TOP = 105
    LEFT_EYEBROW_BOTTOM = 107
    RIGHT_EYEBROW_TOP = 334
    RIGHT_EYEBROW_BOTTOM = 336
    LEFT_EYEBROW_LEFT = 70
    LEFT_EYEBROW_RIGHT = 63
    RIGHT_EYEBROW_LEFT = 300
    RIGHT_EYEBROW_RIGHT = 293

    # Nose landmarks
    NOSE_TIP = 1
    NOSE_BRIDGE = 6
    NOSE_LEFT = 219
    NOSE_RIGHT = 439
    NOSE_BOTTOM = 94
    NOSE_TOP = 5

    # Cheek and face landmarks
    LEFT_CHEEK_INNER = 206
    RIGHT_CHEEK_INNER = 426
    LEFT_CHEEK_OUTER = 187
    RIGHT_CHEEK_OUTER = 411
    CHIN_BOTTOM = 152
    FOREHEAD = 10
    TEMPLE_LEFT = 234
    TEMPLE_RIGHT = 454
    JAW_LEFT = 234
    JAW_RIGHT = 454

    @staticmethod
    def get_landmark(landmarks, index: int, image_width: int, image_height: int) -> Tuple[int, int]:
        """Helper method to get landmark coordinates"""
        point = landmarks.landmark[index]
        return int(point.x * image_width), int(point.y * image_height)

    @staticmethod
    def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    @staticmethod
    def get_eye_aspect_ratio(landmarks, eye_points: List[int], image_width: int, image_height: int) -> float:
        """Calculate the eye aspect ratio to detect blinks/closed eyes"""
        points = [FaceLandmarks.get_landmark(landmarks, p, image_width, image_height) for p in eye_points]
        vertical_dist = (FaceLandmarks.distance(points[1], points[2]) + 
                        FaceLandmarks.distance(points[3], points[4])) / 2
        horizontal_dist = FaceLandmarks.distance(points[0], points[5])
        if horizontal_dist == 0:
            return 0
        return vertical_dist / horizontal_dist

    @staticmethod
    def get_face_angle(landmarks, image_width: int, image_height: int) -> float:
        """Calculate the face angle (tilt)"""
        left_eye = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.LEFT_EYE_LEFT, image_width, image_height)
        right_eye = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.RIGHT_EYE_RIGHT, image_width, image_height)
        nose = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.NOSE_TIP, image_width, image_height)
        
        # Calculate angle between eyes and nose
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        dx = nose[0] - eye_center[0]
        dy = nose[1] - eye_center[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    @staticmethod
    def get_mouth_aspect_ratio(landmarks, image_width: int, image_height: int) -> float:
        """Calculate the mouth aspect ratio"""
        left_corner = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.LEFT_MOUTH_CORNER, image_width, image_height)
        right_corner = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.RIGHT_MOUTH_CORNER, image_width, image_height)
        upper_lip = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.UPPER_LIP_TOP, image_width, image_height)
        lower_lip = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.LOWER_LIP_BOTTOM, image_width, image_height)
        
        mouth_width = FaceLandmarks.distance(left_corner, right_corner)
        mouth_height = FaceLandmarks.distance(upper_lip, lower_lip)
        
        if mouth_width == 0:
            return 0
        return mouth_height / mouth_width

    @staticmethod
    def get_eyebrow_angle(landmarks, image_width: int, image_height: int, is_left: bool = True) -> float:
        """Calculate eyebrow angle"""
        if is_left:
            left = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.LEFT_EYEBROW_LEFT, image_width, image_height)
            right = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.LEFT_EYEBROW_RIGHT, image_width, image_height)
        else:
            left = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.RIGHT_EYEBROW_LEFT, image_width, image_height)
            right = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.RIGHT_EYEBROW_RIGHT, image_width, image_height)
        
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    @staticmethod
    def get_face_symmetry(landmarks, image_width: int, image_height: int) -> float:
        """Calculate face symmetry score"""
        # Get left and right eye positions
        left_eye = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.LEFT_EYE_LEFT, image_width, image_height)
        right_eye = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.RIGHT_EYE_RIGHT, image_width, image_height)
        
        # Get left and right mouth corners
        left_mouth = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.LEFT_MOUTH_CORNER, image_width, image_height)
        right_mouth = FaceLandmarks.get_landmark(landmarks, FaceLandmarks.RIGHT_MOUTH_CORNER, image_width, image_height)
        
        # Calculate face center line
        face_center_x = (left_eye[0] + right_eye[0]) / 2
        
        # Calculate distances from center
        left_eye_dist = abs(left_eye[0] - face_center_x)
        right_eye_dist = abs(right_eye[0] - face_center_x)
        left_mouth_dist = abs(left_mouth[0] - face_center_x)
        right_mouth_dist = abs(right_mouth[0] - face_center_x)
        
        # Calculate symmetry ratios
        eye_symmetry = 1 - abs(left_eye_dist - right_eye_dist) / max(left_eye_dist, right_eye_dist)
        mouth_symmetry = 1 - abs(left_mouth_dist - right_mouth_dist) / max(left_mouth_dist, right_mouth_dist)
        
        return (eye_symmetry + mouth_symmetry) / 2

class Emotion:
    def __init__(self, threshold: float, reset_threshold: float, name: str, emoji: str, color: Tuple[int, int, int]):
        self.threshold = threshold
        self.reset_threshold = reset_threshold
        self.name = name
        self.emoji = emoji
        self.color = color  # BGR color tuple
        self.count = 0
        self.duration = 0
        self.is_active = False
        self.history = []
        self.history_size = 5  # Reduced history size for faster response
        self.current_ratio = 0.0
        self.confidence = 0.0
        self.min_duration = 5  # Reduced minimum frames for faster response
        self.confidence_decay = 0.9  # Faster confidence decay
        self.confidence_threshold = 0.6  # Lower confidence threshold for better responsiveness

    def update_history(self, ratio: float) -> float:
        self.history.append(ratio)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        # Calculate weighted average with more weight on recent values
        weights = [0.2, 0.3, 0.5]  # Weights for the 3 most recent values
        recent_history = self.history[-3:] if len(self.history) >= 3 else self.history
        recent_weights = weights[-len(recent_history):]
        
        self.current_ratio = sum(r * w for r, w in zip(recent_history, recent_weights)) / sum(recent_weights)
        
        # Update confidence based on consistency
        if len(self.history) >= 2:
            variance = np.var(self.history[-3:]) if len(self.history) >= 3 else np.var(self.history)
            self.confidence = max(0, 1 - variance) * self.confidence_decay
        else:
            self.confidence = 0
            
        return self.current_ratio

    def detect(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        """
        Abstract method to be implemented by specific emotions.
        Returns (is_detected, ratio)
        """
        raise NotImplementedError

    def process_frame(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        is_detected, ratio = self.detect(landmarks, image_width, image_height)
        smoothed_ratio = self.update_history(ratio)
        
        # Enhanced emotion detection with confidence and duration requirements
        if smoothed_ratio > self.threshold and self.confidence > self.confidence_threshold:
            self.duration += 1
            if self.duration >= self.min_duration and not self.is_active:
                self.count += 1
                self.is_active = True
                print(f"{self.name} detected! Ratio: {smoothed_ratio:.3f}, Confidence: {self.confidence:.3f}")
        elif smoothed_ratio < self.reset_threshold or self.confidence < self.confidence_threshold:
            self.duration = max(0, self.duration - 1)  # Gradual decrease in duration
            if self.duration == 0:
                self.is_active = False
                self.confidence = 0
            
        return self.is_active, smoothed_ratio

class Happy(Emotion):
    def __init__(self, threshold: float = 0.12, reset_threshold: float = 0.08):
        super().__init__(threshold, reset_threshold, "Happy", "😊", (0, 255, 0))  # Green
        
    def detect(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        fl = FaceLandmarks
        get_landmark = lambda idx: fl.get_landmark(landmarks, idx, image_width, image_height)
            
        # Extract lip positions for smile detection
        upper_lip_bottom = get_landmark(fl.UPPER_LIP_BOTTOM)
        lower_lip_top = get_landmark(fl.LOWER_LIP_TOP)
        left_corner = get_landmark(fl.LEFT_MOUTH_CORNER)
        right_corner = get_landmark(fl.RIGHT_MOUTH_CORNER)
        
        # Get additional points for better smile detection
        upper_lip_top = get_landmark(fl.UPPER_LIP_TOP)
        lower_lip_bottom = get_landmark(fl.LOWER_LIP_BOTTOM)
        upper_lip_mid = get_landmark(fl.UPPER_LIP_MID)
        lower_lip_mid = get_landmark(fl.LOWER_LIP_MID)
        
        # Compute metrics
        mouth_width = fl.distance(left_corner, right_corner)
        
        # Calculate mouth center point
        center_x = (left_corner[0] + right_corner[0]) / 2
        center_y = (upper_lip_bottom[1] + lower_lip_top[1]) / 2
        
        # Calculate corner points relative to center
        left_corner_y = left_corner[1]
        right_corner_y = right_corner[1]
        
        # Calculate smile curve (positive means corners are below center)
        curve = center_y - (left_corner_y + right_corner_y) / 2
        
        # Calculate mouth width relative to face width
        face_width = fl.distance(get_landmark(fl.LEFT_CHEEK), get_landmark(fl.RIGHT_CHEEK))
        width_ratio = mouth_width / face_width if face_width > 0 else 0
        
        # Calculate mouth openness
        mouth_height = fl.distance(upper_lip_bottom, lower_lip_top)
        openness_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # Calculate mid-point curve
        mid_curve = lower_lip_mid[1] - upper_lip_mid[1]
        
        # Enhanced smile ratio calculation with more focus on curve
        if mouth_width > 0:
            # Consider multiple factors with adjusted weights:
            # 1. Upward curve of mouth (50% weight)
            # 2. Appropriate mouth width (20% weight)
            # 3. Moderate mouth openness (20% weight)
            # 4. Mid-point curve (10% weight)
            curve_ratio = min(max(curve / 5, 0), 1)  # Normalize curve with smaller divisor for more sensitivity
            width_ratio = min(max(width_ratio - 0.2, 0), 1)  # Adjusted range for more natural smiles
            openness = min(max(openness_ratio - 0.05, 0), 1)  # Adjusted range for more natural smiles
            mid_curve_ratio = min(max(mid_curve / 5, 0), 1)  # Normalize mid-point curve
            
            smile_ratio = (
                curve_ratio * 0.5 +
                width_ratio * 0.2 +
                openness * 0.2 +
                mid_curve_ratio * 0.1
            )
            
            # Apply a sigmoid function to make the detection more natural
            smile_ratio = 1 / (1 + np.exp(-5 * (smile_ratio - 0.5)))
        else:
            smile_ratio = 0
            
        return True, smile_ratio

class Sad(Emotion):
    def __init__(self, threshold: float = 0.2, reset_threshold: float = 0.15):
        super().__init__(threshold, reset_threshold, "Sad", "😢", (255, 0, 0))  # Blue
        
    def detect(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        def get_landmark(index):
            point = landmarks.landmark[index]
            return int(point.x * image_width), int(point.y * image_height)
            
        # Check for downturned mouth corners
        left_corner = get_landmark(FaceLandmarks.LEFT_MOUTH_CORNER)
        right_corner = get_landmark(FaceLandmarks.RIGHT_MOUTH_CORNER)
        center = get_landmark(FaceLandmarks.LOWER_LIP_BOTTOM)
        
        # Measure how much the corners are turned down
        corner_avg_y = (left_corner[1] + right_corner[1]) / 2
        mouth_width = FaceLandmarks.distance(left_corner, right_corner)
        
        # Enhanced sad ratio calculation
        if mouth_width > 0:
            # Consider both the corner downturn and mouth width
            downturn_ratio = max(0, (corner_avg_y - center[1]) / (mouth_width * 0.3))
            width_ratio = mouth_width / image_width  # Normalize by image width
            sad_ratio = downturn_ratio * 0.7 + width_ratio * 0.3
        else:
            sad_ratio = 0
            
        return True, sad_ratio

class Angry(Emotion):
    def __init__(self, threshold: float = 0.25, reset_threshold: float = 0.2):
        super().__init__(threshold, reset_threshold, "Angry", "😠", (0, 0, 255))  # Red
        
    def detect(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        fl = FaceLandmarks
        get_landmark = lambda idx: fl.get_landmark(landmarks, idx, image_width, image_height)
        
        # Get eyebrow positions
        left_eyebrow_left = get_landmark(fl.LEFT_EYEBROW_LEFT)
        left_eyebrow_right = get_landmark(fl.LEFT_EYEBROW_RIGHT)
        right_eyebrow_left = get_landmark(fl.RIGHT_EYEBROW_LEFT)
        right_eyebrow_right = get_landmark(fl.RIGHT_EYEBROW_RIGHT)
        
        # Get mouth positions
        upper_lip = get_landmark(fl.UPPER_LIP_TOP)
        lower_lip = get_landmark(fl.LOWER_LIP_BOTTOM)
        left_corner = get_landmark(fl.LEFT_MOUTH_CORNER)
        right_corner = get_landmark(fl.RIGHT_MOUTH_CORNER)
        
        # Calculate eyebrow angles
        left_eyebrow_angle = fl.get_eyebrow_angle(landmarks, image_width, image_height, True)
        right_eyebrow_angle = fl.get_eyebrow_angle(landmarks, image_width, image_height, False)
        
        # Calculate eyebrow height relative to eyes
        left_eye_top = get_landmark(fl.LEFT_EYE_TOP)
        right_eye_top = get_landmark(fl.RIGHT_EYE_TOP)
        
        left_eyebrow_height = abs(left_eyebrow_left[1] - left_eye_top[1])
        right_eyebrow_height = abs(right_eyebrow_left[1] - right_eye_top[1])
        
        # Calculate mouth metrics
        mouth_width = fl.distance(left_corner, right_corner)
        lip_distance = fl.distance(upper_lip, lower_lip)
        
        # Enhanced anger detection with multiple factors
        if mouth_width > 0:
            # 1. Eyebrow angle (40% weight)
            # Angry eyebrows are typically angled downward
            left_eyebrow_score = min(max((left_eyebrow_angle + 15) / 30, 0), 1)
            right_eyebrow_score = min(max((right_eyebrow_angle + 15) / 30, 0), 1)
            eyebrow_score = (left_eyebrow_score + right_eyebrow_score) / 2
            
            # 2. Eyebrow height (20% weight)
            # Angry eyebrows are typically lower
            eyebrow_height_score = min(max((left_eyebrow_height + right_eyebrow_height) / 50, 0), 1)
            
            # 3. Mouth compression (20% weight)
            compression_ratio = 1.0 - (lip_distance / mouth_width)
            
            # 4. Mouth width (20% weight)
            # Angry expressions often have a wider mouth
            width_ratio = mouth_width / image_width
            
            # Combine all factors
            anger_ratio = (
                eyebrow_score * 0.4 +
                eyebrow_height_score * 0.2 +
                compression_ratio * 0.2 +
                width_ratio * 0.2
            )
        else:
            anger_ratio = 0
            
        return True, anger_ratio

class Surprised(Emotion):
    def __init__(self, threshold: float = 0.6, reset_threshold: float = 0.4):
        super().__init__(threshold, reset_threshold, "Surprised", "😮", (255, 255, 0))  # Cyan
        
    def detect(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        def get_landmark(index):
            point = landmarks.landmark[index]
            return int(point.x * image_width), int(point.y * image_height)
            
        upper_lip_top = get_landmark(FaceLandmarks.UPPER_LIP_TOP)
        lower_lip_bottom = get_landmark(FaceLandmarks.LOWER_LIP_BOTTOM)
        left_corner = get_landmark(FaceLandmarks.LEFT_MOUTH_CORNER)
        right_corner = get_landmark(FaceLandmarks.RIGHT_MOUTH_CORNER)
        
        # Enhanced surprise detection
        mouth_height = FaceLandmarks.distance(upper_lip_top, lower_lip_bottom)
        mouth_width = FaceLandmarks.distance(left_corner, right_corner)
        
        if mouth_width > 0:
            # Consider both circularity and size
            circularity = mouth_height / mouth_width
            size_ratio = mouth_width / image_width
            surprise_ratio = circularity * 0.7 + size_ratio * 0.3
        else:
            surprise_ratio = 0
            
        return True, surprise_ratio

class Neutral(Emotion):
    def __init__(self, threshold: float = 0.7, reset_threshold: float = 0.5):
        super().__init__(threshold, reset_threshold, "Neutral", "😐", (128, 128, 128))  # Gray
        
    def detect(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        def get_landmark(index):
            point = landmarks.landmark[index]
            return int(point.x * image_width), int(point.y * image_height)
        
        # Enhanced neutral detection
        upper_lip = get_landmark(FaceLandmarks.UPPER_LIP_TOP)
        lower_lip = get_landmark(FaceLandmarks.LOWER_LIP_BOTTOM)
        left_corner = get_landmark(FaceLandmarks.LEFT_MOUTH_CORNER)
        right_corner = get_landmark(FaceLandmarks.RIGHT_MOUTH_CORNER)
        
        lip_distance = FaceLandmarks.distance(upper_lip, lower_lip)
        mouth_width = FaceLandmarks.distance(left_corner, right_corner)
        
        if mouth_width > 0:
            aspect_ratio = lip_distance / mouth_width
            # More precise neutral range
            neutral_ratio = 1.0 - min(abs(0.35 - aspect_ratio) / 0.15, 1.0)
        else:
            neutral_ratio = 0
            
        return True, neutral_ratio

class Confused(Emotion):
    def __init__(self, threshold: float = 0.3, reset_threshold: float = 0.25):
        super().__init__(threshold, reset_threshold, "Confused", "🤨", (255, 165, 0))  # Orange
        
    def detect(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        def get_landmark(index):
            point = landmarks.landmark[index]
            return int(point.x * image_width), int(point.y * image_height)
        
        # Enhanced confusion detection
        left_corner = get_landmark(FaceLandmarks.LEFT_MOUTH_CORNER)
        right_corner = get_landmark(FaceLandmarks.RIGHT_MOUTH_CORNER)
        upper_lip = get_landmark(FaceLandmarks.UPPER_LIP_TOP)
        
        left_height = FaceLandmarks.distance(left_corner, upper_lip)
        right_height = FaceLandmarks.distance(right_corner, upper_lip)
        
        if max(left_height, right_height) > 0:
            # Consider both asymmetry and mouth position
            asymmetry = abs(left_height - right_height) / max(left_height, right_height)
            mouth_width = FaceLandmarks.distance(left_corner, right_corner)
            width_ratio = mouth_width / image_width
            confusion_ratio = asymmetry * 0.7 + width_ratio * 0.3
        else:
            confusion_ratio = 0
            
        return True, confusion_ratio

class Kiss(Emotion):
    def __init__(self, threshold: float = 0.45, reset_threshold: float = 0.35):
        super().__init__(threshold, reset_threshold, "Kiss", "😘", (255, 192, 203))  # Pink
        
    def detect(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        def get_landmark(index):
            point = landmarks.landmark[index]
            return int(point.x * image_width), int(point.y * image_height)
        
        # Enhanced kiss detection
        left_corner = get_landmark(FaceLandmarks.LEFT_MOUTH_CORNER)
        right_corner = get_landmark(FaceLandmarks.RIGHT_MOUTH_CORNER)
        upper_lip = get_landmark(FaceLandmarks.UPPER_LIP_TOP)
        lower_lip = get_landmark(FaceLandmarks.LOWER_LIP_BOTTOM)
        
        mouth_width = FaceLandmarks.distance(left_corner, right_corner)
        mouth_height = FaceLandmarks.distance(upper_lip, lower_lip)
        
        if mouth_height > 0:
            # Consider both puckering and mouth size
            pucker_ratio = (mouth_height / mouth_width) if mouth_width > 0 else 0
            size_ratio = mouth_width / image_width
            kiss_ratio = pucker_ratio * 0.6 + size_ratio * 0.4
        else:
            kiss_ratio = 0
            
        return True, kiss_ratio

class Disgusted(Emotion):
    def __init__(self, threshold: float = 0.35, reset_threshold: float = 0.25):
        super().__init__(threshold, reset_threshold, "Disgusted", "🤢", (0, 255, 255))  # Yellow
        
    def detect(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        fl = FaceLandmarks
        get_landmark = lambda idx: fl.get_landmark(landmarks, idx, image_width, image_height)
        
        # Enhanced disgust detection
        left_cheek_inner = get_landmark(fl.LEFT_CHEEK_INNER)
        right_cheek_inner = get_landmark(fl.RIGHT_CHEEK_INNER)
        left_cheek_outer = get_landmark(fl.LEFT_CHEEK_OUTER)
        right_cheek_outer = get_landmark(fl.RIGHT_CHEEK_OUTER)
        nose_tip = get_landmark(fl.NOSE_TIP)
        upper_lip = get_landmark(fl.UPPER_LIP_TOP)
        
        left_cheek_width = fl.distance(left_cheek_inner, left_cheek_outer)
        right_cheek_width = fl.distance(right_cheek_inner, right_cheek_outer)
        nose_lip_distance = fl.distance(nose_tip, upper_lip)
        face_width = fl.distance(left_cheek_outer, right_cheek_outer)
        
        if face_width > 0:
            # Enhanced disgust ratio calculation
            cheek_ratio = (left_cheek_width + right_cheek_width) / (2 * face_width)
            nose_lip_ratio = 1.0 - (nose_lip_distance / face_width)
            mouth_width = fl.distance(get_landmark(fl.LEFT_MOUTH_CORNER), get_landmark(fl.RIGHT_MOUTH_CORNER))
            mouth_ratio = mouth_width / face_width
            disgust_ratio = (cheek_ratio * 0.4 + nose_lip_ratio * 0.4 + mouth_ratio * 0.2)
        else:
            disgust_ratio = 0
            
        return True, disgust_ratio

class Sleepy(Emotion):
    def __init__(self, threshold: float = 0.65, reset_threshold: float = 0.45):
        super().__init__(threshold, reset_threshold, "Sleepy", "😴", (128, 0, 128))  # Purple
        
    def detect(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        fl = FaceLandmarks
        
        # Enhanced sleepiness detection
        left_eye_points = [
            fl.LEFT_EYE_LEFT, fl.LEFT_EYE_RIGHT,
            fl.LEFT_EYE_TOP, fl.LEFT_EYE_BOTTOM,
            fl.LEFT_EYE_TOP, fl.LEFT_EYE_BOTTOM
        ]
        right_eye_points = [
            fl.RIGHT_EYE_LEFT, fl.RIGHT_EYE_RIGHT,
            fl.RIGHT_EYE_TOP, fl.RIGHT_EYE_BOTTOM,
            fl.RIGHT_EYE_TOP, fl.RIGHT_EYE_BOTTOM
        ]
        
        left_eye_ratio = fl.get_eye_aspect_ratio(landmarks, left_eye_points, image_width, image_height)
        right_eye_ratio = fl.get_eye_aspect_ratio(landmarks, right_eye_points, image_width, image_height)
        
        # Consider both eye ratios and their difference
        avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        eye_difference = abs(left_eye_ratio - right_eye_ratio)
        
        # Enhanced sleepiness ratio
        sleepy_ratio = (1.0 - avg_eye_ratio) * 0.7 + eye_difference * 0.3
        
        return True, sleepy_ratio

class Einstein(Emotion):
    def __init__(self, threshold: float = 0.35, reset_threshold: float = 0.25):
        super().__init__(threshold, reset_threshold, "Einstein", "😛", (255, 165, 0))  # Orange
        
    def detect(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        fl = FaceLandmarks
        get_landmark = lambda idx: fl.get_landmark(landmarks, idx, image_width, image_height)
        
        # Get all mouth landmarks for comprehensive detection
        upper_lip = get_landmark(fl.UPPER_LIP_TOP)
        lower_lip = get_landmark(fl.LOWER_LIP_BOTTOM)
        left_corner = get_landmark(fl.LEFT_MOUTH_CORNER)
        right_corner = get_landmark(fl.RIGHT_MOUTH_CORNER)
        upper_lip_mid = get_landmark(fl.UPPER_LIP_MID)
        lower_lip_mid = get_landmark(fl.LOWER_LIP_MID)
        left_mouth_inner = get_landmark(fl.LEFT_MOUTH_INNER)
        right_mouth_inner = get_landmark(fl.RIGHT_MOUTH_INNER)
        
        # Calculate mouth metrics
        mouth_width = fl.distance(left_corner, right_corner)
        mouth_height = fl.distance(upper_lip, lower_lip)
        
        # Calculate mouth center point
        center_x = (left_corner[0] + right_corner[0]) / 2
        center_y = (upper_lip[1] + lower_lip[1]) / 2
        
        # Calculate mouth curve using mid points
        mid_curve = lower_lip_mid[1] - upper_lip_mid[1]
        
        # Calculate mouth openness ratio
        openness_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # Calculate mouth width relative to face width
        face_width = fl.distance(get_landmark(fl.LEFT_CHEEK), get_landmark(fl.RIGHT_CHEEK))
        width_ratio = mouth_width / face_width if face_width > 0 else 0
        
        # Calculate inner mouth width
        inner_mouth_width = fl.distance(left_mouth_inner, right_mouth_inner)
        inner_width_ratio = inner_mouth_width / mouth_width if mouth_width > 0 else 0
        
        # Calculate tongue-out ratio
        if mouth_width > 0:
            # Consider multiple factors:
            # 1. High mouth openness (40% weight)
            # 2. Appropriate mouth width (20% weight)
            # 3. Mouth shape (20% weight)
            # 4. Inner mouth width (20% weight)
            openness_score = min(max(openness_ratio - 0.25, 0), 1)  # Penalize too closed
            width_score = min(max(width_ratio - 0.15, 0), 1)  # Penalize too narrow
            shape_score = min(max(1 - abs(openness_ratio - 0.6), 0), 1)  # Prefer oval shape
            inner_width_score = min(max(inner_width_ratio - 0.3, 0), 1)  # Prefer wider inner mouth
            
            # Calculate mouth curve (negative means corners are above center)
            curve = center_y - (left_corner[1] + right_corner[1]) / 2
            
            # Calculate mid-point curve (for tongue detection)
            mid_curve_score = min(max(mid_curve / 20, 0), 1)  # Positive mid_curve indicates tongue out
            
            # Adjust score based on mouth curve and mid-point curve
            curve_score = min(max(-curve / 20, 0), 1)  # Prefer downward curve
            
            # Combine all factors with emphasis on mid-point curve
            einstein_ratio = (
                openness_score * 0.4 +
                width_score * 0.2 +
                shape_score * 0.2 +
                inner_width_score * 0.2
            ) * (0.7 + curve_score * 0.3) * (0.6 + mid_curve_score * 0.4)  # Apply curve and mid-point adjustments
        else:
            einstein_ratio = 0
            
        return True, einstein_ratio

class Childish(Emotion):
    def __init__(self, threshold: float = 0.35, reset_threshold: float = 0.25):
        super().__init__(threshold, reset_threshold, "Childish", "😜", (255, 105, 180))  # Hot Pink
        
    def detect(self, landmarks, image_width: int, image_height: int) -> Tuple[bool, float]:
        fl = FaceLandmarks
        get_landmark = lambda idx: fl.get_landmark(landmarks, idx, image_width, image_height)
        
        # Get mouth landmarks
        upper_lip = get_landmark(fl.UPPER_LIP_TOP)
        lower_lip = get_landmark(fl.LOWER_LIP_BOTTOM)
        left_corner = get_landmark(fl.LEFT_MOUTH_CORNER)
        right_corner = get_landmark(fl.RIGHT_MOUTH_CORNER)
        
        # Calculate mouth metrics
        mouth_width = fl.distance(left_corner, right_corner)
        mouth_height = fl.distance(upper_lip, lower_lip)
        
        # Calculate mouth openness ratio
        openness_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # Calculate mouth width relative to face width
        face_width = fl.distance(get_landmark(fl.LEFT_CHEEK), get_landmark(fl.RIGHT_CHEEK))
        width_ratio = mouth_width / face_width if face_width > 0 else 0
        
        # Calculate tongue-out ratio for just the tip
        if mouth_width > 0:
            # Consider multiple factors:
            # 1. Moderate mouth openness (40% weight)
            # 2. Appropriate mouth width (30% weight)
            # 3. Mouth shape (30% weight)
            openness_score = min(max(1 - abs(openness_ratio - 0.2), 0), 1)  # Prefer smaller opening
            width_score = min(max(width_ratio - 0.15, 0), 1)  # Penalize too narrow
            shape_score = min(max(1 - abs(openness_ratio - 0.15), 0), 1)  # Prefer oval shape
            
            childish_ratio = (
                openness_score * 0.4 +
                width_score * 0.3 +
                shape_score * 0.3
            )
        else:
            childish_ratio = 0
            
        return True, childish_ratio

class EmotionTrainer:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def initialize_camera(self) -> bool:
        """Initialize the camera"""
        print(f"Attempting to open camera {self.camera_index}...")
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False

    def capture_emotion(self, emotion_name: str, duration: int = 5):
        """Capture emotion samples for the specified duration"""
        if not self.initialize_camera():
            return False

        print(f"\nCapturing {emotion_name} emotion samples...")
        print(f"Please make the {emotion_name} expression for {duration} seconds")
        print("Press 'q' to quit early")
        
        start_time = time.time()
        samples = []
        
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Convert to RGB for face mesh
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Get face landmarks
                face_landmarks = results.multi_face_landmarks[0]
                
                # Extract landmark coordinates
                landmarks_data = []
                for landmark in face_landmarks.landmark:
                    landmarks_data.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                samples.append(landmarks_data)
                
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

            # Display countdown
            remaining = int(duration - (time.time() - start_time))
            cv2.putText(frame, f"Time remaining: {remaining}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow(f'Training: {emotion_name}', frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save samples
        if samples:
            emotion_dir = os.path.join(TRAINING_DATA_DIR, emotion_name)
            if not os.path.exists(emotion_dir):
                os.makedirs(emotion_dir)
            
            # Save samples with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(emotion_dir, f"{timestamp}.json")
            
            with open(filename, 'w') as f:
                json.dump({
                    'emotion': emotion_name,
                    'timestamp': timestamp,
                    'samples': samples
                }, f)
            
            print(f"\nSaved {len(samples)} samples to {filename}")
        else:
            print("\nNo samples captured. Please try again.")

        # Cleanup
        self.cleanup()
        return True

    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

class EmotionDetector:
    def __init__(self, camera_index: int = 0, debug: bool = False):
        print("Setting up EmotionDetector...")
        self.camera_index = camera_index
        self.debug = debug
        self.cap = None
        
        # Initialize MediaPipe
        print("Initializing MediaPipe...")
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            print("MediaPipe initialized successfully")
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            raise
        
        # Initialize emotions first with default thresholds
        print("Initializing emotion detectors...")
        try:
            self.emotions = [
                Happy(),      # Smile detection
                Sad(),       # Downturned mouth
                Angry(),     # Compressed lips
                Surprised(), # Circular mouth
                Sleepy(),    # Eye aspect ratio
                Einstein(),  # Full tongue-out expression
                Childish()   # Just the tip of tongue
            ]
            print("Emotion detectors initialized successfully")
        except Exception as e:
            print(f"Error initializing emotion detectors: {e}")
            raise
        
        # Load training data and calibrate thresholds
        print("Loading training data...")
        self.training_data = self.load_training_data()
        
        # Update emotion thresholds with calibrated values
        print("Calibrating emotion thresholds...")
        for emotion in self.emotions:
            emotion_name = emotion.name.lower()
            calibrated_threshold = self.get_calibrated_threshold(emotion_name)
            emotion.threshold = calibrated_threshold
            emotion.reset_threshold = calibrated_threshold * 0.8  # Reset threshold at 80% of calibrated threshold
            print(f"Calibrated {emotion_name} threshold: {calibrated_threshold:.3f}")

    def load_training_data(self) -> Dict:
        """Load all training data from JSON files"""
        training_data = {}
        for emotion in ["happy", "sad", "angry", "surprised", "sleepy", "einstein", "childish"]:
            emotion_dir = os.path.join(TRAINING_DATA_DIR, emotion)
            if os.path.exists(emotion_dir):
                samples = []
                for filename in os.listdir(emotion_dir):
                    if filename.endswith('.json'):
                        with open(os.path.join(emotion_dir, filename), 'r') as f:
                            data = json.load(f)
                            samples.extend(data['samples'])
                if samples:
                    training_data[emotion] = samples
                    print(f"Loaded {len(samples)} samples for {emotion}")
        return training_data

    def get_calibrated_threshold(self, emotion_name: str) -> float:
        """Calculate calibrated threshold based on training data"""
        if emotion_name not in self.training_data:
            print(f"No training data for {emotion_name}, using default threshold")
            return 0.35  # Default threshold
            
        samples = self.training_data[emotion_name]
        if not samples:
            return 0.35
            
        # Calculate average ratios for this emotion
        ratios = []
        for sample in samples:
            # Convert sample to landmarks format
            landmarks = type('Landmarks', (), {
                'landmark': [
                    type('Landmark', (), {
                        'x': p['x'],
                        'y': p['y'],
                        'z': p['z'],
                        'visibility': p['visibility']
                    }) for p in sample
                ]
            })
            
            # Get the emotion class
            emotion_class = next(e for e in self.emotions if e.name.lower() == emotion_name)
            _, ratio = emotion_class.detect(landmarks, 640, 480)  # Using standard resolution
            ratios.append(ratio)
        
        if not ratios:
            return 0.35
            
        # Calculate threshold based on statistics
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        # Set threshold to mean minus one standard deviation
        threshold = max(0.2, mean_ratio - std_ratio)
        print(f"Calibrated threshold for {emotion_name}: {threshold:.3f}")
        return threshold

    def initialize_camera(self) -> bool:
        """Initialize the camera with error handling"""
        print(f"Attempting to open camera {self.camera_index}...")
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Try to read a frame to verify camera is working
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from camera")
                return False
            
            print(f"Successfully initialized camera {self.camera_index}")
            print(f"Frame size: {frame.shape}")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False

    def process_emotions(self, frame, face_landmarks, width: int, height: int):
        """Process and display emotions with training data influence"""
        max_emotion = None
        max_ratio = 0.0
        
        # Process all emotions and find the dominant one
        for emotion in self.emotions:
            is_active, ratio = emotion.process_frame(face_landmarks, width, height)
            
            # Adjust ratio based on training data if available
            emotion_name = emotion.name.lower()
            if emotion_name in self.training_data:
                # Calculate similarity with training samples
                current_landmarks = [
                    {'x': p.x, 'y': p.y, 'z': p.z, 'visibility': p.visibility}
                    for p in face_landmarks.landmark
                ]
                
                similarities = []
                for sample in self.training_data[emotion_name]:
                    # Calculate similarity between current landmarks and sample
                    similarity = self.calculate_landmark_similarity(current_landmarks, sample)
                    similarities.append(similarity)
                
                if similarities:
                    # Adjust ratio based on similarity with training samples
                    avg_similarity = np.mean(similarities)
                    ratio = ratio * (0.7 + 0.3 * avg_similarity)
            
            if ratio > max_ratio:
                max_ratio = ratio
                max_emotion = emotion
        
        # Only show emotion if strong enough
        if max_emotion and max_ratio > 0.3:
            # Create a solid background with emotion color
            overlay = frame.copy()
            # Make the background color more visible but not too strong
            bg_color = tuple(int(c * 0.3) for c in max_emotion.color)  # 30% opacity
            cv2.rectangle(overlay, (0, 0), (width, height), bg_color, -1)
            
            # Apply transparency
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Draw thick border with emotion color
            border_thickness = int(max_ratio * 10)  # Thicker border for stronger emotions
            cv2.rectangle(frame, (0, 0), (width-1, height-1), max_emotion.color, border_thickness)
            
            # Draw the emoji in the center
            emoji_size = 150  # Larger emoji
            x = (width - emoji_size) // 2
            y = (height - emoji_size) // 2
            
            # Create a black background for the emoji
            emoji_bg = frame.copy()
            cv2.rectangle(emoji_bg, (x, y), (x + emoji_size, y + emoji_size), (0, 0, 0), -1)
            
            # Apply transparency for emoji background
            frame = cv2.addWeighted(emoji_bg, 0.7, frame, 0.3, 0)
            
            # Draw the emoji
            cv2.putText(frame, max_emotion.emoji, (x + 30, y + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show confidence in debug mode
            if self.debug:
                cv2.putText(frame, f"Confidence: {max_ratio:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def calculate_landmark_similarity(self, landmarks1: List[Dict], landmarks2: List[Dict]) -> float:
        """Calculate similarity between two sets of landmarks using cosine similarity"""
        if len(landmarks1) != len(landmarks2):
            return 0.0
            
        # Extract x, y, z coordinates into separate vectors
        x1 = np.array([p['x'] for p in landmarks1])
        y1 = np.array([p['y'] for p in landmarks1])
        z1 = np.array([p['z'] for p in landmarks1])
        
        x2 = np.array([p['x'] for p in landmarks2])
        y2 = np.array([p['y'] for p in landmarks2])
        z2 = np.array([p['z'] for p in landmarks2])
        
        # Calculate cosine similarity for each dimension
        def cosine_similarity(v1, v2):
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return dot_product / (norm1 * norm2)
        
        # Calculate similarity for each dimension
        x_similarity = cosine_similarity(x1, x2)
        y_similarity = cosine_similarity(y1, y2)
        z_similarity = cosine_similarity(z1, z2)
        
        # Weight the dimensions (y is most important for facial expressions)
        weights = [0.3, 0.5, 0.2]  # x, y, z weights
        similarities = [x_similarity, y_similarity, z_similarity]
        
        # Calculate weighted average
        weighted_similarity = sum(s * w for s, w in zip(similarities, weights))
        
        # Normalize to 0-1 range
        normalized_similarity = (weighted_similarity + 1) / 2
        
        return normalized_similarity

    def run(self):
        """Main detection loop with improved error handling"""
        if not self.cap:
            print("Camera not initialized")
            return

        print("Starting face mesh detection...")
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            print("Face mesh detection initialized")
            
            while self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read frame")
                        break

                    # Convert the BGR image to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width = frame.shape[:2]

                    # Process the frame and detect face landmarks
                    results = face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            # Process emotions
                            self.process_emotions(frame, face_landmarks, width, height)
                            
                            # Draw face mesh in debug mode
                            if self.debug:
                                self.mp_drawing.draw_landmarks(
                                    image=frame,
                                    landmark_list=face_landmarks,
                                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                                )

                    # Display the frame
                    cv2.imshow('Emotion Detection', frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quit command received")
                        break

                except Exception as e:
                    print(f"Error in main loop: {e}")
                    break

        print("Detection loop ended")

    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        emotion_counts = [f"{e.name}: {e.count}" for e in self.emotions]
        print(f"Session ended. Emotion counts: {', '.join(emotion_counts)}")

def main():
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Emotion Detection with MediaPipe')
        parser.add_argument('--camera', type=int, default=0, help='Camera index to use')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        parser.add_argument('--train', action='store_true', help='Enable training mode')
        parser.add_argument('--smile', action='store_true', help='Train smile emotion')
        parser.add_argument('--sad', action='store_true', help='Train sad emotion')
        parser.add_argument('--angry', action='store_true', help='Train angry emotion')
        parser.add_argument('--surprised', action='store_true', help='Train surprised emotion')
        parser.add_argument('--sleepy', action='store_true', help='Train sleepy emotion')
        parser.add_argument('--einstein', action='store_true', help='Train einstein (tongue-out) emotion')
        parser.add_argument('--childish', action='store_true', help='Train childish (tongue-tip) emotion')
        args = parser.parse_args()

        if args.train:
            # Training mode
            trainer = EmotionTrainer(camera_index=args.camera)
            
            if args.smile:
                trainer.capture_emotion("happy")
            elif args.sad:
                trainer.capture_emotion("sad")
            elif args.angry:
                trainer.capture_emotion("angry")
            elif args.surprised:
                trainer.capture_emotion("surprised")
            elif args.sleepy:
                trainer.capture_emotion("sleepy")
            elif args.einstein:
                trainer.capture_emotion("einstein")
            elif args.childish:
                trainer.capture_emotion("childish")
            else:
                print("Please specify an emotion to train using --smile, --sad, --angry, etc.")
        else:
            # Normal emotion detection mode
            print("Starting Emotion Detection Application...")
            print(f"Camera index: {args.camera}")
            print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")

            # Initialize the emotion detector
            print("\nInitializing emotion detector...")
            detector = EmotionDetector(camera_index=args.camera, debug=args.debug)
            
            # Initialize the camera
            print("\nInitializing camera...")
            if not detector.initialize_camera():
                print("Error: Failed to initialize camera")
                print("Please check if:")
                print("1. The camera is properly connected")
                print("2. The camera is not being used by another application")
                print("3. You have the correct camera index")
                return

            print("\nStarting emotion detection...")
            print("Press 'q' to quit")
            print("Press 'd' to toggle debug mode")
            
            # Run the detection loop
            detector.run()

    except KeyboardInterrupt:
        print("\nStopping emotion detection...")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        if 'detector' in locals():
            detector.cleanup()
        cv2.destroyAllWindows()
        print("Application terminated")

if __name__ == "__main__":
    # Set environment variable to disable TensorFlow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
    main() 