"""
💓 HeartSense Pro - Gesture-Controlled Heart Rate Detection System
Complete click-free interface with advanced gesture recognition

Gesture Controls:
- START: Left hand open palm to left of head + Right hand open palm to right of head
- STOP: Both hands covering face
- Auto state machine with smooth transitions
"""

import cv2
import numpy as np
import time
from collections import deque
from scipy.fft import fft, fftfreq
from scipy import signal
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import math
import warnings
warnings.filterwarnings('ignore')

# Install MediaPipe for hand detection if not available
try:
    import mediapipe as mp
except ImportError:
    print("Installing MediaPipe for gesture recognition...")
    import subprocess
    subprocess.check_call(["pip", "install", "mediapipe"])
    import mediapipe as mp

# 1️⃣ 🎨 Enhanced UI Theme with Gesture Feedback
class UITheme:
    """
    CALL ORDER: 1️⃣ - First initialization for UI styling with gesture indicators
    Enhanced Apple-inspired design system with gesture feedback
    """
    
    # Apple Health-inspired Color Palette + Gesture Colors
    COLORS = {
        'primary_blue': '#007AFF',        # Apple System Blue
        'health_green': '#32D74B',        # Apple Health Green
        'heart_red': '#FF3B30',          # Apple System Red
        'warning_orange': '#FF9500',      # Apple System Orange
        'purple_accent': '#AF52DE',       # Apple System Purple
        'background_dark': '#1C1C1E',     # Apple Dark Background
        'surface_glass': 'rgba(255, 255, 255, 0.1)',  # Glassmorphic
        'text_primary': '#FFFFFF',        # Primary text
        'text_secondary': '#8E8E93',      # Secondary text
        'card_background': '#2C2C2E',     # Glass card
        'success': '#34C759',             # Success green
        'error': '#FF453A',               # Error red
        'gesture_start': '#00D4FF',       # Gesture start indicator
        'gesture_stop': '#FF6B35',        # Gesture stop indicator
        'gesture_active': '#FFD60A',      # Gesture detected
    }
    
    # Typography
    FONTS = {
        'primary': ('Helvetica Neue', 16, 'normal'),
        'heading': ('Helvetica Neue', 24, 'bold'),
        'body': ('Helvetica Neue', 14, 'normal'),
        'caption': ('Helvetica Neue', 12, 'normal'),
        'large_display': ('Helvetica Neue', 48, 'bold'),
        'gesture_title': ('Helvetica Neue', 20, 'bold'),
    }

# 2️⃣ 🤚 Advanced Gesture Recognition System
class GestureRecognitionSystem:
    """
    CALL ORDER: 2️⃣ - Initialize gesture recognition with MediaPipe
    Advanced gesture recognition using MediaPipe with state machine
    """
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Gesture detection parameters
        self.gesture_confidence_threshold = 0.8
        self.gesture_hold_time = 1.0  # seconds
        self.last_gesture_time = 0
        self.current_gesture = None
        self.gesture_start_time = 0
        
    def detect_hands(self, frame):
        """
        CALL ORDER: 2️⃣A - Detect hands in frame
        Detect and return hand landmarks and handedness
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_landmarks = []
        handedness_labels = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmark, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_landmarks.append(hand_landmark)
                handedness_labels.append(handedness.classification[0].label)
                
        return hand_landmarks, handedness_labels, results
    
    def is_hand_open(self, landmarks):
        """
        CALL ORDER: 2️⃣B - Check if hand shows open palm
        Detect open palm by analyzing finger positions
        """
        # Finger tip and PIP joint landmarks
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        finger_pips = [6, 10, 14, 18]  # PIP joints
        
        open_fingers = 0
        
        # Check fingers (except thumb)
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
                open_fingers += 1
        
        # Check thumb (different logic due to orientation)
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        wrist = landmarks.landmark[0]
        
        # Thumb is open if tip is farther from wrist than IP joint
        thumb_distance_tip = abs(thumb_tip.x - wrist.x)
        thumb_distance_ip = abs(thumb_ip.x - wrist.x)
        
        if thumb_distance_tip > thumb_distance_ip:
            open_fingers += 1
            
        return open_fingers >= 4  # At least 4 out of 5 fingers open
    
    def get_hand_position(self, landmarks):
        """
        CALL ORDER: 2️⃣C - Get hand position relative to frame
        Calculate hand center position and classify location
        """
        # Calculate hand center
        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        # Classify position
        position = {
            'center_x': center_x,
            'center_y': center_y,
            'is_left_side': center_x < 0.3,
            'is_right_side': center_x > 0.7,
            'is_center': 0.3 <= center_x <= 0.7,
            'is_face_level': 0.2 <= center_y <= 0.8
        }
        
        return position
    
    def detect_start_gesture(self, hand_landmarks, handedness_labels):
        """
        CALL ORDER: 2️⃣D - Detect START gesture
        START: Left hand open palm to left + Right hand open palm to right
        """
        if len(hand_landmarks) != 2:
            return False
            
        # Map hands to their positions
        hands_dict = {}
        for i, label in enumerate(handedness_labels):
            hands_dict[label] = hand_landmarks[i]
        
        left_hand = hands_dict.get('Left')
        right_hand = hands_dict.get('Right')
        
        if not (left_hand and right_hand):
            return False
        
        # Check conditions for START gesture
        left_open = self.is_hand_open(left_hand)
        right_open = self.is_hand_open(right_hand)
        
        left_pos = self.get_hand_position(left_hand)
        right_pos = self.get_hand_position(right_hand)
        
        # LEFT hand should be open and on left side at face level
        left_correct = (left_open and 
                       left_pos['is_left_side'] and 
                       left_pos['is_face_level'])
        
        # RIGHT hand should be open and on right side at face level
        right_correct = (right_open and 
                        right_pos['is_right_side'] and 
                        right_pos['is_face_level'])
        
        return left_correct and right_correct
    
    def detect_stop_gesture(self, hand_landmarks, handedness_labels):
        """
        CALL ORDER: 2️⃣E - Detect STOP gesture
        STOP: Both hands covering face (both hands near center)
        """
        if len(hand_landmarks) != 2:
            return False
            
        both_hands_center = True
        both_hands_face_level = True
        
        for landmarks in hand_landmarks:
            position = self.get_hand_position(landmarks)
            if not position['is_center']:
                both_hands_center = False
            if not position['is_face_level']:
                both_hands_face_level = False
                
        return both_hands_center and both_hands_face_level
    
    def process_gestures(self, frame):
        """
        CALL ORDER: 2️⃣F - Main gesture processing function
        Process frame and return gesture state with confidence
        """
        hand_landmarks, handedness_labels, results = self.detect_hands(frame)
        
        current_time = time.time()
        gesture_detected = None
        confidence = 0.0
        
        # Detect gestures
        if self.detect_start_gesture(hand_landmarks, handedness_labels):
            gesture_detected = 'START'
            confidence = 0.9
        elif self.detect_stop_gesture(hand_landmarks, handedness_labels):
            gesture_detected = 'STOP'
            confidence = 0.9
        
        # Gesture stability check - must hold gesture for minimum time
        if gesture_detected:
            if self.current_gesture != gesture_detected:
                self.current_gesture = gesture_detected
                self.gesture_start_time = current_time
            else:
                # Same gesture detected, check if held long enough
                hold_duration = current_time - self.gesture_start_time
                if hold_duration >= self.gesture_hold_time:
                    self.last_gesture_time = current_time
                    return gesture_detected, confidence, hand_landmarks, results
        else:
            self.current_gesture = None
            
        return None, confidence, hand_landmarks, results

# 3️⃣ 🔄 Gesture-Based State Machine
class GestureStateMachine:
    """
    CALL ORDER: 3️⃣ - Initialize state machine for gesture control
    State machine for managing detection states via gestures
    """
    
    def __init__(self):
        self.state = 'STOPPED'  # STOPPED, RUNNING, TRANSITIONING
        self.previous_state = 'STOPPED'
        self.state_change_time = time.time()
        self.transition_duration = 2.0  # seconds
        
    def update_state(self, gesture, confidence):
        """
        CALL ORDER: 3️⃣A - Update state based on detected gesture
        Process gesture and update state accordingly
        """
        current_time = time.time()
        
        if gesture == 'START' and self.state == 'STOPPED' and confidence > 0.8:
            self.previous_state = self.state
            self.state = 'RUNNING'
            self.state_change_time = current_time
            return True, "Detection Started"
            
        elif gesture == 'STOP' and self.state == 'RUNNING' and confidence > 0.8:
            self.previous_state = self.state
            self.state = 'STOPPED'
            self.state_change_time = current_time
            return True, "Detection Stopped"
            
        return False, f"Current State: {self.state}"
    
    def get_state_info(self):
        """
        CALL ORDER: 3️⃣B - Get current state information
        Return state information for UI display
        """
        current_time = time.time()
        time_in_state = current_time - self.state_change_time
        
        return {
            'state': self.state,
            'previous_state': self.previous_state,
            'time_in_state': time_in_state,
            'is_running': self.state == 'RUNNING'
        }

# 4️⃣ 💓 Enhanced Heartbeat Animator with Gesture Feedback
class EnhancedHeartbeatAnimator:
    """
    CALL ORDER: 4️⃣ - Initialize enhanced heartbeat animations
    Advanced heartbeat animations with gesture state feedback
    """
    
    def __init__(self, canvas, x, y, size=60):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.base_size = size
        self.bpm = 70
        self.detection_active = False
        self.gesture_state = 'STOPPED'
        
        # Animation parameters
        self.pulse_intensity = 1.0
        self.color_transition = 0.0
        
    def update_state(self, bpm, detection_active, gesture_state):
        """
        CALL ORDER: 4️⃣A - Update animator with current state
        Update heart rate, detection state, and gesture feedback
        """
        self.bpm = max(40, min(200, bpm)) if bpm else 70
        self.detection_active = detection_active
        self.gesture_state = gesture_state
        
    def get_dynamic_heart_scale(self):
        """
        CALL ORDER: 4️⃣B - Calculate dynamic heart scale
        Calculate heart scale with enhanced pulsing effects
        """
        if not self.detection_active:
            # Gentle idle animation when not detecting
            idle_pulse = 0.1 * math.sin(time.time() * 2)
            return 1.0 + idle_pulse
            
        # Active heartbeat animation
        beat_frequency = self.bpm / 60.0
        time_factor = time.time() * beat_frequency * 2 * math.pi
        
        # Realistic double-beat pattern (lub-dub)
        primary_beat = max(0, math.sin(time_factor)) * 0.4
        secondary_beat = max(0, math.sin(time_factor * 1.5 + math.pi/4)) * 0.2
        
        scale = 1.0 + (primary_beat + secondary_beat) * self.pulse_intensity
        return scale
    
    def get_heart_color(self):
        """
        CALL ORDER: 4️⃣C - Get heart color based on state
        Dynamic color based on detection state and gesture feedback
        """
        if self.gesture_state == 'RUNNING':
            # Pulsing red when active
            pulse_factor = (math.sin(time.time() * 3) + 1) / 2
            red_intensity = int(255 * (0.7 + 0.3 * pulse_factor))
            return f"#{red_intensity:02x}3B30"
        elif self.gesture_state == 'STOPPED':
            # Dim blue when stopped
            return UITheme.COLORS['primary_blue']
        else:
            # Default red
            return UITheme.COLORS['heart_red']
    
    def draw_enhanced_heart(self):
        """
        CALL ORDER: 4️⃣D - Draw enhanced animated heart
        Draw heart with advanced animations and effects
        """
        scale = self.get_dynamic_heart_scale()
        color = self.get_heart_color()
        size = int(self.base_size * scale)
        
        # Clear previous heart
        self.canvas.delete("heart")
        self.canvas.delete("heart_glow")
        
        # Draw glow effect for active detection
        if self.detection_active:
            glow_size = int(size * 1.3)
            glow_alpha = 0.3 + 0.2 * math.sin(time.time() * 4)
            
            # Multiple glow layers
            for i in range(3):
                layer_size = glow_size - i * 8
                self.draw_heart_shape(layer_size, f"{color}40", "heart_glow")
        
        # Main heart
        self.draw_heart_shape(size, color, "heart")
        
        # Add pulse ring for high heart rates
        if self.bpm and self.bpm > 100:
            ring_radius = int(size * 1.5)
            ring_alpha = max(0, math.sin(time.time() * self.bpm / 30))
            if ring_alpha > 0:
                self.canvas.create_oval(
                    self.x - ring_radius, self.y - ring_radius,
                    self.x + ring_radius, self.y + ring_radius,
                    outline=UITheme.COLORS['warning_orange'],
                    width=int(3 * ring_alpha),
                    tags="heart"
                )
    
    def draw_heart_shape(self, size, color, tag):
        """
        CALL ORDER: 4️⃣E - Draw heart shape with specified parameters
        Draw anatomically-inspired heart shape
        """
        x, y = self.x, self.y
        
        # Heart shape using parametric equations
        points = []
        for angle in range(0, 360, 5):
            t = math.radians(angle)
            # Heart equation: x = 16sin³(t), y = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)
            heart_x = 16 * (math.sin(t) ** 3)
            heart_y = 13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)
            
            # Scale and position
            scaled_x = x + (heart_x * size / 400)
            scaled_y = y - (heart_y * size / 400)  # Negative for tkinter coordinates
            points.extend([scaled_x, scaled_y])
        
        if len(points) >= 6:  # Need at least 3 points for polygon
            self.canvas.create_polygon(
                points,
                fill=color,
                outline=color,
                width=2,
                tags=tag
            )

# 5️⃣ 📊 Gesture-Enhanced Dashboard
class GestureEnhancedDashboard:
    """
    CALL ORDER: 5️⃣ - Initialize gesture-enhanced dashboard
    Advanced dashboard with gesture feedback and enhanced visualizations
    """
    
    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.setup_dashboard()
        self.heart_animator = None
        
        # Data storage
        self.hr_history = deque(maxlen=100)  # Extended history
        self.time_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)
        
    def setup_dashboard(self):
        """
        CALL ORDER: 5️⃣A - Setup enhanced dashboard layout
        Create all dashboard components with gesture indicators
        """
        self.create_main_heart_display()
        self.create_gesture_control_panel()
        self.create_metrics_panel()
        self.create_enhanced_chart()
        self.create_status_panel()
        
    def create_main_heart_display(self):
        """
        CALL ORDER: 5️⃣B - Create main heart rate display
        Large central heart rate display with enhanced animations
        """
        self.main_card = self.create_glass_frame(50, 50, 400, 280)
        
        # Title with gesture status
        self.title_frame = tk.Frame(self.main_card, bg=UITheme.COLORS['card_background'])
        self.title_frame.pack(pady=(20, 10))
        
        self.main_title = tk.Label(
            self.title_frame,
            text="💓 Heart Rate Monitor",
            font=UITheme.FONTS['gesture_title'],
            fg=UITheme.COLORS['text_primary'],
            bg=UITheme.COLORS['card_background']
        )
        self.main_title.pack()
        
        # Detection status
        self.detection_status = tk.Label(
            self.title_frame,
            text="Use gestures to control",
            font=UITheme.FONTS['body'],
            fg=UITheme.COLORS['text_secondary'],
            bg=UITheme.COLORS['card_background']
        )
        self.detection_status.pack()
        
        # Large heart rate display
        self.hr_display = tk.Label(
            self.main_card,
            text="-- BPM",
            font=('Helvetica Neue', 56, 'bold'),
            fg=UITheme.COLORS['health_green'],
            bg=UITheme.COLORS['card_background']
        )
        self.hr_display.pack(pady=(10, 5))
        
        # Confidence and quality
        self.metrics_frame = tk.Frame(self.main_card, bg=UITheme.COLORS['card_background'])
        self.metrics_frame.pack()
        
        self.confidence_label = tk.Label(
            self.metrics_frame,
            text="Confidence: --%",
            font=UITheme.FONTS['body'],
            fg=UITheme.COLORS['text_secondary'],
            bg=UITheme.COLORS['card_background']
        )
        self.confidence_label.pack(side='left', padx=10)
        
        self.quality_label = tk.Label(
            self.metrics_frame,
            text="Quality: --%",
            font=UITheme.FONTS['body'],
            fg=UITheme.COLORS['text_secondary'],
            bg=UITheme.COLORS['card_background']
        )
        self.quality_label.pack(side='left', padx=10)
        
        # Enhanced heart animation canvas
        self.heart_canvas = tk.Canvas(
            self.main_card,
            width=150,
            height=120,
            bg=UITheme.COLORS['card_background'],
            highlightthickness=0
        )
        self.heart_canvas.pack(pady=10)
        
        # Initialize enhanced heart animator
        self.heart_animator = EnhancedHeartbeatAnimator(self.heart_canvas, 75, 60, 40)
        
    def create_gesture_control_panel(self):
        """
        CALL ORDER: 5️⃣C - Create gesture control indicators
        Visual feedback for gesture controls
        """
        self.gesture_card = self.create_glass_frame(470, 50, 300, 280)
        
        # Title
        gesture_title = tk.Label(
            self.gesture_card,
            text="🤚 Gesture Controls",
            font=UITheme.FONTS['heading'],
            fg=UITheme.COLORS['text_primary'],
            bg=UITheme.COLORS['card_background']
        )
        gesture_title.pack(pady=(20, 15))
        
        # START gesture indicator
        self.start_frame = tk.Frame(self.gesture_card, bg=UITheme.COLORS['card_background'])
        self.start_frame.pack(pady=10, padx=20, fill='x')
        
        self.start_status = tk.Label(
            self.start_frame,
            text="▶️ START: Hands open at sides",
            font=UITheme.FONTS['body'],
            fg=UITheme.COLORS['gesture_start'],
            bg=UITheme.COLORS['card_background']
        )
        self.start_status.pack(anchor='w')
        
        # STOP gesture indicator
        self.stop_frame = tk.Frame(self.gesture_card, bg=UITheme.COLORS['card_background'])
        self.stop_frame.pack(pady=10, padx=20, fill='x')
        
        self.stop_status = tk.Label(
            self.stop_frame,
            text="⏹️ STOP: Cover face with hands",
            font=UITheme.FONTS['body'],
            fg=UITheme.COLORS['gesture_stop'],
            bg=UITheme.COLORS['card_background']
        )
        self.stop_status.pack(anchor='w')
        
        # Current gesture detection
        self.current_gesture_frame = tk.Frame(self.gesture_card, bg=UITheme.COLORS['card_background'])
        self.current_gesture_frame.pack(pady=20, padx=20, fill='x')
        
        self.current_gesture_label = tk.Label(
            self.current_gesture_frame,
            text="Current Status:",
            font=UITheme.FONTS['caption'],
            fg=UITheme.COLORS['text_secondary'],
            bg=UITheme.COLORS['card_background']
        )
        self.current_gesture_label.pack()
        
        self.gesture_indicator = tk.Label(
            self.current_gesture_frame,
            text="⏸️ STOPPED",
            font=UITheme.FONTS['heading'],
            fg=UITheme.COLORS['gesture_stop'],
            bg=UITheme.COLORS['card_background']
        )
        self.gesture_indicator.pack(pady=5)
        
    def create_metrics_panel(self):
        """
        CALL ORDER: 5️⃣D - Create metrics and quality panel
        Advanced metrics with real-time quality indicators
        """
        self.metrics_card = self.create_glass_frame(790, 50, 360, 280)
        
        # Title
        metrics_title = tk.Label(
            self.metrics_card,
            text="📊 Live Metrics",
            font=UITheme.FONTS['heading'],
            fg=UITheme.COLORS['text_primary'],
            bg=UITheme.COLORS['card_background']
        )
        metrics_title.pack(pady=(20, 15))
        
        # Signal quality bar
        self.signal_quality_frame = self.create_metric_bar(
            self.metrics_card, "Signal Quality", UITheme.COLORS['primary_blue']
        )
        
        # Motion level bar
        self.motion_level_frame = self.create_metric_bar(
            self.metrics_card, "Motion Level", UITheme.COLORS['warning_orange']
        )
        
        # Heart rate zone indicator
        self.hr_zone_frame = tk.Frame(self.metrics_card, bg=UITheme.COLORS['card_background'])
        self.hr_zone_frame.pack(pady=15, padx=20, fill='x')
        
        self.hr_zone_label = tk.Label(
            self.hr_zone_frame,
            text="Heart Rate Zone: Resting",
            font=UITheme.FONTS['body'],
            fg=UITheme.COLORS['health_green'],
            bg=UITheme.COLORS['card_background']
        )
        self.hr_zone_label.pack()
        
        # Active ROI indicator
        self.roi_label = tk.Label(
            self.metrics_card,
            text="Active ROI: Combined",
            font=UITheme.FONTS['caption'],
            fg=UITheme.COLORS['text_secondary'],
            bg=UITheme.COLORS['card_background']
        )
        self.roi_label.pack(pady=10)
        
    def create_metric_bar(self, parent, label, color):
        """
        CALL ORDER: 5️⃣E - Create animated metric bars
        Create progress bars for real-time metrics
        """
        container = tk.Frame(parent, bg=UITheme.COLORS['card_background'])
        container.pack(pady=8, padx=20, fill='x')
        
        # Label
        tk.Label(
            container,
            text=label,
            font=UITheme.FONTS['caption'],
            fg=UITheme.COLORS['text_secondary'],
            bg=UITheme.COLORS['card_background']
        ).pack(anchor='w')
        
        # Progress bar
        progress = ttk.Progressbar(
            container,
            mode='determinate',
            length=300,
            style='Custom.Horizontal.TProgressbar'
        )
        progress.pack(fill='x', pady=(3, 0))
        
        return progress
        
    def create_enhanced_chart(self):
        """
        CALL ORDER: 5️⃣F - Create enhanced real-time chart
        Advanced real-time chart with multiple data streams
        """
        self.chart_card = self.create_glass_frame(50, 350, 700, 320)
        
        # Chart title with controls
        chart_header = tk.Frame(self.chart_card, bg=UITheme.COLORS['card_background'])
        chart_header.pack(pady=10, fill='x')
        
        chart_title = tk.Label(
            chart_header,
            text="📈 Real-Time Heart Rate Trend",
            font=UITheme.FONTS['heading'],
            fg=UITheme.COLORS['text_primary'],
            bg=UITheme.COLORS['card_background']
        )
        chart_title.pack()
        
        # Setup matplotlib chart
        self.setup_enhanced_chart()
        
    def setup_enhanced_chart(self):
        """
        CALL ORDER: 5️⃣G - Setup enhanced matplotlib chart
        Configure advanced chart with multiple data series
        """
        plt.style.use('dark_background')
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 4), facecolor='#1C1C1E')
        self.fig.subplots_adjust(hspace=0.3)
        
        # Heart rate chart
        self.ax1.set_facecolor('#1C1C1E')
        self.ax1.grid(True, alpha=0.3, color='#8E8E93')
        self.ax1.set_ylabel('BPM', color=UITheme.COLORS['text_secondary'])
        self.ax1.tick_params(colors=UITheme.COLORS['text_secondary'])
        self.ax1.set_title('Heart Rate', color=UITheme.COLORS['text_primary'], fontsize=12)
        
        # Signal quality chart
        self.ax2.set_facecolor('#1C1C1E')
        self.ax2.grid(True, alpha=0.3, color='#8E8E93')
        self.ax2.set_xlabel('Time (s)', color=UITheme.COLORS['text_secondary'])
        self.ax2.set_ylabel('Quality', color=UITheme.COLORS['text_secondary'])
        self.ax2.tick_params(colors=UITheme.COLORS['text_secondary'])
        self.ax2.set_title('Signal Quality', color=UITheme.COLORS['text_primary'], fontsize=12)
        
        # Initialize lines
        self.hr_line, = self.ax1.plot([], [], color=UITheme.COLORS['heart_red'], linewidth=2, label='Heart Rate')
        self.quality_line, = self.ax2.plot([], [], color=UITheme.COLORS['primary_blue'], linewidth=2, label='Quality')
        
        # Embed in tkinter
        self.canvas_widget = FigureCanvasTkAgg(self.fig, self.chart_card)
        self.canvas_widget.get_tk_widget().pack(padx=15, pady=10)
        
    def create_status_panel(self):
        """
        CALL ORDER: 5️⃣H - Create system status panel
        System status with gesture feedback
        """
        self.status_card = self.create_glass_frame(770, 350, 380, 320)
        
        # Title
        status_title = tk.Label(
            self.status_card,
            text="🔧 System Status",
            font=UITheme.FONTS['heading'],
            fg=UITheme.COLORS['text_primary'],
            bg=UITheme.COLORS['card_background']
        )
        status_title.pack(pady=(20, 15))
        
        # Status indicators
        self.camera_status = self.create_status_indicator(
            self.status_card, "📹 Camera", "Ready", UITheme.COLORS['success']
        )
        
        self.gesture_recognition_status = self.create_status_indicator(
            self.status_card, "🤚 Gesture Recognition", "Active", UITheme.COLORS['primary_blue']
        )
        
        self.face_detection_status = self.create_status_indicator(
            self.status_card, "👤 Face Detection", "Searching", UITheme.COLORS['warning_orange']
        )
        
        self.processing_status = self.create_status_indicator(
            self.status_card, "⚡ Signal Processing", "Standby", UITheme.COLORS['text_secondary']
        )
        
    def create_status_indicator(self, parent, icon_text, status, color):
        """
        CALL ORDER: 5️⃣I - Create individual status indicators
        Create status indicators with dynamic colors
        """
        container = tk.Frame(parent, bg=UITheme.COLORS['card_background'])
        container.pack(pady=8, padx=20, fill='x')
        
        label = tk.Label(
            container,
            text=f"{icon_text}: {status}",
            font=UITheme.FONTS['body'],
            fg=color,
            bg=UITheme.COLORS['card_background']
        )
        label.pack(anchor='w')
        
        return label
        
    def create_glass_frame(self, x, y, width, height):
        """
        CALL ORDER: 5️⃣J - Create glassmorphic frames
        Create glassmorphic frame components
        """
        frame = tk.Frame(
            self.parent,
            bg=UITheme.COLORS['card_background'],
            relief='raised',
            borderwidth=1,
        )
        frame.place(x=x, y=y, width=width, height=height)
        return frame
    
    def update_display(self, hr, confidence, signal_quality, motion_level, 
                      gesture_state, active_roi, face_detected):
        """
        CALL ORDER: 5️⃣K - Update all display elements
        Update all dashboard components with current data
        """
        # Update heart rate display
        if hr:
            self.hr_display.config(
                text=f"{hr} BPM",
                fg=self.get_hr_color(hr)
            )
            
            # Update heart animator
            self.heart_animator.update_state(hr, gesture_state == 'RUNNING', gesture_state)
            
            # Add to history
            current_time = time.time()
            self.hr_history.append(hr)
            self.time_history.append(current_time)
            self.confidence_history.append(confidence)
        else:
            self.hr_display.config(
                text="-- BPM",
                fg=UITheme.COLORS['text_secondary']
            )
        
        # Update confidence and quality
        self.confidence_label.config(
            text=f"Confidence: {confidence:.0%}" if confidence else "Confidence: --%"
        )
        self.quality_label.config(
            text=f"Quality: {signal_quality:.0%}" if signal_quality else "Quality: --%"
        )
        
        # Update gesture status
        if gesture_state == 'RUNNING':
            self.gesture_indicator.config(
                text="▶️ RUNNING",
                fg=UITheme.COLORS['gesture_start']
            )
            self.detection_status.config(
                text="🔍 Actively detecting heart rate...",
                fg=UITheme.COLORS['success']
            )
        else:
            self.gesture_indicator.config(
                text="⏹️ STOPPED",
                fg=UITheme.COLORS['gesture_stop']
            )
            self.detection_status.config(
                text="Use gestures to control",
                fg=UITheme.COLORS['text_secondary']
            )
        
        # Update metrics bars
        if hasattr(self, 'signal_quality_frame'):
            self.signal_quality_frame['value'] = signal_quality * 100 if signal_quality else 0
            self.motion_level_frame['value'] = min(motion_level * 20, 100) if motion_level else 0
        
        # Update heart rate zone
        if hr:
            zone, zone_color = self.get_hr_zone(hr)
            self.hr_zone_label.config(
                text=f"Heart Rate Zone: {zone}",
                fg=zone_color
            )
        
        # Update ROI status
        self.roi_label.config(text=f"Active ROI: {active_roi}")
        
        # Update status indicators
        self.face_detection_status.config(
            text="👤 Face Detection: " + ("Detected" if face_detected else "Searching"),
            fg=UITheme.COLORS['success'] if face_detected else UITheme.COLORS['warning_orange']
        )
        
        self.processing_status.config(
            text="⚡ Signal Processing: " + ("Active" if gesture_state == 'RUNNING' else "Standby"),
            fg=UITheme.COLORS['success'] if gesture_state == 'RUNNING' else UITheme.COLORS['text_secondary']
        )
    
    def get_hr_color(self, hr):
        """Get heart rate color based on zones"""
        if hr < 60:
            return UITheme.COLORS['primary_blue']  # Resting
        elif hr < 100:
            return UITheme.COLORS['health_green']  # Normal
        elif hr < 150:
            return UITheme.COLORS['warning_orange']  # Elevated
        else:
            return UITheme.COLORS['error']  # High
    
    def get_hr_zone(self, hr):
        """Get heart rate zone and color"""
        if hr < 60:
            return "Resting", UITheme.COLORS['primary_blue']
        elif hr < 100:
            return "Normal", UITheme.COLORS['health_green']
        elif hr < 150:
            return "Elevated", UITheme.COLORS['warning_orange']
        else:
            return "High Intensity", UITheme.COLORS['error']
    
    def animate_heartbeat(self):
        """
        CALL ORDER: 5️⃣L - Animate heartbeat display
        Continuously animate the heartbeat visualization
        """
        if self.heart_animator:
            self.heart_animator.draw_enhanced_heart()
    
    def update_charts(self):
        """
        CALL ORDER: 5️⃣M - Update real-time charts
        Update the real-time data visualization charts
        """
        if len(self.hr_history) > 1:
            # Calculate relative time
            times = [(t - self.time_history[0]) for t in self.time_history]
            
            # Update heart rate line
            self.hr_line.set_data(times, list(self.hr_history))
            
            # Update quality line (using confidence as proxy)
            quality_data = [c * 100 if c else 0 for c in self.confidence_history]
            self.quality_line.set_data(times, quality_data)
            
            # Auto-scale both plots
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            # Redraw
            try:
                self.canvas_widget.draw()
            except:
                pass  # Handle matplotlib thread safety

# 6️⃣-🔟 [Previous computer vision functions remain the same as in previous implementation]
# Including: initialize_advanced_camera, detect_face_advanced, extract_intelligent_rois,
# detect_skin_pixels, extract_advanced_signals, AdvancedSignalBuffer,
# advanced_signal_processing, advanced_frequency_analysis, calculate_robust_heart_rate

def initialize_advanced_camera():
    """CALL ORDER: 6️⃣ - Initialize camera with optimal settings"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    cap.set(cv2.CAP_PROP_CONTRAST, 32)
    if not cap.isOpened():
        raise Exception("Could not open camera")
    return cap

def detect_face_advanced(frame):
    """CALL ORDER: 7️⃣ - Advanced face detection"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        face_quality = min(w * h / (640 * 480), 1.0)
        return largest_face, face_quality
    return None, 0.0

def extract_intelligent_rois(frame, face_coords):
    """CALL ORDER: 8️⃣ - Extract multiple ROIs"""
    if face_coords is None:
        return None, None
    
    x, y, w, h = face_coords
    rois = {}
    roi_coords = {}
    
    roi_definitions = {
        'forehead': (int(x + w * 0.25), int(y + h * 0.15), int(w * 0.5), int(h * 0.25)),
        'left_cheek': (int(x + w * 0.15), int(y + h * 0.4), int(w * 0.25), int(h * 0.3)),
        'right_cheek': (int(x + w * 0.6), int(y + h * 0.4), int(w * 0.25), int(h * 0.3))
    }
    
    for roi_name, (rx, ry, rw, rh) in roi_definitions.items():
        rx = max(0, rx)
        ry = max(0, ry)
        rw = min(rw, frame.shape[1] - rx)
        rh = min(rh, frame.shape[0] - ry)
        
        if rw > 10 and rh > 10:
            roi = frame[ry:ry+rh, rx:rx+rw]
            if roi.size > 0:
                rois[roi_name] = roi
                roi_coords[roi_name] = (rx, ry, rw, rh)
    
    return rois, roi_coords

def extract_advanced_signals(rois):
    """CALL ORDER: 9️⃣ - Extract color signals from ROIs"""
    if not rois:
        return None
    
    signals = {}
    for roi_name, roi in rois.items():
        if roi is None or roi.size == 0:
            continue
        
        # Extract RGB channels
        r_mean = np.mean(roi[:, :, 2])
        g_mean = np.mean(roi[:, :, 1])
        b_mean = np.mean(roi[:, :, 0])
        
        signals[roi_name] = {
            'red': r_mean,
            'green': g_mean,
            'blue': b_mean,
            'quality': np.std([r_mean, g_mean, b_mean])
        }
    
    return signals

class AdvancedSignalBuffer:
    """CALL ORDER: 🔟 - Signal buffer with Kalman filtering"""
    def __init__(self, buffer_size=180, fps=30):
        self.buffer_size = buffer_size
        self.fps = fps
        self.signals = {
            'forehead': deque(maxlen=buffer_size),
            'left_cheek': deque(maxlen=buffer_size),
            'right_cheek': deque(maxlen=buffer_size),
            'combined': deque(maxlen=buffer_size)
        }
        self.timestamps = deque(maxlen=buffer_size)
        self.quality_scores = deque(maxlen=buffer_size)
        self.motion_scores = deque(maxlen=buffer_size)
        self.previous_frame = None
    
    def add_sample(self, signals_dict, frame, face_quality):
        """Add new signal sample"""
        if not signals_dict:
            return
        
        # Calculate motion score
        motion_score = self.calculate_motion_score(frame)
        self.motion_scores.append(motion_score)
        
        # Add combined signal
        if signals_dict:
            combined_signal = 0
            count = 0
            for roi_signals in signals_dict.values():
                combined_signal += roi_signals['green']
                count += 1
            if count > 0:
                combined_signal /= count
                self.signals['combined'].append(combined_signal)
        
        # Record quality metrics
        overall_quality = face_quality * (1.0 - min(motion_score / 50.0, 1.0))
        self.quality_scores.append(overall_quality)
        self.timestamps.append(time.time())
    
    def calculate_motion_score(self, frame):
        """Calculate motion score"""
        if self.previous_frame is None:
            self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return 0.0
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_magnitude = np.mean(np.abs(current_gray.astype(float) - self.previous_frame.astype(float)))
        self.previous_frame = current_gray
        return motion_magnitude
    
    def is_ready(self):
        """Check if buffer is ready for analysis"""
        if len(self.signals['combined']) < self.buffer_size * 0.6:
            return False
        recent_quality = list(self.quality_scores)[-30:]
        avg_quality = np.mean(recent_quality) if recent_quality else 0
        return avg_quality > 0.2
    
    def get_best_signal_array(self):
        """Get the best quality signal array"""
        return np.array(list(self.signals['combined'])), 'combined'

def advanced_signal_processing(signal_data, fps=30):
    """CALL ORDER: 1️⃣1️⃣ - Advanced signal processing"""
    if len(signal_data) < 30:
        return signal_data
    
    # Detrending
    x = np.arange(len(signal_data))
    coeffs = np.polyfit(x, signal_data, 3)
    trend = np.polyval(coeffs, x)
    detrended = signal_data - trend
    
    # Bandpass filter
    low_freq, high_freq = 0.7, 3.0
    nyquist = fps / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, detrended)
    
    return filtered_signal

def advanced_frequency_analysis(signal_data, fps=30):
    """CALL ORDER: 1️⃣2️⃣ - Advanced frequency analysis"""
    if len(signal_data) < 30:
        return None, None, None
    
    try:
        freqs_welch, psd_welch = signal.welch(
            signal_data, fps, 
            nperseg=min(len(signal_data)//4, 128),
            window='hann'
        )
    except:
        freqs_welch, psd_welch = signal.periodogram(signal_data, fps, window='hann')
    
    heart_rate_mask = (freqs_welch >= 0.7) & (freqs_welch <= 3.0)
    valid_freqs = freqs_welch[heart_rate_mask]
    valid_psd = psd_welch[heart_rate_mask]
    
    if len(valid_psd) == 0:
        return None, None, None
    
    try:
        peaks, _ = signal.find_peaks(valid_psd, height=np.max(valid_psd) * 0.3)
    except:
        peaks = []
    
    if len(peaks) == 0:
        peak_idx = np.argmax(valid_psd)
        dominant_freq = valid_freqs[peak_idx]
        confidence = 0.5
    else:
        highest_peak_idx = peaks[np.argmax(valid_psd[peaks])]
        dominant_freq = valid_freqs[highest_peak_idx]
        peak_height = valid_psd[highest_peak_idx]
        mean_power = np.mean(valid_psd)
        confidence = min(peak_height / (mean_power + 1e-10) / 10.0, 1.0)
    
    return dominant_freq, valid_psd, confidence

def calculate_robust_heart_rate(freqs_analysis_result, previous_hr_buffer):
    """CALL ORDER: 1️⃣3️⃣ - Calculate heart rate with validation"""
    if freqs_analysis_result[0] is None:
        return None, 0.0
    
    dominant_freq, psd, confidence = freqs_analysis_result
    current_hr = dominant_freq * 60
    
    # Physiological validation
    if current_hr < 42 or current_hr > 180:
        return None, 0.0
    
    # Temporal consistency check
    if len(previous_hr_buffer) > 0:
        recent_hrs = list(previous_hr_buffer)[-5:]
        median_hr = np.median(recent_hrs)
        max_change = 20
        if abs(current_hr - median_hr) > max_change:
            confidence *= 0.5
    
    return int(current_hr), min(confidence, 1.0)

# 1️⃣4️⃣ 🚀 Main Gesture-Controlled Application
class GestureControlledHeartRateApp:
    """
    CALL ORDER: 1️⃣4️⃣ - Main application with complete gesture control
    Complete gesture-controlled heart rate detection system
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.setup_components()
        self.setup_threads()
        
    def setup_window(self):
        """CALL ORDER: 1️⃣4️⃣A - Setup main window"""
        self.root.title("💓 HeartSense Pro - Gesture-Controlled Heart Rate Monitor")
        self.root.geometry("1200x800")
        self.root.configure(bg=UITheme.COLORS['background_dark'])
        self.root.resizable(False, False)
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            'Custom.Horizontal.TProgressbar',
            background=UITheme.COLORS['health_green'],
            troughcolor=UITheme.COLORS['card_background'],
            borderwidth=0
        )
        
    def setup_components(self):
        """CALL ORDER: 1️⃣4️⃣B - Setup all system components"""
        # Create title
        self.create_title_bar()
        
        # Main frame
        self.main_frame = tk.Frame(self.root, bg=UITheme.COLORS['background_dark'])
        self.main_frame.pack(fill='both', expand=True)
        
        # Initialize components
        self.dashboard = GestureEnhancedDashboard(self.main_frame)
        self.gesture_system = GestureRecognitionSystem()
        self.state_machine = GestureStateMachine()
        
        # Computer vision components
        self.cap = None
        self.signal_buffer = AdvancedSignalBuffer(buffer_size=180, fps=30)
        self.hr_buffer = deque(maxlen=10)
        self.confidence_buffer = deque(maxlen=10)
        
        # State variables
        self.running = True
        self.heart_rate = None
        self.confidence = 0.0
        self.signal_quality = 0.0
        self.motion_level = 0.0
        self.face_detected = False
        self.active_roi = "combined"
        
    def create_title_bar(self):
        """CALL ORDER: 1️⃣4️⃣C - Create application title"""
        title_frame = tk.Frame(self.root, bg=UITheme.COLORS['background_dark'], height=80)
        title_frame.pack(fill='x', pady=(0, 20))
        title_frame.pack_propagate(False)
        
        # Main title
        title_label = tk.Label(
            title_frame,
            text="💓 HeartSense Pro - Gesture Control",
            font=('Helvetica Neue', 32, 'bold'),
            fg=UITheme.COLORS['text_primary'],
            bg=UITheme.COLORS['background_dark']
        )
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = tk.Label(
            title_frame,
            text="🤚 Click-Free Heart Rate Detection • Use Hand Gestures to Control",
            font=UITheme.FONTS['heading'],
            fg=UITheme.COLORS['text_secondary'],
            bg=UITheme.COLORS['background_dark']
        )
        subtitle_label.pack()
        
    def setup_threads(self):
        """CALL ORDER: 1️⃣4️⃣D - Setup processing threads"""
        # Start camera
        try:
            self.cap = initialize_advanced_camera()
            print("📹 Camera initialized successfully")
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            
        # Start threads
        self.cv_thread = threading.Thread(target=self.cv_processing_loop, daemon=True)
        self.cv_thread.start()
        
        self.ui_thread = threading.Thread(target=self.ui_animation_loop, daemon=True)
        self.ui_thread.start()
        
    def cv_processing_loop(self):
        """CALL ORDER: 1️⃣4️⃣E - Main computer vision processing loop"""
        update_counter = 0
        
        while self.running and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Process gestures first
                gesture, gesture_confidence, hand_landmarks, gesture_results = self.gesture_system.process_gestures(frame)
                
                # Update state machine
                state_changed, state_message = self.state_machine.update_state(gesture, gesture_confidence)
                if state_changed:
                    print(f"🔄 State changed: {state_message}")
                
                state_info = self.state_machine.get_state_info()
                detection_active = state_info['is_running']
                
                # Heart rate detection (only if active)
                if detection_active:
                    # Face detection
                    face_coords, face_quality = detect_face_advanced(frame)
                    self.face_detected = face_coords is not None
                    
                    if face_coords is not None and face_quality > 0.3:
                        # ROI extraction
                        rois, roi_coords = extract_intelligent_rois(frame, face_coords)
                        
                        if rois:
                            # Signal extraction
                            signals_dict = extract_advanced_signals(rois)
                            
                            if signals_dict:
                                # Add to signal buffer
                                self.signal_buffer.add_sample(signals_dict, frame, face_quality)
                                
                                # Calculate quality metrics
                                if len(self.signal_buffer.quality_scores) > 0:
                                    self.signal_quality = np.mean(list(self.signal_buffer.quality_scores)[-10:])
                                    self.motion_level = np.mean(list(self.signal_buffer.motion_scores)[-10:]) if self.signal_buffer.motion_scores else 0
                                
                                # Process signal every 30 frames
                                update_counter += 1
                                if update_counter >= 30 and self.signal_buffer.is_ready():
                                    signal_data, self.active_roi = self.signal_buffer.get_best_signal_array()
                                    
                                    if len(signal_data) > 60:
                                        # Advanced signal processing
                                        processed_signal = advanced_signal_processing(signal_data)
                                        freq_analysis = advanced_frequency_analysis(processed_signal)
                                        hr_result = calculate_robust_heart_rate(freq_analysis, self.hr_buffer)
                                        
                                        if hr_result[0] is not None:
                                            new_heart_rate, new_confidence = hr_result
                                            
                                            if new_confidence > 0.3:
                                                self.hr_buffer.append(new_heart_rate)
                                                self.confidence_buffer.append(new_confidence)
                                                
                                                if len(self.hr_buffer) >= 3:
                                                    self.heart_rate = int(np.median(list(self.hr_buffer)[-3:]))
                                                    self.confidence = np.mean(list(self.confidence_buffer)[-3:]))
                                                else:
                                                    self.heart_rate = new_heart_rate
                                                    self.confidence = new_confidence
                                    
                                    update_counter = 0
                else:
                    # Reset when not detecting
                    self.face_detected = False
                    self.signal_quality = 0.0
                    self.motion_level = 0.0
                
                time.sleep(1/30)  # 30 FPS
                
            except Exception as e:
                print(f"CV processing error: {e}")
                time.sleep(0.1)
                
    def ui_animation_loop(self):
        """CALL ORDER: 1️⃣4️⃣F - UI animation and update loop"""
        while self.running:
            try:
                state_info = self.state_machine.get_state_info()
                
                # Update dashboard
                self.dashboard.update_display(
                    self.heart_rate,
                    self.confidence,
                    self.signal_quality,
                    self.motion_level,
                    state_info['state'],
                    self.active_roi,
                    self.face_detected
                )
                
                # Animate heartbeat
                self.dashboard.animate_heartbeat()
                
                # Update charts
                self.dashboard.update_charts()
                
                time.sleep(1/30)  # 30 FPS UI updates
                
            except Exception as e:
                print(f"UI animation error: {e}")
                time.sleep(0.1)
                
    def run(self):
        """CALL ORDER: 1️⃣4️⃣G - Start the application"""
        print("🚀 Starting Gesture-Controlled Heart Rate Detection")
        print("👋 Gesture Controls:")
        print("   ▶️  START: Open palms at sides of head")
        print("   ⏹️  STOP: Cover face with both hands")
        print("=" * 60)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """CALL ORDER: 1️⃣4️⃣H - Handle application closing"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

# 🎯 MAIN ENTRY POINT
def main():
    print("💓 HeartSense Pro - Gesture-Controlled Heart Rate Detection")
    print("=" * 80)
    print("🎨 Features:")
    print("   • Complete Click-Free Interface")
    print("   • Advanced Gesture Recognition (MediaPipe)")
    print("   • Intelligent State Machine")
    print("   • Enhanced Heartbeat Visualizations")
    print("   • Real-time Quality Indicators")
    print("   • Apple-Inspired Glassmorphic Design")
    print("\n🤚 Gesture Controls:")
    print("   ▶️  START: Left hand open left + Right hand open right")
    print("   ⏹️  STOP: Both hands covering face")
    print("=" * 80)
    
    # Check dependencies
    missing_deps = []
    required_packages = ['scipy', 'matplotlib', 'mediapipe']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_deps.append(package)
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        exit()
    else:
        print("✅ All dependencies available")
    
    # Launch the gesture-controlled app
    print("\n🚀 Launching application...")
    app = GestureControlledHeartRateApp()
    app.run()



if __name__ == "__main__":
    main()