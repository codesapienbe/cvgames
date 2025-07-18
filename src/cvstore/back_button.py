"""
CVGames Back Button System

This module provides a standardized back button with approval dialog
that can be used by all games to return to the app store.

Usage:
    from cvstore.back_button import BackButton
    
    back_button = BackButton()
    
    # In your game loop:
    if back_button.handle_input(key, hand_landmarks, hand_position):
        # User approved exit, return to app store
        return True
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Tuple, Optional

class BackButton:
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        """
        Initialize the back button system
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Back button position (top-left corner)
        self.button_rect = (50, 50, 120, 60)
        
        # Approval dialog state
        self.show_approval_dialog = False
        self.approval_timer = 0
        self.approval_timeout = 5.0  # 5 seconds to approve
        
        # Colors
        self.colors = {
            'button_bg': (100, 150, 255),
            'button_hover': (120, 170, 255),
            'button_text': (255, 255, 255),
            'dialog_bg': (30, 30, 45),
            'dialog_border': (100, 150, 255),
            'dialog_text': (255, 255, 255),
            'yes_button': (80, 200, 120),
            'no_button': (200, 80, 80),
            'warning': (255, 180, 80)
        }
        
        # Hand tracking for gesture detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.4,
            model_complexity=0
        )
    
    def detect_index_middle_pinch(self, landmarks) -> bool:
        """Detect if index and middle fingers are pinched together (click gesture)"""
        if len(landmarks.landmark) < 21:
            return False
        
        # Get index and middle finger tip positions
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        
        # Calculate distance between index and middle finger tips
        distance = ((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2) ** 0.5
        
        # Check if other fingers are extended (not closed)
        thumb_open = landmarks.landmark[4].y < landmarks.landmark[3].y
        ring_open = landmarks.landmark[16].y < landmarks.landmark[14].y
        pinky_open = landmarks.landmark[20].y < landmarks.landmark[18].y
        
        # Pinch threshold (adjust as needed)
        pinch_threshold = 0.05  # 5% of screen size
        
        return distance < pinch_threshold and thumb_open and ring_open and pinky_open
    
    def is_point_in_rect(self, point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
        """Check if a point is inside a rectangle"""
        x, y = point
        rect_x, rect_y, rect_w, rect_h = rect
        return rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h
    
    def handle_input(self, key: int, hand_landmarks: Optional[mp.solutions.hands.HandLandmark] = None, 
                    hand_position: Optional[Tuple[int, int]] = None) -> bool:
        """
        Handle input for back button (keyboard and gesture)
        
        Args:
            key: Keyboard key pressed
            hand_landmarks: MediaPipe hand landmarks
            hand_position: Hand position (x, y)
        
        Returns:
            True if user approved exit, False otherwise
        """
        # Keyboard input
        if key == ord('b') or key == ord('B'):
            self.show_approval_dialog = True
            self.approval_timer = time.time()
            return False
        
        # Gesture input
        if hand_landmarks and hand_position:
            hand_x, hand_y = hand_position
            
            # Check if clicking on back button
            if self.is_point_in_rect((hand_x, hand_y), self.button_rect):
                if self.detect_index_middle_pinch(hand_landmarks):
                    self.show_approval_dialog = True
                    self.approval_timer = time.time()
                    return False
            
            # Handle approval dialog gestures
            if self.show_approval_dialog:
                # Yes button (left side)
                yes_rect = (self.screen_width // 2 - 200, self.screen_height // 2 + 50, 150, 60)
                # No button (right side)
                no_rect = (self.screen_width // 2 + 50, self.screen_height // 2 + 50, 150, 60)
                
                if self.detect_index_middle_pinch(hand_landmarks):
                    if self.is_point_in_rect((hand_x, hand_y), yes_rect):
                        return True  # User approved exit
                    elif self.is_point_in_rect((hand_x, hand_y), no_rect):
                        self.show_approval_dialog = False  # User cancelled
                        return False
        
        # Check for timeout
        if self.show_approval_dialog and time.time() - self.approval_timer > self.approval_timeout:
            self.show_approval_dialog = False
        
        return False
    
    def draw(self, frame: np.ndarray, hand_position: Optional[Tuple[int, int]] = None):
        """
        Draw the back button and approval dialog
        
        Args:
            frame: OpenCV frame to draw on
            hand_position: Current hand position for hover effects
        """
        # Draw back button
        x, y, w, h = self.button_rect
        
        # Check if hand is hovering over button
        is_hovered = False
        if hand_position:
            is_hovered = self.is_point_in_rect(hand_position, self.button_rect)
        
        # Button background
        color = self.colors['button_hover'] if is_hovered else self.colors['button_bg']
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['button_text'], 2)
        
        # Button text
        text = "BACK"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, 0.8, 2)
        text_x = x + (w - text_width) // 2
        text_y = y + (h + text_height) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, 0.8, self.colors['button_text'], 2)
        
        # Draw approval dialog if active
        if self.show_approval_dialog:
            self.draw_approval_dialog(frame, hand_position)
    
    def draw_approval_dialog(self, frame: np.ndarray, hand_position: Optional[Tuple[int, int]] = None):
        """Draw the approval dialog"""
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Dialog background
        dialog_width = 600
        dialog_height = 300
        dialog_x = (self.screen_width - dialog_width) // 2
        dialog_y = (self.screen_height - dialog_height) // 2
        
        # Dialog background with glassmorphic effect
        for i in range(2):
            alpha = 0.4 - i * 0.2
            overlay = frame.copy()
            cv2.rectangle(overlay, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height), 
                         self.colors['dialog_bg'], -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Dialog border
        cv2.rectangle(frame, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height), 
                     self.colors['dialog_border'], 3)
        
        # Dialog title
        title = "Exit Game?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (title_width, title_height), _ = cv2.getTextSize(title, font, 1.5, 3)
        title_x = dialog_x + (dialog_width - title_width) // 2
        title_y = dialog_y + 60
        
        cv2.putText(frame, title, (title_x, title_y), font, 1.5, self.colors['dialog_text'], 3)
        
        # Dialog message
        message = "Are you sure you want to return to the App Store?"
        (msg_width, msg_height), _ = cv2.getTextSize(message, font, 0.8, 2)
        msg_x = dialog_x + (dialog_width - msg_width) // 2
        msg_y = dialog_y + 120
        
        cv2.putText(frame, message, (msg_x, msg_y), font, 0.8, self.colors['dialog_text'], 2)
        
        # Yes button
        yes_rect = (dialog_x + 100, dialog_y + 180, 150, 60)
        yes_x, yes_y, yes_w, yes_h = yes_rect
        
        # Check if hovering over Yes button
        yes_hovered = False
        if hand_position:
            yes_hovered = self.is_point_in_rect(hand_position, yes_rect)
        
        yes_color = (100, 220, 140) if yes_hovered else self.colors['yes_button']
        cv2.rectangle(frame, (yes_x, yes_y), (yes_x + yes_w, yes_y + yes_h), yes_color, -1)
        cv2.rectangle(frame, (yes_x, yes_y), (yes_x + yes_w, yes_y + yes_h), self.colors['button_text'], 2)
        
        yes_text = "YES"
        (yes_text_width, yes_text_height), _ = cv2.getTextSize(yes_text, font, 0.9, 2)
        yes_text_x = yes_x + (yes_w - yes_text_width) // 2
        yes_text_y = yes_y + (yes_h + yes_text_height) // 2
        
        cv2.putText(frame, yes_text, (yes_text_x, yes_text_y), font, 0.9, self.colors['button_text'], 2)
        
        # No button
        no_rect = (dialog_x + 350, dialog_y + 180, 150, 60)
        no_x, no_y, no_w, no_h = no_rect
        
        # Check if hovering over No button
        no_hovered = False
        if hand_position:
            no_hovered = self.is_point_in_rect(hand_position, no_rect)
        
        no_color = (220, 100, 100) if no_hovered else self.colors['no_button']
        cv2.rectangle(frame, (no_x, no_y), (no_x + no_w, no_y + no_h), no_color, -1)
        cv2.rectangle(frame, (no_x, no_y), (no_x + no_w, no_y + no_h), self.colors['button_text'], 2)
        
        no_text = "NO"
        (no_text_width, no_text_height), _ = cv2.getTextSize(no_text, font, 0.9, 2)
        no_text_x = no_x + (no_w - no_text_width) // 2
        no_text_y = no_y + (no_h + no_text_height) // 2
        
        cv2.putText(frame, no_text, (no_text_x, no_text_y), font, 0.9, self.colors['button_text'], 2)
        
        # Timer warning
        remaining_time = max(0, int(self.approval_timeout - (time.time() - self.approval_timer)))
        timer_text = f"Auto-cancel in {remaining_time}s"
        (timer_width, timer_height), _ = cv2.getTextSize(timer_text, font, 0.6, 1)
        timer_x = dialog_x + (dialog_width - timer_width) // 2
        timer_y = dialog_y + dialog_height - 20
        
        cv2.putText(frame, timer_text, (timer_x, timer_y), font, 0.6, self.colors['warning'], 1)
    
    def get_hand_position(self, landmarks) -> Tuple[int, int]:
        """Get hand position in screen coordinates"""
        if len(landmarks.landmark) < 21:
            return (0, 0)
        
        # Use palm center (landmark 9 - middle finger base)
        x = int(landmarks.landmark[9].x * self.screen_width)
        y = int(landmarks.landmark[9].y * self.screen_height)
        return (x, y) 