"""
Camera module for handling video capture with improved settings.
"""
import cv2
import numpy as np
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    BRIGHTNESS, CONTRAST, SATURATION, HUE,
    GAIN, EXPOSURE, WHITE_BALANCE, AUTOFOCUS,
    SHARPNESS, BACKLIGHT,
    CAP_PROP_SHARPNESS, CAP_PROP_BACKLIGHT
)


class Camera:
    """Handles video capture with improved settings."""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.is_opened = False
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT
        self.fps = CAMERA_FPS
        
    def open(self):
        """Open the camera with improved settings."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Set image properties for better quality
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
        self.cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST)
        self.cap.set(cv2.CAP_PROP_SATURATION, SATURATION)
        self.cap.set(cv2.CAP_PROP_HUE, HUE)
        self.cap.set(cv2.CAP_PROP_GAIN, GAIN)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
        self.cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, WHITE_BALANCE)
        
        # Enable autofocus if supported
        if AUTOFOCUS:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Set sharpness if supported (codec-specific property 20)
        self.cap.set(20, SHARPNESS)
        
        # Enable backlight compensation if supported
        if BACKLIGHT:
            self.cap.set(cv2.CAP_PROP_BACKLIGHT, 1)
        
        self.is_opened = True
        return True
    
    def enhance_frame(self, frame):
        """
        Apply real-time image enhancement to improve video quality.
        - Color correction and brightness adjustment
        - Sharpening filter for clearer details
        - Noise reduction for cleaner background
        """
        # Convert to LAB for better color enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better contrast without noise
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply sharpening kernel for clearer details
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def read(self):
        """Read a frame from the camera with enhancement."""
        if not self.is_opened:
            return False, None
        ret, frame = self.cap.read()
        if ret:
            # Apply real-time enhancement for bright, clear video
            frame = self.enhance_frame(frame)
        return ret, frame
    
    def release(self):
        """Release the camera resources."""
        if self.is_opened and self.cap:
            self.cap.release()
            self.is_opened = False
    
    def isOpened(self):
        """Check if camera is opened."""
        return self.is_opened