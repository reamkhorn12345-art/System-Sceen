"""
Configuration constants and settings for the face recognition system
"""

import cv2

# File paths
DB_FILE = "people_db.json"
MODEL_FILE = "face_model.xml"
FACES_DIR = "known_faces"

# Face detection settings
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# Improved detection parameters for better accuracy
DETECTION_SCALE_FACTOR = 1.1  # Lower = more scales checked
DETECTION_MIN_NEIGHBORS = 4   # Slightly lower for multi-scale detection
MIN_FACE_SIZE = 60            # Minimum face size in pixels
MAX_FACE_SIZE = 400           # Maximum face size in pixels

# Face recognition settings
FACE_SIZE = (100, 100)
SAMPLES_PER_PERSON = 40
RECOGNITION_THRESHOLD = 70    # Lower = stricter match required

# HUD settings
HEADER_HEIGHT = 50
FOOTER_HEIGHT = 40
HEADER_COLOR = (14, 20, 36)           # Dark blue-black
FOOTER_COLOR = (14, 20, 36)           # Dark blue-black
TEXT_COLOR = (255, 255, 255)          # White
KNOWN_BOX_COLOR = (0, 230, 130)       # Mint green - recognised / OK
UNKNOWN_BOX_COLOR = (60, 60, 220)     # Red - unknown / alert
BOX_LINE_THICKNESS = 4
CORNER_LENGTH_RATIO = 0.2

# Camera settings - Enhanced for better quality with HD / 4K support
CAMERA_INDEX = 0
CAMERA_WIDTH = 1920   # Full HD width (use 3840 for 4K if supported)
CAMERA_HEIGHT = 1080  # Full HD height (use 2160 for 4K if supported)
CAMERA_FPS = 30       # Target FPS for smoother video

# Image enhancement settings for bright, clear video
BRIGHTNESS = 0.7      # Increased brightness (0-1) - was 0.5
CONTRAST = 1.5        # Higher contrast for sharper details (0-3) - was 1.2
SATURATION = 1.3      # Enhanced saturation for vivid colors (0-3) - was 1.1
HUE = 0               # Adjust hue (-180 to 180)
GAIN = 0              # Digital gain (0-100)
EXPOSURE = -3         # Slightly higher exposure (-7 to 1) - was -4
WHITE_BALANCE = 5000  # Better white balance (2800-6500K) - was 4500
SHARPNESS = 128       # Increased sharpness (0-255) - new setting
BACKLIGHT = 1         # Enable backlight compensation (0=off, 1=on) - new
AUTOFOCUS = 1         # Enable autofocus (0=off, 1=on)

# Camera property constants for additional settings
CAP_PROP_SHARPNESS = 20      # Sharpness property ID
CAP_PROP_BACKLIGHT = 44      # Backlight compensation property ID

# Window names
WINDOW_NAME_RECOGNITION = "Face Recognition System"
WINDOW_NAME_REGISTRATION = "Registration"

# Performance settings
USE_ASYNC_PROCESSING = True
MAX_FACE_HISTORY = 10  # For stabilizing face bounding boxes
FACE_HISTORY_DECAY = 0.8  # How quickly old face positions decay

# Face smoothing settings for registration
FACE_SMOOTH_FACTOR = 0.6   # Skin smoothing strength (0.0-1.0)
FACE_CLARITY_FACTOR = 1.2  # Clarity enhancement strength (0.0-2.0)
