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
DETECTION_SCALE_FACTOR = 1.3
DETECTION_MIN_NEIGHBORS = 5
MIN_FACE_SIZE = 50

# Face recognition settings
FACE_SIZE = (100, 100)
SAMPLES_PER_PERSON = 40
RECOGNITION_THRESHOLD = 80

# HUD settings
HEADER_HEIGHT = 50
FOOTER_HEIGHT = 40
HEADER_COLOR = (30, 30, 40)
FOOTER_COLOR = (30, 30, 40)
TEXT_COLOR = (255, 255, 255)
KNOWN_BOX_COLOR = (0, 255, 0)
UNKNOWN_BOX_COLOR = (0, 0, 255)
BOX_LINE_THICKNESS = 3
CORNER_LENGTH_RATIO = 0.25

# Camera settings
CAMERA_INDEX = 0
WINDOW_NAME_RECOGNITION = "Face Recognition System"
WINDOW_NAME_REGISTRATION = "Registration"
