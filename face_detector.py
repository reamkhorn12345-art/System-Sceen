"""
Face detection module using OpenCV Haar Cascade
"""

import cv2
import numpy as np
from config import (
    HAAR_CASCADE_PATH,
    DETECTION_SCALE_FACTOR,
    DETECTION_MIN_NEIGHBORS,
    MIN_FACE_SIZE,
    FACE_SIZE
)


class FaceDetector:
    """Detects faces in images using Haar Cascade"""
    
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of tuples (x, y, width, height) for each detected face
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=DETECTION_SCALE_FACTOR,
            minNeighbors=DETECTION_MIN_NEIGHBORS
        )
        
        # Filter by minimum size
        valid_faces = []
        for (x, y, w, h) in faces:
            if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
                valid_faces.append((x, y, w, h))
        
        return valid_faces
    
    def extract_face(self, frame, bbox):
        """
        Extract and preprocess a face from the frame
        
        Args:
            frame: Input image (BGR format)
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Preprocessed grayscale face image resized to FACE_SIZE
        """
        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, FACE_SIZE)
        return face_resized
    
    def draw_detection_box(self, frame, bbox, color=(0, 255, 0), thickness=2):
        """
        Draw a detection box around a face
        
        Args:
            frame: Input image
            bbox: Bounding box (x, y, width, height)
            color: Box color (B, G, R)
            thickness: Line thickness
        """
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
