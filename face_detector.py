"""
Face detection module with improved accuracy and stability.
"""
import cv2
import numpy as np
from collections import deque
from config import (
    HAAR_CASCADE_PATH,
    DETECTION_SCALE_FACTOR,
    DETECTION_MIN_NEIGHBORS,
    MIN_FACE_SIZE,
    MAX_FACE_SIZE,
    MAX_FACE_HISTORY,
    FACE_HISTORY_DECAY,
    FACE_SMOOTH_FACTOR,
    FACE_CLARITY_FACTOR
)


class FaceDetector:
    """Handles face detection with improved accuracy and stability."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")
        
        # For stabilizing face bounding boxes
        self.face_history = deque(maxlen=MAX_FACE_HISTORY)
        
        # CLAHE for improving detection in low light
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def enhance_frame(self, gray):
        """
        Apply CLAHE and histogram equalization to improve contrast in bad lighting.
        This helps catch faces better in various lighting conditions.
        """
        # Apply CLAHE for local contrast enhancement
        clahe_enhanced = self.clahe.apply(gray)
        
        # Apply global histogram equalization for overall brightness normalization
        hist_eq = cv2.equalizeHist(gray)
        
        # Blend both methods for best results
        enhanced = cv2.addWeighted(clahe_enhanced, 0.7, hist_eq, 0.3, 0)
        
        return enhanced
    
    def detect_faces_multi_scale(self, gray):
        """
        Detect faces using multiple scales for better catching.
        Combines detections from different scale factors to find faces at various distances.
        """
        all_faces = []
        
        # Try multiple scale factors for better detection
        scale_factors = [1.05, 1.1, 1.15]
        
        for scale in scale_factors:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale,
                minNeighbors=DETECTION_MIN_NEIGHBORS,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
                maxSize=(MAX_FACE_SIZE, MAX_FACE_SIZE)
            )
            all_faces.extend(faces)
        
        # Remove duplicate detections using non-maximum suppression
        if len(all_faces) > 1:
            all_faces = self._non_max_suppression(all_faces, 0.3)
        
        return all_faces
    
    def _non_max_suppression(self, faces, overlap_threshold):
        """
        Remove overlapping face detections using non-maximum suppression.
        Keeps the largest face in each overlapping group.
        """
        if not faces:
            return faces
        
        # Sort by area (largest first)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        
        keep = []
        while faces:
            current = faces.pop(0)
            keep.append(current)
            
            # Remove faces that overlap significantly with current
            faces = [f for f in faces if self._iou(current, f) < overlap_threshold]
        
        return keep
    
    def _iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) for two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def stabilize_face_boxes(self, faces):
        """
        Stabilize face bounding boxes to reduce shaking/flickering.
        Uses historical data to smooth the bounding box coordinates.
        """
        if not faces:
            self.face_history.clear()
            return []
        
        # Add current faces to history
        self.face_history.append(faces)
        
        # If we don't have enough history yet, return current faces
        if len(self.face_history) < 2:
            return faces
        
        # Stabilize each face by averaging with recent history
        stabilized_faces = []
        for i, face in enumerate(faces):
            if i < len(self.face_history[-2]):  # Make sure we have a corresponding face in previous frame
                prev_face = self.face_history[-2][i]
                
                # Apply exponential moving average for smoothing
                stabilized_x = int(face[0] * (1 - FACE_HISTORY_DECAY) + prev_face[0] * FACE_HISTORY_DECAY)
                stabilized_y = int(face[1] * (1 - FACE_HISTORY_DECAY) + prev_face[1] * FACE_HISTORY_DECAY)
                stabilized_w = int(face[2] * (1 - FACE_HISTORY_DECAY) + prev_face[2] * FACE_HISTORY_DECAY)
                stabilized_h = int(face[3] * (1 - FACE_HISTORY_DECAY) + prev_face[3] * FACE_HISTORY_DECAY)
                
                stabilized_faces.append((stabilized_x, stabilized_y, stabilized_w, stabilized_h))
            else:
                stabilized_faces.append(face)
        
        return stabilized_faces
    
    def smooth_face_skin(self, face_img, smooth_factor=0.5):
        """
        Apply skin smoothing to a face image while preserving details like eyes/mouth.
        Uses bilateral filter for smoothing while keeping edges sharp.
        
        Args:
            face_img: Grayscale face image
            smooth_factor: 0.0 (no smoothing) to 1.0 (heavy smoothing)
        
        Returns:
            Smoothed face image
        """
        if smooth_factor <= 0:
            return face_img
        
        # Convert to BGR for filtering then back to grayscale
        face_bgr = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
        
        # Determine filter parameters based on smooth factor
        d = int(9 + smooth_factor * 7)  # Diameter: 9-16
        sigma_color = int(30 + smooth_factor * 40)  # Color sigma: 30-70
        sigma_space = int(30 + smooth_factor * 40)  # Space sigma: 30-70
        
        # Apply bilateral filter for skin smoothing
        smoothed = cv2.bilateralFilter(face_bgr, d, sigma_color, sigma_space)
        
        # Convert back to grayscale
        return cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    
    def enhance_face_clarity(self, face_img, clarity_factor=1.0):
        """
        Enhance face clarity with sharpening and contrast adjustment.
        Combines unsharp mask with CLAHE for better facial details.
        
        Args:
            face_img: Grayscale face image
            clarity_factor: 0.0-2.0 strength of enhancement
        
        Returns:
            Enhanced face image
        """
        if clarity_factor <= 0:
            return face_img
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0 + clarity_factor, tileGridSize=(4, 4))
        enhanced = clahe.apply(face_img)
        
        # Unsharp mask for sharpening
        if clarity_factor > 0.3:
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
            sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            enhanced = sharpened
        
        return enhanced
    
    def detect_faces(self, frame):
        """
        Main method to detect faces in a frame with all improvements.
        Returns list of stabilized face bounding boxes.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance frame for better detection in various lighting conditions
        enhanced_gray = self.enhance_frame(gray)
        
        # Detect faces using multi-scale approach
        faces = self.detect_faces_multi_scale(enhanced_gray)
        
        # Stabilize face bounding boxes to reduce shaking
        stabilized_faces = self.stabilize_face_boxes(faces)
        
        return stabilized_faces