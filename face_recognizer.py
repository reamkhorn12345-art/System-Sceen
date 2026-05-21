"""
Face recognition module using OpenCV LBPH
"""

import cv2
import numpy as np
import os
from config import MODEL_FILE, RECOGNITION_THRESHOLD


class FaceRecognizer:
    """Recognizes faces using LBPH algorithm"""
    
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.is_trained = False
        
        # Load model if exists
        if os.path.exists(MODEL_FILE):
            self.load_model()
    
    def train(self, faces, labels):
        """
        Train the face recognition model
        
        Args:
            faces: List of face images (grayscale)
            labels: List of corresponding labels (person IDs)
        """
        if not faces:
            raise ValueError("No faces provided for training")
        
        self.recognizer.train(faces, np.array(labels))
        self.save_model()
        self.is_trained = True
    
    def predict(self, face):
        """
        Predict the identity of a face
        
        Args:
            face: Face image (grayscale)
            
        Returns:
            Tuple (label, confidence) where confidence is lower for better matches
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained")
        
        return self.recognizer.predict(face)
    
    def recognize(self, face, database):
        """
        Recognize a face and return person info
        
        Args:
            face: Face image (grayscale)
            database: PeopleDatabase instance
            
        Returns:
            Dictionary with 'known', 'name', 'person_id', 'confidence'
        """
        label, confidence = self.predict(face)
        known = confidence < RECOGNITION_THRESHOLD
        
        result = {
            'known': known,
            'confidence': confidence,
            'label': label
        }
        
        if known:
            person = database.get_person(label)
            if person:
                result['name'] = person['name']
                result['person_id'] = person['id']
            else:
                result['name'] = 'Unknown'
                result['person_id'] = 'N/A'
        else:
            result['name'] = 'Unknown'
            result['person_id'] = 'N/A'
        
        return result
    
    def save_model(self):
        """Save the trained model to file"""
        self.recognizer.save(MODEL_FILE)
    
    def load_model(self):
        """Load a trained model from file"""
        self.recognizer.read(MODEL_FILE)
        self.is_trained = True
    
    def load_training_data(self, database):
        """
        Load all face samples from database for training
        
        Args:
            database: PeopleDatabase instance
            
        Returns:
            Tuple (faces, labels) for training
        """
        faces = []
        labels = []
        
        for person_id_str in database.get_all_people().keys():
            person_id = int(person_id_str)
            sample_paths = database.get_face_samples(person_id)
            
            for sample_path in sample_paths:
                face = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
                if face is not None:
                    faces.append(face)
                    labels.append(person_id)
        
        return faces, labels
