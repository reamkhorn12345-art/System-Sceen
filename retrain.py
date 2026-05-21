#!/usr/bin/env python3
"""
Retrain the face recognition model
"""

from face_camera import FaceRecognitionSystem

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.train_model()
