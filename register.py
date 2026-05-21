#!/usr/bin/env python3
"""
Register a new person for face recognition
"""

from face_camera import FaceRecognitionSystem

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.register_person()
