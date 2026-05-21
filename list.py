#!/usr/bin/env python3
"""
List all registered people
"""

from face_camera import FaceRecognitionSystem

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.list_people()
