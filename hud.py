"""
HUD (Heads-Up Display) module for overlay display
"""

import cv2
import numpy as np
from datetime import datetime
from config import (
    HEADER_HEIGHT,
    FOOTER_HEIGHT,
    HEADER_COLOR,
    FOOTER_COLOR,
    TEXT_COLOR,
    KNOWN_BOX_COLOR,
    UNKNOWN_BOX_COLOR,
    BOX_LINE_THICKNESS,
    CORNER_LENGTH_RATIO
)


class HUD:
    """Manages the HUD overlay display"""
    
    @staticmethod
    def draw_header(frame, title="FACE RECOGNITION"):
        """
        Draw the header bar with title and timestamp
        
        Args:
            frame: Input image
            title: Title text to display
        """
        h, w = frame.shape[:2]
        
        # Header background
        header_rect = np.zeros((HEADER_HEIGHT, w, 3), dtype=np.uint8)
        header_rect[:] = HEADER_COLOR
        frame[0:HEADER_HEIGHT, 0:w] = header_rect
        
        # Title
        cv2.putText(frame, title, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Live indicator with timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"[LIVE]  {current_time}", (w - 250, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    @staticmethod
    def draw_footer(frame, face_count, fps, db_count):
        """
        Draw the footer bar with status information
        
        Args:
            frame: Input image
            face_count: Number of faces detected
            fps: Current FPS
            db_count: Number of people in database
        """
        h, w = frame.shape[:2]
        footer_y = h - FOOTER_HEIGHT
        
        # Footer background
        footer_rect = np.zeros((FOOTER_HEIGHT, w, 3), dtype=np.uint8)
        footer_rect[:] = FOOTER_COLOR
        frame[footer_y:h, 0:w] = footer_rect
        
        # Status text
        status_text = f"Faces: {face_count} | FPS: {fps:.1f} | DB: {db_count} people"
        cv2.putText(frame, status_text, (10, h - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    
    @staticmethod
    def draw_corner_box(frame, bbox, color):
        """
        Draw a corner bracket box around a face
        
        Args:
            frame: Input image
            bbox: Bounding box (x, y, width, height)
            color: Box color (B, G, R)
        """
        x, y, w, h = bbox
        line_length = min(w, h) * CORNER_LENGTH_RATIO
        line_length = int(line_length)
        
        # Top-left corner
        cv2.line(frame, (x, y), (x + line_length, y), color, BOX_LINE_THICKNESS)
        cv2.line(frame, (x, y), (x, y + line_length), color, BOX_LINE_THICKNESS)
        
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - line_length, y), color, BOX_LINE_THICKNESS)
        cv2.line(frame, (x + w, y), (x + w, y + line_length), color, BOX_LINE_THICKNESS)
        
        # Bottom-left corner
        cv2.line(frame, (x, y + h), (x + line_length, y + h), color, BOX_LINE_THICKNESS)
        cv2.line(frame, (x, y + h), (x, y + h - line_length), color, BOX_LINE_THICKNESS)
        
        # Bottom-right corner
        cv2.line(frame, (x + w, y + h), (x + w - line_length, y + h), color, BOX_LINE_THICKNESS)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - line_length), color, BOX_LINE_THICKNESS)
    
    @staticmethod
    def draw_confidence(frame, bbox, confidence):
        """
        Draw confidence score above the face box
        
        Args:
            frame: Input image
            bbox: Bounding box (x, y, width, height)
            confidence: Confidence score
        """
        x, y, w, h = bbox
        cv2.putText(frame, f"{confidence:.0f}", (x + w - 30, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    @staticmethod
    def draw_info_card(frame, bbox, name, person_id):
        """
        Draw an info card with name and ID below the face box
        
        Args:
            frame: Input image
            bbox: Bounding box (x, y, width, height)
            name: Person name
            person_id: Person ID
        """
        x, y, w, h = bbox
        card_height = 60
        card_y = y + h + 10
        frame_h, frame_w = frame.shape[:2]
        
        # Check if card fits in frame
        if card_y + card_height >= frame_h:
            return
        
        card_x = max(0, x - 10)
        if card_x + w + 20 > frame_w:
            return
        
        # Card background (semi-transparent)
        card_rect = np.zeros((card_height, w + 20, 3), dtype=np.uint8)
        card_rect[:] = (0, 0, 0)
        card_rect = cv2.addWeighted(card_rect, 0.7, card_rect, 0, 0)
        frame[card_y:card_y+card_height, card_x:card_x+w+20] = card_rect
        
        # Name
        cv2.putText(frame, name, (card_x + 10, card_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
        
        # ID
        cv2.putText(frame, f"ID: {person_id}", (card_x + 10, card_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
    
    @staticmethod
    def draw_face_overlay(frame, face_info):
        """
        Draw complete overlay for a detected face
        
        Args:
            frame: Input image
            face_info: Dictionary with face recognition results
        """
        bbox = face_info['bbox']
        known = face_info['known']
        confidence = face_info['confidence']
        
        # Choose color based on recognition
        color = KNOWN_BOX_COLOR if known else UNKNOWN_BOX_COLOR
        
        # Draw corner box
        HUD.draw_corner_box(frame, bbox, color)
        
        # Draw confidence
        HUD.draw_confidence(frame, bbox, confidence)
        
        # Draw info card if known
        if known:
            HUD.draw_info_card(frame, bbox, face_info['name'], face_info['person_id'])
    
    @staticmethod
    def draw_full_hud(frame, faces_info, fps, db_count):
        """
        Draw complete HUD overlay on frame
        
        Args:
            frame: Input image
            faces_info: List of face recognition results
            fps: Current FPS
            db_count: Number of people in database
        """
        # Draw header
        HUD.draw_header(frame)
        
        # Draw face overlays
        for face_info in faces_info:
            HUD.draw_face_overlay(frame, face_info)
        
        # Draw footer
        HUD.draw_footer(frame, len(faces_info), fps, db_count)
