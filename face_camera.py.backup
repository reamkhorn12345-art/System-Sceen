#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║           FACE RECOGNITION CAMERA SYSTEM  v2.0                  ║
║                                                                  ║
║  NEW IN v2.0                                                     ║
║  ─────────────────────────────────────────────────────────────  ║
║  Camera features:                                                ║
║    • Auto brightness / contrast normalisation (CLAHE)           ║
║    • Mirror flip toggle  (press M)                               ║
║    • Screenshot capture  (press S  →  saves PNG)                 ║
║    • Freeze frame        (press F  →  pause / unpause)           ║
║    • Zoom in / out       (press Z / X)                           ║
║    • Confidence threshold slider (press + / -)                   ║
║    • Unknown-face alert  (flashing red border)                   ║
║    • Recognition log     (last 5 events shown on screen)         ║
║    • Session stats panel (total seen, known %, uptime)           ║
║                                                                  ║
║  Registration features:                                          ║
║    • Animated guide ring that turns green when face is centred   ║
║    • Live face-quality meter (blur / brightness score)           ║
║    • Auto-pause if face moves out of frame                       ║
║    • Countdown beep before capture starts                        ║
║    • Preview thumbnail strip of captured samples                 ║
║    • Role / department field added to profile                    ║
║                                                                  ║
║  MODES                                                           ║
║    python face_camera.py              → live recognition         ║
║    python face_camera.py --register   → register new person      ║
║    python face_camera.py --retrain    → train model              ║
║    python face_camera.py --list       → list all people          ║
║    python face_camera.py --delete ID  → remove person            ║
║    python face_camera.py --export     → export log to CSV        ║
║                                                                  ║
║  INSTALL                                                         ║
║    pip install opencv-contrib-python pyttsx3                     ║
╚══════════════════════════════════════════════════════════════════╝
"""
 
import cv2
import numpy as np
import json
import os
import sys
import csv
import argparse
import time
import shutil
import threading
from datetime import datetime
from pathlib import Path
import pyttsx3
 
 
# ─────────────────────────────────────────────────────────────────────────────
# COLOUR THEME  (BGR format for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
C_BG        = (14,  20,  36)
C_PANEL     = (20,  30,  55)
C_ACCENT    = (0,  180, 255)   # electric blue
C_GREEN     = (0,  230, 130)   # mint green — recognised / OK
C_RED       = (60,  60, 220)   # red — unknown / alert
C_YELLOW    = (30, 210, 255)   # amber — warning / badge
C_WHITE     = (255, 255, 255)
C_MUTED     = (140, 155, 175)
C_DARK_RED  = (30,  25, 140)
C_ORANGE    = (30, 160, 255)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# HELPER — semi-transparent rectangle (frosted-glass look)
# ─────────────────────────────────────────────────────────────────────────────
def frosted(frame, x1, y1, x2, y2, color=C_BG, alpha=0.78):
    """Blend a solid colour rectangle over the frame at given opacity."""
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    if y2 <= y1 or x2 <= x1:
        return
    roi     = frame[y1:y2, x1:x2]
    overlay = np.full(roi.shape, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────
class FaceRecognitionSystem:
 
    def __init__(self):
        # ── Haar cascade face detector ────────────────────────────────────────
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # ── Improved detection parameters for better face catching ────────────
        self.detect_scale_factor = 1.1  # Lower = more scales checked (was 1.3)
        self.detect_min_neighbors = 5   # Higher = more strict (was 5)
        self.detect_min_size = 60       # Minimum face size in pixels (was not set)
        self.detect_max_size = 400      # Maximum face size in pixels (new)
 
        # ── Paths ─────────────────────────────────────────────────────────────
        self.db_file    = "people_db.json"
        self.model_file = "face_model.xml"
        self.faces_dir  = "known_faces"
        self.log_file   = "recognition_log.json"
        Path(self.faces_dir).mkdir(exist_ok=True)
        Path("screenshots").mkdir(exist_ok=True)
 
        # ── Data ──────────────────────────────────────────────────────────────
        self.people_db   = self._load_json(self.db_file,  {})
        self.rec_log     = self._load_json(self.log_file, [])  # recognition events
 
        # ── LBPH recogniser ───────────────────────────────────────────────────
        self.recognizer  = cv2.face.LBPHFaceRecognizer_create()
 
        # ── CLAHE — improves contrast in poor lighting ─────────────────────
        # clipLimit: how much contrast boost (higher = more aggressive)
        # tileGridSize: divides image into blocks for local enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
 
        # ── TTS ───────────────────────────────────────────────────────────────
        self.tts_engine            = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.last_recognized_name  = None
        self.last_recognition_time = 0
        self.recognition_cooldown  = 5   # seconds
 
        # ── Runtime state (used during recognition loop) ──────────────────────
        self.confidence_threshold  = 70   # lower = stricter match required (improved from 80)
        self.mirror_mode           = False
        self.freeze_frame          = False
        self.zoom_factor           = 1.0  # 1.0 = no zoom, 2.0 = 2× zoom
        self.session_start         = time.time()
        self.session_total_faces   = 0
        self.session_known_faces   = 0
        self.recent_events         = []   # last 5 recognition strings for HUD log
 
    # ─────────────────────────────────────────────────────────────────────────
    # JSON HELPERS
    # ─────────────────────────────────────────────────────────────────────────
 
    def _load_json(self, path, default):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return default
 
    def _save_json(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
 
    def save_database(self):
        self._save_json(self.db_file, self.people_db)
 
    def save_log(self):
        self._save_json(self.log_file, self.rec_log[-500:])  # keep last 500 events
 
    def get_next_person_id(self):
        if not self.people_db:
            return 0
        return max(int(k) for k in self.people_db.keys()) + 1
 
    # ─────────────────────────────────────────────────────────────────────────
    # IMAGE PROCESSING HELPERS
    # ─────────────────────────────────────────────────────────────────────────
 
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
                minNeighbors=4,  # Slightly lower for multi-scale
                minSize=(self.detect_min_size, self.detect_min_size),
                maxSize=(self.detect_max_size, self.detect_max_size)
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
 
    def apply_zoom(self, frame):
        """
        Crop the centre of the frame and resize back to original size.
        zoom_factor=1.0 → no change; zoom_factor=2.0 → 2× magnification.
        """
        if self.zoom_factor <= 1.0:
            return frame
        fh, fw = frame.shape[:2]
        # Calculate crop window
        new_h = int(fh / self.zoom_factor)
        new_w = int(fw / self.zoom_factor)
        y1 = (fh - new_h) // 2
        x1 = (fw - new_w) // 2
        cropped = frame[y1:y1+new_h, x1:x1+new_w]
        return cv2.resize(cropped, (fw, fh))
 
    def face_quality_score(self, face_gray):
        """
        Return a 0–100 quality score for a face crop.
        Based on:
          • Laplacian variance  → measures sharpness (blur detection)
          • Mean brightness     → too dark or too bright = low quality
          • Contrast            → good contrast = better quality
        """
        # Sharpness: Laplacian variance — blurry faces score near 0
        blur_score = cv2.Laplacian(face_gray, cv2.CV_64F).var()
        sharpness  = min(100, blur_score / 5)   # normalise to 0-100

        # Brightness: ideal is around 120 out of 255
        brightness     = np.mean(face_gray)
        brightness_ok  = 100 - abs(brightness - 120) / 1.2   # 0-100

        # Contrast: standard deviation - higher is better
        contrast = np.std(face_gray)
        contrast_ok = min(100, contrast * 2)   # normalise to 0-100

        return int((sharpness * 0.5) + (brightness_ok * 0.3) + (contrast_ok * 0.2))
 
    # ─────────────────────────────────────────────────────────────────────────
    # DRAWING HELPERS
    # ─────────────────────────────────────────────────────────────────────────
 
    def corner_box(self, frame, x, y, w, h, color, thickness=4):
        """Corner-bracket rectangle (no full border — cleaner look)."""
        arm = min(w, h) // 5
        pts = [
            ((x, y),       (x+arm, y),     (x,   y+arm)),
            ((x+w, y),     (x+w-arm, y),   (x+w, y+arm)),
            ((x, y+h),     (x+arm, y+h),   (x,   y+h-arm)),
            ((x+w, y+h),   (x+w-arm, y+h), (x+w, y+h-arm)),
        ]
        for corner in pts:
            cv2.line(frame, corner[0], corner[1], color, thickness)
            cv2.line(frame, corner[0], corner[2], color, thickness)
 
    def corner_box_glow(self, frame, x, y, w, h, color, thickness=5):
        """Corner box with darker shadow layer for a glow effect."""
        shadow = tuple(max(0, c - 70) for c in color)
        self.corner_box(frame, x, y, w, h, shadow, thickness + 3)
        self.corner_box(frame, x, y, w, h, color,  thickness)
 
    def text_shadow(self, frame, text, pos, scale, color, thickness=2):
        """Draw text with a black drop-shadow for readability."""
        x, y = pos
        cv2.putText(frame, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness)
 
    def quality_bar(self, frame, x, y, width, score, label="QUALITY"):
        """
        Draw a horizontal quality bar (0-100).
        Green when good (>70), amber mid, red low.
        """
        bar_h  = 8
        fill_w = int(width * score / 100)
        color  = C_GREEN if score > 70 else (C_YELLOW if score > 40 else C_RED)
        # Background
        cv2.rectangle(frame, (x, y), (x+width, y+bar_h), (30, 40, 60), -1)
        # Fill
        cv2.rectangle(frame, (x, y), (x+fill_w, y+bar_h), color, -1)
        # Label
        cv2.putText(frame, f"{label} {score}%", (x, y+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
 
    # ─────────────────────────────────────────────────────────────────────────
    # REGISTRATION UI  (modernised v2)
    # ─────────────────────────────────────────────────────────────────────────
 
    def draw_registration_ui(self, frame, sample_count, samples_needed,
                              name, person_id, role,
                              quality_score, face_centered, thumbnails):
        """
        Full-featured registration HUD.
 
        New elements vs v1:
          • Role / department field in info card
          • Live face-quality bar (blur + brightness score)
          • Animated spinning outer ring — turns green once centred
          • Sweeping progress arc on the guide ring
          • Thumbnail strip of the last 5 captured samples
          • 3-step pill tracker: ALIGN → CAPTURE → DONE
          • Mirror / quality / countdown hints in the corner
 
        Returns (bx, by, bw, bh) — STOP button coords for click detection.
        """
        fh, fw = frame.shape[:2]
        progress = sample_count / samples_needed   # 0.0 → 1.0
        tick     = time.time()
 
        # ── HEADER ────────────────────────────────────────────────────────────
        frosted(frame, 0, 0, fw, 75, C_BG, 0.88)
        cv2.rectangle(frame, (0, 73), (fw, 75), C_ACCENT, -1)   # accent line
 
        cv2.putText(frame, "BIOMETRIC CAPTURE",
                    (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_MUTED, 1)
        # "Face" in accent colour, rest white
        cv2.putText(frame, "Face Registration",
                    (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, C_WHITE, 2)
        cv2.putText(frame, "Face",
                    (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, C_ACCENT, 2)
 
        # Clock + pulsing dot
        clk = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, clk, (fw-148, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_MUTED, 1)
        dot_col = C_GREEN if int(tick * 2) % 2 == 0 else (0, 140, 70)
        cv2.circle(frame, (fw-162, 23), 5, dot_col, -1)
 
        # ── STOP BUTTON ───────────────────────────────────────────────────────
        bw2, bh2 = 112, 34
        bx, by   = fw - bw2 - 14, 84
        frosted(frame, bx, by, bx+bw2, by+bh2, C_DARK_RED, 0.93)
        cv2.rectangle(frame, (bx, by), (bx+bw2, by+bh2), (80, 80, 220), 1)
        cv2.rectangle(frame, (bx+10, by+9), (bx+22, by+25), (80,80,220), -1)  # stop icon ■
        cv2.putText(frame, "STOP", (bx+30, by+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 2)
 
        # Sample counter under STOP
        cv2.putText(frame, f"{sample_count}/{samples_needed}",
                    (bx+12, by+bh2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_YELLOW, 2)
        cv2.putText(frame, "SAMPLES",
                    (bx+12, by+bh2+36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_MUTED, 1)
 
        # ── SEGMENTED PROGRESS BAR (below header) ────────────────────────────
        seg_count  = samples_needed
        bar_left   = 20
        bar_right  = bx - 10
        bar_y_top  = 82
        bar_w_full = bar_right - bar_left
        seg_gap    = 2
        seg_w      = max(2, (bar_w_full - (seg_count-1)*seg_gap) // seg_count)
 
        for i in range(seg_count):
            sx     = bar_left + i * (seg_w + seg_gap)
            filled = i < sample_count
            col    = C_GREEN if filled else (35, 45, 65)
            cv2.rectangle(frame, (sx, bar_y_top),
                          (sx+seg_w, bar_y_top+7), col, -1)
 
        pct = f"{int(progress*100)}%"
        cv2.putText(frame, pct, (bar_left, bar_y_top+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    C_GREEN if progress > 0 else C_MUTED, 1)
 
        # ── FACE QUALITY BAR (left side, vertical) ────────────────────────────
        q_x, q_y, q_h = 14, fh//2 - 80, 160
        q_fill = int(q_h * quality_score / 100)
        q_col  = C_GREEN if quality_score > 70 else (C_YELLOW if quality_score > 40 else C_RED)
        frosted(frame, q_x, q_y, q_x+22, q_y+q_h, C_PANEL, 0.8)
        cv2.rectangle(frame, (q_x+4, q_y + (q_h - q_fill)),
                      (q_x+18, q_y+q_h), q_col, -1)
        cv2.rectangle(frame, (q_x+4, q_y), (q_x+18, q_y+q_h), C_MUTED, 1)
        cv2.putText(frame, "Q", (q_x+6, q_y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_MUTED, 1)
        cv2.putText(frame, str(quality_score),
                    (q_x+2, q_y+q_h+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, q_col, 1)
 
        # ── ANIMATED GUIDE RING ───────────────────────────────────────────────
        cx, cy = fw // 2, fh // 2
        R_OUTER = 175
        R_MID   = 164
        R_INNER = 150
 
        # Spinning dashed outer ring — colour depends on centred state
        ring_accent = C_GREEN if face_centered else C_ACCENT
        spin_angle  = (tick * 45) % 360
        for seg_i in range(12):
            a_start = int(spin_angle + seg_i * 30)
            a_end   = a_start + 18
            dim     = tuple(int(c * 0.5) for c in ring_accent)
            cv2.ellipse(frame, (cx, cy), (R_OUTER, R_OUTER),
                        0, a_start, a_end, dim, 1)
 
        # Middle ring — shifts blue → green as progress grows
        g_frac = progress
        mid_col = (
            0,
            int(180 + 50 * g_frac),
            int(255 - 125 * g_frac),
        )
        cv2.circle(frame, (cx, cy), R_MID, mid_col, 2)
 
        # Progress arc sweeps around the middle ring
        if sample_count > 0:
            sweep = int(360 * progress)
            cv2.ellipse(frame, (cx, cy), (R_MID, R_MID),
                        -90, 0, sweep, C_GREEN, 4)
 
        # Inner thin guide ring
        cv2.circle(frame, (cx, cy), R_INNER, (30, 45, 70), 1)
 
        # 4 corner alignment marks (viewfinder style)
        mark_len = 24
        mark_off = 122
        for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1)]:
            mx, my = cx + dx*mark_off, cy + dy*mark_off
            col    = C_GREEN if face_centered else C_ACCENT
            cv2.line(frame, (mx, my), (mx + dx*mark_len, my), col, 2)
            cv2.line(frame, (mx, my), (mx, my + dy*mark_len), col, 2)
 
        # Centre crosshair
        cv2.circle(frame, (cx, cy), 4, C_GREEN, -1)
        cv2.circle(frame, (cx, cy), 9, C_GREEN,  1)
 
        # Status hint below circle
        hint     = "Hold still — Capturing..." if face_centered else "Centre your face in the ring"
        hint_col = C_GREEN if face_centered else C_YELLOW
        hint_x   = cx - len(hint) * 6
        frosted(frame, hint_x-8, cy+R_OUTER+8, hint_x+len(hint)*12+8, cy+R_OUTER+32, C_BG, 0.7)
        cv2.putText(frame, hint, (hint_x, cy+R_OUTER+26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, hint_col, 1)
 
        # ── THUMBNAIL STRIP (right side) ──────────────────────────────────────
        # Shows up to 5 of the most recent captured face samples
        if thumbnails:
            strip_x = fw - 72
            strip_y = by + bh2 + 55
            cv2.putText(frame, "RECENT", (strip_x-2, strip_y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_MUTED, 1)
            for i, thumb in enumerate(thumbnails[-5:]):
                ty = strip_y + i * 58
                if ty + 54 > fh - 105:
                    break
                # Convert grayscale thumb → colour and blit
                thumb_bgr = cv2.cvtColor(
                    cv2.resize(thumb, (50, 50)), cv2.COLOR_GRAY2BGR)
                frame[ty:ty+50, strip_x:strip_x+50] = thumb_bgr
                cv2.rectangle(frame, (strip_x, ty), (strip_x+50, ty+50), C_ACCENT, 1)
 
        # ── PERSON INFO CARD (bottom) ─────────────────────────────────────────
        card_h = 95
        card_y = fh - card_h - 10
        frosted(frame, 10, card_y, fw-10, card_y+card_h, C_PANEL, 0.90)
        cv2.rectangle(frame, (10, card_y), (fw-10, card_y+1), C_ACCENT, -1)  # top border
        cv2.rectangle(frame, (10, card_y), (14, card_y+card_h), C_ACCENT, -1)  # left bar
 
        # Name
        cv2.putText(frame, "NAME", (28, card_y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_MUTED, 1)
        cv2.putText(frame, name, (28, card_y+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, C_WHITE, 2)
 
        # Vertical divider 1
        d1 = fw // 3
        cv2.rectangle(frame, (d1, card_y+10), (d1+1, card_y+card_h-10), (40,55,80), -1)
 
        # Badge ID
        cv2.putText(frame, "BADGE ID", (d1+12, card_y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_MUTED, 1)
        cv2.putText(frame, str(person_id), (d1+12, card_y+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, C_YELLOW, 2)
 
        # Vertical divider 2
        d2 = (fw * 2) // 3
        cv2.rectangle(frame, (d2, card_y+10), (d2+1, card_y+card_h-10), (40,55,80), -1)
 
        # Role
        cv2.putText(frame, "ROLE", (d2+12, card_y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_MUTED, 1)
        cv2.putText(frame, str(role) if role else "—", (d2+12, card_y+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_ORANGE, 2)
 
        # Step tracker pills
        steps      = ["ALIGN", "CAPTURE", "DONE"]
        step_done  = 0 if sample_count == 0 else (1 if sample_count < samples_needed else 2)
        pill_start = 28
        for si, label in enumerate(steps):
            pill_x = pill_start + si * 72
            done   = si < step_done
            active = si == step_done
            fc     = C_GREEN if done else (C_ACCENT if active else (35,45,65))
            tc     = (10,10,10) if done else (C_WHITE if active else C_MUTED)
            cv2.rectangle(frame, (pill_x, card_y+58),
                          (pill_x+64, card_y+78), fc, -1)
            cv2.putText(frame, label, (pill_x+8, card_y+72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, tc, 1)
 
        return bx, by, bw2, bh2   # stop button coords
 
    # ─────────────────────────────────────────────────────────────────────────
    # RECOGNITION HUD  (modernised v2)
    # ─────────────────────────────────────────────────────────────────────────
 
    def draw_hud(self, frame, faces_info, fps, alert_unknown):
        """
        Full recognition HUD with:
          • Top header: title, clock, FPS / face count / DB size
          • Right panel: session stats + confidence threshold
          • Left panel: recent recognition event log (last 5)
          • Bottom bar: status + keybind hints
          • Per-face: corner boxes, name cards, confidence badge
          • Red flashing border when an unknown face is detected
        """
        fh, fw = frame.shape[:2]
        tick   = time.time()
        uptime = int(tick - self.session_start)
        known_pct = (
            int(100 * self.session_known_faces / self.session_total_faces)
            if self.session_total_faces > 0 else 0
        )
 
        # ── UNKNOWN ALERT — full-frame red flashing border ────────────────────
        if alert_unknown and int(tick * 4) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (fw-1, fh-1), C_RED, 6)
 
        # ── TOP HEADER ────────────────────────────────────────────────────────
        frosted(frame, 0, 0, fw, 75, C_BG, 0.88)
        cv2.rectangle(frame, (0, 73), (fw, 75), C_ACCENT, -1)
 
        cv2.putText(frame, "FACE RECOGNITION SYSTEM",
                    (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_MUTED, 1)
        cv2.putText(frame, "Face Recognition",
                    (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, C_WHITE, 2)
        cv2.putText(frame, "Face",
                    (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, C_ACCENT, 2)
 
        # Clock + pulsing dot
        clk = datetime.now().strftime("%H:%M:%S")
        dot_col = C_GREEN if int(tick*2)%2==0 else (0,140,70)
        cv2.circle(frame, (fw-165, 23), 6, dot_col, -1)
        cv2.putText(frame, "LIVE", (fw-150, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, dot_col, 2)
        cv2.putText(frame, clk, (fw-150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_MUTED, 1)
 
        # ── RIGHT PANEL — session stats ────────────────────────────────────────
        rp_x, rp_y = fw - 160, 85
        rp_w, rp_h = 150, 145
        frosted(frame, rp_x, rp_y, rp_x+rp_w, rp_y+rp_h, C_PANEL, 0.85)
        cv2.rectangle(frame, (rp_x, rp_y), (rp_x+rp_w, rp_y+rp_h), C_MUTED, 1)
        cv2.putText(frame, "SESSION",
                    (rp_x+8, rp_y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_MUTED, 1)
 
        stats = [
            ("FPS",       f"{fps:.1f}"),
            ("FACES",     str(len(faces_info))),
            ("DB SIZE",   str(len(self.people_db))),
            ("KNOWN %",   f"{known_pct}%"),
            ("UPTIME",    f"{uptime//60:02d}:{uptime%60:02d}"),
            ("THRESHOLD", str(self.confidence_threshold)),
        ]
        for i, (label, val) in enumerate(stats):
            sy = rp_y + 28 + i * 19
            cv2.putText(frame, label, (rp_x+8, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_MUTED, 1)
            cv2.putText(frame, val, (rp_x+90, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GREEN, 1)
 
        # ── LEFT PANEL — event log ─────────────────────────────────────────────
        if self.recent_events:
            lp_x, lp_y = 10, 85
            lp_h       = 20 + len(self.recent_events) * 20
            frosted(frame, lp_x, lp_y, lp_x+230, lp_y+lp_h, C_PANEL, 0.82)
            cv2.putText(frame, "RECENT EVENTS",
                        (lp_x+8, lp_y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_MUTED, 1)
            for i, ev in enumerate(self.recent_events[-5:]):
                cv2.putText(frame, ev, (lp_x+8, lp_y+28+i*18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1)
 
        # ── BOTTOM STATUS BAR ─────────────────────────────────────────────────
        frosted(frame, 0, fh-58, fw, fh, C_BG, 0.88)
        cv2.rectangle(frame, (0, fh-58), (fw, fh-56), C_ACCENT, -1)
 
        status = "SCANNING — NO FACES DETECTED" if len(faces_info) == 0 else \
                 f"ACTIVE — {len(faces_info)} FACE(S) IN FRAME"
        cv2.putText(frame, status, (20, fh-32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_GREEN, 1)
 
        # Key-hint strip
        hints = "S=screenshot  F=freeze  M=mirror  Z/X=zoom  +/-=threshold  Q=quit"
        cv2.putText(frame, hints, (20, fh-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_MUTED, 1)
 
        # Freeze indicator
        if self.freeze_frame:
            frosted(frame, fw//2-70, fh//2-24, fw//2+70, fh//2+24, C_BG, 0.8)
            cv2.putText(frame, "FROZEN", (fw//2-52, fh//2+8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_YELLOW, 3)
 
        # Zoom indicator
        if self.zoom_factor > 1.0:
            cv2.putText(frame, f"ZOOM {self.zoom_factor:.1f}x",
                        (fw-140, fh-65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_YELLOW, 2)
 
        # ── PER-FACE OVERLAYS ─────────────────────────────────────────────────
        for fi in faces_info:
            x, y, w, h  = fi['bbox']
            known        = fi['known']
            confidence   = fi['confidence']
            name         = fi.get('name', 'Unknown')
            pid          = fi.get('person_id', 'N/A')
            role         = fi.get('role', '')
 
            box_col = C_GREEN if known else C_RED
 
            # Corner box with glow
            self.corner_box_glow(frame, x, y, w, h, box_col)
 
            # Confidence badge (top-right corner of box)
            conf_txt = f"{confidence:.0f}%"
            bx2      = x + w - 64
            by2      = y - 28
            if by2 > 5:
                frosted(frame, bx2, by2, bx2+60, by2+22, box_col, 0.9)
                cv2.putText(frame, conf_txt, (bx2+6, by2+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 2)
 
            # Name card below face
            if known:
                cx2    = max(0, x - 20)
                cy2    = y + h + 14
                cw2    = w + 40
                ch2    = 88 if role else 68
 
                if cy2 + ch2 < fh and cx2 + cw2 <= fw:
                    frosted(frame, cx2, cy2, cx2+cw2, cy2+ch2, C_PANEL, 0.90)
                    cv2.rectangle(frame, (cx2, cy2), (cx2+cw2, cy2+ch2), box_col, 2)
 
                    # Name
                    cv2.putText(frame, name, (cx2+12, cy2+28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_WHITE, 2)
                    # ID
                    cv2.putText(frame, f"ID: {pid}", (cx2+12, cy2+50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_MUTED, 1)
                    # Role
                    if role:
                        cv2.putText(frame, str(role), (cx2+12, cy2+70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_ORANGE, 1)
 
                    # "RECOGNISED" badge
                    cv2.putText(frame, "RECOGNISED",
                                (cx2+cw2-118, cy2+22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GREEN, 1)
            else:
                # "UNKNOWN" label above box
                frosted(frame, x, y-34, x+w, y-6, C_DARK_RED, 0.85)
                cv2.putText(frame, "UNKNOWN", (x+8, y-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,80,240), 2)
 
    # ─────────────────────────────────────────────────────────────────────────
    # TTS
    # ─────────────────────────────────────────────────────────────────────────
 
    def speak(self, message):
        """Speak a message in a background thread so camera never pauses."""
        def _speak():
            try:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
        t = threading.Thread(target=_speak)
        t.daemon = True
        t.start()
 
    # ─────────────────────────────────────────────────────────────────────────
    # MODE 1 — REGISTER  (v2 — adds role, quality meter, thumbnail strip)
    # ─────────────────────────────────────────────────────────────────────────
 
    def register_person(self):
        """
        Register a new person with:
          • Name, badge ID, and optional role / department
          • 40 auto-captured face samples (100×100 grayscale)
          • Live quality score shown on screen
          • Thumbnail strip of last 5 captures
          • STOP button or Q key to cancel
        """
        print("\n╔══════════════════════════════╗")
        print("║     FACE REGISTRATION v2     ║")
        print("╚══════════════════════════════╝\n")
 
        print("Enter full name:         ", end='', flush=True)
        name = sys.stdin.readline().strip()
        if not name:
            print("Name cannot be empty!"); return
 
        print("Enter badge / ID number: ", end='', flush=True)
        person_id = sys.stdin.readline().strip()
        if not person_id:
            print("ID cannot be empty!"); return
 
        print("Enter role / department  (or press Enter to skip): ", end='', flush=True)
        role = sys.stdin.readline().strip()
 
        # Assign internal ID and create folder
        internal_id = self.get_next_person_id()
        person_dir  = os.path.join(self.faces_dir, str(internal_id))
        Path(person_dir).mkdir(exist_ok=True)
 
        # Save to database
        self.people_db[str(internal_id)] = {
            "name": name, "id": person_id, "role": role,
            "registered": datetime.now().isoformat()
        }
        self.save_database()
 
        print(f"\nRegistering: {name}  |  Badge: {person_id}  |  Role: {role or '—'}")
        print("Look at the camera. Centre your face in the ring.")
        print("Press Q or click STOP to cancel.\n")
 
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera!"); return
 
        samples_needed  = 40
        sample_count    = 0
        last_capture    = 0
        capture_interval= 0.12   # one photo every 120 ms
        stop_clicked    = False
        quality_score   = 0
        thumbnails      = []      # list of grayscale numpy arrays
        stop_btn        = [0, 0, 0, 0]
 
        def on_mouse(event, mx, my, flags, param):
            nonlocal stop_clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                bx, by, bw2, bh2 = stop_btn
                if bx <= mx <= bx+bw2 and by <= my <= by+bh2:
                    stop_clicked = True
 
        cv2.namedWindow("Registration")
        cv2.setMouseCallback("Registration", on_mouse)
 
        frozen_frame = None   # used when face leaves frame briefly
 
        while sample_count < samples_needed and not stop_clicked:
            ret, frame = cap.read()
            if not ret:
                continue
 
            frame = cv2.flip(frame, 1)   # mirror for registration (natural selfie view)
            fh2, fw2 = frame.shape[:2]
 
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enh   = self.enhance_frame(gray)   # CLAHE contrast boost
            faces = self.detect_faces_multi_scale(enh)
 
            face_centered = False
            best_face     = None
 
            for (fx, fy, fw3, fh3) in faces:
                if fw3 < 100 or fh3 < 100:
                    continue
                face_cx = fx + fw3 / 2
                face_cy = fy + fh3 / 2
                centered = (
                    abs(face_cx - fw2/2) < fw2 * 0.28 and
                    abs(face_cy - fh2/2) < fh2 * 0.28
                )
                if centered:
                    face_centered = True
                    best_face     = (fx, fy, fw3, fh3)
                    face_crop     = enh[fy:fy+fh3, fx:fx+fw3]
                    quality_score = self.face_quality_score(face_crop)
 
            # Auto-capture when centred and quality is acceptable (improved threshold from 30 to 45)
            if face_centered and best_face and quality_score > 45:
                fx, fy, fw3, fh3 = best_face
                now = time.time()
                if now - last_capture > capture_interval:
                    face_100 = cv2.resize(enh[fy:fy+fh3, fx:fx+fw3], (100,100))
                    cv2.imwrite(
                        os.path.join(person_dir, f"sample_{sample_count}.jpg"),
                        face_100
                    )
                    thumbnails.append(face_100)
                    sample_count += 1
                    last_capture  = now
 
                # Draw corner box
                box_col = C_GREEN if face_centered else C_ACCENT
                self.corner_box(frame, fx, fy, fw3, fh3, box_col, 3)
 
            # Draw full HUD
            coords = self.draw_registration_ui(
                frame, sample_count, samples_needed,
                name, person_id, role,
                quality_score, face_centered, thumbnails
            )
            stop_btn[:] = list(coords)
 
            cv2.imshow("Registration", frame)
            cv2.setWindowProperty("Registration", cv2.WND_PROP_TOPMOST, 1)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_clicked = True
 
        cap.release()
        cv2.destroyAllWindows()
 
        if stop_clicked:
            print(f"\nCancelled. Rolling back...")
            del self.people_db[str(internal_id)]
            self.save_database()
            if os.path.exists(person_dir):
                shutil.rmtree(person_dir)
        else:
            print(f"\n✓ Registered {name} with {sample_count} samples.")
            print("Next: python face_camera.py --retrain")
 
    # ─────────────────────────────────────────────────────────────────────────
    # MODE 2 — TRAIN
    # ─────────────────────────────────────────────────────────────────────────
 
    def train_model(self):
        """
        Load all saved face samples and train the LBPH model.
        Saves to face_model.xml. Run after every register or delete.
        """
        print("\n=== Training Model ===")
        faces, labels = [], []
 
        for pid in os.listdir(self.faces_dir):
            person_dir = os.path.join(self.faces_dir, pid)
            if not os.path.isdir(person_dir):
                continue
            for fname in os.listdir(person_dir):
                if not fname.endswith('.jpg'):
                    continue
                img = cv2.imread(os.path.join(person_dir, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(int(pid))
 
        if not faces:
            print("No samples found — register someone first."); return False
 
        n_people = len(set(labels))
        print(f"Training on {len(faces)} photos across {n_people} people...")
        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save(self.model_file)
        print(f"✓ Model saved → {self.model_file}")
        return True
 
    # ─────────────────────────────────────────────────────────────────────────
    # MODE 3 — LIVE RECOGNITION  (v2 — many new features)
    # ─────────────────────────────────────────────────────────────────────────
 
    def run_recognition(self):
        """
        Live recognition loop with:
          • CLAHE contrast enhancement
          • Mirror mode  (M key)
          • Freeze frame (F key)
          • Screenshot   (S key → saved to screenshots/)
          • Zoom         (Z = zoom in, X = zoom out)
          • Confidence threshold adjust (+ / -)
          • Unknown-face flashing red border alert
          • On-screen event log (last 5 recognitions)
          • Session stats panel
          • TTS announcement with 5-second cooldown
          • All events written to recognition_log.json
        """
        print("\n╔══════════════════════════════╗")
        print("║   LIVE RECOGNITION v2        ║")
        print("╠══════════════════════════════╣")
        print("║  S  screenshot               ║")
        print("║  F  freeze / unfreeze        ║")
        print("║  M  mirror toggle            ║")
        print("║  Z  zoom in                  ║")
        print("║  X  zoom out                 ║")
        print("║  +  loosen threshold         ║")
        print("║  -  tighten threshold        ║")
        print("║  Q  quit                     ║")
        print("╚══════════════════════════════╝\n")
 
        if not os.path.exists(self.model_file):
            print("Model not found! Run --retrain first."); return
 
        self.recognizer.read(self.model_file)
        self.session_start = time.time()
 
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera!"); return
 
        prev_time    = time.time()
        frozen_frame = None
 
        while True:
            if not self.freeze_frame:
                ret, frame = cap.read()
                if not ret:
                    continue
                frozen_frame = frame.copy()
            else:
                frame = frozen_frame.copy() if frozen_frame is not None else np.zeros((480,640,3),np.uint8)
 
            # Mirror mode
            if self.mirror_mode:
                frame = cv2.flip(frame, 1)
 
            # Zoom
            frame = self.apply_zoom(frame)
 
            # FPS
            now       = time.time()
            fps       = 1 / (now - prev_time) if now > prev_time else 0
            prev_time = now
 
            # Face detection on CLAHE-enhanced grayscale
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enh   = self.enhance_frame(gray)
            faces = self.detect_faces_multi_scale(enh)
 
            faces_info    = []
            alert_unknown = False
 
            for (x, y, w, h) in faces:
                if w < 80 or h < 80:
                    continue
 
                self.session_total_faces += 1
 
                face_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                face_100  = cv2.resize(face_gray, (100, 100))
                label, confidence = self.recognizer.predict(face_100)
 
                known = confidence < self.confidence_threshold
 
                info = {
                    'bbox':       (x, y, w, h),
                    'known':      known,
                    'confidence': confidence,
                    'name':       'Unknown',
                    'person_id':  'N/A',
                    'role':       '',
                }
 
                if known and str(label) in self.people_db:
                    person = self.people_db[str(label)]
                    info['name']      = person['name']
                    info['person_id'] = person['id']
                    info['role']      = person.get('role', '')
                    self.session_known_faces += 1
 
                    # TTS + log event
                    if (self.last_recognized_name != person['name'] or
                            now - self.last_recognition_time > self.recognition_cooldown):
                        self.speak(f"Welcome, {person['name']}")
                        self.last_recognized_name  = person['name']
                        self.last_recognition_time = now
 
                        # Log to file
                        entry = {
                            "time":   datetime.now().isoformat(),
                            "name":   person['name'],
                            "badge":  person['id'],
                            "conf":   round(confidence, 1),
                        }
                        self.rec_log.append(entry)
                        self.save_log()
 
                        # On-screen event log (last 5)
                        ts = datetime.now().strftime("%H:%M:%S")
                        self.recent_events.append(f"{ts}  {person['name']}")
                        self.recent_events = self.recent_events[-5:]
 
                else:
                    alert_unknown = True
 
                faces_info.append(info)
 
            # Draw full HUD
            self.draw_hud(frame, faces_info, fps, alert_unknown)
 
            cv2.imshow("Face Recognition System", frame)
            key = cv2.waitKey(1) & 0xFF
 
            # ── KEY BINDINGS ──────────────────────────────────────────────────
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Screenshot
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"screenshots/snapshot_{ts}.png"
                cv2.imwrite(path, frame)
                print(f"Screenshot saved → {path}")
            elif key == ord('f'):
                # Freeze / unfreeze
                self.freeze_frame = not self.freeze_frame
                print(f"Freeze: {'ON' if self.freeze_frame else 'OFF'}")
            elif key == ord('m'):
                # Mirror
                self.mirror_mode = not self.mirror_mode
                print(f"Mirror: {'ON' if self.mirror_mode else 'OFF'}")
            elif key == ord('z'):
                # Zoom in (max 4×)
                self.zoom_factor = min(4.0, round(self.zoom_factor + 0.25, 2))
                print(f"Zoom: {self.zoom_factor}×")
            elif key == ord('x'):
                # Zoom out (min 1×)
                self.zoom_factor = max(1.0, round(self.zoom_factor - 0.25, 2))
                print(f"Zoom: {self.zoom_factor}×")
            elif key in (ord('+'), ord('=')):
                # Loosen threshold — recognises more (may get false matches)
                self.confidence_threshold = min(120, self.confidence_threshold + 5)
                print(f"Threshold: {self.confidence_threshold}")
            elif key == ord('-'):
                # Tighten threshold — stricter matching
                self.confidence_threshold = max(30, self.confidence_threshold - 5)
                print(f"Threshold: {self.confidence_threshold}")
 
        cap.release()
        cv2.destroyAllWindows()
        print("\nSession ended.")
 
    # ─────────────────────────────────────────────────────────────────────────
    # MODE 4 — LIST
    # ─────────────────────────────────────────────────────────────────────────
 
    def list_people(self):
        """Print every registered person."""
        print("\n=== Registered People ===")
        if not self.people_db:
            print("Nobody registered yet."); return
        fmt = "  {:>4}  {:<22}  {:<12}  {:<16}  {}"
        print(fmt.format("ID", "Name", "Badge", "Role", "Registered"))
        print("  " + "─"*72)
        for pid, info in self.people_db.items():
            print(fmt.format(
                pid,
                info.get('name', ''),
                info.get('id', ''),
                info.get('role', '—'),
                info.get('registered', '—')[:16],
            ))
 
    # ─────────────────────────────────────────────────────────────────────────
    # MODE 5 — DELETE
    # ─────────────────────────────────────────────────────────────────────────
 
    def delete_person(self, internal_id):
        """
        Remove a person by their internal ID (get it from --list).
        Deletes database entry + all face photos.
        Then run --retrain to update the model.
        """
        pid = str(internal_id)
        if pid not in self.people_db:
            print(f"No person with ID {internal_id}."); return False
 
        p = self.people_db[pid]
        print(f"\nDeleting: {p['name']} (Badge: {p['id']})")
        del self.people_db[pid]
        self.save_database()
 
        d = os.path.join(self.faces_dir, pid)
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Removed face samples from {d}")
 
        print("Done. Run --retrain to update the model.")
        return True
 
    # ─────────────────────────────────────────────────────────────────────────
    # MODE 6 — EXPORT LOG
    # ─────────────────────────────────────────────────────────────────────────
 
    def export_log(self):
        """
        Export recognition_log.json → recognition_log.csv
        Useful for attendance tracking / reporting.
        """
        if not self.rec_log:
            print("No events logged yet."); return
 
        out = "recognition_log.csv"
        with open(out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["time","name","badge","conf"])
            writer.writeheader()
            writer.writerows(self.rec_log)
 
        print(f"✓ Exported {len(self.rec_log)} events → {out}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition Camera System v2",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python face_camera.py                → live recognition\n"
            "  python face_camera.py --register     → add a new person\n"
            "  python face_camera.py --retrain      → train / retrain model\n"
            "  python face_camera.py --list         → show all people\n"
            "  python face_camera.py --delete 0     → remove internal ID 0\n"
            "  python face_camera.py --export       → export log to CSV\n"
        )
    )
    parser.add_argument('--register', action='store_true', help='Register a new person')
    parser.add_argument('--retrain',  action='store_true', help='Train / retrain model')
    parser.add_argument('--list',     action='store_true', help='List all people')
    parser.add_argument('--delete',   type=int, metavar='ID', help='Delete by internal ID')
    parser.add_argument('--export',   action='store_true', help='Export log to CSV')
 
    args   = parser.parse_args()
    system = FaceRecognitionSystem()
 
    if   args.register:           system.register_person()
    elif args.retrain:            system.train_model()
    elif args.list:               system.list_people()
    elif args.delete is not None: system.delete_person(args.delete)
    elif args.export:             system.export_log()
    else:                         system.run_recognition()
 
 
if __name__ == "__main__":
    main()