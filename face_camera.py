#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║           FACE RECOGNITION CAMERA SYSTEM  v3.1                  ║
║                                                                  ║
║  FIXED in v3.1                                                   ║
║  ────────────────────────────────────────────────────────────    ║
║  _init_tts_engine method added                                   ║
║  draw_loading_screen method added                                ║
║  capture_face_snapshot method added                              ║
║  clahe / face_cascade attributes initialised in __init__         ║
║  detect_min/max_size attributes added                            ║
║  Smooth gradient background (no more black bars)                 ║
║                                                                  ║
║  INSTALL                                                         ║
║    pip install opencv-contrib-python pyttsx3 numpy               ║
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
 
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("WARNING: pyttsx3 not installed. TTS disabled.")
 
# ── Try importing optional modules, fall back gracefully ──────────
try:
    from camera import Camera as _ExtCamera
    USE_EXT_CAMERA = True
except ImportError:
    USE_EXT_CAMERA = False
 
try:
    from face_detector import FaceDetector as _ExtFaceDetector
    USE_EXT_FACE_DETECTOR = True
except ImportError:
    USE_EXT_FACE_DETECTOR = False
 
try:
    from hud import HUD as _ExtHUD
    USE_EXT_HUD = True
except ImportError:
    USE_EXT_HUD = False
 
try:
    import config as _config
    DB_FILE              = _config.DB_FILE
    MODEL_FILE           = _config.MODEL_FILE
    FACES_DIR            = _config.FACES_DIR
    RECOGNITION_THRESHOLD= _config.RECOGNITION_THRESHOLD
except ImportError:
    DB_FILE               = "people_db.json"
    MODEL_FILE            = "face_model.xml"
    FACES_DIR             = "face_samples"
    RECOGNITION_THRESHOLD = 75
 
 
# ─────────────────────────────────────────────────────────────────
# MINIMAL CAMERA WRAPPER  (used when camera.py is absent)
# ─────────────────────────────────────────────────────────────────
class _SimpleCamera:
    def __init__(self, index=0):
        self.index = index
        self.cap   = None
 
    def open(self):
        self.cap = cv2.VideoCapture(self.index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS,          30)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)
        return self.cap.isOpened()
 
    def read(self):
        return self.cap.read() if self.cap else (False, None)
 
    def release(self):
        if self.cap:
            self.cap.release()
 
 
Camera = _ExtCamera if USE_EXT_CAMERA else _SimpleCamera
 
 
# ─────────────────────────────────────────────────────────────────
# MINIMAL FACE-DETECTOR WRAPPER  (used when face_detector.py is absent)
# ─────────────────────────────────────────────────────────────────
class _SimpleFaceDetector:
    """Haar-cascade face detector with CLAHE enhancement."""
 
    CASCADE_PATHS = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
    ]
 
    def __init__(self):
        self.cascade = None
        for p in self.CASCADE_PATHS:
            if os.path.exists(p):
                self.cascade = cv2.CascadeClassifier(p)
                break
        if self.cascade is None:
            raise RuntimeError("No Haar cascade XML found. Reinstall opencv-contrib-python.")
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
 
    def enhance_frame(self, gray):
        clahe_out = self.clahe.apply(gray)
        hist_eq   = cv2.equalizeHist(gray)
        return cv2.addWeighted(clahe_out, 0.7, hist_eq, 0.3, 0)
 
    def detect_faces_multi_scale(self, gray):
        all_faces = []
        for scale in [1.05, 1.1, 1.15]:
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=scale, minNeighbors=4,
                minSize=(80, 80), maxSize=(500, 500)
            )
            if len(faces):
                all_faces.extend(faces.tolist())
        if len(all_faces) > 1:
            all_faces = self._nms(all_faces, 0.3)
        return all_faces
 
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enh  = self.enhance_frame(gray)
        return self.detect_faces_multi_scale(enh)
 
    def smooth_face_skin(self, gray, smooth_factor=0.5):
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        return cv2.addWeighted(blur, smooth_factor, gray, 1 - smooth_factor, 0)
 
    def enhance_face_clarity(self, gray, clarity_factor=1.0):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharp  = cv2.filter2D(gray, -1, kernel)
        return cv2.addWeighted(sharp, clarity_factor, gray, 1 - clarity_factor, 0)
 
    def _nms(self, faces, thresh):
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        keep  = []
        while faces:
            cur = faces.pop(0)
            keep.append(cur)
            faces = [f for f in faces if self._iou(cur, f) < thresh]
        return keep
 
    @staticmethod
    def _iou(b1, b2):
        x1,y1,w1,h1 = b1;  x2,y2,w2,h2 = b2
        ix1,iy1 = max(x1,x2), max(y1,y2)
        ix2,iy2 = min(x1+w1,x2+w2), min(y1+h1,y2+h2)
        if ix2<=ix1 or iy2<=iy1:
            return 0.0
        inter = (ix2-ix1)*(iy2-iy1)
        union = w1*h1 + w2*h2 - inter
        return inter/union if union else 0.0
 
 
FaceDetector = _ExtFaceDetector if USE_EXT_FACE_DETECTOR else _SimpleFaceDetector
 
 
# ─────────────────────────────────────────────────────────────────
# COLOUR THEME  (BGR)
# ─────────────────────────────────────────────────────────────────
C_BG        = (14,  20,  36)
C_PANEL     = (20,  30,  55)
C_ACCENT    = (0,  180, 255)
C_GREEN     = (0,  230, 130)
C_RED       = (60,  60, 220)
C_YELLOW    = (30, 210, 255)
C_WHITE     = (255, 255, 255)
C_MUTED     = (140, 155, 175)
C_DARK_RED  = (30,  25, 140)
C_ORANGE    = (30, 160, 255)
 
 
# ─────────────────────────────────────────────────────────────────
# BACKGROUND GRADIENT  (smooth deep-navy → midnight-blue)
# ─────────────────────────────────────────────────────────────────
def draw_gradient_background(frame):
    """
    Replace the frame background with a smooth vertical gradient
    (deep navy at top → slightly lighter blue-navy at bottom).
    This eliminates harsh black borders.
    """
    fh, fw = frame.shape[:2]
    top    = np.array([14,  20,  36], dtype=np.float32)   # BGR dark navy
    bottom = np.array([22,  35,  62], dtype=np.float32)   # BGR slightly lighter
 
    for y in range(fh):
        t   = y / max(fh - 1, 1)
        col = (top * (1 - t) + bottom * t).astype(np.uint8)
        frame[y, :] = col
 
 
def draw_gradient_panel(frame, x1, y1, x2, y2,
                        col_top=(20,30,55), col_bot=(14,22,44), alpha=0.88):
    """
    Frosted gradient panel: blends top→bottom colour inside the region.
    Replaces the plain frosted() helper for panel backgrounds.
    """
    y1c = max(0, y1); y2c = min(frame.shape[0], y2)
    x1c = max(0, x1); x2c = min(frame.shape[1], x2)
    if y2c <= y1c or x2c <= x1c:
        return
    h = y2c - y1c
    ct = np.array(col_top, dtype=np.float32)
    cb = np.array(col_bot, dtype=np.float32)
    for dy in range(h):
        t   = dy / max(h - 1, 1)
        col = (ct * (1 - t) + cb * t).astype(np.uint8)
        row = frame[y1c+dy, x1c:x2c]
        blended = cv2.addWeighted(
            np.full(row.shape, col, dtype=np.uint8), alpha,
            row, 1.0 - alpha, 0
        )
        frame[y1c+dy, x1c:x2c] = blended
 
 
# Keep the old frosted() for backward-compat inside registration UI
def frosted(frame, x1, y1, x2, y2, color=C_BG, alpha=0.78):
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    if y2 <= y1 or x2 <= x1:
        return
    roi     = frame[y1:y2, x1:x2]
    overlay = np.full(roi.shape, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi
 
 
# ─────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────
class FaceRecognitionSystem:
 
    def __init__(self):
        # ── Camera ───────────────────────────────────────────────
        self.camera = Camera()
 
        # ── Face detector ─────────────────────────────────────────
        self.face_detector = FaceDetector()
 
        # ── CLAHE (also used directly in some methods) ────────────
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
 
        # ── Cascade (direct access for registration loop) ─────────
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
 
        # ── Detection size limits ─────────────────────────────────
        self.detect_min_size = 80
        self.detect_max_size = 500
 
        # ── Paths ─────────────────────────────────────────────────
        self.db_file    = DB_FILE
        self.model_file = MODEL_FILE
        self.faces_dir  = FACES_DIR
        self.log_file   = "recognition_log.json"
        Path(self.faces_dir).mkdir(exist_ok=True)
        Path("snapshots").mkdir(exist_ok=True)
        Path("screenshots").mkdir(exist_ok=True)
 
        # ── Data ──────────────────────────────────────────────────
        self.people_db = self._load_json(self.db_file,  {})
        self.rec_log   = self._load_json(self.log_file, [])
 
        # ── LBPH recogniser ───────────────────────────────────────
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
 
        # ── TTS ───────────────────────────────────────────────────
        self.tts_engine = self._init_tts_engine()
        if self.tts_engine:
            self.tts_engine.setProperty('rate', 150)
        else:
            print("WARNING: TTS initialization failed. TTS functionality disabled.")
 
        self.last_recognized_name  = None
        self.last_recognition_time = 0
        self.recognition_cooldown  = 5
 
        # ── Runtime state ─────────────────────────────────────────
        self.confidence_threshold = RECOGNITION_THRESHOLD
        self.mirror_mode          = False
        self.freeze_frame         = False
        self.zoom_factor          = 1.0
        self.session_start        = time.time()
        self.session_total_faces  = 0
        self.session_known_faces  = 0
        self.recent_events        = []
 
    # ─────────────────────────────────────────────────────────────
    # TTS INIT  (was missing – root cause of the AttributeError)
    # ─────────────────────────────────────────────────────────────
 
    def _init_tts_engine(self):
        """Initialise pyttsx3 TTS engine; return None on failure."""
        if not PYTTSX3_AVAILABLE:
            return None
        try:
            engine = pyttsx3.init()
            return engine
        except Exception as e:
            print(f"TTS init error: {e}")
            return None
 
    # ─────────────────────────────────────────────────────────────
    # JSON HELPERS
    # ─────────────────────────────────────────────────────────────
 
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
        self._save_json(self.log_file, self.rec_log[-500:])
 
    def get_next_person_id(self):
        if not self.people_db:
            return 0
        return max(int(k) for k in self.people_db.keys()) + 1
 
    # ─────────────────────────────────────────────────────────────
    # IMAGE PROCESSING HELPERS
    # ─────────────────────────────────────────────────────────────
 
    def enhance_frame(self, gray):
        clahe_enhanced = self.clahe.apply(gray)
        hist_eq        = cv2.equalizeHist(gray)
        return cv2.addWeighted(clahe_enhanced, 0.7, hist_eq, 0.3, 0)
 
    def detect_faces_multi_scale(self, gray):
        all_faces = []
        for scale in [1.05, 1.1, 1.15]:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=scale, minNeighbors=4,
                minSize=(self.detect_min_size, self.detect_min_size),
                maxSize=(self.detect_max_size, self.detect_max_size)
            )
            if len(faces):
                all_faces.extend(faces.tolist())
        if len(all_faces) > 1:
            all_faces = self._non_max_suppression(all_faces, 0.3)
        return all_faces
 
    def _non_max_suppression(self, faces, overlap_threshold):
        if not faces:
            return faces
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        keep  = []
        while faces:
            current = faces.pop(0)
            keep.append(current)
            faces = [f for f in faces if self._iou(current, f) < overlap_threshold]
        return keep
 
    def _iou(self, box1, box2):
        x1, y1, w1, h1 = box1;  x2, y2, w2, h2 = box2
        xl = max(x1, x2); yt = max(y1, y2)
        xr = min(x1+w1, x2+w2); yb = min(y1+h1, y2+h2)
        if xr < xl or yb < yt:
            return 0.0
        inter = (xr - xl) * (yb - yt)
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0.0
 
    def apply_zoom(self, frame):
        if self.zoom_factor <= 1.0:
            return frame
        fh, fw = frame.shape[:2]
        new_h  = int(fh / self.zoom_factor)
        new_w  = int(fw / self.zoom_factor)
        y1     = (fh - new_h) // 2
        x1     = (fw - new_w) // 2
        return cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (fw, fh))
 
    def face_quality_score(self, face_gray):
        blur_score    = cv2.Laplacian(face_gray, cv2.CV_64F).var()
        sharpness     = min(100, blur_score / 5)
        brightness    = np.mean(face_gray)
        brightness_ok = 100 - abs(brightness - 120) / 1.2
        contrast      = np.std(face_gray)
        contrast_ok   = min(100, contrast * 2)
        return int(sharpness*0.5 + brightness_ok*0.3 + contrast_ok*0.2)
 
    # ─────────────────────────────────────────────────────────────
    # DRAWING HELPERS
    # ─────────────────────────────────────────────────────────────
 
    def corner_box(self, frame, x, y, w, h, color, thickness=4):
        arm = min(w, h) // 5
        pts = [
            ((x,   y),   (x+arm, y),   (x,   y+arm)),
            ((x+w, y),   (x+w-arm, y), (x+w, y+arm)),
            ((x,   y+h), (x+arm, y+h), (x,   y+h-arm)),
            ((x+w, y+h), (x+w-arm,y+h),(x+w, y+h-arm)),
        ]
        for corner in pts:
            cv2.line(frame, corner[0], corner[1], color, thickness)
            cv2.line(frame, corner[0], corner[2], color, thickness)
 
    def corner_box_glow(self, frame, x, y, w, h, color, thickness=5):
        shadow = tuple(max(0, c - 70) for c in color)
        self.corner_box(frame, x, y, w, h, shadow, thickness + 3)
        self.corner_box(frame, x, y, w, h, color,  thickness)
 
    def text_shadow(self, frame, text, pos, scale, color, thickness=2):
        x, y = pos
        cv2.putText(frame, text, (x+2,y+2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0,0,0), thickness+2)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness)
 
    def quality_bar(self, frame, x, y, width, score, label="QUALITY"):
        bar_h  = 8
        fill_w = int(width * score / 100)
        color  = C_GREEN if score > 70 else (C_YELLOW if score > 40 else C_RED)
        cv2.rectangle(frame, (x, y), (x+width, y+bar_h), (30,40,60), -1)
        cv2.rectangle(frame, (x, y), (x+fill_w, y+bar_h), color, -1)
        cv2.putText(frame, f"{label} {score}%", (x, y+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # ─────────────────────────────────────────────────────────────
    # LOADING SCREEN  (was missing)
    # ─────────────────────────────────────────────────────────────

    def draw_loading_screen(self, frame, elapsed, total_duration):
        """
        Animated loading overlay with smooth gradient background,
        spinning arc, and status text.
        Shows actual camera feed with overlay instead of solid background.
        """
        fh, fw = frame.shape[:2]
        cx, cy = fw // 2, fh // 2
        progress = min(1.0, elapsed / total_duration)

        # Subtle dark overlay on actual frame (keep camera feed visible)
        overlay = np.full(frame.shape, (14, 20, 36), dtype=np.uint8)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Spinning outer ring
        tick       = time.time()
        spin_angle = (tick * 120) % 360
        for seg_i in range(8):
            a_start = int(spin_angle + seg_i * 45)
            a_end   = a_start + 30
            alpha_v = 0.3 + 0.7 * (seg_i / 7)
            col     = tuple(int(c * alpha_v) for c in C_ACCENT)
            cv2.ellipse(frame, (cx, cy), (72, 72), 0, a_start, a_end, col, 3)

        # Progress arc (fills as loading completes)
        if progress > 0:
            sweep = int(360 * progress)
            cv2.ellipse(frame, (cx, cy), (60, 60), -90, 0, sweep, C_GREEN, 4)

        # Centre pulse
        pulse_r = int(18 + 6 * abs(np.sin(tick * 4)))
        cv2.circle(frame, (cx, cy), pulse_r, C_ACCENT, -1)
        cv2.circle(frame, (cx, cy), pulse_r + 6, C_ACCENT, 1)

        # Title
        title = "FACE RECOGNITION SYSTEM"
        tw    = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.putText(frame, title, (cx - tw//2, cy - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_MUTED, 1)

        # Subtitle
        sub = "Initialising camera..."
        sw  = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(frame, sub, (cx - sw//2, cy + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GREEN, 1)

        # Loading bar below spinner
        bar_w = 240
        bar_x = cx - bar_w // 2
        bar_y = cy + 130
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6),
                      (30, 45, 70), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * progress), bar_y + 6),
                      C_ACCENT, -1)

        pct_txt = f"{int(progress * 100)}%"
        cv2.putText(frame, pct_txt, (cx - 14, bar_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_MUTED, 1)

    # ─────────────────────────────────────────────────────────────
    # FACE SNAPSHOT  (was missing)
    # ─────────────────────────────────────────────────────────────

    def capture_face_snapshot(self, frame, faces_info):
        """
        Save a cropped snapshot of each detected face with
        the person's name stamped on it.
        """
        if not faces_info:
            print("No faces to snapshot.")
            return
 
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, fi in enumerate(faces_info):
            x, y, w, h = fi['bbox']
            pad = 20
            fh, fw = frame.shape[:2]
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(fw, x+w+pad); y2 = min(fh, y+h+pad)
            crop = frame[y1:y2, x1:x2].copy()
            name = fi.get('name', 'unknown').replace(' ', '_')
            path = f"snapshots/face_{name}_{ts}_{i}.png"
            cv2.imwrite(path, crop)
            print(f"Face snapshot saved → {path}")
 
    # ─────────────────────────────────────────────────────────────
    # REGISTRATION UI
    # ─────────────────────────────────────────────────────────────
 
    def draw_registration_ui(self, frame, sample_count, samples_needed,
                              name, person_id, role,
                              quality_score, face_centered, thumbnails):
        fh, fw = frame.shape[:2]
        progress = sample_count / samples_needed
        tick     = time.time()

        # Semi-transparent overlay on actual camera feed (not solid gradient)
        overlay = np.full(frame.shape, (14, 20, 36), dtype=np.uint8)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
 
        # ── HEADER ───────────────────────────────────────────────
        draw_gradient_panel(frame, 0, 0, fw, 75,
                            col_top=(18,26,50), col_bot=(12,18,38), alpha=0.92)
        cv2.rectangle(frame, (0, 73), (fw, 75), C_ACCENT, -1)
 
        cv2.putText(frame, "BIOMETRIC CAPTURE",
                    (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_MUTED, 1)
        cv2.putText(frame, "Face Registration",
                    (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, C_WHITE, 2)
        cv2.putText(frame, "Face",
                    (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, C_ACCENT, 2)
 
        clk     = datetime.now().strftime("%H:%M:%S")
        dot_col = C_GREEN if int(tick*2)%2==0 else (0,140,70)
        cv2.circle(frame, (fw-162, 23), 5, dot_col, -1)
        cv2.putText(frame, clk, (fw-148, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_MUTED, 1)
 
        # ── STOP BUTTON ──────────────────────────────────────────
        bw2, bh2 = 112, 34
        bx, by   = fw - bw2 - 14, 84
        frosted(frame, bx, by, bx+bw2, by+bh2, C_DARK_RED, 0.93)
        cv2.rectangle(frame, (bx, by), (bx+bw2, by+bh2), (80,80,220), 1)
        cv2.rectangle(frame, (bx+10,by+9), (bx+22,by+25), (80,80,220), -1)
        cv2.putText(frame, "STOP", (bx+30, by+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 2)
 
        cv2.putText(frame, f"{sample_count}/{samples_needed}",
                    (bx+12, by+bh2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_YELLOW, 2)
        cv2.putText(frame, "SAMPLES",
                    (bx+12, by+bh2+36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_MUTED, 1)
 
        # ── SEGMENTED PROGRESS BAR ───────────────────────────────
        seg_count  = samples_needed
        bar_left   = 20
        bar_right  = bx - 10
        bar_y_top  = 82
        bar_w_full = bar_right - bar_left
        seg_gap    = 2
        seg_w      = max(2, (bar_w_full - (seg_count-1)*seg_gap) // seg_count)
 
        for i in range(seg_count):
            sx     = bar_left + i * (seg_w + seg_gap)
            col    = C_GREEN if i < sample_count else (35, 45, 65)
            cv2.rectangle(frame, (sx, bar_y_top),
                          (sx+seg_w, bar_y_top+7), col, -1)
 
        cv2.putText(frame, f"{int(progress*100)}%",
                    (bar_left, bar_y_top+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    C_GREEN if progress > 0 else C_MUTED, 1)
 
        # ── FACE QUALITY BAR (left side, vertical) ───────────────
        q_x, q_y, q_h = 14, fh//2 - 80, 160
        q_fill = int(q_h * quality_score / 100)
        q_col  = C_GREEN if quality_score>70 else (C_YELLOW if quality_score>40 else C_RED)
        frosted(frame, q_x, q_y, q_x+22, q_y+q_h, C_PANEL, 0.8)
        cv2.rectangle(frame, (q_x+4, q_y+(q_h-q_fill)), (q_x+18, q_y+q_h), q_col, -1)
        cv2.rectangle(frame, (q_x+4, q_y), (q_x+18, q_y+q_h), C_MUTED, 1)
        cv2.putText(frame, "Q", (q_x+6, q_y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_MUTED, 1)
        cv2.putText(frame, str(quality_score), (q_x+2, q_y+q_h+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, q_col, 1)
 
        # ── ANIMATED GUIDE RING ───────────────────────────────────
        cx_g, cy_g = fw // 2, fh // 2
        R_OUTER = 175;  R_MID = 164;  R_INNER = 150
 
        ring_accent = C_GREEN if face_centered else C_ACCENT
        spin_angle  = (tick * 45) % 360
        for seg_i in range(12):
            a_start = int(spin_angle + seg_i * 30)
            a_end   = a_start + 18
            dim     = tuple(int(c * 0.5) for c in ring_accent)
            cv2.ellipse(frame, (cx_g, cy_g), (R_OUTER, R_OUTER),
                        0, a_start, a_end, dim, 1)
 
        g_frac  = progress
        mid_col = (0, int(180+50*g_frac), int(255-125*g_frac))
        cv2.circle(frame, (cx_g, cy_g), R_MID, mid_col, 2)
 
        if sample_count > 0:
            sweep = int(360 * progress)
            cv2.ellipse(frame, (cx_g, cy_g), (R_MID, R_MID),
                        -90, 0, sweep, C_GREEN, 4)
 
        cv2.circle(frame, (cx_g, cy_g), R_INNER, (30,45,70), 1)
 
        mark_len = 24;  mark_off = 122
        for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
            mx, my = cx_g + dx*mark_off, cy_g + dy*mark_off
            col    = C_GREEN if face_centered else C_ACCENT
            cv2.line(frame, (mx, my), (mx+dx*mark_len, my), col, 2)
            cv2.line(frame, (mx, my), (mx, my+dy*mark_len), col, 2)
 
        cv2.circle(frame, (cx_g, cy_g), 4, C_GREEN, -1)
        cv2.circle(frame, (cx_g, cy_g), 9, C_GREEN, 1)
 
        hint     = "Hold still - Capturing..." if face_centered else "Centre your face in the ring"
        hint_col = C_GREEN if face_centered else C_YELLOW
        hint_x   = cx_g - len(hint) * 6
        frosted(frame, hint_x-8, cy_g+R_OUTER+8,
                hint_x+len(hint)*12+8, cy_g+R_OUTER+32, C_BG, 0.7)
        cv2.putText(frame, hint, (hint_x, cy_g+R_OUTER+26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, hint_col, 1)
 
        # ── THUMBNAIL STRIP ───────────────────────────────────────
        if thumbnails:
            strip_x = fw - 72
            strip_y = by + bh2 + 55
            cv2.putText(frame, "RECENT", (strip_x-2, strip_y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_MUTED, 1)
            for i, thumb in enumerate(thumbnails[-5:]):
                ty = strip_y + i * 58
                if ty + 54 > fh - 105:
                    break
                # Thumbnail is already color (BGR)
                thumb_resized = cv2.resize(thumb, (50, 50))
                frame[ty:ty+50, strip_x:strip_x+50] = thumb_resized
                cv2.rectangle(frame, (strip_x,ty), (strip_x+50,ty+50), C_ACCENT, 1)
 
        # ── PERSON INFO CARD ──────────────────────────────────────
        card_h = 95
        card_y = fh - card_h - 10
        draw_gradient_panel(frame, 10, card_y, fw-10, card_y+card_h,
                            col_top=(24,36,66), col_bot=(14,22,44), alpha=0.92)
        cv2.rectangle(frame, (10, card_y), (fw-10, card_y+1), C_ACCENT, -1)
        cv2.rectangle(frame, (10, card_y), (14, card_y+card_h), C_ACCENT, -1)
 
        cv2.putText(frame, "NAME",  (28, card_y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_MUTED, 1)
        cv2.putText(frame, name,    (28, card_y+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, C_WHITE, 2)
 
        d1 = fw // 3
        cv2.rectangle(frame, (d1, card_y+10), (d1+1, card_y+card_h-10), (40,55,80), -1)
        cv2.putText(frame, "BADGE ID", (d1+12, card_y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_MUTED, 1)
        cv2.putText(frame, str(person_id), (d1+12, card_y+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, C_YELLOW, 2)
 
        d2 = (fw*2)//3
        cv2.rectangle(frame, (d2, card_y+10), (d2+1, card_y+card_h-10), (40,55,80), -1)
        cv2.putText(frame, "ROLE", (d2+12, card_y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_MUTED, 1)
        cv2.putText(frame, str(role) if role else "-", (d2+12, card_y+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_ORANGE, 2)
 
        steps     = ["ALIGN", "CAPTURE", "DONE"]
        step_done = 0 if sample_count==0 else (1 if sample_count<samples_needed else 2)
        pill_start= 28
        for si, label in enumerate(steps):
            pill_x = pill_start + si * 72
            done   = si < step_done
            active = si == step_done
            fc     = C_GREEN if done else (C_ACCENT if active else (35,45,65))
            tc     = (10,10,10) if done else (C_WHITE if active else C_MUTED)
            cv2.rectangle(frame, (pill_x, card_y+58), (pill_x+64, card_y+78), fc, -1)
            cv2.putText(frame, label, (pill_x+8, card_y+72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, tc, 1)
 
        return bx, by, bw2, bh2
 
    # ─────────────────────────────────────────────────────────────
    # RECOGNITION HUD
    # ─────────────────────────────────────────────────────────────
 
    def draw_hud(self, frame, faces_info, fps, alert_unknown):
        fh, fw = frame.shape[:2]
        tick   = time.time()
        uptime = int(tick - self.session_start)
        known_pct = (
            int(100 * self.session_known_faces / self.session_total_faces)
            if self.session_total_faces > 0 else 0
        )

        # Unknown alert flash
        if alert_unknown and int(tick*4)%2==0:
            cv2.rectangle(frame, (0,0), (fw-1,fh-1), C_RED, 6)

        # Header
        draw_gradient_panel(frame, 0, 0, fw, 75,
                            col_top=(18,26,50), col_bot=(12,18,38), alpha=0.92)
        cv2.rectangle(frame, (0,73), (fw,75), C_ACCENT, -1)
 
        cv2.putText(frame, "FACE RECOGNITION SYSTEM",
                    (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_MUTED, 1)
        cv2.putText(frame, "Face Recognition",
                    (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, C_WHITE, 2)
        cv2.putText(frame, "Face",
                    (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, C_ACCENT, 2)
 
        clk     = datetime.now().strftime("%H:%M:%S")
        dot_col = C_GREEN if int(tick*2)%2==0 else (0,140,70)
        cv2.circle(frame, (fw-165, 23), 6, dot_col, -1)
        cv2.putText(frame, "LIVE", (fw-150, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, dot_col, 2)
        cv2.putText(frame, clk, (fw-150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_MUTED, 1)
 
        # Right panel
        rp_x, rp_y = fw-160, 85
        rp_w, rp_h = 150, 145
        draw_gradient_panel(frame, rp_x, rp_y, rp_x+rp_w, rp_y+rp_h,
                            col_top=(24,36,66), col_bot=(14,22,44), alpha=0.88)
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
            sy = rp_y + 28 + i*19
            cv2.putText(frame, label, (rp_x+8, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_MUTED, 1)
            cv2.putText(frame, val, (rp_x+90, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GREEN, 1)
 
        # Left event log
        if self.recent_events:
            lp_x, lp_y = 10, 85
            lp_h       = 20 + len(self.recent_events)*20
            draw_gradient_panel(frame, lp_x, lp_y, lp_x+230, lp_y+lp_h,
                                col_top=(24,36,66), col_bot=(14,22,44), alpha=0.85)
            cv2.putText(frame, "RECENT EVENTS",
                        (lp_x+8, lp_y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_MUTED, 1)
            for i, ev in enumerate(self.recent_events[-5:]):
                cv2.putText(frame, ev, (lp_x+8, lp_y+28+i*18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1)
 
        # Bottom bar
        draw_gradient_panel(frame, 0, fh-58, fw, fh,
                            col_top=(12,18,38), col_bot=(18,26,50), alpha=0.92)
        cv2.rectangle(frame, (0, fh-58), (fw, fh-56), C_ACCENT, -1)
 
        status = "SCANNING - NO FACES DETECTED" if len(faces_info)==0 else \
                 f"ACTIVE - {len(faces_info)} FACE(S) IN FRAME"
        cv2.putText(frame, status, (20, fh-32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_GREEN, 1)
        hints = "S=screenshot  F=freeze  M=mirror  Z/X=zoom  +/-=threshold  C=snapshot  Q=quit"
        cv2.putText(frame, hints, (20, fh-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_MUTED, 1)
 
        if self.freeze_frame:
            frosted(frame, fw//2-70, fh//2-24, fw//2+70, fh//2+24, C_BG, 0.8)
            cv2.putText(frame, "FROZEN", (fw//2-52, fh//2+8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_YELLOW, 3)
 
        if self.zoom_factor > 1.0:
            cv2.putText(frame, f"ZOOM {self.zoom_factor:.1f}x",
                        (fw-140, fh-65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_YELLOW, 2)
 
        # Per-face overlays
        for fi in faces_info:
            x, y, w, h = fi['bbox']
            known       = fi['known']
            confidence  = fi['confidence']
            name        = fi.get('name', 'Unknown')
            pid         = fi.get('person_id', 'N/A')
            role        = fi.get('role', '')
            box_col     = C_GREEN if known else C_RED
 
            self.corner_box_glow(frame, x, y, w, h, box_col)
 
            conf_txt = f"{confidence:.0f}%"
            bx2      = x + w - 64
            by2      = y - 28
            if by2 > 5:
                frosted(frame, bx2, by2, bx2+60, by2+22, box_col, 0.9)
                cv2.putText(frame, conf_txt, (bx2+6, by2+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 2)
 
            if known:
                cx2 = max(0, x-20)
                cy2 = y + h + 14
                cw2 = w + 40
                ch2 = 88 if role else 68
 
                if cy2+ch2 < fh and cx2+cw2 <= fw:
                    draw_gradient_panel(frame, cx2, cy2, cx2+cw2, cy2+ch2,
                                        col_top=(24,36,66), col_bot=(14,22,44), alpha=0.92)
                    cv2.rectangle(frame, (cx2,cy2), (cx2+cw2,cy2+ch2), box_col, 2)
                    cv2.putText(frame, name,         (cx2+12, cy2+28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_WHITE, 2)
                    cv2.putText(frame, f"ID: {pid}", (cx2+12, cy2+50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_MUTED, 1)
                    if role:
                        cv2.putText(frame, str(role), (cx2+12, cy2+70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_ORANGE, 1)
                    cv2.putText(frame, "RECOGNISED",
                                (cx2+cw2-118, cy2+22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GREEN, 1)
            else:
                frosted(frame, x, y-34, x+w, y-6, C_DARK_RED, 0.85)
                cv2.putText(frame, "UNKNOWN", (x+8, y-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,80,240), 2)
 
    # ─────────────────────────────────────────────────────────────
    # TTS  (speak)
    # ─────────────────────────────────────────────────────────────
 
    def speak(self, message):
        if not self.tts_engine:
            return
        def _speak():
            try:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
        t = threading.Thread(target=_speak)
        t.daemon = True
        t.start()
 
    # ─────────────────────────────────────────────────────────────
    # MODE 1 – REGISTER
    # ─────────────────────────────────────────────────────────────
 
    def register_person(self):
        print("\n╔══════════════════════════════╗")
        print("║     FACE REGISTRATION v3     ║")
        print("╚══════════════════════════════╝\n")
 
        print("Enter full name:         ", end='', flush=True)
        name = sys.stdin.readline().strip()
        if not name:
            print("Name cannot be empty!"); return
 
        print("Enter badge / ID number: ", end='', flush=True)
        person_id = sys.stdin.readline().strip()
        if not person_id:
            print("ID cannot be empty!"); return
 
        print("Enter role / department (or press Enter to skip): ", end='', flush=True)
        role = sys.stdin.readline().strip()
 
        internal_id = self.get_next_person_id()
        person_dir  = os.path.join(self.faces_dir, str(internal_id))
        Path(person_dir).mkdir(exist_ok=True)
 
        self.people_db[str(internal_id)] = {
            "name": name, "id": person_id, "role": role,
            "registered": datetime.now().isoformat()
        }
        self.save_database()
 
        print(f"\nRegistering: {name}  |  Badge: {person_id}  |  Role: {role or '—'}")
        print("Look at the camera. Centre your face in the ring.")
        print("Press Q or click STOP to cancel.\n")
 
        samples_needed   = 40
        sample_count     = 0
        last_capture     = 0
        capture_interval = 0.12
        stop_clicked     = False
        quality_score    = 0
        thumbnails       = []
        stop_btn         = [0, 0, 0, 0]
 
        def on_mouse(event, mx, my, flags, param):
            nonlocal stop_clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                bx, by, bw2, bh2 = stop_btn
                if bx <= mx <= bx+bw2 and by <= my <= by+bh2:
                    stop_clicked = True
 
        cv2.namedWindow("Registration")
        cv2.setMouseCallback("Registration", on_mouse)
 
        reg_camera       = Camera()
        reg_face_detector= FaceDetector()
 
        if not reg_camera.open():
            print("Error: Cannot open camera for registration!"); return
 
        while sample_count < samples_needed and not stop_clicked:
            ret, frame = reg_camera.read()
            if not ret:
                continue
 
            frame = cv2.flip(frame, 1)
            fh2, fw2 = frame.shape[:2]
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enh   = reg_face_detector.enhance_frame(gray)
            faces = reg_face_detector.detect_faces_multi_scale(enh)
 
            face_centered = False
            best_face     = None
 
            for (fx, fy, fw3, fh3) in faces:
                if fw3 < 100 or fh3 < 100:
                    continue
                face_cx = fx + fw3 / 2
                face_cy = fy + fh3 / 2
                centered = (abs(face_cx - fw2/2) < fw2*0.28 and
                            abs(face_cy - fh2/2) < fh2*0.28)
                if centered:
                    face_centered = True
                    best_face     = (fx, fy, fw3, fh3)
                    quality_score = self.face_quality_score(enh[fy:fy+fh3, fx:fx+fw3])
 
            if face_centered and best_face and quality_score > 45:
                fx, fy, fw3, fh3 = best_face
                now = time.time()
                if now - last_capture > capture_interval:
                    raw_face = enh[fy:fy+fh3, fx:fx+fw3]
                    smoothed_face = reg_face_detector.smooth_face_skin(raw_face, 0.6)
                    enhanced_face = reg_face_detector.enhance_face_clarity(smoothed_face, 1.2)
                    face_100 = cv2.resize(enhanced_face, (100, 100))
                    # Save grayscale for LBPH recognizer (required)
                    cv2.imwrite(os.path.join(person_dir, f"sample_{sample_count}.jpg"), face_100)
                    # Convert to BGR for color thumbnail display
                    thumb_color = cv2.cvtColor(face_100, cv2.COLOR_GRAY2BGR)
                    thumbnails.append(thumb_color)
                    sample_count += 1
                    last_capture  = now
                self.corner_box(frame, fx, fy, fw3, fh3,
                                C_GREEN if face_centered else C_ACCENT, 3)
 
            coords = self.draw_registration_ui(
                frame, sample_count, samples_needed,
                name, person_id, role, quality_score, face_centered, thumbnails
            )
            stop_btn[:] = list(coords)
 
            cv2.imshow("Registration", frame)
            cv2.setWindowProperty("Registration", cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_clicked = True
 
        reg_camera.release()
        cv2.destroyAllWindows()
 
        if stop_clicked:
            print("\nCancelled. Rolling back...")
            del self.people_db[str(internal_id)]
            self.save_database()
            if os.path.exists(person_dir):
                shutil.rmtree(person_dir)
        else:
            print(f"\n✔ Registered {name} with {sample_count} samples.")
            print("Next: python face_camera.py --retrain")
 
    # ─────────────────────────────────────────────────────────────
    # MODE 2 – TRAIN
    # ─────────────────────────────────────────────────────────────
 
    def train_model(self):
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
 
        print(f"Training on {len(faces)} photos across {len(set(labels))} people...")
        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save(self.model_file)
        print(f"✔ Model saved → {self.model_file}")
        return True
 
    # ─────────────────────────────────────────────────────────────
    # MODE 3 – LIVE RECOGNITION
    # ─────────────────────────────────────────────────────────────
 
    def run_recognition(self):
        print("\n╔══════════════════════════════╗")
        print("║   LIVE RECOGNITION v3.1      ║")
        print("╠══════════════════════════════╣")
        print("║  S  screenshot               ║")
        print("║  F  freeze / unfreeze        ║")
        print("║  M  mirror toggle            ║")
        print("║  Z  zoom in                  ║")
        print("║  X  zoom out                 ║")
        print("║  +  loosen threshold         ║")
        print("║  -  tighten threshold        ║")
        print("║  C  capture face snapshot    ║")
        print("║  Q  quit                     ║")
        print("╚══════════════════════════════╝\n")
 
        if not os.path.exists(self.model_file):
            print("Model not found! Run --retrain first."); return
 
        self.recognizer.read(self.model_file)
        self.session_start = time.time()
 
        if not self.camera.open():
            print("Error: Cannot open camera!"); return
 
        prev_time          = time.time()
        frozen_frame       = None
        show_loading       = True
        loading_start_time = time.time()
        loading_duration   = 2.0
 
        while True:
            current_time = time.time()
 
            # Loading phase
            if show_loading and (current_time - loading_start_time) < loading_duration:
                ret, frame = self.camera.read()
                if not ret:
                    frame = np.zeros((720, 1280, 3), np.uint8)
                if self.mirror_mode:
                    frame = cv2.flip(frame, 1)
                frame = self.apply_zoom(frame)
                self.draw_loading_screen(frame,
                                         current_time - loading_start_time,
                                         loading_duration)
                cv2.imshow("Face Recognition System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            else:
                show_loading = False
 
            if not self.freeze_frame:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                frozen_frame = frame.copy()
            else:
                frame = frozen_frame.copy() if frozen_frame is not None \
                        else np.zeros((720, 1280, 3), np.uint8)
 
            if self.mirror_mode:
                frame = cv2.flip(frame, 1)
 
            frame = self.apply_zoom(frame)
 
            now       = time.time()
            fps       = 1 / (now - prev_time) if now > prev_time else 0
            prev_time = now
 
            faces         = self.face_detector.detect_faces(frame)
            faces_info    = []
            alert_unknown = False
 
            for (x, y, w, h) in faces:
                if w < 80 or h < 80:
                    continue
                self.session_total_faces += 1
 
                face_gray     = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                smooth_face   = self.face_detector.smooth_face_skin(face_gray, 0.4)
                enhanced_face = self.face_detector.enhance_face_clarity(smooth_face, 0.8)
                face_100      = cv2.resize(enhanced_face, (100, 100))
                label, confidence = self.recognizer.predict(face_100)
 
                known = confidence < self.confidence_threshold
 
                info = {
                    'bbox':      (x, y, w, h),
                    'known':     known,
                    'confidence':confidence,
                    'name':      'Unknown',
                    'person_id': 'N/A',
                    'role':      '',
                }
 
                if known and str(label) in self.people_db:
                    person            = self.people_db[str(label)]
                    info['name']      = person['name']
                    info['person_id'] = person['id']
                    info['role']      = person.get('role', '')
                    self.session_known_faces += 1
 
                    if (self.last_recognized_name != person['name'] or
                            now - self.last_recognition_time > self.recognition_cooldown):
                        self.speak(f"Welcome, {person['name']}")
                        self.last_recognized_name  = person['name']
                        self.last_recognition_time = now
 
                        entry = {
                            "time":  datetime.now().isoformat(),
                            "name":  person['name'],
                            "badge": person['id'],
                            "conf":  round(confidence, 1),
                        }
                        self.rec_log.append(entry)
                        self.save_log()
 
                        ts = datetime.now().strftime("%H:%M:%S")
                        self.recent_events.append(f"{ts}  {person['name']}")
                        self.recent_events = self.recent_events[-5:]
                else:
                    alert_unknown = True
 
                faces_info.append(info)
 
            # Draw HUD with smooth gradient background
            if USE_EXT_HUD:
                _ExtHUD.draw_full_hud(frame, faces_info, fps, len(self.people_db))
            else:
                self.draw_hud(frame, faces_info, fps, alert_unknown)
 
            cv2.imshow("Face Recognition System", frame)
            key = cv2.waitKey(1) & 0xFF
 
            if   key == ord('q'):
                break
            elif key == ord('s'):
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"screenshots/snapshot_{ts}.png"
                cv2.imwrite(path, frame)
                print(f"Screenshot saved → {path}")
            elif key == ord('f'):
                self.freeze_frame = not self.freeze_frame
                print(f"Freeze: {'ON' if self.freeze_frame else 'OFF'}")
            elif key == ord('m'):
                self.mirror_mode = not self.mirror_mode
                print(f"Mirror: {'ON' if self.mirror_mode else 'OFF'}")
            elif key == ord('z'):
                self.zoom_factor = min(4.0, round(self.zoom_factor + 0.25, 2))
                print(f"Zoom: {self.zoom_factor}×")
            elif key == ord('x'):
                self.zoom_factor = max(1.0, round(self.zoom_factor - 0.25, 2))
                print(f"Zoom: {self.zoom_factor}×")
            elif key in (ord('+'), ord('=')):
                self.confidence_threshold = min(120, self.confidence_threshold + 5)
                print(f"Threshold: {self.confidence_threshold}")
            elif key == ord('-'):
                self.confidence_threshold = max(30, self.confidence_threshold - 5)
                print(f"Threshold: {self.confidence_threshold}")
            elif key == ord('c'):
                self.capture_face_snapshot(frame, faces_info)
 
        self.camera.release()
        cv2.destroyAllWindows()
        print("\nSession ended.")
 
    # ─────────────────────────────────────────────────────────────
    # MODE 4 – LIST
    # ─────────────────────────────────────────────────────────────
 
    def list_people(self):
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
 
    # ─────────────────────────────────────────────────────────────
    # MODE 5 – DELETE
    # ─────────────────────────────────────────────────────────────
 
    def delete_person(self, internal_id):
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
 
    # ─────────────────────────────────────────────────────────────
    # MODE 6 – EXPORT LOG
    # ─────────────────────────────────────────────────────────────
 
    def export_log(self):
        if not self.rec_log:
            print("No events logged yet."); return
        out = "recognition_log.csv"
        with open(out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["time","name","badge","conf"])
            writer.writeheader()
            writer.writerows(self.rec_log)
        print(f"✔ Exported {len(self.rec_log)} events → {out}")
 
 
# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition Camera System v3.1",
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