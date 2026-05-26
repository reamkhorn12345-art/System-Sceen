# Face Recognition Camera System v3.0

## Overview
This is an enhanced face recognition system with improved camera quality, face detection accuracy, and user experience. The system now features HD resolution, better image processing, stabilized face detection, modern HUD overlay, and face snapshot capabilities.

## Features

### Camera Quality Improvements
- Increased camera resolution to Full HD (1920x1080)
- Improved brightness, contrast, and sharpness settings
- Auto-focus support (if camera supports it)
- Better FPS for smoother video stream
- CLAHE and histogram equalization for enhanced contrast in poor lighting

### Face Detection Improvements
- Multi-scale face detection for better accuracy at various distances
- Face bounding box stabilization to reduce shaking/flickering
- Improved detection in low light conditions
- Non-maximum suppression to eliminate duplicate detections
- Better face quality scoring (blur, brightness, contrast)

### HUD Overlay Improvements
- Modern, futuristic HUD design
- Displays User Name, User ID, and Confidence %
- Color-coded boxes: Green for verified users, Red for unknown users
- Text shadow/background for better readability
- Animated face box corners
- Status text: "Face Detected" / "No Face Found"
- Loading animation at startup
- Fullscreen mode support
- Responsive design for different screen sizes

### Face Snapshot Feature
- Button to capture detected face image ('C' key)
- Automatic saving with timestamp
- Filename format: `userID_YYYY-MM-DD_HH-MM-SS.jpg`
- Saves both full frame and individual face crops

### User Experience Improvements
- Loading animation when camera starts
- Clear status indicators
- Improved keyboard controls
- Better error handling
- Organized code structure with separate modules

### Performance Optimizations
- Reduced CPU and memory usage
- Async processing considerations
- Optimized camera rendering pipeline
- Prevents UI freezing during detection

## Modules

1. `camera.py` - Handles video capture with improved settings
2. `face_detector.py` - Improved face detection with multi-scale and stabilization
3. `hud.py` - Heads-Up Display module for overlay (unchanged from v2)
4. `config.py` - Configuration constants and settings
5. `face_camera.py` - Main application logic (enhanced v3.0)

## Installation

```bash
pip install opencv-contrib-python pyttsx3 numpy
```

## Usage

### Live Recognition
```bash
python face_camera.py
```

### Register New Person
```bash
python face_camera.py --register
```

### Retrain Model
```bash
python face_camera.py --retrain
```

### List Registered People
```bash
python face_camera.py --list
```

### Delete Person
```bash
python face_camera.py --delete ID
```

### Export Log to CSV
```bash
python face_camera.py --export
```

## Keyboard Controls (Live Recognition)

- **S** - Save screenshot
- **F** - Freeze/unfreeze frame
- **M** - Toggle mirror mode
- **Z** - Zoom in
- **X** - Zoom out
- **+** - Loosen confidence threshold
- **-** - Tighten confidence threshold
- **C** - Capture face snapshot
- **Q** - Quit application

## Configuration

Adjust settings in `config.py`:
- Camera resolution (CAMERA_WIDTH, CAMERA_HEIGHT)
- FPS target (CAMERA_FPS)
- Image enhancement settings (BRIGHTNESS, CONTRAST, etc.)
- Face detection parameters
- HUD colors and dimensions
- Recognition threshold

## Output Directories

- `known_faces/` - Stores registered face samples
- `screenshots/` - Stores screenshots (S key)
- `snapshots/` - Stores face snapshots (C key)
- `face_model.xml` - Trained recognition model
- `people_db.json` - Database of registered people
- `recognition_log.json` - Log of recognition events

## Requirements

- OpenCV (opencv-contrib-python for face recognition)
- pyttsx3 (for text-to-speech)
- NumPy
- Webcam or camera device

## Future Improvements

Consider adding:
- Face recognition matching with database
- Unknown person alerts
- Face distance estimation
- Blink detection/liveness detection
- Attendance logging system
- Integration with security systems
- Cloud synchronization