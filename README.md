# 🎯 Face Recognition Camera System

Real-time camera app that detects faces and displays **Name + ID** as a HUD overlay.

---

## 📦 Install Dependencies

```bash
pip install opencv-python opencv-contrib-python mediapipe numpy
```

---

## 🚀 Quick Start

### Step 1 — Register people

```bash
python face_camera.py --register
```

- Enter the person's **full name** and **ID / badge number**
- Look at the camera — it captures **40 face samples** automatically
- Repeat for every person you want recognised

### Step 2 — Run the live camera

```bash
python face_camera.py
```

- Camera opens, faces are detected in real-time
- **Green box** = recognised person → shows Name + ID
- **Red box**   = unknown face
- Press **Q** to quit

---

## 📋 Other Commands

| Command | Action |
|---------|--------|
| `python face_camera.py` | Start live recognition |
| `python face_camera.py --register` | Register a new person |
| `python face_camera.py --list` | Show all registered people |
| `python face_camera.py --retrain` | Re-train model after manual changes |

---

## 📁 File Structure

```
face_recognition_system/
├── face_camera.py      ← main program
├── requirements.txt    ← dependencies
├── people_db.json      ← auto-created: name/ID database
├── face_model.xml      ← auto-created: trained model
└── known_faces/
    ├── 0/              ← face images for person 0
    ├── 1/              ← face images for person 1
    └── ...
```

---

## 🖥️ HUD Display Explained

```
┌─────────────────────────────────────────┐
│ FACE RECOGNITION  [LIVE]        12:34:56│
├─────────────────────────────────────────┤
│                                         │
│   ┌──────────┐   ← corner bracket box  │
│   │  (face)  │      GREEN = known       │
│   └──────────┘      RED   = unknown     │
│  ┌────────────────┐                     │
│  │ John Smith     │  ← Name             │
│  │ ID: EMP-1042   │  ← ID              │
│  └────────────────┘                     │
│                                         │
├─────────────────────────────────────────┤
│ Faces: 1 | FPS: 28.4 | DB: 3 people    │
└─────────────────────────────────────────┘
```

---

## ⚙️ How It Works

1. **Detection** — MediaPipe FaceDetection finds all faces in the frame
2. **Recognition** — OpenCV LBPH (Local Binary Pattern Histogram) compares face to trained model
3. **Overlay** — Name + ID drawn as a semi-transparent card beneath each face box

---

## 💡 Tips

- Register in **good lighting** for best accuracy
- Register from **multiple angles** (run `--register` 2–3 times per person)
- Confidence score shown top-right of box (higher = more confident)
- Threshold is set at **80** — adjust `known = confidence < 80` in code to tune sensitivity