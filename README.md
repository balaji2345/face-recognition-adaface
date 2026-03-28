# 🎓 Classroom Face Recognition System using AdaFace IR-101

A real-time face recognition system designed for classroom attendance
and monitoring using state-of-the-art AdaFace IR-101 deep learning model
combined with YOLOv8 face detection and ByteTrack multi-object tracking.

---

## 📌 Project Overview

This system can automatically detect and recognize student faces from
CCTV footage or video recordings. It handles challenging real-world
conditions like:
- Far/small faces (back-row students)
- Low-quality or blurry frames
- Different lighting conditions
- Partial face visibility and head pose variations

---

## 🧠 How It Works
```
Video Input
    ↓
YOLOv8 Face Detection (multi-scale tiling)
    ↓
ByteTrack Multi-Object Tracking (stable IDs)
    ↓
InsightFace Landmark Detection + Face Alignment
    ↓
AdaFace IR-101 Embedding Extraction
    ↓
Cosine Similarity Matching against Student Database
    ↓
Temporal Smoothing + Confirmation Voting
    ↓
Recognized / Unknown Label on Video
```

---

## 📁 Project Structure
```
face_recognition/
│
├── adaface_build_embeddings.py   # Step 1: Build student face database
├── adaface_recognition.py        # Step 2: Run real-time recognition
│
├── adaface_model/                # AdaFace IR-101 model files (not uploaded)
│   ├── pretrained_model/
│   │   ├── model.pt
│   │   └── model.yaml
│   └── models/
│
├── students.npz                  # Generated face embeddings (not uploaded)
├── yolov8m-face.pt               # YOLOv8 face detection model (not uploaded)
│
├── faces_split/
│   └── train/                    # Training images (one folder per student)
│       ├── Student_Name_1/
│       ├── Student_Name_2/
│       └── ...
│
├── unknown_faces_adaface/        # Auto-saved unknown face crops (generated)
└── .gitignore
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Face Detection | YOLOv8m-face + InsightFace Buffalo_L |
| Face Recognition | AdaFace IR-101 WebFace12M |
| Multi-Object Tracking | ByteTrack (via Ultralytics) |
| Face Alignment | InsightFace norm_crop (112×112) |
| Image Enhancement | CLAHE (Contrast Limited Adaptive Histogram Equalization) |
| Framework | PyTorch |
| Language | Python 3.9+ |

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/balaji2345/face_recognition.git
cd face_recognition
```

### 2. Install Dependencies
```bash
pip install torch torchvision
pip install opencv-python
pip install insightface
pip install ultralytics
pip install huggingface_hub
pip install omegaconf
pip install numpy
```

### 3. Download AdaFace Model
```bash
python adaface_build_embeddings.py --download
```

### 4. Prepare Training Images
Organize student photos like this:
```
faces_split/train/
    ├── John_Smith/
    │     ├── photo1.jpg
    │     ├── photo2.jpg
    │     └── photo3.jpg
    ├── Jane_Doe/
    │     ├── photo1.jpg
    │     └── photo2.jpg
    └── ...
```
> 💡 Tip: 5–15 clear face photos per student gives best results.

### 5. Generate Face Embeddings
```bash
python adaface_build_embeddings.py --encode
```

### 6. Run Recognition on Video
Set your video path in `adaface_recognition.py`:
```python
VIDEO_PATH = r"path/to/your/video.mp4"
```
Then run:
```bash
python adaface_recognition.py
```

---

## 🎯 Key Features

- ✅ **Dual-Zone Detection** — Separate thresholds for front-row and
  back-row students (ceiling-mounted CCTV support)
- ✅ **Multi-Scale Tiling** — Detects small/distant faces missed by
  standard single-pass detection
- ✅ **Temporal Smoothing** — Reduces false positives using voting
  window across multiple frames
- ✅ **CLAHE Enhancement** — Improves recognition in poor lighting
- ✅ **Unknown Face Saving** — Automatically saves unrecognized face
  crops for review
- ✅ **Real-Time Display** — Color-coded bounding boxes with similarity
  scores

---

## 🎨 Label Color Guide

| Color | Meaning |
|---|---|
| 🟢 Green | Confirmed recognized student |
| 🟠 Orange | Confirming... (collecting votes) |
| 🟡 Yellow | Best guess (low confidence) |
| 🔴 Red | Unknown person |
| 🔵 Cyan | Face too far/small to process |

---

## 📊 Configuration (adaface_recognition.py)

| Parameter | Default | Description |
|---|---|---|
| `SIM_FRONT` | 0.38 | Similarity threshold for front-row |
| `SIM_BACK` | 0.32 | Similarity threshold for back-row |
| `ZONE_SPLIT_RATIO` | 0.55 | Top 55% = back zone |
| `SMOOTH_WINDOW` | 7 | Frames used for voting |
| `CONFIRM_VOTES` | 3 | Votes needed to confirm identity |
| `MAX_YAW` | 55° | Max head turn angle allowed |
| `MAX_PITCH` | 38° | Max head tilt angle allowed |

---

## 📋 Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended for real-time performance)
- Minimum 6GB GPU VRAM
- Windows / Linux

---

## 📌 Notes

- Model files (`adaface_model/`, `*.pt`) are **not included** in this
  repo due to size. Download them using the `--download` command.
- `students.npz` is generated locally and not uploaded.
- For best accuracy, use **front-facing, well-lit photos** for training.

---

## 👨‍💻 Author

**Balaji** — [github.com/balaji2345](https://github.com/balaji2345)

---

## 📄 License

This project is for educational purposes only.
