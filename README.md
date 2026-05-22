# yoga_pose_predictor
identify the yoga pose with imge ,webcam,video
# Yoga Pose Detection using OpenPose + OpenCV

Detect and classify yoga poses in real-time from a **webcam**, a **video file**, or a **static image** — powered by OpenPose (COCO body-25) running through OpenCV's DNN module.

---

## Supported Poses

| Pose | Sanskrit Name |
|------|--------------|
| Warrior I | Virabhadrasana I |
| Warrior II | Virabhadrasana II |
| Tree Pose | Vrksasana |
| Downward Dog | Adho Mukha Svanasana |
| Mountain Pose | Tadasana |
| Cobra Pose | Bhujangasana |
| Child's Pose | Balasana |
| Triangle Pose | Trikonasana |

---

## Project Structure

```
yoga_pose_detection/
├── main.py                     # Entry point
├── requirements.txt
├── models/                     # Place OpenPose model files here
│   ├── pose_deploy.prototxt
│   └── pose_iter_440000.caffemodel
├── input/                      # Drop test images/videos here
├── output/                     # Results saved here
└── utils/
    ├── pose_detector.py        # OpenPose inference wrapper
    ├── yoga_classifier.py      # Joint-angle rule-based classifier
    ├── visualizer.py           # Skeleton + label overlay
    └── download_models.py      # One-time model downloader
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download OpenPose model files
```bash
python utils/download_models.py
```
> This downloads `pose_deploy.prototxt` (~28 KB) and `pose_iter_440000.caffemodel` (~200 MB) into `models/`.

### 3. Run

**Webcam (real-time)**
```bash
python main.py --mode webcam
```
Press `q` to quit, `s` to save a snapshot.

**Video file**
```bash
python main.py --mode video --input input/yoga_video.mp4
```

**Image file**
```bash
python main.py --mode image --input input/warrior_pose.jpg
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `webcam` | `webcam` / `video` / `image` |
| `--input` | — | Path to video or image (required for those modes) |
| `--output` | `output/` | Directory to save results |
| `--model-dir` | `models/` | Directory with OpenPose model files |
| `--cam` | `0` | Webcam device index |
| `--threshold` | `0.1` | Keypoint confidence threshold (0–1) |

---

## How It Works

1. **Pose Estimation** — `PoseDetector` feeds each frame through the OpenPose Caffe model via `cv2.dnn`. It returns 25 body keypoints (x, y) with confidence scores.

2. **Pose Classification** — `YogaClassifier` computes joint angles (e.g. knee bend, arm raise, hip angle) and scores each pose against a set of angle constraints. The highest-scoring pose above a threshold is returned.

3. **Visualisation** — `draw_skeleton` overlays coloured limb lines and joint dots; `draw_pose_label` adds a semi-transparent banner with the pose name and a confidence bar.

---

## GPU Acceleration

The detector automatically attempts to use CUDA (OpenCV DNN CUDA backend). To enable it, build OpenCV with CUDA support or install `opencv-python` with GPU wheels.

---

## Extending with New Poses

Add a new rule class in `utils/yoga_classifier.py`:

```python
class _MyNewPose(_PoseRule):
    def __init__(self): super().__init__("My New Pose")

    def score(self, kp):
        scores = []
        knee_angle = _angle(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])
        _, s = _in_range(knee_angle, 80, 100)
        scores.append(s)
        return _avg(scores) * 100
```

Then add an instance to `POSE_RULES`:
```python
POSE_RULES = [
    ...
    _MyNewPose(),
]
```

---

## License
MIT
