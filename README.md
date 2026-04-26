# Fitness App — Phone Camera Viewer

Stream your phone's camera to your Mac/PC in real-time over Wi-Fi, or process a local video file, using Python and OpenCV. Optionally overlay 3-D pose landmarks via MediaPipe, YOLOv11-pose, or MMPose.

---

## 1. Set up your phone

### Android — IP Webcam (free, recommended)

1. Install **[IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)** from the Play Store.
2. Open the app, scroll to the bottom, tap **Start server**.
3. Note the IP address shown on screen (e.g. `http://192.168.1.42:8080`).

### iOS — IP Camera Lite (free)

1. Install **[IP Camera Lite](https://apps.apple.com/app/ip-camera-lite/id1013455789)** from the App Store.
2. Open the app and tap **Start**. Note the URL shown (e.g. `http://192.168.1.42:8080`).

> Both your phone and computer must be on the **same Wi-Fi network**.

---

## 2. Set up the conda environment

### Create and activate

```bash
conda create -n fitness python=3.11 -y
conda activate fitness
```

### Core dependencies

```bash
pip install opencv-python "numpy>=1.24.0,<2.0.0"
```

### Pose estimation backends (install only what you need)

#### MediaPipe (default, easiest)

```bash
pip install mediapipe==0.10.21
```

#### YOLOv11-pose (via Ultralytics)

```bash
pip install ultralytics
```

The model file `yolo11n-pose.pt` is downloaded automatically on first run.

#### MMPose (top-down, RTMPose)

MMPose requires building `mmcv` from source. Run in order:

```bash
pip install --upgrade pip setuptools wheel
pip install -U openmim
mim install mmengine
pip install mmcv --no-build-isolation
mim install "mmdet>=3.1.0" "mmpose>=1.1.0"
```

> On Apple Silicon (MPS backend), NMS ops are unsupported — the estimator forces CPU automatically.  
> First run auto-downloads RTMPose-m (~50 MB).

---

## 3. Run the viewer

### Live phone camera

```bash
python phone_camera.py --ip 10.88.111.11:8080
python phone_camera.py --url http://10.88.111.11:8080/video
```

### Webcam

```bash
# Default device (index 0)
python phone_camera.py --webcam

# Specific device
python phone_camera.py --webcam 1
```

### Local video file

```bash
python phone_camera.py --file /path/to/clip.mp4
```

### Common options

```bash
# Record output to output.mp4
python phone_camera.py --ip 10.88.111.11:8080 --record

# Resize display window (recording stays full-res)
python phone_camera.py --ip 10.88.111.11:8080 --width 960
```

---

## 4. Pose estimation

Pass `--pose [model]` to any source. Available models:

| Model | Flag | Notes |
|-------|------|-------|
| MediaPipe | `--pose` or `--pose mediapipe` | Default. Full-body skeleton + z-depth HUD. |
| YOLOv11-pose | `--pose yolo` | ByteTrack multi-person tracking. |
| MMPose | `--pose mmpose` | RTMPose-m, top-down, highest accuracy. |

```bash
# MediaPipe (default)
python phone_camera.py --webcam --pose

# YOLO
python phone_camera.py --webcam --pose yolo

# MMPose
python phone_camera.py --webcam --pose mmpose

# File + MMPose + record
python phone_camera.py --file clip.mp4 --pose mmpose --record --width 960
```

---

## 5. Keyboard shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot (`screenshot_<timestamp>.jpg`) |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Could not connect" | Confirm both devices are on the same Wi-Fi network |
| Black screen / no image | Open the stream URL directly in a browser to verify |
| Wrong IP | Phone IP can change; recheck it in the app each session |
| Low FPS | Reduce resolution in IP Webcam settings, or use `--width` |
| Port in use | Change port in app settings and pass `--port <new-port>` |
| `No module named 'pkg_resources'` (mmcv build) | `pip install --upgrade setuptools`, then `pip install mmcv --no-build-isolation` |
| MMPose MPS crash | Already handled — estimator forces CPU on Apple Silicon |

---

## Common stream URLs by app

| App | Default URL |
|-----|-------------|
| IP Webcam (Android) | `http://<ip>:8080/video` |
| DroidCam (Android/iOS) | `http://<ip>:4747/video` |
| IP Camera Lite (iOS) | `http://<ip>:8080/live` |
| iVCam (iOS) | Uses a desktop client — not HTTP stream |

---

## Requirements

- Python 3.11 (conda recommended)
- `opencv-python` — capture and display
- `numpy<2.0` — required by OpenCV and MediaPipe
- `mediapipe==0.10.21` — `--pose mediapipe`
- `ultralytics` — `--pose yolo`
- `mmengine`, `mmcv`, `mmdet`, `mmpose` — `--pose mmpose`
