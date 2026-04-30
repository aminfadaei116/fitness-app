# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Python tool that streams a phone's MJPEG camera feed (or local video/webcam) to a desktop window via Wi-Fi and optionally overlays pose landmarks. Entry point: [phone_camera.py](phone_camera.py).

## Setup

conda is recommended (Python 3.11):

```bash
conda create -n fitness python=3.11 -y
conda activate fitness
```

Install only the backends you need:

```bash
# Core (always required)
pip install opencv-python "numpy>=1.24.0,<2.0.0"

# MediaPipe (default pose backend)
pip install mediapipe==0.10.21

# YOLO (--pose yolo)
pip install ultralytics

# MMPose (--pose mmpose) — requires openmim build order
pip install --upgrade pip setuptools wheel
pip install -U openmim
mim install mmengine
pip install mmcv --no-build-isolation
mim install "mmdet>=3.1.0" "mmpose>=1.1.0"
```

## Running

`--ip`, `--url`, `--file`, and `--webcam` are mutually exclusive; one is required.

```bash
# Phone stream
python phone_camera.py --ip 10.88.111.11:8080
python phone_camera.py --url http://10.88.111.11:8080/video

# Webcam (device index, default 0)
python phone_camera.py --webcam
python phone_camera.py --webcam 1

# Local file
python phone_camera.py --file /path/to/clip.mp4

# Pose overlay (model defaults to mediapipe)
python phone_camera.py --webcam --pose
python phone_camera.py --webcam --pose yolo
python phone_camera.py --webcam --pose mmpose

# Record full-res + resize display
python phone_camera.py --ip 10.88.111.11:8080 --record --width 960
```

Keyboard shortcuts: `q` quit, `s` screenshot.

## Datasets (Fit3D / M3GYM)

Place unpacked archives under [`datasets/`](datasets/) or set `FIT3D_ROOT` / `M3GYM_ROOT`. See [`datasets/README.md`](datasets/README.md).

Lightweight readers live in [`fitness_app/datasets/`](fitness_app/datasets/) (`Fit3DRoot`, `M3GYMRoot`). Smoke-check:

```bash
python scripts/inspect_dataset.py --dataset fit3d --limit 10
python scripts/inspect_dataset.py --dataset m3gym
```

## Architecture

[phone_camera.py](phone_camera.py) — CLI entry point (`main`). Sets `live=True` for network/webcam sources (reconnects on drop) and `live=False` for files (stops at end).

**[fitness_app/stream.py](fitness_app/stream.py)**
- `build_url` — constructs MJPEG URL, avoids double-appending port
- `open_stream` — retries `cv2.VideoCapture` up to 5× with 2 s delay
- `open_file` / `open_webcam` — open local file or webcam by device index

**[fitness_app/pose.py](fitness_app/pose.py)**
- `MediaPipePoseEstimator` — MediaPipe Pose (model_complexity=1), draws skeleton + z-depth HUD via `draw_depth_overlay`
- `YOLOPoseEstimator` — YOLOv11-pose with ByteTrack multi-person tracking; loads `yolo11n-pose.pt` (committed to repo)
- `MMPosePoseEstimator` — MMPoseInferencer (RTMPose-m); forces CPU on Apple Silicon because MPS lacks NMS ops
- `_ESTIMATORS` registry dict + `build_estimator(name)` factory — all models imported lazily

**To add a new pose model:** implement a class with `process(frame: np.ndarray) -> np.ndarray` and `close()`, then register it in `_ESTIMATORS`.

**[yolo_pose.py](yolo_pose.py)** — standalone exploration script using the Ultralytics stream API directly; not wired into the main CLI.

Recording (`--record`) always writes full-resolution frames to `output.mp4`; `--width` only downscales the display window.
