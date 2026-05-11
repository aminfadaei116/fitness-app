# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Python tool that streams a phone's MJPEG camera feed (or local video / webcam) to a desktop window via Wi-Fi and optionally overlays pose landmarks. Entry point: [phone_camera.py](phone_camera.py).

## Setup

Conda recommended (Python 3.11):

```bash
conda create -n fitness python=3.11 -y
conda activate fitness
```

Install only the backends needed:

```bash
# Core (always required)
pip install opencv-python "numpy>=1.24.0,<2.0.0"

# MediaPipe (default pose backend, Tasks PoseLandmarker)
pip install "mediapipe>=0.10.30,<0.11"

# YOLO (--pose yolo) — yolo11n-pose.pt committed at repo root
pip install ultralytics

# MMPose (--pose mmpose) — requires openmim build order
pip install --upgrade pip setuptools wheel
pip install -U openmim
mim install mmengine
pip install mmcv --no-build-isolation
mim install "mmdet>=3.1.0" "mmpose>=1.1.0"
```

Python 3.13: [requirements.txt](requirements.txt) switches to `numpy>=2.0` and `mediapipe>=0.10.30` (Tasks API). Pose code currently uses legacy `mp.solutions.pose` — verify before bumping.

## Running

`--ip`, `--url`, `--file`, `--webcam` are mutually exclusive; one required.

```bash
python phone_camera.py --ip 10.88.111.11:8080
python phone_camera.py --url http://10.88.111.11:8080/video
python phone_camera.py --webcam            # device 0
python phone_camera.py --webcam 1
python phone_camera.py --file /path/to/clip.mp4

# Pose overlay (model defaults to mediapipe)
python phone_camera.py --webcam --pose
python phone_camera.py --webcam --pose yolo
python phone_camera.py --webcam --pose mmpose

# Record full-res + downscale display
python phone_camera.py --ip 10.88.111.11:8080 --record --width 960
```

Keys: `q` quit, `s` screenshot.

## Tests

```bash
python -m unittest discover -s tests                                                   # all
python -m unittest tests.test_dataset_registry                                         # one module
python -m unittest tests.test_dataset_registry.DatasetRegistryTests.test_registry_keys # one test
```

## Coaching (experimental)

Heuristic squat feedback uses MediaPipe landmarks (joint angles + phase machine). Requires upright-ish camera; not medical advice.

```bash
python phone_camera.py --webcam --coach squat
```

Uses `--pose mediapipe` automatically. `--coach` is incompatible with `--pose yolo` or `--pose mmpose`.

## Architecture

[phone_camera.py](phone_camera.py) — CLI entry (`main`). Sets `live=True` for network/webcam (reconnect loop on read failure) and `live=False` for files (stop at EOF). Recording always writes the full-res frame; `--width` only resizes the display window.

**[fitness_app/stream.py](fitness_app/stream.py)**
- `build_url` — constructs MJPEG URL, avoids double-appending port
- `open_stream` — retries `cv2.VideoCapture` up to 5× with 2 s delay
- `open_file` / `open_webcam` — local file or webcam by device index

**[fitness_app/pose.py](fitness_app/pose.py)**
- `MediaPipePoseEstimator` — MediaPipe Tasks `PoseLandmarker` (VIDEO mode); auto-downloads `pose_landmarker_lite.task` to `~/.cache/fitness-app/`; draws skeleton + z-depth HUD via `draw_depth_overlay`; sets **`last_landmarks`** `(33, 4)` array `x,y,z,visibility` when a pose is found (for coaching).
- `YOLOPoseEstimator` — YOLOv11-pose with ByteTrack multi-person tracking; loads `yolo11n-pose.pt` (committed to repo)
- `MMPosePoseEstimator` — MMPoseInferencer (RTMPose-m); forces CPU on Apple Silicon because MPS lacks NMS ops
- `_ESTIMATORS` registry dict + `build_estimator(name)` factory — all models imported lazily

**[fitness_app/coaching/](fitness_app/coaching/)**
- `geometry`, `squat_features`, `squat_phase`, `squat_rules`, `squat_coach` — heuristic squat cues; **`--coach squat`** in `phone_camera.py`. See [`fitness_app/coaching/README.md`](fitness_app/coaching/README.md) for extension notes.

**To add a new pose model:** implement a class with `process(frame: np.ndarray) -> np.ndarray` and `close()`, then register it in `_ESTIMATORS`.

[yolo_pose.py](yolo_pose.py) — standalone Ultralytics-stream-API exploration; not wired into the CLI.

## Datasets registry

Path-only registry for unpacked public pose corpora — no downloads, no parsers. Place archives under [datasets/](datasets/) `<slug>/` or override per-dataset with env vars (table in [datasets/README.md](datasets/README.md)).

Module split in [fitness_app/datasets/](fitness_app/datasets/):
- `base.py` — `DatasetRoot` ABC, `DATASET_REGISTRY`, `register_dataset`, `get_dataset(name, root=None)`, `list_dataset_names`. `validate()` only checks the root directory exists.
- `paths.py` — `repo_root`, `default_dataset_root`, `resolve_dataset_root` (env var → default), `require_exists`.
- `public_datasets.py` — concrete `DatasetRoot` subclasses; imported for side-effect registration from package `__init__.py`.

Registry keys: `coco_keypoints`, `mpii_human_pose`, `yoga_pose`, `exercise_skeleton`, `human36m`.

```python
from pathlib import Path
from fitness_app.datasets import get_dataset

get_dataset("coco_keypoints").validate()
get_dataset("coco_keypoints", root=Path("/custom/path")).validate()
```
