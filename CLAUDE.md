# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A single-script Python tool that streams a phone's MJPEG camera feed to a desktop window via Wi-Fi and optionally overlays MediaPipe 3-D pose landmarks. The entry point is [phone_camera.py](phone_camera.py).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requirements: `opencv-python>=4.8.0`, `numpy>=1.24.0,<2.0.0`, `mediapipe==0.10.21`

## Running

```bash
# Basic stream
python phone_camera.py --ip 10.88.111.11:8080

# With 3-D pose overlay (defaults to mediapipe)
python phone_camera.py --ip 10.88.111.11:8080 --pose
python phone_camera.py --ip 10.88.111.11:8080 --pose mediapipe

# Full URL + record + resize display
python phone_camera.py --url http://10.88.111.11:8080/video --record --width 960
```

`--ip` and `--url` are mutually exclusive and required. `--pose` takes an optional model name (default: `mediapipe`); the model is imported lazily only when `--pose` is passed.

## Architecture

[phone_camera.py](phone_camera.py) is the CLI entry point (`main`). Supporting logic lives in the `fitness_app` package:

- [fitness_app/stream.py](fitness_app/stream.py) — `build_url` (constructs MJPEG URL, avoids double-appending port) and `open_stream` (retries `cv2.VideoCapture` up to 5 times with a 2 s delay, exits on failure).
- [fitness_app/pose.py](fitness_app/pose.py) — `_DEPTH_LANDMARKS` (13 key joint indices), `draw_depth_overlay` (z-depth HUD in top-left corner), `MediaPipePoseEstimator` (encapsulates MediaPipe Pose state and processing), `_ESTIMATORS` registry dict, and `build_estimator(name)` factory.

`main` calls `build_estimator(args.pose)` to get an estimator with a `process(frame) -> frame` and `close()` interface. To add a new pose model: implement a class with those two methods and add it to `_ESTIMATORS` in [fitness_app/pose.py](fitness_app/pose.py). Recording always captures full-resolution frames; `--width` only downscales the display window.
