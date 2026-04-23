# Fitness App — Phone Camera Viewer

Stream your phone's camera to your Mac/PC in real-time over Wi-Fi using Python and OpenCV.

---

## How it works

Your phone runs a free app that broadcasts its camera as an MJPEG HTTP stream on your local network. The Python script connects to that stream and displays it in a window.

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

## 2. Set up Python

```bash
# Create a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Run the viewer

```bash
# Basic — just pass the phone's IP
python phone_camera.py --ip 10.88.111.11:8080

# Custom port
python phone_camera.py --ip 10.88.111.11:8080 --port 4747

# Full URL (useful for apps that use a different path)
python phone_camera.py --url http://10.88.111.11:8080/video

# Record to output.mp4 while viewing
python phone_camera.py --ip 10.88.111.11:8080 --record

# Resize the display window (keep original resolution when recording)
python phone_camera.py --ip 10.88.111.11:8080 --width 960
```

### Pose estimation

Pass `--pose` to overlay 3-D skeleton landmarks on the stream. An optional model name selects the backend (default: `mediapipe`).

```bash
# Enable pose estimation (uses mediapipe by default)
python phone_camera.py --ip 10.88.111.11:8080 --pose

# Explicitly select a model
python phone_camera.py --ip 10.88.111.11:8080 --pose mediapipe

# Combine with other flags
python phone_camera.py --ip 10.88.111.11:8080 --pose --record --width 960
```

The MediaPipe backend draws a full-body skeleton and a real-time z-depth HUD for 13 key joints. The model is loaded lazily — only when `--pose` is passed.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save a screenshot (saved as `screenshot_<timestamp>.jpg`) |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Could not connect" | Confirm both devices are on the same Wi-Fi network |
| Black screen / no image | Try the full URL from the app's screen directly in your browser first |
| Wrong IP | Your phone's IP can change; recheck it in the app each session |
| Low FPS | Reduce resolution in the IP Webcam app settings, or use `--width` to downscale display |
| Port already in use | Change port in the app settings and pass `--port <new-port>` |

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

- Python 3.8+
- `opencv-python` — for capture and display
- `numpy` — required by OpenCV
- `mediapipe` — required only when using `--pose mediapipe`
