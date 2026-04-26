import sys
import time
import cv2


def build_url(ip: str, port: int) -> str:
    # If the user already passed host:port, don't append port again
    if ":" in ip:
        return f"http://{ip}/video"
    return f"http://{ip}:{port}/video"


def open_stream(url: str, retries: int = 5, delay: float = 2.0) -> cv2.VideoCapture:
    for attempt in range(1, retries + 1):
        print(f"Connecting to {url}  (attempt {attempt}/{retries})...")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print("Connected.")
            return cap
        cap.release()
        if attempt < retries:
            time.sleep(delay)
    sys.exit(f"Could not connect to {url}. Check the IP and that the camera app is running.")


def open_file(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Could not open video file: {path}")
    print(f"Opened file: {path}")
    return cap


def open_webcam(device_id: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        sys.exit(f"Could not open webcam (device {device_id}).")
    print(f"Opened webcam (device {device_id}).")
    return cap
