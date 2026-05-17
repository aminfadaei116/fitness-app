import cv2
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve


# Key landmark indices to display z-depth for (MediaPipe Pose)
_DEPTH_LANDMARKS = {
    0:  "nose",
    11: "L-shoulder",
    12: "R-shoulder",
    13: "L-elbow",
    14: "R-elbow",
    15: "L-wrist",
    16: "R-wrist",
    23: "L-hip",
    24: "R-hip",
    25: "L-knee",
    26: "R-knee",
    27: "L-ankle",
    28: "R-ankle",
}


_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ensure_pose_landmarker_model() -> str:
    cache_dir = Path.home() / ".cache" / "fitness-app"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "pose_landmarker_lite.task"
    if not dest.is_file():
        print(f"Downloading pose landmarker model to {dest} ...")
        urlretrieve(_POSE_MODEL_URL, dest)
    return str(dest)


def landmarks_to_array(landmarks_list) -> np.ndarray:
    """Build ``(33, 4)`` array ``x, y, z, visibility`` from Tasks ``NormalizedLandmark`` sequence."""
    out = np.zeros((33, 4), dtype=np.float64)
    for i in range(33):
        lm = landmarks_list[i]
        vis = lm.visibility if lm.visibility is not None else 1.0
        out[i] = (
            lm.x if lm.x is not None else 0.0,
            lm.y if lm.y is not None else 0.0,
            lm.z if lm.z is not None else 0.0,
            float(vis),
        )
    return out


_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def draw_pose_from_array(frame: np.ndarray, landmarks: np.ndarray, visibility_threshold: float = 0.5) -> None:
    """Draw pose skeleton onto *frame* in-place from a ``(33, 4)`` x/y/z/vis array."""
    h, w = frame.shape[:2]
    pts = []
    for i in range(33):
        x, y, _, vis = landmarks[i]
        if vis >= visibility_threshold:
            pts.append((int(x * w), int(y * h)))
        else:
            pts.append(None)

    for a, b in _POSE_CONNECTIONS:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(frame, pts[a], pts[b], (255, 255, 0), 2, cv2.LINE_AA)

    for pt in pts:
        if pt is not None:
            cv2.circle(frame, pt, 3, (0, 255, 0), -1, cv2.LINE_AA)


def draw_depth_overlay(frame: np.ndarray, landmarks: np.ndarray) -> None:
    """Draw z-depth HUD from a ``(33, 4)`` landmarks array."""
    x_offset, y_start, line_h = 10, 20, 18
    cv2.rectangle(frame, (0, 0), (170, line_h * len(_DEPTH_LANDMARKS) + 8), (0, 0, 0), -1)
    for i, (idx, name) in enumerate(_DEPTH_LANDMARKS.items()):
        z = float(landmarks[idx, 2])
        label = f"{name}: z={z:+.3f}"
        cv2.putText(
            frame, label,
            (x_offset, y_start + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA,
        )


class MediaPipePoseEstimator:
    """Pose estimator backed by MediaPipe Tasks ``PoseLandmarker`` (VIDEO mode)."""

    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        model_path = _ensure_pose_landmarker_model()
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._ts_ms = 0
        self.last_landmarks: np.ndarray | None = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        import mediapipe as mp

        self.last_landmarks = None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._ts_ms)
        out = frame.copy()
        if result.pose_landmarks:
            self.last_landmarks = landmarks_to_array(result.pose_landmarks[0])
            draw_pose_from_array(out, self.last_landmarks)
            draw_depth_overlay(out, self.last_landmarks)

        return out

    def close(self) -> None:
        self._landmarker.close()


class YOLOPoseEstimator:
    """Pose estimator backed by YOLOv11-pose with ByteTrack."""

    def __init__(self, model_path: str = "yolo11n-pose.pt") -> None:
        from ultralytics import YOLO
        self._model = YOLO(model_path)

    def process(self, frame: np.ndarray) -> np.ndarray:
        results = self._model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        if results and results[0] is not None:
            return results[0].plot()
        return frame

    def close(self) -> None:
        pass


# To add a new model: implement a class with process(frame) -> frame and close(),
# then register it here.
_ESTIMATORS: dict[str, type] = {
    "mediapipe": MediaPipePoseEstimator,
    "yolo": YOLOPoseEstimator,
}


def build_estimator(name: str):
    """Instantiate a pose estimator by name."""
    if name not in _ESTIMATORS:
        available = ", ".join(_ESTIMATORS)
        raise SystemExit(f"Unknown pose model '{name}'. Available: {available}")
    estimator = _ESTIMATORS[name]()
    print(f"Pose detection ON  (model={name})")
    return estimator
