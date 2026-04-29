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


def draw_depth_overlay(frame: np.ndarray, landmarks) -> None:
    x_offset, y_start, line_h = 10, 20, 18
    cv2.rectangle(frame, (0, 0), (170, line_h * len(_DEPTH_LANDMARKS) + 8), (0, 0, 0), -1)
    for i, (idx, name) in enumerate(_DEPTH_LANDMARKS.items()):
        lm = landmarks[idx]
        z = lm.z if lm.z is not None else 0.0
        label = f"{name}: z={z:+.3f}"
        cv2.putText(
            frame, label,
            (x_offset, y_start + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA,
        )


class MediaPipePoseEstimator:
    """Pose estimator backed by MediaPipe Pose (legacy solutions API or Tasks PoseLandmarker)."""

    def __init__(self) -> None:
        import mediapipe as mp

        self._legacy = hasattr(mp, "solutions")
        if self._legacy:
            mp_pose = mp.solutions.pose
            self._drawing = mp.solutions.drawing_utils
            self._mp_pose = mp_pose
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = None
            self._ts_ms = 0
            self._drawing_utils = None
            self._pose_connections = None
            self._mp_image_lib = None
        else:
            self._drawing = None
            self._mp_pose = None
            self._pose = None
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.vision import drawing_utils as tasks_drawing

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
            self._drawing_utils = tasks_drawing
            self._pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
            from mediapipe.tasks.python.vision.core import image as mp_image_lib

            self._mp_image_lib = mp_image_lib

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self._legacy:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self._pose.process(rgb)
            rgb.flags.writeable = True
            out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                self._drawing.draw_landmarks(
                    out,
                    results.pose_landmarks,
                    self._mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self._drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    connection_drawing_spec=self._drawing.DrawingSpec(color=(255, 255, 0), thickness=2),
                )
                draw_depth_overlay(out, results.pose_landmarks.landmark)

            return out

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = self._mp_image_lib.Image(
            image_format=self._mp_image_lib.ImageFormat.SRGB,
            data=rgb,
        )
        self._ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._ts_ms)
        out = frame.copy()
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            self._drawing_utils.draw_landmarks(
                out,
                landmarks,
                self._pose_connections,
                landmark_drawing_spec=self._drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=self._drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=2),
            )
            draw_depth_overlay(out, landmarks)

        return out

    def close(self) -> None:
        if self._legacy:
            self._pose.close()
        elif self._landmarker is not None:
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


class MMPosePoseEstimator:
    """Pose estimator backed by MMPose via MMPoseInferencer (top-down, ViTPose-compatible)."""

    def __init__(self, pose2d: str = "human") -> None:
        from mmpose.apis import MMPoseInferencer
        # MPS backend lacks NMS op — force CPU
        self._inferencer = MMPoseInferencer(pose2d=pose2d, device="cpu")

    def process(self, frame: np.ndarray) -> np.ndarray:
        gen = self._inferencer(frame, show=False, return_vis=True)
        result = next(gen)
        vis = result.get("visualization")
        if vis:
            return cv2.cvtColor(vis[0], cv2.COLOR_RGB2BGR)
        return frame

    def close(self) -> None:
        pass


# To add a new model: implement a class with process(frame) -> frame and close(),
# then register it here.
_ESTIMATORS: dict[str, type] = {
    "mediapipe": MediaPipePoseEstimator,
    "yolo": YOLOPoseEstimator,
    "mmpose": MMPosePoseEstimator,
}


def build_estimator(name: str):
    """Instantiate a pose estimator by name."""
    if name not in _ESTIMATORS:
        available = ", ".join(_ESTIMATORS)
        raise SystemExit(f"Unknown pose model '{name}'. Available: {available}")
    estimator = _ESTIMATORS[name]()
    print(f"Pose detection ON  (model={name})")
    return estimator
