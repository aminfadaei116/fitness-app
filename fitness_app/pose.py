import cv2
import numpy as np


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


def draw_depth_overlay(frame: np.ndarray, landmarks) -> None:
    x_offset, y_start, line_h = 10, 20, 18
    cv2.rectangle(frame, (0, 0), (170, line_h * len(_DEPTH_LANDMARKS) + 8), (0, 0, 0), -1)
    for i, (idx, name) in enumerate(_DEPTH_LANDMARKS.items()):
        lm = landmarks[idx]
        label = f"{name}: z={lm.z:+.3f}"
        cv2.putText(
            frame, label,
            (x_offset, y_start + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA,
        )


class MediaPipePoseEstimator:
    """Pose estimator backed by MediaPipe Pose."""

    def __init__(self) -> None:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        self._drawing = mp.solutions.drawing_utils
        self._mp_pose = mp_pose
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,          # 0=fast, 1=balanced, 2=accurate
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
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

    def close(self) -> None:
        self._pose.close()


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
