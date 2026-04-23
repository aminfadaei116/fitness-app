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
    """Print z-depth values for key joints as a HUD in the top-left corner."""
    x_offset, y_start, line_h = 10, 20, 18
    cv2.rectangle(frame, (0, 0), (170, line_h * len(_DEPTH_LANDMARKS) + 8), (0, 0, 0), -1)
    for i, (idx, name) in enumerate(_DEPTH_LANDMARKS.items()):
        lm = landmarks[idx]
        # z is in the same scale as x/y (normalized, relative to hip midpoint)
        # negative = toward camera
        label = f"{name}: z={lm.z:+.3f}"
        cv2.putText(
            frame, label,
            (x_offset, y_start + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA,
        )


def run_pose(frame: np.ndarray, pose, drawing, mp_pose) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        drawing.draw_landmarks(
            out,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 0), thickness=2),
        )
        draw_depth_overlay(out, results.pose_landmarks.landmark)

    return out
