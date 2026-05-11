"""Image-plane geometry on MediaPipe BlazePose topology (33 landmarks).

Angles assume an upright-ish camera (vertical is negative image Y).

Landmark indices match MediaPipe Pose:
https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
"""

from __future__ import annotations

import numpy as np

# BlazePose subset used by squat coaching
NOSE = 0
L_SHOULDER = 11
R_SHOULDER = 12
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28


def pixel_xy(lms: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """Denormalize x,y columns of ``lms`` shape ``(33, 4+)`` to pixel coords."""
    xy = np.zeros((33, 2), dtype=np.float64)
    xy[:, 0] = np.clip(lms[:, 0], 0.0, 1.0) * frame_w
    xy[:, 1] = np.clip(lms[:, 1], 0.0, 1.0) * frame_h
    return xy


def midpoint_xy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint of two (x,y) points."""
    return 0.5 * (a + b)


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle ABC at vertex B in degrees (image plane)."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-8 or nbc < 1e-8:
        return float("nan")
    cos_t = float(np.dot(ba, bc) / (nba * nbc))
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def torso_vertical_deg(mid_hip: np.ndarray, mid_shoulder: np.ndarray) -> float:
    """Angle (deg) between hip->shoulder vector and image \"up\" (0, -1).

    ~0 when torso is vertical in the image; larger when leaning forward.
    """
    v = mid_shoulder - mid_hip
    n = np.linalg.norm(v)
    if n < 1e-8:
        return float("nan")
    up = np.array([0.0, -1.0], dtype=np.float64)
    cos_t = float(np.dot(v, up) / n)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def symmetry_delta(left_angle: float, right_angle: float) -> float:
    return abs(left_angle - right_angle)


def normalize_xy_about_center(xy: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Translate by center and divide by ``scale`` (add eps)."""
    if scale < 1e-8:
        return xy - center
    return (xy - center) / scale


def shoulder_width_px(xy: np.ndarray) -> float:
    return float(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER]))


def hip_width_px(xy: np.ndarray) -> float:
    return float(np.linalg.norm(xy[L_HIP] - xy[R_HIP]))
