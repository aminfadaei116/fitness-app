"""Squat-specific scalar features from normalized landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fitness_app.coaching.geometry import (
    L_ANKLE,
    L_HIP,
    L_KNEE,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    angle_deg,
    hip_width_px,
    midpoint_xy,
    pixel_xy,
    torso_vertical_deg,
)


@dataclass(frozen=True)
class SquatFeatures:
    knee_flexion_left_deg: float
    knee_flexion_right_deg: float
    mean_knee_flexion_deg: float
    torso_vs_vertical_deg: float
    knee_ankle_dx_norm_left: float
    knee_ankle_dx_norm_right: float
    min_knee_visibility: float


def compute_squat_features(lms: np.ndarray, frame_w: int, frame_h: int) -> SquatFeatures:
    """Derive squat cues from ``lms`` shape ``(33, 4)`` (x,y,z,visibility).

    Knee flexion = 180° - interior angle hip-knee-ankle (~0 standing, larger when bent).

    knee_ankle_dx_norm_* is (knee_x - ankle_x) / hip_width (weak sagittal heuristic;
    interpretation depends heavily on camera azimuth).
    """
    xy = pixel_xy(lms, frame_w, frame_h)

    lk = xy[L_KNEE]
    rk = xy[R_KNEE]
    lh = xy[L_HIP]
    rh = xy[R_HIP]
    la = xy[L_ANKLE]
    ra = xy[R_ANKLE]

    int_l = angle_deg(lh, lk, la)
    int_r = angle_deg(rh, rk, ra)
    flex_l = 180.0 - int_l if not np.isnan(int_l) else float("nan")
    flex_r = 180.0 - int_r if not np.isnan(int_r) else float("nan")

    if np.isnan(flex_l) and np.isnan(flex_r):
        mean_f = float("nan")
    elif np.isnan(flex_l):
        mean_f = flex_r
    elif np.isnan(flex_r):
        mean_f = flex_l
    else:
        mean_f = 0.5 * (flex_l + flex_r)

    mid_h = midpoint_xy(lh, rh)
    mid_s = midpoint_xy(xy[11], xy[12])
    torso_v = torso_vertical_deg(mid_h, mid_s)

    hw = hip_width_px(xy)
    if hw < 1e-6:
        hw = 1.0
    knee_ankle_l = (lk[0] - la[0]) / hw
    knee_ankle_r = (rk[0] - ra[0]) / hw

    vis_knees = min(float(lms[L_KNEE, 3]), float(lms[R_KNEE, 3]))

    return SquatFeatures(
        knee_flexion_left_deg=flex_l,
        knee_flexion_right_deg=flex_r,
        mean_knee_flexion_deg=mean_f,
        torso_vs_vertical_deg=torso_v,
        knee_ankle_dx_norm_left=knee_ankle_l,
        knee_ankle_dx_norm_right=knee_ankle_r,
        min_knee_visibility=vis_knees,
    )
