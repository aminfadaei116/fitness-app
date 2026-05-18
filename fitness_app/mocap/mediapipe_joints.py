"""Map BlazePose `(33, 4)` landmarks to named 3D joint positions for retargeting."""

from __future__ import annotations

import numpy as np

# BlazePose landmark indices — see MediaPipe Pose Landmarker topology
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def normalized_to_xyz(
    lms: np.ndarray,
    scale_xy: float = 2.0,
    scale_z: float = 2.0,
) -> np.ndarray:
    """
    Column vectors ``(33, 3)`` world coords: **Y-up**, X right, approximate forward +Z.

    MediaPipe stores ``x``, ``y`` normalized to image width/height [0–1]; ``z``
    is a weak relative-depth hint scaled like ``x``.
    """
    x = (lms[:, 0].astype(np.float64) - 0.5) * scale_xy
    y = -(lms[:, 1].astype(np.float64) - 0.5) * scale_xy
    z = lms[:, 2].astype(np.float64) * scale_z
    return np.stack([x, y, z], axis=1)


def landmarks_to_skeleton_joints(
    landmarks: np.ndarray,
    *,
    scale_xy: float = 2.0,
    scale_z: float = 2.0,
) -> dict[str, np.ndarray]:
    """
    Named joints used by the simplified BVH skeleton (approximate torso from landmarks).

    Returns unitless 3-vectors consistent with ``normalized_to_xyz``.
    """
    if landmarks.shape != (33, 4):
        raise ValueError(f"expected (33, 4) landmarks, got {landmarks.shape}")
    xyz = normalized_to_xyz(landmarks, scale_xy=scale_xy, scale_z=scale_z)

    lh = xyz[L_HIP]
    rh = xyz[R_HIP]
    ls = xyz[L_SHOULDER]
    rs = xyz[R_SHOULDER]
    hips = 0.5 * (lh + rh)
    chest = 0.5 * (ls + rs)
    spine = 0.5 * (hips + chest)
    neck = 0.5 * (chest + xyz[NOSE])

    joints: dict[str, np.ndarray] = {
        "Hips": hips.copy(),
        "LeftHip": lh.copy(),
        "RightHip": rh.copy(),
        "Spine": spine.copy(),
        "Spine1": chest.copy(),
        "Neck": neck.copy(),
        "Head": xyz[NOSE].copy(),
        "LeftShoulder": ls.copy(),
        "RightShoulder": rs.copy(),
        "LeftArm": xyz[L_ELBOW].copy(),
        "RightArm": xyz[R_ELBOW].copy(),
        "LeftForeArm": xyz[L_WRIST].copy(),
        "RightForeArm": xyz[R_WRIST].copy(),
        "LeftHand": xyz[L_WRIST].copy(),
        "RightHand": xyz[R_WRIST].copy(),
        "LeftUpLeg": 0.5 * (hips + lh),
        "RightUpLeg": 0.5 * (hips + rh),
        "LeftLeg": xyz[L_KNEE].copy(),
        "RightLeg": xyz[R_KNEE].copy(),
        "LeftFoot": xyz[L_ANKLE].copy(),
        "RightFoot": xyz[R_ANKLE].copy(),
    }

    toe_down = np.array([0.0, -0.06, 0.02], dtype=np.float64)
    joints["LeftToeEnd"] = joints["LeftFoot"] + toe_down
    joints["RightToeEnd"] = joints["RightFoot"] + toe_down

    return joints


def skeleton_joint_names_ordered() -> list[str]:
    """BVH DFS order matching ``SkeletonRig`` hierarchy."""
    return [
        "Hips",
        "Spine",
        "Spine1",
        "Neck",
        "Head",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToeEnd",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "RightToeEnd",
    ]
