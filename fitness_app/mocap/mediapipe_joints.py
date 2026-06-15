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
    is a weak relative-depth hint scaled like ``x``. The ``z`` here is unreliable and
    corrupts the root facing/yaw — prefer ``world_to_xyz`` (metric world landmarks).
    """
    x = (lms[:, 0].astype(np.float64) - 0.5) * scale_xy
    y = -(lms[:, 1].astype(np.float64) - 0.5) * scale_xy
    z = lms[:, 2].astype(np.float64) * scale_z
    return np.stack([x, y, z], axis=1)


def world_to_xyz(lms: np.ndarray) -> np.ndarray:
    """
    Column vectors ``(33, 3)`` from MediaPipe **world** landmarks (meters, hip-centered).

    MediaPipe's world frame is X-right, Y-down, Z-toward-camera-negative. Convert to the
    pipeline's **Y-up, +Z-forward** convention (matching the rest-pose template's toe-forward
    +Z) by flipping Y and Z. Unlike ``normalized_to_xyz`` the depth is metric, so the root
    basis (right/forward, i.e. the character's facing) comes out correct instead of yaw-spun.
    """
    x = lms[:, 0].astype(np.float64)
    y = -lms[:, 1].astype(np.float64)
    z = -lms[:, 2].astype(np.float64)
    return np.stack([x, y, z], axis=1)


def landmarks_to_skeleton_joints(
    landmarks: np.ndarray,
    *,
    space: str = "image",
    scale_xy: float = 2.0,
    scale_z: float = 2.0,
) -> dict[str, np.ndarray]:
    """
    Named joints used by the simplified BVH skeleton (approximate torso from landmarks).

    ``space="world"`` treats ``landmarks`` as metric MediaPipe world landmarks (preferred);
    ``space="image"`` treats them as normalized image landmarks (legacy, weak depth).
    Returns 3-vectors in the pipeline's Y-up, +Z-forward frame.
    """
    if landmarks.shape != (33, 4):
        raise ValueError(f"expected (33, 4) landmarks, got {landmarks.shape}")
    if space == "world":
        xyz = world_to_xyz(landmarks)
    elif space == "image":
        xyz = normalized_to_xyz(landmarks, scale_xy=scale_xy, scale_z=scale_z)
    else:
        raise ValueError(f"space must be 'world' or 'image', got {space!r}")

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
