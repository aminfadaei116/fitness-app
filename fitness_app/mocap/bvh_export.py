"""Build a simplified humanoid BVH from per-frame named joint positions (Y-up, arbitrary units)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from fitness_app.mocap.mediapipe_joints import skeleton_joint_names_ordered


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_from_two_unit_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton quaternion ``q = w + xi + yj + zk`` rotating unit vector ``a`` onto ``b``."""
    a = _normalize(a)
    b = _normalize(b)
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if d > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if d < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(axis)) < 1e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = _normalize(axis)
        return quat_normalize(np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64))
    axis = np.cross(a, b)
    s = np.sqrt((1.0 + d) * 2.0)
    inv = 1.0 / s
    return quat_normalize(
        np.array([0.5 * s, axis[0] * inv, axis[1] * inv, axis[2] * inv], dtype=np.float64)
    )


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_deg_zyx_from_mat(R: np.ndarray) -> np.ndarray:
    """
    Extrinsic-ish extraction matching channel order ``Zrotation Yrotation Xrotation`` as
    ``R ≈ Rz(rz) @ Ry(ry) @ Rx(rx)`` with angles in **degrees** (common BVH convention).
    """
    m = R.reshape(3, 3)
    sy = math.sqrt(m[0, 0] * m[0, 0] + m[1, 0] * m[1, 0])
    singular = sy < 1e-8
    if not singular:
        rx = math.atan2(m[2, 1], m[2, 2])
        ry = math.atan2(-m[2, 0], sy)
        rz = math.atan2(m[1, 0], m[0, 0])
    else:
        rx = math.atan2(-m[1, 2], m[1, 1])
        ry = math.atan2(-m[2, 0], sy)
        rz = 0.0
    return np.degrees(np.array([rz, ry, rx], dtype=np.float64))


# --------------------------------------------------------------------------- rig template
def _default_tpose_offsets(scale: float) -> dict[str, np.ndarray]:
    """Proportional T-pose (Y-up): arms along ±X, legs −Y, head +Y."""
    s = float(scale)
    return {
        "Hips": np.zeros(3, dtype=np.float64),
        "Spine": np.array([0.0, 0.12 * s, 0.0], dtype=np.float64),
        "Spine1": np.array([0.0, 0.12 * s, 0.0], dtype=np.float64),
        "Neck": np.array([0.0, 0.10 * s, 0.0], dtype=np.float64),
        "Head": np.array([0.0, 0.10 * s, 0.0], dtype=np.float64),
        "LeftShoulder": np.array([-0.06 * s, 0.08 * s, 0.0], dtype=np.float64),
        "LeftArm": np.array([-0.14 * s, 0.0, 0.0], dtype=np.float64),
        "LeftForeArm": np.array([-0.12 * s, 0.0, 0.0], dtype=np.float64),
        "LeftHand": np.array([-0.10 * s, 0.0, 0.0], dtype=np.float64),
        "RightShoulder": np.array([0.06 * s, 0.08 * s, 0.0], dtype=np.float64),
        "RightArm": np.array([0.14 * s, 0.0, 0.0], dtype=np.float64),
        "RightForeArm": np.array([0.12 * s, 0.0, 0.0], dtype=np.float64),
        "RightHand": np.array([0.10 * s, 0.0, 0.0], dtype=np.float64),
        "LeftUpLeg": np.array([-0.08 * s, -0.10 * s, 0.0], dtype=np.float64),
        "LeftLeg": np.array([0.0, -0.12 * s, 0.0], dtype=np.float64),
        "LeftFoot": np.array([0.0, -0.12 * s, 0.0], dtype=np.float64),
        "LeftToeEnd": np.array([0.02 * s, -0.05 * s, 0.04 * s], dtype=np.float64),
        "RightUpLeg": np.array([0.08 * s, -0.10 * s, 0.0], dtype=np.float64),
        "RightLeg": np.array([0.0, -0.12 * s, 0.0], dtype=np.float64),
        "RightFoot": np.array([0.0, -0.12 * s, 0.0], dtype=np.float64),
        "RightToeEnd": np.array([-0.02 * s, -0.05 * s, 0.04 * s], dtype=np.float64),
    }


def _median_scale_from_joints(joints_seq: list[dict[str, np.ndarray]]) -> float:
    lengths: list[float] = []
    pairs = [
        ("Hips", "Spine1"),
        ("LeftShoulder", "LeftArm"),
        ("LeftArm", "LeftForeArm"),
        ("RightShoulder", "RightArm"),
        ("LeftUpLeg", "LeftLeg"),
        ("RightUpLeg", "RightLeg"),
    ]
    for jf in joints_seq:
        for a, b in pairs:
            if a in jf and b in jf:
                lengths.append(float(np.linalg.norm(jf[b] - jf[a])))
    if not lengths:
        return 1.0
    return float(np.median(np.array(lengths, dtype=np.float64)))


def _hierarchy_children() -> dict[str | None, list[str]]:
    return {
        "Hips": ["Spine", "LeftUpLeg", "RightUpLeg"],
        "Spine": ["Spine1"],
        "Spine1": ["Neck", "LeftShoulder", "RightShoulder"],
        "Neck": ["Head"],
        "Head": [],
        "LeftShoulder": ["LeftArm"],
        "LeftArm": ["LeftForeArm"],
        "LeftForeArm": ["LeftHand"],
        "LeftHand": [],
        "RightShoulder": ["RightArm"],
        "RightArm": ["RightForeArm"],
        "RightForeArm": ["RightHand"],
        "RightHand": [],
        "LeftUpLeg": ["LeftLeg"],
        "LeftLeg": ["LeftFoot"],
        "LeftFoot": ["LeftToeEnd"],
        "LeftToeEnd": [],
        "RightUpLeg": ["RightLeg"],
        "RightLeg": ["RightFoot"],
        "RightFoot": ["RightToeEnd"],
        "RightToeEnd": [],
    }


def _emit_joint(
    name: str,
    offsets: dict[str, np.ndarray],
    children_map: dict[str | None, list[str]],
    indent: int,
) -> list[str]:
    pad = "  " * indent
    off = offsets[name]
    lines: list[str] = []
    kids = children_map.get(name, [])
    end_names = {"Head", "LeftHand", "RightHand", "LeftToeEnd", "RightToeEnd"}
    if name == "Hips":
        lines.append(f"{pad}ROOT {name}")
    else:
        lines.append(f"{pad}JOINT {name}")
    lines.append(f"{pad}{{")
    lines.append(f"{pad}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
    if name == "Hips":
        lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
    else:
        lines.append(f"{pad}  CHANNELS 3 Zrotation Yrotation Xrotation")
    if not kids and name in end_names:
        lines.append(f"{pad}  End Site")
        lines.append(f"{pad}  {{")
        lines.append(f"{pad}    OFFSET 0.0 0.05 0.0")
        lines.append(f"{pad}  }}")
    for ch in kids:
        lines.extend(_emit_joint(ch, offsets, children_map, indent + 1))
    lines.append(f"{pad}}}")
    return lines


def build_bvh_hierarchy_string(offsets: dict[str, np.ndarray]) -> str:
    """Return HIERARCHY block only (no MOTION)."""
    lines = ["HIERARCHY"]
    lines.extend(_emit_joint("Hips", offsets, _hierarchy_children(), 0))
    return "\n".join(lines) + "\n"


def _root_basis_from_joints(J: dict[str, np.ndarray]) -> np.ndarray:
    """3x3 matrix with columns [right, up, forward] mapping character -> world."""
    right = _normalize(J["RightHip"] - J["LeftHip"])
    up = _normalize(J["Spine1"] - J["Hips"])
    forward = _normalize(np.cross(right, up))
    right = _normalize(np.cross(up, forward))
    return np.stack([right, up, forward], axis=1)


def _compute_frame_rotations(
    J: dict[str, np.ndarray],
    offsets: dict[str, np.ndarray],
    R0: np.ndarray,
    origin0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns root translation (3,) and flat euler degrees for all joints in DFS order
    (root 6: x y z rz ry rx, then each joint 3: rz ry rx) — **channel order matches file**.
    """
    children = _hierarchy_children()
    trans = J["Hips"] - origin0

    R_root_world = _root_basis_from_joints(J) @ R0.T
    e_root = euler_deg_zyx_from_mat(R_root_world)
    root_line = np.concatenate([trans, e_root])

    world_R: dict[str, np.ndarray] = {"Hips": R_root_world}

    local_euler: dict[str, np.ndarray] = {}

    def visit(name: str, parent: str | None) -> None:
        if parent is None:
            return
        o = offsets[name]
        rest_dir = _normalize(o)
        parent_R = world_R[parent]
        target_w = _normalize(J[name] - J[parent])
        local_target = parent_R.T @ target_w
        q = quat_from_two_unit_vectors(rest_dir, local_target)
        R_local = quat_to_mat(q)
        world_R[name] = parent_R @ R_local
        local_euler[name] = euler_deg_zyx_from_mat(R_local)

    # BFS / DFS from Hips
    stack: list[tuple[str, str | None]] = [("Hips", None)]
    order: list[str] = []
    while stack:
        name, par = stack.pop(0)
        order.append(name)
        for ch in children.get(name, []):
            stack.append((ch, name))
    # skip Hips in local pass
    for name in order[1:]:
        par = _parent_of(name, children)
        visit(name, par)

    parts: list[np.ndarray] = [root_line]
    for name in skeleton_joint_names_ordered()[1:]:
        parts.append(local_euler[name])
    return trans, np.concatenate(parts)


def _parent_of(name: str, children: dict[str | None, list[str]]) -> str | None:
    for p, ks in children.items():
        if p is not None and name in ks:
            return p
    return None


def _pick_reference_skeleton(joints_sequence: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    for J in joints_sequence:
        torso = float(np.linalg.norm(J["Spine1"] - J["Hips"]))
        if torso > 1e-4:
            return J
    return joints_sequence[0]


def build_animation_from_joints_list(
    joints_sequence: Iterable[dict[str, np.ndarray]],
    fps: float,
) -> tuple[str, np.ndarray]:
    """
    Full BVH file body: hierarchy string + motion array shape ``(n_frames, n_channels)``.

    Root translation is **relative to the reference frame** hips position. Root rotation is
    relative to the reference torso basis (first frame with a sane torso span).
    """
    seq = list(joints_sequence)
    if not seq:
        raise ValueError("empty joint sequence")
    scale = max(_median_scale_from_joints(seq), 0.05)
    offsets = _default_tpose_offsets(scale / 0.14)
    hierarchy = build_bvh_hierarchy_string(offsets)

    J0 = _pick_reference_skeleton(seq)
    origin0 = J0["Hips"].copy()
    R0 = _root_basis_from_joints(J0)

    rows: list[np.ndarray] = []
    for J in seq:
        _, row = _compute_frame_rotations(J, offsets, R0, origin0)
        rows.append(row)

    motion = np.stack(rows, axis=0)
    return hierarchy, motion


def write_bvh(
    path: str | Path,
    joints_sequence: Iterable[dict[str, np.ndarray]],
    fps: float,
) -> None:
    """Write a BVH file with ZYX Euler degrees (see module docstring)."""
    hierarchy, motion = build_animation_from_joints_list(joints_sequence, fps)
    n_frames, n_ch = motion.shape
    frame_time = 1.0 / max(float(fps), 1e-6)
    lines = [hierarchy.rstrip(), "MOTION", f"Frames: {n_frames}", f"Frame Time: {frame_time:.6f}"]
    for i in range(n_frames):
        lines.append(" ".join(f"{v:.6f}" for v in motion[i]))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
