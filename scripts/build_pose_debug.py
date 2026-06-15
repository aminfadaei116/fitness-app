#!/usr/bin/env python3
"""Build a debug bundle (image + single-frame BVH + landmark JSON) for the web debug page.

Runs the real (world-landmark) mocap pipeline on a single still image and writes the result
to ``web/public/debug/`` so the web "Debug" view can show the 2D image landmarks next to the
3D character they drive, with hover-linked correspondence.

    python scripts/build_pose_debug.py datasets/TestPose/pose_model.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root on path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

from fitness_app.pose import (
    _ensure_pose_landmarker_model,
    landmarks_to_array,
    world_landmarks_to_array,
    _POSE_CONNECTIONS,
)
from fitness_app.mocap.mediapipe_joints import landmarks_to_skeleton_joints
from fitness_app.mocap.bvh_export import build_animation_from_joints_list

# BlazePose 33-landmark names (subset that matters labelled; rest generic).
LANDMARK_NAMES = {
    0: "nose", 11: "L-shoulder", 12: "R-shoulder", 13: "L-elbow", 14: "R-elbow",
    15: "L-wrist", 16: "R-wrist", 23: "L-hip", 24: "R-hip", 25: "L-knee",
    26: "R-knee", 27: "L-ankle", 28: "R-ankle",
}
# Which BVH joint each key landmark most directly drives (for the hover correspondence).
LANDMARK_TO_BVH = {
    0: "Head", 11: "LeftShoulder", 12: "RightShoulder", 13: "LeftArm", 14: "RightArm",
    15: "LeftForeArm", 16: "RightForeArm", 23: "LeftUpLeg", 24: "RightUpLeg",
    25: "LeftLeg", 26: "RightLeg", 27: "LeftFoot", 28: "RightFoot",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", default="datasets/TestPose/pose_model.png")
    ap.add_argument("--max-size", type=int, default=640, help="Resize longest image side to this many px")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        raise SystemExit(f"No such image: {img_path}")

    out_dir = Path("web/public/debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- MediaPipe (single image) ---
    opts = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_ensure_pose_landmarker_model()),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)
    img = cv2.imread(str(img_path))
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        raise SystemExit("No pose detected in image")

    norm = landmarks_to_array(res.pose_landmarks[0])          # (33,4) normalized image space
    world = world_landmarks_to_array(res.pose_world_landmarks[0])  # (33,4) metric

    # --- single-frame BVH via the world path (matches the real export) ---
    joints = landmarks_to_skeleton_joints(world, space="world")
    hierarchy, motion = build_animation_from_joints_list([joints], fps=25.0)
    bvh = hierarchy.rstrip() + "\nMOTION\nFrames: 1\nFrame Time: 0.040000\n" + \
        " ".join(f"{v:.6f}" for v in motion[0]) + "\n"
    (out_dir / "pose.bvh").write_text(bvh, encoding="utf-8")

    # --- resized image ---
    h, w = img.shape[:2]
    scale = min(1.0, args.max_size / max(h, w))
    disp = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
    cv2.imwrite(str(out_dir / "image.png"), disp)

    # --- landmark JSON (normalized coords so the web scales to the shown image) ---
    points = [[float(norm[i, 0]), float(norm[i, 1]), float(norm[i, 3])] for i in range(33)]
    bundle = {
        "image": "/debug/image.png",
        "bvh": "/debug/pose.bvh",
        "width": disp.shape[1],
        "height": disp.shape[0],
        "points": points,                       # [x_norm, y_norm, visibility] x33
        "connections": _POSE_CONNECTIONS,
        "names": LANDMARK_NAMES,
        "landmarkToBvh": LANDMARK_TO_BVH,
    }
    (out_dir / "landmarks.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}/image.png, pose.bvh, landmarks.json  (from {img_path})")


if __name__ == "__main__":
    main()
