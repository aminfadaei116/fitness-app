#!/usr/bin/env python3
"""Kalman smoothing + BlazePose landmarks to simplified humanoid BVH."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from fitness_app.mocap.bvh_export import write_bvh
from fitness_app.mocap.kalman_smooth import smooth_landmark_sequence
from fitness_app.mocap.mediapipe_joints import landmarks_to_skeleton_joints


def _resolve_input(path: Path) -> tuple[Path, Path]:
    """Return ``(path_to_keypoints_pkl, directory_for_default_output)``."""
    if path.is_dir():
        pkl = path / "keypoints.pkl"
        default_dir = path
        if not pkl.is_file():
            raise SystemExit(f"Missing keypoints.pkl in directory {path}")
        return pkl, default_dir
    if path.is_file():
        if path.suffix.lower() != ".pkl":
            raise SystemExit(f"Not a pickle file or directory: {path}")
        return path, path.parent
    raise SystemExit(f"No such path: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        metavar="PATH",
        required=True,
        help="Sample directory with keypoints.pkl or direct path to keypoints.pkl",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Output ``.bvh`` (default: ``motion.bvh`` next to the pickle)",
    )
    parser.add_argument("--fps-override", type=float, default=None, help="Override FPS from pickle metadata")
    parser.add_argument("--no-kalman", action="store_true", help="Skip filtering (still retarget + BVH)")
    parser.add_argument("--visibility-threshold", type=float, default=0.35, help="Kalman measurement gate")
    parser.add_argument("--mahalanobis-max", type=float, default=25.0, help="Innovation rejection cutoff")
    args = parser.parse_args()

    kp_path, out_parent = _resolve_input(Path(args.input))
    out_path = Path(args.output) if args.output else out_parent / "motion.bvh"

    with kp_path.open("rb") as f:
        data = pickle.load(f)

    raw: list = data["keypoints"]
    fps = float(args.fps_override if args.fps_override is not None else data["fps"])
    if not raw:
        raise SystemExit("keypoints sequence is empty")

    if args.no_kalman:
        smoothed_frames: list = []
        for lm in raw:
            if lm is None:
                smoothed_frames.append(np.zeros((33, 4), dtype=np.float64))
            else:
                smoothed_frames.append(lm.astype(np.float64).copy())
    else:
        smoothed_frames = smooth_landmark_sequence(
            raw,
            fps=fps,
            visibility_threshold=args.visibility_threshold,
            mahalanobis_max=args.mahalanobis_max,
        )

    joints_sequence = [landmarks_to_skeleton_joints(lm) for lm in smoothed_frames]

    write_bvh(out_path, joints_sequence, fps=fps)

    n = len(smoothed_frames)
    pose_frames = sum(1 for lm in raw if lm is not None)
    pct = 100.0 * pose_frames / n if n else 0.0
    print(
        f"wrote {out_path.resolve()} "
        f"({n} frames, {pct:.1f}% source frames had pose landmarks, fps={fps:.3f})"
    )


if __name__ == "__main__":
    main()
