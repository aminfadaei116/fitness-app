"""Smoke tests for BVH export."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fitness_app.mocap.bvh_export import _default_tpose_offsets, build_bvh_hierarchy_string, write_bvh
from fitness_app.mocap.mediapipe_joints import landmarks_to_skeleton_joints


def _fake_joint_track(n_frames: int) -> list[dict[str, np.ndarray]]:
    lm = np.zeros((33, 4), dtype=np.float64)
    lm[:, :3] = 0.5
    lm[:, 3] = 0.9
    lm[11:13, 0] = [0.35, 0.65]
    lm[25:29, 1] += 0.12
    j0 = landmarks_to_skeleton_joints(lm)
    rng = np.random.default_rng(0)
    seq: list[dict[str, np.ndarray]] = []
    for _ in range(n_frames):
        noise = rng.standard_normal((len(j0), 3)) * 0.005
        jitter = dict(zip(j0.keys(), noise, strict=True))
        seq.append({k: v.copy() + jitter[k] for k, v in j0.items()})
    return seq


class BVHExportTests(unittest.TestCase):
    def test_write_bvh_frame_line_matches_row_count(self) -> None:
        tracks = _fake_joint_track(12)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "motion.bvh"
            write_bvh(path, tracks, fps=24.0)
            text = path.read_text(encoding="utf-8")
            m = re.search(r"(?m)^Frames:\s*(\d+)\s*$", text)
            assert m is not None
            frames = int(m.group(1))
            self.assertEqual(frames, 12)

            motion_i = text.index("MOTION")
            chunk = text[motion_i:].splitlines()
            numeric_lines = [ln for ln in chunk[3 : 3 + frames] if ln.strip()]
            self.assertEqual(len(numeric_lines), frames)

    def test_hierarchy_contains_root_and_known_joints(self) -> None:
        off = _default_tpose_offsets(1.0)
        hier = build_bvh_hierarchy_string(off)
        self.assertIn("ROOT Hips", hier)
        self.assertIn("JOINT Spine", hier)
        self.assertIn("End Site", hier)


if __name__ == "__main__":
    unittest.main()
