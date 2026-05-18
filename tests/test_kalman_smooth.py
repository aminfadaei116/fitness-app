"""Tests for constant-velocity Kalman landmark smoothing."""

from __future__ import annotations

import unittest

import numpy as np

from fitness_app.mocap.kalman_smooth import smooth_landmark_sequence


class KalmanSmoothTests(unittest.TestCase):
    def test_smooth_reduces_measurement_noise_variance(self) -> None:
        fps = 30.0
        n = 180
        t = np.arange(n, dtype=np.float64) / fps

        baseline = np.zeros((33, 4), dtype=np.float64)
        baseline[:, :3] = 0.5
        baseline[:, 3] = 1.0

        noisy: list[np.ndarray | None] = []
        rng = np.random.default_rng(0)
        for i in range(n):
            lm = baseline.copy()
            true_x = float(0.5 + 0.2 * np.sin(2 * np.pi * 0.4 * t[i]))
            lm[11, 0] = true_x + float(rng.normal(0.0, 0.06))
            noisy.append(lm)

        smoothed = smooth_landmark_sequence(
            noisy, fps=fps, meas_var=4e-3, mahalanobis_max=900.0, visibility_threshold=0.0
        )

        d_raw = np.diff(np.array([noisy[i][11, 0] for i in range(n)], dtype=np.float64))
        d_sm = np.diff(
            np.asarray([smoothed[i][11, 0] for i in range(n)], dtype=np.float64)
        )
        self.assertLess(np.var(d_sm), np.var(d_raw) * 0.85)

    def test_none_gaps_keep_sequence_length(self) -> None:
        lm = np.zeros((33, 4), dtype=np.float64)
        lm[:, 3] = 1.0
        lm[:, 0] = 0.5

        seq: list[np.ndarray | None] = [None, lm, lm, None, lm]
        out = smooth_landmark_sequence(seq, fps=60.0)
        self.assertEqual(len(out), 5)

    def test_untracked_frames_do_not_diverge_quickly(self) -> None:
        seq: list[np.ndarray | None] = [None] * 5
        lm = np.zeros((33, 4), dtype=np.float64)
        lm[0, :] = [0.4, 0.5, 0.02, 0.9]
        seq.append(lm)

        out = smooth_landmark_sequence(seq, fps=120.0, visibility_threshold=0.0)
        self.assertLess(np.max(np.abs(out[4][0, :3])), 10.0)


if __name__ == "__main__":
    unittest.main()
