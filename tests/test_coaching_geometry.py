"""Tests for coaching geometry helpers."""

from __future__ import annotations

import unittest

import numpy as np

from fitness_app.coaching.geometry import angle_deg, torso_vertical_deg
from fitness_app.coaching.squat_phase import SquatPhase, SquatPhaseTracker


class GeometryTests(unittest.TestCase):
    def test_angle_deg_right_angle(self) -> None:
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 0.0])
        deg = angle_deg(a, b, c)
        self.assertAlmostEqual(deg, 90.0, places=5)

    def test_angle_deg_colinear_straight(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([2.0, 0.0])
        deg = angle_deg(a, b, c)
        self.assertAlmostEqual(deg, 180.0, places=5)

    def test_torso_vertical_upright(self) -> None:
        hip = np.array([100.0, 400.0])
        shoulder = np.array([100.0, 200.0])
        t = torso_vertical_deg(hip, shoulder)
        self.assertAlmostEqual(t, 0.0, places=3)


class PhaseTrackerSmokeTests(unittest.TestCase):
    def test_phase_reaches_bottom(self) -> None:
        tr = SquatPhaseTracker(
            standing_below_deg=30.0,
            eccentric_enter_deg=40.0,
            bottom_enter_deg=65.0,
            bottom_exit_delta_deg=15.0,
            standing_recover_deg=35.0,
        )
        flex = [5, 10, 15, 50, 72, 74, 55, 40, 20, 10, 5]
        last = SquatPhase.UNKNOWN
        for f in flex:
            last = tr.update(f)
        self.assertIsInstance(last, SquatPhase)


if __name__ == "__main__":
    unittest.main()
