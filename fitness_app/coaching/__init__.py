"""Heuristic pose coaching (squat MVP)."""

from fitness_app.coaching.geometry import (
    angle_deg,
    midpoint_xy,
    normalize_xy_about_center,
    pixel_xy,
    shoulder_width_px,
    symmetry_delta,
    torso_vertical_deg,
)
from fitness_app.coaching.squat_coach import SquatCoach, build_coach, draw_coaching_hud
from fitness_app.coaching.squat_features import SquatFeatures, compute_squat_features
from fitness_app.coaching.squat_phase import SquatPhase, SquatPhaseTracker
from fitness_app.coaching.squat_rules import SquatRuleThresholds, evaluate_squat_rules

__all__ = [
    "SquatCoach",
    "build_coach",
    "draw_coaching_hud",
    "SquatFeatures",
    "compute_squat_features",
    "SquatPhase",
    "SquatPhaseTracker",
    "SquatRuleThresholds",
    "evaluate_squat_rules",
    "angle_deg",
    "midpoint_xy",
    "normalize_xy_about_center",
    "pixel_xy",
    "shoulder_width_px",
    "symmetry_delta",
    "torso_vertical_deg",
]
