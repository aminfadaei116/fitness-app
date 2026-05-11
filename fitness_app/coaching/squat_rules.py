"""Heuristic squat rule thresholds (not medical advice).

Extension hooks (future):
- Gold clips: JSON profiles with mean/std knee flexion and torso angle per SquatPhase,
  then compare z-scores instead of fixed thresholds.
- Procrustes: align normalized 2D joints per phase vs a reference skeleton, score mean L2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fitness_app.coaching.squat_features import SquatFeatures
from fitness_app.coaching.squat_phase import SquatPhase


@dataclass
class SquatRuleThresholds:
    min_bottom_depth_deg: float = 52.0
    max_torso_lean_deg: float = 38.0
    max_knee_symmetry_delta_deg: float = 14.0
    knee_visibility_floor: float = 0.35
    knee_over_toe_dx_warn: float = 1.35


def evaluate_squat_rules(
    phase: SquatPhase,
    feats: SquatFeatures,
    thresholds: SquatRuleThresholds | None = None,
    *,
    max_cues: int = 3,
) -> list[str]:
    """Return short HUD cue strings for current phase + features."""
    t = thresholds or SquatRuleThresholds()
    cues: list[str] = []

    vis_ok = feats.min_knee_visibility >= t.knee_visibility_floor

    # Depth / symmetry near bottom
    if phase == SquatPhase.BOTTOM:
        if not np.isnan(feats.mean_knee_flexion_deg) and feats.mean_knee_flexion_deg < t.min_bottom_depth_deg:
            cues.append("Try a deeper squat")
        if vis_ok:
            lf = feats.knee_flexion_left_deg
            rf = feats.knee_flexion_right_deg
            if not np.isnan(lf) and not np.isnan(rf):
                d = abs(lf - rf)
                if d > t.max_knee_symmetry_delta_deg:
                    cues.append("Level out both legs")

    # Torso during movement
    if phase in (SquatPhase.ECCENTRIC, SquatPhase.BOTTOM, SquatPhase.CONCENTRIC):
        tv = feats.torso_vs_vertical_deg
        if not np.isnan(tv) and tv > t.max_torso_lean_deg:
            cues.append("Keep chest taller")

    # Weak knee-over-toe (2D)
    if vis_ok:
        if abs(feats.knee_ankle_dx_norm_left) > t.knee_over_toe_dx_warn:
            cues.append("Check L knee tracking")
        if abs(feats.knee_ankle_dx_norm_right) > t.knee_over_toe_dx_warn:
            cues.append("Check R knee tracking")

    return cues[:max_cues]
