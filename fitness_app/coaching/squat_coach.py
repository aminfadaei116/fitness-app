"""Squat coaching orchestrator + HUD drawing."""

from __future__ import annotations

import numpy as np

from fitness_app.coaching.squat_features import compute_squat_features
from fitness_app.coaching.squat_phase import SquatPhaseTracker
from fitness_app.coaching.squat_rules import SquatRuleThresholds, evaluate_squat_rules


class SquatCoach:
    """Phase tracker + threshold rules on MediaPipe landmarks."""

    def __init__(
        self,
        thresholds: SquatRuleThresholds | None = None,
        tracker: SquatPhaseTracker | None = None,
    ) -> None:
        self._thresholds = thresholds or SquatRuleThresholds()
        self._tracker = tracker or SquatPhaseTracker()

    def update(self, frame_bgr: np.ndarray, landmarks_33x4: np.ndarray | None) -> list[str]:
        if landmarks_33x4 is None:
            return ["No pose"]

        h, w = frame_bgr.shape[:2]
        feats = compute_squat_features(landmarks_33x4, w, h)

        flex = feats.mean_knee_flexion_deg
        phase = self._tracker.update(flex)

        cues = evaluate_squat_rules(phase, feats, self._thresholds)
        phase_label = phase.name.lower().replace("_", " ")
        prefix = f"[{phase_label}]"
        return [prefix] + cues if cues else [prefix]


def draw_coaching_hud(frame: np.ndarray, lines: list[str], *, margin: int = 10) -> None:
    """Draw multi-line coaching text bottom-left."""
    import cv2

    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    line_h = 18
    pad = 6
    sizes = [cv2.getTextSize(s, font, scale, thickness)[0] for s in lines]
    max_w = max(w for w, _ in sizes)
    total_h = line_h * len(lines)
    x0 = margin
    y0 = frame.shape[0] - margin - total_h - 2 * pad
    x1 = x0 + max_w + 2 * pad
    y1 = frame.shape[0] - margin
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        y = y0 + pad + (i + 1) * line_h - 4
        cv2.putText(frame, s, (x0 + pad, y), font, scale, (200, 255, 200), thickness, cv2.LINE_AA)


def build_coach(name: str) -> SquatCoach:
    key = name.strip().lower()
    if key != "squat":
        raise ValueError(f"Unknown coach {name!r}. Available: squat")
    return SquatCoach()
