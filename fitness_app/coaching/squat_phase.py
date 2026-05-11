"""Simple squat phase tracker from smoothed knee flexion."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np


class SquatPhase(Enum):
    UNKNOWN = auto()
    STANDING = auto()
    ECCENTRIC = auto()
    BOTTOM = auto()
    CONCENTRIC = auto()


class SquatPhaseTracker:
    """EMA-smoothed knee flexion + coarse phase state machine."""

    def __init__(
        self,
        *,
        ema_alpha: float = 0.28,
        standing_below_deg: float = 32.0,
        eccentric_enter_deg: float = 42.0,
        bottom_enter_deg: float = 68.0,
        bottom_exit_delta_deg: float = 18.0,
        standing_recover_deg: float = 36.0,
    ) -> None:
        self._ema_alpha = ema_alpha
        self._standing_below = standing_below_deg
        self._eccentric_enter = eccentric_enter_deg
        self._bottom_enter = bottom_enter_deg
        self._bottom_exit_delta = bottom_exit_delta_deg
        self._standing_recover = standing_recover_deg

        self._smooth_flex: float | None = None
        self._peak_flex: float = 0.0
        self.phase = SquatPhase.UNKNOWN

    def reset(self) -> None:
        self._smooth_flex = None
        self._peak_flex = 0.0
        self.phase = SquatPhase.UNKNOWN

    def update(self, mean_knee_flexion_deg: float) -> SquatPhase:
        if np.isnan(mean_knee_flexion_deg):
            return self.phase

        if self._smooth_flex is None:
            self._smooth_flex = mean_knee_flexion_deg
        else:
            a = self._ema_alpha
            self._smooth_flex = a * mean_knee_flexion_deg + (1.0 - a) * self._smooth_flex

        f = self._smooth_flex
        ph = self.phase

        if ph == SquatPhase.UNKNOWN:
            if f < self._standing_below:
                self.phase = SquatPhase.STANDING
        elif ph == SquatPhase.STANDING:
            if f > self._eccentric_enter:
                self.phase = SquatPhase.ECCENTRIC
                self._peak_flex = f
        elif ph == SquatPhase.ECCENTRIC:
            self._peak_flex = max(self._peak_flex, f)
            if f > self._bottom_enter:
                self.phase = SquatPhase.BOTTOM
        elif ph == SquatPhase.BOTTOM:
            self._peak_flex = max(self._peak_flex, f)
            if f < self._peak_flex - self._bottom_exit_delta:
                self.phase = SquatPhase.CONCENTRIC
        elif ph == SquatPhase.CONCENTRIC:
            if f < self._standing_recover:
                self.phase = SquatPhase.STANDING

        return self.phase
