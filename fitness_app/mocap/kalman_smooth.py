"""Constant-velocity Kalman filter per BlazePose landmark (NumPy-only)."""

from __future__ import annotations

import numpy as np


def _mahalanobis_sq(residual: np.ndarray, S: np.ndarray) -> float:
    """Squared Mahalanobis distance ``r^T S^{-1} r``."""
    try:
        L = np.linalg.cholesky(S)
        y = np.linalg.solve(L, residual)
        return float(y @ y)
    except np.linalg.LinAlgError:
        inv = np.linalg.inv(S + np.eye(3) * 1e-9)
        return float(residual @ inv @ residual)


class _CVKalman3D:
    """6D state: position (x,y,z), velocity (vx,vy,vz). Observes position only."""

    def __init__(
        self,
        dt: float,
        process_pos_var: float,
        process_vel_var: float,
        meas_var: float,
    ) -> None:
        self.dt = dt
        self.F = np.block(
            [
                [np.eye(3), dt * np.eye(3)],
                [np.zeros((3, 3)), np.eye(3)],
            ]
        )
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        q_p, q_v = process_pos_var, process_vel_var
        self.Q = np.diag([q_p, q_p, q_p, q_v, q_v, q_v])
        self.R_base = meas_var * np.eye(3)
        self.s = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 500.0
        self.initialized = False

    def predict(self) -> None:
        if not self.initialized:
            return
        self.s = self.F @ self.s
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(
        self,
        z: np.ndarray,
        visibility: float,
        *,
        visibility_threshold: float,
        mahalanobis_max: float,
        low_vis_r_mult: float,
    ) -> bool:
        """Returns True if a measurement update was applied."""
        if not self.initialized:
            self.s[:3] = z
            self.s[3:] = 0.0
            self.P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
            self.initialized = True
            return True

        if visibility < visibility_threshold:
            return False

        R = self.R_base * (low_vis_r_mult if visibility < 0.7 else max(1.0, 1.0 / max(visibility, 0.05)))

        s_pred = self.s.copy()
        P_pred = self.P.copy()
        z_pred = self.H @ s_pred
        residual = z - z_pred
        S = self.H @ P_pred @ self.H.T + R

        if _mahalanobis_sq(residual, S) > mahalanobis_max:
            return False

        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.s = s_pred + K @ residual
        I = np.eye(6)
        self.P = (I - K @ self.H) @ P_pred
        return True


def smooth_landmark_sequence(
    keypoints: list[np.ndarray | None],
    *,
    fps: float,
    visibility_threshold: float = 0.35,
    mahalanobis_max: float = 25.0,
    process_pos_var: float = 1e-5,
    process_vel_var: float = 1e-3,
    meas_var: float = 2e-4,
    low_vis_r_mult: float = 25.0,
) -> list[np.ndarray]:
    """
    Smooth a sequence of ``(33, 4)`` landmarks (x,y,z,visibility) or ``None``.

    Missing frames: predict-only; visibility column is forward-filled from the last
    successful update, else ``0.0``. Output has a full ``(33, 4)`` array every frame.
    """
    n_frames = len(keypoints)
    if n_frames == 0:
        return []

    dt = 1.0 / max(fps, 1e-6)
    filters = [
        _CVKalman3D(dt, process_pos_var, process_vel_var, meas_var) for _ in range(33)
    ]
    last_vis = np.zeros(33, dtype=np.float64)

    out: list[np.ndarray] = []
    for t in range(n_frames):
        lm = keypoints[t]
        for i in range(33):
            filters[i].predict()

        if lm is not None:
            pos = lm[:, :3].astype(np.float64, copy=False)
            vis = lm[:, 3].astype(np.float64, copy=False)
            for i in range(33):
                z = pos[i]
                v = float(vis[i])
                did = filters[i].update(
                    z,
                    v,
                    visibility_threshold=visibility_threshold,
                    mahalanobis_max=mahalanobis_max,
                    low_vis_r_mult=low_vis_r_mult,
                )
                if did:
                    last_vis[i] = v
        row = np.zeros((33, 4), dtype=np.float64)
        for i in range(33):
            kf = filters[i]
            if kf.initialized:
                row[i, :3] = kf.s[:3]
                row[i, 3] = last_vis[i]
        out.append(row)

    return out
