"""Dataset layout helpers for Fit3D and optional M3GYM."""

from fitness_app.datasets.fit3d import Fit3DRoot
from fitness_app.datasets.m3gym import M3GYMRoot
from fitness_app.datasets.paths import fit3d_root, m3gym_root, repo_root, require_exists

__all__ = [
    "Fit3DRoot",
    "M3GYMRoot",
    "fit3d_root",
    "m3gym_root",
    "repo_root",
    "require_exists",
]
