"""Resolve dataset roots relative to the repo or env overrides."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Root of the fitness-app repository (parent of `fitness_app/`)."""
    return Path(__file__).resolve().parents[2]


def fit3d_root() -> Path:
    """Fit3D root: ``FIT3D_ROOT`` env or ``datasets/fit3d`` under the repo."""
    raw = os.environ.get("FIT3D_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return repo_root() / "datasets" / "fit3d"


def m3gym_root() -> Path:
    """M3GYM root: ``M3GYM_ROOT`` env or ``datasets/m3gym`` under the repo."""
    raw = os.environ.get("M3GYM_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return repo_root() / "datasets" / "m3gym"


def require_exists(path: Path, what: str) -> Path:
    """Return ``path`` if it exists; else raise ``FileNotFoundError`` with ``what``."""
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path
