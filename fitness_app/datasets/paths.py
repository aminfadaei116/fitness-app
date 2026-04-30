"""Resolve dataset roots relative to the repo or env overrides."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Root of the fitness-app repository (parent of `fitness_app/`)."""
    return Path(__file__).resolve().parents[2]


def default_dataset_root(slug: str) -> Path:
    """Default path ``datasets/<slug>`` under the repo."""
    return repo_root() / "datasets" / slug


def resolve_dataset_root(env_var_name: str | None, slug: str) -> Path:
    """Resolve root from ``env_var_name`` if set and non-empty, else ``default_dataset_root(slug)``."""
    if env_var_name:
        raw = os.environ.get(env_var_name)
        if raw:
            return Path(raw).expanduser().resolve()
    return default_dataset_root(slug)


def require_exists(path: Path, what: str) -> Path:
    """Return ``path`` if it exists; else raise ``FileNotFoundError`` with ``what``."""
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path
