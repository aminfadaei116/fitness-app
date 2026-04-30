"""Fit3D filesystem layout + metadata reader.

Ground-truth pose tensors (GHUM/SMPLX) are not decoded here; see upstream
``imar_vision_datasets_tools`` notebooks for mesh/visualization pipelines.

Expected layout after unzip (same as IMAR docs):

    <root>/train/
    <root>/test/
    <root>/info.json
    <root>/template.json   # optional
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SPLIT_NAMES = frozenset({"train", "test"})


def _readme_hint() -> str:
    root = Path(__file__).resolve().parents[2]
    return str(root / "datasets" / "README.md")


@dataclass(frozen=True)
class Fit3DRoot:
    """Validated Fit3D directory containing ``train/`` and ``test/`` splits."""

    root: Path

    def validate(self, *, require_info: bool = False) -> None:
        """Ensure ``train`` and ``test`` exist; optionally require ``info.json``."""
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Fit3D root is not a directory: {self.root}. "
                f"Download layout per {_readme_hint()}."
            )
        for split in ("train", "test"):
            p = self.root / split
            if not p.is_dir():
                raise FileNotFoundError(
                    f"Fit3D split directory missing: {p}. "
                    f"See {_readme_hint()} for expected layout."
                )
        if require_info and not (self.root / "info.json").is_file():
            raise FileNotFoundError(
                f"Fit3D info.json missing under {self.root}. "
                f"See {_readme_hint()}."
            )

    def list_splits(self) -> dict[str, Path]:
        """Return paths for ``train`` and ``test``."""
        self.validate()
        return {"train": self.root / "train", "test": self.root / "test"}

    def load_info(self) -> Mapping[str, Any]:
        """Load ``info.json`` if present; raise if missing."""
        path = self.root / "info.json"
        if not path.is_file():
            raise FileNotFoundError(f"Fit3D info.json not found: {path}")
        with path.open(encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return data

    def load_info_optional(self) -> Mapping[str, Any] | None:
        """Return parsed ``info.json`` or ``None`` if absent."""
        path = self.root / "info.json"
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return data

    def iter_sequences(self, split: str) -> Iterator[Path]:
        """Yield immediate child paths under ``train`` or ``test`` (sequence folders or files).

        Fit3D packs sequences under each split; naming follows the shipped archive.
        Only non-hidden entries are yielded.
        """
        if split not in _SPLIT_NAMES:
            raise ValueError(f"split must be one of {_SPLIT_NAMES}, got {split!r}")
        self.validate()
        split_dir = self.root / split
        for child in sorted(split_dir.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            yield child
