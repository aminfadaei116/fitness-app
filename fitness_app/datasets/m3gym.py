"""M3GYM dataset placeholder until an official on-disk layout is published.

Project: https://finalyou.github.io/M3GYM/

Validate only checks that the root exists. Listing sequences raises until layout is wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_LAYOUT_MSG = (
    "M3GYM layout is not configured in this repo yet. "
    "When authors publish archives and a stable folder convention exists, "
    "update fitness_app/datasets/m3gym.py and datasets/README.md."
)


def _readme_hint() -> str:
    root = Path(__file__).resolve().parents[2]
    return str(root / "datasets" / "README.md")


@dataclass(frozen=True)
class M3GYMRoot:
    """Reserved M3GYM root directory."""

    root: Path

    def validate(self) -> None:
        """Ensure ``root`` exists as a directory."""
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"M3GYM root is not a directory: {self.root}. "
                f"See {_readme_hint()}."
            )

    def load_info_optional(self) -> dict[str, object] | None:
        """Return parsed ``info.json`` if present."""
        path = self.root / "info.json"
        if not path.is_file():
            return None
        import json

        with path.open(encoding="utf-8") as f:
            data: dict[str, object] = json.load(f)
        return data

    def list_splits(self) -> dict[str, Path]:
        """Raise until split directories are defined for M3GYM releases."""
        self.validate()
        raise NotImplementedError(_LAYOUT_MSG)

    def iter_sequences(self, split: str) -> None:
        """Raise until sequence traversal is implemented."""
        raise NotImplementedError(_LAYOUT_MSG)
