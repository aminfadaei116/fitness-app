"""Abstract dataset layout root + registry/factory.

Concrete layouts live in ``public_datasets``. Validation is path-only (manual download/unzip),
see ``datasets/README.md``.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import ClassVar, TypeVar

from fitness_app.datasets.paths import resolve_dataset_root

DATASET_REGISTRY: dict[str, type["DatasetRoot"]] = {}

TDataset = TypeVar("TDataset", bound="DatasetRoot")


def register_dataset(cls: type[TDataset]) -> type[TDataset]:
    """Register ``cls`` under ``cls.REGISTRY_KEY``."""
    DATASET_REGISTRY[cls.REGISTRY_KEY] = cls
    return cls


class DatasetRoot(ABC):
    """Filesystem root for an unpacked public pose/heuristic corpus."""

    REGISTRY_KEY: ClassVar[str]
    SLUG: ClassVar[str]
    ENV_VAR: ClassVar[str | None]

    def __init__(self, root: Path | None = None) -> None:
        if root is not None:
            self._root = root.expanduser().resolve()
        else:
            self._root = resolve_dataset_root(self.ENV_VAR, self.SLUG)

    @property
    def root(self) -> Path:
        return self._root

    def validate(self) -> None:
        """Ensure ``root`` exists as a directory (see ``datasets/README.md`` for layout)."""
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Dataset root is not a directory: {self.root}. "
                f"Install '{self.REGISTRY_KEY}' data per datasets/README.md."
            )


def list_dataset_names() -> list[str]:
    """Sorted registry keys."""
    return sorted(DATASET_REGISTRY)


def get_dataset(name: str, root: Path | None = None) -> DatasetRoot:
    """Instantiate registered dataset ``name``. Raises ``ValueError`` if unknown."""
    try:
        cls = DATASET_REGISTRY[name]
    except KeyError as e:
        avail = ", ".join(list_dataset_names()) or "(none)"
        raise ValueError(f"Unknown dataset {name!r}. Available: {avail}") from e
    instance: DatasetRoot = cls(root=root)
    return instance


