"""Dataset layout registry for public pose / heuristic corpora (path-only validation)."""

from fitness_app.datasets import public_datasets as _public_datasets  # noqa: F401 - register datasets

from fitness_app.datasets.base import (
    DATASET_REGISTRY,
    DatasetRoot,
    get_dataset,
    list_dataset_names,
    register_dataset,
)
from fitness_app.datasets.paths import (
    default_dataset_root,
    repo_root,
    require_exists,
    resolve_dataset_root,
)

__all__ = [
    "DATASET_REGISTRY",
    "DatasetRoot",
    "get_dataset",
    "list_dataset_names",
    "register_dataset",
    "repo_root",
    "default_dataset_root",
    "resolve_dataset_root",
    "require_exists",
]
