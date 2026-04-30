"""Tests for fitness_app.datasets registry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fitness_app.datasets import DATASET_REGISTRY, get_dataset, list_dataset_names


EXPECTED_KEYS = frozenset(
    {
        "coco_keypoints",
        "mpii_human_pose",
        "yoga_pose",
        "exercise_skeleton",
        "human36m",
    }
)


class DatasetRegistryTests(unittest.TestCase):
    def test_registry_keys(self) -> None:
        self.assertEqual(set(DATASET_REGISTRY.keys()), EXPECTED_KEYS)
        self.assertEqual(set(list_dataset_names()), EXPECTED_KEYS)

    def test_unknown_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_dataset("nonexistent_dataset_xyz")
        self.assertIn("Available:", str(ctx.exception))

    def test_validate_accepts_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = get_dataset("coco_keypoints", root=root)
            ds.validate()

    def test_validate_rejects_missing_directory(self) -> None:
        root = Path(tempfile.gettempdir()) / "fitness_app_dataset_missing_xyz_nonexistent"
        if root.exists():
            root.rmdir()
        ds = get_dataset("coco_keypoints", root=root)
        with self.assertRaises(FileNotFoundError):
            ds.validate()


if __name__ == "__main__":
    unittest.main()
