"""Tests for Fit3D dataset readers (stdlib unittest)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fitness_app.datasets.fit3d import Fit3DRoot


class Fit3DRootTests(unittest.TestCase):
    def test_validate_and_iterate_fixture(self) -> None:
        fixture = Path(__file__).resolve().parent / "fixtures" / "fit3d_fake"
        ds = Fit3DRoot(fixture)
        ds.validate()
        train_names = [p.name for p in ds.iter_sequences("train")]
        test_names = [p.name for p in ds.iter_sequences("test")]
        self.assertEqual(train_names, ["seq_one"])
        self.assertEqual(test_names, ["seq_two"])

        payload = ds.load_info()
        self.assertEqual(payload.get("version"), 1)

    def test_validate_raises_without_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = Fit3DRoot(root)
            with self.assertRaises(FileNotFoundError):
                ds.validate()


if __name__ == "__main__":
    unittest.main()
