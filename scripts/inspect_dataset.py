#!/usr/bin/env python3
"""Smoke-check Fit3D / M3GYM roots from the repo (see datasets/README.md)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Fit3D or M3GYM dataset roots.")
    parser.add_argument("--dataset", choices=("fit3d", "m3gym"), required=True)
    parser.add_argument("--limit", type=int, default=10, metavar="N", help="Max sequence samples to print (Fit3D only)")
    args = parser.parse_args()

    if args.dataset == "fit3d":
        from fitness_app.datasets import Fit3DRoot, fit3d_root

        root = fit3d_root()
        print(f"FIT3D_ROOT resolved to: {root}")
        ds = Fit3DRoot(root)
        try:
            ds.validate()
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

        splits = ds.list_splits()
        for name, path in splits.items():
            print(f"  split {name}: {path}")

        info_opt = ds.load_info_optional()
        if info_opt is None:
            print("  info.json: (missing)")
        else:
            print(f"  info.json: loaded ({len(info_opt)} top-level keys)")

        for split in ("train", "test"):
            names = []
            for i, seq in enumerate(ds.iter_sequences(split)):
                names.append(seq.name)
                if i + 1 >= args.limit:
                    break
            print(f"  {split} samples (first {len(names)}): {names}")

        sys.exit(0)

    # m3gym
    from fitness_app.datasets import M3GYMRoot, m3gym_root

    root = m3gym_root()
    print(f"M3GYM_ROOT resolved to: {root}")
    ds = M3GYMRoot(root)
    try:
        ds.validate()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    info_opt = ds.load_info_optional()
    if info_opt is None:
        print("  info.json: (missing)")
    else:
        print(f"  info.json: loaded ({len(info_opt)} top-level keys)")

    try:
        ds.list_splits()
    except NotImplementedError as e:
        print(f"  splits: not available - {e}")
        sys.exit(0)


if __name__ == "__main__":
    main()
