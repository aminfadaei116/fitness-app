# Local datasets (Fit3D / M3GYM)

Large archives are **not** committed to git. Place unpacked data under this folder or point env vars elsewhere.

## Fit3D

1. Register and download from [Fit3D](https://fit3d.imar.ro/).
2. Unzip so the layout matches [IMAR tooling expectations](https://github.com/sminchisescu-research/imar_vision_datasets_tools):

   ```text
   datasets/fit3d/
     train/
     test/
     info.json
     template.json    # optional
   ```

3. Override path if needed:

   ```bash
   set FIT3D_ROOT=D:\path\to\fit3d    # Windows
   export FIT3D_ROOT=/path/to/fit3d   # Unix
   ```

Official visualization/evaluation notebooks live in `imar_vision_datasets_tools`; this repo only provides lightweight filesystem + `info.json` readers in `fitness_app.datasets`.

## M3GYM

[M3GYM](https://finalyou.github.io/M3GYM/) (CVPR 2025): when a public archive with a stable on-disk layout is available, unzip under `datasets/m3gym/` or set:

```bash
set M3GYM_ROOT=D:\path\to\m3gym
```

The stub reader (`fitness_app.datasets.m3gym`) documents that layout details are still TBD until release.

## Inspect from the repo root

```bash
python scripts/inspect_dataset.py --dataset fit3d --limit 10
python scripts/inspect_dataset.py --dataset m3gym
```
