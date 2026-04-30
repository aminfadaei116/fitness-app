# Local datasets (public pose / heuristic workflows)

Unpack archives **outside git** under `datasets/<slug>/` (see folders next to this file) or override paths with env vars below.

Python API (path validation only; no downloads or parsers):

```python
from fitness_app.datasets import get_dataset

get_dataset("coco_keypoints").validate()
```

See [`fitness_app/datasets/public_datasets.py`](../fitness_app/datasets/public_datasets.py) for registry keys.

## Registry keys and env overrides

| Registry key | Default folder | Env var |
|--------------|----------------|---------|
| `coco_keypoints` | `datasets/coco_keypoints/` | `COCO_POSE_ROOT` |
| `mpii_human_pose` | `datasets/mpii_human_pose/` | `MPII_POSE_ROOT` |
| `yoga_pose` | `datasets/yoga_pose/` | `YOGA_POSE_ROOT` |
| `exercise_skeleton` | `datasets/exercise_skeleton/` | `EXERCISE_SKELETON_ROOT` |
| `human36m` | `datasets/human36m/` | `HUMAN36M_ROOT` |

## Sources (manual download)

| Dataset | Notes |
|---------|--------|
| [MS COCO](https://cocodataset.org/) | Images + person keypoint annotations (train/val). Typical layout includes `annotations/` and `train2017/` or `val2017/` after unzip; place under `coco_keypoints/` (any internal naming is fine). |
| [MPII Human Pose](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset) | Images + annotation JSON/mat per official package; unzip under `mpii_human_pose/`. |
| Yoga / static poses | Many forks on GitHub or Kaggle; structures vary. Place chosen corpus root under `yoga_pose/`. |
| Exercise + skeleton features | Example discovery: Kaggle search *exercise recognition*, *mediapipe landmarks*. Place unpacked CSV/feature trees under `exercise_skeleton/`. |
| [Human3.6M](http://vision.imar.ro/human3.6m/) | Academic license; layout depends on release. Place extracted root under `human36m/`. |

This repo does **not** fetch or parse annotations yet; `validate()` only checks that the resolved directory exists.
