"""Concrete dataset roots for heuristic / public pose corpora (path validation only)."""

from __future__ import annotations

from fitness_app.datasets.base import DatasetRoot, register_dataset


@register_dataset
class COCOKeypointsRoot(DatasetRoot):
    """MS COCO train/val images + keypoint annotations (manual download)."""

    REGISTRY_KEY = "coco_keypoints"
    SLUG = "coco_keypoints"
    ENV_VAR = "COCO_POSE_ROOT"


@register_dataset
class MPIIHumanPoseRoot(DatasetRoot):
    """MPII Human Pose (images + annotations, manual download)."""

    REGISTRY_KEY = "mpii_human_pose"
    SLUG = "mpii_human_pose"
    ENV_VAR = "MPII_POSE_ROOT"


@register_dataset
class YogaPoseRoot(DatasetRoot):
    """Yoga / static pose image sets (many public forks; layout varies)."""

    REGISTRY_KEY = "yoga_pose"
    SLUG = "yoga_pose"
    ENV_VAR = "YOGA_POSE_ROOT"


@register_dataset
class ExerciseSkeletonRoot(DatasetRoot):
    """Exercise-classification or skeleton-feature dumps (e.g. Kaggle CSV/landmarks)."""

    REGISTRY_KEY = "exercise_skeleton"
    SLUG = "exercise_skeleton"
    ENV_VAR = "EXERCISE_SKELETON_ROOT"


@register_dataset
class Human36MRoot(DatasetRoot):
    """Human3.6M (academic agreement; layout depends on release variant)."""

    REGISTRY_KEY = "human36m"
    SLUG = "human36m"
    ENV_VAR = "HUMAN36M_ROOT"


__all__ = [
    "COCOKeypointsRoot",
    "MPIIHumanPoseRoot",
    "YogaPoseRoot",
    "ExerciseSkeletonRoot",
    "Human36MRoot",
]
