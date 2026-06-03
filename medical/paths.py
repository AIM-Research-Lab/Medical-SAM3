"""Portable paths for MedSAM3 training and evaluation.

Set environment variables before running scripts:

  export MEDSAM3_ROOT=/path/to/sam3/repo          # this repository root
  export MEDSAM3_DATA_ROOT=/path/to/project       # parent of converted_cvpr_biomedsegfm
  export MEDSAM3_CVPR_ROOT=$MEDSAM3_DATA_ROOT/converted_cvpr_biomedsegfm  # optional
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """SAM3 / medicalsam3 repository root (parent of ``medical/``)."""
    env = os.environ.get("MEDSAM3_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    env = os.environ.get("MEDSAM3_DATA_ROOT")
    if env:
        return Path(env).resolve()
    # Default: sibling of sam3 repo
    return repo_root().parent


def cvpr_root() -> Path:
    env = os.environ.get("MEDSAM3_CVPR_ROOT")
    if env:
        return Path(env).resolve()
    return data_root() / "converted_cvpr_biomedsegfm"


def sam3_package_dir() -> Path:
    return repo_root() / "sam3"


def bpe_path() -> Path:
    return repo_root() / "assets" / "bpe_simple_vocab_16e6.txt.gz"


def default_unified_ckpt() -> Path:
    return repo_root() / "experiments" / "medsam3_stage1_train_all_unified" / "checkpoints" / "checkpoint.pt"


def val_npz_dir() -> Path:
    return cvpr_root() / "Dataset3D" / "CVPR-BiomedSegFM" / "3D_val_npz"


def val_gt_dir() -> Path:
    return cvpr_root() / "Dataset3D" / "CVPR-BiomedSegFM" / "3D_val_gt" / "3D_val_gt_interactive"


def npz_resolve_roots() -> list[Path]:
    return [
        cvpr_root(),
        data_root(),
        cvpr_root() / "annotations",
        cvpr_root() / "Dataset3D" / "CVPR-BiomedSegFM",
        cvpr_root() / "volumes_train10p",
        data_root() / "dataset" / "datasetavail",
        repo_root(),
    ]
