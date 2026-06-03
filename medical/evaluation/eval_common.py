"""Shared helpers for CVPR 3D / JSON evaluation."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from medical.paths import bpe_path, default_unified_ckpt, val_gt_dir, val_npz_dir, repo_root, sam3_package_dir

BPE = bpe_path()
DEFAULT_CKPT = default_unified_ckpt()
VAL_NPZ = val_npz_dir()
VAL_GT = val_gt_dir()
SAM3 = sam3_package_dir()
REPO = repo_root()

_TRAIN_QUERY_VOCAB = [
    "Adrenocortical carcinoma", "Airway Tree", "Alzheimer's disease plaque",
    "Aorta", "Aortic vessel trees", "Bladder", "Brachiocephalic trunk", "Brain",
    "Brainstem", "COVID-19 infection", "Cervical cancer tumor",
    "Circle of Willis", "Colon", "Colon cancer primaries", "Duodenum",
    "Endolysosomes", "Enhancing tissue", "Esophagus",
    "Gallbladder", "Head and neck structure", "Head-neck cancer", "Heart",
    "Inferior vena cava", "Intervertebral discs", "Kidney lesions",
    "Left Atrium", "Left Ventricle", "Left adrenal gland", "Left atrium",
    "Left cochlea", "Left kidney", "Left kidney cyst", "Left lung",
    "Left ventricle cavity", "Lesion", "Liver", "Liver tumors",
    "Lung lesions", "Lung lower lobe left", "Lung lower lobe right",
    "Lung middle lobe right", "Lung upper lobe left", "Lung upper lobe right",
    "Lymph node", "Mitochondria", "Myocardium", "Neural activity",
    "Non-enhancing tumor core", "Nuclei", "Nucleus",
    "Pancreas", "Pancreas tumors", "Portal vein and splenic vein",
    "Prostate", "Prostate lesion", "Prostate/uterus", "Pulmonary vein",
    "Resection cavity", "Ribs", "Right adrenal gland", "Right cochlea",
    "Right kidney", "Right kidney cyst", "Right lung",
    "Right ventricle cavity", "Sacrum", "Small bowel", "Spinal canal",
    "Spinal cord", "Spleen", "Stomach", "Stroke lesion",
    "Superior vena cava", "Surrounding non-enhancing FLAIR hyperintensity",
    "Thyroid", "Thyroid gland", "Trachea", "Urinary bladder",
    "Vertebrae", "Vessel", "White matter hyperintensities",
    "foreground",
]
_TRAIN_QUERY_VOCAB_SORTED = sorted(_TRAIN_QUERY_VOCAB, key=lambda x: -len(x))

_GROUPS = [
    ("CT_LiverTumor", ["CT_AbdTumor_liver", "CT_AbdTumor_HCC", "CT_AbdTumor_hepaticvessel", "CT_LiverTumor"]),
    ("CT_TotalSeg_organs", ["CT_TotalSeg_organs", "CT_AbdomenAtlas_BDMAP"]),
    ("MR_Heart_ACDC", ["MR_heart-ACDC", "MR_Heart_ACDC"]),
    ("CT_Aorta", ["CT_Aorta"]),
    ("CT_LungLesion", ["CT_LungLesions", "CT_LungLesion"]),
    ("MR_AMOS", ["MR_AMOSMR", "MR_AMOS"]),
    ("CT_PancreasTumor", ["CT_AbdTumor_pancreas", "CT_PancreasTumor"]),
    ("CT_KidneyTumor", ["CT_KidneyTumor"]),
    ("MR_BraTS", ["MR_BraTS"]),
    ("CT_TotalSeg_muscles", ["CT_TotalSeg_muscles"]),
]


def _install_np_core_shim():
    if "numpy._core" in sys.modules:
        return
    import numpy.core as _nc
    sys.modules["numpy._core"] = _nc
    for s in ("multiarray", "umath", "numeric", "_methods", "fromnumeric"):
        try:
            sys.modules[f"numpy._core.{s}"] = __import__(
                f"numpy.core.{s}", fromlist=[s]
            )
        except Exception:
            pass


def load_val_npz(p):
    _install_np_core_shim()
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def val_text_to_train_query(val_text: str) -> str | None:
    low = val_text.lower()
    for vocab in _TRAIN_QUERY_VOCAB_SORTED:
        if vocab.lower() in low:
            return vocab
    return None


def _group(stem):
    for label, pxs in _GROUPS:
        for p in pxs:
            if stem.startswith(p):
                return label
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def group_files(files):
    g = defaultdict(list)
    for f in sorted(files):
        g[_group(f.stem)].append(f)
    return dict(g)


def slice_to_rgb(vol, idx):
    arr = vol[idx]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def dice_iou(gt: np.ndarray, pred: np.ndarray | None):
    gt = gt.astype(bool)
    pred = pred.astype(bool) if pred is not None else np.zeros_like(gt)
    inter = int(np.logical_and(gt, pred).sum())
    union = int(np.logical_or(gt, pred).sum())
    iou = inter / union if union > 0 else 0.0
    denom = int(gt.sum()) + int(pred.sum())
    dice = 2 * inter / denom if denom > 0 else 0.0
    return dice, iou


def ensure_sam3_on_path():
    root = str(REPO)
    pkg = str(SAM3)
    for p in (root, pkg):
        if p not in sys.path:
            sys.path.insert(0, p)
