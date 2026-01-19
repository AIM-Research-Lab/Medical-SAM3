"""
Evaluation metrics for medical image segmentation.
Includes: Dice, IoU, PSNR, SSIM
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class SegmentationMetrics:
    """Container for segmentation metrics."""
    dice: float
    iou: float
    psnr: float
    ssim: float
    precision: float
    recall: float


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Dice similarity coefficient.

    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask

    Returns:
        Dice score in [0, 1]
    """
    pred = np.squeeze(pred).astype(bool)
    gt = np.squeeze(gt).astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0  # Both masks are empty

    return 2 * intersection / total


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index).

    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask

    Returns:
        IoU score in [0, 1]
    """
    pred = np.squeeze(pred).astype(bool)
    gt = np.squeeze(gt).astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0  # Both masks are empty

    return intersection / union


def compute_precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """
    Compute precision and recall.

    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask

    Returns:
        (precision, recall) tuple
    """
    pred = np.squeeze(pred).astype(bool)
    gt = np.squeeze(gt).astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio for binary masks.

    For binary masks, this treats them as images and computes PSNR.

    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask

    Returns:
        PSNR in dB (higher is better)
    """
    pred = np.squeeze(pred).astype(np.float64)
    gt = np.squeeze(gt).astype(np.float64)

    mse = np.mean((pred - gt) ** 2)

    if mse == 0:
        return float('inf')  # Perfect match

    # For binary images, max value is 1
    max_val = 1.0
    psnr = 10 * np.log10(max_val ** 2 / mse)

    return psnr


def compute_ssim(pred: np.ndarray, gt: np.ndarray, window_size: int = 11) -> float:
    """
    Compute Structural Similarity Index (SSIM) for binary masks.

    Uses a simplified SSIM computation suitable for binary masks.

    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask
        window_size: Size of the sliding window

    Returns:
        SSIM in [-1, 1] (1 is perfect)
    """
    pred = np.squeeze(pred).astype(np.float64)
    gt = np.squeeze(gt).astype(np.float64)

    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Global statistics (simplified SSIM)
    mu_pred = np.mean(pred)
    mu_gt = np.mean(gt)

    sigma_pred = np.std(pred)
    sigma_gt = np.std(gt)

    sigma_pred_gt = np.mean((pred - mu_pred) * (gt - mu_gt))

    # SSIM formula
    numerator = (2 * mu_pred * mu_gt + C1) * (2 * sigma_pred_gt + C2)
    denominator = (mu_pred ** 2 + mu_gt ** 2 + C1) * (sigma_pred ** 2 + sigma_gt ** 2 + C2)

    ssim = numerator / denominator

    return ssim


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> SegmentationMetrics:
    """
    Compute all segmentation metrics.

    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask

    Returns:
        SegmentationMetrics object
    """
    dice = compute_dice(pred, gt)
    iou = compute_iou(pred, gt)
    psnr = compute_psnr(pred, gt)
    ssim = compute_ssim(pred, gt)
    precision, recall = compute_precision_recall(pred, gt)

    return SegmentationMetrics(
        dice=dice,
        iou=iou,
        psnr=psnr,
        ssim=ssim,
        precision=precision,
        recall=recall
    )


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    print("=" * 60)

    # Create test masks
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:80, 20:80] = 1

    # Perfect prediction
    pred_perfect = gt.copy()
    metrics = compute_all_metrics(pred_perfect, gt)
    print(f"\nPerfect prediction:")
    print(f"  Dice: {metrics.dice:.4f}")
    print(f"  IoU: {metrics.iou:.4f}")
    print(f"  PSNR: {metrics.psnr:.2f} dB")
    print(f"  SSIM: {metrics.ssim:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall: {metrics.recall:.4f}")

    # Partial overlap
    pred_partial = np.zeros((100, 100), dtype=np.uint8)
    pred_partial[30:90, 30:90] = 1
    metrics = compute_all_metrics(pred_partial, gt)
    print(f"\nPartial overlap:")
    print(f"  Dice: {metrics.dice:.4f}")
    print(f"  IoU: {metrics.iou:.4f}")
    print(f"  PSNR: {metrics.psnr:.2f} dB")
    print(f"  SSIM: {metrics.ssim:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall: {metrics.recall:.4f}")

    # No overlap
    pred_none = np.zeros((100, 100), dtype=np.uint8)
    pred_none[0:20, 0:20] = 1
    metrics = compute_all_metrics(pred_none, gt)
    print(f"\nNo overlap:")
    print(f"  Dice: {metrics.dice:.4f}")
    print(f"  IoU: {metrics.iou:.4f}")
    print(f"  PSNR: {metrics.psnr:.2f} dB")
    print(f"  SSIM: {metrics.ssim:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall: {metrics.recall:.4f}")

    print("\n" + "=" * 60)
    print("Metrics test complete!")
