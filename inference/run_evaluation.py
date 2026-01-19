#!/usr/bin/env python3
"""
SAM3 Zero-Shot Evaluation on Medical Image Datasets.

Evaluates SAM3 with both box prompts and text prompts on:
- CHASE_DB1 (Fundus - Retinal Vessel)
- STARE (Fundus - Retinal Vessel)
- CVC-ClinicDB (Endoscopy - Polyp)
- ETIS-Larib (Endoscopy - Polyp)
- PH2 (Dermoscopy - Skin Lesion)

Usage:
    conda activate medsam3
    python run_evaluation.py [--max-samples N] [--datasets DATASET1,DATASET2]
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_loaders import load_dataset, DATASET_LOADERS, DATASET_PROMPTS
from sam3_inference import SAM3Model, generate_bbox_from_mask, resize_mask
from metrics import compute_all_metrics, SegmentationMetrics


# Output directory
OUTPUT_DIR = Path(__file__).parent / "results"


def evaluate_dataset(
    sam3: SAM3Model,
    dataset_name: str,
    max_samples: Optional[int] = None
) -> dict:
    """
    Evaluate SAM3 on a single dataset.

    Returns dict with metrics for box and text prompts.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"Text Prompt: {DATASET_PROMPTS[dataset_name]}")
    print(f"{'='*60}")

    # Metrics storage
    box_metrics = []
    text_metrics = []

    # Load samples
    samples = list(load_dataset(dataset_name, max_samples))
    print(f"Loaded {len(samples)} samples")

    for sample in tqdm(samples, desc=dataset_name):
        img_size = sample.gt_mask.shape

        # Encode image
        inference_state = sam3.encode_image(sample.image)

        # Generate bbox from GT
        bbox = generate_bbox_from_mask(sample.gt_mask)
        if bbox is None:
            continue

        # Box prompt inference
        pred_mask_box = sam3.predict_box(inference_state, bbox, img_size)

        if pred_mask_box is not None:
            # Resize if needed
            if pred_mask_box.shape != img_size:
                pred_mask_box = resize_mask(pred_mask_box, img_size)

            metrics_box = compute_all_metrics(pred_mask_box, sample.gt_mask)
            box_metrics.append({
                'sample_id': sample.sample_id,
                'dice': metrics_box.dice,
                'iou': metrics_box.iou,
                'psnr': metrics_box.psnr,
                'ssim': metrics_box.ssim,
                'precision': metrics_box.precision,
                'recall': metrics_box.recall,
            })

        # Text prompt inference
        pred_mask_text = sam3.predict_text(inference_state, sample.text_prompt)

        if pred_mask_text is not None:
            if pred_mask_text.shape != img_size:
                pred_mask_text = resize_mask(pred_mask_text, img_size)

            metrics_text = compute_all_metrics(pred_mask_text, sample.gt_mask)
            text_metrics.append({
                'sample_id': sample.sample_id,
                'dice': metrics_text.dice,
                'iou': metrics_text.iou,
                'psnr': metrics_text.psnr,
                'ssim': metrics_text.ssim,
                'precision': metrics_text.precision,
                'recall': metrics_text.recall,
            })
        else:
            # No text prediction
            text_metrics.append({
                'sample_id': sample.sample_id,
                'dice': 0.0,
                'iou': 0.0,
                'psnr': 0.0,
                'ssim': 0.0,
                'precision': 0.0,
                'recall': 0.0,
            })

    # Compute averages
    def avg_metrics(metrics_list):
        if not metrics_list:
            return {}
        df = pd.DataFrame(metrics_list)
        return {
            'dice': df['dice'].mean(),
            'iou': df['iou'].mean(),
            'psnr': df['psnr'].replace([np.inf, -np.inf], np.nan).mean(),
            'ssim': df['ssim'].mean(),
            'precision': df['precision'].mean(),
            'recall': df['recall'].mean(),
            'n_samples': len(df),
        }

    results = {
        'dataset': dataset_name,
        'text_prompt': DATASET_PROMPTS[dataset_name],
        'box_prompt': avg_metrics(box_metrics),
        'text_prompt_results': avg_metrics(text_metrics),
        'box_details': box_metrics,
        'text_details': text_metrics,
    }

    # Print summary
    print(f"\nResults for {dataset_name}:")
    print(f"  Box Prompt:  Dice={results['box_prompt'].get('dice', 0):.2%}, "
          f"IoU={results['box_prompt'].get('iou', 0):.2%}, "
          f"SSIM={results['box_prompt'].get('ssim', 0):.4f}")
    print(f"  Text Prompt: Dice={results['text_prompt_results'].get('dice', 0):.2%}, "
          f"IoU={results['text_prompt_results'].get('iou', 0):.2%}, "
          f"SSIM={results['text_prompt_results'].get('ssim', 0):.4f}")

    return results


def generate_report(all_results: List[dict], output_dir: Path):
    """Generate evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Dataset': result['dataset'],
            'Text Prompt': result['text_prompt'],
            'Box Dice (%)': result['box_prompt'].get('dice', 0) * 100,
            'Box IoU (%)': result['box_prompt'].get('iou', 0) * 100,
            'Box PSNR': result['box_prompt'].get('psnr', 0),
            'Box SSIM': result['box_prompt'].get('ssim', 0),
            'Text Dice (%)': result['text_prompt_results'].get('dice', 0) * 100,
            'Text IoU (%)': result['text_prompt_results'].get('iou', 0) * 100,
            'Text PSNR': result['text_prompt_results'].get('psnr', 0),
            'Text SSIM': result['text_prompt_results'].get('ssim', 0),
            'N Samples': result['box_prompt'].get('n_samples', 0),
        })

    df_summary = pd.DataFrame(summary_data)

    # Save CSV
    csv_path = output_dir / "sam3_evaluation_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary to: {csv_path}")

    # Save detailed JSON
    json_path = output_dir / "sam3_evaluation_detailed.json"
    with open(json_path, 'w') as f:
        # Remove detailed lists for cleaner JSON
        clean_results = []
        for r in all_results:
            clean_results.append({
                'dataset': r['dataset'],
                'text_prompt': r['text_prompt'],
                'box_prompt': r['box_prompt'],
                'text_prompt_results': r['text_prompt_results'],
            })
        json.dump(clean_results, f, indent=2)
    print(f"Saved detailed results to: {json_path}")

    # Generate markdown report
    report_path = output_dir / "sam3_evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write("# SAM3 Zero-Shot Medical Image Segmentation Evaluation\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        f.write("| Dataset | Text Prompt | Box Dice | Box IoU | Text Dice | Text IoU | Samples |\n")
        f.write("|---------|-------------|----------|---------|-----------|----------|--------|\n")

        for row in summary_data:
            f.write(f"| {row['Dataset']} | {row['Text Prompt']} | "
                    f"{row['Box Dice (%)']:.1f}% | {row['Box IoU (%)']:.1f}% | "
                    f"{row['Text Dice (%)']:.1f}% | {row['Text IoU (%)']:.1f}% | "
                    f"{row['N Samples']} |\n")

        # Overall averages
        f.write("\n## Overall Averages\n\n")
        avg_box_dice = np.mean([r['Box Dice (%)'] for r in summary_data])
        avg_box_iou = np.mean([r['Box IoU (%)'] for r in summary_data])
        avg_text_dice = np.mean([r['Text Dice (%)'] for r in summary_data])
        avg_text_iou = np.mean([r['Text IoU (%)'] for r in summary_data])

        f.write(f"- **Box Prompt Average**: Dice={avg_box_dice:.1f}%, IoU={avg_box_iou:.1f}%\n")
        f.write(f"- **Text Prompt Average**: Dice={avg_text_dice:.1f}%, IoU={avg_text_iou:.1f}%\n")

        f.write("\n## Notes\n\n")
        f.write("- Box prompts use bounding boxes derived from ground truth masks\n")
        f.write("- Text prompts use natural language descriptions (zero-shot)\n")
        f.write("- Metrics: Dice (F1), IoU (Jaccard), PSNR, SSIM\n")

    print(f"Saved report to: {report_path}")

    return df_summary


def main():
    parser = argparse.ArgumentParser(description="SAM3 Medical Image Evaluation")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples per dataset (for testing)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated list of datasets to evaluate")
    args = parser.parse_args()

    print("=" * 60)
    print("SAM3 Zero-Shot Medical Image Segmentation Evaluation")
    print("=" * 60)

    # Determine datasets to evaluate
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = list(DATASET_LOADERS.keys())

    print(f"\nDatasets to evaluate: {datasets}")
    print(f"Max samples per dataset: {args.max_samples or 'All'}")

    # Initialize SAM3
    sam3 = SAM3Model(confidence_threshold=0.1)

    # Evaluate each dataset
    all_results = []
    for dataset_name in datasets:
        if dataset_name not in DATASET_LOADERS:
            print(f"Unknown dataset: {dataset_name}, skipping...")
            continue

        result = evaluate_dataset(sam3, dataset_name, args.max_samples)
        all_results.append(result)

    # Generate report
    print("\n" + "=" * 60)
    print("Generating Report")
    print("=" * 60)

    df_summary = generate_report(all_results, OUTPUT_DIR)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
