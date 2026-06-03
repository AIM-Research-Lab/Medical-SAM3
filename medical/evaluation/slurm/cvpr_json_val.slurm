#!/bin/bash
#SBATCH --job-name=medsam3_jsonval
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=experiments/results/cvpr_json_val/slurm_%j.log
#SBATCH --error=experiments/results/cvpr_json_val/slurm_%j.err

set -euo pipefail

export MEDSAM3_ROOT="${MEDSAM3_ROOT:-/anvil/projects/x-cis250950/sam3}"
export MEDSAM3_DATA_ROOT="${MEDSAM3_DATA_ROOT:-/anvil/projects/x-cis250950}"
export MEDSAM3_CVPR_ROOT="${MEDSAM3_CVPR_ROOT:-${MEDSAM3_DATA_ROOT}/converted_cvpr_biomedsegfm}"

cd "$MEDSAM3_ROOT"
CKPT=experiments/medsam3_stage1_train_all_unified/checkpoints/checkpoint.pt

python medical/evaluation/cvpr_json_val.py \
  --ckpt "$CKPT" \
  --ann-dir annotations/unified_sam3_annotations/cvpr_train10p \
  --ann-glob '*one_per_cat*sam3_video.json' \
  --videos-per-dataset 2 --max-frames 32 \
  --out-dir experiments/results/cvpr_train10p_eval \
  --run-tag unified

python medical/evaluation/cvpr_json_val.py \
  --ckpt "$CKPT" \
  --ann-dir annotations/cvpr_biomedsegfm/val_external \
  --ann-glob 'val_*.json' \
  --videos-per-dataset 15 --max-frames 32 \
  --out-dir experiments/results/cvpr_val_external_eval \
  --run-tag unified_val
