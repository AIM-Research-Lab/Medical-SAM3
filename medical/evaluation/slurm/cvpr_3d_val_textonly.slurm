#!/bin/bash
#SBATCH --job-name=medsam3_3dval
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=experiments/results/cvpr_3d_val/slurm_%j.log
#SBATCH --error=experiments/results/cvpr_3d_val/slurm_%j.err

set -euo pipefail

export MEDSAM3_ROOT="${MEDSAM3_ROOT:-/anvil/projects/x-cis250950/sam3}"
export MEDSAM3_DATA_ROOT="${MEDSAM3_DATA_ROOT:-/anvil/projects/x-cis250950}"
export MEDSAM3_CVPR_ROOT="${MEDSAM3_CVPR_ROOT:-${MEDSAM3_DATA_ROOT}/converted_cvpr_biomedsegfm}"

cd "$MEDSAM3_ROOT"
OUT=experiments/results/cvpr_3d_val
CKPT=experiments/medsam3_stage1_train_all_unified/checkpoints/checkpoint.pt
mkdir -p "$OUT"

python medical/evaluation/cvpr_3d_val.py \
  --ckpt "$CKPT" \
  --out-dir "$OUT/unified_textonly" \
  --run-tag unified_textonly \
  --eval-all-volumes --text-only --use-train-prompts

python medical/evaluation/cvpr_3d_val.py \
  --load-from-hf \
  --out-dir "$OUT/base_textonly" \
  --run-tag base_textonly \
  --eval-all-volumes --text-only --use-train-prompts
