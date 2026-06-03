#!/bin/bash
#SBATCH --job-name=medsam3_unified
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=experiments/medsam3_stage1_train_all_unified/slurm_%j.log
#SBATCH --error=experiments/medsam3_stage1_train_all_unified/slurm_%j.err

set -euo pipefail

export MEDSAM3_ROOT="${MEDSAM3_ROOT:-/anvil/projects/x-cis250950/sam3}"
export MEDSAM3_DATA_ROOT="${MEDSAM3_DATA_ROOT:-/anvil/projects/x-cis250950}"
export MEDSAM3_CVPR_ROOT="${MEDSAM3_CVPR_ROOT:-${MEDSAM3_DATA_ROOT}/converted_cvpr_biomedsegfm}"

cd "$MEDSAM3_ROOT"
mkdir -p experiments/medsam3_stage1_train_all_unified

CONFIG=configs/medsam3_stage1_train_all_unified
if [[ -f sam3/train/configs/medsam3_stage1_train_all_unified.local.yaml ]]; then
  CONFIG=configs/medsam3_stage1_train_all_unified.local
fi

python sam3/train/train.py -c "$CONFIG" --num-gpus 4
