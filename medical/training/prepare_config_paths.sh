#!/usr/bin/env bash
# Rewrite cluster-specific paths in the unified training config for your environment.
set -euo pipefail

: "${MEDSAM3_ROOT:?Set MEDSAM3_ROOT to the repository root}"
: "${MEDSAM3_CVPR_ROOT:=${MEDSAM3_DATA_ROOT:-$(dirname "$MEDSAM3_ROOT")}/converted_cvpr_biomedsegfm}"

OLD_ROOT="${OLD_ROOT:-/anvil/projects/x-cis250950}"
OLD_SAM3="${OLD_ROOT}/sam3"
OLD_CVPR="${OLD_ROOT}/converted_cvpr_biomedsegfm"
OLD_DATA="${OLD_ROOT}/dataset/datasetavail"

MEDICAL_DATA="${MEDSAM3_DATA_ROOT:-$(dirname "$MEDSAM3_ROOT")}/dataset/datasetavail"

SRC="${MEDSAM3_ROOT}/sam3/train/configs/medsam3_stage1_train_all_unified.yaml"
DST="${MEDSAM3_ROOT}/sam3/train/configs/medsam3_stage1_train_all_unified.local.yaml"

sed -e "s|${OLD_SAM3}|${MEDSAM3_ROOT}|g" \
    -e "s|${OLD_CVPR}|${MEDSAM3_CVPR_ROOT}|g" \
    -e "s|${OLD_DATA}|${MEDICAL_DATA}|g" \
    "$SRC" > "$DST"

echo "Wrote ${DST}"
echo "Train with: python sam3/train/train.py -c configs/medsam3_stage1_train_all_unified.local --num-gpus N"
