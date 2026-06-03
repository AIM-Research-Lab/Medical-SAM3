# MedSAM3 evaluation

Run from repository root with `MEDSAM3_*` env vars set (`medical/README.md`).

## CVPR 3D val

```bash
python medical/evaluation/cvpr_3d_val.py \
  --ckpt experiments/medsam3_stage1_train_all_unified/checkpoints/checkpoint.pt \
  --out-dir experiments/results/cvpr_3d_val/unified_textonly \
  --run-tag unified_textonly \
  --eval-all-volumes --text-only --use-train-prompts
```

Base SAM3: add `--load-from-hf` and omit `--ckpt`.

## CVPR JSON

```bash
python medical/evaluation/cvpr_json_val.py \
  --ann-dir annotations/unified_sam3_annotations/cvpr_train10p \
  --ann-glob '*one_per_cat*sam3_video.json' \
  --videos-per-dataset 2 --max-frames 32 \
  --out-dir experiments/results/cvpr_train10p_eval --run-tag unified
```

Val external: `--ann-dir annotations/cvpr_biomedsegfm/val_external --ann-glob 'val_*.json'`

## SLURM

`medical/evaluation/slurm/cvpr_3d_val_textonly.slurm` · `cvpr_json_val.slurm`
