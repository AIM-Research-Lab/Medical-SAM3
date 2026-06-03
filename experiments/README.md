# Runtime outputs (do not commit)

Everything under this folder is created locally when you train or evaluate. **None of it belongs in git.**

After training:

```
medsam3_stage1_train_all_unified/
  checkpoints/checkpoint.pt    # your fine-tuned weights
  logs/
  tensorboard/
```

After evaluation:

```
results/
  cvpr_3d_val/...
  cvpr_train10p_eval/...
```

Optional init weights (download or copy locally): `sam3_stage0.pt`

Entry points: `medical/README.md` · `README.md`
