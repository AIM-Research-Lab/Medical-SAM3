# Runtime outputs (not source code)

This directory holds checkpoints and evaluation results created at run time. It is not part of the MedSAM3 source tree.

| Path | Purpose |
|------|---------|
| `medsam3_stage1_train_all_unified/` | Training logs and `checkpoints/checkpoint.pt` |
| `sam3_stage0.pt` | Optional init weights (see training config) |
| `results/` | Evaluation outputs (`summary.json`, CSV) |

Train and eval entry points: `medical/README.md` and `README_MEDSAM3.md`.
