# MedSAM3 — training & evaluation

All train/eval code lives here. Checkpoints and metrics go under `experiments/` (runtime only).

## Layout

```
medical/
├── paths.py
├── training/
│   ├── README.md
│   ├── prepare_config_paths.sh
│   └── slurm/train_unified.slurm
└── evaluation/
    ├── README.md
    ├── eval_common.py
    ├── cvpr_3d_val.py
    ├── cvpr_json_val.py
    └── slurm/
```

## Environment

```bash
export MEDSAM3_ROOT=/path/to/this/repo
export MEDSAM3_DATA_ROOT=/path/to/project
export MEDSAM3_CVPR_ROOT=$MEDSAM3_DATA_ROOT/converted_cvpr_biomedsegfm
```

## Training

```bash
cd "$MEDSAM3_ROOT"
bash medical/training/prepare_config_paths.sh   # once, if paths differ from yaml defaults
python sam3/train/train.py -c configs/medsam3_stage1_train_all_unified --num-gpus 4
```

Config: `sam3/train/configs/medsam3_stage1_train_all_unified.yaml`  
Details: `medical/training/README.md`

## Evaluation

| Benchmark | Script |
|-----------|--------|
| CVPR 3D val | `medical/evaluation/cvpr_3d_val.py` |
| CVPR JSON | `medical/evaluation/cvpr_json_val.py` |

Details: `medical/evaluation/README.md`
