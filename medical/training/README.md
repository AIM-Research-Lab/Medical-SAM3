# MedSAM3 training

## Prerequisites

- CVPR BiomedSegFM converted data: `converted_cvpr_biomedsegfm/` with `annotations/unified_sam3_annotations/cvpr_train_all/*.json`
- NPZ volumes referenced by those JSONs (under `converted_cvpr_biomedsegfm` or `dataset/datasetavail`)
- SAM3 BPE vocab: `assets/bpe_simple_vocab_16e6.txt.gz`
- Base or stage-0 checkpoint (config sets `checkpoint_path`)

## Configure paths

The unified YAML contains absolute paths from the development cluster. Patch them before training:

```bash
export MEDSAM3_ROOT=/path/to/repo
export MEDSAM3_DATA_ROOT=/path/to/project
export MEDSAM3_CVPR_ROOT=$MEDSAM3_DATA_ROOT/converted_cvpr_biomedsegfm

bash medical/training/prepare_config_paths.sh
```

This writes `sam3/train/configs/medsam3_stage1_train_all_unified.local.yaml` (gitignored pattern: use the generated file or copy over the default name).

Alternatively override the top-level `paths:` block via Hydra:

```bash
cd "$MEDSAM3_ROOT"
python sam3/train/train.py -c configs/medsam3_stage1_train_all_unified \
  paths.experiment_log_dir="$MEDSAM3_ROOT/experiments/medsam3_stage1_train_all_unified" \
  paths.bpe_path="$MEDSAM3_ROOT/assets/bpe_simple_vocab_16e6.txt.gz" \
  paths.medical_data_root="$MEDSAM3_DATA_ROOT/dataset/datasetavail" \
  --num-gpus 4
```

Note: per-dataset `ann_file` / `img_folder` entries inside the YAML still need `prepare_config_paths.sh` unless you regenerate annotations with the same root.

## Run training

```bash
cd "$MEDSAM3_ROOT"
python sam3/train/train.py \
  -c configs/medsam3_stage1_train_all_unified \
  --num-gpus 4
```

Key training settings (see YAML `scratch` / `medical_train`):

- `num_stages_sample: 8` — 8-frame video windows
- `TextQueryToVisual` probability 0.5 — text + noisy GT box on half the steps
- Short `query_text` labels (see eval vocab in `medical/evaluation/eval_common.py`)

## SLURM (Anvil example)

Edit account/partition in `medical/training/slurm/train_unified.slurm`, then:

```bash
sbatch medical/training/slurm/train_unified.slurm
```

## Checkpoints

Default log dir: `experiments/medsam3_stage1_train_all_unified/checkpoints/checkpoint.pt`

Upload helper (optional): `scripts/upload_checkpoints_to_hf.py`.
