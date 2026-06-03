# Medical-SAM3

Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation

## Setup

```bash
# Create conda environment
conda create -n medsam3 python=3.10
conda activate medsam3

# From repository root (parent of inference/)
cd ..
pip install -r requirements.txt
pip install -e ".[train]"   # SAM3 + training deps (bundled sam3/)
```

## Evaluation

### Base SAM3

```bash
python run_evaluation.py
```

### MedSAM3

```bash
python run_medsam3_evaluation.py \
    --checkpoint /path/to/checkpoint.pt \
    --model-name medsam3
```

**Options:** `--max-samples N`, `--datasets "Dataset1,Dataset2"`

## Visualization

```bash
cd visualization
python visualize_all_datasets.py
```

## Output

- **Results:** `results/*.csv`, `*.json`, `*.md`
- **Visualizations:** `visualization/{dataset}/comparison_*.png`

## Datasets

**Data path:** `../medsam_data/`
