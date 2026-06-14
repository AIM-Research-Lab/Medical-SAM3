<div align="center">
  
  <h1>🏥 Medical-SAM3</h1>
  
  <a href="https://github.com/AIM-Research-Lab/Medical-SAM3">
    <img src="./assests/overview.svg" width="100%" alt="Medical-SAM3 Teaser">
  </a>

  <h3>A Foundation Model for Universal Prompt-Driven Medical Image Segmentation</h3>

  <p align="center">
  <a href="https://arxiv.org/abs/2601.10880"><img src="https://img.shields.io/badge/arXiv-2601.10880-b31b1b?style=flat-square&logo=arxiv"></a>&nbsp;<a href="https://chongcongjiang.github.io/MedicalSAM3/"><img src="https://img.shields.io/badge/Website-Project%20Page-blue?style=flat-square&logo=google-chrome"></a>&nbsp;<a href="https://huggingface.co/Chongcong/Medical-SAM3"><img src="https://img.shields.io/badge/Hugging%20Face-Models-yellow?style=flat-square&logo=huggingface"></a>
  </p>

</div>

## 📰 News

* **[2026-06]**: 🎓 Medical SAM 3 V2 **training & evaluation** code released under `medical/`.
* **[2026-01-20]**: 🚀 Pretrained weights for Medical-SAM3 are released!
* **[2026-01-15]**: 📄 Paper is available on arXiv.

## ⚡ Inference & evaluation (2D medical benchmarks)

Toolkit for **2D** datasets (CHASE_DB1, STARE, CVC-ClinicDB, etc.) — box/text prompts, baseline comparison, visualization.

<a href="./inference/README.md"><img src="https://img.shields.io/badge/📖-2D_Inference_Guide-blue?style=for-the-badge&logo=markdown"></a>

```bash
cd inference
python run_medsam3_evaluation.py --checkpoint /path/to/checkpoint.pt --model-name medsam3
```

SAM3 is bundled in this repo (`sam3/`); no separate clone required.

## 🎬 Training & 3D evaluation

Fine-tune SAM3 on **3D** annotations and run held-out **3D / JSON** eval (train-aligned protocol).

<a href="./medical/README.md"><img src="https://img.shields.io/badge/📖-Training_&_CVPR_Eval_Guide-blue?style=for-the-badge&logo=markdown"></a>

```bash
export MEDSAM3_ROOT=$(pwd)
export MEDSAM3_DATA_ROOT=/path/to/project
export MEDSAM3_CVPR_ROOT=$MEDSAM3_DATA_ROOT/converted_cvpr_biomedsegfm

pip install -e ".[train]"
bash medical/training/prepare_config_paths.sh   # once, if yaml paths differ

# Train
python sam3/train/train.py -c configs/medsam3_stage1_train_all_unified --num-gpus 4

# Eval 
python medical/evaluation/3d_val.py \
  --ckpt experiments/medsam3_stage1_train_all_unified/checkpoints/checkpoint.pt \
  --out-dir experiments/results/3d_val/unified \
  --eval-all-volumes --text-only --use-train-prompts
```

| Path | Role |
|------|------|
| `inference/` | 2D image inference & public benchmark eval |
| `medical/` | 3D **training** + **3D/JSON evaluation** |
| `sam3/` | SAM3 model, trainer, Hydra config |
| `assets/` | BPE vocabulary |
| `experiments/` | Checkpoints & eval outputs (runtime) |

## 📅 Todo List

| Feature | Status | Description |
| :--- | :---: | :--- |
| **Demo** | 🚧 Doing | Online interactive demo. |
| **Data Scaling** | 🚧 Doing | Expand training corpus and benchmarks. |
| **3D Training** | ✅ Released | `medical/` + `medsam3_stage1_train_all_unified` config. |
| **Medical-SAM3 Agent** | 📅 Planned | LLM agentic segmentation. |

## 📝 Citation

```bibtex
@article{jiang2026medicalsam3,
  title={Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation},
  author={Jiang, Chongcong and Ding, Tianxingjian and Song, Chuhan and Tu, Jiachen and Yan, Ziyang and Shao, Yihua and Wang, Zhenyi and Shang, Yuzhang and Han, Tianyu and Tian, Yu},
  journal={arXiv preprint arXiv:2601.10880},
  year={2026},
  url={https://arxiv.org/abs/2601.10880}
}
```
