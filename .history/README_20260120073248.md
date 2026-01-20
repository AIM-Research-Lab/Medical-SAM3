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
* **[2026-01-20]**: 🚀 Pretrained weights for Medical-SAM3 are released!
* **[2026-01-15]**: 📄 Paper is available on arXiv.

## ⚡ Inference & Evaluation

We provide a comprehensive toolkit to run **inference** on diverse medical datasets (e.g., CHASE_DB1, Synapse, etc.).

The inference pipeline supports:
* **📊 Model Evaluation**: Run Medical-SAM3 on supported datasets with a single command.
* **⚖️ Baseline Comparison**: Compare performance against the vanilla SAM3 or other baselines.
* **🖼️ Visualization**: Generate and save segmentation masks for qualitative analysis.

<div align="left">
  <a href="./inference/README.md">
    <img src="https://img.shields.io/badge/📖-Read_Full_Evaluation_Guide-blue?style=for-the-badge&logo=markdown">
  </a>
</div>

## 📅 Todo List

| Feature | Status | Description |
| :--- | :---: | :--- |
| **Demo** | 🚧 Doing | Online interactive demo. |
| **Data Scaling** | 🚧 Doing | Significantly expand the training corpus and evaluate on broader and more diverse medical datasets. |
| **Training Code** |  📅 Planned | Release full training scripts and data construction guidelines. |
| **Medical-SAM3 Agent** | 📅 Planned | Integrate LLMs to enable agentic reasoning and interaction for segmentation tasks. |

<br>
<p align="left">
  <i>📢 We are actively updating this repository. If you are interested in any features above, feel free to open an issue!</i>
</p>

## 📝 Citation

If you find Medical-SAM3 useful for your research or work, please consider citing our paper:

```bibtex
@article{jiang2026medicalsam3,
  title={Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation},
  author={Jiang, Chongcong and Ding, Tianxingjian and Song, Chuhan and Tu, Jiachen and Yan, Ziyang and Shao, Yihua and Wang, Zhenyi and Shang, Yuzhang and Han, Tianyu and Tian, Yu},
  journal={arXiv preprint arXiv:2601.10880},
  year={2026},
  url={https://arxiv.org/abs/2601.10880}
}