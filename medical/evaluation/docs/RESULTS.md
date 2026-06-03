# MedSAM3 · 3D Val 评估结果（训练对齐 · text-only）

> Job **17867500** · 协议：8 帧连续窗口 + 纯 text prompt  
> 脚本：`eval_val3d_train_protocol.py`

---

## 1. 评估协议

与 Unified 训练一致的数据用法，**不读** val npz 的 `boxes`：

| 步骤 | 做法 |
|------|------|
| 帧范围 | 整卷 `@0…@D-1`，每个 volume **只取 1 个** 连续 **8 帧** 窗口 |
| 窗口位置 | GT mask 面积最大的 slice 为中心 |
| Prompt | **仅 text**（`--text-only`），映射为训练短词 |
| 推理 | 在 init 帧 `add_prompt(text)` → 双向 propagate |
| 指标 | 该 8 帧内逐层 Dice **平均** |

**Prompt 来源**

- Val npz 长句 → `--use-train-prompts` → 训练类别短词（如 `"Spleen"`）
- 与训练 JSON 的 `query_text` 一致

**数据**

- 图像：`3D_val_npz`（2145 volumes）
- GT：`3D_val_gt/3D_val_gt_interactive`

---

## 2. 总体结果

| 模型 | ok / 2145 | Mean Dice |
|------|-----------|-----------|
| **Unified**（微调） | **1755** | **0.411** |
| **Base**（预训练 SAM3） | **1755** | 0.160 |

- 同一批 **1755** 例上对比；Unified 明显优于 Base  
- 每例在 **8/8 帧** 上评分（`n=8/8`）  
- Job 耗时约 **37 min**（Unified + Base）

**未纳入（390）**

| 原因 | N |
|------|---|
| 长句无法映射到训练短词 | 371 |
| npz 无 text | 19 |

---

## 3. Unified vs Base

| | Unified | Base |
|--|---------|------|
| Mean Dice | **0.411** | 0.160 |
| 相对提升 | — | Unified 约为 Base 的 **2.6×** |

text-only 设定下，Unified 微调在 3D val 上带来 **稳定、大幅** 提升。

---

## 4. 结果文件

| 用途 | 路径 |
|------|------|
| Unified summary | `experiments/results/val3d_train_protocol_textonly/unified/summary.json` |
| Base summary | `experiments/results/val3d_train_protocol_textonly/base/summary.json` |
| 逐例 CSV | 同上目录 `results_per_volume.csv` |
| 日志 | `experiments/logs/val3d_trainproto_to_17867500.out` |

---

## 5. 复现

```bash
cd /anvil/projects/x-cis250950/sam3
PYTHON=/anvil/projects/x-cis250950/tding/envs/sam3/bin/python
SCRIPT=experiments/medsam3_stage1_train10p_video_test/eval_val3d_train_protocol.py

# Unified
$PYTHON $SCRIPT \
  --ckpt experiments/medsam3_stage1_train_all_unified/checkpoints/checkpoint.pt \
  --out-dir experiments/results/val3d_train_protocol_textonly/unified \
  --run-tag unified_trainproto_textonly \
  --eval-all-volumes --num-stages 8 --text-only --use-train-prompts

# Base
$PYTHON $SCRIPT --load-from-hf \
  --out-dir experiments/results/val3d_train_protocol_textonly/base \
  --run-tag base_trainproto_textonly \
  --eval-all-volumes --num-stages 8 --text-only --use-train-prompts
```

SLURM：`sbatch experiments/slurm_3dval_train_protocol_textonly.sh`

---

## 6. 汇报要点（一句话）

**在 CVPR 3D val（2145 volumes）上，采用与训练对齐的 8 帧 text-only 协议，Unified 微调 Dice 0.41，Base 0.16，1755 例有效评估。**
