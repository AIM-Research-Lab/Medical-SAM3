# Medical-SAM3 3D inference

This entry point runs text-prompted inference on complete 3D volumes without
using ground-truth masks, boxes, points, or ground-truth-positive slice ranges.
It treats the depth axis as video time and saves a label volume with the same
`(D, H, W)` geometry.

## Install

Inference requires an NVIDIA CUDA GPU.

```bash
git clone https://github.com/AIM-Research-Lab/Medical-SAM3.git
cd Medical-SAM3
pip install -e .
```

Download a Medical-SAM3 3D checkpoint from the
[model page](https://huggingface.co/Chongcong/Medical-SAM3).

## Input

Each `.npz` file must contain an `imgs` array in one of these layouts:

- `(D, H, W)` grayscale
- `(D, H, W, 1|3)` channel-last
- `(D, 1|3, H, W)` channel-first

Prompts may be embedded as an object-scalar dictionary named `text_prompts`:

```python
import numpy as np

np.savez_compressed(
    "case.npz",
    imgs=volume,
    text_prompts={"1": "liver", "2": "spleen", "instance_label": 0},
)
```

Non-`uint8` slices are independently min-max normalized. If modality-specific
windowing matters (for example CT), apply it before writing the NPZ.

## Run

Embedded prompts:

```bash
python medical/inference_3d.py \
  --input /path/to/case.npz \
  --output-dir outputs/3d \
  --checkpoint /path/to/medical_sam3_3d.pt
```

One prompt supplied on the command line:

```bash
python medical/inference_3d.py \
  --input /path/to/npz_directory \
  --output-dir outputs/3d \
  --checkpoint /path/to/medical_sam3_3d.pt \
  --prompt '1=liver'
```

Repeat `--prompt` for semantic multiclass output. Add `--instance` when a
single prompt should produce a binary instance-style mask.

For different prompts per case, pass a JSON file keyed by NPZ filename or stem:

```json
{
  "case_001.npz": {"1": "liver", "2": "spleen", "instance_label": 0},
  "case_002": {"1": "pancreas", "instance_label": 0}
}
```

```bash
python medical/inference_3d.py \
  --input /path/to/npz_directory \
  --output-dir outputs/3d \
  --checkpoint /path/to/medical_sam3_3d.pt \
  --prompts-json prompts.json \
  --continue-on-error
```

Each output NPZ contains `segs`; `inference_summary.json` records prompts,
scores, runtime, and any per-case failure. The released policy uses independent
four-slice clips and per-pixel probability fusion. The corresponding defaults
are `--clip-length 4 --fusion pixel_score`.
