#!/usr/bin/env python3
"""3D val — training-aligned protocol (8-frame window, text-only by default).

Usage (from repo root):
  export MEDSAM3_ROOT=$(pwd)
  export MEDSAM3_CVPR_ROOT=/path/to/converted_cvpr_biomedsegfm
  python medical/evaluation/cvpr_3d_val.py \\
    --ckpt experiments/medsam3_stage1_train_all_unified/checkpoints/checkpoint.pt \\
    --out-dir experiments/results/cvpr_3d_val \\
    --eval-all-volumes --text-only --use-train-prompts
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from medical.evaluation.eval_common import (
    BPE,
    DEFAULT_CKPT,
    REPO,
    SAM3,
    VAL_GT,
    VAL_NPZ,
    dice_iou,
    ensure_sam3_on_path,
    group_files,
    load_val_npz,
    slice_to_rgb,
    val_text_to_train_query,
)


def init_frame_from_gt(gt_full: np.ndarray, obj_id: int) -> int | None:
    mask = gt_full == obj_id
    if not mask.any():
        return None
    z_present = np.where(mask.any(axis=(1, 2)))[0]
    z_min = int(z_present[0])
    z_max = int(z_present[-1])
    areas = mask[z_min : z_max + 1].reshape(z_max - z_min + 1, -1).sum(axis=1)
    return z_min + int(np.argmax(areas))


def sample_train_like_frames(
    D: int,
    init_frame: int,
    num_stages: int,
    stage_stride: int,
    rng: random.Random | None,
) -> tuple[list[int], int] | None:
    """Return (frame_indices, init_local) matching training subsampling style."""
    if D < 2:
        return None
    num_stages = min(num_stages, D)
    all_ids = list(range(D))

    if num_stages >= D:
        init_local = max(0, min(init_frame, D - 1))
        return all_ids, init_local

    gap = (num_stages - 1) * stage_stride
    if gap > D - 1:
        stage_stride = max(1, math.floor((D - 1) / (num_stages - 1)))
        gap = (num_stages - 1) * stage_stride

    b_max = D - 1 - gap
    if rng is not None:
        # Training: random start among valid windows
        b = rng.randint(0, b_max)
    else:
        # Eval default: center window on init_frame
        b = init_frame - gap // 2
        b = max(0, min(b, b_max))

    e = b + gap
    indices = all_ids[b : e + 1 : stage_stride]
    if len(indices) < 2:
        return None

    init_local = init_frame - indices[0]
    if init_local < 0 or init_local >= len(indices):
        init_local = min(len(indices) - 1, max(0, len(indices) // 2))
    return indices, init_local


def bbox_xywh_norm_from_mask(mask2d: np.ndarray, w: int, h: int) -> np.ndarray | None:
    ys, xs = np.where(mask2d)
    if len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return np.array(
        [x0 / w, y0 / h, (x1 - x0 + 1) / w, (y1 - y0 + 1) / h],
        dtype=np.float32,
    )


def eval_one_volume_train_protocol(
    video_model,
    vol_path: Path,
    gt_path: Path,
    obj_id: int,
    num_stages: int,
    stage_stride: int,
    text_only: bool,
    use_train_prompts: bool,
    train_random_window: bool,
    seed: int,
) -> dict:
    res = {
        "file": vol_path.name,
        "obj_id": obj_id,
        "status": "skip",
        "reason": "",
        "protocol": "train_like_8frame_gt_bbox",
    }

    vol = load_val_npz(vol_path)
    imgs = vol["imgs"]
    D, H, W = int(imgs.shape[0]), int(imgs.shape[1]), int(imgs.shape[2])

    gt_full = load_val_npz(gt_path).get("gts")
    if gt_full is None:
        res["reason"] = "no gt"
        return res
    if hasattr(gt_full, "ndim") and gt_full.ndim == 0:
        gt_full = gt_full.item()

    tp = vol.get("text_prompts")
    if tp is not None:
        try:
            tp = tp.item() if getattr(tp, "ndim", 0) == 0 else tp
        except Exception:
            pass
    qt = tp.get(str(obj_id)) if isinstance(tp, dict) else None
    if not isinstance(qt, str) or not qt.strip():
        res["reason"] = f"no text for obj {obj_id}"
        return res
    if use_train_prompts:
        mapped = val_text_to_train_query(qt)
        if mapped is None:
            res["reason"] = f"no train vocab match for '{qt[:60]}'"
            return res
        res["query_text_original"] = qt
        qt = mapped
    res["query_text"] = qt

    init_frame = init_frame_from_gt(gt_full, obj_id)
    if init_frame is None:
        res["reason"] = f"no gt label {obj_id}"
        return res

    rng = random.Random(seed + hash(vol_path.name) % 100000) if train_random_window else None
    sampled = sample_train_like_frames(D, init_frame, num_stages, stage_stride, rng)
    if sampled is None:
        res["reason"] = "too few frames"
        return res
    frame_indices, init_local = sampled
    res["n_frames"] = len(frame_indices)
    res["init_frame"] = init_frame
    res["init_local"] = init_local
    res["frame_indices"] = frame_indices
    res["window_mode"] = "random" if train_random_window else "center_on_init"

    gt_mask_init = (gt_full[init_frame] == obj_id)
    box_xywh_norm = bbox_xywh_norm_from_mask(gt_mask_init, W, H)
    if box_xywh_norm is None:
        res["reason"] = "empty mask on init frame"
        return res

    frames = [slice_to_rgb(imgs, z) for z in frame_indices]

    with tempfile.TemporaryDirectory(prefix="eval_val3d_train_") as td:
        mp4 = Path(td) / "clip.mp4"
        wr = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (W, H))
        for fr in frames:
            wr.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        wr.release()

        with torch.inference_mode():
            st = video_model.init_state(resource_path=str(mp4))
            prompt_kw = dict(
                inference_state=st,
                frame_idx=init_local,
                text_str=qt,
            )
            if not text_only:
                prompt_kw["boxes_xywh"] = box_xywh_norm.reshape(1, 4)
                prompt_kw["box_labels"] = torch.ones(1, dtype=torch.long)
            _fi, prompt_out = video_model.add_prompt(**prompt_kw)

            pred_masks: dict[int, np.ndarray | None] = {}

            def _collect(gen):
                for fidx, post_out in gen:
                    if post_out is None:
                        continue
                    masks = post_out.get("out_binary_masks")
                    if masks is not None and masks.shape[0] > 0:
                        m = np.any(masks, axis=0)
                        if fidx not in pred_masks or (
                            pred_masks[fidx] is None or m.sum() > pred_masks[fidx].sum()
                        ):
                            pred_masks[fidx] = m
                    elif fidx not in pred_masks:
                        pred_masks[fidx] = None

            n_local = len(frames)
            _collect(
                video_model.propagate_in_video(
                    st, start_frame_idx=0, max_frame_num_to_track=n_local, reverse=False
                )
            )
            _collect(
                video_model.propagate_in_video(
                    st,
                    start_frame_idx=n_local - 1,
                    max_frame_num_to_track=n_local,
                    reverse=True,
                )
            )

    dices, ious = [], []
    dice_by_local: dict[int, float] = {}
    for li, z in enumerate(frame_indices):
        gt_slice = gt_full[z] == obj_id
        pm = pred_masks.get(li)
        if pm is not None and pm.shape != gt_slice.shape:
            pm = cv2.resize(
                pm.astype(np.uint8),
                (gt_slice.shape[1], gt_slice.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if not gt_slice.any() and (pm is None or not pm.any()):
            continue
        d, io = dice_iou(gt_slice, pm)
        dices.append(d)
        ious.append(io)
        dice_by_local[li] = d

    if not dices:
        res["reason"] = "no scored slices"
        return res

    res.update(
        status="ok",
        prompt_mode="text_only" if text_only else "text+gt_bbox",
        mean_dice=float(np.mean(dices)),
        mean_iou=float(np.mean(ious)),
        dice_init=float(dice_by_local.get(init_local, float("nan"))),
        n_scored=len(dices),
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--load-from-hf", action="store_true")
    ap.add_argument("--bpe", default=str(BPE))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-tag", default="train_protocol")
    ap.add_argument("--vols", type=int, default=1)
    ap.add_argument("--eval-all-volumes", action="store_true")
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--num-stages", type=int, default=8,
                    help="Match training num_stages_sample (default 8)")
    ap.add_argument("--stage-stride", type=int, default=1)
    ap.add_argument("--text-only", action="store_true",
                    help="Ablation: no GT bbox at prompt (training uses box ~50%%)")
    ap.add_argument("--use-train-prompts", action="store_true")
    ap.add_argument("--train-random-window", action="store_true",
                    help="Random 8-frame window like training (else center on init)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-det-threshold", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ensure_sam3_on_path()
    from sam3.model_builder import build_sam3_video_model

    ckpt_path = None if args.load_from_hf else args.ckpt
    ckpt_label = "HuggingFace (base SAM3)" if args.load_from_hf else args.ckpt
    print(f"[{args.run_tag}] train-protocol eval | ckpt={ckpt_label}")
    print(
        f"  num_stages={args.num_stages} stride={args.stage_stride} "
        f"prompt={'text-only' if args.text_only else 'text+gt_bbox'} "
        f"window={'random' if args.train_random_window else 'center'}"
    )

    video_model = build_sam3_video_model(
        checkpoint_path=ckpt_path,
        load_from_HF=args.load_from_hf,
        bpe_path=args.bpe,
        strict_state_dict_loading=False,
        training_mode=False,
        device="cuda",
        compile=False,
        apply_temporal_disambiguation=True,
    )
    video_model.eval()
    video_model.hotstart_delay = 0
    video_model.hotstart_unmatch_thresh = 0
    video_model.hotstart_dup_thresh = 0
    if args.no_det_threshold:
        video_model.score_threshold_detection = 0.0
        video_model.new_det_thresh = 0.0

    val_files = sorted(VAL_NPZ.glob("*.npz"))
    groups = group_files(val_files)
    if args.groups:
        groups = {k: v for k, v in groups.items() if k in set(args.groups)}

    work_items = []
    if args.eval_all_volumes:
        for grp, files in sorted(groups.items()):
            for vol in files:
                work_items.append((grp, vol))
    else:
        for grp, files in sorted(groups.items()):
            for vol in files[: args.vols]:
                work_items.append((grp, vol))

    print(f"[{args.run_tag}] volumes={len(work_items)}")

    rows = []
    t0 = time.time()
    for wi, (grp, vol) in enumerate(work_items, 1):
        if wi == 1 or wi % 50 == 0 or wi == len(work_items):
            print(f"\n--- {wi}/{len(work_items)} {grp} ---")
        gt = VAL_GT / vol.name
        if not gt.exists():
            rows.append({"group": grp, "file": vol.name, "status": "skip", "reason": "gt_missing"})
            continue
        try:
            r = eval_one_volume_train_protocol(
                video_model,
                vol,
                gt,
                1,
                args.num_stages,
                args.stage_stride,
                args.text_only,
                args.use_train_prompts,
                args.train_random_window,
                args.seed,
            )
        except Exception as e:
            traceback.print_exc()
            r = {"status": "error", "reason": str(e), "file": vol.name}
        r["group"] = grp
        rows.append(r)
        if r.get("status") == "ok" and (wi <= 3 or wi % 100 == 0):
            print(
                f"  {vol.name}: dice={r['mean_dice']:.4f} "
                f"n={r['n_scored']}/{r['n_frames']} init_z={r['init_frame']}"
            )

    elapsed = time.time() - t0
    ok = [r for r in rows if r.get("status") == "ok"]
    md = float(np.mean([r["mean_dice"] for r in ok])) if ok else float("nan")
    print(f"\n[{args.run_tag}] ok={len(ok)}/{len(rows)} Dice={md:.4f} ({elapsed/60:.1f}min)")

    csv_path = out / "results_per_volume.csv"
    with open(csv_path, "w", newline="") as f:
        keys = [
            "group", "file", "status", "reason", "query_text", "prompt_mode",
            "n_frames", "n_scored", "init_frame", "mean_dice", "mean_iou", "dice_init",
        ]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "run_tag": args.run_tag,
        "protocol": "train_like",
        "num_stages": args.num_stages,
        "stage_stride": args.stage_stride,
        "text_only": args.text_only,
        "train_random_window": args.train_random_window,
        "n_ok": len(ok),
        "dice": md,
        "elapsed": elapsed,
        "rows": rows,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
