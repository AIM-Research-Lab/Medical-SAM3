#!/usr/bin/env python3
"""Evaluate SAM3 video model on CVPR BiomedSegFM JSON annotations (train10p / val_external).

Protocol (matches training): init_state → add_prompt(text + GT box on query frame) →
bidirectional propagate_in_video.

Usage:
  # Train 10p (one_per_cat), 2 videos per dataset:
  python eval_cvpr_json_video.py \\
    --ann-dir converted_cvpr_biomedsegfm/annotations/unified_sam3_annotations/cvpr_train10p \\
    --ann-glob '*one_per_cat*sam3_video.json' \\
    --videos-per-dataset 2 --max-frames 32 \\
    --ckpt experiments/medsam3_stage1_train_all_unified/checkpoints/checkpoint.pt \\
    --out-dir experiments/results/cvpr_train10p_eval --run-tag unified_train

  # Val external (all val_*.json), more videos:
  python eval_cvpr_json_video.py \\
    --ann-dir converted_cvpr_biomedsegfm/annotations/cvpr_biomedsegfm/val_external \\
    --ann-glob 'val_*.json' \\
    --videos-per-dataset 15 --max-frames 32 \\
    --out-dir experiments/results/cvpr_val_external_eval --run-tag unified_val
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from medical.paths import bpe_path, cvpr_root, default_unified_ckpt, npz_resolve_roots, repo_root, sam3_package_dir
from medical.evaluation.eval_common import ensure_sam3_on_path

REPO = repo_root()
SAM3 = sam3_package_dir()
CONVERTED = cvpr_root()
BPE = bpe_path()
DEFAULT_CKPT = default_unified_ckpt()
ROOTS = npz_resolve_roots()


def parse_npz_at(s: str) -> tuple[str, int]:
    p, i = s.rsplit("@", 1)
    return p, int(i)


def resolve_npz_path(npz_rel: str, extra_roots: list[Path]) -> Path:
    p = Path(npz_rel)
    if p.is_absolute() and p.exists():
        return p
    for r in ROOTS + extra_roots:
        cand = r / npz_rel
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Cannot resolve npz: {npz_rel!r}")


def np_slice_to_rgb(npz_file: Path, slice_idx: int) -> np.ndarray:
    with np.load(npz_file, allow_pickle=True) as npz:
        arr = npz["imgs"][slice_idx] if "imgs" in npz else npz[npz.files[0]][slice_idx]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def rle_decode(rle: dict[str, Any]) -> np.ndarray:
    h, w = int(rle["size"][0]), int(rle["size"][1])
    counts = rle["counts"]
    vals: list[int] = []
    cur = 0
    for c in counts:
        vals.extend([cur] * int(c))
        cur = 1 - cur
    arr = np.array(vals, dtype=np.uint8)
    if arr.size != h * w:
        arr = np.pad(arr, (0, max(0, h * w - arr.size)))[: h * w]
    return arr.reshape((h, w), order="F").astype(bool)


def dice_iou(gt: np.ndarray, pred: np.ndarray | None) -> tuple[float, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool) if pred is not None else np.zeros_like(gt)
    inter = int(np.logical_and(gt, pred).sum())
    union = int(np.logical_or(gt, pred).sum())
    iou = inter / union if union > 0 else 0.0
    denom = int(gt.sum()) + int(pred.sum())
    dice = 2 * inter / denom if denom > 0 else 0.0
    return dice, iou


def align_mask(mask: np.ndarray | None, h: int, w: int) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.squeeze(mask)
    if m.ndim != 2:
        return None
    m = m > 0
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def pick_queries_for_video(
    data: dict, vid: int, queries_per_video: int
) -> list[dict]:
    """Pick evaluation queries: one per unique query_text (earliest frame)."""
    img_to_frame = {}
    for a in data["annotations"]:
        if int(a.get("video_id", -1)) == vid:
            img_to_frame[int(a["image_id"])] = int(a["frame_index"])

    v_imgids = {int(a["image_id"]) for a in data["annotations"] if int(a.get("video_id", -1)) == vid}
    candidates = []
    for q in data.get("queries", []):
        iid = int(q["image_id"])
        if iid not in v_imgids:
            continue
        fi = img_to_frame.get(iid)
        if fi is None:
            continue
        candidates.append((fi, int(q.get("query_processing_order", 0)), q))

    if not candidates:
        return []

    candidates.sort(key=lambda x: (x[0], x[1]))
    seen_text: set[str] = set()
    picked: list[dict] = []
    for _fi, _ord, q in candidates:
        t = q.get("query_text", "")
        if t in seen_text:
            continue
        seen_text.add(t)
        picked.append(q)
        if queries_per_video > 0 and len(picked) >= queries_per_video:
            break
    return picked


def make_fallback_query_for_video(data: dict, vid: int) -> dict | None:
    """Build a robust fallback query from per-video annotations.

    This handles val_external JSONs where `query.image_id` can be reused across
    videos and cannot safely map a query to one specific video.
    """
    video = next((v for v in data.get("videos", []) if int(v.get("id", -1)) == vid), None)
    if video is None:
        return None

    anns = [a for a in data.get("annotations", []) if int(a.get("video_id", -1)) == vid]
    if not anns:
        return None

    # Prefer the earliest frame that has both segmentation and bbox.
    anns = sorted(anns, key=lambda a: int(a.get("frame_index", 10**9)))
    chosen = None
    for a in anns:
        if a.get("segmentation") and a.get("bbox"):
            chosen = a
            break
    if chosen is None:
        chosen = anns[0]

    q_text = str(video.get("query", "")).strip() or "foreground"
    obj_id = int(chosen.get("object_id", 1))
    init_frame = int(chosen.get("frame_index", 0))

    # Keep the same keys used by eval_one_query.
    return {
        "query_text": q_text,
        "image_id": int(chosen.get("image_id", 0)),
        "object_ids_output": [obj_id],
        "query_processing_order": 0,
        "init_frame": init_frame,
        "source": "fallback_video_query",
    }


def get_gt_mask(
    anns_by_frame: dict[int, list],
    frame_idx: int,
    object_ids: list[int] | None,
) -> np.ndarray | None:
    anns = anns_by_frame.get(frame_idx, [])
    gt_raw = None
    for a in anns:
        if object_ids is not None and int(a.get("object_id", -1)) not in object_ids:
            continue
        seg = a.get("segmentation")
        if not seg:
            continue
        m = rle_decode(seg)
        gt_raw = m if gt_raw is None else np.logical_or(gt_raw, m)
    return gt_raw


def get_box_xywh_norm(anns_by_frame: dict[int, list], frame_idx: int, object_ids: list[int] | None):
    anns = anns_by_frame.get(frame_idx, [])
    boxes = []
    for a in anns:
        if object_ids is not None and int(a.get("object_id", -1)) not in object_ids:
            continue
        b = a.get("bbox")
        if b and len(b) == 4:
            boxes.append(b)  # xywh normalized
    if not boxes:
        return None
    arr = np.array(boxes, dtype=np.float32)
    x0 = arr[:, 0].min()
    y0 = arr[:, 1].min()
    x1 = (arr[:, 0] + arr[:, 2]).max()
    y1 = (arr[:, 1] + arr[:, 3]).max()
    return np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)


def eval_one_query(
    video_model,
    data: dict,
    vid: int,
    query: dict,
    num_frames_cap: int,
    ann_parent: Path,
    text_only: bool = False,
) -> dict | None:
    video = next((v for v in data["videos"] if int(v["id"]) == vid), None)
    if video is None:
        return None

    query_text = query.get("query_text", "")
    object_ids = [int(x) for x in query.get("object_ids_output", [])]
    if not object_ids:
        object_ids = None

    anns_by_frame: dict[int, list] = {}
    img_to_frame: dict[int, int] = {}
    for a in data["annotations"]:
        if int(a.get("video_id", -1)) != vid:
            continue
        fi = int(a["frame_index"])
        anns_by_frame.setdefault(fi, []).append(a)
        img_to_frame[int(a["image_id"])] = fi

    init_frame = query.get("init_frame")
    if init_frame is not None:
        init_frame = int(init_frame)
    else:
        init_frame = img_to_frame.get(int(query["image_id"]))
    if init_frame is None:
        return None

    file_names = video["file_names"]
    n_total = len(file_names)
    if n_total < 2:
        return None

    # Window around init frame
    if num_frames_cap > 0 and n_total > num_frames_cap:
        half = num_frames_cap // 2
        f0 = max(0, init_frame - half)
        f1 = min(n_total - 1, f0 + num_frames_cap - 1)
        f0 = max(0, f1 - num_frames_cap + 1)
        frame_indices = list(range(f0, f1 + 1))
    else:
        frame_indices = list(range(n_total))

    init_local = frame_indices.index(init_frame)
    extra_roots = [ann_parent, CONVERTED, PROJECT]

    frames = []
    for fi in frame_indices:
        fn = file_names[fi]
        rel, sidx = parse_npz_at(fn)
        npz_path = resolve_npz_path(rel, extra_roots)
        frames.append(np_slice_to_rgb(npz_path, sidx))

    box_xywh = get_box_xywh_norm(anns_by_frame, init_frame, object_ids)
    if box_xywh is None and not text_only:
        return {"status": "skip", "reason": "no_bbox", "video_id": vid, "query_text": query_text}

    H, W = frames[0].shape[:2]

    try:
        with tempfile.TemporaryDirectory(prefix="cvpr_eval_") as td:
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
                    text_str=query_text,
                )
                if not text_only and box_xywh is not None:
                    prompt_kw["boxes_xywh"] = box_xywh.reshape(1, 4)
                    prompt_kw["box_labels"] = torch.ones(1, dtype=torch.long)
                _fi, prompt_out = video_model.add_prompt(**prompt_kw)

                max_prob = 0.0
                n_objs = 0
                if prompt_out is not None:
                    probs = prompt_out.get("out_probs")
                    bm = prompt_out.get("out_binary_masks")
                    if bm is not None:
                        n_objs = bm.shape[0]
                    if probs is not None and isinstance(probs, torch.Tensor) and probs.numel():
                        max_prob = float(probs.max().item())

                pred_masks: dict[int, np.ndarray | None] = {}

                def _collect(gen):
                    for fidx, post_out in gen:
                        if post_out is None:
                            continue
                        masks = post_out.get("out_binary_masks")
                        if masks is not None and masks.shape[0] > 0:
                            m = np.any(masks, axis=0)
                            if fidx not in pred_masks or m.sum() > (pred_masks[fidx].sum() if pred_masks[fidx] is not None else 0):
                                pred_masks[fidx] = m
                        elif fidx not in pred_masks:
                            pred_masks[fidx] = None

                n_local = len(frames)
                _collect(video_model.propagate_in_video(
                    st, start_frame_idx=0, max_frame_num_to_track=n_local, reverse=False,
                ))
                _collect(video_model.propagate_in_video(
                    st, start_frame_idx=n_local - 1, max_frame_num_to_track=n_local, reverse=True,
                ))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"status": "skip", "reason": "CUDA OOM", "video_id": vid, "query_text": query_text}
    except Exception as e:
        return {"status": "error", "reason": str(e), "video_id": vid, "query_text": query_text}

    dices, ious = [], []
    gt_annotated_frames = 0
    gt_positive_frames = 0
    pred_positive_frames = 0
    for li, global_fi in enumerate(frame_indices):
        gt_raw = get_gt_mask(anns_by_frame, global_fi, object_ids)
        if gt_raw is None:
            continue
        gt_annotated_frames += 1
        Hi, Wi = frames[li].shape[:2]
        gt = align_mask(gt_raw, Hi, Wi)
        pm = align_mask(pred_masks.get(li), Hi, Wi)
        if gt is None:
            continue
        if gt.any():
            gt_positive_frames += 1
        if pm is not None and pm.any():
            pred_positive_frames += 1
        if not gt.any() and (pm is None or not pm.any()):
            continue
        d, io = dice_iou(gt, pm)
        dices.append(d)
        ious.append(io)

    if not dices:
        if gt_annotated_frames == 0:
            detail = "no_gt_annotations_in_window"
        elif gt_positive_frames == 0:
            detail = "gt_sparse_no_positive_pixels_in_window"
        elif pred_positive_frames == 0 and n_objs == 0:
            detail = "detector_zero_objects_and_empty_propagation"
        elif pred_positive_frames == 0:
            detail = "propagation_all_empty_masks"
        else:
            detail = "other_no_scored_condition"
        return {
            "status": "skip",
            "reason": "no_scored_frames",
            "no_scored_detail": detail,
            "video_id": vid,
            "query_text": query_text,
            "n_objects": n_objs,
            "max_prob": max_prob,
            "gt_annotated_frames": gt_annotated_frames,
            "gt_positive_frames": gt_positive_frames,
            "pred_positive_frames": pred_positive_frames,
            "window_frames": len(frame_indices),
        }

    return {
        "status": "ok",
        "video_id": vid,
        "query_text": query_text,
        "init_frame": init_frame,
        "n_frames": len(dices),
        "n_objects": n_objs,
        "max_prob": max_prob,
        "gt_annotated_frames": gt_annotated_frames,
        "gt_positive_frames": gt_positive_frames,
        "pred_positive_frames": pred_positive_frames,
        "window_frames": len(frame_indices),
        "mean_dice": float(np.mean(dices)),
        "mean_iou": float(np.mean(ious)),
        "dice_init": dices[init_local] if init_local < len(dices) else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--load-from-hf", action="store_true")
    ap.add_argument("--bpe", default=str(BPE))
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--ann-glob", default="*one_per_cat*sam3_video.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-tag", default="run")
    ap.add_argument("--videos-per-dataset", type=int, default=2)
    ap.add_argument("--queries-per-video", type=int, default=1,
                    help="Unique query_texts per video (0 = all)")
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--max-datasets", type=int, default=0, help="0 = all matching JSONs")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--no-det-threshold", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    ann_dir = Path(args.ann_dir)
    if not ann_dir.is_absolute():
        for base in (CONVERTED, REPO):
            cand = base / ann_dir
            if cand.exists():
                ann_dir = cand
                break
        else:
            ann_dir = REPO / ann_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_sam3_on_path()
    from sam3.model_builder import build_sam3_video_model

    ckpt = None if args.load_from_hf else args.ckpt
    print(f"[{args.run_tag}] Loading model: {'HF base' if args.load_from_hf else args.ckpt}")
    video_model = build_sam3_video_model(
        bpe_path=str(args.bpe),
        checkpoint_path=ckpt,
        load_from_HF=args.load_from_hf,
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

    all_jsons = sorted(ann_dir.glob(args.ann_glob))
    if args.max_datasets > 0:
        all_jsons = all_jsons[: args.max_datasets]

    mode = "text-only" if args.text_only else "text+box"
    print(f"[{args.run_tag}] {len(all_jsons)} JSONs | {mode} | "
          f"{args.videos_per_dataset} videos/ds | {args.queries_per_video} queries/video")

    rows: list[dict] = []
    ds_summaries: list[dict] = []
    t0 = time.time()

    for j_idx, ann_json in enumerate(all_jsons):
        ds_name = ann_json.stem
        print(f"\n[{j_idx+1}/{len(all_jsons)}] {ds_name}")

        try:
            data = json.loads(ann_json.read_text())
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        videos = data.get("videos", [])
        if not videos:
            print("  no videos")
            continue

        vids = [int(v["id"]) for v in videos[: args.videos_per_dataset]]
        q_per_vid = args.queries_per_video if args.queries_per_video > 0 else 999

        ds_dices, ds_ious = [], []
        for vid in vids:
            queries = pick_queries_for_video(data, vid, q_per_vid)
            if not queries:
                fb = make_fallback_query_for_video(data, vid)
                if fb is not None:
                    queries = [fb]
            if not queries:
                print(f"  video {vid}: no queries")
                continue
            for q in queries:
                try:
                    r = eval_one_query(
                        video_model, data, vid, q, args.max_frames,
                        ann_json.parent, text_only=args.text_only,
                    )
                except Exception as e:
                    traceback.print_exc()
                    r = {"status": "error", "reason": str(e), "video_id": vid}
                if r is None:
                    continue
                r["dataset"] = ds_name
                rows.append(r)
                if r.get("status") == "ok":
                    ds_dices.append(r["mean_dice"])
                    ds_ious.append(r["mean_iou"])
                    print(
                        f"  vid={vid} '{r['query_text'][:40]}' "
                        f"dice={r['mean_dice']:.3f} iou={r['mean_iou']:.3f} "
                        f"conf={r.get('max_prob', 0):.3f} objs={r.get('n_objects', 0)} "
                        f"frames={r['n_frames']}"
                    )
                else:
                    print(f"  vid={vid} '{q.get('query_text','')[:40]}': {r.get('status')} {r.get('reason','')[:50]}")

        if ds_dices:
            ds_summaries.append({
                "dataset": ds_name,
                "n_evals": len(ds_dices),
                "mean_dice": float(np.mean(ds_dices)),
                "mean_iou": float(np.mean(ds_ious)),
            })
            print(f"  → dataset mean dice={np.mean(ds_dices):.4f} n={len(ds_dices)}")

    elapsed = time.time() - t0
    ok = [r for r in rows if r.get("status") == "ok"]
    md = float(np.mean([r["mean_dice"] for r in ok])) if ok else float("nan")
    mi = float(np.mean([r["mean_iou"] for r in ok])) if ok else float("nan")

    print(f"\n{'='*70}")
    print(f"[{args.run_tag}] ok={len(ok)}/{len(rows)} mean_dice={md:.4f} mean_iou={mi:.4f} ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

    if ds_summaries:
        print(f"\n{'Dataset':<55} {'Dice':>7} {'IoU':>7} {'N':>5}")
        print("-" * 78)
        for s in sorted(ds_summaries, key=lambda x: -x["mean_dice"]):
            print(f"{s['dataset'][:54]:<55} {s['mean_dice']:7.4f} {s['mean_iou']:7.4f} {s['n_evals']:5d}")

    tag = args.run_tag
    csv_path = out_dir / f"per_query_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        fields = ["dataset", "video_id", "query_text", "status", "reason", "no_scored_detail",
                  "mean_dice", "mean_iou", "dice_init", "n_frames", "n_objects", "max_prob", "init_frame",
                  "gt_annotated_frames", "gt_positive_frames", "pred_positive_frames", "window_frames"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "run_tag": tag,
        "mode": mode,
        "ann_dir": str(ann_dir),
        "ann_glob": args.ann_glob,
        "n_ok": len(ok),
        "n_total": len(rows),
        "mean_dice": md,
        "mean_iou": mi,
        "elapsed_s": elapsed,
        "per_dataset": ds_summaries,
        "rows": rows,
    }
    json_path = out_dir / f"summary_{tag}.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {csv_path}\n       {json_path}")


if __name__ == "__main__":
    main()
