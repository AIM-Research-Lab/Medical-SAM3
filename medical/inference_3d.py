#!/usr/bin/env python3
"""Text-prompted 3D inference for Medical-SAM3.

The volume is treated as an ordered video. Every text query is propagated over
the full depth without using a ground-truth slice, box, or point prompt. By
default, independent four-slice clips are used; this is the inference policy
used by the current Medical-SAM3 3D checkpoint.

Input NPZ files must contain ``imgs`` with shape ``(D,H,W)``, ``(D,H,W,C)``, or
``(D,C,H,W)``. Prompts can be embedded in ``text_prompts`` or supplied with
``--prompt`` / ``--prompts-json``. Output NPZ files contain ``segs`` in the
original ``(D,H,W)`` geometry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BPE = REPO_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
FUSION_POLICIES = ("pixel_score", "tracker_score", "detection_score", "small_first")


@dataclass(frozen=True)
class InferenceConfig:
    clip_length: int = 4
    fusion: str = "pixel_score"


class PILFrameList(list):
    """PIL frames with the original output geometry attached."""

    def __init__(
        self, frames: Iterable[Image.Image], original_size_hw: tuple[int, int]
    ):
        super().__init__(frames)
        self.original_size_hw = original_size_hw


def load_npz(path: Path) -> dict[str, Any]:
    # CVPR/MedSegDB prompt dictionaries are stored as NumPy object scalars.
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _unwrap_object(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def normalize_prompts(value: Any) -> dict[str, Any]:
    value = _unwrap_object(value)
    if not isinstance(value, dict):
        raise ValueError("prompts must be a JSON object or an embedded dictionary")
    result: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key == "instance_label":
            result[key] = int(raw_value)
            continue
        try:
            label = int(key)
        except ValueError as exc:
            raise ValueError(f"prompt label must be an integer, got {key!r}") from exc
        text = str(raw_value).strip()
        if label <= 0 or not text:
            raise ValueError(f"invalid prompt {key!r}: {raw_value!r}")
        result[str(label)] = text
    result.setdefault("instance_label", 0)
    if not any(key != "instance_label" for key in result):
        raise ValueError("at least one LABEL=TEXT prompt is required")
    if result["instance_label"] and len(result) != 2:
        raise ValueError("instance inference accepts exactly one text prompt")
    return result


def parse_inline_prompts(items: list[str], instance: bool) -> dict[str, Any] | None:
    if not items:
        return None
    prompts: dict[str, Any] = {"instance_label": int(instance)}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--prompt expects LABEL=TEXT, got {item!r}")
        label, text = item.split("=", 1)
        prompts[label.strip()] = text.strip()
    return normalize_prompts(prompts)


def prompts_for_volume(
    volume_path: Path,
    payload: dict[str, Any],
    inline_prompts: dict[str, Any] | None,
    prompt_document: dict[str, Any] | None,
) -> dict[str, Any]:
    if inline_prompts is not None:
        return inline_prompts
    if prompt_document is not None:
        # A direct prompt map applies to every volume. Otherwise use filename or stem.
        if any(str(key).isdigit() for key in prompt_document):
            return normalize_prompts(prompt_document)
        selected = prompt_document.get(
            volume_path.name, prompt_document.get(volume_path.stem)
        )
        if selected is None:
            raise ValueError(f"no prompts found for {volume_path.name}")
        return normalize_prompts(selected)
    if "text_prompts" not in payload:
        raise ValueError("missing text_prompts; pass --prompt or --prompts-json")
    return normalize_prompts(payload["text_prompts"])


def _slice_to_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"each slice must be grayscale or RGB, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.nan_to_num(array.astype(np.float32, copy=False))
        minimum, maximum = float(array.min()), float(array.max())
        if maximum > minimum:
            array = (array - minimum) / (maximum - minimum)
        else:
            array = np.zeros_like(array)
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def volume_to_frames(imgs: np.ndarray) -> tuple[PILFrameList, tuple[int, int]]:
    if not isinstance(imgs, np.ndarray) or imgs.ndim not in (3, 4):
        raise ValueError(f"imgs must be a 3D or 4D NumPy array, got {type(imgs)!r}")
    if imgs.shape[0] < 1:
        raise ValueError("imgs contains no slices")
    converted = [_slice_to_rgb(imgs[index]) for index in range(int(imgs.shape[0]))]
    spatial_shape = tuple(int(value) for value in converted[0].shape[:2])
    if any(tuple(frame.shape[:2]) != spatial_shape for frame in converted):
        raise ValueError("all volume slices must have identical spatial dimensions")
    frames = [Image.fromarray(frame) for frame in converted]
    return PILFrameList(frames, spatial_shape), spatial_shape


def temporal_windows(depth: int, clip_length: int) -> list[tuple[int, int]]:
    if depth < 1 or clip_length < 1:
        raise ValueError("depth and clip length must be positive")
    return [
        (start, min(depth, start + clip_length))
        for start in range(0, depth, clip_length)
    ]


def _score_max(value: Any, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    return float(array.max()) if array.size else fallback


def _mask_union(masks: Any) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim == 2:
        return masks
    if masks.ndim == 3:
        return np.any(masks, axis=0)
    if masks.ndim == 4 and masks.shape[1] == 1:
        return np.any(masks[:, 0], axis=0)
    raise ValueError(f"unexpected mask shape: {masks.shape}")


def install_probability_outputs() -> None:
    """Expose tracker scores and continuous mask probabilities during inference."""
    from sam3.model.sam3_video_inference import Sam3VideoInference
    from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores

    if getattr(
        Sam3VideoInference._postprocess_output, "_medical_3d_probabilities", False
    ):
        return
    original_build_outputs = Sam3VideoInference.build_outputs
    original_postprocess = Sam3VideoInference._postprocess_output

    def patched_build_outputs(
        self,
        frame_idx,
        num_frames,
        reverse,
        det_out,
        tracker_low_res_masks_global,
        tracker_obj_scores_global,
        tracker_metadata_prev,
        tracker_update_plan,
        orig_vid_height,
        orig_vid_width,
        reconditioned_obj_ids=None,
        det_to_matched_trk_obj_ids=None,
    ):
        masks = original_build_outputs(
            self,
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_obj_scores_global=tracker_obj_scores_global,
            tracker_metadata_prev=tracker_metadata_prev,
            tracker_update_plan=tracker_update_plan,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
            reconditioned_obj_ids=reconditioned_obj_ids,
            det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
        )
        logits: dict[int, torch.Tensor] = {}
        existing_ids = tracker_metadata_prev["obj_ids_all_gpu"]
        existing = F.interpolate(
            tracker_low_res_masks_global.unsqueeze(1),
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )
        for object_id, logit in zip(existing_ids, existing):
            logits[int(object_id)] = logit

        detection_indices = tracker_update_plan["new_det_fa_inds"]
        detection_ids = tracker_update_plan["new_det_obj_ids"]
        index_tensor = torch.as_tensor(detection_indices, dtype=torch.long)
        detected = det_out["mask"][index_tensor].unsqueeze(1)
        detected = fill_holes_in_mask_scores(
            detected,
            max_area=self.fill_hole_area,
            fill_holes=True,
            remove_sprinkles=True,
        )
        detected = F.interpolate(
            detected,
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )
        for object_id, logit in zip(detection_ids, detected):
            logits[int(object_id)] = logit

        if reconditioned_obj_ids:
            replacements = tracker_update_plan.get(
                "trk_id_to_max_iou_high_conf_det", {}
            )
            for object_id in reconditioned_obj_ids:
                detection_index = replacements.get(object_id)
                if detection_index is None:
                    continue
                logit = det_out["mask"][detection_index].unsqueeze(0).unsqueeze(0)
                logits[int(object_id)] = F.interpolate(
                    logit.float(),
                    size=(orig_vid_height, orig_vid_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        cache = getattr(self, "_medical_3d_logits", None)
        if cache is None:
            cache = {}
            self._medical_3d_logits = cache
        cache[id(masks)] = logits
        return masks

    def patched_postprocess(
        self,
        inference_state,
        out,
        removed_obj_ids=None,
        suppressed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ):
        result = original_postprocess(
            self,
            inference_state,
            out,
            removed_obj_ids=removed_obj_ids,
            suppressed_obj_ids=suppressed_obj_ids,
            unconfirmed_obj_ids=unconfirmed_obj_ids,
        )
        object_ids = [int(value) for value in result["out_obj_ids"]]
        tracker_scores = out.get("obj_id_to_tracker_score") or {}
        result["out_tracker_probs"] = np.asarray(
            [_score_max(tracker_scores.get(object_id)) for object_id in object_ids],
            dtype=np.float32,
        )
        cache = getattr(self, "_medical_3d_logits", {})
        logits = cache.pop(id(out["obj_id_to_mask"]), None)
        if logits is not None and all(object_id in logits for object_id in object_ids):
            if object_ids:
                scores = torch.cat(
                    [logits[object_id] for object_id in object_ids], dim=0
                )
                result["out_mask_scores"] = scores.sigmoid().float().cpu().numpy()
            else:
                height, width = result["out_binary_masks"].shape[-2:]
                result["out_mask_scores"] = np.zeros(
                    (0, height, width), dtype=np.float32
                )
            result["out_mask_score_source"] = "mask_logits"
        else:
            result["out_mask_scores"] = result["out_binary_masks"].astype(np.float32)
            result["out_mask_score_source"] = "binary_fallback"
        return result

    Sam3VideoInference.build_outputs = patched_build_outputs
    patched_postprocess._medical_3d_probabilities = True
    Sam3VideoInference._postprocess_output = patched_postprocess


def predict_prompt(
    model: Any,
    state: dict[str, Any],
    text: str,
    depth: int,
    spatial_shape: tuple[int, int],
    config: InferenceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    prediction = np.zeros((depth, *spatial_shape), dtype=bool)
    detection_scores = np.zeros(depth, dtype=np.float32)
    tracker_scores = np.zeros(depth, dtype=np.float32)
    pixel_scores = (
        np.zeros((depth, *spatial_shape), dtype=np.float32)
        if config.fusion == "pixel_score"
        else None
    )
    seen = np.zeros(depth, dtype=bool)

    for start, end in temporal_windows(depth, config.clip_length):
        model.reset_state(state)
        model.add_prompt(inference_state=state, frame_idx=start, text_str=text)
        for frame_index, output in model.propagate_in_video(
            state,
            start_frame_idx=start,
            max_frame_num_to_track=end - start - 1,
            reverse=False,
        ):
            if output is None:
                continue
            frame_index = int(frame_index)
            if not start <= frame_index < end:
                raise RuntimeError(
                    f"propagation escaped [{start}, {end}): {frame_index}"
                )
            if seen[frame_index]:
                raise RuntimeError(f"frame {frame_index} was returned more than once")
            seen[frame_index] = True
            masks = output.get("out_binary_masks")
            detection_scores[frame_index] = _score_max(output.get("out_probs"))
            tracker_scores[frame_index] = _score_max(
                output.get("out_tracker_probs"), fallback=detection_scores[frame_index]
            )
            if masks is None or len(masks) == 0:
                continue
            merged = _mask_union(masks)
            if merged.shape != spatial_shape:
                merged = np.asarray(
                    Image.fromarray(merged.astype(np.uint8)).resize(
                        (spatial_shape[1], spatial_shape[0]),
                        resample=Image.Resampling.NEAREST,
                    ),
                    dtype=bool,
                )
            prediction[frame_index] |= merged
            if pixel_scores is None:
                continue
            if output.get("out_mask_score_source") != "mask_logits":
                raise RuntimeError(
                    "continuous mask logits were unavailable for pixel_score fusion"
                )
            object_scores = np.asarray(output["out_mask_scores"], dtype=np.float32)
            frame_scores = object_scores.max(axis=0)
            if frame_scores.shape != spatial_shape:
                frame_scores = np.asarray(
                    Image.fromarray(frame_scores).resize(
                        (spatial_shape[1], spatial_shape[0]),
                        resample=Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                )
            pixel_scores[frame_index] = frame_scores

    if not np.all(seen):
        raise RuntimeError(
            f"model did not return frames {np.flatnonzero(~seen).tolist()}"
        )
    return prediction, detection_scores, tracker_scores, pixel_scores


def merge_prediction(
    segmentation: np.ndarray,
    priority: np.ndarray,
    prediction: np.ndarray,
    label: int,
    detection_scores: np.ndarray,
    tracker_scores: np.ndarray,
    fusion: str,
    pixel_scores: np.ndarray | None,
) -> None:
    if fusion == "pixel_score" and pixel_scores is None:
        raise ValueError("pixel_score fusion requires mask probabilities")
    for z_index, mask in enumerate(prediction):
        area = int(mask.sum())
        if area == 0:
            continue
        if fusion == "pixel_score":
            candidate = pixel_scores[z_index]
            update = mask & (candidate > priority[z_index])
            priority[z_index][update] = candidate[update]
        elif fusion == "tracker_score":
            candidate = float(tracker_scores[z_index])
            update = mask & (candidate > priority[z_index])
            priority[z_index][update] = candidate
        elif fusion == "detection_score":
            candidate = float(detection_scores[z_index])
            update = mask & (candidate > priority[z_index])
            priority[z_index][update] = candidate
        elif fusion == "small_first":
            candidate = 1.0 / area
            update = mask & (candidate > priority[z_index])
            priority[z_index][update] = candidate
        else:
            raise ValueError(f"unknown fusion policy: {fusion}")
        segmentation[z_index][update] = label


def infer_volume(
    model: Any,
    imgs: np.ndarray,
    prompts: dict[str, Any],
    config: InferenceConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    frames, spatial_shape = volume_to_frames(imgs)
    depth = len(frames)
    instance_mode = bool(prompts["instance_label"])
    labels = sorted(int(key) for key in prompts if key != "instance_label")
    max_label = max(labels, default=1)
    output_dtype = np.uint8 if max_label <= np.iinfo(np.uint8).max else np.uint16
    segmentation = np.zeros((depth, *spatial_shape), dtype=output_dtype)
    initial_priority = 0.5 if config.fusion == "pixel_score" else -np.inf
    priority = np.full(segmentation.shape, initial_priority, dtype=np.float32)
    diagnostics: list[dict[str, Any]] = []

    with torch.inference_mode():
        state = model.init_state(resource_path=frames)
        for label in labels:
            started = time.time()
            prediction, detection, tracker, pixels = predict_prompt(
                model,
                state,
                prompts[str(label)],
                depth,
                spatial_shape,
                config,
            )
            if instance_mode:
                segmentation[prediction] = 1
            else:
                merge_prediction(
                    segmentation,
                    priority,
                    prediction,
                    label,
                    detection,
                    tracker,
                    config.fusion,
                    pixels,
                )
            diagnostics.append(
                {
                    "label": label,
                    "text": prompts[str(label)],
                    "positive_voxels": int(prediction.sum()),
                    "positive_slices": int(np.any(prediction, axis=(1, 2)).sum()),
                    "max_detection_score": float(detection.max()),
                    "max_tracker_score": float(tracker.max()),
                    "seconds": time.time() - started,
                }
            )
        del state
    return segmentation, {"prompts": diagnostics, "instance_mode": instance_mode}


def _input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.npz"))
        if files:
            return files
        raise ValueError(f"no NPZ files found in {path}")
    raise ValueError(f"input does not exist: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="NPZ file or directory"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bpe", type=Path, default=DEFAULT_BPE)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        metavar="LABEL=TEXT",
        help="repeat for multiclass inference; overrides embedded prompts",
    )
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument(
        "--instance", action="store_true", help="write a binary instance mask"
    )
    parser.add_argument("--clip-length", type=int, default=4)
    parser.add_argument("--fusion", choices=FUSION_POLICIES, default="pixel_score")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-det-threshold", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("Medical-SAM3 3D inference requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but is not available")
    if ":" in args.device:
        torch.cuda.set_device(args.device)
    if args.clip_length < 1:
        raise ValueError("--clip-length must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.bpe.is_file():
        raise FileNotFoundError(args.bpe)

    inline_prompts = parse_inline_prompts(args.prompt, args.instance)
    prompt_document = None
    if args.prompts_json:
        prompt_document = json.loads(args.prompts_json.read_text(encoding="utf-8"))
        if not isinstance(prompt_document, dict):
            raise ValueError("--prompts-json must contain a JSON object")

    install_probability_outputs()
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        bpe_path=str(args.bpe),
        strict_state_dict_loading=True,
        training_mode=False,
        device=args.device,
        compile=False,
        apply_temporal_disambiguation=True,
        image_size=504,
    )
    model.eval()
    model.hotstart_delay = 0
    model.hotstart_unmatch_thresh = 0
    model.hotstart_dup_thresh = 0
    if args.no_det_threshold:
        model.score_threshold_detection = 0.0
        model.new_det_thresh = 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = InferenceConfig(clip_length=args.clip_length, fusion=args.fusion)
    records: list[dict[str, Any]] = []
    failures = 0
    for index, input_path in enumerate(_input_files(args.input), start=1):
        started = time.time()
        try:
            payload = load_npz(input_path)
            if "imgs" not in payload:
                raise ValueError("NPZ is missing the imgs array")
            prompts = prompts_for_volume(
                input_path, payload, inline_prompts, prompt_document
            )
            segmentation, details = infer_volume(
                model, payload["imgs"], prompts, config
            )
            output_path = args.output_dir / input_path.name
            temporary = output_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, segs=segmentation)
            os.replace(temporary, output_path)
            record = {
                "file": input_path.name,
                "status": "ok",
                "output": str(output_path),
                "shape": list(segmentation.shape),
                "clip_length": config.clip_length,
                "fusion": config.fusion,
                "seconds": time.time() - started,
                **details,
            }
            print(f"[{index}] {input_path.name} -> {output_path}")
        except Exception as exc:
            failures += 1
            record = {
                "file": input_path.name,
                "status": "error",
                "error": str(exc),
                "seconds": time.time() - started,
            }
            print(f"[{index}] ERROR {input_path.name}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise
        records.append(record)

    summary_path = args.output_dir / "inference_summary.json"
    summary_path.write_text(
        json.dumps({"records": records, "failures": failures}, indent=2),
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
