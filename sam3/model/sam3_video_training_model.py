import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.logger import get_logger
from sam3.model.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from sam3.model.data_misc import BatchedDatapoint, BatchedFindTarget
from sam3.model.geometry_encoders import Prompt
from sam3.model.model_misc import SAM3Output
from sam3.model.sam3_tracker_utils import mask_to_box

logger = get_logger(__name__)


class Sam3VideoTrainingModel(nn.Module):
    """
    Minimal training wrapper for video batches.

    This uses the tracker on multi-frame inputs and converts its per-frame outputs
    into the same loss-facing format expected by the trainer and existing image
    losses. The detector is kept around for checkpoint compatibility and target
    conversion, but the training path here is tracker-centric.
    """

    def __init__(self, detector: nn.Module, tracker: nn.Module, **kwargs):
        super().__init__()
        self.detector = detector
        self.tracker = tracker
        logger.info("Initialized Sam3VideoTrainingModel.")

    @property
    def device(self):
        return self.tracker.device

    def back_convert(self, targets: BatchedFindTarget):
        return self.detector.back_convert(self._merge_targets(targets))

    def _get_query_batch_size(self, input: BatchedDatapoint) -> int:
        batch_sizes = [int(t.num_boxes.shape[0]) for t in input.find_targets]
        assert len(set(batch_sizes)) == 1, (
            f"Expected the same number of queries per frame, got {batch_sizes}"
        )
        return batch_sizes[0]

    def _get_query_slice(self, targets: BatchedFindTarget, query_idx: int):
        num_boxes = int(targets.num_boxes[query_idx].item())
        offset = int(targets.num_boxes[:query_idx].sum().item())
        return offset, num_boxes

    def _iter_query_masks(self, targets: BatchedFindTarget, query_idx: int):
        offset, num_boxes = self._get_query_slice(targets, query_idx)
        if num_boxes == 0:
            return None, False

        boxes = targets.boxes_padded[query_idx, :num_boxes]
        visible = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        is_visible = bool(visible.any().item())
        if targets.segments is None or targets.is_valid_segment is None:
            return None, is_visible

        merged_mask = None
        for inner_idx in range(num_boxes):
            if not bool(targets.is_valid_segment[offset + inner_idx].item()):
                continue
            seg = targets.segments[offset + inner_idx].bool()
            merged_mask = seg if merged_mask is None else (merged_mask | seg)
        return merged_mask, is_visible

    def _merge_targets(self, targets: BatchedFindTarget) -> BatchedFindTarget:
        batch_size = int(targets.num_boxes.shape[0])
        boxes = []
        repeated_boxes = []
        segments = [] if targets.segments is not None else None
        is_valid_segment = [] if targets.is_valid_segment is not None else None
        object_ids = []
        num_boxes = []
        is_exhaustive = targets.is_exhaustive.clone()
        semantic_segments = targets.semantic_segments
        zero_box = targets.boxes_padded.new_zeros(4)

        for query_idx in range(batch_size):
            offset, orig_num_boxes = self._get_query_slice(targets, query_idx)
            if orig_num_boxes == 0:
                num_boxes.append(0)
                continue

            query_boxes = targets.boxes_padded[query_idx, :orig_num_boxes]
            visible = (query_boxes[:, 2] > 0) & (query_boxes[:, 3] > 0)
            if not bool(visible.any().item()):
                num_boxes.append(0)
                continue

            visible_xyxy = box_cxcywh_to_xyxy(query_boxes[visible])
            merged_xyxy = torch.stack(
                [
                    visible_xyxy[:, 0].min(),
                    visible_xyxy[:, 1].min(),
                    visible_xyxy[:, 2].max(),
                    visible_xyxy[:, 3].max(),
                ]
            )
            merged_box = box_xyxy_to_cxcywh(merged_xyxy)
            boxes.append(merged_box)
            repeated_boxes.append(merged_box)
            object_ids.append(targets.object_ids_padded[query_idx, 0].clone())
            num_boxes.append(1)

            if segments is not None and is_valid_segment is not None:
                merged_mask = None
                for inner_idx in range(orig_num_boxes):
                    if not bool(targets.is_valid_segment[offset + inner_idx].item()):
                        continue
                    seg = targets.segments[offset + inner_idx].bool()
                    merged_mask = seg if merged_mask is None else (merged_mask | seg)
                if merged_mask is None:
                    dummy = torch.zeros_like(targets.segments[0], dtype=torch.bool)
                    segments.append(dummy)
                    is_valid_segment.append(False)
                else:
                    segments.append(merged_mask)
                    is_valid_segment.append(True)

        num_boxes = torch.as_tensor(num_boxes, dtype=torch.long, device=targets.num_boxes.device)
        if boxes:
            boxes = torch.stack(boxes, dim=0).to(targets.boxes.device)
            repeated_boxes = torch.stack(repeated_boxes, dim=0).to(targets.boxes.device)
            object_ids = torch.stack(object_ids, dim=0).to(targets.object_ids.device)
        else:
            boxes = targets.boxes.new_zeros((0, 4))
            repeated_boxes = targets.repeated_boxes.new_zeros((0, 4))
            object_ids = targets.object_ids.new_zeros((0,), dtype=torch.long)

        boxes_padded = zero_box.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
        object_ids_padded = (
            targets.object_ids_padded.new_full((batch_size, 1), -1)
        )
        packed_idx = 0
        for query_idx, n in enumerate(num_boxes.tolist()):
            if n == 0:
                continue
            boxes_padded[query_idx, 0] = boxes[packed_idx]
            object_ids_padded[query_idx, 0] = object_ids[packed_idx]
            packed_idx += 1

        if segments is not None and is_valid_segment is not None:
            if segments:
                segments = torch.stack(segments, dim=0).to(targets.boxes.device)
                is_valid_segment = torch.as_tensor(
                    is_valid_segment, dtype=torch.bool, device=targets.boxes.device
                )
            else:
                if targets.segments.ndim >= 2:
                    h, w = targets.segments.shape[-2:]
                    segments = torch.zeros(
                        (0, h, w), dtype=torch.bool, device=targets.boxes.device
                    )
                else:
                    segments = torch.zeros(
                        (0,), dtype=torch.bool, device=targets.boxes.device
                    )
                is_valid_segment = torch.zeros((0,), dtype=torch.bool, device=targets.boxes.device)

        return BatchedFindTarget(
            num_boxes=num_boxes,
            boxes=boxes,
            boxes_padded=boxes_padded,
            repeated_boxes=repeated_boxes,
            segments=segments,
            semantic_segments=semantic_segments,
            is_valid_segment=is_valid_segment,
            is_exhaustive=is_exhaustive,
            object_ids=object_ids,
            object_ids_padded=object_ids_padded,
        )

    def _det_pred_mask_for_init(self, det_out: dict, query_bs: int, image_h: int, image_w: int, device) -> torch.Tensor:
        """Extract top-scoring detection mask from detector output and upsample to image resolution."""
        pred_logits = det_out["pred_logits"].squeeze(-1)  # (query_bs, N_queries)
        pred_masks_low = det_out["pred_masks"]  # (query_bs, N_queries, H_low, W_low)
        top_idx = pred_logits.argmax(dim=1)  # (query_bs,)
        top_masks = pred_masks_low[
            torch.arange(query_bs, device=device), top_idx
        ].unsqueeze(1)  # (query_bs, 1, H_low, W_low)
        upsampled = F.interpolate(
            top_masks.float(), size=(image_h, image_w), mode="bilinear", align_corners=False
        )
        return (upsampled > 0).float()

    def _build_identity_indices(self, targets: BatchedFindTarget, device):
        batch_idx = []
        src_idx = []
        tgt_idx = []
        offset = 0
        for query_idx in range(int(targets.num_boxes.shape[0])):
            num_boxes = int(targets.num_boxes[query_idx].item())
            if num_boxes == 1:
                batch_idx.append(query_idx)
                src_idx.append(0)
                tgt_idx.append(offset)
            offset += num_boxes
        return (
            torch.as_tensor(batch_idx, dtype=torch.long, device=device),
            torch.as_tensor(src_idx, dtype=torch.long, device=device),
            torch.as_tensor(tgt_idx, dtype=torch.long, device=device),
        )

    def _tracker_out_to_loss_dict(self, out_dict, targets: BatchedFindTarget):
        pred_masks = out_dict["pred_masks"]
        pred_masks_high_res = out_dict.get("pred_masks_high_res", pred_masks)
        object_score_logits = out_dict.get("object_score_logits")
        if object_score_logits is None:
            object_score_logits = pred_masks.new_zeros(pred_masks.size(0), 1)
        object_score_logits = object_score_logits.view(pred_masks.size(0), 1, 1)

        _, _, height, width = pred_masks_high_res.shape
        pred_masks_binary = pred_masks_high_res > 0
        pred_boxes_xyxy = mask_to_box(pred_masks_binary)
        norm = pred_boxes_xyxy.new_tensor(
            [max(width - 1, 1), max(height - 1, 1), max(width - 1, 1), max(height - 1, 1)]
        )
        pred_boxes_xyxy = pred_boxes_xyxy / norm
        pred_boxes = box_xyxy_to_cxcywh(pred_boxes_xyxy)

        indices = self._build_identity_indices(targets, pred_masks.device)
        return {
            "pred_masks": pred_masks,
            "pred_logits": object_score_logits,
            "pred_boxes": pred_boxes,
            "pred_boxes_xyxy": pred_boxes_xyxy,
            "indices": indices,
            "semantic_seg": pred_masks,
            # Reuse the tracker object score as the decoder-level presence logit
            # so the existing IABCEMdetr presence supervision is active.
            "presence_logit_dec": object_score_logits.squeeze(-1),
            "presence_logit": object_score_logits.squeeze(-1),
        }

    def _ensure_presence_logits(self, sam_output: SAM3Output) -> SAM3Output:
        for stage_outputs in sam_output.output:
            for out in stage_outputs:
                if out.get("presence_logit", None) is not None:
                    continue
                if out.get("presence_logit_dec", None) is not None:
                    out["presence_logit"] = out["presence_logit_dec"].amax(
                        dim=1, keepdim=True
                    )
                elif out.get("pred_logits", None) is not None:
                    out["presence_logit"] = out["pred_logits"].amax(
                        dim=1, keepdim=True
                    ).squeeze(-1)
        return sam_output

    def forward(self, input: BatchedDatapoint, **kwargs):
        merged_targets = [self._merge_targets(t) for t in input.find_targets]
        if len(input.find_targets) <= 1:
            detector_input = BatchedDatapoint(
                img_batch=input.img_batch,
                find_text_batch=input.find_text_batch,
                find_inputs=input.find_inputs,
                find_targets=merged_targets,
                find_metadatas=input.find_metadatas,
                raw_images=input.raw_images,
            )
            return self._ensure_presence_logits(self.detector(detector_input))

        num_frames = len(input.find_targets)
        device = input.img_batch.device
        image_h, image_w = input.img_batch.shape[-2:]
        query_bs = self._get_query_batch_size(input)

        # Step 1: Run shared backbone (vision + text) on all frames at once.
        # This matches SAM3's inference where the detector backbone processes every frame.
        backbone_out = {"img_batch_all_stages": input.img_batch}
        backbone_out.update(self.detector.backbone.forward_image(input.img_batch))
        backbone_out.update(
            self.detector.backbone.forward_text(input.find_text_batch, device=device)
        )

        sam2_bb = backbone_out.get("sam2_backbone_out")
        assert sam2_bb is not None, (
            "detector backbone did not return sam2_backbone_out; "
            "make sure the vision backbone is SAM3-compatible"
        )

        # Step 2: Run detector per-frame with GT targets for detection loss.
        # Frame 0's predicted mask is also extracted (detached) as the tracker init mask.
        det_outputs = []
        init_mask: torch.Tensor | None = None

        for stage_idx in range(num_frames):
            find_input = input.find_inputs[stage_idx]
            geo_prompt = Prompt(
                box_embeddings=find_input.input_boxes,
                box_mask=find_input.input_boxes_mask,
                box_labels=find_input.input_boxes_label,
            )
            det_out = self.detector.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=merged_targets[stage_idx],
                geometric_prompt=geo_prompt,
            )
            det_outputs.append(det_out)
            if stage_idx == 0:
                with torch.no_grad():
                    init_mask = self._det_pred_mask_for_init(
                        det_out, query_bs, image_h, image_w, device
                    )

        assert init_mask is not None

        # Step 3: Build tracker backbone from the SAM2 features produced by the shared backbone.
        # Apply conv_s0/conv_s1 as done in SAM3VideoBase.run_backbone_and_detection.
        tracker_backbone_fpn = [
            self.tracker.sam_mask_decoder.conv_s0(sam2_bb["backbone_fpn"][0]),
            self.tracker.sam_mask_decoder.conv_s1(sam2_bb["backbone_fpn"][1]),
            sam2_bb["backbone_fpn"][2],
        ]
        tracker_backbone = {
            "vision_features": tracker_backbone_fpn[-1],
            "vision_pos_enc": sam2_bb["vision_pos_enc"],
            "backbone_fpn": tracker_backbone_fpn,
            # Tracker control: frame 0 is init cond, remaining frames are propagation targets.
            # ALL frames (including frame 0) contribute to the training loss.
            "num_frames": num_frames,
            "init_cond_frames": [0],
            "frames_not_in_init_cond": list(range(1, num_frames)),
            "frames_to_add_correction_pt": [],
            "point_inputs_per_frame": {},
            "mask_inputs_per_frame": {0: init_mask},
        }

        tracker_input = BatchedDatapoint(
            img_batch=input.img_batch,
            find_text_batch=input.find_text_batch,
            find_inputs=input.find_inputs,
            find_targets=merged_targets,
            find_metadatas=input.find_metadatas,
            raw_images=input.raw_images,
        )

        tracker_outputs = self.tracker.forward_tracking(
            backbone_out=tracker_backbone,
            input=tracker_input,
            return_dict=False,
        )

        # Step 4: Build stage outputs combining detector and tracker for each frame.
        # Each frame contributes both detection loss (det_out) and tracking loss (tracker_out).
        stage_outputs = []
        for stage_idx in range(num_frames):
            tracker_out_dict = self._tracker_out_to_loss_dict(
                tracker_outputs[stage_idx], merged_targets[stage_idx]
            )
            stage_outputs.append([det_outputs[stage_idx], tracker_out_dict])

        return self._ensure_presence_logits(SAM3Output(
            output=stage_outputs,
            iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
            loss_stages=list(range(num_frames)),
        ))
