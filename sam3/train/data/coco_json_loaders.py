# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# pyre-unsafe

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from pycocotools import mask as mask_util


def convert_boxlist_to_normalized_tensor(box_list, image_width, image_height):
    boxes = torch.tensor(box_list, dtype=torch.float32)
    boxes[:, [0, 2]] /= image_width
    boxes[:, [1, 3]] /= image_height
    boxes = boxes.clamp(0, 1)
    return boxes


def load_coco_and_group_by_image(json_path: str) -> Tuple[List[Dict], Dict[int, str]]:
    with open(json_path, "r") as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    sorted_image_ids = sorted(images.keys())
    grouped = []
    for image_id in sorted_image_ids:
        image_info = images[image_id]
        grouped.append({"image": image_info, "annotations": anns_by_image.get(image_id, [])})
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    return grouped, cat_id_to_name


def ann_to_rle(segm, im_info: Dict) -> Dict:
    h, w = im_info["height"], im_info["width"]
    if isinstance(segm, list):
        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif isinstance(segm["counts"], list):
        rle = mask_util.frPyObjects(segm, h, w)
    else:
        rle = segm
    return rle


class COCO_FROM_JSON:
    def __init__(self, annotation_file, prompts=None, include_negatives=True, category_chunk_size=None):
        self._raw_data, self._cat_idx_to_text = load_coco_and_group_by_image(annotation_file)
        self._sorted_cat_ids = sorted(list(self._cat_idx_to_text.keys()))
        self.prompts = None
        self.include_negatives = include_negatives
        self.category_chunk_size = (
            category_chunk_size if category_chunk_size is not None else len(self._sorted_cat_ids)
        )
        self.category_chunks = [
            self._sorted_cat_ids[i : i + self.category_chunk_size]
            for i in range(0, len(self._sorted_cat_ids), self.category_chunk_size)
        ]
        if prompts is not None:
            prompts = eval(prompts)
            self.prompts = {}
            for loc_dict in prompts:
                self.prompts[int(loc_dict["id"])] = loc_dict["name"]
            assert len(self.prompts) == len(self._sorted_cat_ids)

    def getDatapointIds(self):
        return list(range(len(self._raw_data) * len(self.category_chunks)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        img_idx = idx // len(self.category_chunks)
        chunk_idx = idx % len(self.category_chunks)
        cat_chunk = self.category_chunks[chunk_idx]
        queries = []
        annotations = []
        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }
        annot_template = {
            "image_id": 0,
            "bbox": None,
            "area": None,
            "segmentation": None,
            "object_id": None,
            "is_crowd": None,
            "id": None,
        }
        raw_annotations = self._raw_data[img_idx]["annotations"]
        image_info = self._raw_data[img_idx]["image"]
        width, height = image_info["width"], image_info["height"]
        cat_id_to_anns = defaultdict(list)
        for ann in raw_annotations:
            cat_id_to_anns[ann["category_id"]].append(ann)
        annotations_by_cat_sorted = [(cat_id, cat_id_to_anns[cat_id]) for cat_id in cat_chunk]
        for cat_id, anns in annotations_by_cat_sorted:
            if len(anns) == 0 and not self.include_negatives:
                continue
            cur_ann_ids = []
            for ann in anns:
                annotation = annot_template.copy()
                annotation["id"] = len(annotations)
                annotation["object_id"] = annotation["id"]
                annotation["is_crowd"] = ann["iscrowd"]
                normalized_boxes = convert_boxlist_to_normalized_tensor([ann["bbox"]], width, height)
                bbox = normalized_boxes[0]
                annotation["area"] = (bbox[2] * bbox[3]).item()
                annotation["bbox"] = bbox
                if "segmentation" in ann and ann["segmentation"] not in (None, []):
                    annotation["segmentation"] = ann_to_rle(ann["segmentation"], im_info=image_info)
                annotations.append(annotation)
                cur_ann_ids.append(annotation["id"])
            query = query_template.copy()
            query["id"] = len(queries)
            query["original_cat_id"] = cat_id
            query["query_text"] = self._cat_idx_to_text[cat_id] if self.prompts is None else self.prompts[cat_id]
            query["object_ids_output"] = cur_ann_ids
            queries.append(query)
        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        img_idx = idx // len(self.category_chunks)
        img_data = self._raw_data[img_idx]["image"]
        return [{"id": 0, "file_name": img_data["file_name"], "original_img_id": img_data["id"], "coco_img_id": img_data["id"]}]


class SAM3_VIDEO_API_FROM_JSON:
    def __init__(self, annotation_file):
        with open(annotation_file, "r") as f:
            self._raw_data = json.load(f)
        self._videos = {v["id"]: v for v in self._raw_data.get("videos", [])}
        self._annotations = self._raw_data.get("annotations", [])
        self._queries = self._raw_data.get("queries", [])

        for ann in self._annotations:
            seg = ann.get("segmentation")
            if isinstance(seg, dict) and isinstance(seg.get("counts"), list):
                size = seg.get("size") or [0, 0]
                h, w = int(size[0]), int(size[1])
                rle = mask_util.frPyObjects(seg, h, w)
                # frPyObjects can return a list for some inputs
                if isinstance(rle, list):
                    rle = rle[0]
                ann["segmentation"] = rle

        self._anns_by_video = defaultdict(list)
        self._ann_id_to_video = {}
        for ann in self._annotations:
            video_id = ann["video_id"]
            self._anns_by_video[video_id].append(ann)
            self._ann_id_to_video[ann["id"]] = video_id

        self._queries_by_video = defaultdict(list)
        for query in self._queries:
            obj_ids = query.get("object_ids_output") or []
            if not obj_ids:
                continue
            video_id = self._ann_id_to_video.get(obj_ids[0])
            if video_id is None:
                continue
            if "ptr_x_query_id" not in query:
                query["ptr_x_query_id"] = None
            if "ptr_y_query_id" not in query:
                query["ptr_y_query_id"] = None
            self._queries_by_video[video_id].append(query)

        ann_info = {}
        for ann in self._annotations:
            ann_info[ann["id"]] = (
                ann["video_id"],
                ann.get("object_id", -1),
                ann.get("frame_index", ann.get("image_id", 0)),
            )

        for video_id, queries in self._queries_by_video.items():
            obj_to_queries = defaultdict(list)
            for query in queries:
                obj_ids = query.get("object_ids_output") or []
                if not obj_ids:
                    continue
                info = ann_info.get(obj_ids[0])
                if info is None:
                    continue
                ann_video_id, object_id, frame_index = info
                if ann_video_id != video_id or object_id < 0:
                    continue
                obj_to_queries[object_id].append((frame_index, query))

            for _, qlist in obj_to_queries.items():
                qlist.sort(key=lambda x: (x[0], x[1].get("id", 0)))
                for i, (_, query) in enumerate(qlist):
                    prev_q = qlist[i - 1][1] if i > 0 else None
                    next_q = qlist[i + 1][1] if i + 1 < len(qlist) else None
                    if query.get("ptr_x_query_id") in (None, -1):
                        query["ptr_x_query_id"] = prev_q.get("id") if prev_q else None
                    if query.get("ptr_y_query_id") in (None, -1):
                        query["ptr_y_query_id"] = next_q.get("id") if next_q else None

        self._datapoint_ids = []
        self._skipped_single_frame_videos = 0
        for video_id, video in sorted(self._videos.items()):
            num_frames = len(video.get("file_names", []))
            if num_frames <= 1:
                self._skipped_single_frame_videos += 1
                continue
            self._datapoint_ids.append(video_id)

    def getDatapointIds(self):
        return self._datapoint_ids

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        queries = self._queries_by_video.get(idx, [])
        annotations = self._anns_by_video.get(idx, [])
        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        video = self._videos[idx]
        frames = []
        for frame_idx, fname in enumerate(video.get("file_names", [])):
            frames.append(
                {
                    "id": frame_idx,
                    "file_name": fname,
                    "original_img_id": frame_idx,
                    "coco_img_id": frame_idx,
                }
            )
        return frames


class SAM3_SINGLE_FRAME_API_FROM_JSON(SAM3_VIDEO_API_FROM_JSON):
    """Like SAM3_VIDEO_API_FROM_JSON but includes single-frame videos.

    SAM3_VIDEO_API_FROM_JSON skips videos with <= 1 frame (designed for real
    video datasets). This subclass re-includes those single-frame entries so
    that 2D/1-frame datasets stored in the unified SAM3 video API JSON format
    can be loaded by VideoGroundingDataset with num_stages_sample=1.
    """

    def __init__(self, annotation_file):
        super().__init__(annotation_file)
        existing = set(self._datapoint_ids)
        for video_id, video in sorted(self._videos.items()):
            if video_id not in existing:
                self._datapoint_ids.append(video_id)
        self._datapoint_ids.sort()
