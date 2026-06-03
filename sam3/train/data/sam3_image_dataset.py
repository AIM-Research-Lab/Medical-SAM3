# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# pyre-unsafe

"""Dataset class for modulated detection"""

import json
import os
import random
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.data
import torchvision
import numpy as np
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from PIL.Image import DecompressionBombError
from sam3.model.box_ops import box_xywh_to_xyxy
from torchvision.datasets.vision import VisionDataset

from .coco_json_loaders import COCO_FROM_JSON

try:
    from decord import cpu, VideoReader
except ImportError:
    cpu = None
    VideoReader = None


@dataclass
class InferenceMetadata:
    coco_image_id: int
    original_image_id: int
    original_category_id: int
    original_size: Tuple[int, int]
    object_id: int
    frame_index: int
    is_conditioning_only: Optional[bool] = False


@dataclass
class FindQuery:
    query_text: str
    image_id: int
    object_ids_output: List[int]
    is_exhaustive: bool
    query_processing_order: int = 0
    input_bbox: Optional[torch.Tensor] = None
    input_bbox_label: Optional[torch.Tensor] = None
    input_points: Optional[torch.Tensor] = None
    semantic_target: Optional[torch.Tensor] = None
    is_pixel_exhaustive: Optional[bool] = None


@dataclass
class FindQueryLoaded(FindQuery):
    inference_metadata: Optional[InferenceMetadata] = None


@dataclass
class Object:
    bbox: torch.Tensor
    area: float
    object_id: Optional[int] = -1
    frame_index: Optional[int] = -1
    segment: Optional[Union[torch.Tensor, dict]] = None
    is_crowd: bool = False
    source: Optional[str] = None


@dataclass
class Image:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]
    size: Tuple[int, int]
    blurring_mask: Optional[Dict[str, Any]] = None


@dataclass
class Datapoint:
    """Refers to an image/video and all its annotations"""

    find_queries: List[FindQueryLoaded]
    images: List[Image]
    raw_images: Optional[List[PILImage.Image]] = None
    get_queries: Optional[List[Any]] = None


class CustomCocoDetectionAPI(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        load_segmentation: bool,
        fix_fname: bool = False,
        training: bool = True,
        blurring_masks_path: Optional[str] = None,
        use_caching: bool = True,
        zstd_dict_path=None,
        filter_query=None,
        coco_json_loader: Callable = COCO_FROM_JSON,
        limit_ids: int = None,
    ) -> None:
        super().__init__(root)
        self.annFile = annFile
        self.use_caching = use_caching
        self.zstd_dict_path = zstd_dict_path
        self.curr_epoch = 0
        self.load_segmentation = load_segmentation
        self.fix_fname = fix_fname
        self.filter_query = filter_query
        self.coco = None
        self.coco_json_loader = coco_json_loader
        self.limit_ids = limit_ids
        self.training = training
        self.blurring_masks_path = blurring_masks_path
        self.set_sharded_annotation_file(0)

    def _load_images(
        self, datapoint_id: int, img_ids_to_load: Optional[Set[int]] = None
    ) -> Tuple[List[Tuple[int, PILImage.Image]], List[Dict[str, Any]]]:
        all_images = []
        all_img_metadata = []
        for current_meta in self.coco.loadImagesFromDatapoint(datapoint_id):
            img_id = current_meta["id"]
            if img_ids_to_load is not None and img_id not in img_ids_to_load:
                continue
            if self.fix_fname:
                current_meta["file_name"] = current_meta["file_name"].split("/")[-1]
            path = current_meta["file_name"]
            if self.blurring_masks_path is not None:
                mask_fname = os.path.basename(path).replace(".jpg", "-mask.json")
                mask_path = os.path.join(self.blurring_masks_path, mask_fname)
                if os.path.exists(mask_path):
                    with open(mask_path, "r") as fopen:
                        current_meta["blurring_mask"] = json.load(fopen)
            all_img_metadata.append(current_meta)
            path = os.path.join(self.root, path)
            try:
                if ".mp4" in path and path[-4:] == ".mp4":
                    if VideoReader is None:
                        raise FileNotFoundError("decord not installed; cannot load video frames")
                    video_path, frame = path.split("@")
                    video = VideoReader(video_path, ctx=cpu(0))
                    all_images.append(
                        (img_id, torchvision.transforms.ToPILImage()(video[int(frame)].asnumpy()))
                    )
                elif ".npz" in path and "@" in path:
                    npz_path, frame = path.split("@")
                    with np.load(npz_path, allow_pickle=True) as npz:
                        if "imgs" in npz:
                            frame_arr = npz["imgs"][int(frame)]
                        else:
                            # fallback to first array if key missing
                            frame_arr = npz[npz.files[0]][int(frame)]
                    frame_img = self._np_array_to_pil(frame_arr)
                    all_images.append((img_id, frame_img))
                else:
                    with g_pathmgr.open(path, "rb") as fopen:
                        all_images.append((img_id, PILImage.open(fopen).convert("RGB")))
            except FileNotFoundError as e:
                print(f"File not found: {path} from dataset: {self.annFile}")
                raise e
        return all_images, all_img_metadata

    def set_curr_epoch(self, epoch: int):
        self.curr_epoch = epoch

    def set_epoch(self, epoch: int):
        pass

    def _np_array_to_pil(self, arr: np.ndarray) -> PILImage.Image:
        if isinstance(arr, PILImage.Image):
            return arr.convert("RGB")
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            img = arr
        elif arr.ndim == 2:
            img = np.expand_dims(arr, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):
            img = np.transpose(arr, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported npz frame shape: {arr.shape}")

        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        if img.dtype != np.uint8:
            img_min = float(np.min(img))
            img_max = float(np.max(img))
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            img = (img * 255.0).clip(0, 255).astype(np.uint8)

        return PILImage.fromarray(img, mode="RGB")

    def set_sharded_annotation_file(self, data_epoch: int):
        if self.coco is not None:
            return
        assert g_pathmgr.isfile(self.annFile), f"please provide valid annotation file. Missing: {self.annFile}"
        annFile = g_pathmgr.get_local_path(self.annFile)
        if self.coco is not None:
            del self.coco
        self.coco = self.coco_json_loader(annFile)
        ids_list = list(sorted(self.coco.getDatapointIds()))
        if self.limit_ids is not None:
            local_random = random.Random(len(ids_list))
            local_random.shuffle(ids_list)
            ids_list = ids_list[: self.limit_ids]
        self.ids = torch.as_tensor(ids_list, dtype=torch.long)

    def __getitem__(self, index: int) -> Datapoint:
        return self._load_datapoint(index)

    def _load_datapoint(self, index: int) -> Datapoint:
        id = self.ids[index].item()
        pil_images, img_metadata = self._load_images(id)
        queries, annotations = self.coco.loadQueriesAndAnnotationsFromDatapoint(id)
        return self.load_queries(pil_images, annotations, queries, img_metadata)

    def load_queries(self, pil_images, annotations, queries, img_metadata):
        images: List[Image] = []
        id2index_img = {}
        id2index_obj = {}
        id2imsize = {}
        assert len(pil_images) == len(img_metadata)
        for i in range(len(pil_images)):
            w, h = pil_images[i][1].size
            blurring_mask = img_metadata[i].get("blurring_mask")
            images.append(
                Image(data=pil_images[i][1], objects=[], size=(h, w), blurring_mask=blurring_mask)
            )
            id2index_img[pil_images[i][0]] = i
            id2imsize[pil_images[i][0]] = (h, w)
        for annotation in annotations:
            image_id = id2index_img[annotation["image_id"]]
            bbox = box_xywh_to_xyxy(torch.as_tensor(annotation["bbox"])).view(1, 4)
            h, w = id2imsize[annotation["image_id"]]
            bbox[:, 0::2].mul_(w).clamp_(min=0, max=w)
            bbox[:, 1::2].mul_(h).clamp_(min=0, max=h)
            segment = None
            if self.load_segmentation and "segmentation" in annotation:
                segment = annotation["segmentation"]
            images[image_id].objects.append(
                Object(
                    bbox=bbox[0],
                    area=annotation["area"],
                    object_id=annotation.get("object_id", -1),
                    frame_index=annotation.get("frame_index", -1),
                    segment=segment,
                    is_crowd=annotation.get("is_crowd"),
                    source=annotation.get("source", ""),
                )
            )
            id2index_obj[annotation["id"]] = len(images[image_id].objects) - 1
        find_queries = []
        stage2num_queries = Counter()
        for query in queries:
            stage2num_queries[query["query_processing_order"]] += 1
        num_queries_per_stage = stage2num_queries.most_common(1)[0][1] if stage2num_queries else 0
        for stage, num_queries in stage2num_queries.items():
            assert num_queries == num_queries_per_stage, (
                f"Number of queries in stage {stage} is {num_queries}, expected {num_queries_per_stage}"
            )
        for query in queries:
            h, w = id2imsize[query["image_id"]]
            if "input_box" in query and query["input_box"] is not None and len(query["input_box"]) > 0:
                bbox = box_xywh_to_xyxy(torch.as_tensor(query["input_box"])).view(-1, 4)
                bbox[:, 0::2].mul_(w).clamp_(min=0, max=w)
                bbox[:, 1::2].mul_(h).clamp_(min=0, max=h)
                bbox_label = (
                    torch.as_tensor(query["input_box_label"], dtype=torch.long).view(-1)
                    if query.get("input_box_label") is not None
                    else torch.ones(len(bbox), dtype=torch.long)
                )
            else:
                bbox = None
                bbox_label = None
            if "input_points" in query and query["input_points"] is not None:
                points = torch.as_tensor(query["input_points"]).view(1, -1, 3)
                points[:, :, 0:1].mul_(w).clamp_(min=0, max=w)
                points[:, :, 1:2].mul_(h).clamp_(min=0, max=h)
            else:
                points = None
            try:
                original_image_id = int(img_metadata[id2index_img[query["image_id"]]]["original_img_id"])
            except (ValueError, KeyError):
                original_image_id = -1
            try:
                img_metadata_query = img_metadata[id2index_img[query["image_id"]]]
                coco_image_id = int(img_metadata_query.get("coco_img_id", query["id"]))
            except (KeyError, ValueError):
                coco_image_id = -1
            try:
                original_category_id = int(query["original_cat_id"])
            except (ValueError, KeyError):
                original_category_id = -1
            if query.get("object_ids_output"):
                obj_id = query["object_ids_output"][0]
                obj_idx = id2index_obj[obj_id]
                image_idx = id2index_img[query["image_id"]]
                object_id = images[image_idx].objects[obj_idx].object_id
                frame_index = images[image_idx].objects[obj_idx].frame_index
            else:
                object_id = -1
                frame_index = -1
            find_queries.append(
                FindQueryLoaded(
                    query_text=query.get("query_text") or "",
                    image_id=id2index_img[query["image_id"]],
                    input_bbox=bbox,
                    input_bbox_label=bbox_label,
                    input_points=points,
                    object_ids_output=[id2index_obj[obj_id] for obj_id in query.get("object_ids_output", [])],
                    is_exhaustive=query.get("is_exhaustive", True),
                    is_pixel_exhaustive=query.get("is_pixel_exhaustive", query.get("is_exhaustive")),
                    query_processing_order=query["query_processing_order"],
                    inference_metadata=InferenceMetadata(
                        coco_image_id=-1 if self.training else coco_image_id,
                        original_image_id=-1 if self.training else original_image_id,
                        frame_index=frame_index,
                        original_category_id=original_category_id,
                        original_size=(h, w),
                        object_id=object_id,
                    ),
                )
            )
        return Datapoint(find_queries=find_queries, images=images, raw_images=[p[1] for p in pil_images])

    def __len__(self) -> int:
        return len(self.ids)


class Sam3ImageDataset(CustomCocoDetectionAPI):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        max_ann_per_img: int,
        multiplier: int,
        training: bool,
        load_segmentation: bool = False,
        max_train_queries: int = 300,
        max_val_queries: int = 300,
        fix_fname: bool = False,
        is_sharded_annotation_dir: bool = False,
        blurring_masks_path: Optional[str] = None,
        use_caching: bool = True,
        zstd_dict_path=None,
        filter_query=None,
        coco_json_loader: Callable = COCO_FROM_JSON,
        limit_ids: int = None,
    ):
        super(Sam3ImageDataset, self).__init__(
            img_folder,
            ann_file,
            fix_fname=fix_fname,
            load_segmentation=load_segmentation,
            training=training,
            blurring_masks_path=blurring_masks_path,
            use_caching=use_caching,
            zstd_dict_path=zstd_dict_path,
            filter_query=filter_query,
            coco_json_loader=coco_json_loader,
            limit_ids=limit_ids,
        )
        self._transforms = transforms
        self.training = training
        self.max_ann_per_img = max_ann_per_img
        self.max_train_queries = max_train_queries
        self.max_val_queries = max_val_queries
        self.repeat_factors = torch.ones(len(self.ids), dtype=torch.float32) * multiplier
        print(f"Raw dataset length = {len(self.ids)}")
        self._MAX_RETRIES = 100

    def __getitem__(self, idx):
        return self.__orig_getitem__(idx)

    def __orig_getitem__(self, idx):
        for _ in range(self._MAX_RETRIES):
            try:
                datapoint = super(Sam3ImageDataset, self).__getitem__(idx)
                for q in datapoint.find_queries:
                    if len(q.object_ids_output) > self.max_ann_per_img:
                        raise DecompressionBombError(f"Too many outputs ({len(q.object_ids_output)})")
                max_queries = self.max_train_queries if self.training else self.max_val_queries
                if len(datapoint.find_queries) > max_queries:
                    raise DecompressionBombError(f"Too many find queries ({len(datapoint.find_queries)})")
                if len(datapoint.find_queries) == 0:
                    raise DecompressionBombError("No find queries")
                for transform in self._transforms:
                    datapoint = transform(datapoint, epoch=self.curr_epoch)
                break
            except (DecompressionBombError, OSError, ValueError) as error:
                sys.stderr.write(f"ERROR: got loading error on datapoint {idx}\n")
                sys.stderr.write(f"Exception: {error}\n")
                sys.stderr.write(traceback.format_exc())
                idx = (idx + 1) % len(self)
        else:
            raise RuntimeError(f"Failed {self._MAX_RETRIES} times trying to load an image.")
        return datapoint
