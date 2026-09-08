from __future__ import annotations

from pathlib import Path

import numpy as np

from medical.inference_3d import (
    InferenceConfig,
    merge_prediction,
    normalize_prompts,
    predict_prompt,
    prompts_for_volume,
    temporal_windows,
    volume_to_frames,
)


def test_temporal_windows_cover_volume_once():
    windows = temporal_windows(10, 4)
    assert windows == [(0, 4), (4, 8), (8, 10)]
    assert [z for start, end in windows for z in range(start, end)] == list(range(10))


def test_prompt_document_supports_direct_and_per_file_maps():
    payload = {}
    direct = {"1": "liver", "instance_label": 0}
    assert prompts_for_volume(Path("case.npz"), payload, None, direct) == direct
    per_file = {"case.npz": {"2": "spleen", "instance_label": 0}}
    assert (
        prompts_for_volume(Path("case.npz"), payload, None, per_file)
        == per_file["case.npz"]
    )


def test_prompt_validation_rejects_multiple_instance_queries():
    try:
        normalize_prompts({"1": "lesion", "2": "other", "instance_label": 1})
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_volume_to_frames_accepts_channel_first_rgb():
    imgs = np.zeros((2, 3, 5, 7), dtype=np.uint8)
    frames, shape = volume_to_frames(imgs)
    assert len(frames) == 2
    assert shape == (5, 7)
    assert frames.original_size_hw == (5, 7)


def test_pixel_score_fusion_uses_background_threshold_and_class_argmax():
    segmentation = np.zeros((1, 2, 3), dtype=np.uint8)
    priority = np.full(segmentation.shape, 0.5, dtype=np.float32)
    left = np.array([[[True, True, False], [True, True, False]]])
    right = np.array([[[False, True, True], [False, True, True]]])
    merge_prediction(
        segmentation,
        priority,
        left,
        1,
        np.ones(1),
        np.ones(1),
        "pixel_score",
        np.array([[[0.8, 0.6, 0.1], [0.7, 0.9, 0.1]]]),
    )
    merge_prediction(
        segmentation,
        priority,
        right,
        2,
        np.ones(1),
        np.ones(1),
        "pixel_score",
        np.array([[[0.1, 0.7, 0.9], [0.1, 0.6, 0.4]]]),
    )
    assert np.array_equal(segmentation, np.array([[[1, 2, 2], [1, 1, 0]]]))


class FakeModel:
    def __init__(self):
        self.prompts = []
        self.reset_count = 0

    def reset_state(self, state):
        self.reset_count += 1

    def add_prompt(self, inference_state, frame_idx, text_str):
        self.prompts.append((frame_idx, text_str))

    def propagate_in_video(
        self, state, start_frame_idx, max_frame_num_to_track, reverse
    ):
        for frame_index in range(
            start_frame_idx, start_frame_idx + max_frame_num_to_track + 1
        ):
            mask = np.zeros((1, 2, 2), dtype=bool)
            mask[:, frame_index % 2, :] = True
            yield frame_index, {
                "out_binary_masks": mask,
                "out_probs": np.array([0.9]),
                "out_tracker_probs": np.array([0.5]),
                "out_mask_scores": mask.astype(np.float32) * 0.4 + 0.55,
                "out_mask_score_source": "mask_logits",
            }


def test_predict_prompt_uses_independent_clips_and_all_slices():
    model = FakeModel()
    prediction, detection, tracker, pixels = predict_prompt(
        model,
        {},
        "organ",
        depth=10,
        spatial_shape=(2, 2),
        config=InferenceConfig(),
    )
    assert model.reset_count == 3
    assert model.prompts == [(0, "organ"), (4, "organ"), (8, "organ")]
    assert np.all(prediction.any(axis=(1, 2)))
    assert np.allclose(detection, 0.9)
    assert np.allclose(tracker, 0.5)
    assert pixels is not None and pixels.shape == (10, 2, 2)
