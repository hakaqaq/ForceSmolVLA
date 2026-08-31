from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "tools/reward_classifier/conrft_lerobot_v3_adapter.py"
SPEC = importlib.util.spec_from_file_location("conrft_lerobot_v3_adapter", ADAPTER_PATH)
assert SPEC is not None and SPEC.loader is not None
adapter_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter_module
SPEC.loader.exec_module(adapter_module)

from forcesmolvla.rft.reward_detector import (  # noqa: E402
    SyntheticCausalRewardDetector,
    align_confirmation_to_policy_boundary,
)


def _reference(episode: str = "episode_000000", frame: int = 0):
    return adapter_module.RowReference(
        dataset_root_id="task2_lerobotv3",
        parquet_relative_path="data/chunk-000/file-000.parquet",
        row_index=frame,
        episode_id=episode,
        frame_index=frame,
        timestamp=frame / 30.0,
    )


def _identity():
    return adapter_module.CameraRowIdentity(101, 102)


def _images():
    camera1 = np.zeros((3, 480, 640), dtype=np.uint8)
    camera2 = np.zeros((3, 480, 640), dtype=np.uint8)
    camera1[0] = 11
    camera1[1] = 22
    camera1[2] = 33
    camera2[0] = 44
    camera2[1] = 55
    camera2[2] = 66
    return {
        "observation.images.camera1": camera1,
        "observation.images.camera2": camera2,
    }


def test_dual_camera_order_rgb_layout_and_dtype_sentinel():
    adapted = adapter_module.ConRFTLeRobotV3Adapter().adapt(
        _images(), row_reference=_reference(), camera_row_identity=_identity()
    )
    assert tuple(adapted.observation) == ("d435_third_person", "d405_wrist")
    first, second = adapted.observation.values()
    assert first.shape == second.shape == (1, 1, 480, 640, 3)
    assert first.dtype == second.dtype == np.uint8
    assert first[0, 0, 0, 0].tolist() == [11, 22, 33]
    assert second[0, 0, 0, 0].tolist() == [44, 55, 66]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda x: {**x, "observation.images.camera1": x["observation.images.camera1"].astype(np.float32)}, TypeError),
        (lambda x: {**x, "observation.images.camera1": np.zeros((480, 640, 3), dtype=np.uint8)}, ValueError),
        (lambda x: dict(reversed(list(x.items()))), ValueError),
    ],
)
def test_dtype_shape_and_order_errors_fail_explicitly(mutation, error):
    with pytest.raises(error):
        adapter_module.ConRFTLeRobotV3Adapter().adapt(
            mutation(_images()), row_reference=_reference(), camera_row_identity=_identity()
        )


def test_bgr_is_explicitly_rejected():
    with pytest.raises(ValueError, match="RGB"):
        adapter_module.ConRFTLeRobotV3Adapter().adapt(
            _images(),
            row_reference=_reference(),
            camera_row_identity=_identity(),
            source_color_order="BGR",
        )


def test_eval_adapter_is_deterministic_and_future_independent():
    adapter = adapter_module.ConRFTLeRobotV3Adapter()
    current = _images()
    future = _images()
    first = adapter.adapt(current, row_reference=_reference(), camera_row_identity=_identity())
    future["observation.images.camera1"][:] = 255
    second = adapter.adapt(current, row_reference=_reference(), camera_row_identity=_identity())
    for key in first.observation:
        np.testing.assert_array_equal(first.observation[key], second.observation[key])


def test_episode_switch_resets_identity_without_retaining_frames():
    adapter = adapter_module.ConRFTLeRobotV3Adapter()
    a = adapter.adapt(_images(), row_reference=_reference("ep_a"), camera_row_identity=_identity())
    b = adapter.adapt(_images(), row_reference=_reference("ep_a", 1), camera_row_identity=_identity())
    c = adapter.adapt(_images(), row_reference=_reference("ep_b"), camera_row_identity=_identity())
    assert (a.reset_at_episode_boundary, b.reset_at_episode_boundary, c.reset_at_episode_boundary) == (True, False, True)
    assert adapter.episode_reset_count == 2
    assert adapter.retained_frame_count == 0


def test_causal_streak_confirms_now_not_at_onset_and_resets_by_episode():
    detector = SyntheticCausalRewardDetector(probability_threshold=0.8, consecutive_frames=3)
    assert detector.update(episode_id="a", frame_index=0, probability=0.9) is None
    assert detector.update(episode_id="a", frame_index=1, probability=0.9) is None
    event = detector.update(episode_id="a", frame_index=2, probability=0.9)
    assert event is not None
    assert event.confirmation_frame == 2
    assert event.aligned_policy_frame == 3
    assert event.alignment_delay_frames == 1
    assert detector.update(episode_id="b", frame_index=0, probability=0.9) is None


def test_alignment_delay_is_zero_to_two_frames_and_never_backward():
    for frame in range(30):
        aligned, delay = align_confirmation_to_policy_boundary(frame)
        assert aligned >= frame
        assert aligned % 3 == 0
        assert delay in (0, 1, 2)


def test_detector_rejects_noncausal_frame_gaps():
    detector = SyntheticCausalRewardDetector(probability_threshold=0.8, consecutive_frames=2)
    detector.update(episode_id="a", frame_index=0, probability=0.1)
    with pytest.raises(ValueError, match="consecutive causal"):
        detector.update(episode_id="a", frame_index=2, probability=0.9)


def test_future_probabilities_cannot_change_current_confirmation():
    prefix = [0.1, 0.9, 0.9]
    confirmations = []
    for suffix in ([0.0, 0.0], [1.0, 1.0]):
        detector = SyntheticCausalRewardDetector(probability_threshold=0.8, consecutive_frames=2)
        event_at_prefix = None
        for frame, probability in enumerate([*prefix, *suffix]):
            event = detector.update(episode_id="a", frame_index=frame, probability=probability)
            if frame == len(prefix) - 1:
                event_at_prefix = event
        confirmations.append(event_at_prefix)
    assert confirmations[0] == confirmations[1]
    assert confirmations[0] is not None
    assert confirmations[0].confirmation_frame == 2


def test_real_parameters_remain_null_and_label_template_is_empty():
    detector = json.loads((ROOT / "configs/stage2_reward_detector.development.yaml").read_text())
    assert detector["probability_threshold"] is None
    assert detector["consecutive_positive_frames"] is None
    assert detector["max_detection_delay_frames"] is None
    assert detector["last_valid_frame_fallback"] == "disabled"
    review = json.loads((ROOT / "schemas/stage2_reward_classifier_review_template.json").read_text())
    assert review["records"] == []
    assert review["generation_permitted"] is False
