import json

import numpy as np
import pytest
import torch

from forcesmolvla.normalizer import CartesianNormalizerBundle, FrozenFeatureNormalizer
from forcesmolvla.training_data import load_normalizer_bundle, prepare_training_sample


def _normalizer(name, width):
    return FrozenFeatureNormalizer(name, np.zeros(width), np.ones(width), ("episode_000000",))


def _bundle():
    return CartesianNormalizerBundle(
        state7=_normalizer("state7", 7),
        wrench6=_normalizer("wrench6", 6),
        delta_action7=_normalizer("delta_action7", 7),
        split_sha256="1" * 64,
        calibration_bundle_sha256="2" * 64,
        wrench_geometry_spec_sha256="3" * 64,
    )


def test_real_training_transform_delta_normalizes_once_and_preserves_mask():
    state = np.array([0.5, -0.2, 0.3, 0.1, -0.2, 0.3, 0.04], dtype=np.float32)
    actions = np.tile(state, (50, 1))
    actions[:, :6] += 0.25
    actions[:, 6] = 0.04
    sample = {
        "observation.state": torch.from_numpy(state),
        "observation.wrench": torch.arange(6, dtype=torch.float32),
        "action": torch.from_numpy(actions),
        "action_is_pad": torch.tensor([False] * 47 + [True] * 3),
        "episode_index": torch.tensor(0),
        "frame_index": torch.tensor(3),
        "task": "fixture",
        "observation.images.camera1": torch.zeros(3, 8, 8),
        "observation.images.camera2": torch.zeros(3, 8, 8),
        "provenance.validity_bits": torch.tensor([0xFF]),
        "provenance.state_pose_age_ms": torch.tensor([1.0]),
        "provenance.camera1_age_ms": torch.tensor([2.0]),
        "provenance.camera2_age_ms": torch.tensor([3.0]),
        "provenance.intercamera_skew_ms": torch.tensor([1.0]),
        "provenance.pose_age_ms": torch.tensor([4.0]),
        "provenance.action_ack_age_ms": torch.tensor([5.0]),
        "provenance.pose_source_stamp_ns": torch.tensor([10]),
        "provenance.wrench_raw_source_stamp_ns": torch.tensor([20]),
        "provenance.wrench_filter_output_stamp_ns": torch.tensor([20]),
    }
    result = prepare_training_sample(sample, _bundle())
    np.testing.assert_allclose(result["delta_action7"][:, :6], 0.25)
    np.testing.assert_allclose(result["delta_action7"][:, 6], 0.04)
    assert result["action_valid_mask"].sum() == 47
    assert len(result["batch_sha256"]) == 64


def test_normalizer_manifest_tamper_fails(tmp_path):
    payload = _bundle().manifest()
    payload["features"]["state7"]["mean"][0] = 1
    (tmp_path / "normalizer_manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="population binding"):
        load_normalizer_bundle(tmp_path)


def test_legacy_same_frame_normalizer_manifest_fails_closed(tmp_path):
    payload = _bundle().manifest()
    payload.pop("schema_version")
    payload.pop("fit_contract")
    (tmp_path / "normalizer_manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="legacy or unknown"):
        load_normalizer_bundle(tmp_path)
