import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from forcesmolvla.action_delta import ActionSafetyProfile
from forcesmolvla.context import ChunkContext
from forcesmolvla.modeling_forcesmolvla import ActionInferenceError, ForceSmolVLAPolicy
from forcesmolvla.normalizer import CartesianNormalizerBundle, FrozenFeatureNormalizer
from forcesmolvla.training_data import RuntimeArtifactBundle


def _feature(name: str, width: int) -> FrozenFeatureNormalizer:
    return FrozenFeatureNormalizer(name, np.zeros(width), np.ones(width), ("episode-0",))


def _artifacts(digest: str) -> RuntimeArtifactBundle:
    return RuntimeArtifactBundle(
        normalizer=CartesianNormalizerBundle(
            state7=_feature("state7", 7),
            wrench6=_feature("wrench6", 6),
            delta_action7=_feature("delta_action7", 7),
            split_sha256=digest,
            calibration_bundle_sha256=digest,
            wrench_geometry_spec_sha256=digest,
        ),
        normalizer_manifest_sha256=digest,
        calibration_bundle_sha256=digest,
        wrench_geometry_spec_sha256=digest,
        split_sha256=digest,
        action_delta_spec_sha256=digest,
        action_delta_source_sha256=hashlib.sha256(
            (Path(__file__).parents[1] / "src/forcesmolvla/action_delta.py").read_bytes()
        ).hexdigest(),
    )


def _context(digest: str) -> ChunkContext:
    valid = torch.ones(2, 3, dtype=torch.bool)
    return ChunkContext(
        policy_generation=0,
        raw_state_snapshot=torch.tensor(
            [[0.5, 0, 0, 0, 0, 0, 0.04], [0.4, 0, 0, 0, 0, 0, 0.03]],
            dtype=torch.float32,
        ),
        t_ref_ns=torch.tensor([1, 2], dtype=torch.int64),
        tau0_ns=torch.tensor([3, 4], dtype=torch.int64),
        clock_domain_id=("sensor", "sensor"),
        episode_id=("episode-0", "episode-0"),
        session_id=("session-0", "session-0"),
        sample_id=("sample-0", "sample-1"),
        chunk_id=("chunk-0", "chunk-1"),
        action_valid_mask=valid,
        suffix_valid_mask=valid.clone(),
        calibration_bundle_hash=(digest, digest),
        wrench_geometry_spec_hash=(digest, digest),
        normalizer_hash=(digest, digest),
        calibration_mapping_hash_or_none=(None, None),
        wrench_geometry_valid=torch.ones(2, dtype=torch.bool),
        runtime_artifact_compatible=torch.ones(2, dtype=torch.bool),
        selected_provenance=({}, {}),
    )


class _PredictionHarness(torch.nn.Module):
    _predict_action_chunks = ForceSmolVLAPolicy._predict_action_chunks
    predict_action_chunk = ForceSmolVLAPolicy.predict_action_chunk

    def __init__(self, normalized_delta7, artifacts, safety_profile):
        super().__init__()
        self.normalized_delta7 = normalized_delta7
        self._runtime_artifacts = artifacts
        self._action_safety_profile = safety_profile
        self._consumed_chunk_ids = set()

    def _predict_normalized_delta_chunk(self, _batch, chunk_context=None, noise=None, **kwargs):
        return self.normalized_delta7


def test_public_action_api_is_absolute_safe_no_grad_and_batch_atomic():
    root = Path(__file__).parents[1]
    rules_path = root / "tests/fixtures/shadow_safety_thresholds.test_only.yaml"
    profile = ActionSafetyProfile.from_rulespec(
        yaml.safe_load(rules_path.read_text()),
        rules_sha256=hashlib.sha256(rules_path.read_bytes()).hexdigest(),
    )
    digest = "a" * 64
    normalized = torch.zeros(2, 3, 7, dtype=torch.float32, requires_grad=True)
    normalized = normalized.clone()
    normalized[..., 0] = 0.1
    normalized[..., 6] = 0.05
    normalized[1, 1, 0] = 11.0
    policy = _PredictionHarness(normalized, _artifacts(digest), profile).eval()
    context = _context(digest)

    with pytest.raises(ActionInferenceError, match="workspace") as failure:
        policy.predict_action_chunk({}, context)
    assert failure.value.code == "ACTION_INFERENCE_VALUE_INVALID"
    assert policy._consumed_chunk_ids == set()
    private_result = policy._predict_normalized_delta_chunk({}, context)
    assert private_result is policy.normalized_delta7
    assert policy._consumed_chunk_ids == set()

    policy.normalized_delta7 = normalized.detach().clone()
    policy.normalized_delta7[1, 1, 0] = 0.1
    absolute = policy.predict_action_chunk({}, context)
    assert absolute.shape == (2, 3, 7)
    assert not absolute.requires_grad
    torch.testing.assert_close(absolute[0, :, 0], torch.full((3,), 0.6))
    torch.testing.assert_close(absolute[1, :, 0], torch.full((3,), 0.5))
    torch.testing.assert_close(absolute[..., 6], torch.full((2, 3), 0.085))
    assert policy._consumed_chunk_ids == {"chunk-0", "chunk-1"}


def test_public_action_api_saturates_finite_gripper_and_rejects_nonfinite():
    root = Path(__file__).parents[1]
    rules_path = root / "tests/fixtures/shadow_safety_thresholds.test_only.yaml"
    profile = ActionSafetyProfile.from_rulespec(
        yaml.safe_load(rules_path.read_text()),
        rules_sha256=hashlib.sha256(rules_path.read_bytes()).hexdigest(),
    )
    digest = "e" * 64
    normalized = torch.zeros(2, 3, 7, dtype=torch.float32)
    normalized[0, :, 6] = torch.tensor([-0.0026, 0.0031, 0.042499])
    normalized[1, :, 6] = torch.tensor([0.0425, 0.085, 0.095])
    policy = _PredictionHarness(normalized, _artifacts(digest), profile).eval()
    context = _context(digest)
    absolute = policy.predict_action_chunk({}, context)
    torch.testing.assert_close(absolute[0, :, 6], torch.zeros(3))
    torch.testing.assert_close(absolute[1, :, 6], torch.full((3,), 0.085))
    assert policy._consumed_chunk_ids == {"chunk-0", "chunk-1"}

    context = replace(
        context,
        chunk_id=("chunk-2", "chunk-3"),
        sample_id=("sample-2", "sample-3"),
    )
    policy.normalized_delta7 = normalized.clone()
    policy.normalized_delta7[0, 0, 6] = -0.0101
    policy.normalized_delta7[1, 0, 6] = 0.097404
    saturated = policy.predict_action_chunk({}, context)
    assert saturated[0, 0, 6] == 0.0
    assert saturated[1, 0, 6] == 0.085
    assert policy._consumed_chunk_ids == {
        "chunk-0", "chunk-1", "chunk-2", "chunk-3",
    }

    context = replace(
        context,
        chunk_id=("chunk-4", "chunk-5"),
        sample_id=("sample-4", "sample-5"),
    )
    policy.normalized_delta7[1, 0, 6] = float("inf")
    with pytest.raises(ActionInferenceError, match="nonfinite|must be finite"):
        policy.predict_action_chunk({}, context)


def test_public_action_api_sets_eval_and_zeroes_invalid_tail():
    root = Path(__file__).parents[1]
    rules_path = root / "tests/fixtures/shadow_safety_thresholds.test_only.yaml"
    profile = ActionSafetyProfile.from_rulespec(
        yaml.safe_load(rules_path.read_text()),
        rules_sha256=hashlib.sha256(rules_path.read_bytes()).hexdigest(),
    )
    digest = "b" * 64
    normalized = torch.zeros(2, 4, 7, dtype=torch.float32)
    normalized[..., 6] = 0.05
    policy = _PredictionHarness(normalized, _artifacts(digest), profile).train()
    context = _context(digest)
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    context = replace(
        context,
        action_valid_mask=valid,
        suffix_valid_mask=valid.clone(),
    )
    absolute = policy.predict_action_chunk({}, context)
    assert not policy.training
    assert absolute.shape == (2, 4, 7)
    assert absolute.dtype == torch.float32
    assert torch.count_nonzero(absolute[1, 3]) == 0


def test_runtime_action_delta_source_hash_tamper_fails_closed():
    artifacts = replace(_artifacts("c" * 64), action_delta_source_sha256="d" * 64)
    with pytest.raises(RuntimeError, match="RUNTIME_ACTION_DELTA_SOURCE_HASH_MISMATCH"):
        artifacts.validate_action_contract()
