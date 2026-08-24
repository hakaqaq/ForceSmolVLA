import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from forcesmolvla.action_delta import (
    ActionDeltaProcessor,
    ActionSafetyProfile,
    canonicalize_zyx,
    decode_binary_gripper_width,
)
from forcesmolvla.masks import action_masks, pack_active_features


def test_action_delta_whole_chunk_roundtrip():
    state = np.array([0.5, -0.2, 0.1, 3.1, -0.2, -3.1, 0.04])
    actions = np.array(
        [
            [0.6, -0.1, 0.2, -3.1, -0.1, 3.1, 0.05],
            [0.7, 0.0, 0.3, 3.0, 0.0, -3.0, 0.06],
        ]
    )
    delta = ActionDeltaProcessor.to_delta(actions, state)
    restored = ActionDeltaProcessor.from_delta(delta, state)
    np.testing.assert_allclose(restored, actions, atol=1e-12)


def test_action_delta_batched_state_binding():
    state = np.array([[0, 0, 0, 0, 0, 0, 0.02], [1, 2, 3, 0.1, 0.2, 0.3, 0.04]])
    actions = np.stack([np.tile(state[0], (3, 1)), np.tile(state[1], (3, 1))])
    delta = ActionDeltaProcessor.to_delta(actions, state)
    np.testing.assert_allclose(delta[..., :6], 0)
    np.testing.assert_allclose(delta[..., 6], actions[..., 6])
    np.testing.assert_allclose(ActionDeltaProcessor.from_delta(delta, state), actions)


def test_zyx_inverse_returns_principal_chart_and_rejects_singular_region():
    state = np.array([0, 0, 0, 0.1, 0.2, 0.3, 0.04])
    action = np.array([[0, 0, 0, 0.2, 2.0, -0.4, 0.05]])
    restored = ActionDeltaProcessor.from_delta(
        ActionDeltaProcessor.to_delta(action, state), state
    )
    np.testing.assert_allclose(restored[0, 3:6], canonicalize_zyx(action[0, 3:6]))
    singular = action.copy()
    singular[0, 4] = np.pi / 2 - np.deg2rad(1)
    with pytest.raises(ValueError, match="singular"):
        ActionDeltaProcessor.to_delta(singular, state)


def test_test_only_action_safety_profile_checks_width_and_workspace():
    root = Path(__file__).parents[1]
    path = root / "tests/fixtures/shadow_safety_thresholds.test_only.yaml"
    rules = yaml.safe_load(path.read_text())
    profile = ActionSafetyProfile.from_rulespec(
        rules, rules_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    state = np.array([[0, 0, 0, 0, 0, 0, 0.04]])
    actions = np.tile(state[:, None, :], (1, 3, 1))
    mask = np.ones((1, 3), dtype=np.bool_)
    profile.validate_chunk(actions, mask, state)
    outside = actions.copy()
    outside[0, 1, 0] = 11
    with pytest.raises(ValueError, match="workspace"):
        profile.validate_chunk(outside, mask, state)


def test_binary_gripper_decoder_is_fail_closed_and_not_clipping():
    candidates = np.zeros((1, 5, 7), dtype=np.float64)
    candidates[0, :, 6] = [-0.01, 0.0, 0.042499, 0.0425, 0.095]
    decoded = decode_binary_gripper_width(candidates)
    np.testing.assert_array_equal(decoded[..., :6], candidates[..., :6])
    np.testing.assert_array_equal(decoded[0, :, 6], [0.0, 0.0, 0.0, 0.085, 0.085])

    outside = candidates.copy()
    outside[0, 0, 6] = -0.0100001
    with pytest.raises(ValueError, match="frozen.*tolerance"):
        decode_binary_gripper_width(outside)
    outside[0, 0, 6] = 0.0950001
    with pytest.raises(ValueError, match="frozen.*tolerance"):
        decode_binary_gripper_width(outside)


def test_live_binary_gripper_rate_accepts_exact_switch_only():
    root = Path(__file__).parents[1]
    path = root / "configs/live_action_safety.task2.development.yaml"
    rules = yaml.safe_load(path.read_text())
    rules["mode"] = "test_only"
    profile = ActionSafetyProfile.from_rulespec(
        rules,
        rules_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    state = np.array([[0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]])
    actions = np.tile(state[:, None, :], (1, 2, 1))
    actions[0, 1, 6] = 0.085
    mask = np.ones((1, 2), dtype=np.bool_)
    profile.validate_chunk(actions, mask, state)

    above = actions.copy()
    above[0, 1, 6] = np.nextafter(0.085, np.inf)
    with pytest.raises(RuntimeError, match="SHADOW_GRIPPER_INVALID"):
        profile.validate_chunk(above, mask, state)


def test_padding_is_zero_and_masked():
    active = np.arange(14, dtype=np.float32).reshape(1, 2, 7)
    packed = pack_active_features(active)
    assert packed.shape == (1, 2, 32)
    assert np.all(packed[..., 7:] == 0)
    masks = action_masks(np.array([[True, False]]))
    assert masks["action_feature_mask"].shape == (1, 2, 32)
    assert masks["action_feature_mask"][0, 0, :7].all()
    assert not masks["action_feature_mask"][0, 0, 7:].any()
    assert not masks["action_feature_mask"][0, 1].any()
