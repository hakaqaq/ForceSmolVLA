from __future__ import annotations

import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from forcesmolvla import action_delta
from forcesmolvla.action_delta import decode_binary_gripper_width
from forcesmolvla.rft.critic_action_adapter_v2 import (
    critic_action_for_q_guidance_v2,
    normalized_gripper_endpoints_v2,
    project_binary_gripper_width_v2,
    raw_gripper_out_of_public_tolerance_mask,
)
from forcesmolvla.rft.gripper_domain_audit import global_rng_digest


MEAN = torch.tensor([0.0] * 6 + [0.028491082421846097], dtype=torch.float32)
STD = torch.tensor([1.0] * 6 + [0.04012480845771951], dtype=torch.float32)
OFFENDER_NORMALIZED = 1.71746826171875
OFFENDER_METERS = 0.09740415960550308


def test_offender_internal_projects_open_while_public_still_rejects() -> None:
    chunk = torch.zeros(1, 50, 7)
    chunk[..., 6] = OFFENDER_NORMALIZED
    internal = critic_action_for_q_guidance_v2(
        chunk, delta_action_mean7=MEAN, delta_action_std7=STD
    )
    endpoints = normalized_gripper_endpoints_v2(MEAN, STD)
    assert internal.shape == (1, 3, 7)
    assert torch.all(internal[..., 6] == endpoints[1])
    physical = (chunk.numpy() * STD.numpy() + MEAN.numpy())
    assert float(physical[0, 0, 6]) == OFFENDER_METERS
    with pytest.raises(ValueError, match="outside the frozen"):
        decode_binary_gripper_width(physical)


def test_threshold_direction_and_tie_match_public_decoder() -> None:
    threshold = action_delta.BINARY_GRIPPER_SWITCH_WIDTH_M
    eps = 1e-7
    values = np.array([threshold - eps, threshold, threshold + eps], dtype=np.float64)
    action = np.zeros((3, 7), dtype=np.float64); action[:, 6] = values
    public = decode_binary_gripper_width(action)[:, 6]
    internal = project_binary_gripper_width_v2(torch.from_numpy(values)).numpy()
    np.testing.assert_array_equal(public.astype(np.float32), internal)
    np.testing.assert_array_equal(internal, np.array([0.0, 0.085, 0.085], dtype=np.float32))


def test_all_finite_values_total_binary_nonfinite_fail_closed_and_rng_free() -> None:
    values = torch.tensor(
        [-torch.finfo(torch.float32).max, -1.0, 0.0, 0.0425, 1.0,
         torch.finfo(torch.float32).max], dtype=torch.float32,
    )
    before = global_rng_digest()
    first = project_binary_gripper_width_v2(values)
    second = project_binary_gripper_width_v2(values)
    assert global_rng_digest() == before
    assert torch.equal(first, second)
    assert set(first.tolist()) == {0.0, torch.tensor(0.085).item()}
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="NONFINITE"):
            project_binary_gripper_width_v2(torch.tensor([value]))


def test_tcp_gradient_gripper_stop_and_no_public_execution_calls() -> None:
    chunk = torch.randn(2, 50, 7, requires_grad=True)
    weights = torch.arange(1, 22, dtype=torch.float32).view(3, 7)
    with (
        patch.object(action_delta.ActionDeltaProcessor, "from_delta", side_effect=AssertionError),
        patch.object(action_delta.ActionSafetyProfile, "validate_chunk", side_effect=AssertionError),
        patch.object(action_delta, "decode_binary_gripper_width", side_effect=AssertionError),
    ):
        action = critic_action_for_q_guidance_v2(
            chunk, delta_action_mean7=MEAN, delta_action_std7=STD
        )
    (action * weights).sum().backward()
    assert torch.all(chunk.grad[:, :3, :6] != 0)
    assert torch.count_nonzero(chunk.grad[:, :3, 6]) == 0
    assert torch.count_nonzero(chunk.grad[:, 3:]) == 0


def test_only_normalized_endpoints_enter_q_and_duplicates_are_preserved() -> None:
    chunk = torch.zeros(3, 50, 7)
    chunk[0, :, 6] = -100.0
    chunk[1:, :, 6] = 100.0
    action = critic_action_for_q_guidance_v2(
        chunk, delta_action_mean7=MEAN, delta_action_std7=STD
    )
    endpoints = normalized_gripper_endpoints_v2(MEAN, STD)
    values = action[..., 6]
    assert torch.all((values == endpoints[0]) | (values == endpoints[1]))
    assert torch.equal(action[1], action[2])
    mask = raw_gripper_out_of_public_tolerance_mask(
        chunk[:, :3, 6], gripper_mean=MEAN[6], gripper_std=STD[6]
    )
    assert mask.shape == (3, 3) and bool(mask.all())
