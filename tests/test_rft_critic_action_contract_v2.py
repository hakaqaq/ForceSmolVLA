from __future__ import annotations

import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from forcesmolvla import action_delta
from forcesmolvla.action_delta import decode_binary_gripper_width
from forcesmolvla.rft.critic_action_adapter_v2 import (
    aligned_fresh_chunk_execution_index_map_v2,
    critic_action_for_q_guidance_v2,
    normalized_gripper_endpoints_v2,
    project_binary_gripper_width_v2,
    raw_gripper_out_of_public_tolerance_mask,
)
from forcesmolvla.rft.gripper_domain_audit import global_rng_digest
from forcesmolvla.rft.online.transition_authority import AcceptedAck, causal_zoh_ack_macro


MEAN = torch.tensor([0.0] * 6 + [0.028491082421846097], dtype=torch.float32)
STD = torch.tensor([1.0] * 6 + [0.04012480845771951], dtype=torch.float32)
OFFENDER_NORMALIZED = 1.71746826171875
OFFENDER_METERS = 0.09740415960550308
EXECUTION_INDEX_MAP = aligned_fresh_chunk_execution_index_map_v2()


def test_offender_internal_projects_open_while_public_still_rejects() -> None:
    chunk = torch.zeros(1, 50, 7)
    chunk[..., 6] = OFFENDER_NORMALIZED
    internal = critic_action_for_q_guidance_v2(
        chunk, execution_index_map=EXECUTION_INDEX_MAP,
        delta_action_mean7=MEAN, delta_action_std7=STD
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
            chunk, execution_index_map=EXECUTION_INDEX_MAP,
            delta_action_mean7=MEAN, delta_action_std7=STD
        )
    (action * weights).sum().backward()
    assert torch.all(chunk.grad[:, 0, :6] != 0)
    assert torch.count_nonzero(chunk.grad[..., 6]) == 0
    assert torch.count_nonzero(chunk.grad[:, 1:]) == 0


def test_only_normalized_endpoints_enter_q_and_duplicates_are_preserved() -> None:
    chunk = torch.zeros(3, 50, 7)
    chunk[0, :, 6] = -100.0
    chunk[1:, :, 6] = 100.0
    action = critic_action_for_q_guidance_v2(
        chunk, execution_index_map=EXECUTION_INDEX_MAP,
        delta_action_mean7=MEAN, delta_action_std7=STD
    )
    endpoints = normalized_gripper_endpoints_v2(MEAN, STD)
    values = action[..., 6]
    assert torch.all((values == endpoints[0]) | (values == endpoints[1]))
    assert torch.equal(action[1], action[2])
    mask = raw_gripper_out_of_public_tolerance_mask(
        chunk[:, :3, 6], gripper_mean=MEAN[6], gripper_std=STD[6]
    )
    assert mask.shape == (3, 3) and bool(mask.all())


def test_effective_control_is_explicit_dispatch_zoh_not_first_three_slots() -> None:
    chunk = torch.arange(50, dtype=torch.float32).view(1, 50, 1).expand(-1, -1, 7).clone()
    action = critic_action_for_q_guidance_v2(
        chunk,
        execution_index_map=EXECUTION_INDEX_MAP,
        delta_action_mean7=MEAN,
        delta_action_std7=STD,
    )
    assert EXECUTION_INDEX_MAP == (0, 0, 0)
    assert torch.equal(action[..., :6], chunk[:, [0, 0, 0], :6])
    assert not torch.equal(action[..., :6], chunk[:, :3, :6])


def test_actor_projection_matches_known_ack_zoh_trajectory_slot_by_slot() -> None:
    accepted = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.0)
    replay = causal_zoh_ack_macro(
        [
            AcceptedAck(
                ack_id="ack",
                receive_monotonic_ns=999_000_000,
                accepted_absolute_action7=accepted,
                gripper_command_id="gripper",
                gripper_ack_command_id="gripper",
                slot_owner="policy",
                accepted_action_source="policy",
                intervention=False,
                source_command_id="pose-command",
                source_dispatch_sequence=0,
                source_model_index=0,
                episode_id="episode",
                policy_revision="revision",
                chunk_id="chunk",
                chunk_compatibility_key="generation-0",
                clock_domain="upper-host-monotonic",
                controller_authority="fr3-reference-controller",
            )
        ],
        (1_000_000_000, 1_033_333_333, 1_066_666_667),
        max_ack_age_ms=100.0,
    )
    chunk = torch.zeros(1, 50, 7)
    chunk[:, 0] = torch.tensor(accepted)
    actor = critic_action_for_q_guidance_v2(
        chunk,
        execution_index_map=EXECUTION_INDEX_MAP,
        delta_action_mean7=torch.zeros(7),
        delta_action_std7=torch.ones(7),
    )
    np.testing.assert_allclose(
        actor.squeeze(0).numpy(), replay.accepted_absolute_action_k7
    )
