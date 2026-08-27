from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from forcesmolvla import action_delta
from forcesmolvla.rft import losses
from forcesmolvla.rft.flow_sampling import critic_action_for_q_guidance
from forcesmolvla.rft.gripper_domain_audit import (
    EXPECTED_PUBLIC_REJECTION,
    global_rng_digest,
    gripper_domain_layers,
)


ROOT = Path(__file__).resolve().parents[1]
MEAN = torch.tensor([0.0] * 6 + [0.028491082421846097], dtype=torch.float32)
STD = torch.tensor([1.0] * 6 + [0.04012480845771951], dtype=torch.float32)
R1_OFFENDING_G_FLOW = 1.71746826171875
R1_OFFENDING_WIDTH_M = 0.09740415960550308


def offending_action() -> torch.Tensor:
    value = torch.zeros(7, dtype=torch.float32)
    value[6] = R1_OFFENDING_G_FLOW
    return value


def test_r1_offending_fixture_is_permanent_and_domain_labeled() -> None:
    audit = gripper_domain_layers(
        offending_action(), delta_action_mean7=MEAN, delta_action_std7=STD
    )
    assert audit["g_flow_normalized"] == R1_OFFENDING_G_FLOW
    assert audit["g_unnormalized_continuous_width_m"] == R1_OFFENDING_WIDTH_M
    assert audit["valid"] is False
    assert audit["g_public_decoded_endpoint_m"] is None
    assert audit["g_critic_normalized"] is None
    assert audit["replacement_action_created"] is False

    chunk = offending_action().view(1, 1, 7).expand(1, 50, 7).clone()
    with pytest.raises(ValueError, match=r"outside the frozen"):
        critic_action_for_q_guidance(
            chunk, delta_action_mean7=MEAN, delta_action_std7=STD
        )


def test_detached_public_audit_records_without_replacement_or_rng_use() -> None:
    before = global_rng_digest()
    first = gripper_domain_layers(
        offending_action(), delta_action_mean7=MEAN, delta_action_std7=STD
    )
    second = gripper_domain_layers(
        offending_action(), delta_action_mean7=MEAN, delta_action_std7=STD
    )
    assert global_rng_digest() == before
    assert first["failure_code"] == second["failure_code"]
    assert first["public_input_action7_sha256"] == second["public_input_action7_sha256"]
    assert first["clipping_or_resampling"] is second["clipping_or_resampling"] is False

    continuous = (offending_action() * STD + MEAN).numpy()[None, :]
    with pytest.raises(ValueError, match=r"outside the frozen") as caught:
        action_delta.decode_binary_gripper_width(continuous)
    assert str(caught.value) == EXPECTED_PUBLIC_REJECTION


def test_internal_loss_sources_do_not_call_public_execution_apis() -> None:
    forbidden = (
        "predict_action_chunk", "ActionDeltaProcessor.from_delta",
        "ActionSafetyProfile", "validate_chunk",
    )
    source = inspect.getsource(losses)
    assert all(name not in source for name in forbidden)

    chunk = torch.zeros(1, 50, 7)
    chunk[..., 6] = (0.085 - MEAN[6]) / STD[6]
    with (
        patch.object(action_delta.ActionDeltaProcessor, "from_delta", side_effect=AssertionError),
        patch.object(action_delta.ActionSafetyProfile, "validate_chunk", side_effect=AssertionError),
        patch.object(action_delta, "decode_binary_gripper_width", side_effect=AssertionError),
    ):
        result = critic_action_for_q_guidance(
            chunk, delta_action_mean7=MEAN, delta_action_std7=STD
        )
    assert result.shape == (1, 3, 7)
    assert torch.all(result[..., 6] == (0.085 - MEAN[6]) / STD[6])


def test_hash_bound_contract_exposes_the_boundary_conflict() -> None:
    contract = json.loads(
        (ROOT / "configs/stage2_action_contract.development.json").read_text()
    )
    gripper = contract["gripper_actor_q_input"]
    assert gripper["decode_owner"] == "forcesmolvla.action_delta.decode_binary_gripper_width"
    assert gripper["candidate_space"] == "frozen_delta_action7_normalized_inverse"
    assert [gripper["closed_width_m"], gripper["open_width_m"]] == [0.0, 0.085]
    assert gripper["stop_gradient_after_decode"] is True
    source = inspect.getsource(critic_action_for_q_guidance)
    assert "MODEL_GRIPPER_CANDIDATE_RANGE_M" in source
    assert "physical_candidate" in source
