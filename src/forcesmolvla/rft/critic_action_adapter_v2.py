"""ActionContract-v2 projection for Stage-2 Critic-only action views."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from forcesmolvla.action_delta import (
    BINARY_GRIPPER_CLOSED_WIDTH_M,
    BINARY_GRIPPER_OPEN_WIDTH_M,
    BINARY_GRIPPER_SWITCH_WIDTH_M,
    MODEL_GRIPPER_CANDIDATE_RANGE_M,
)
from forcesmolvla.rft.online.action_runtime import (
    H50_MODEL_TIMEBASE_HZ,
    POSE_REFERENCE_DISPATCH_HZ,
    rational_h50_index,
)


ACTION_SLOTS = 3
ACTION_DIM = 7
FLOW_HORIZON = 50
NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class CriticActionContract:
    """Single source of truth for every behavior and candidate Q action."""

    version: str = "critic-action-contract-v3-command-effective-r30-k3"
    model_grid_hz: int = H50_MODEL_TIMEBASE_HZ
    execution_hz: int = POSE_REFERENCE_DISPATCH_HZ
    critic_slots: int = ACTION_SLOTS
    macro_duration_ns: int = 100_000_000
    action_dim: int = ACTION_DIM
    tcp_dim: int = 6
    flow_horizon: int = FLOW_HORIZON
    gamma: float = 0.99
    max_ack_age_ms: float = 110.0
    action_sources: tuple[str, str, str] = (
        "policy", "human", "offline_demonstration",
    )
    behavior_authority: str = "identifier-matched-controller-accepted-ack"
    temporal_projection: str = "strict-rational-30hz-causal-latest-ack-zoh"
    candidate_projection: str = "command-effective-anchor-zoh"
    partial_macro: str = "masked-prefix-deterministic-zero-padding"

    def validate(self) -> "CriticActionContract":
        if (
            not self.version
            or self.model_grid_hz != 30
            or self.execution_hz != 10
            or self.critic_slots != 3
            or self.macro_duration_ns != 100_000_000
            or self.action_dim != 7
            or self.tcp_dim != 6
            or self.flow_horizon != 50
            or self.model_grid_hz // self.execution_hz != self.critic_slots
            or self.model_grid_hz % self.execution_hz
            or not 0.0 < self.gamma <= 1.0
            or self.max_ack_age_ms < 100.0
        ):
            raise ValueError("CRITIC_ACTION_CONTRACT_INVALID")
        return self


CRITIC_ACTION_CONTRACT = CriticActionContract().validate()


def _rational_tick_ns(index: int, *, hz: int) -> int:
    if index < 0 or hz <= 0:
        raise ValueError("CRITIC_ACTION_RATIONAL_TICK_INVALID")
    return (int(index) * NANOSECONDS_PER_SECOND + hz // 2) // hz


def build_critic_transition_grid(
    anchor_timestamp_ns: int,
    *,
    contract: CriticActionContract = CRITIC_ACTION_CONTRACT,
) -> tuple[tuple[int, int, int], int]:
    """Return the three strict rational ticks and the fourth/next tick."""

    contract.validate()
    if int(anchor_timestamp_ns) <= 0:
        raise ValueError("CRITIC_ACTION_ANCHOR_TIMESTAMP_INVALID")
    anchor = int(anchor_timestamp_ns)
    ticks = tuple(
        anchor + _rational_tick_ns(offset, hz=contract.model_grid_hz)
        for offset in range(contract.critic_slots)
    )
    next_tick = anchor + _rational_tick_ns(
        contract.critic_slots, hz=contract.model_grid_hz
    )
    if next_tick - ticks[0] != contract.macro_duration_ns:
        raise AssertionError("CRITIC_ACTION_MACRO_DURATION_DRIFT")
    return ticks, next_tick


def command_effective_execution_index_map(
    *,
    contract: CriticActionContract = CRITIC_ACTION_CONTRACT,
    anchor_timestamp_ns: int = NANOSECONDS_PER_SECOND,
) -> tuple[int, int, int]:
    """Derive the held H50 index from the rational 10 Hz dispatcher clock."""

    ticks, _ = build_critic_transition_grid(
        anchor_timestamp_ns, contract=contract
    )
    execution_period_slots = contract.model_grid_hz // contract.execution_hz
    dispatch_anchor_index = 0
    result = []
    for grid_index, _tick in enumerate(ticks):
        effective_dispatch_index = grid_index // execution_period_slots
        dispatch_timestamp_ns = anchor_timestamp_ns + _rational_tick_ns(
            effective_dispatch_index * execution_period_slots,
            hz=contract.model_grid_hz,
        )
        chunk_anchor_ns = anchor_timestamp_ns + _rational_tick_ns(
            dispatch_anchor_index * execution_period_slots,
            hz=contract.model_grid_hz,
        )
        result.append(rational_h50_index(dispatch_timestamp_ns, chunk_anchor_ns))
    if tuple(result) != (0, 0, 0):
        raise ValueError("CRITIC_ACTION_COMMAND_EFFECTIVE_PHASE_INVALID")
    return tuple(result)


def aligned_fresh_chunk_execution_index_map_v2() -> tuple[int, int, int]:
    """Explicit 10 Hz ZOH map for a fresh decision on the 30 Hz Critic grid."""

    return command_effective_execution_index_map()


def _normalizer(mean7: Tensor, std7: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
    if tuple(mean7.shape) != (ACTION_DIM,) or tuple(std7.shape) != (ACTION_DIM,):
        raise ValueError("ACTION_CONTRACT_V2_NORMALIZER_SHAPE")
    if mean7.device != reference.device or std7.device != reference.device:
        raise ValueError("ACTION_CONTRACT_V2_NORMALIZER_DEVICE")
    if not torch.all(torch.isfinite(mean7)) or not torch.all(torch.isfinite(std7)):
        raise ValueError("ACTION_CONTRACT_V2_NORMALIZER_NONFINITE")
    if torch.any(std7 <= 0):
        raise ValueError("ACTION_CONTRACT_V2_NORMALIZER_STD_NONPOSITIVE")
    return mean7.float(), std7.float()


def project_binary_gripper_width_v2(continuous_width_m: Tensor) -> Tensor:
    """Total deterministic binary projection for every finite internal sample."""
    value = continuous_width_m.float()
    if not torch.all(torch.isfinite(value)):
        raise ValueError("ACTION_CONTRACT_V2_GRIPPER_NONFINITE")
    closed = torch.as_tensor(
        BINARY_GRIPPER_CLOSED_WIDTH_M, device=value.device, dtype=torch.float32
    )
    opened = torch.as_tensor(
        BINARY_GRIPPER_OPEN_WIDTH_M, device=value.device, dtype=torch.float32
    )
    return torch.where(value < BINARY_GRIPPER_SWITCH_WIDTH_M, closed, opened)


def normalized_gripper_endpoints_v2(mean7: Tensor, std7: Tensor) -> Tensor:
    reference = mean7
    mean, std = _normalizer(mean7, std7, reference)
    endpoints = torch.tensor(
        [BINARY_GRIPPER_CLOSED_WIDTH_M, BINARY_GRIPPER_OPEN_WIDTH_M],
        device=mean.device, dtype=torch.float32,
    )
    return (endpoints - mean[6]) / std[6]


def critic_action_for_q_guidance_v2(
    normalized_flow_action_chunk7: Tensor,
    *,
    execution_index_map: Tensor | tuple[int, int, int],
    delta_action_mean7: Tensor,
    delta_action_std7: Tensor,
) -> Tensor:
    """Project H50 through the effective dispatcher/ZOH control contract."""
    action = normalized_flow_action_chunk7
    if action.ndim != 3 or tuple(action.shape[1:]) != (FLOW_HORIZON, ACTION_DIM):
        raise ValueError("ACTION_CONTRACT_V2_FLOW_ACTION_SHAPE")
    if not torch.all(torch.isfinite(action)):
        raise ValueError("ACTION_CONTRACT_V2_FLOW_ACTION_NONFINITE")
    mean, std = _normalizer(delta_action_mean7, delta_action_std7, action)
    indices = torch.as_tensor(
        execution_index_map, dtype=torch.long, device=action.device
    )
    if indices.ndim == 1:
        indices = indices.unsqueeze(0).expand(action.shape[0], -1)
    if tuple(indices.shape) != (action.shape[0], ACTION_SLOTS):
        raise ValueError("ACTION_CONTRACT_V2_EXECUTION_INDEX_MAP_SHAPE")
    if torch.any(indices < 0) or torch.any(indices >= FLOW_HORIZON):
        raise ValueError("ACTION_CONTRACT_V2_EXECUTION_INDEX_MAP_RANGE")
    action_k = torch.gather(
        action, 1, indices.unsqueeze(-1).expand(-1, -1, ACTION_DIM)
    )
    with torch.no_grad():
        continuous_width_m = action_k[..., 6].float() * std[6] + mean[6]
        endpoint_m = project_binary_gripper_width_v2(continuous_width_m)
        normalized_endpoint = (endpoint_m - mean[6]) / std[6]
    return torch.cat(
        (action_k[..., :6].float(), normalized_endpoint.unsqueeze(-1)), dim=-1
    )


def command_effective_candidate_action(
    normalized_flow_action_chunk7: Tensor,
    *,
    contract: CriticActionContract = CRITIC_ACTION_CONTRACT,
    anchor_timestamp_ns: int = NANOSECONDS_PER_SECOND,
    delta_action_mean7: Tensor,
    delta_action_std7: Tensor,
) -> Tensor:
    """Differentiable candidate action in the canonical command-effective space."""

    return critic_action_for_q_guidance_v2(
        normalized_flow_action_chunk7,
        execution_index_map=command_effective_execution_index_map(
            contract=contract, anchor_timestamp_ns=anchor_timestamp_ns
        ),
        delta_action_mean7=delta_action_mean7,
        delta_action_std7=delta_action_std7,
    )


# Identity, not a wrapper: TD bootstrap and Actor guidance cannot drift apart.
bootstrap_command_effective_candidate_action = command_effective_candidate_action


def raw_gripper_out_of_public_tolerance_mask(
    normalized_flow_gripper: Tensor,
    *,
    gripper_mean: Tensor,
    gripper_std: Tensor,
) -> Tensor:
    """Detached distribution diagnostic; never an internal validity gate."""
    value = normalized_flow_gripper.float()
    if not torch.all(torch.isfinite(value)):
        raise ValueError("ACTION_CONTRACT_V2_GRIPPER_NONFINITE")
    continuous = value * gripper_std.float() + gripper_mean.float()
    low, high = MODEL_GRIPPER_CANDIDATE_RANGE_M
    return ((continuous < low) | (continuous > high)).detach()
