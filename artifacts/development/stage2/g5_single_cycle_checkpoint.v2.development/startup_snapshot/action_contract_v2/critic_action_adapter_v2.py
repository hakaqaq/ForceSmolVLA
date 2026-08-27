"""ActionContract-v2 projection for Stage-2 Critic-only action views."""

from __future__ import annotations

import torch
from torch import Tensor

from forcesmolvla.action_delta import (
    BINARY_GRIPPER_CLOSED_WIDTH_M,
    BINARY_GRIPPER_OPEN_WIDTH_M,
    BINARY_GRIPPER_SWITCH_WIDTH_M,
    MODEL_GRIPPER_CANDIDATE_RANGE_M,
)


ACTION_SLOTS = 3
ACTION_DIM = 7
FLOW_HORIZON = 50


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
    delta_action_mean7: Tensor,
    delta_action_std7: Tensor,
) -> Tensor:
    """Return [B,3,7]: differentiable TCP6 plus detached binary gripper."""
    action = normalized_flow_action_chunk7
    if action.ndim != 3 or tuple(action.shape[1:]) != (FLOW_HORIZON, ACTION_DIM):
        raise ValueError("ACTION_CONTRACT_V2_FLOW_ACTION_SHAPE")
    if not torch.all(torch.isfinite(action)):
        raise ValueError("ACTION_CONTRACT_V2_FLOW_ACTION_NONFINITE")
    mean, std = _normalizer(delta_action_mean7, delta_action_std7, action)
    action_k = action[:, :ACTION_SLOTS]
    with torch.no_grad():
        continuous_width_m = action_k[..., 6].float() * std[6] + mean[6]
        endpoint_m = project_binary_gripper_width_v2(continuous_width_m)
        normalized_endpoint = (endpoint_m - mean[6]) / std[6]
    return torch.cat(
        (action_k[..., :6].float(), normalized_endpoint.unsqueeze(-1)), dim=-1
    )


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
