"""Differentiable access to the native ten-step ForceSmolVLA Flow sampler."""

from __future__ import annotations

from typing import Literal

import torch

from forcesmolvla.action_delta import (
    BINARY_GRIPPER_CLOSED_WIDTH_M,
    BINARY_GRIPPER_OPEN_WIDTH_M,
    BINARY_GRIPPER_SWITCH_WIDTH_M,
    MODEL_GRIPPER_CANDIDATE_RANGE_M,
)
from forcesmolvla.force_token import PreparedForceContextBinding
from lerobot.policies.smolvla.modeling_smolvla import pad_vector
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


SamplingPurpose = Literal["actor_guidance", "td_next", "cql_current", "cql_next"]
_PURPOSES = {"actor_guidance", "td_next", "cql_current", "cql_next"}


def critic_action_for_q_guidance(
    action_chunk7: torch.Tensor,
    *,
    delta_action_mean7: torch.Tensor,
    delta_action_std7: torch.Tensor,
) -> torch.Tensor:
    """Build the fixed ``[B,3,7]`` v4 Q input with TCP-only Actor gradients."""

    if action_chunk7.ndim != 3 or action_chunk7.shape[1:] != (50, 7):
        raise ValueError("RFT Actor action chunk must have shape [B,50,7]")
    for name, value in (
        ("delta_action_mean7", delta_action_mean7),
        ("delta_action_std7", delta_action_std7),
    ):
        if (
            value.shape != (7,)
            or value.device != action_chunk7.device
            or not torch.all(torch.isfinite(value))
        ):
            raise ValueError(f"{name} must be finite [7] on the Actor device")
    if torch.any(delta_action_std7 <= 0):
        raise ValueError("delta_action_std7 must be positive")

    action_k7 = action_chunk7[:, :3]
    physical_candidate = (
        action_k7[..., 6] * delta_action_std7[6] + delta_action_mean7[6]
    )
    candidate_low, candidate_high = MODEL_GRIPPER_CANDIDATE_RANGE_M
    if torch.any(
        (physical_candidate < candidate_low) | (physical_candidate > candidate_high)
    ):
        raise ValueError(
            "model gripper candidate is outside the frozen [-0.01,0.095] m tolerance"
        )
    decoded_width = torch.where(
        physical_candidate < BINARY_GRIPPER_SWITCH_WIDTH_M,
        torch.as_tensor(
            BINARY_GRIPPER_CLOSED_WIDTH_M,
            device=action_chunk7.device,
            dtype=action_chunk7.dtype,
        ),
        torch.as_tensor(
            BINARY_GRIPPER_OPEN_WIDTH_M,
            device=action_chunk7.device,
            dtype=action_chunk7.dtype,
        ),
    )
    normalized_gripper = (
        (decoded_width - delta_action_mean7[6]) / delta_action_std7[6]
    ).detach()
    mixed_action = torch.cat(
        (action_k7[..., :6], normalized_gripper.unsqueeze(-1)), dim=-1
    )
    return mixed_action


def sample_normalized_action_chunk_with_grad(
    policy,
    batch: dict[str, torch.Tensor],
    noise7: torch.Tensor,
    *,
    call_id: str,
    purpose: SamplingPurpose,
) -> torch.Tensor:
    """Return graph-connected normalized action_target7 with shape ``[B,50,7]``."""

    if purpose not in _PURPOSES:
        raise ValueError(f"unsupported RFT sampling purpose: {purpose!r}")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("RFT Flow call_id must be a nonempty string")
    if policy.training:
        raise RuntimeError("RFT_FLOW_SAMPLING_REQUIRES_ACTOR_EVAL_MODE")
    if purpose == "actor_guidance" and (
        not torch.is_grad_enabled() or torch.is_inference_mode_enabled()
    ):
        raise RuntimeError("RFT_ACTOR_GUIDANCE_REQUIRES_AUTOGRAD")
    if (
        policy.config.chunk_size != 50
        or policy.config.num_steps != 10
        or policy.config.max_state_dim != 32
        or policy.config.max_action_dim != 32
    ):
        raise RuntimeError("RFT_FROZEN_FLOW_TOPOLOGY_DRIFT")

    state7 = batch.get("observation.state")
    wrench6 = batch.get("observation.wrench")
    if not isinstance(state7, torch.Tensor) or state7.ndim != 2 or state7.shape[1] != 7:
        raise ValueError("RFT observation.state must have shape [B,7]")
    batch_size = state7.shape[0]
    if (
        not isinstance(wrench6, torch.Tensor)
        or wrench6.shape != (batch_size, 6)
        or not torch.all(torch.isfinite(wrench6))
    ):
        raise ValueError("RFT observation.wrench must be finite with shape [B,6]")
    if (
        noise7.shape != (batch_size, 50, 7)
        or noise7.dtype != torch.float32
        or noise7.device != state7.device
        or not torch.all(torch.isfinite(noise7))
    ):
        raise ValueError("RFT noise7 must be finite float32 [B,50,7] on the Actor device")
    for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK):
        value = batch.get(key)
        if not isinstance(value, torch.Tensor) or value.shape[0] != batch_size:
            raise ValueError(f"RFT batch field {key!r} must be batch-aligned")

    identities = batch.get("sample_identity")
    if (
        not isinstance(identities, (list, tuple))
        or len(identities) != batch_size
        or any(not isinstance(value, str) or not value for value in identities)
    ):
        raise ValueError("RFT sample_identity must contain one nonempty string per row")

    images, image_masks = policy.prepare_images(batch)
    state32 = policy.prepare_state(batch)
    if state32.shape != (batch_size, 32):
        raise RuntimeError("RFT prepared state must have shape [B,32]")
    active_state = (torch.arange(32, device=state32.device) < 7).view(1, 32)
    state32 = state32 * active_state.to(dtype=state32.dtype)
    wrench = policy._prepare_wrench(batch, device=state32.device)

    suffix_valid = torch.ones(batch_size, 50, dtype=torch.bool, device=state32.device)
    feature_mask = suffix_valid.unsqueeze(-1) & (
        torch.arange(32, device=state32.device).view(1, 1, 32) < 7
    )
    noise32 = pad_vector(noise7, 32) * feature_mask.to(dtype=noise7.dtype)
    binding = PreparedForceContextBinding(
        chunk_id=tuple(f"rft:{purpose}:{call_id}:{row}" for row in range(batch_size)),
        sample_id=tuple(identities),
        context_generation=policy._context_generation,
        model_generation=policy.model.parameter_generation(),
        device=state32.device,
        dtype=torch.float32,
    )
    actions32 = policy.model.sample_actions_masked(
        images,
        image_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        state32,
        noise32,
        action_feature_mask=feature_mask,
        suffix_valid_mask=suffix_valid,
        wrench=wrench,
        force_context_binding=binding,
    )
    action7 = actions32[..., :7].float()
    if action7.shape != (batch_size, 50, 7) or not torch.all(torch.isfinite(action7)):
        raise RuntimeError("RFT_NORMALIZED_ACTION_OUTPUT_INVALID")
    return action7
