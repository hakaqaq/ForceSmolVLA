"""Append-only Stage-2 frozen-prefix Actor trainability contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import types
from typing import Any

import torch
from torch import Tensor, nn

from forcesmolvla.force_token import PreparedForceContextBinding
from forcesmolvla.prefix import PrefixContext
from forcesmolvla.rft.critic_action_adapter_v2 import critic_action_for_q_guidance_v2
from forcesmolvla.rft.losses import CriticObservation, critics_as_action_differentiators
from lerobot.policies.smolvla.modeling_smolvla import pad_vector
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


FROZEN_PREFIXES = (
    "model.vlm_with_expert.vlm.",
    "model.state_proj.",
)
TRAINABLE_PREFIXES = (
    "model.force_branch.",
    "model.force_adapter.",
    "model.vlm_with_expert.lm_expert.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.action_time_mlp_in.",
    "model.action_time_mlp_out.",
)


def _tensor_digest(items) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(items):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_state_digest(policy: nn.Module) -> dict[str, Any]:
    parameters = [
        (name, value) for name, value in policy.named_parameters()
        if name.startswith(FROZEN_PREFIXES)
    ]
    buffers = [
        (name, value) for name, value in policy.named_buffers()
        if name.startswith(FROZEN_PREFIXES)
    ]
    return {
        "parameters_sha256": _tensor_digest(parameters),
        "buffers_sha256": _tensor_digest(buffers),
        "parameter_tensor_count": len(parameters),
        "buffer_tensor_count": len(buffers),
    }


def _force_frozen_modules_eval(policy: nn.Module) -> None:
    policy.model.vlm_with_expert.vlm.eval()
    policy.model.state_proj.eval()


@dataclass(frozen=True)
class TrainabilityManifest:
    frozen_parameter_count: int
    trainable_actor_parameter_count: int
    frozen_parameter_tensors: int
    trainable_actor_parameter_tensors: int
    frozen_names: tuple[str, ...]
    trainable_names: tuple[str, ...]


def apply_frozen_vlm_trainability(policy: nn.Module) -> TrainabilityManifest:
    """Freeze the prefix owner and keep it in eval under later Actor mode changes."""

    named = dict(policy.named_parameters())
    if not named:
        raise RuntimeError("STAGE2_TRAINABILITY_EMPTY_ACTOR")
    frozen_names = tuple(sorted(name for name in named if name.startswith(FROZEN_PREFIXES)))
    if not frozen_names or not any("state_proj" in name for name in frozen_names):
        raise RuntimeError("STAGE2_TRAINABILITY_FROZEN_PREFIX_NOT_RESOLVED")
    for name, parameter in named.items():
        parameter.requires_grad_(name not in frozen_names)
    unexpected = tuple(
        sorted(
            name for name, parameter in named.items()
            if parameter.requires_grad and not name.startswith(TRAINABLE_PREFIXES)
        )
    )
    if unexpected:
        raise RuntimeError(f"STAGE2_TRAINABILITY_UNCLASSIFIED_TRAINABLE:{unexpected}")

    if not getattr(policy, "_stage2_frozen_vlm_train_patch", False):
        original_train = policy.train

        def train_with_frozen_prefix(self, mode: bool = True):
            result = original_train(mode)
            _force_frozen_modules_eval(self)
            return result

        policy.train = types.MethodType(train_with_frozen_prefix, policy)
        policy._stage2_frozen_vlm_train_patch = True
    policy.train(policy.training)
    if policy.model.vlm_with_expert.vlm.training or policy.model.state_proj.training:
        raise RuntimeError("STAGE2_TRAINABILITY_FROZEN_MODULE_NOT_EVAL")
    if any(named[name].requires_grad for name in frozen_names):
        raise RuntimeError("STAGE2_TRAINABILITY_FROZEN_REQUIRES_GRAD")
    trainable_names = tuple(sorted(name for name, value in named.items() if value.requires_grad))
    return TrainabilityManifest(
        frozen_parameter_count=sum(named[name].numel() for name in frozen_names),
        trainable_actor_parameter_count=sum(named[name].numel() for name in trainable_names),
        frozen_parameter_tensors=len(frozen_names),
        trainable_actor_parameter_tensors=len(trainable_names),
        frozen_names=frozen_names,
        trainable_names=trainable_names,
    )


def build_frozen_vlm_actor_optimizer(policy: nn.Module, *, lr: float = 1e-5):
    """Reuse Stage-1 AdamW grouping over only the approved Stage-2 parameters."""

    from forcesmolvla.router_training import _no_decay_parameter_names

    named = {name: value for name, value in policy.named_parameters() if value.requires_grad}
    no_decay = set(_no_decay_parameter_names(policy)) & set(named)
    decay = set(named) - no_decay
    optimizer = torch.optim.AdamW(
        [
            {"params": [named[name] for name in sorted(decay)], "weight_decay": 1e-10},
            {"params": [named[name] for name in sorted(no_decay)], "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    owned = [id(value) for group in optimizer.param_groups for value in group["params"]]
    expected = {id(value) for value in named.values()}
    if len(owned) != len(set(owned)) or set(owned) != expected:
        raise RuntimeError("STAGE2_TRAINABILITY_ACTOR_OPTIMIZER_OWNERSHIP")
    return optimizer, scheduler, {
        "type": "AdamW",
        "parameter_count": sum(value.numel() for value in named.values()),
        "parameter_tensor_count": len(named),
        "decay_tensor_count": len(decay),
        "no_decay_tensor_count": len(no_decay),
        "frozen_parameter_in_optimizer": 0,
    }


def _prefix_context_is_detached(context: PrefixContext) -> bool:
    tensors = [
        context.prefix_out,
        context.prefix_valid_mask,
        context.prefix_segment_ids,
        context.prefix_position_ids,
    ] + [
        value for layer in context.past_key_values.values() for value in layer.values()
    ]
    return all(not value.requires_grad for value in tensors)


def frozen_prefix_flow_matching_terms(
    policy,
    batch: dict,
    *,
    noise: Tensor,
    time: Tensor,
    call_id: str,
) -> tuple[Tensor, Tensor, Any, dict[str, Any]]:
    """FM terms with one no-grad prefix prefill and one trainable Force K/V projection."""

    policy._validate_visual_batch(batch)
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    wrench = policy._prepare_wrench(batch, device=state.device)
    state_mask = (torch.arange(32, device=state.device) < 7).view(1, 32)
    state = state * state_mask.to(dtype=state.dtype)
    actions = pad_vector(batch[ACTION], 32)
    feature_mask, suffix_valid = policy._action_masks(
        batch, horizon=actions.shape[1], device=actions.device
    )
    if noise.shape[-1] == 7:
        noise = pad_vector(noise, 32)
    if tuple(noise.shape) != tuple(actions.shape):
        raise ValueError("STAGE2_FROZEN_PREFIX_FM_NOISE_SHAPE")
    mask = feature_mask.to(device=actions.device, dtype=actions.dtype)
    actions = actions * mask
    noise = noise * mask
    x_t = (time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions) * mask
    target_velocity = (noise - actions) * mask

    with torch.no_grad():
        context = policy.model.encode_prefix(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            audit_cache=False,
        )
    if not _prefix_context_is_detached(context):
        raise RuntimeError("STAGE2_FROZEN_PREFIX_CONTEXT_NOT_DETACHED")
    force_context = policy.model.build_force_context(
        context.prefix_out.detach(), context.prefix_valid_mask, wrench
    )
    if force_context is None or force_context.router_state is None:
        raise RuntimeError("STAGE2_FROZEN_PREFIX_FORCE_CONTEXT_MISSING")
    identities = batch.get("sample_identity")
    if not isinstance(identities, (tuple, list)) or len(identities) != actions.shape[0]:
        raise ValueError("STAGE2_FROZEN_PREFIX_SAMPLE_IDENTITY")
    binding = PreparedForceContextBinding(
        chunk_id=tuple(f"stage2-frozen-prefix:{call_id}:{index}" for index in range(actions.shape[0])),
        sample_id=tuple(str(value) for value in identities),
        context_generation=policy._context_generation,
        model_generation=policy.model.parameter_generation(),
        device=actions.device,
        dtype=torch.float32,
    )
    prepared = policy.model.force_adapter.cross_attention.prepare(
        force_context, binding=binding
    )
    velocity = policy.model.velocity_cached(
        context,
        x_t,
        time,
        action_feature_mask=feature_mask,
        suffix_valid_mask=suffix_valid,
        force_context=prepared,
        force_context_binding=binding,
    )
    losses = torch.nn.functional.mse_loss(target_velocity, velocity, reduction="none") * mask
    return losses, feature_mask, force_context.router_state, {
        "prefix_prefill_count": 1,
        "prefix_grad_enabled": False,
        "prefix_representation_detached": True,
        "prefix_cache_detached": True,
        "force_kv_projection_count": 1,
    }


def compute_min_twin_q_actor_loss(
    *,
    q1: nn.Module,
    q2: nn.Module,
    observation: CriticObservation,
    normalized_flow_action_chunk7: Tensor,
    delta_action_mean7: Tensor,
    delta_action_std7: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``-mean(min(Q1,Q2))`` with TCP-only action gradients."""

    action = critic_action_for_q_guidance_v2(
        normalized_flow_action_chunk7,
        delta_action_mean7=delta_action_mean7,
        delta_action_std7=delta_action_std7,
    )
    mask = torch.ones(action.shape[0], 3, dtype=torch.bool, device=action.device)
    with critics_as_action_differentiators(q1, q2):
        q1_value = q1(*observation.as_tuple(), action.float(), mask)
        q2_value = q2(*observation.as_tuple(), action.float(), mask)
        loss = -torch.minimum(q1_value.float(), q2_value.float()).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("STAGE2_FROZEN_PREFIX_ACTOR_Q_NONFINITE")
    return loss, q1_value, q2_value, action


def gradient_norm_for_prefixes(policy: nn.Module, prefixes: tuple[str, ...]) -> float:
    values = [
        value.grad.detach().float().square().sum()
        for name, value in policy.named_parameters()
        if name.startswith(prefixes) and value.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).cpu())
