"""Pure online-replay CPU losses: Twin-Q TD and expert-only Actor terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OnlineTwinQTDLoss:
    total: Tensor
    q1_loss: Tensor
    q2_loss: Tensor
    q1_value: Tensor
    q2_value: Tensor
    target: Tensor
    next_actor_calls: int
    target_q1_calls: int
    target_q2_calls: int
    calql_candidate_calls: int = 0
    random_candidate_calls: int = 0
    mc_return_reads: int = 0


@dataclass(frozen=True)
class OnlineActorLoss:
    total: Tensor
    expert_flow_matching: Tensor
    actor_q: Tensor
    balance: Tensor
    z: Tensor
    expert_feature_count: int
    actor_q_valid_count: int


def _finite_fp32(value: Tensor, name: str, shape: tuple[int, ...] | None = None) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"ONLINE_REPLAY_{name}_MUST_BE_FLOATING_TENSOR")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"ONLINE_REPLAY_{name}_SHAPE_INVALID")
    value = value.float()
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"ONLINE_REPLAY_{name}_NONFINITE")
    return value


def _slice_observation(observation: Any, mask: Tensor) -> Any:
    if isinstance(observation, Tensor):
        return observation[mask]
    if hasattr(observation, "index") and callable(observation.index):
        return observation.index(mask)
    if isinstance(observation, tuple):
        return tuple(_slice_observation(value, mask) for value in observation)
    if isinstance(observation, list):
        return [_slice_observation(value, mask) for value in observation]
    if isinstance(observation, dict):
        batch = int(mask.numel())
        return {
            key: value[mask] if isinstance(value, Tensor) and value.ndim and value.shape[0] == batch else value
            for key, value in observation.items()
        }
    raise TypeError(f"ONLINE_REPLAY_OBSERVATION_TYPE_UNSUPPORTED:{type(observation).__name__}")


def _call_critic(critic: nn.Module, observation: Any, action: Tensor, mask: Tensor) -> Tensor:
    if hasattr(observation, "as_tuple") and callable(observation.as_tuple):
        return critic(*observation.as_tuple(), action, mask)
    if isinstance(observation, (tuple, list)):
        return critic(*observation, action, mask)
    return critic(observation, action, mask)


def compute_online_twin_q_td_loss(
    *,
    q1: nn.Module,
    q2: nn.Module,
    q1_target: nn.Module,
    q2_target: nn.Module,
    observation: Any,
    next_observation: Any,
    ack_behavior_action_k7: Tensor,
    behavior_mask: Tensor,
    reward: Tensor,
    discount: Tensor,
    terminated: Tensor,
    bootstrap_mask: Tensor,
    next_policy_action_fn: Callable[[Any], Tensor],
    gamma_decision: float = 0.99,
) -> OnlineTwinQTDLoss:
    """Compute pure online TD; no Cal-QL or MC-return input exists in this API."""

    batch = int(reward.numel())
    reward = _finite_fp32(reward, "REWARD", (batch,))
    discount = _finite_fp32(discount, "DISCOUNT", (batch,))
    action = _finite_fp32(ack_behavior_action_k7, "ACK_ACTION", (batch, 3, 7))
    if terminated.dtype != torch.bool or tuple(terminated.shape) != (batch,):
        raise ValueError("ONLINE_REPLAY_TERMINATED_MUST_BE_BOOL_VECTOR")
    if bootstrap_mask.dtype != torch.bool or tuple(bootstrap_mask.shape) != (batch,):
        raise ValueError("ONLINE_REPLAY_BOOTSTRAP_MASK_MUST_BE_BOOL_VECTOR")
    if behavior_mask.dtype != torch.bool or tuple(behavior_mask.shape) != (batch, 3):
        raise ValueError("ONLINE_REPLAY_BEHAVIOR_MASK_MUST_BE_BOOL_BK")
    if not bool(behavior_mask.any(dim=1).all()):
        raise ValueError("ONLINE_REPLAY_ONLINE_TD_REQUIRES_EXECUTED_ACTION")
    if torch.any(terminated & bootstrap_mask) or not torch.equal(terminated, ~bootstrap_mask):
        raise ValueError("ONLINE_REPLAY_TERMINAL_BOOTSTRAP_CONTRACT")
    expected_discount = bootstrap_mask.float() * float(gamma_decision)
    if not torch.equal(discount, expected_discount):
        raise ValueError("ONLINE_REPLAY_DISCOUNT_ALREADY_ENCODES_BOOTSTRAP")

    nonterminal = ~terminated
    count = int(nonterminal.sum())
    target = reward.clone()
    next_actor_calls = target_q1_calls = target_q2_calls = 0
    if count:
        next_subset = _slice_observation(next_observation, nonterminal)
        with torch.no_grad():
            next_action = next_policy_action_fn(next_subset)
            next_actor_calls = 1
            next_action = _finite_fp32(next_action, "NEXT_POLICY_ACTION", (count, 3, 7)).detach()
            policy_mask = torch.ones(count, 3, dtype=torch.bool, device=next_action.device)
            next_q1 = _finite_fp32(
                _call_critic(q1_target, next_subset, next_action, policy_mask),
                "NEXT_TARGET_Q1", (count,),
            )
            target_q1_calls = 1
            next_q2 = _finite_fp32(
                _call_critic(q2_target, next_subset, next_action, policy_mask),
                "NEXT_TARGET_Q2", (count,),
            )
            target_q2_calls = 1
            target[nonterminal] = (
                reward[nonterminal]
                + discount[nonterminal] * torch.minimum(next_q1, next_q2)
            )
    target = _finite_fp32(target, "TD_TARGET", (batch,)).detach()
    q1_value = _finite_fp32(
        _call_critic(q1, observation, action, behavior_mask), "ONLINE_Q1", (batch,),
    )
    q2_value = _finite_fp32(
        _call_critic(q2, observation, action, behavior_mask), "ONLINE_Q2", (batch,),
    )
    q1_loss = torch.nn.functional.mse_loss(q1_value, target)
    q2_loss = torch.nn.functional.mse_loss(q2_value, target)
    total = (q1_loss + q2_loss) * 0.5
    _finite_fp32(total.reshape(1), "TWIN_Q_TD_LOSS", (1,))
    return OnlineTwinQTDLoss(
        total=total,
        q1_loss=q1_loss,
        q2_loss=q2_loss,
        q1_value=q1_value,
        q2_value=q2_value,
        target=target,
        next_actor_calls=next_actor_calls,
        target_q1_calls=target_q1_calls,
        target_q2_calls=target_q2_calls,
    )


def compute_expert_only_flow_matching_loss(
    per_feature_flow_loss: Tensor,
    action_valid_mask_h50: Tensor,
    expert_feature_mask_h50x7: Tensor,
) -> tuple[Tensor, int]:
    loss = _finite_fp32(per_feature_flow_loss, "FLOW_FEATURE_LOSS")
    if loss.ndim != 3 or loss.shape[-1] != 7:
        raise ValueError("ONLINE_REPLAY_FLOW_FEATURE_LOSS_MUST_BE_BH7")
    batch, horizon, _ = loss.shape
    if action_valid_mask_h50.dtype != torch.bool or tuple(action_valid_mask_h50.shape) != (batch, horizon):
        raise ValueError("ONLINE_REPLAY_ACTION_VALID_MASK_SHAPE")
    if expert_feature_mask_h50x7.dtype != torch.bool or tuple(expert_feature_mask_h50x7.shape) != tuple(loss.shape):
        raise ValueError("ONLINE_REPLAY_EXPERT_FEATURE_MASK_SHAPE")
    mask = expert_feature_mask_h50x7 & action_valid_mask_h50.unsqueeze(-1)
    count = int(mask.sum())
    if count == 0:
        return loss.sum() * 0.0, 0
    return (loss * mask.to(loss.dtype)).sum() / count, count


def compute_min_twin_q_guidance_from_values(
    q1_value: Tensor,
    q2_value: Tensor,
    actor_q_valid: Tensor,
) -> tuple[Tensor, int]:
    if tuple(q1_value.shape) != tuple(q2_value.shape) or q1_value.ndim != 1:
        raise ValueError("ONLINE_REPLAY_ACTOR_Q_VALUE_SHAPE")
    batch = q1_value.shape[0]
    q1_value = _finite_fp32(q1_value, "ACTOR_Q1", (batch,))
    q2_value = _finite_fp32(q2_value, "ACTOR_Q2", (batch,))
    if actor_q_valid.dtype != torch.bool or tuple(actor_q_valid.shape) != (batch,):
        raise ValueError("ONLINE_REPLAY_ACTOR_Q_VALID_MASK")
    count = int(actor_q_valid.sum())
    minimum = torch.minimum(q1_value, q2_value)
    if count == 0:
        return minimum.sum() * 0.0, 0
    return -minimum[actor_q_valid].mean(), count


def compute_online_actor_objective(
    *,
    per_feature_flow_loss: Tensor,
    action_valid_mask_h50: Tensor,
    expert_feature_mask_h50x7: Tensor,
    q1_actor_value: Tensor,
    q2_actor_value: Tensor,
    actor_q_valid: Tensor,
    balance_loss: Tensor,
    z_loss: Tensor,
    beta: float,
    eta: float,
    balance_weight: float = 0.01,
    z_weight: float = 0.001,
) -> OnlineActorLoss:
    fm, expert_count = compute_expert_only_flow_matching_loss(
        per_feature_flow_loss, action_valid_mask_h50, expert_feature_mask_h50x7,
    )
    actor_q, actor_q_count = compute_min_twin_q_guidance_from_values(
        q1_actor_value, q2_actor_value, actor_q_valid,
    )
    balance = _finite_fp32(balance_loss.reshape(1), "BALANCE_LOSS", (1,))[0]
    z = _finite_fp32(z_loss.reshape(1), "Z_LOSS", (1,))[0]
    total = float(beta) * fm + float(eta) * actor_q + balance_weight * balance + z_weight * z
    _finite_fp32(total.reshape(1), "ACTOR_TOTAL", (1,))
    return OnlineActorLoss(total, fm, actor_q, balance, z, expert_count, actor_q_count)


def compute_online_min_twin_q_actor_loss(**kwargs):
    """Reuse the accepted ActionContract-v2 path for TCP-only Q gradients."""

    from forcesmolvla.rft.frozen_vlm_trainability import compute_min_twin_q_actor_loss

    return compute_min_twin_q_actor_loss(**kwargs)
