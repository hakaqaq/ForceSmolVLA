"""Losses for the image-free residual Actor and ACK-aligned Twin-Q."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ResidualActorLoss:
    total: Tensor
    value: Tensor
    residual: Tensor
    human: Tensor
    actor_q_valid_count: int
    human_residual_valid_count: int


def residual_critic_loss(
    q1: nn.Module,
    q2: nn.Module,
    q1_target: nn.Module,
    q2_target: nn.Module,
    residual_actor_target: nn.Module,
    batch: Any,
    gamma: float,
) -> Tensor:
    """Pure ACK TD loss; neither cameras nor the frozen base Actor are inputs."""

    with torch.no_grad():
        bootstrap = (~batch.terminated & ~batch.truncated)
        if torch.any(bootstrap & ~batch.next_base_valid):
            raise ValueError("FORCERFT_TD_NEXT_BASE_MISSING")
        next_residual6 = residual_actor_target(
            normalized_state7=batch.next_state7,
            normalized_wrench6=batch.next_wrench6,
            normalized_wrench_delta6=batch.next_wrench_delta6,
            base_action6=batch.next_base_action_k6[:, 0],
        )
        next_residual_k6 = next_residual6[:, None, :].expand(-1, 3, -1)
        next_q = torch.minimum(
            q1_target(
                batch.next_state7,
                batch.next_wrench6,
                batch.next_wrench_delta6,
                batch.next_base_action_k6,
                next_residual_k6,
                batch.next_action_mask,
            ),
            q2_target(
                batch.next_state7,
                batch.next_wrench6,
                batch.next_wrench_delta6,
                batch.next_base_action_k6,
                next_residual_k6,
                batch.next_action_mask,
            ),
        )
        target = batch.reward.float() + float(gamma) * bootstrap.float() * next_q

    q1_value = q1(
        batch.state7,
        batch.wrench6,
        batch.wrench_delta6,
        batch.base_action_k6,
        batch.behavior_residual_k6,
        batch.action_mask,
    )
    q2_value = q2(
        batch.state7,
        batch.wrench6,
        batch.wrench_delta6,
        batch.base_action_k6,
        batch.behavior_residual_k6,
        batch.action_mask,
    )
    loss = 0.5 * (
        F.mse_loss(q1_value, target) + F.mse_loss(q2_value, target)
    )
    if loss.ndim or not torch.isfinite(loss):
        raise FloatingPointError("FORCERFT_CRITIC_LOSS_NONFINITE")
    return loss


def residual_actor_loss(
    q1: nn.Module,
    q2: nn.Module,
    residual_actor: nn.Module,
    policy_batch: Any,
    human_batch: Any | None,
    *,
    actor_q_weight: float,
    residual_l2_weight: float,
    human_residual_weight: float,
) -> ResidualActorLoss:
    """Min-Q value term plus small residual and valid-human supervision terms."""

    zero = next(residual_actor.parameters()).sum() * 0.0
    candidate_residual6 = None
    valid_count = 0
    value = residual = zero
    if policy_batch is not None:
        candidate_residual6 = residual_actor(
            normalized_state7=policy_batch.state7,
            normalized_wrench6=policy_batch.wrench6,
            normalized_wrench_delta6=policy_batch.wrench_delta6,
            base_action6=policy_batch.base_action_k6[:, 0],
        )
        valid = policy_batch.actor_q_valid
        if valid.dtype != torch.bool or valid.shape != (candidate_residual6.shape[0],):
            raise ValueError("FORCERFT_ACTOR_Q_VALID_MASK_INVALID")
        valid_count = int(valid.sum())
        if valid_count:
            candidate_k6 = candidate_residual6[valid, None, :].expand(-1, 3, -1)
            value = -torch.minimum(
                q1(
                    policy_batch.state7[valid],
                    policy_batch.wrench6[valid],
                    policy_batch.wrench_delta6[valid],
                    policy_batch.base_action_k6[valid],
                    candidate_k6,
                    policy_batch.action_mask[valid],
                ),
                q2(
                    policy_batch.state7[valid],
                    policy_batch.wrench6[valid],
                    policy_batch.wrench_delta6[valid],
                    policy_batch.base_action_k6[valid],
                    candidate_k6,
                    policy_batch.action_mask[valid],
                ),
            ).mean()
        residual = candidate_residual6.square().mean()

    human_count = 0
    human = zero
    if human_batch is not None:
        human_valid = human_batch.human_residual_valid
        if human_valid.dtype != torch.bool:
            raise ValueError("FORCERFT_HUMAN_RESIDUAL_VALID_MASK_INVALID")
        human_count = int(human_valid.sum())
        if human_count:
            human_prediction = residual_actor(
                normalized_state7=human_batch.state7[human_valid],
                normalized_wrench6=human_batch.wrench6[human_valid],
                normalized_wrench_delta6=human_batch.wrench_delta6[human_valid],
                base_action6=human_batch.base_action_k6[human_valid, 0],
            )
            human = F.mse_loss(
                human_prediction,
                human_batch.human_residual_target6[human_valid].detach(),
            )
            if candidate_residual6 is None:
                residual = human_prediction.square().mean()
    total = (
        float(actor_q_weight) * value
        + float(residual_l2_weight) * residual
        + float(human_residual_weight) * human
    )
    if total.ndim or not torch.isfinite(total):
        raise FloatingPointError("FORCERFT_ACTOR_LOSS_NONFINITE")
    return ResidualActorLoss(
        total,
        value,
        residual,
        human,
        valid_count,
        human_count,
    )
