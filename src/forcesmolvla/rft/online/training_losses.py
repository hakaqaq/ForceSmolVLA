"""Losses for the image-free residual Actor and ACK-aligned Twin-Q."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from forcesmolvla.rft.online.controller_acceptance import (
    policy_candidate_guard_valid,
)


@dataclass(frozen=True)
class ResidualCriticLoss:
    total: Tensor
    td_valid_count: int
    target_candidate_guard_rejected_count: int
    target_candidate_guard_unknown_count: int


@dataclass(frozen=True)
class ResidualActorLoss:
    total: Tensor
    value: Tensor
    residual: Tensor
    human: Tensor
    output_norm: Tensor
    actor_q_valid_count: int
    human_residual_valid_count: int
    actor_q_guard_rejected_count: int
    actor_q_guard_unknown_count: int
    human_residual_projected_count: int
    human_residual_projected_axis_count: int
    candidate_q1_mean: Tensor | None
    candidate_q2_mean: Tensor | None
    zero_q1_mean: Tensor | None
    zero_q2_mean: Tensor | None
    behavior_q1_mean: Tensor | None
    behavior_q2_mean: Tensor | None


def residual_critic_loss(
    q1: nn.Module,
    q2: nn.Module,
    q1_target: nn.Module,
    q2_target: nn.Module,
    residual_actor_target: nn.Module,
    batch: Any,
    gamma: float,
    *,
    return_details: bool = False,
) -> Tensor | ResidualCriticLoss:
    """Policy-only TD loss in residual-proposal action space."""

    boundary = batch.terminated | batch.truncated
    bootstrap = ~boundary
    if torch.any(bootstrap & ~batch.next_base_valid):
        raise ValueError("FORCERFT_TD_NEXT_BASE_MISSING")
    target = batch.reward.float().clone()
    target_candidate_valid = torch.zeros_like(bootstrap)
    target_guard_rejected = torch.zeros_like(bootstrap)
    target_guard_unknown = torch.zeros_like(bootstrap)
    with torch.no_grad():
        if bool(bootstrap.any()):
            indices = torch.nonzero(bootstrap, as_tuple=False).squeeze(1)
            next_residual6 = residual_actor_target(
                normalized_state7=batch.next_state7[indices],
                normalized_wrench6=batch.next_wrench6[indices],
                normalized_wrench_delta6=batch.next_wrench_delta6[indices],
                base_action6=batch.next_base_action6[indices],
            )
            context = batch.next_candidate_guard.select(indices)
            guard_valid = policy_candidate_guard_valid(
                next_residual6,
                batch.next_base_action6[indices],
                context,
            )
            target_guard_unknown[indices] = ~context.valid
            target_guard_rejected[indices] = context.valid & ~guard_valid
            if bool(guard_valid.any()):
                valid_indices = indices[guard_valid]
                next_q = torch.minimum(
                    q1_target(
                        batch.next_state7[valid_indices],
                        batch.next_wrench6[valid_indices],
                        batch.next_wrench_delta6[valid_indices],
                        batch.next_base_action6[valid_indices],
                        batch.next_base_gripper[valid_indices],
                        next_residual6[guard_valid],
                    ),
                    q2_target(
                        batch.next_state7[valid_indices],
                        batch.next_wrench6[valid_indices],
                        batch.next_wrench_delta6[valid_indices],
                        batch.next_base_action6[valid_indices],
                        batch.next_base_gripper[valid_indices],
                        next_residual6[guard_valid],
                    ),
                )
                target[valid_indices] += float(gamma) * next_q
                target_candidate_valid[valid_indices] = True

    td_valid = boundary | target_candidate_valid
    valid_count = int(td_valid.sum())
    if valid_count:
        q1_value = q1(
            batch.state7[td_valid],
            batch.wrench6[td_valid],
            batch.wrench_delta6[td_valid],
            batch.base_action6[td_valid],
            batch.base_gripper[td_valid],
            batch.behavior_proposal6[td_valid],
        )
        q2_value = q2(
            batch.state7[td_valid],
            batch.wrench6[td_valid],
            batch.wrench_delta6[td_valid],
            batch.base_action6[td_valid],
            batch.base_gripper[td_valid],
            batch.behavior_proposal6[td_valid],
        )
        selected_target = target[td_valid]
        loss = 0.5 * (
            F.mse_loss(q1_value, selected_target)
            + F.mse_loss(q2_value, selected_target)
        )
    else:
        loss = sum(parameter.sum() * 0.0 for parameter in q1.parameters())
        loss += sum(parameter.sum() * 0.0 for parameter in q2.parameters())
    if loss.ndim or not torch.isfinite(loss):
        raise FloatingPointError("FORCERFT_CRITIC_LOSS_NONFINITE")
    details = ResidualCriticLoss(
        loss,
        valid_count,
        int(target_guard_rejected.sum()),
        int(target_guard_unknown.sum()),
    )
    return details if return_details else details.total


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
    """Min-Q value term plus bounded residual and human-correction terms."""

    zero = next(residual_actor.parameters()).sum() * 0.0
    candidate_residual6 = None
    output_norm = zero
    valid_count = guard_rejected_count = guard_unknown_count = 0
    value = residual = zero
    candidate_q1_mean = candidate_q2_mean = None
    zero_q1_mean = zero_q2_mean = None
    behavior_q1_mean = behavior_q2_mean = None
    if policy_batch is not None:
        candidate_residual6 = residual_actor(
            normalized_state7=policy_batch.state7,
            normalized_wrench6=policy_batch.wrench6,
            normalized_wrench_delta6=policy_batch.wrench_delta6,
            base_action6=policy_batch.base_action6,
        )
        eligible = policy_batch.actor_q_valid
        if eligible.dtype != torch.bool or eligible.shape != (candidate_residual6.shape[0],):
            raise ValueError("FORCERFT_ACTOR_Q_VALID_MASK_INVALID")
        guard_valid = policy_candidate_guard_valid(
            candidate_residual6,
            policy_batch.base_action6,
            policy_batch.candidate_guard,
        )
        valid = eligible & guard_valid
        valid_count = int(valid.sum())
        guard_unknown_count = int(
            (eligible & ~policy_batch.candidate_guard.valid).sum()
        )
        guard_rejected_count = int(
            (eligible & policy_batch.candidate_guard.valid & ~guard_valid).sum()
        )
        if valid_count:
            candidate_q1 = q1(
                policy_batch.state7[valid],
                policy_batch.wrench6[valid],
                policy_batch.wrench_delta6[valid],
                policy_batch.base_action6[valid],
                policy_batch.base_gripper[valid],
                candidate_residual6[valid],
            )
            candidate_q2 = q2(
                policy_batch.state7[valid],
                policy_batch.wrench6[valid],
                policy_batch.wrench_delta6[valid],
                policy_batch.base_action6[valid],
                policy_batch.base_gripper[valid],
                candidate_residual6[valid],
            )
            value = -torch.minimum(candidate_q1, candidate_q2).mean()
            candidate_q1_mean = candidate_q1.detach().mean()
            candidate_q2_mean = candidate_q2.detach().mean()

        with torch.no_grad():
            behavior_valid = eligible
            if bool(behavior_valid.any()):
                arguments = (
                    policy_batch.state7[behavior_valid],
                    policy_batch.wrench6[behavior_valid],
                    policy_batch.wrench_delta6[behavior_valid],
                    policy_batch.base_action6[behavior_valid],
                    policy_batch.base_gripper[behavior_valid],
                    policy_batch.behavior_proposal6[behavior_valid],
                )
                behavior_q1_mean = q1(*arguments).mean()
                behavior_q2_mean = q2(*arguments).mean()
            zero_proposal = torch.zeros_like(candidate_residual6)
            zero_guard_valid = policy_candidate_guard_valid(
                zero_proposal,
                policy_batch.base_action6,
                policy_batch.candidate_guard,
            )
            zero_valid = eligible & zero_guard_valid
            if bool(zero_valid.any()):
                arguments = (
                    policy_batch.state7[zero_valid],
                    policy_batch.wrench6[zero_valid],
                    policy_batch.wrench_delta6[zero_valid],
                    policy_batch.base_action6[zero_valid],
                    policy_batch.base_gripper[zero_valid],
                    zero_proposal[zero_valid],
                )
                zero_q1_mean = q1(*arguments).mean()
                zero_q2_mean = q2(*arguments).mean()
        residual = candidate_residual6.square().mean()
        output_norm = candidate_residual6.norm(dim=-1).mean()

    human_count = projected_count = projected_axis_count = 0
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
                base_action6=human_batch.base_action6[human_valid],
            )
            if candidate_residual6 is None:
                residual = human_prediction.square().mean()
                output_norm = human_prediction.norm(dim=-1).mean()
            raw_target = human_batch.human_residual_target6[human_valid].detach()
            cap = torch.as_tensor(
                getattr(
                    residual_actor,
                    "residual_cap6",
                    [float(residual_actor.max_normalized_residual)] * 6,
                ),
                device=raw_target.device,
                dtype=raw_target.dtype,
            )
            target_bc = torch.maximum(torch.minimum(raw_target, cap), -cap)
            projected_count = int((raw_target != target_bc).any(dim=1).sum())
            projected_axis_count = int((raw_target != target_bc).sum())
            human = F.mse_loss(human_prediction, target_bc)
            if candidate_residual6 is None:
                residual = human_prediction.square().mean()
                output_norm = human_prediction.norm(dim=-1).mean()
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
        output_norm,
        valid_count,
        human_count,
        guard_rejected_count,
        guard_unknown_count,
        projected_count,
        projected_axis_count,
        candidate_q1_mean,
        candidate_q2_mean,
        zero_q1_mean,
        zero_q2_mean,
        behavior_q1_mean,
        behavior_q2_mean,
    )
