"""Losses for the image-free residual Actor and ACK-aligned Twin-Q."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


RECORDED_CANDIDATE_ATOL = 1.0e-6


@dataclass(frozen=True)
class AcceptedCandidate:
    residual_k6: Tensor
    valid: Tensor


@dataclass(frozen=True)
class ResidualCriticLoss:
    total: Tensor
    td_valid_count: int
    target_candidate_unavailable_count: int


@dataclass(frozen=True)
class ResidualActorLoss:
    total: Tensor
    value: Tensor
    residual: Tensor
    human: Tensor
    output_norm: Tensor
    actor_q_valid_count: int
    human_residual_valid_count: int
    actor_q_mapping_unavailable_count: int
    human_residual_projected_count: int


def accepted_candidate_for_q(
    candidate_residual6: Tensor,
    *,
    differentiable_identity_valid: Tensor,
    recorded_proposal_residual6: Tensor | None = None,
    recorded_accepted_residual_k6: Tensor | None = None,
    recorded_point_valid: Tensor | None = None,
    allow_recorded_point: bool,
) -> AcceptedCandidate:
    """Map a proposal to ACK space only where the recorded interface proves it.

    The deployed controller applies a stateful low-pass/reference leash after the
    locally recorded workspace guard.  Its hidden pre-update reference and exact
    time step are not present in the ACK.  Consequently a differentiable value
    query is allowed only for an explicitly verified identity domain.  A target
    query may additionally reuse the real ACK at the exact proposal point that
    produced it; that path is intentionally used only under ``no_grad``.
    """

    batch = int(candidate_residual6.shape[0])
    if candidate_residual6.shape != (batch, 6):
        raise ValueError("FORCERFT_CANDIDATE_RESIDUAL_SHAPE_INVALID")
    if (
        differentiable_identity_valid.dtype != torch.bool
        or differentiable_identity_valid.shape != (batch,)
    ):
        raise ValueError("FORCERFT_CANDIDATE_IDENTITY_MASK_INVALID")
    candidate_k6 = candidate_residual6[:, None, :].expand(-1, 3, -1)
    accepted = candidate_k6
    valid = differentiable_identity_valid.clone()
    if allow_recorded_point:
        if (
            recorded_proposal_residual6 is None
            or recorded_accepted_residual_k6 is None
            or recorded_point_valid is None
        ):
            raise ValueError("FORCERFT_RECORDED_ACCEPTANCE_POINT_MISSING")
        if (
            recorded_proposal_residual6.shape != (batch, 6)
            or recorded_accepted_residual_k6.shape != (batch, 3, 6)
            or recorded_point_valid.dtype != torch.bool
            or recorded_point_valid.shape != (batch,)
        ):
            raise ValueError("FORCERFT_RECORDED_ACCEPTANCE_POINT_INVALID")
        exact = recorded_point_valid & (
            (candidate_residual6 - recorded_proposal_residual6)
            .abs()
            .amax(dim=1)
            <= RECORDED_CANDIDATE_ATOL
        )
        accepted = torch.where(exact[:, None, None], recorded_accepted_residual_k6, accepted)
        valid |= exact
    return AcceptedCandidate(accepted, valid)


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
    """ACK TD loss with current behavior and successor candidate in ACK space."""

    boundary = batch.terminated | batch.truncated
    bootstrap = ~boundary
    if torch.any(bootstrap & ~batch.next_base_valid):
        raise ValueError("FORCERFT_TD_NEXT_BASE_MISSING")
    target = batch.reward.float().clone()
    target_mapping_valid = torch.zeros_like(bootstrap)
    with torch.no_grad():
        if bool(bootstrap.any()):
            indices = torch.nonzero(bootstrap, as_tuple=False).squeeze(1)
            next_residual6 = residual_actor_target(
                normalized_state7=batch.next_state7[indices],
                normalized_wrench6=batch.next_wrench6[indices],
                normalized_wrench_delta6=batch.next_wrench_delta6[indices],
                base_action6=batch.next_base_action_k6[indices, 0],
            )
            mapped = accepted_candidate_for_q(
                next_residual6,
                differentiable_identity_valid=(
                    batch.next_candidate_acceptance_identity_valid[indices]
                ),
                recorded_proposal_residual6=(
                    batch.next_recorded_proposal_residual6[indices]
                ),
                recorded_accepted_residual_k6=(
                    batch.next_recorded_behavior_residual_k6[indices]
                ),
                recorded_point_valid=batch.next_recorded_point_valid[indices],
                allow_recorded_point=True,
            )
            if bool(mapped.valid.any()):
                mapped_indices = indices[mapped.valid]
                mapped_residual = mapped.residual_k6[mapped.valid]
                next_q = torch.minimum(
                    q1_target(
                        batch.next_state7[mapped_indices],
                        batch.next_wrench6[mapped_indices],
                        batch.next_wrench_delta6[mapped_indices],
                        batch.next_base_action_k6[mapped_indices],
                        mapped_residual,
                        batch.next_action_mask[mapped_indices],
                        batch.next_control_source[mapped_indices],
                        batch.next_gripper_command[mapped_indices],
                    ),
                    q2_target(
                        batch.next_state7[mapped_indices],
                        batch.next_wrench6[mapped_indices],
                        batch.next_wrench_delta6[mapped_indices],
                        batch.next_base_action_k6[mapped_indices],
                        mapped_residual,
                        batch.next_action_mask[mapped_indices],
                        batch.next_control_source[mapped_indices],
                        batch.next_gripper_command[mapped_indices],
                    ),
                )
                target[mapped_indices] += float(gamma) * next_q
                target_mapping_valid[mapped_indices] = True

    td_valid = boundary | target_mapping_valid
    valid_count = int(td_valid.sum())
    unavailable_count = int((bootstrap & ~target_mapping_valid).sum())
    if valid_count:
        q1_value = q1(
            batch.state7[td_valid],
            batch.wrench6[td_valid],
            batch.wrench_delta6[td_valid],
            batch.base_action_k6[td_valid],
            batch.behavior_residual_k6[td_valid],
            batch.action_mask[td_valid],
            batch.control_source[td_valid],
            batch.gripper_command[td_valid],
        )
        q2_value = q2(
            batch.state7[td_valid],
            batch.wrench6[td_valid],
            batch.wrench_delta6[td_valid],
            batch.base_action_k6[td_valid],
            batch.behavior_residual_k6[td_valid],
            batch.action_mask[td_valid],
            batch.control_source[td_valid],
            batch.gripper_command[td_valid],
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
    details = ResidualCriticLoss(loss, valid_count, unavailable_count)
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
    valid_count = unavailable_count = 0
    value = residual = zero
    if policy_batch is not None:
        candidate_residual6 = residual_actor(
            normalized_state7=policy_batch.state7,
            normalized_wrench6=policy_batch.wrench6,
            normalized_wrench_delta6=policy_batch.wrench_delta6,
            base_action6=policy_batch.base_action_k6[:, 0],
        )
        eligible = policy_batch.actor_q_valid
        if eligible.dtype != torch.bool or eligible.shape != (candidate_residual6.shape[0],):
            raise ValueError("FORCERFT_ACTOR_Q_VALID_MASK_INVALID")
        mapped = accepted_candidate_for_q(
            candidate_residual6,
            differentiable_identity_valid=(
                policy_batch.candidate_acceptance_identity_valid
            ),
            allow_recorded_point=False,
        )
        valid = eligible & mapped.valid
        valid_count = int(valid.sum())
        unavailable_count = int((eligible & ~mapped.valid).sum())
        if valid_count:
            value = -torch.minimum(
                q1(
                    policy_batch.state7[valid],
                    policy_batch.wrench6[valid],
                    policy_batch.wrench_delta6[valid],
                    policy_batch.base_action_k6[valid],
                    mapped.residual_k6[valid],
                    policy_batch.action_mask[valid],
                    policy_batch.control_source[valid],
                    policy_batch.gripper_command[valid],
                ),
                q2(
                    policy_batch.state7[valid],
                    policy_batch.wrench6[valid],
                    policy_batch.wrench_delta6[valid],
                    policy_batch.base_action_k6[valid],
                    mapped.residual_k6[valid],
                    policy_batch.action_mask[valid],
                    policy_batch.control_source[valid],
                    policy_batch.gripper_command[valid],
                ),
            ).mean()
        residual = candidate_residual6.square().mean()
        output_norm = candidate_residual6.norm(dim=-1).mean()

    human_count = projected_count = 0
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
            raw_target = human_batch.human_residual_target6[human_valid].detach()
            cap = float(residual_actor.max_normalized_residual)
            target_bc = raw_target.clamp(-cap, cap)
            projected_count = int((raw_target != target_bc).any(dim=1).sum())
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
        unavailable_count,
        projected_count,
    )
