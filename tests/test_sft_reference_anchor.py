from __future__ import annotations

import pytest
import torch

from forcesmolvla.rft.online.training_losses import (
    compute_online_actor_objective,
    compute_sft_reference_anchor_loss,
)


def test_identical_current_and_reference_actions_have_zero_anchor() -> None:
    actor = torch.randn(2, 3, 7, requires_grad=True)
    loss = compute_sft_reference_anchor_loss(
        actor,
        actor.detach().clone(),
        torch.ones(2, dtype=torch.bool),
    )

    assert loss.item() == pytest.approx(0.0)


def test_tcp_change_is_anchored_and_reference_is_stop_gradient() -> None:
    actor = torch.ones(2, 3, 7, requires_grad=True)
    reference = torch.zeros_like(actor, requires_grad=True)
    loss = compute_sft_reference_anchor_loss(
        actor,
        reference,
        torch.ones(2, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == pytest.approx(1.0)
    assert torch.count_nonzero(actor.grad[..., :6]) == 36
    assert torch.count_nonzero(actor.grad[..., 6]) == 0
    assert reference.grad is None


def test_gripper_difference_is_not_anchored() -> None:
    actor = torch.zeros(1, 3, 7, requires_grad=True)
    reference = torch.zeros_like(actor)
    reference[..., 6] = 5.0
    loss = compute_sft_reference_anchor_loss(
        actor,
        reference,
        torch.ones(1, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert torch.count_nonzero(actor.grad) == 0


def test_reference_anchor_applies_to_every_valid_actor_row() -> None:
    actor = torch.ones(2, 3, 7, requires_grad=True)
    reference = torch.zeros_like(actor)
    reference_loss = compute_sft_reference_anchor_loss(
        actor,
        reference,
        torch.tensor([True, True]),
    )
    terms = compute_online_actor_objective(
        per_feature_flow_loss=torch.zeros(2, 50, 7),
        action_valid_mask_h50=torch.ones(2, 50, dtype=torch.bool),
        expert_feature_mask_h50x7=torch.zeros(2, 50, 7, dtype=torch.bool),
        q1_actor_value=torch.zeros(2),
        q2_actor_value=torch.zeros(2),
        actor_q_valid=torch.tensor([True, False]),
        sft_reference_anchor_loss=reference_loss,
        balance_loss=torch.tensor(0.0),
        z_loss=torch.tensor(0.0),
        beta=1.0,
        eta=0.0,
        lambda_sft_reference_anchor=1.0,
    )

    assert terms.sft_reference_anchor.item() == pytest.approx(1.0)
    assert terms.total.item() == pytest.approx(1.0)

