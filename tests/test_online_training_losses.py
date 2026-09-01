from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn

from forcesmolvla.rft.losses import CriticObservation
from forcesmolvla.rft.online.training_losses import (
    compute_expert_only_flow_matching_loss,
    compute_online_twin_q_td_loss,
    compute_online_actor_objective,
    compute_online_min_twin_q_actor_loss,
)
from forcesmolvla.rft.critic_action_adapter_v2 import (
    aligned_fresh_chunk_execution_index_map_v2,
)


class ToyCritic(nn.Module):
    def __init__(self, bias: float) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias))
        self.calls = 0

    def forward(self, observation, action, mask):
        self.calls += 1
        return observation[:, 0] + (action * mask.unsqueeze(-1)).sum((1, 2)) * 0.01 + self.bias


class ActionOnlyCritic(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(offset))

    def forward(self, _c1, _c2, _task, _state, _wrench, action, mask):
        weights = torch.arange(1, 22, device=action.device, dtype=action.dtype).view(3, 7)
        return (action * weights * mask.unsqueeze(-1)).sum((1, 2)) + self.offset


def test_pure_online_td_has_no_calql_random_or_mc_and_uses_target_min() -> None:
    q1, q2 = ToyCritic(0.0), ToyCritic(1.0)
    target1, target2 = ToyCritic(2.0).eval(), ToyCritic(3.0).eval()
    observation = torch.zeros(2, 1)
    action = torch.zeros(2, 3, 7)
    actor_calls = 0

    def next_action(next_observation):
        nonlocal actor_calls
        actor_calls += 1
        assert next_observation.shape == (1, 1)
        return torch.zeros(1, 3, 7)

    with (
        patch("forcesmolvla.rft.losses.evaluate_calql_candidates", side_effect=AssertionError),
        patch("forcesmolvla.rft.losses.compute_calql_penalty", side_effect=AssertionError),
    ):
        result = compute_online_twin_q_td_loss(
            q1=q1, q2=q2, q1_target=target1, q2_target=target2,
            observation=observation, next_observation=observation,
            ack_behavior_action_k7=action,
            behavior_mask=torch.ones(2, 3, dtype=torch.bool),
            reward=torch.tensor([1.0, 0.0]),
            discount=torch.tensor([0.0, 0.99]),
            terminated=torch.tensor([True, False]),
            truncated=torch.tensor([False, False]),
            bootstrap_mask=torch.tensor([False, True]),
            next_policy_action_fn=next_action,
        )
    torch.testing.assert_close(result.target, torch.tensor([1.0, 1.98]))
    assert actor_calls == result.next_actor_calls == 1
    assert result.target_q1_calls == result.target_q2_calls == 1
    assert result.calql_candidate_calls == result.random_candidate_calls == result.mc_return_reads == 0
    result.total.backward()
    assert q1.bias.grad is not None and q2.bias.grad is not None
    assert target1.bias.grad is None and target2.bias.grad is None


def test_all_terminal_rows_never_call_next_actor_or_target_critics() -> None:
    q1, q2 = ToyCritic(0.0), ToyCritic(0.0)
    target1, target2 = ToyCritic(2.0), ToyCritic(3.0)

    def forbidden(_):
        raise AssertionError("terminal Actor call")

    result = compute_online_twin_q_td_loss(
        q1=q1, q2=q2, q1_target=target1, q2_target=target2,
        observation=torch.zeros(2, 1), next_observation=torch.zeros(2, 1),
        ack_behavior_action_k7=torch.zeros(2, 3, 7),
        behavior_mask=torch.ones(2, 3, dtype=torch.bool),
        reward=torch.tensor([1.0, -1.0]), discount=torch.zeros(2),
        terminated=torch.ones(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
        bootstrap_mask=torch.zeros(2, dtype=torch.bool),
        next_policy_action_fn=forbidden,
    )
    assert result.next_actor_calls == result.target_q1_calls == result.target_q2_calls == 0
    assert target1.calls == target2.calls == 0


def test_expert_only_fm_zero_batch_is_graph_connected_exact_zero() -> None:
    raw = torch.ones(2, 50, 7, requires_grad=True)
    valid = torch.ones(2, 50, dtype=torch.bool)
    no_expert = torch.zeros_like(raw, dtype=torch.bool)
    loss, count = compute_expert_only_flow_matching_loss(raw, valid, no_expert)
    assert count == 0 and loss.item() == 0.0 and loss.grad_fn is not None
    loss.backward()
    assert torch.count_nonzero(raw.grad) == 0

    raw.grad = None
    expert = no_expert.clone(); expert[0, 0] = True
    loss, count = compute_expert_only_flow_matching_loss(raw, valid, expert)
    assert count == 7
    loss.backward()
    assert torch.all(raw.grad[0, 0] != 0)
    assert raw.grad[0, 0, 6] != 0
    assert torch.count_nonzero(raw.grad[1]) == 0


def test_human_masked_fm_is_nonzero_and_autonomous_fm_is_zero() -> None:
    per_feature = torch.arange(700, dtype=torch.float32).reshape(2, 50, 7)
    per_feature.requires_grad_()
    valid = torch.ones(2, 50, dtype=torch.bool)
    expert = torch.zeros_like(per_feature, dtype=torch.bool)
    expert[0, 3, :] = True

    loss, count = compute_expert_only_flow_matching_loss(
        per_feature, valid, expert
    )
    loss.backward()

    assert count == 7
    assert torch.count_nonzero(per_feature.grad[0, 3]) == 7
    assert torch.count_nonzero(per_feature.grad[1]) == 0


def test_human_partial_ack_action_still_enters_critic_td() -> None:
    q1, q2 = ToyCritic(0.0), ToyCritic(1.0)
    target1, target2 = ToyCritic(2.0), ToyCritic(3.0)
    result = compute_online_twin_q_td_loss(
        q1=q1,
        q2=q2,
        q1_target=target1,
        q2_target=target2,
        observation=torch.zeros(1, 1),
        next_observation=torch.zeros(1, 1),
        ack_behavior_action_k7=torch.ones(1, 3, 7),
        behavior_mask=torch.tensor([[True, False, False]]),
        reward=torch.tensor([1.0]),
        discount=torch.tensor([0.0]),
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
        bootstrap_mask=torch.tensor([False]),
        next_policy_action_fn=lambda _: (_ for _ in ()).throw(AssertionError),
    )

    assert torch.isfinite(result.total)
    assert q1.calls == q2.calls == 1


def test_actor_objective_uses_min_q_and_actioncontract_v2_stops_gripper_q_gradient() -> None:
    per_feature = torch.ones(2, 50, 7, requires_grad=True)
    expert = torch.zeros_like(per_feature, dtype=torch.bool); expert[0] = True
    terms = compute_online_actor_objective(
        per_feature_flow_loss=per_feature,
        action_valid_mask_h50=torch.ones(2, 50, dtype=torch.bool),
        expert_feature_mask_h50x7=expert,
        q1_actor_value=torch.tensor([2.0, 5.0]),
        q2_actor_value=torch.tensor([3.0, 4.0]),
        actor_q_valid=torch.ones(2, dtype=torch.bool),
        balance_loss=torch.tensor(0.0), z_loss=torch.tensor(0.0), beta=1.0, eta=3.0,
    )
    assert terms.actor_q.item() == pytest.approx(-3.0)

    chunk = torch.randn(2, 50, 7, requires_grad=True)
    zeros = torch.zeros(2, 1)
    observation = CriticObservation(zeros, zeros, zeros, torch.zeros(2, 7), torch.zeros(2, 6))
    mean = torch.tensor([0.0] * 6 + [0.028491082421846097])
    std = torch.tensor([1.0] * 6 + [0.04012480845771951])
    loss, _q1, _q2, action = compute_online_min_twin_q_actor_loss(
        q1=ActionOnlyCritic(0.0), q2=ActionOnlyCritic(1.0), observation=observation,
        normalized_flow_action_chunk7=chunk,
        execution_index_map=aligned_fresh_chunk_execution_index_map_v2(),
        delta_action_mean7=mean, delta_action_std7=std,
    )
    loss.backward()
    assert torch.count_nonzero(chunk.grad[:, 0, :6]) > 0
    assert torch.count_nonzero(chunk.grad[..., 6]) == 0
    assert torch.count_nonzero(chunk.grad[:, 1:]) == 0
    assert action.shape == (2, 3, 7)


def test_intervention_truncation_uses_immediate_reward_without_next_calls() -> None:
    q1, q2 = ToyCritic(0.0), ToyCritic(0.0)
    target1, target2 = ToyCritic(2.0), ToyCritic(3.0)

    result = compute_online_twin_q_td_loss(
        q1=q1,
        q2=q2,
        q1_target=target1,
        q2_target=target2,
        observation=torch.zeros(1, 1),
        next_observation=torch.zeros(1, 1),
        ack_behavior_action_k7=torch.zeros(1, 3, 7),
        behavior_mask=torch.ones(1, 3, dtype=torch.bool),
        reward=torch.tensor([0.25]),
        discount=torch.zeros(1),
        terminated=torch.tensor([False]),
        truncated=torch.tensor([True]),
        bootstrap_mask=torch.tensor([False]),
        next_policy_action_fn=lambda _: (_ for _ in ()).throw(AssertionError),
    )

    torch.testing.assert_close(result.target, torch.tensor([0.25]))
    assert result.next_actor_calls == result.target_q1_calls == result.target_q2_calls == 0
    assert target1.calls == target2.calls == 0


def test_post_takeover_fresh_generation_row_still_bootstraps() -> None:
    q1, q2 = ToyCritic(0.0), ToyCritic(0.0)
    target1, target2 = ToyCritic(2.0), ToyCritic(3.0)
    result = compute_online_twin_q_td_loss(
        q1=q1,
        q2=q2,
        q1_target=target1,
        q2_target=target2,
        observation=torch.zeros(2, 1),
        next_observation=torch.zeros(2, 1),
        ack_behavior_action_k7=torch.zeros(2, 3, 7),
        behavior_mask=torch.ones(2, 3, dtype=torch.bool),
        reward=torch.tensor([0.0, 0.5]),
        discount=torch.tensor([0.0, 0.99]),
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([True, False]),
        bootstrap_mask=torch.tensor([False, True]),
        next_policy_action_fn=lambda observation: torch.zeros(
            observation.shape[0], 3, 7
        ),
    )
    torch.testing.assert_close(result.target, torch.tensor([0.0, 2.48]))
    assert result.next_actor_calls == result.target_q1_calls == result.target_q2_calls == 1


def test_terminal_and_truncation_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="TERMINATED_AND_TRUNCATED"):
        compute_online_twin_q_td_loss(
            q1=ToyCritic(0.0),
            q2=ToyCritic(0.0),
            q1_target=ToyCritic(0.0),
            q2_target=ToyCritic(0.0),
            observation=torch.zeros(1, 1),
            next_observation=torch.zeros(1, 1),
            ack_behavior_action_k7=torch.zeros(1, 3, 7),
            behavior_mask=torch.ones(1, 3, dtype=torch.bool),
            reward=torch.zeros(1),
            discount=torch.zeros(1),
            terminated=torch.tensor([True]),
            truncated=torch.tensor([True]),
            bootstrap_mask=torch.tensor([False]),
            next_policy_action_fn=lambda _: torch.zeros(1, 3, 7),
        )
