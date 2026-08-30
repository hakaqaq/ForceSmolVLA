from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import rft_losses_numpy_oracle as oracle

from forcesmolvla.rft.losses import (
    CriticObservation,
    build_actor_q_action,
    compute_actor_q_loss,
    compute_calql_penalty,
    compute_offline_actor_objective,
    compute_td_target,
    compute_td_target_from_current_actor,
    compute_twin_q_critic_loss,
    derive_loss_masks,
    evaluate_calql_candidates,
    load_authorized_reward_train_transitions,
    validate_mc_return_recurrence,
)


def observation(batch=3):
    generator = torch.Generator().manual_seed(12)
    return CriticObservation(
        camera1=torch.randn(batch, 1, generator=generator),
        camera2=torch.randn(batch, 1, generator=generator),
        task_feature=torch.randn(batch, 1, generator=generator),
        normalized_state7=torch.randn(batch, 7, generator=generator),
        normalized_wrench6=torch.randn(batch, 6, generator=generator),
    )


class TinyCritic(nn.Module):
    def __init__(self, action_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(
            action_scale * torch.arange(1, 22, dtype=torch.float32).reshape(3, 7) / 21
        )
        self.state_weight = nn.Parameter(torch.linspace(0.1, 0.7, 7))
        self.calls = 0
        self.last_action = None
        self.last_mask = None
        self.last_state = None

    def forward(self, camera1, camera2, task, state, wrench, action, mask):
        self.calls += 1
        self.last_action = action.detach().clone()
        self.last_mask = mask.detach().clone()
        self.last_state = state.detach().clone()
        return ((action * self.weight) * mask[..., None]).sum((1, 2)) + (state * self.state_weight).sum(1)


class FakeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(0.0))


def test_td_target_numpy_parity_and_no_duplicate_discount():
    reward = torch.tensor([1.0, 0.0, 0.25])
    discount = torch.tensor([0.0, 0.99, 0.99])
    terminated = torch.tensor([True, False, False])
    bootstrap = torch.tensor([0, 1, 1])
    q1 = torch.tensor([2.0, 4.0])
    q2 = torch.tensor([3.0, 1.0])
    actual = compute_td_target(reward, discount, terminated, bootstrap, q1, q2)
    expected = oracle.td_target(reward, discount, terminated, q1, q2)
    assert np.array_equal(actual.numpy(), expected)
    assert torch.equal(actual, torch.tensor([1.0, 1.98, 1.24]))


def test_terminal_rows_make_zero_actor_and_target_calls_and_restore_actor_mode():
    actor = FakeActor().train(True)
    q1, q2 = TinyCritic(), TinyCritic(2.0)
    for target in (q1, q2):
        target.eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
    sample_calls = []

    def sampler(*args, **kwargs):
        sample_calls.append((args, kwargs))
        raise AssertionError("terminal-only batch must never sample")

    batch = 2
    target = compute_td_target_from_current_actor(
        reward=torch.ones(batch),
        discount=torch.zeros(batch),
        terminated=torch.ones(batch, dtype=torch.bool),
        bootstrap_mask=torch.zeros(batch, dtype=torch.int8),
        next_observation=observation(batch),
        next_actor_batch={"marker": torch.arange(batch)},
        next_noise7=torch.zeros(batch, 50, 7),
        actor=actor,
        q1_target=q1,
        q2_target=q2,
        delta_action_mean7=torch.zeros(7),
        delta_action_std7=torch.ones(7),
        call_id="terminal-only",
        sample_action_fn=sampler,
    )
    assert torch.equal(target, torch.ones(batch))
    assert q1.calls == q2.calls == len(sample_calls) == 0
    assert actor.training


def test_next_actor_uses_next_rows_slot_zero_and_full_policy_mask():
    actor = FakeActor().train(True)
    q1, q2 = TinyCritic(), TinyCritic(2.0)
    for target in (q1, q2):
        target.eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
    seen = {}

    def sampler(policy, batch, noise, *, call_id, purpose):
        assert not policy.training and not torch.is_grad_enabled()
        assert purpose == "td_next" and call_id == "next"
        seen["markers"] = batch["marker"].clone()
        chunk = torch.zeros(noise.shape[0], 50, 7)
        chunk[:, 0, 0] = batch["marker"].float()
        chunk[:, 1, 0] = batch["marker"].float() + 1
        chunk[:, 2, 0] = batch["marker"].float() + 2
        chunk[..., 6] = 0.085
        return chunk

    obs = observation(3)
    obs.normalized_state7[:, 0] = torch.tensor([10.0, 20.0, 30.0])
    compute_td_target_from_current_actor(
        reward=torch.zeros(3),
        discount=torch.tensor([0.99, 0.0, 0.99]),
        terminated=torch.tensor([False, True, False]),
        bootstrap_mask=torch.tensor([1, 0, 1]),
        next_observation=obs,
        next_actor_batch={"marker": torch.tensor([101, 202, 303]), "sample_identity": ("a", "b", "c")},
        next_noise7=torch.zeros(3, 50, 7),
        actor=actor,
        q1_target=q1,
        q2_target=q2,
        delta_action_mean7=torch.zeros(7),
        delta_action_std7=torch.ones(7),
        call_id="next",
        sample_action_fn=sampler,
    )
    assert seen["markers"].tolist() == [101, 303]
    assert q1.last_state[:, 0].tolist() == [10.0, 30.0]
    assert q1.last_action[:, :, 0].tolist() == [[101.0, 102.0, 103.0], [303.0, 304.0, 305.0]]
    assert torch.all(q1.last_mask)
    assert actor.training


def test_calql_formula_numpy_parity_dataset_once_candidate_only_bound_and_clip():
    qd = torch.tensor([2.0, -2.0])
    candidates = torch.tensor([[0.0, 1.0, 3.0, 4.0, -1.0, 2.5], [-4.0, -3.0, -1.0, 0.0, 1.0, 2.0]])
    mc = torch.tensor([2.5, -0.5])
    valid = torch.tensor([True, True])
    actual = compute_calql_penalty(
        qd, candidates, mc, valid, temperature=0.7, clip_min=-0.25, clip_max=1.25
    )
    expected = oracle.calql_penalty(
        qd, candidates, mc, valid, temperature=0.7, clip_min=-0.25, clip_max=1.25
    )
    assert np.isclose(float(actual), float(expected), atol=2e-7)
    # Raising only dataset Q cannot be hidden by the MC lower bound.
    changed = compute_calql_penalty(
        qd + 10, candidates, mc, valid, temperature=0.7, clip_min=-20, clip_max=20
    )
    assert changed != actual


def test_twin_loss_numpy_parity_and_empty_calql_exact_zero():
    q1 = torch.tensor([0.2, -0.3], requires_grad=True)
    q2 = torch.tensor([-0.1, 0.4], requires_grad=True)
    target = torch.tensor([1.0, 0.5])
    c1 = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 10
    c2 = -c1
    mc = torch.tensor([0.7, 1.0])
    valid = torch.tensor([True, False])
    terms = compute_twin_q_critic_loss(
        q1_dataset=q1,
        q2_dataset=q2,
        td_target=target,
        q1_candidates=c1,
        q2_candidates=c2,
        mc_return=mc,
        calql_valid=valid,
        alpha_calql=0.2,
        temperature=0.5,
        clip_min=-2.0,
        clip_max=3.0,
    )
    expected = oracle.twin_q_loss(
        q1.detach(), q2.detach(), target, c1, c2, mc, valid,
        alpha=0.2, temperature=0.5, clip_min=-2.0, clip_max=3.0,
    )
    for name in expected:
        assert np.isclose(float(getattr(terms, name)), float(expected[name]), atol=2e-7)
    zero = compute_calql_penalty(
        q1, c1, mc, torch.zeros(2, dtype=torch.bool),
        temperature=1.0, clip_min=-1.0, clip_max=1.0,
    )
    assert torch.equal(zero, torch.tensor(0.0)) and torch.isfinite(zero)


def test_candidate_sets_detached_all_ones_and_evaluated_at_current_observation():
    critic = TinyCritic()
    obs = observation(2)
    endpoint = torch.tensor([-1.0, 1.0])
    tensors = []
    for offset in (0.0, 1.0, 2.0):
        value = torch.full((2, 2, 3, 7), offset, requires_grad=True)
        value = value.clone()
        value[..., 6] = endpoint[0]
        tensors.append(value)
    q = evaluate_calql_candidates(critic, obs, *tensors, endpoint)
    assert q.shape == (2, 6)
    assert critic.last_state.shape[0] == 12
    assert torch.equal(critic.last_state[0], obs.normalized_state7[0])
    assert torch.equal(critic.last_state[6], obs.normalized_state7[1])
    assert torch.all(critic.last_mask)
    q.sum().backward()
    assert all(value.grad is None for value in tensors)
    bad = tensors[0].detach().clone()
    bad[..., 6] = 0.0
    with pytest.raises(ValueError, match="GRIPPER"):
        evaluate_calql_candidates(critic, obs, bad, tensors[1], tensors[2], endpoint)


def test_mask_ownership_partial_tail_td_but_not_calql_or_actor_q():
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    terminated = torch.tensor([False, True, True])
    result = derive_loss_masks(mask, terminated)
    assert result["full_macro_valid"].tolist() == [True, False, False]
    assert result["calql_valid"].tolist() == [True, False, False]
    assert result["actor_q_valid"].tolist() == [True, False, False]


def test_actor_q_mean_sign_tcp_gradient_gripper_stop_and_critic_restore():
    q1, q2 = TinyCritic(1.0).train(True), TinyCritic(2.0).train(False)
    flags1 = [parameter.requires_grad for parameter in q1.parameters()]
    flags2 = [parameter.requires_grad for parameter in q2.parameters()]
    chunk = torch.zeros(2, 50, 7, requires_grad=True)
    loss = compute_actor_q_loss(
        q1=q1,
        q2=q2,
        current_observation=observation(2),
        actor_action_chunk7=chunk,
        actor_q_valid=torch.tensor([True, False]),
        delta_action_mean7=torch.zeros(7),
        delta_action_std7=torch.ones(7),
    )
    loss.backward()
    assert loss < 0
    assert torch.all(chunk.grad[0, :3, :6] != 0)
    assert torch.count_nonzero(chunk.grad[..., 6]) == 0
    assert torch.count_nonzero(chunk.grad[:, 3:]) == 0
    assert all(parameter.grad is None for critic in (q1, q2) for parameter in critic.parameters())
    assert q1.training and not q2.training
    assert flags1 == [parameter.requires_grad for parameter in q1.parameters()]
    assert flags2 == [parameter.requires_grad for parameter in q2.parameters()]


def test_actor_objective_does_not_double_negate_and_q_action_shape():
    chunk = torch.zeros(1, 50, 7)
    action = build_actor_q_action(
        chunk,
        delta_action_mean7=torch.zeros(7),
        delta_action_std7=torch.ones(7),
    )
    assert action.shape == (1, 3, 7)
    terms = compute_offline_actor_objective(
        flow_matching_loss=torch.tensor(2.0),
        actor_q_loss=torch.tensor(-3.0),
        balance_loss=torch.tensor(5.0),
        z_loss=torch.tensor(7.0),
        beta=0.4,
        eta=0.2,
    )
    assert torch.isclose(terms.total, torch.tensor(0.257))


def test_mc_return_and_authorized_loader(tmp_path):
    rows = [
        {"episode_id": "e", "reward": 0.0, "discount": 0.99, "mc_return": 0.99, "terminated": False, "reward_source": "frozen_classifier_detector"},
        {"episode_id": "e", "reward": 1.0, "discount": 0.0, "mc_return": 1.0, "terminated": True, "reward_source": "frozen_classifier_detector"},
    ]
    assert validate_mc_return_recurrence(rows)["maximum_absolute_error"] == 0.0
    table = load_authorized_reward_train_transitions()
    assert table.num_rows == 10075 and set(table.column("split").to_pylist()) == {"train"}
    with pytest.raises(RuntimeError, match="BEFORE_OPEN"):
        load_authorized_reward_train_transitions(tmp_path / "manual")
