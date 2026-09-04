from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import serve_forcerft_actor_learner as learner_server  # noqa: E402

from forcesmolvla.rft.online.actor_learner_runtime import OnlineTrainingPolicy
from forcesmolvla.rft.online.replay_training import (
    algorithm_hyperparameters,
    load_common_actor_critic_config,
)
from forcesmolvla.rft.online.training_losses import (
    residual_actor_loss,
    residual_critic_loss,
)


class ConstantQ(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value))
        self.batch_sizes: list[int] = []

    def forward(self, state, wrench, wrench_delta, base, residual, mask):
        del wrench, wrench_delta, base, mask
        self.batch_sizes.append(len(state))
        return self.value.expand(len(state)) + residual.flatten(1).mean(1) * 0.0


class TargetActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        return torch.zeros(len(kwargs["normalized_state7"]), 6)


class ScalarResidualActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))
        self.batch_sizes: list[int] = []

    def forward(self, **kwargs):
        batch = len(kwargs["normalized_state7"])
        self.batch_sizes.append(batch)
        return self.value.expand(batch, 6)


def batch(batch_size: int = 2) -> SimpleNamespace:
    zeros7 = torch.zeros(batch_size, 7)
    zeros6 = torch.zeros(batch_size, 6)
    zeros_k6 = torch.zeros(batch_size, 3, 6)
    mask = torch.ones(batch_size, 3, dtype=torch.bool)
    return SimpleNamespace(
        state7=zeros7,
        wrench6=zeros6,
        wrench_delta6=zeros6,
        base_action_k6=zeros_k6,
        behavior_residual_k6=zeros_k6,
        action_mask=mask,
        next_state7=zeros7,
        next_wrench6=zeros6,
        next_wrench_delta6=zeros6,
        next_base_action_k6=zeros_k6,
        next_action_mask=mask,
        reward=torch.ones(batch_size),
        terminated=torch.tensor([False, True][:batch_size]),
        truncated=torch.zeros(batch_size, dtype=torch.bool),
        actor_q_valid=torch.ones(batch_size, dtype=torch.bool),
        human_residual_target6=zeros6,
        human_residual_valid=torch.zeros(batch_size, dtype=torch.bool),
    )


def test_residual_critic_td_target_is_ack_only_and_bootstrap_safe() -> None:
    q1, q2 = ConstantQ(0.0), ConstantQ(1.0)
    q1_target, q2_target = ConstantQ(2.0), ConstantQ(3.0)
    target_actor = TargetActor()
    loss = residual_critic_loss(
        q1, q2, q1_target, q2_target, target_actor, batch(), gamma=0.5
    )
    assert torch.isclose(loss, torch.tensor(1.5))
    assert target_actor.calls == 1
    assert q1.batch_sizes == q2.batch_sizes == [2]
    assert q1_target.batch_sizes == q2_target.batch_sizes == [2]
    assert "base_actor" not in inspect.signature(residual_critic_loss).parameters
    assert not any(
        "camera" in name
        for name in inspect.signature(residual_critic_loss).parameters
    )


def test_actor_q_mask_and_invalid_human_residual_are_skipped() -> None:
    q1, q2 = ConstantQ(0.0), ConstantQ(1.0)
    actor = ScalarResidualActor()
    policy = batch(2)
    policy.actor_q_valid = torch.tensor([False, True])
    human = batch(2)
    human.human_residual_valid = torch.tensor([False, False])
    losses = residual_actor_loss(
        q1,
        q2,
        actor,
        policy,
        human,
        actor_q_weight=1.0,
        residual_l2_weight=0.01,
        human_residual_weight=1.0,
    )
    assert losses.actor_q_valid_count == 1
    assert losses.human_residual_valid_count == 0
    assert q1.batch_sizes == q2.batch_sizes == [1]
    assert actor.batch_sizes == [2]
    assert torch.equal(losses.human, torch.zeros_like(losses.human))


def test_online_schedule_is_2q_1actor_and_episode_bounded() -> None:
    policy = OnlineTrainingPolicy()
    assert policy.critic_updates_per_cycle == 2
    assert policy.actor_updates_per_cycle == 1
    assert policy.joint_cycles_for_admission(100) == 2
    assert policy.joint_cycles_for_admission(400) == 7
    assert policy.joint_cycles_for_admission(641) == 10
    assert policy.joint_cycle_budget((100, 400, 641)) == 19
    assert not policy.candidate_due(9)
    assert policy.candidate_due(10)


def test_task_profiles_cannot_override_algorithm_parameters() -> None:
    task2 = load_common_actor_critic_config("task2")
    task3 = load_common_actor_critic_config("task3")
    assert algorithm_hyperparameters(task2) == algorithm_hyperparameters(task3)
    assert task2["task"] != task3["task"]
    assert task2["online_training"] == {
        "critic_updates_per_cycle": 2,
        "actor_updates_per_cycle": 1,
        "max_joint_cycles_per_admitted_episode": 10,
        "actor_candidate_period": 10,
        "checkpoint_period": 50,
        "keep_latest_checkpoints": 2,
    }


def tiny_continuous_learner(*, phase: str, burnin_updates: int = 0):
    learner = learner_server.ContinuousLearner.__new__(
        learner_server.ContinuousLearner
    )
    actor = torch.nn.Linear(2, 2)
    learner.replay_root = Path("/unused")
    learner.replay = None
    learner._materialized_replay_signature = None
    learner.training_policy = OnlineTrainingPolicy()
    learner.learner = {
        "residual_actor": actor,
        "runtime": {
            "phase": phase,
            "critic_burnin_complete": phase == "joint",
            "critic_burnin_updates": burnin_updates,
            "online_joint_cycles": 0,
            "counters": {
                "critic_optimizer_steps": burnin_updates,
                "actor_optimizer_steps": 0,
                "target_polyak_steps": burnin_updates,
            },
            "replay": {
                "critic_td_valid_rows": 0,
                "actor_q_valid_rows": 0,
                "human_residual_valid_rows": 0,
            },
        },
    }
    return learner


def test_collecting_does_not_update_actor_or_critic(monkeypatch) -> None:
    learner = tiny_continuous_learner(phase="collecting")
    actor_before = {
        name: value.detach().clone()
        for name, value in learner.residual_actor.state_dict().items()
    }
    monkeypatch.setattr(
        learner_server.warmup,
        "count_sealed_critic_td_valid_transitions",
        lambda _root: 99,
    )
    result = learner(object())
    assert result["phase"] == "collecting"
    assert result["learner_critic_steps"] == result["learner_actor_steps"] == 0
    assert learner.learner["runtime"]["counters"] == {
        "critic_optimizer_steps": 0,
        "actor_optimizer_steps": 0,
        "target_polyak_steps": 0,
    }
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in learner.residual_actor.state_dict().items()
    )


def test_100_rows_runs_exactly_256_critic_burnin_then_enters_joint(
    monkeypatch,
) -> None:
    learner = tiny_continuous_learner(phase="collecting")
    replay = SimpleNamespace(critic_rows_per_episode=(100,))
    monkeypatch.setattr(
        learner_server.warmup,
        "count_sealed_critic_td_valid_transitions",
        lambda _root: 100,
    )
    monkeypatch.setattr(learner, "_refresh_replay", lambda: replay)
    actor_before = {
        name: value.detach().clone()
        for name, value in learner.residual_actor.state_dict().items()
    }
    calls = []

    def critic_update(_coordinator, _replay, *, burnin):
        assert burnin is True
        calls.append(1)
        runtime = learner.learner["runtime"]
        runtime["critic_burnin_updates"] += 1
        runtime["counters"]["critic_optimizer_steps"] += 1
        runtime["counters"]["target_polyak_steps"] += 1
        return 0.25

    monkeypatch.setattr(learner, "_critic_update", critic_update)
    result = learner(object())
    assert len(calls) == 256
    assert result["critic_burnin_updates"] == 256
    assert result["learner_actor_steps"] == 0
    assert learner.learner["runtime"]["phase"] == "joint"
    assert learner.learner["runtime"]["critic_burnin_complete"] is True
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in learner.residual_actor.state_dict().items()
    )


def test_joint_cycle_is_exactly_two_critic_and_one_actor(monkeypatch) -> None:
    learner = tiny_continuous_learner(phase="joint", burnin_updates=256)
    replay = SimpleNamespace(critic_rows_per_episode=(100,))
    monkeypatch.setattr(
        learner_server.warmup,
        "count_sealed_critic_td_valid_transitions",
        lambda _root: 100,
    )
    monkeypatch.setattr(learner, "_refresh_replay", lambda: replay)
    critic_calls = []
    actor_calls = []
    monkeypatch.setattr(
        learner,
        "_critic_update",
        lambda _coordinator, _replay, *, burnin: critic_calls.append(burnin)
        or 0.5,
    )
    monkeypatch.setattr(
        learner,
        "_actor_update",
        lambda _coordinator, _replay: actor_calls.append(1)
        or {"total": 0.1, "value": -0.2},
    )
    result = learner(object())
    assert critic_calls == [False, False]
    assert actor_calls == [1]
    assert result["learner_critic_steps"] == 2
    assert result["learner_actor_steps"] == 1
    assert result["online_joint_cycle"] == 1
