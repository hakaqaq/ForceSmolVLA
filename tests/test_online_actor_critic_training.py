from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import serve_forcerft_residual_actor_critic as learner_server  # noqa: E402

from forcesmolvla.rft.online.residual_actor_critic_runtime import ResidualActorCriticSchedule
from forcesmolvla.rft.online.replay_training import (
    ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
    LEGACY_ACK_RESIDUAL_TRANSITION_SCHEMA_VERSIONS,
    OnlineResidualReplay,
    ProductionAckMacro,
    algorithm_hyperparameters,
    load_common_actor_critic_config,
)
from forcesmolvla.rft.online.transition_authority import (
    AckMacro,
    ActorQEligibility,
)
from forcesmolvla.rft.critic import (
    RESIDUAL_ACTION_OFFSET,
    RESIDUAL_ACTION_WIDTH,
    build_twin_q,
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


class IdentityTransform:
    def apply(self, value):
        return np.asarray(value)


def human_replay(*, terminated: bool = True) -> OnlineResidualReplay:
    observation = {
        "state7_absolute": [0.0] * 7,
        "wrench6_calibrated_tcp": [0.0] * 6,
        "materialized_timestamp_monotonic_ns": 1_000_000_000,
    }
    next_observation = {
        **observation,
        "materialized_timestamp_monotonic_ns": 1_100_000_000,
    }
    accepted = np.repeat(
        np.asarray([[0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]]),
        3,
        axis=0,
    )
    behavior = AckMacro(
        grid_monotonic_ns=(1_000_000_000, 1_033_333_333, 1_066_666_667),
        ack_ids=("a", "a", "a"),
        gripper_command_ids=("g", "g", "g"),
        gripper_ack_command_ids=("g", "g", "g"),
        accepted_absolute_action_k7=accepted,
        slot_owner=("human_intervention",) * 3,
        workspace_clip_flags=(False,) * 3,
    )
    transition = {
        "identity": {"episode_id": "human-episode"},
        "action_source": "human",
        "observation": observation,
        "next_observation": next_observation,
        "outcome": {
            "reward": 1.0,
            "terminated": terminated,
            "truncated": False,
        },
        "eligibility": {"actor_q_valid": True},
        "human_residual_valid": True,
        "pre_takeover_base_absolute_action7": [
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    }
    macro = ProductionAckMacro(
        transition=transition,
        behavior=behavior,
        next_grid_monotonic_ns=1_100_000_000,
        ack_provenance=(),
        actor_q_eligibility=ActorQEligibility(True, "valid"),
    )
    normalizer = SimpleNamespace(
        state7=IdentityTransform(),
        wrench6=IdentityTransform(),
        delta_action7=IdentityTransform(),
    )
    return OnlineResidualReplay((macro,), normalizer)


def policy_replay(*, schema_version: str, base_action: object) -> OnlineResidualReplay:
    observation = {
        "state7_absolute": [0.0] * 7,
        "wrench6_calibrated_tcp": [0.0] * 6,
        "materialized_timestamp_monotonic_ns": 1_000_000_000,
    }
    accepted = np.repeat(
        np.asarray([[0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]]), 3, axis=0
    )
    transition = {
        "schema_version": schema_version,
        "identity": {"episode_id": "policy-episode"},
        "action_source": "policy",
        "observation": observation,
        "next_observation": {
            **observation,
            "materialized_timestamp_monotonic_ns": 1_100_000_000,
        },
        "outcome": {"reward": 0.0, "terminated": True, "truncated": False},
        "eligibility": {"actor_q_valid": True},
    }
    if base_action is not None:
        transition["base_normalized_action_k7"] = base_action
    macro = ProductionAckMacro(
        transition=transition,
        behavior=AckMacro(
            grid_monotonic_ns=(1_000_000_000, 1_033_333_333, 1_066_666_667),
            ack_ids=("a", "a", "a"),
            gripper_command_ids=("g", "g", "g"),
            gripper_ack_command_ids=("g", "g", "g"),
            accepted_absolute_action_k7=accepted,
            slot_owner=("policy",) * 3,
            workspace_clip_flags=(False,) * 3,
        ),
        next_grid_monotonic_ns=1_100_000_000,
        ack_provenance=(),
        actor_q_eligibility=ActorQEligibility(True, "valid"),
    )
    normalizer = SimpleNamespace(
        state7=IdentityTransform(),
        wrench6=IdentityTransform(),
        delta_action7=IdentityTransform(),
    )
    return OnlineResidualReplay((macro,), normalizer)


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
        next_base_valid=torch.ones(batch_size, dtype=torch.bool),
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


def test_valid_human_residual_reaches_critic_and_unlocks_action_columns() -> None:
    replay = human_replay()
    row = replay.rows[0]
    assert row["human_residual_valid"] is True
    assert np.count_nonzero(row["behavior_residual_k6"]) > 0
    assert np.count_nonzero(row["human_residual_target6"]) > 0

    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=16, seed=13)
    target_actor = TargetActor()
    optimizer = torch.optim.Adam((*q1.parameters(), *q2.parameters()), lr=3e-4)
    before = q1.layers[0].weight[
        :, RESIDUAL_ACTION_OFFSET : RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH
    ].detach().clone()
    critic_batch = replay.sample(8, device=torch.device("cpu"), seed=1)
    assert critic_batch is not None
    optimizer.zero_grad(set_to_none=True)
    residual_critic_loss(
        q1, q2, q1_target, q2_target, target_actor, critic_batch, gamma=0.99
    ).backward()
    optimizer.step()
    after = q1.layers[0].weight[
        :, RESIDUAL_ACTION_OFFSET : RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH
    ].detach()
    assert not torch.equal(before, after)


def test_policy_value_sampling_excludes_human_and_missing_next_base() -> None:
    replay = human_replay()
    assert replay.sample(
        1,
        device=torch.device("cpu"),
        seed=0,
        policy_only=True,
        actor_q_valid_only=True,
    ) is None
    assert replay.sample(
        1, device=torch.device("cpu"), seed=0, human_only=True
    ) is not None

    missing_next_base = human_replay(terminated=False)
    assert missing_next_base.rows == ()
    assert missing_next_base.next_base_missing_rows == 1


def test_only_explicit_legacy_policy_rows_fallback_to_zero_residual() -> None:
    legacy_schema = next(iter(LEGACY_ACK_RESIDUAL_TRANSITION_SCHEMA_VERSIONS))
    legacy = policy_replay(schema_version=legacy_schema, base_action=None)
    assert legacy.critic_td_valid_rows == 1
    assert legacy.nonzero_behavior_residual_rows == 0
    assert legacy.quarantined_current_schema_rows == 0

    missing_current = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=None,
    )
    assert missing_current.rows == ()
    assert missing_current.quarantined_current_schema_rows == 1

    corrupted_current = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=[[float("nan")] * 7 for _ in range(3)],
    )
    assert corrupted_current.rows == ()
    assert corrupted_current.quarantined_current_schema_rows == 1

    current = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=[[0.0] * 7 for _ in range(3)],
    )
    assert current.critic_td_valid_rows == 1
    assert current.nonzero_behavior_residual_rows == 1


def test_replay_sampling_is_without_replacement_when_population_is_large_enough() -> None:
    replay = policy_replay(
        schema_version=ACK_RESIDUAL_TRANSITION_SCHEMA_VERSION,
        base_action=[[0.0] * 7 for _ in range(3)],
    )
    prototype = replay.rows[0]
    replay.rows = tuple(
        {**prototype, "state7": np.full(7, index, dtype=np.float32)}
        for index in range(8)
    )
    sampled = replay.sample(8, device=torch.device("cpu"), seed=7)
    assert sampled is not None
    assert len(set(sampled.state7[:, 0].tolist())) == 8

    replay.rows = tuple(
        {
            **prototype,
            "episode_id": episode_id,
            "state7": np.full(7, value, dtype=np.float32),
        }
        for episode_id, value, count in (
            ("short", 0.0, 20),
            ("long", 1.0, 100),
        )
        for _ in range(count)
    )
    balanced = replay.sample(10, device=torch.device("cpu"), seed=7)
    assert balanced is not None
    assert balanced.state7[:, 0].tolist().count(0.0) == 5
    assert balanced.state7[:, 0].tolist().count(1.0) == 5


def test_online_schedule_is_2q_1actor_and_episode_bounded() -> None:
    policy = ResidualActorCriticSchedule()
    assert policy.twin_q_updates_per_cycle == 2
    assert policy.residual_actor_updates_per_cycle == 1
    assert policy.cycles_for_admission(100) == 2
    assert policy.cycles_for_admission(400) == 7
    assert policy.cycles_for_admission(641) == 10
    assert policy.cycles_for_observed_admission(
        new_critic_td_valid_rows=99,
        total_critic_td_valid_rows=99,
    ) == 0
    assert policy.cycles_for_observed_admission(
        new_critic_td_valid_rows=1,
        total_critic_td_valid_rows=100,
    ) == 1
    assert policy.residual_actor_critic_cycle_budget((100, 400, 641)) == 19
    assert not policy.candidate_due(9)
    assert policy.candidate_due(10)
    assert ResidualActorCriticSchedule(
        admitted_rows_per_cycle=32,
        max_cycles_per_admitted_episode=20,
    ).cycles_for_admission(400) == 13


def test_task_profiles_cannot_override_algorithm_parameters() -> None:
    task2 = load_common_actor_critic_config("task2")
    task3 = load_common_actor_critic_config("task3")
    assert algorithm_hyperparameters(task2) == algorithm_hyperparameters(task3)
    assert task2["task"] != task3["task"]
    assert task2["residual_actor_critic_training"] == {
        "admitted_rows_per_cycle": 64,
        "twin_q_updates_per_cycle": 2,
        "residual_actor_updates_per_cycle": 1,
        "max_cycles_per_admitted_episode": 10,
        "residual_candidate_interval_actor_steps": 10,
        "training_checkpoint_interval_cycles": 20,
        "retained_training_checkpoint_count": 10,
        "checkpoint_on_warmup_complete": True,
        "checkpoint_on_candidate_activation": True,
    }


def tiny_continuous_learner(*, learner_state: str, warmup_updates: int = 0):
    learner = learner_server.ResidualActorCriticLearner.__new__(
        learner_server.ResidualActorCriticLearner
    )
    actor = torch.nn.Linear(2, 2)
    learner.replay_root = Path("/unused")
    learner.replay = None
    learner.training_policy = ResidualActorCriticSchedule(
        checkpoint_on_warmup_complete=False,
        checkpoint_on_candidate_activation=False,
    )
    learner._loaded_episode_keys = set()
    learner._admission_progress = {}
    learner._expected_admission_id = None
    learner._joint_cycle_budget = 0
    learner.latest_replay_refresh_ms = 0.0
    learner.latest_critic_update_ms = 0.0
    learner.latest_actor_update_ms = 0.0
    learner.latest_cycle_ms = 0.0
    learner.learner = {
        "residual_actor": actor,
        "runtime": {
            "learner_state": learner_state,
            "ack_critic_warmup_complete": learner_state == "residual_actor_critic_training",
            "ack_critic_warmup_steps": warmup_updates,
            "residual_actor_critic_cycles": 0,
            "counters": {
                "twin_q_optimizer_steps": warmup_updates,
                "residual_actor_optimizer_steps": 0,
                "twin_q_target_update_steps": warmup_updates,
            },
            "replay": {
                "critic_td_valid_rows": 0,
                "actor_q_valid_rows": 0,
                "human_residual_valid_rows": 0,
            },
        },
    }
    return learner


def test_replay_refresh_loads_only_newly_sealed_episodes(monkeypatch) -> None:
    learner = tiny_continuous_learner(learner_state="ack_replay_collection")
    learner.normalizer = object()
    learner.current_session_id = None
    learner.unique_r_count = 0
    learner.r_macro_count = 0
    learner.next_base_missing_rows = 0
    learner.quarantined_current_schema_rows = 0
    learner.nonzero_behavior_residual_rows = 0
    signatures = [["a"]]
    monkeypatch.setattr(
        learner, "_episode_signature", lambda: tuple(signatures[0])
    )

    class FakeReplay:
        def __init__(self, _macros, _normalizer) -> None:
            self.counts: list[int] = []
            self.next_base_missing_rows = 0
            self.quarantined_current_schema_rows = 0
            self.nonzero_behavior_residual_rows = 0

        def append_macros(self, macros):
            macro = tuple(macros)[0]
            episode_id = macro.transition["identity"]["episode_id"]
            count = int(macro.transition["materialized_count"])
            self.counts.append(count)
            return {episode_id: count}

        @property
        def critic_rows_per_episode(self):
            return tuple(self.counts)

        @property
        def critic_td_valid_rows(self):
            return sum(self.counts)

        actor_q_valid_rows = property(lambda self: sum(self.counts))
        human_residual_valid_rows = property(lambda _self: 0)

    calls: list[str] = []

    def load_episode(_root, admission_id):
        calls.append(admission_id)
        episode_id = f"{admission_id}/episode"
        row = {
            "identity": {"episode_id": episode_id, "session_id": "old"},
            "materialized_count": {"a": 99, "b": 1, "c": 400}[admission_id],
        }
        macro = SimpleNamespace(transition=row)
        return [row], (macro,), {episode_id: Path("episode")}, []

    monkeypatch.setattr(learner_server.warmup, "OnlineResidualReplay", FakeReplay)
    monkeypatch.setattr(
        learner_server.warmup, "load_formal_online_episode", load_episode
    )
    monkeypatch.setattr(
        learner_server.warmup,
        "load_formal_online_r",
        lambda _root: (_ for _ in ()).throw(AssertionError("full reload")),
    )
    monkeypatch.setattr(
        learner_server.warmup, "build_ack_macros", lambda _rows: ()
    )

    learner._refresh_replay()
    assert calls == ["a"]
    assert learner.admission_budget_status("a")["computed_cycle_budget"] == 0
    signatures[0].append("b")
    learner._refresh_replay()
    assert learner.admission_budget_status("b")["computed_cycle_budget"] == 1
    learner.learner["runtime"]["residual_actor_critic_cycles"] = 1
    signatures[0].append("c")
    learner.expect_admission("c")
    learner._refresh_replay()
    assert calls == ["a", "b", "c"]
    assert learner.learner["runtime"]["replay"]["loaded_episode_keys"] == [
        "a",
        "b",
        "c",
    ]
    assert learner.admission_budget_status("c") == {
        "episode_key": "c",
        "admitted_rows_for_latest_episode": 400,
        "computed_cycle_budget": 7,
        "cycle_count_at_admission_start": 1,
        "target_cycle_count_after_admission": 8,
        "completed_cycle_count_for_latest_admission": 0,
        "remaining_cycle_budget": 7,
    }


def test_collecting_does_not_update_actor_or_critic(monkeypatch) -> None:
    learner = tiny_continuous_learner(learner_state="ack_replay_collection")
    actor_before = {
        name: value.detach().clone()
        for name, value in learner.residual_actor.state_dict().items()
    }
    monkeypatch.setattr(
        learner,
        "_refresh_replay",
        lambda: SimpleNamespace(critic_td_valid_rows=99),
    )
    result = learner(object())
    assert result["learner_state"] == "ack_replay_collection"
    assert result["learner_critic_steps"] == result["learner_actor_steps"] == 0
    assert learner.learner["runtime"]["counters"] == {
        "twin_q_optimizer_steps": 0,
        "residual_actor_optimizer_steps": 0,
        "twin_q_target_update_steps": 0,
    }
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in learner.residual_actor.state_dict().items()
    )


def test_100_rows_runs_exactly_256_critic_warmup_then_starts_residual_training(
    monkeypatch,
) -> None:
    learner = tiny_continuous_learner(learner_state="ack_replay_collection")
    learner.training_policy = ResidualActorCriticSchedule(
        checkpoint_on_warmup_complete=True,
        checkpoint_on_candidate_activation=False,
    )
    replay = SimpleNamespace(
        critic_td_valid_rows=100, critic_rows_per_episode=(100,)
    )
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

    def critic_update(_coordinator, _replay, *, warmup):
        assert warmup is True
        calls.append(1)
        runtime = learner.learner["runtime"]
        runtime["ack_critic_warmup_steps"] += 1
        runtime["counters"]["twin_q_optimizer_steps"] += 1
        runtime["counters"]["twin_q_target_update_steps"] += 1
        return 0.25

    monkeypatch.setattr(learner, "_critic_update", critic_update)
    checkpoint_calls = []
    monkeypatch.setattr(
        learner,
        "save_checkpoint",
        lambda: checkpoint_calls.append(1) or Path("warmup-checkpoint"),
    )
    result = learner(object())
    assert len(calls) == 256
    assert result["ack_critic_warmup_steps"] == 256
    assert result["learner_actor_steps"] == 0
    assert learner.learner["runtime"]["learner_state"] == "residual_actor_critic_training"
    assert learner.learner["runtime"]["ack_critic_warmup_complete"] is True
    assert checkpoint_calls == [1]
    assert result["latest_checkpoint_path"] == "warmup-checkpoint"
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in learner.residual_actor.state_dict().items()
    )


def test_residual_training_cycle_is_exactly_two_critic_and_one_actor(
    monkeypatch,
) -> None:
    learner = tiny_continuous_learner(
        learner_state="residual_actor_critic_training", warmup_updates=256
    )
    replay = SimpleNamespace(
        critic_td_valid_rows=100, critic_rows_per_episode=(100,)
    )
    monkeypatch.setattr(
        learner_server.warmup,
        "count_sealed_critic_td_valid_transitions",
        lambda _root: 100,
    )
    monkeypatch.setattr(learner, "_refresh_replay", lambda: replay)
    learner._joint_cycle_budget = 1
    critic_calls = []
    actor_calls = []
    monkeypatch.setattr(
        learner,
        "_critic_update",
        lambda _coordinator, _replay, *, warmup: critic_calls.append(warmup)
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
    assert result["residual_actor_critic_cycle"] == 1
