from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from forcesmolvla.rft.critic import build_twin_q, state_exact
from forcesmolvla.rft.online.residual_actor_critic_runtime import (
    AsyncRuntimeError,
    ResidualActorCriticSchedule,
    training_checkpoint_path,
    prepare_learner,
    retain_latest_training_checkpoints,
    select_resume_or_bootstrap_checkpoint,
)
from forcesmolvla.rft.online.residual_actor_critic_checkpoint import (
    save_residual_actor_critic_checkpoint,
)
from forcesmolvla.rft.residual_actor import make_residual_actor_pair


ROOT = Path(__file__).parents[1]


def write_checkpoint(path: Path, *, learner_state: str = "ack_replay_collection") -> Path:
    config = yaml.safe_load(
        (ROOT / "configs/forcerft/online_ack_residual_actor_critic.yaml").read_text()
    )
    actor, actor_target = make_residual_actor_pair(hidden_dim=256)
    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=256, seed=4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(
        (*q1.parameters(), *q2.parameters()), lr=3e-4
    )
    warmup = 256 if learner_state == "residual_actor_critic_training" else 0
    runtime = {
        "frozen_base_policy_checkpoint": "/fixed/base",
        "learner_state": learner_state,
        "ack_critic_warmup_complete": learner_state == "residual_actor_critic_training",
        "ack_critic_warmup_steps": warmup,
        "residual_actor_critic_cycles": 0,
        "active_residual_policy_revision": "task3-residual-policy-step-000000",
        "online_adaptation_id": "task3-ack-residual-test",
        "counters": {
            "twin_q_optimizer_steps": warmup,
            "residual_actor_optimizer_steps": 0,
            "twin_q_target_update_steps": warmup,
        },
        "replay": {
            "critic_td_valid_rows": 100 if learner_state == "residual_actor_critic_training" else 0,
            "actor_q_valid_rows": 0,
            "human_residual_valid_rows": 0,
        },
    }
    return save_residual_actor_critic_checkpoint(
        path,
        residual_actor=actor,
        residual_actor_target=actor_target,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        residual_actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        runtime_state=runtime,
        config=config,
    )


def test_final_online_policy_has_fixed_schedule_and_bounded_admission_budget() -> None:
    policy = ResidualActorCriticSchedule()
    assert not policy.training_ready(99) and policy.training_ready(100)
    assert policy.ack_critic_warmup_steps == 256
    assert policy.twin_q_updates_per_cycle == 2
    assert policy.residual_actor_updates_per_cycle == 1
    assert not policy.candidate_due(9) and policy.candidate_due(10)
    assert policy.cycles_for_admission(100) == 2
    assert policy.cycles_for_admission(400) == 7
    assert policy.cycles_for_admission(999) == 10


def test_resume_selection_prefers_latest_final_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "outputs/task3"
    checkpoint_root = root / "online_ack_residual/training_checkpoints"
    first = write_checkpoint(training_checkpoint_path(checkpoint_root, 2))
    latest = write_checkpoint(training_checkpoint_path(checkpoint_root, 7))
    incomplete = training_checkpoint_path(checkpoint_root, 9)
    incomplete.mkdir(parents=True)
    selected = select_resume_or_bootstrap_checkpoint(
        root, configured_bootstrap_checkpoint=None
    )
    assert selected.path == latest.resolve()
    assert selected.kind == "residual_actor_critic_training"
    assert first.exists() and incomplete.exists()


def test_seed_is_used_without_offline_critic_fallback(tmp_path: Path) -> None:
    seed = write_checkpoint(
        tmp_path / "base_policy_zero_residual_random_twin_q"
    )
    selected = select_resume_or_bootstrap_checkpoint(
        tmp_path / "empty-output", configured_bootstrap_checkpoint=seed
    )
    assert selected.path == seed.resolve() and selected.kind == "online_residual_bootstrap"
    with pytest.raises(
        AsyncRuntimeError, match="RESUME_OR_ONLINE_RESIDUAL_BOOTSTRAP_REQUIRED"
    ):
        select_resume_or_bootstrap_checkpoint(
            tmp_path / "empty-output", configured_bootstrap_checkpoint=None
        )


def test_prepare_learner_restores_only_residual_system(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "seed", learner_state="residual_actor_critic_training")
    learner = prepare_learner(torch.device("cpu"), resume_checkpoint=checkpoint)
    assert set(learner["modules"]) == {
        "residual_actor",
        "residual_actor_target",
        "q1",
        "q2",
        "q1_target",
        "q2_target",
    }
    assert learner["runtime"]["learner_state"] == "residual_actor_critic_training"
    assert learner["runtime"]["ack_critic_warmup_steps"] == 256
    assert state_exact(learner["q1"], learner["q1_target"])


def test_checkpoint_retention_keeps_two_latest(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    for cycle in (1, 2, 3):
        write_checkpoint(training_checkpoint_path(root, cycle))
    kept = retain_latest_training_checkpoints(root, keep=2)
    assert [path.name for path in kept] == [
        "residual_actor_critic_cycle_000002",
        "residual_actor_critic_cycle_000003",
    ]
    assert not training_checkpoint_path(root, 1).exists()
