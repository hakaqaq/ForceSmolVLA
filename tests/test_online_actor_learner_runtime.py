from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from forcesmolvla.rft.critic import build_twin_q, state_exact
from forcesmolvla.rft.online.actor_learner_runtime import (
    AsyncRuntimeError,
    OnlineTrainingPolicy,
    online_checkpoint_path,
    prepare_learner,
    retain_latest_online_checkpoints,
    select_resume_or_seed_checkpoint,
)
from forcesmolvla.rft.online.learner_checkpoint import save_residual_checkpoint
from forcesmolvla.rft.residual_actor import make_residual_actor_pair


ROOT = Path(__file__).parents[1]


def write_checkpoint(path: Path, *, phase: str = "collecting") -> Path:
    config = yaml.safe_load(
        (ROOT / "configs/forcerft/actor_critic_common.yaml").read_text()
    )
    actor, actor_target = make_residual_actor_pair(hidden_dim=256)
    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=256, seed=4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(
        (*q1.parameters(), *q2.parameters()), lr=3e-4
    )
    burnin = 256 if phase == "joint" else 0
    runtime = {
        "base_actor_checkpoint": "/fixed/base",
        "phase": phase,
        "critic_burnin_complete": phase == "joint",
        "critic_burnin_updates": burnin,
        "online_joint_cycles": 0,
        "active_residual_revision": "task3-residual-step-000000",
        "counters": {
            "critic_optimizer_steps": burnin,
            "actor_optimizer_steps": 0,
            "target_polyak_steps": burnin,
        },
        "replay": {
            "critic_td_valid_rows": 100 if phase == "joint" else 0,
            "actor_q_valid_rows": 0,
            "human_residual_valid_rows": 0,
        },
    }
    return save_residual_checkpoint(
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
    policy = OnlineTrainingPolicy()
    assert not policy.training_ready(99) and policy.training_ready(100)
    assert policy.critic_burnin_updates == 256
    assert policy.critic_updates_per_cycle == 2
    assert policy.actor_updates_per_cycle == 1
    assert not policy.candidate_due(9) and policy.candidate_due(10)
    assert policy.joint_cycles_for_admission(100) == 2
    assert policy.joint_cycles_for_admission(400) == 7
    assert policy.joint_cycles_for_admission(999) == 10


def test_resume_selection_prefers_latest_final_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "outputs/task3"
    first = write_checkpoint(online_checkpoint_path(root / "online/checkpoints", 2))
    latest = write_checkpoint(online_checkpoint_path(root / "online/checkpoints", 7))
    incomplete = online_checkpoint_path(root / "online/checkpoints", 9)
    incomplete.mkdir(parents=True)
    selected = select_resume_or_seed_checkpoint(
        root, configured_seed_bundle=None
    )
    assert selected.path == latest.resolve()
    assert selected.kind == "online_residual_actor_critic"
    assert first.exists() and incomplete.exists()


def test_seed_is_used_without_offline_critic_fallback(tmp_path: Path) -> None:
    seed = write_checkpoint(tmp_path / "stage3_base_actor_residual_q_cycle_000000")
    selected = select_resume_or_seed_checkpoint(
        tmp_path / "empty-output", configured_seed_bundle=seed
    )
    assert selected.path == seed.resolve() and selected.kind == "stage3_seed"
    with pytest.raises(AsyncRuntimeError, match="RESUME_OR_SAFE_SEED_REQUIRED"):
        select_resume_or_seed_checkpoint(
            tmp_path / "empty-output", configured_seed_bundle=None
        )


def test_prepare_learner_restores_only_residual_system(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "seed", phase="joint")
    learner = prepare_learner(torch.device("cpu"), resume_checkpoint=checkpoint)
    assert set(learner["modules"]) == {
        "residual_actor",
        "residual_actor_target",
        "q1",
        "q2",
        "q1_target",
        "q2_target",
    }
    assert learner["runtime"]["phase"] == "joint"
    assert learner["runtime"]["critic_burnin_updates"] == 256
    assert state_exact(learner["q1"], learner["q1_target"])


def test_checkpoint_retention_keeps_two_latest(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    for cycle in (1, 2, 3):
        write_checkpoint(online_checkpoint_path(root, cycle))
    kept = retain_latest_online_checkpoints(root, keep=2)
    assert [path.name for path in kept] == [
        "online_actor_critic_cycle_000002",
        "online_actor_critic_cycle_000003",
    ]
    assert not online_checkpoint_path(root, 1).exists()
