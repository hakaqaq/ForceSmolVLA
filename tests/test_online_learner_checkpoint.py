from __future__ import annotations

from pathlib import Path

import torch
import yaml

from forcesmolvla.rft.critic import build_twin_q, state_exact
from forcesmolvla.rft.online.actor_learner_runtime import prepare_learner
from forcesmolvla.rft.online.learner_checkpoint import (
    RESIDUAL_CHECKPOINT_FILES,
    residual_checkpoint_is_recoverable,
    save_residual_checkpoint,
)
from forcesmolvla.rft.residual_actor import make_residual_actor_pair


ROOT = Path(__file__).parents[1]


def test_residual_checkpoint_restores_phase_burnin_and_only_requested_state(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/forcerft/actor_critic_common.yaml").read_text()
    )
    actor, actor_target = make_residual_actor_pair(hidden_dim=256)
    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=256, seed=3)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(
        (*q1.parameters(), *q2.parameters()), lr=3e-4
    )
    runtime = {
        "base_actor_checkpoint": "/fixed/base",
        "phase": "critic_burnin",
        "critic_burnin_complete": False,
        "critic_burnin_updates": 137,
        "online_joint_cycles": 0,
        "active_residual_revision": "task3-residual-step-000000",
        "counters": {
            "critic_optimizer_steps": 137,
            "actor_optimizer_steps": 0,
            "target_polyak_steps": 137,
        },
        "replay": {
            "critic_td_valid_rows": 100,
            "actor_q_valid_rows": 80,
            "human_residual_valid_rows": 0,
        },
    }
    checkpoint = tmp_path / "online_actor_critic_cycle_000000"
    save_residual_checkpoint(
        checkpoint,
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

    assert residual_checkpoint_is_recoverable(checkpoint)
    assert {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file()
    } == set(RESIDUAL_CHECKPOINT_FILES)
    assert not (checkpoint / "metadata.json").exists()
    assert not (checkpoint / "manifest.json").exists()

    restored = prepare_learner(torch.device("cpu"), resume_checkpoint=checkpoint)
    assert restored["runtime"] == runtime
    assert state_exact(actor, restored["residual_actor"])
    assert state_exact(q1, restored["q1"])
    assert state_exact(q2_target, restored["q2_target"])

    runtime["phase"] = "joint"
    runtime["critic_burnin_complete"] = True
    runtime["critic_burnin_updates"] = 256
    runtime["counters"]["critic_optimizer_steps"] = 256
    runtime["counters"]["target_polyak_steps"] = 256
    save_residual_checkpoint(
        checkpoint,
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
    resumed_again = prepare_learner(
        torch.device("cpu"), resume_checkpoint=checkpoint
    )
    assert resumed_again["runtime"]["phase"] == "joint"
    assert resumed_again["runtime"]["critic_burnin_updates"] == 256
