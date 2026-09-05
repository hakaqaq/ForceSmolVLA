from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from forcesmolvla.rft.critic import build_twin_q, state_exact
from forcesmolvla.rft.online.residual_actor_critic_runtime import (
    AsyncRuntimeError,
    load_checkpoint_training_config,
    prepare_learner,
    require_exact_resume_algorithm_config,
)
from forcesmolvla.rft.online.residual_actor_critic_checkpoint import (
    RESIDUAL_ACTOR_CRITIC_CHECKPOINT_FILES,
    residual_actor_critic_checkpoint_is_recoverable,
    save_residual_actor_critic_checkpoint,
)
from forcesmolvla.rft.residual_actor import make_residual_actor_pair


ROOT = Path(__file__).parents[1]


def test_residual_checkpoint_restores_learner_state_and_warmup_progress(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/forcerft/online_ack_residual_actor_critic.yaml").read_text()
    )
    actor, actor_target = make_residual_actor_pair(hidden_dim=256)
    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=256, seed=3)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(
        (*q1.parameters(), *q2.parameters()), lr=3e-4
    )
    runtime = {
        "frozen_base_policy_checkpoint": "/fixed/base",
        "learner_state": "ack_critic_warmup",
        "ack_critic_warmup_complete": False,
        "ack_critic_warmup_steps": 137,
        "residual_actor_critic_cycles": 0,
        "active_residual_policy_revision": "task3-residual-policy-step-000000",
        "online_adaptation_id": "task3-ack-residual-test",
        "counters": {
            "twin_q_optimizer_steps": 137,
            "residual_actor_optimizer_steps": 0,
            "residual_actor_update_attempts": 0,
            "residual_actor_updates_skipped_no_gradient": 0,
            "twin_q_target_update_steps": 137,
        },
        "replay": {
            "critic_td_valid_rows": 100,
            "actor_q_valid_rows": 80,
            "human_residual_valid_rows": 0,
            "loaded_episode_keys": ["003__episode_000000"],
            "per_episode_critic_row_counts": {
                "003__episode_000000": 100
            },
            "admission_cycle_budgets": {
                "003__episode_000000": 2
            },
            "replay_generation": 1,
        },
    }
    checkpoint = tmp_path / "residual_actor_critic_cycle_000000"
    save_residual_actor_critic_checkpoint(
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

    assert residual_actor_critic_checkpoint_is_recoverable(checkpoint)
    assert {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file()
    } == set(RESIDUAL_ACTOR_CRITIC_CHECKPOINT_FILES)
    assert not (checkpoint / "metadata.json").exists()
    assert not (checkpoint / "manifest.json").exists()

    restored = prepare_learner(torch.device("cpu"), resume_checkpoint=checkpoint)
    assert restored["runtime"] == runtime
    assert state_exact(actor, restored["residual_actor"])
    assert state_exact(q1, restored["q1"])
    assert state_exact(q2_target, restored["q2_target"])

    runtime["learner_state"] = "residual_actor_critic_training"
    runtime["ack_critic_warmup_complete"] = True
    runtime["ack_critic_warmup_steps"] = 256
    runtime["counters"]["twin_q_optimizer_steps"] = 256
    runtime["counters"]["twin_q_target_update_steps"] = 256
    save_residual_actor_critic_checkpoint(
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
    assert resumed_again["runtime"]["learner_state"] == "residual_actor_critic_training"
    assert resumed_again["runtime"]["ack_critic_warmup_steps"] == 256


def test_exact_resume_rejects_current_yaml_algorithm_drift(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/forcerft/online_ack_residual_actor_critic.yaml").read_text()
    )
    actor, actor_target = make_residual_actor_pair(hidden_dim=256)
    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=256, seed=3)
    checkpoint = tmp_path / "residual_actor_critic_cycle_000000"
    save_residual_actor_critic_checkpoint(
        checkpoint,
        residual_actor=actor,
        residual_actor_target=actor_target,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        residual_actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1e-4),
        critic_optimizer=torch.optim.Adam(
            (*q1.parameters(), *q2.parameters()), lr=3e-4
        ),
        runtime_state={
            "frozen_base_policy_checkpoint": "/fixed/base",
            "learner_state": "ack_replay_collection",
            "ack_critic_warmup_complete": False,
            "ack_critic_warmup_steps": 0,
            "residual_actor_critic_cycles": 0,
            "active_residual_policy_revision": "task3-residual-policy-step-000000",
            "online_adaptation_id": "task3-ack-residual-config-test",
            "counters": {
                "twin_q_optimizer_steps": 0,
                "residual_actor_optimizer_steps": 0,
                "residual_actor_update_attempts": 0,
                "residual_actor_updates_skipped_no_gradient": 0,
                "twin_q_target_update_steps": 0,
            },
            "replay": {
                "critic_td_valid_rows": 0,
                "actor_q_valid_rows": 0,
                "human_residual_valid_rows": 0,
            },
        },
        config=config,
    )
    checkpoint_config = load_checkpoint_training_config(checkpoint)
    current_config = deepcopy(config)
    current_config["residual_actor_critic_training"][
        "admitted_rows_per_cycle"
    ] = 32
    current_config["wrist_wrench_residual_actor"][
        "max_normalized_residual"
    ] = 0.3

    with pytest.raises(
        AsyncRuntimeError, match="FORCERFT_EXACT_RESUME_CONFIG_MISMATCH"
    ):
        require_exact_resume_algorithm_config(
            checkpoint_config=checkpoint_config,
            current_config=current_config,
        )

    restored = prepare_learner(
        torch.device("cpu"), resume_checkpoint=checkpoint
    )
    assert restored["training_policy"].admitted_rows_per_cycle == 64
    assert restored["residual_actor"].max_normalized_residual == 0.5
