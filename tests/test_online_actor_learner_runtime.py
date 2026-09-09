from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest
import torch
import yaml

from forcesmolvla.rft.critic import CRITIC_INPUT_SPEC, build_twin_q, state_exact
from forcesmolvla.rft.online.residual_actor_critic_runtime import (
    AsyncRuntimeError,
    InferencePriorityCoordinator,
    ONLINE_ADAPTATION_DIRECTORY_NAME,
    ResidualActorCriticSchedule,
    training_checkpoint_path,
    prepare_learner,
    retain_latest_training_checkpoints,
    select_resume_or_bootstrap_checkpoint,
)
from forcesmolvla.rft.online.residual_actor_critic_checkpoint import (
    BOOTSTRAP_CHECKPOINT_KIND,
    TRAINING_CHECKPOINT_KIND,
    save_residual_actor_critic_checkpoint,
)
from forcesmolvla.rft.online.transition_authority import ONLINE_SEMANTICS_VERSION
from forcesmolvla.rft.residual_actor import make_residual_actor_pair


ROOT = Path(__file__).parents[1]


def write_checkpoint(
    path: Path,
    *,
    learner_state: str = "ack_replay_collection",
    checkpoint_kind: str = TRAINING_CHECKPOINT_KIND,
) -> Path:
    config = yaml.safe_load(
        (ROOT / "configs/forcerft/online_ack_residual_actor_critic.yaml").read_text()
    )
    actor, actor_target = make_residual_actor_pair(
        hidden_dim=256,
        max_normalized_residual=0.1,
        residual_cap6=[0.1] * 6,
    )
    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=256, seed=4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-5)
    critic_optimizer = torch.optim.Adam(
        (*q1.parameters(), *q2.parameters()), lr=3e-4
    )
    warmup = 256 if learner_state == "residual_actor_critic_training" else 0
    episode_counts = (
        {"a": 334, "b": 333, "c": 333}
        if learner_state == "residual_actor_critic_training"
        else {}
    )
    credit_admissions = {
        admission: {
            "episode_id": f"{admission}/episode",
            "critic_td_valid_rows": count,
            "td_uids": [f"{admission}:{index}" for index in range(count)],
        }
        for admission, count in episode_counts.items()
    }
    credited_td_uids = [
        uid
        for record in credit_admissions.values()
        for uid in record["td_uids"]
    ]
    runtime = {
        "checkpoint_kind": checkpoint_kind,
        "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        "critic_input_spec": CRITIC_INPUT_SPEC,
        "frozen_base_policy_checkpoint": "/fixed/base",
        "learner_state": learner_state,
        "ack_critic_warmup_complete": learner_state == "residual_actor_critic_training",
        "ack_critic_warmup_steps": warmup,
        "residual_actor_critic_cycles": 0,
        "partial_cycle_q_updates": 0,
        "active_residual_policy_revision": "task3-residual-policy-step-000000",
        "online_adaptation_id": "task3-ack-residual-test",
        "scheduling": {
            "mode": "continuous_async",
            "last_publish_attempt_cycle": 0,
            "last_published_cycle": 0,
            "publication_event_count": 0,
            "last_periodic_checkpoint_cycle": 0,
            "periodic_checkpoint_event_count": 0,
            "last_saved_checkpoint_cycle": 0,
            "active_publication_cycle": None,
            "active_actor_optimizer_step": 0,
            "active_actor_checkpoint": str(
                path.resolve() / "models/residual_actor.pt"
            ),
            "active_policy_epoch": 0,
            "active_policy_epoch_status": "known",
            "pending_publication": None,
        },
        "counters": {
            "twin_q_optimizer_steps": warmup,
            "residual_actor_optimizer_steps": 0,
            "residual_actor_update_attempts": 0,
            "residual_actor_updates_skipped_no_gradient": 0,
            "twin_q_target_update_steps": warmup,
            "critic_sample_draws": 0,
            "policy_sample_draws": 0,
            "human_sample_draws": 0,
        },
        "replay": {
            "critic_td_valid_rows": len(credited_td_uids),
            "actor_q_valid_rows": 0,
            "human_residual_valid_rows": 0,
            "loaded_episode_keys": list(episode_counts),
            "per_episode_critic_row_counts": episode_counts,
            "replay_generation": len(episode_counts),
            "training_credit_ledger": {
                "schema": "forcesmolvla-td-cycle-credit-ledger-v1",
                "new_td_rows_per_cycle": 8,
                "admissions": credit_admissions,
                "in_flight_cycle": None,
            },
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


def test_final_online_policy_has_continuous_100_1000_schedule() -> None:
    policy = ResidualActorCriticSchedule()
    assert not policy.training_ready(999, 3)
    assert not policy.training_ready(1000, 2)
    assert policy.training_ready(1000, 3)
    assert policy.ack_critic_warmup_steps == 256
    assert policy.twin_q_updates_per_cycle == 2
    assert policy.residual_actor_updates_per_cycle == 1
    assert not policy.candidate_due(99) and policy.candidate_due(100)
    assert not policy.candidate_due(200, last_publish_attempt_cycle=200)
    assert not policy.checkpoint_due(999) and policy.checkpoint_due(1000)
    assert not hasattr(policy, "max_cycles_per_admitted_episode")


def test_resume_selection_prefers_latest_final_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "outputs/task3"
    checkpoint_root = (
        root / ONLINE_ADAPTATION_DIRECTORY_NAME / "training_checkpoints"
    )
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
        tmp_path / "base_policy_zero_residual_random_twin_q",
        checkpoint_kind=BOOTSTRAP_CHECKPOINT_KIND,
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


def test_coordinator_coverage_wait_is_observable_and_cancellable() -> None:
    coordinator = InferencePriorityCoordinator()
    coordinator.begin_actor_window(0.0)
    errors: list[str] = []

    def wait_for_slot() -> None:
        try:
            with coordinator.learner_step_slot("critic"):
                raise AssertionError("cancelled waiter must not enter slot")
        except AsyncRuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=wait_for_slot)
    thread.start()
    deadline = time.monotonic() + 1.0
    while (
        coordinator.status_snapshot()["learner_wait_reason"] is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert coordinator.status_snapshot()["learner_wait_reason"] == (
        "insufficient_action_coverage"
    )
    coordinator.cancel_waiters()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert errors == ["ONLINE_REPLAY_ASYNC_LEARNER_CANCELLED"]
