from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from forcesmolvla.rft.critic import CRITIC_INPUT_SPEC, build_twin_q, state_exact
from forcesmolvla.rft.online.residual_actor_critic_runtime import (
    AsyncRuntimeError,
    load_checkpoint_training_config,
    prepare_learner,
    require_exact_resume_algorithm_config,
)
from forcesmolvla.rft.online.residual_actor_critic_checkpoint import (
    OnlineCheckpointSchemaError,
    RESIDUAL_ACTOR_CRITIC_CHECKPOINT_FILES,
    TRAINING_CHECKPOINT_KIND,
    residual_actor_critic_checkpoint_is_recoverable,
    save_residual_actor_critic_checkpoint,
)
from forcesmolvla.rft.online.sample_credit import TdCycleCreditLedger
from forcesmolvla.rft.online.transition_authority import ONLINE_SEMANTICS_VERSION
from forcesmolvla.rft.residual_actor import make_residual_actor_pair


ROOT = Path(__file__).parents[1]


def _credit_replay(per_episode_counts: dict[str, int]) -> dict:
    ledger = TdCycleCreditLedger(new_td_rows_per_cycle=8)
    for admission_id, count in per_episode_counts.items():
        ledger.register_admission(
            admission_id=admission_id,
            episode_id=admission_id,
            td_uids={f"{admission_id}:td:{index}" for index in range(count)},
        )
    total = sum(per_episode_counts.values())
    return {
        "critic_td_valid_rows": total,
        "actor_q_valid_rows": total,
        "human_residual_valid_rows": 0,
        "loaded_episode_keys": list(per_episode_counts),
        "per_episode_critic_row_counts": dict(per_episode_counts),
        "replay_generation": len(per_episode_counts),
        "training_credit_ledger": ledger.state_dict(),
    }


def _counters(*, q_steps: int = 0) -> dict:
    return {
        "twin_q_optimizer_steps": q_steps,
        "residual_actor_optimizer_steps": 0,
        "residual_actor_update_attempts": 0,
        "residual_actor_updates_skipped_no_gradient": 0,
        "twin_q_target_update_steps": q_steps,
        "critic_sample_draws": 0,
        "policy_sample_draws": 0,
        "human_sample_draws": 0,
    }


def test_residual_checkpoint_restores_learner_state_and_warmup_progress(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/forcerft/online_ack_residual_actor_critic.yaml").read_text()
    )
    actor, actor_target = make_residual_actor_pair(
        hidden_dim=256, max_normalized_residual=0.1
    )
    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=256, seed=3)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(
        (*q1.parameters(), *q2.parameters()), lr=3e-4
    )
    runtime = {
        "checkpoint_kind": TRAINING_CHECKPOINT_KIND,
        "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        "critic_input_spec": CRITIC_INPUT_SPEC,
        "frozen_base_policy_checkpoint": "/fixed/base",
        "learner_state": "ack_critic_warmup",
        "ack_critic_warmup_complete": False,
        "ack_critic_warmup_steps": 137,
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
                (
                    tmp_path
                    / "residual_actor_critic_cycle_000000/models/residual_actor.pt"
                ).resolve()
            ),
            "active_policy_epoch": 0,
            "active_policy_epoch_status": "known",
            "pending_publication": None,
        },
        "counters": _counters(q_steps=137),
        "replay": _credit_replay(
            {
                "001__episode_000000": 333,
                "002__episode_000000": 333,
                "003__episode_000000": 334,
            }
        ),
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
    assert residual_actor_critic_checkpoint_is_recoverable(
        checkpoint, expected_kind=TRAINING_CHECKPOINT_KIND
    )
    assert not residual_actor_critic_checkpoint_is_recoverable(
        checkpoint, expected_kind="online_residual_bootstrap"
    )
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
    actor, actor_target = make_residual_actor_pair(
        hidden_dim=256, max_normalized_residual=0.1
    )
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
            "checkpoint_kind": TRAINING_CHECKPOINT_KIND,
            "online_semantics_version": ONLINE_SEMANTICS_VERSION,
            "critic_input_spec": CRITIC_INPUT_SPEC,
            "frozen_base_policy_checkpoint": "/fixed/base",
            "learner_state": "ack_replay_collection",
            "ack_critic_warmup_complete": False,
            "ack_critic_warmup_steps": 0,
            "residual_actor_critic_cycles": 0,
            "partial_cycle_q_updates": 0,
            "active_residual_policy_revision": "task3-residual-policy-step-000000",
            "online_adaptation_id": "task3-ack-residual-config-test",
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
                    (checkpoint / "models/residual_actor.pt").resolve()
                ),
                "active_policy_epoch": 0,
                "active_policy_epoch_status": "known",
                "pending_publication": None,
            },
            "counters": _counters(),
            "replay": _credit_replay({}),
        },
        config=config,
    )
    checkpoint_config = load_checkpoint_training_config(checkpoint)
    current_config = deepcopy(config)
    current_config["residual_actor_critic_training"][
        "residual_candidate_interval_cycles"
    ] = 50
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
    assert restored["training_policy"].residual_candidate_interval_cycles == 100
    assert restored["residual_actor"].max_normalized_residual == 0.1

    state_path = checkpoint / "state/runtime_state.pt"
    original = torch.load(state_path, map_location="cpu", weights_only=False)
    incompatible = deepcopy(original)
    incompatible["online_semantics_version"] = "old-semantics"
    torch.save(incompatible, state_path)
    assert not residual_actor_critic_checkpoint_is_recoverable(
        checkpoint, expected_kind=TRAINING_CHECKPOINT_KIND
    )

    missing_pending_dependency = deepcopy(original)
    missing_pending_dependency["scheduling"]["pending_publication"] = {
        "revision_id": "task3-residual-policy-cycle-000000-gmissing",
        "checkpoint": str(tmp_path / "missing-candidate"),
        "residual_actor_critic_cycle": 0,
        "residual_actor_optimizer_steps": 0,
    }
    torch.save(missing_pending_dependency, state_path)
    assert not residual_actor_critic_checkpoint_is_recoverable(
        checkpoint, expected_kind=TRAINING_CHECKPOINT_KIND
    )

    missing_counter = deepcopy(original)
    del missing_counter["counters"]["residual_actor_update_attempts"]
    torch.save(missing_counter, state_path)
    assert not residual_actor_critic_checkpoint_is_recoverable(
        checkpoint, expected_kind=TRAINING_CHECKPOINT_KIND
    )

    missing_critic_spec = deepcopy(original)
    del missing_critic_spec["critic_input_spec"]
    torch.save(missing_critic_spec, state_path)
    assert not residual_actor_critic_checkpoint_is_recoverable(
        checkpoint, expected_kind=TRAINING_CHECKPOINT_KIND
    )

    inconsistent = deepcopy(original)
    inconsistent["counters"]["residual_actor_update_attempts"] = 1
    torch.save(inconsistent, state_path)
    assert not residual_actor_critic_checkpoint_is_recoverable(
        checkpoint, expected_kind=TRAINING_CHECKPOINT_KIND
    )


@pytest.mark.parametrize("partial_q", [0, 1, 2])
def test_exact_resume_preserves_reserved_partial_cycle(
    tmp_path: Path, partial_q: int
) -> None:
    config = yaml.safe_load(
        (ROOT / "configs/forcerft/online_ack_residual_actor_critic.yaml").read_text()
    )
    checkpoint = tmp_path / f"partial-{partial_q}"
    actor, actor_target = make_residual_actor_pair(
        hidden_dim=256, max_normalized_residual=0.1
    )
    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=256, seed=3)
    replay = _credit_replay(
        {
            "001__episode_000000": 333,
            "002__episode_000000": 333,
            "003__episode_000000": 334,
        }
    )
    ledger = TdCycleCreditLedger.from_state_dict(
        replay["training_credit_ledger"]
    )
    assert ledger.reserve_cycle(completed_cycles=0)
    replay["training_credit_ledger"] = ledger.state_dict()
    runtime = {
        "checkpoint_kind": TRAINING_CHECKPOINT_KIND,
        "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        "critic_input_spec": CRITIC_INPUT_SPEC,
        "frozen_base_policy_checkpoint": "/fixed/base",
        "learner_state": "residual_actor_critic_training",
        "ack_critic_warmup_complete": True,
        "ack_critic_warmup_steps": 256,
        "residual_actor_critic_cycles": 0,
        "partial_cycle_q_updates": partial_q,
        "active_residual_policy_revision": "task3-residual-policy-step-000000",
        "online_adaptation_id": "task3-partial-resume-test",
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
                (checkpoint / "models/residual_actor.pt").resolve()
            ),
            "active_policy_epoch": 0,
            "active_policy_epoch_status": "known",
            "pending_publication": None,
        },
        "counters": _counters(q_steps=256 + partial_q),
        "replay": replay,
    }
    save_residual_actor_critic_checkpoint(
        checkpoint,
        residual_actor=actor,
        residual_actor_target=actor_target,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        residual_actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-5),
        critic_optimizer=torch.optim.Adam(
            (*q1.parameters(), *q2.parameters()), lr=3e-4
        ),
        runtime_state=runtime,
        config=config,
    )
    restored = prepare_learner(
        torch.device("cpu"), resume_checkpoint=checkpoint
    )["runtime"]
    assert restored["partial_cycle_q_updates"] == partial_q
    assert restored["replay"]["training_credit_ledger"][
        "in_flight_cycle"
    ] == 1

def test_legacy_checkpoint_without_td_credit_cannot_be_saved_or_resumed(
    tmp_path: Path,
) -> None:
    online_root = tmp_path / "online_ack_residual_filter_leash"
    source = online_root / "training_checkpoints/legacy_cycle_000003"
    config = yaml.safe_load(
        (ROOT / "configs/forcerft/online_ack_residual_actor_critic.yaml").read_text()
    )
    actor, actor_target = make_residual_actor_pair(hidden_dim=256)
    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=256, seed=3)
    runtime = {
        "checkpoint_kind": TRAINING_CHECKPOINT_KIND,
        "online_semantics_version": ONLINE_SEMANTICS_VERSION,
        "critic_input_spec": CRITIC_INPUT_SPEC,
        "frozen_base_policy_checkpoint": "/fixed/base",
        "learner_state": "residual_actor_critic_training",
        "ack_critic_warmup_complete": True,
        "ack_critic_warmup_steps": 256,
        "residual_actor_critic_cycles": 3,
        "active_residual_policy_revision": "task3-residual-policy-step-000000",
        "online_adaptation_id": "task3-migration-test",
        "counters": {
            "twin_q_optimizer_steps": 262,
            "residual_actor_optimizer_steps": 2,
            "residual_actor_update_attempts": 3,
            "residual_actor_updates_skipped_no_gradient": 1,
            "twin_q_target_update_steps": 262,
        },
        "replay": {
            "critic_td_valid_rows": 123,
            "actor_q_valid_rows": 100,
            "human_residual_valid_rows": 7,
            "loaded_episode_keys": ["001__episode_000000"],
            "per_episode_critic_row_counts": {"001__episode_000000": 123},
            "admission_cycle_budgets": {"001__episode_000000": 2},
            "replay_generation": 1,
        },
    }
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam((*q1.parameters(), *q2.parameters()), lr=3e-4)
    with pytest.raises(
        OnlineCheckpointSchemaError, match="FORCERFT_CHECKPOINT_WRITE_INCOMPLETE"
    ):
        save_residual_actor_critic_checkpoint(
            source,
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
    assert not residual_actor_critic_checkpoint_is_recoverable(source)
