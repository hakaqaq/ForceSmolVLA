from __future__ import annotations

from copy import deepcopy

import pytest

from forcesmolvla.rft.online.learner_checkpoint import (
    OnlineCheckpointSchemaError,
    cpu_round_trip_online_checkpoint,
    validate_online_checkpoint_metadata,
)


SHA = "a" * 64


def state_ref(name: str) -> dict:
    return {"relative_path": f"state/{name}.pt", "sha256": SHA}


def checkpoint_payload() -> dict:
    return {
        "schema_version": "forcesmolvla_stage3_online_checkpoint.v1",
        "checkpoint_id": "checkpoint-2",
        "boundary": {
            "episode_sealed": True, "learner_update_committed": True,
            "pending_graphs": 0, "pending_optimizer_steps": 0,
        },
        "parent": {
            "binding_status": "approved_hybrid",
            "cross_stage_optimizer_rebuilt": True,
        },
        "models": {name: state_ref(name) for name in ("actor", "q1", "q2", "q1_target", "q2_target")},
        "optimizers": {name: state_ref(f"{name}_optimizer") for name in ("actor", "critic")},
        "schedulers": {name: state_ref(f"{name}_scheduler") for name in ("actor", "critic")},
        "rng": state_ref("rng"),
        "samplers": state_ref("samplers"),
        "replay": {
            "canonical_index_sha256": SHA, "R_watermark": 2, "D_watermark": 1,
            "wal_committed_offset": 2, "episode_finalization_state": "sealed",
            "outbox_cursor": 2,
        },
        "credits": {"minted": 2, "consumed": 2, "available": 0},
        "publication": {
            "active_revision": "r0", "pending_revision": None,
            "previous_revision": None, "policy_epoch": 0,
        },
        "counters": {
            "learner_cycles": 2, "critic_updates": 4, "actor_updates": 2,
            "polyak_updates_per_target": 4, "publication_count": 0,
        },
        "bindings": {"source_tree_sha256": SHA, "action_contract_sha256": SHA},
        "authorization": {"deployment_release": False, "robot_execution": False},
    }


def test_checkpoint_schema_and_cpu_json_round_trip() -> None:
    value = checkpoint_payload()
    assert validate_online_checkpoint_metadata(value) == value
    decoded, encoded = cpu_round_trip_online_checkpoint(value)
    assert decoded == value and isinstance(encoded, bytes)


def test_checkpoint_fails_nonboundary_and_counter_or_credit_drift() -> None:
    pending = checkpoint_payload(); pending["boundary"]["pending_optimizer_steps"] = 1
    with pytest.raises(OnlineCheckpointSchemaError, match="SCHEMA"):
        validate_online_checkpoint_metadata(pending)
    counter = checkpoint_payload(); counter["counters"]["critic_updates"] = 3
    with pytest.raises(OnlineCheckpointSchemaError, match="CRITIC_COUNTER"):
        validate_online_checkpoint_metadata(counter)
    credit = checkpoint_payload(); credit["credits"]["available"] = 1
    with pytest.raises(OnlineCheckpointSchemaError, match="CREDIT_LEDGER"):
        validate_online_checkpoint_metadata(credit)
