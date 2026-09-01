from __future__ import annotations

from copy import deepcopy

import pytest

from forcesmolvla.rft.online.transition_authority import (
    AcceptedAck,
    TransitionContractError,
    canonical_payload_sha256,
    finalize_ack_transition,
    validate_ack_transition,
    validate_reward_terminal,
)


SHA = "a" * 64


def observation(identifier: str, episode: str = "episode-1", timestamp: int = 100) -> dict:
    camera = {
        "blob_reference": f"blob:{identifier}",
        "sha256": SHA,
        "timestamp_monotonic_ns": timestamp,
        "age_ms": 0.0,
        "valid": True,
    }
    return {
        "observation_id": identifier,
        "episode_id": episode,
        "timestamp_monotonic_ns": timestamp,
        "state7": [0.0] * 7,
        "wrench6": [0.0] * 6,
        "camera1": camera,
        "camera2": camera,
        "valid": True,
    }


def transition_payload(*, owner: str = "policy", macro_index: int = 0) -> dict:
    expert = owner in {"human_intervention", "offline_demonstration"}
    return {
        "schema_version": "forcesmolvla_stage3_ack_transition.v1",
        "identity": {
            "run_id": "run-1", "session_id": "session-1", "episode_id": "episode-1",
            "macro_index": macro_index, "task": "task",
        },
        "bindings": {
            "action_contract_sha256": SHA, "normalizer_sha256": SHA,
            "calibration_sha256": SHA, "wrench_contract_sha256": SHA,
            "rulespec_sha256": SHA, "reward_contract_sha256": SHA,
            "deployment_binding_sha256": SHA, "source_tree_sha256": SHA,
        },
        "observation": observation(f"obs-{macro_index}"),
        "next_observation": observation(f"next-{macro_index}", timestamp=200),
        "policy_proposal": {
            "revision_id": "revision-1", "model_sha256": SHA, "policy_epoch": 0,
            "request_id": f"request-{macro_index}", "chunk_id": f"chunk-{macro_index}",
            "action_h50_sha256": SHA, "flow_noise_sha256": SHA,
        },
        "behavior_ack": {
            "K": 3, "ack_ids": [f"ack-{macro_index}"] * 3,
            "gripper_command_ids": [f"gripper-{macro_index}"] * 3,
            "gripper_ack_command_ids": [f"gripper-{macro_index}"] * 3,
            "accepted_absolute_action_k7": [[0.0] * 7 for _ in range(3)],
            "normalized_delta_action_k7": [[0.0] * 7 for _ in range(3)],
            "slot_owner": [owner] * 3,
            "accepted_action_source": [
                "human" if owner.startswith("human_") else
                "safety" if owner == "safety_hold" else
                "offline" if owner == "offline_demonstration" else "policy"
            ] * 3,
            "intervention_flags": [owner == "human_intervention"] * 3,
            "workspace_clip_flags": [False] * 3,
        },
        "fm_target": {
            "target_action_h50": [[0.0] * 7 for _ in range(50)],
            "action_valid_mask_h50": [True] * 50,
            "expert_slot_mask_h50": [expert] * 50,
            "expert_feature_mask_h50x7": [[expert] * 7 for _ in range(50)],
        },
        "outcome": {
            "reward": 0.0, "reward_revision": "reward-pending-v1",
            "terminated": False, "truncated": False, "bootstrap": True, "discount": 0.99,
        },
        "eligibility": {
            "critic_td_valid": True, "actor_q_valid": True,
            "expert_fm_available": expert, "quarantined": False,
            "quarantine_reason": None,
        },
        "commit": {
            "episode_sealed": True, "execution_event_sequence": macro_index,
            "ack_watermark": macro_index,
        },
    }


def test_finalize_uid_digest_and_schema_are_stable() -> None:
    value = finalize_ack_transition(transition_payload())
    assert len(value["identity"]["transition_uid"]) == 64
    assert value["integrity"]["canonical_payload_sha256"] == canonical_payload_sha256(value)
    assert validate_ack_transition(value) == value
    tampered = deepcopy(value)
    tampered["outcome"]["reward"] = 1.0
    with pytest.raises(TransitionContractError, match="DIGEST_MISMATCH"):
        validate_ack_transition(tampered)


def test_human_expert_requires_ack_source_and_intervention() -> None:
    invalid = AcceptedAck(
        ack_id="ack", receive_monotonic_ns=1,
        accepted_absolute_action7=(0.0,) * 7, gripper_command_id="g",
        gripper_ack_command_id="g",
        slot_owner="human_intervention", accepted_action_source="policy",
        intervention=True,
    )
    with pytest.raises(TransitionContractError, match="NOT_ACK_CONFIRMED"):
        invalid.validate()
    with pytest.raises(TransitionContractError, match="IDENTITY_MISSING"):
        AcceptedAck(**{**invalid.__dict__, "gripper_command_id": ""}).validate()
    with pytest.raises(TransitionContractError, match="GRIPPER_COMMAND_ID_MISMATCH"):
        AcceptedAck(**{**invalid.__dict__, "accepted_action_source": "human", "gripper_ack_command_id": "other"}).validate()


def test_reward_terminal_matrix_and_quarantine() -> None:
    validate_reward_terminal({
        "reward": 1.0, "terminated": True, "truncated": False,
        "bootstrap": False, "discount": 0.0, "next_observation_valid": False,
    })
    validate_reward_terminal({
        "reward": 0.0, "terminated": None, "truncated": True,
        "bootstrap": None, "discount": None, "next_observation_valid": False,
        "quarantined": True,
    })
    validate_reward_terminal({
        "reward": 0.0, "terminated": False, "truncated": True,
        "bootstrap": False, "discount": 0.0, "next_observation_valid": True,
    })
    with pytest.raises(TransitionContractError, match="TERMINATED_AND_TRUNCATED"):
        validate_reward_terminal({
            "reward": 0.0, "terminated": True, "truncated": True,
            "bootstrap": False, "discount": 0.0, "next_observation_valid": False,
        })
    with pytest.raises(TransitionContractError, match="NEXT_OBSERVATION"):
        validate_reward_terminal({
            "reward": 0.0, "terminated": False, "truncated": True,
            "bootstrap": False, "discount": 0.0, "next_observation_valid": False,
        })
