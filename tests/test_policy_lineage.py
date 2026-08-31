from __future__ import annotations

from dataclasses import replace

import pytest

from forcesmolvla.rft.online.gripper_authority import (
    GripperGeneration,
    GripperProvenanceError,
)
from forcesmolvla.rft.online.policy_lineage import (
    InitialGripperAuthority,
    PolicyLineageAudit,
    PolicyLineageError,
)


def _request() -> dict:
    return {
        "request_id": "request-1",
        "chunk_id": "chunk-1",
        "clock_domain_id": "upper_host_monotonic_ns",
        "provenance": {"t_ref_ns": 1_000_000_000},
    }


def _result() -> dict:
    return {
        "request_id": "request-1",
        "chunk_id": "chunk-1",
        "t_ref_ns": 1_000_000_000,
    }


def _audit() -> PolicyLineageAudit:
    return PolicyLineageAudit(
        episode_id="dataset/episode_000001",
        policy_revision="model-sha-1",
        reset_generation=1,
    )


def test_real_request_result_identity_closes_without_fake_ack() -> None:
    audit = _audit()
    audit.record_request(
        _request(),
        policy_epoch=2,
        takeover_generation=2,
        recorded_monotonic_ns=1_000_000_010,
    )
    result = audit.record_result(
        _request(), _result(), recorded_monotonic_ns=1_100_000_000
    )
    fields = audit.bind_dispatch(
        result, policy_epoch=2, takeover_generation=2
    )
    assert fields["request_id"] == "request-1"
    assert fields["result_id"] == "policy-result:request-1"
    assert fields["proposal_id"] == "policy-proposal:request-1"
    assert fields["chunk_id"] == "chunk-1"
    assert fields["policy_revision"] == "model-sha-1"
    assert fields["reset_generation"] == 1
    assert "pose_ack_id" not in fields
    assert "gripper_ack_id" not in fields


def test_result_before_request_and_mismatched_result_fail_closed() -> None:
    audit = _audit()
    with pytest.raises(PolicyLineageError, match="RESULT_BEFORE_REQUEST"):
        audit.record_result(
            _request(), _result(), recorded_monotonic_ns=1_100_000_000
        )
    audit.record_request(
        _request(),
        policy_epoch=2,
        takeover_generation=2,
        recorded_monotonic_ns=1_000_000_010,
    )
    with pytest.raises(PolicyLineageError, match="RESULT_BINDING_MISMATCH"):
        audit.record_result(
            _request(),
            {**_result(), "chunk_id": "wrong"},
            recorded_monotonic_ns=1_100_000_000,
        )


def test_takeover_generation_change_rejects_stale_result_dispatch() -> None:
    audit = _audit()
    audit.record_request(
        _request(),
        policy_epoch=2,
        takeover_generation=2,
        recorded_monotonic_ns=1_000_000_010,
    )
    result = audit.record_result(
        _request(), _result(), recorded_monotonic_ns=1_100_000_000
    )
    with pytest.raises(PolicyLineageError, match="STALE_GENERATION"):
        audit.bind_dispatch(result, policy_epoch=3, takeover_generation=3)


def _initial() -> InitialGripperAuthority:
    generation = GripperGeneration(
        episode_id="dataset/episode_000001",
        reset_generation=1,
        takeover_generation=2,
        policy_revision="model-sha-1",
        policy_epoch=2,
    )
    return InitialGripperAuthority(
        episode_id=generation.episode_id,
        origin_local_goal_sequence=1,
        origin_action_goal_id="real-ros-goal-id",
        origin_accepted_monotonic_ns=800_000_000,
        requested_state="OPEN",
        requested_width_m=0.085,
        terminal_outcome="reached",
        terminal_finished_monotonic_ns=900_000_000,
        feedback_width_m=0.084,
        feedback_state="OPEN",
        feedback_monotonic_ns=990_000_000,
        captured_monotonic_ns=1_000_000_000,
        feedback_age_ns=10_000_000,
        clock_domain_id="upper_host_monotonic",
        generation=generation,
    )


def test_initial_gripper_authority_round_trip_is_real_origin_bound() -> None:
    authority = _initial().validate(max_feedback_age_ns=100_000_000)
    restored = InitialGripperAuthority.from_mapping(authority.to_dict()).validate(
        max_feedback_age_ns=100_000_000
    )
    assert restored == authority
    assert restored.origin_action_goal_id == "real-ros-goal-id"


def test_initial_gripper_authority_rejects_missing_origin_and_stale_feedback() -> None:
    with pytest.raises(GripperProvenanceError, match="INITIAL_GRIPPER"):
        replace(_initial(), origin_action_goal_id="").validate(
            max_feedback_age_ns=100_000_000
        )
    with pytest.raises(GripperProvenanceError, match="INITIAL_GRIPPER"):
        replace(
            _initial(),
            feedback_monotonic_ns=800_000_000,
            feedback_age_ns=200_000_000,
        ).validate(max_feedback_age_ns=100_000_000)
