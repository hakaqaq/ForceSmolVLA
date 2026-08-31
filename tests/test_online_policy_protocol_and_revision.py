from __future__ import annotations

import pytest

from forcesmolvla.rft.online.policy_protocol import (
    InferenceDisposition,
    PolicyEpochGate,
    TransportEnvelope,
)
from forcesmolvla.rft.online.policy_revision import (
    InMemoryRevisionStateMachine,
    QuiescentBoundary,
    RevisionRecord,
    RevisionState,
)


SHA0 = "0" * 64
SHA1 = "1" * 64


def envelope(epoch: int, revision: str) -> TransportEnvelope:
    return TransportEnvelope(
        run_id="run", session_id="session", episode_id="episode",
        request_id="request", chunk_id="chunk",
        arbitration_epoch_at_request=epoch, policy_revision_id=revision,
        model_sha256=SHA0, t_ref_monotonic_ns=1, observation_id="obs",
    )


def quiet(**overrides) -> QuiescentBoundary:
    values = {
        "active_episode": False, "inflight_inference": 0, "queued_actions": 0,
        "unconsumed_acks": 0, "robot_home": True, "wal_sealed": True,
    }
    values.update(overrides)
    return QuiescentBoundary(**values)


def test_policy_epoch_stale_result_is_normal_drop() -> None:
    gate = PolicyEpochGate(active_revision_id="r0")
    assert gate.classify_result(envelope(0, "r0")) is InferenceDisposition.ACCEPT
    gate.invalidate_queued_policy()
    assert gate.classify_result(envelope(0, "r0")) is InferenceDisposition.STALE_DROP


def test_revision_lifecycle_enforces_episode_boundary_and_rollback() -> None:
    machine = InMemoryRevisionStateMachine(
        RevisionRecord("r0", SHA0, RevisionState.ACTIVE)
    )
    machine.register_candidate("r1", SHA1)
    machine.stage("r1")
    assert machine.begin_episode() == "r0"
    machine.assert_episode_revision("r0")
    with pytest.raises(RuntimeError, match="NOT_QUIESCENT|DURING_EPISODE"):
        machine.activate_pending(quiet(active_episode=True))
    with pytest.raises(RuntimeError, match="ONE_EPISODE_ONE_REVISION"):
        machine.assert_episode_revision("r1")
    machine.end_episode()
    activated = machine.activate_pending(quiet())
    assert activated.revision_id == "r1"
    assert machine.active_revision_id == "r1"
    assert machine.previous_revision_id == "r0"
    assert machine.policy_epoch == 1
    restored = machine.rollback(quiet())
    assert restored.revision_id == "r0"
    assert machine.policy_epoch == 2


def test_revision_identity_is_immutable_and_invalid_candidate_rejects() -> None:
    machine = InMemoryRevisionStateMachine(
        RevisionRecord("r0", SHA0, RevisionState.ACTIVE)
    )
    machine.register_candidate("r1", SHA1)
    with pytest.raises(RuntimeError, match="SHA_COLLISION"):
        machine.register_candidate("r1", SHA0)
    rejected = machine.reject("r1", "offline parity failed")
    assert rejected.state is RevisionState.REJECTED
    with pytest.raises(RuntimeError, match="CANDIDATE"):
        machine.stage("r1")
