"""In-memory-only immutable policy revision state machine for G2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class RevisionState(str, Enum):
    CANDIDATE = "candidate"
    PENDING = "pending"
    ACTIVE = "active"
    PREVIOUS = "previous"
    REJECTED = "rejected"


@dataclass(frozen=True)
class RevisionRecord:
    revision_id: str
    model_sha256: str
    state: RevisionState
    rejection_reason: str | None = None

    def validate(self) -> "RevisionRecord":
        if not self.revision_id:
            raise ValueError("STAGE3_REVISION_ID_EMPTY")
        if len(self.model_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.model_sha256):
            raise ValueError("STAGE3_REVISION_MODEL_SHA_INVALID")
        if self.state is RevisionState.REJECTED and not self.rejection_reason:
            raise ValueError("STAGE3_REJECTED_REVISION_REASON_MISSING")
        return self


@dataclass(frozen=True)
class QuiescentBoundary:
    active_episode: bool
    inflight_inference: int
    queued_actions: int
    unconsumed_acks: int
    robot_home: bool
    wal_sealed: bool

    def validate_for_activation(self) -> None:
        if min(self.inflight_inference, self.queued_actions, self.unconsumed_acks) < 0:
            raise ValueError("STAGE3_QUIESCENT_COUNTER_NEGATIVE")
        if (
            self.active_episode
            or self.inflight_inference != 0
            or self.queued_actions != 0
            or self.unconsumed_acks != 0
            or not self.robot_home
            or not self.wal_sealed
        ):
            raise RuntimeError("STAGE3_REVISION_ACTIVATION_NOT_QUIESCENT")


class InMemoryRevisionStateMachine:
    def __init__(self, active: RevisionRecord) -> None:
        active.validate()
        if active.state is not RevisionState.ACTIVE:
            raise ValueError("STAGE3_INITIAL_REVISION_NOT_ACTIVE")
        self._records = {active.revision_id: active}
        self.active_revision_id = active.revision_id
        self.pending_revision_id: str | None = None
        self.previous_revision_id: str | None = None
        self.episode_revision_id: str | None = None
        self.policy_epoch = 0

    def register_candidate(self, revision_id: str, model_sha256: str) -> RevisionRecord:
        if revision_id in self._records:
            existing = self._records[revision_id]
            if existing.model_sha256 != model_sha256:
                raise RuntimeError("STAGE3_REVISION_ID_SHA_COLLISION")
            return existing
        record = RevisionRecord(revision_id, model_sha256, RevisionState.CANDIDATE).validate()
        self._records[revision_id] = record
        return record

    def stage(self, revision_id: str) -> RevisionRecord:
        record = self._records[revision_id]
        if record.state is not RevisionState.CANDIDATE:
            raise RuntimeError("STAGE3_ONLY_CANDIDATE_CAN_BE_STAGED")
        if self.pending_revision_id is not None:
            raise RuntimeError("STAGE3_PENDING_REVISION_ALREADY_EXISTS")
        staged = replace(record, state=RevisionState.PENDING)
        self._records[revision_id] = staged
        self.pending_revision_id = revision_id
        return staged

    def reject(self, revision_id: str, reason: str) -> RevisionRecord:
        if not reason:
            raise ValueError("STAGE3_REVISION_REJECTION_REASON_EMPTY")
        record = self._records[revision_id]
        if record.state not in {RevisionState.CANDIDATE, RevisionState.PENDING}:
            raise RuntimeError("STAGE3_ACTIVE_OR_PREVIOUS_REVISION_CANNOT_BE_REJECTED")
        rejected = replace(record, state=RevisionState.REJECTED, rejection_reason=reason)
        self._records[revision_id] = rejected
        if self.pending_revision_id == revision_id:
            self.pending_revision_id = None
        return rejected

    def activate_pending(self, boundary: QuiescentBoundary) -> RevisionRecord:
        boundary.validate_for_activation()
        if self.episode_revision_id is not None:
            raise RuntimeError("STAGE3_REVISION_ACTIVATION_DURING_EPISODE")
        if self.pending_revision_id is None:
            raise RuntimeError("STAGE3_NO_PENDING_REVISION")
        current = self._records[self.active_revision_id]
        pending = self._records[self.pending_revision_id]
        self._records[current.revision_id] = replace(current, state=RevisionState.PREVIOUS)
        activated = replace(pending, state=RevisionState.ACTIVE)
        self._records[activated.revision_id] = activated
        self.previous_revision_id = current.revision_id
        self.active_revision_id = activated.revision_id
        self.pending_revision_id = None
        self.policy_epoch += 1
        return activated

    def begin_episode(self) -> str:
        if self.episode_revision_id is not None:
            raise RuntimeError("STAGE3_EPISODE_ALREADY_ACTIVE")
        self.episode_revision_id = self.active_revision_id
        return self.episode_revision_id

    def assert_episode_revision(self, revision_id: str) -> None:
        if self.episode_revision_id is None or revision_id != self.episode_revision_id:
            raise RuntimeError("STAGE3_ONE_EPISODE_ONE_REVISION_VIOLATION")

    def end_episode(self) -> None:
        if self.episode_revision_id is None:
            raise RuntimeError("STAGE3_NO_ACTIVE_EPISODE")
        self.episode_revision_id = None

    def rollback(self, boundary: QuiescentBoundary) -> RevisionRecord:
        boundary.validate_for_activation()
        if self.episode_revision_id is not None:
            raise RuntimeError("STAGE3_ROLLBACK_DURING_EPISODE")
        if self.previous_revision_id is None:
            raise RuntimeError("STAGE3_NO_PREVIOUS_REVISION")
        current = self._records[self.active_revision_id]
        previous = self._records[self.previous_revision_id]
        self._records[current.revision_id] = replace(current, state=RevisionState.PREVIOUS)
        restored = replace(previous, state=RevisionState.ACTIVE)
        self._records[restored.revision_id] = restored
        self.previous_revision_id = current.revision_id
        self.active_revision_id = restored.revision_id
        self.policy_epoch += 1
        return restored

    def record(self, revision_id: str) -> RevisionRecord:
        return self._records[revision_id]
