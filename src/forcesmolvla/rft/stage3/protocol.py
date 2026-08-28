"""Transport identities and stale-policy rejection with no network dependency."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class TransportEnvelope:
    run_id: str
    session_id: str
    episode_id: str
    request_id: str
    chunk_id: str
    arbitration_epoch_at_request: int
    policy_revision_id: str
    model_sha256: str
    t_ref_monotonic_ns: int
    observation_id: str

    def validate(self) -> "TransportEnvelope":
        strings = (
            self.run_id, self.session_id, self.episode_id, self.request_id, self.chunk_id,
            self.policy_revision_id, self.model_sha256, self.observation_id,
        )
        if any(not value for value in strings):
            raise ValueError("STAGE3_PROTOCOL_IDENTITY_EMPTY")
        if self.arbitration_epoch_at_request < 0 or self.t_ref_monotonic_ns <= 0:
            raise ValueError("STAGE3_PROTOCOL_COUNTER_OR_TIMESTAMP_INVALID")
        if len(self.model_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.model_sha256):
            raise ValueError("STAGE3_PROTOCOL_MODEL_SHA_INVALID")
        return self


class InferenceDisposition(str, Enum):
    ACCEPT = "accept"
    STALE_DROP = "stale_drop"


class PolicyEpochGate:
    def __init__(self, *, active_revision_id: str, initial_epoch: int = 0) -> None:
        if not active_revision_id or initial_epoch < 0:
            raise ValueError("STAGE3_POLICY_EPOCH_INITIAL_STATE_INVALID")
        self.active_revision_id = active_revision_id
        self.policy_epoch = int(initial_epoch)

    def invalidate_queued_policy(self) -> int:
        self.policy_epoch += 1
        return self.policy_epoch

    def activate_revision(self, revision_id: str) -> int:
        if not revision_id:
            raise ValueError("STAGE3_REVISION_ID_EMPTY")
        self.active_revision_id = revision_id
        return self.invalidate_queued_policy()

    def classify_result(self, envelope: TransportEnvelope) -> InferenceDisposition:
        envelope.validate()
        if (
            envelope.arbitration_epoch_at_request != self.policy_epoch
            or envelope.policy_revision_id != self.active_revision_id
        ):
            return InferenceDisposition.STALE_DROP
        return InferenceDisposition.ACCEPT
