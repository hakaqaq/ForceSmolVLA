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
            raise ValueError("ONLINE_REPLAY_PROTOCOL_IDENTITY_EMPTY")
        if self.arbitration_epoch_at_request < 0 or self.t_ref_monotonic_ns <= 0:
            raise ValueError("ONLINE_REPLAY_PROTOCOL_COUNTER_OR_TIMESTAMP_INVALID")
        if len(self.model_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.model_sha256):
            raise ValueError("ONLINE_REPLAY_PROTOCOL_MODEL_SHA_INVALID")
        return self


class InferenceDisposition(str, Enum):
    ACCEPT = "accept"
    STALE_DROP = "stale_drop"


class PolicyEpochGate:
    def __init__(
        self,
        *,
        active_revision_id: str,
        active_model_sha256: str | None = None,
        initial_epoch: int = 0,
    ) -> None:
        if not active_revision_id or initial_epoch < 0:
            raise ValueError("ONLINE_REPLAY_POLICY_EPOCH_INITIAL_STATE_INVALID")
        if active_model_sha256 is not None and (
            len(active_model_sha256) != 64
            or any(char not in "0123456789abcdef" for char in active_model_sha256)
        ):
            raise ValueError("ONLINE_REPLAY_POLICY_MODEL_SHA_INVALID")
        self.active_revision_id = active_revision_id
        self.active_model_sha256 = active_model_sha256
        self.policy_epoch = int(initial_epoch)
        self._pinned_request: tuple[str, str, str, str, str] | None = None

    def invalidate_queued_policy(self) -> int:
        self.policy_epoch += 1
        self._pinned_request = None
        return self.policy_epoch

    def activate_revision(
        self, revision_id: str, model_sha256: str | None = None,
    ) -> int:
        if not revision_id:
            raise ValueError("ONLINE_REPLAY_REVISION_ID_EMPTY")
        if model_sha256 is not None and (
            len(model_sha256) != 64
            or any(char not in "0123456789abcdef" for char in model_sha256)
        ):
            raise ValueError("ONLINE_REPLAY_POLICY_MODEL_SHA_INVALID")
        self.active_revision_id = revision_id
        self.active_model_sha256 = model_sha256
        return self.invalidate_queued_policy()

    def pin_request(self, envelope: TransportEnvelope) -> InferenceDisposition:
        """Pin the one request/chunk identity whose result may currently be accepted."""

        envelope.validate()
        if (
            envelope.arbitration_epoch_at_request != self.policy_epoch
            or envelope.policy_revision_id != self.active_revision_id
            or (
                self.active_model_sha256 is not None
                and envelope.model_sha256 != self.active_model_sha256
            )
        ):
            return InferenceDisposition.STALE_DROP
        self._pinned_request = (
            envelope.episode_id,
            envelope.request_id,
            envelope.chunk_id,
            envelope.observation_id,
            envelope.model_sha256,
        )
        return InferenceDisposition.ACCEPT

    @property
    def has_pinned_request(self) -> bool:
        return self._pinned_request is not None

    def classify_result(self, envelope: TransportEnvelope) -> InferenceDisposition:
        envelope.validate()
        if (
            envelope.arbitration_epoch_at_request != self.policy_epoch
            or envelope.policy_revision_id != self.active_revision_id
            or (
                self.active_model_sha256 is not None
                and envelope.model_sha256 != self.active_model_sha256
            )
        ):
            return InferenceDisposition.STALE_DROP
        if self._pinned_request is not None and self._pinned_request != (
            envelope.episode_id,
            envelope.request_id,
            envelope.chunk_id,
            envelope.observation_id,
            envelope.model_sha256,
        ):
            return InferenceDisposition.STALE_DROP
        return InferenceDisposition.ACCEPT
