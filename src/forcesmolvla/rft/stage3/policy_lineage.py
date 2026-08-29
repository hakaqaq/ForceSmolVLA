"""CPU-only policy lineage and episode-initial gripper authority contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import threading
from typing import Any, Mapping

from .gripper_provenance import (
    GripperGeneration,
    GripperProvenanceError,
    VALID_TERMINAL_OUTCOMES,
)


POLICY_LINEAGE_SCHEMA = "forcesmolvla-stage3-policy-lineage-v1"
INITIAL_GRIPPER_AUTHORITY_SCHEMA = (
    "forcesmolvla-stage3-initial-gripper-authority-v1"
)
UPPER_CLOCK_DOMAIN = "upper_host_monotonic"
PRODUCTION_REQUEST_CLOCK_DOMAIN = "upper_host_monotonic_ns"


class PolicyLineageError(RuntimeError):
    """A fail-closed policy-lineage binding violation."""


def _nonempty(value: Any, reason: str) -> str:
    text = str(value).strip()
    if not text:
        raise PolicyLineageError(reason)
    return text


def _counter(value: Any, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolicyLineageError(reason)
    return value


@dataclass(frozen=True)
class PolicyResultLineage:
    request_id: str
    result_id: str
    chunk_id: str
    proposal_id: str
    policy_revision: str
    policy_epoch: int
    reset_generation: int
    takeover_generation: int
    t_ref_ns: int
    request_clock_domain_id: str
    clock_domain_id: str
    request_recorded_monotonic_ns: int
    result_recorded_monotonic_ns: int

    def selection_fields(self) -> dict[str, Any]:
        return {
            "lineage_schema": POLICY_LINEAGE_SCHEMA,
            **asdict(self),
        }


class PolicyLineageAudit:
    """Thread-safe identity registry; it never invokes inference or robot I/O."""

    def __init__(
        self,
        *,
        episode_id: str,
        policy_revision: str,
        reset_generation: int,
        clock_domain_id: str = UPPER_CLOCK_DOMAIN,
    ) -> None:
        self.episode_id = _nonempty(episode_id, "POLICY_LINEAGE_EPISODE_ID_MISSING")
        self.policy_revision = _nonempty(
            policy_revision, "POLICY_LINEAGE_REVISION_MISSING"
        )
        self.reset_generation = _counter(
            reset_generation, "POLICY_LINEAGE_RESET_GENERATION_INVALID"
        )
        if clock_domain_id != UPPER_CLOCK_DOMAIN:
            raise PolicyLineageError("POLICY_LINEAGE_CLOCK_DOMAIN_INVALID")
        self.clock_domain_id = clock_domain_id
        self._lock = threading.Lock()
        self._requests: dict[str, dict[str, Any]] = {}
        self._results: dict[str, PolicyResultLineage] = {}

    def record_request(
        self,
        request: Mapping[str, Any],
        *,
        policy_epoch: int,
        takeover_generation: int,
        recorded_monotonic_ns: int,
    ) -> dict[str, Any]:
        request_id = _nonempty(
            request.get("request_id"), "POLICY_LINEAGE_REQUEST_ID_MISSING"
        )
        chunk_id = _nonempty(
            request.get("chunk_id"), "POLICY_LINEAGE_CHUNK_ID_MISSING"
        )
        clock = _nonempty(
            request.get("clock_domain_id"), "POLICY_LINEAGE_REQUEST_CLOCK_MISSING"
        )
        if clock != PRODUCTION_REQUEST_CLOCK_DOMAIN:
            raise PolicyLineageError("POLICY_LINEAGE_REQUEST_CLOCK_MISMATCH")
        provenance = request.get("provenance")
        if not isinstance(provenance, Mapping):
            raise PolicyLineageError("POLICY_LINEAGE_REQUEST_PROVENANCE_MISSING")
        t_ref_ns = int(provenance.get("t_ref_ns", 0))
        epoch = _counter(policy_epoch, "POLICY_LINEAGE_POLICY_EPOCH_INVALID")
        takeover = _counter(
            takeover_generation, "POLICY_LINEAGE_TAKEOVER_GENERATION_INVALID"
        )
        if t_ref_ns <= 0 or recorded_monotonic_ns < t_ref_ns:
            raise PolicyLineageError("POLICY_LINEAGE_REQUEST_TIME_INVALID")
        identity = {
            "request_id": request_id,
            "chunk_id": chunk_id,
            "proposal_id": f"policy-proposal:{request_id}",
            "policy_revision": self.policy_revision,
            "policy_epoch": epoch,
            "reset_generation": self.reset_generation,
            "takeover_generation": takeover,
            "t_ref_ns": t_ref_ns,
            "request_clock_domain_id": clock,
            "clock_domain_id": self.clock_domain_id,
            "request_recorded_monotonic_ns": int(recorded_monotonic_ns),
        }
        with self._lock:
            previous = self._requests.get(request_id)
            if previous is not None and previous != identity:
                raise PolicyLineageError("POLICY_LINEAGE_REQUEST_ID_CONFLICT")
            self._requests[request_id] = identity
        return dict(identity)

    def record_result(
        self,
        request: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        recorded_monotonic_ns: int,
    ) -> PolicyResultLineage:
        request_id = _nonempty(
            request.get("request_id"), "POLICY_LINEAGE_REQUEST_ID_MISSING"
        )
        with self._lock:
            registered = self._requests.get(request_id)
        if registered is None:
            raise PolicyLineageError("POLICY_LINEAGE_RESULT_BEFORE_REQUEST")
        if (
            result.get("request_id") != registered["request_id"]
            or result.get("chunk_id") != registered["chunk_id"]
            or int(result.get("t_ref_ns", 0)) != registered["t_ref_ns"]
        ):
            raise PolicyLineageError("POLICY_LINEAGE_RESULT_BINDING_MISMATCH")
        if recorded_monotonic_ns < registered["request_recorded_monotonic_ns"]:
            raise PolicyLineageError("POLICY_LINEAGE_RESULT_TIME_INVALID")
        lineage = PolicyResultLineage(
            request_id=request_id,
            result_id=f"policy-result:{request_id}",
            chunk_id=registered["chunk_id"],
            proposal_id=registered["proposal_id"],
            policy_revision=registered["policy_revision"],
            policy_epoch=registered["policy_epoch"],
            reset_generation=registered["reset_generation"],
            takeover_generation=registered["takeover_generation"],
            t_ref_ns=registered["t_ref_ns"],
            request_clock_domain_id=registered["request_clock_domain_id"],
            clock_domain_id=registered["clock_domain_id"],
            request_recorded_monotonic_ns=registered[
                "request_recorded_monotonic_ns"
            ],
            result_recorded_monotonic_ns=int(recorded_monotonic_ns),
        )
        with self._lock:
            previous = self._results.get(lineage.result_id)
            if previous is not None and previous != lineage:
                raise PolicyLineageError("POLICY_LINEAGE_RESULT_ID_CONFLICT")
            self._results[lineage.result_id] = lineage
        return lineage

    def bind_dispatch(
        self,
        lineage: PolicyResultLineage,
        *,
        policy_epoch: int,
        takeover_generation: int,
    ) -> dict[str, Any]:
        if (
            _counter(policy_epoch, "POLICY_LINEAGE_POLICY_EPOCH_INVALID")
            != lineage.policy_epoch
            or _counter(
                takeover_generation,
                "POLICY_LINEAGE_TAKEOVER_GENERATION_INVALID",
            )
            != lineage.takeover_generation
            or lineage.policy_revision != self.policy_revision
            or lineage.reset_generation != self.reset_generation
        ):
            raise PolicyLineageError("POLICY_LINEAGE_STALE_GENERATION")
        with self._lock:
            if self._results.get(lineage.result_id) != lineage:
                raise PolicyLineageError("POLICY_LINEAGE_RESULT_NOT_REGISTERED")
        return lineage.selection_fields()


@dataclass(frozen=True)
class InitialGripperAuthority:
    episode_id: str
    origin_local_goal_sequence: int
    origin_action_goal_id: str
    origin_accepted_monotonic_ns: int
    requested_state: str
    requested_width_m: float
    terminal_outcome: str
    terminal_finished_monotonic_ns: int
    feedback_width_m: float
    feedback_state: str
    feedback_monotonic_ns: int
    captured_monotonic_ns: int
    feedback_age_ns: int
    clock_domain_id: str
    generation: GripperGeneration

    def validate(self, *, max_feedback_age_ns: int) -> "InitialGripperAuthority":
        self.generation.validate()
        if (
            self.episode_id != self.generation.episode_id
            or self.origin_local_goal_sequence <= 0
            or not self.origin_action_goal_id
            or self.origin_accepted_monotonic_ns <= 0
            or self.requested_state not in {"OPEN", "CLOSED"}
            or not math.isfinite(self.requested_width_m)
            or not 0.0 <= self.requested_width_m <= 0.1
            or self.terminal_outcome not in VALID_TERMINAL_OUTCOMES
            or self.terminal_finished_monotonic_ns
            < self.origin_accepted_monotonic_ns
            or not math.isfinite(self.feedback_width_m)
            or not 0.0 <= self.feedback_width_m <= 0.1
            or not self.feedback_state
            or self.feedback_monotonic_ns <= 0
            or self.captured_monotonic_ns < self.feedback_monotonic_ns
            or self.feedback_age_ns
            != self.captured_monotonic_ns - self.feedback_monotonic_ns
            or self.feedback_age_ns < 0
            or self.feedback_age_ns > max_feedback_age_ns
            or self.clock_domain_id != UPPER_CLOCK_DOMAIN
        ):
            raise GripperProvenanceError("INITIAL_GRIPPER_AUTHORITY_INVALID")
        return self

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = INITIAL_GRIPPER_AUTHORITY_SCHEMA
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InitialGripperAuthority":
        if value.get("schema") != INITIAL_GRIPPER_AUTHORITY_SCHEMA:
            raise GripperProvenanceError("INITIAL_GRIPPER_AUTHORITY_SCHEMA_INVALID")
        generation = value.get("generation")
        if not isinstance(generation, Mapping):
            raise GripperProvenanceError(
                "INITIAL_GRIPPER_AUTHORITY_GENERATION_MISSING"
            )
        try:
            return cls(
                episode_id=str(value["episode_id"]),
                origin_local_goal_sequence=int(value["origin_local_goal_sequence"]),
                origin_action_goal_id=str(value["origin_action_goal_id"]),
                origin_accepted_monotonic_ns=int(
                    value["origin_accepted_monotonic_ns"]
                ),
                requested_state=str(value["requested_state"]),
                requested_width_m=float(value["requested_width_m"]),
                terminal_outcome=str(value["terminal_outcome"]),
                terminal_finished_monotonic_ns=int(
                    value["terminal_finished_monotonic_ns"]
                ),
                feedback_width_m=float(value["feedback_width_m"]),
                feedback_state=str(value["feedback_state"]),
                feedback_monotonic_ns=int(value["feedback_monotonic_ns"]),
                captured_monotonic_ns=int(value["captured_monotonic_ns"]),
                feedback_age_ns=int(value["feedback_age_ns"]),
                clock_domain_id=str(value["clock_domain_id"]),
                generation=GripperGeneration(
                    episode_id=str(generation["episode_id"]),
                    reset_generation=int(generation["reset_generation"]),
                    takeover_generation=int(generation["takeover_generation"]),
                    policy_revision=str(generation["policy_revision"]),
                    policy_epoch=int(generation["policy_epoch"]),
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise GripperProvenanceError(
                "INITIAL_GRIPPER_AUTHORITY_FIELDS_INVALID"
            ) from error
