"""CPU-only runtime temporal ledger and fail-closed Stage-3 primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
from typing import Sequence


NANOSECONDS_PER_SECOND = 1_000_000_000

H50_MODEL_TIMEBASE_HZ = 30
POSE_REFERENCE_DISPATCH_HZ = 10
CONTROLLER_INTERNAL_SERVO_HZ = "UNVERIFIED"
STAGE3_PROJECTION_GRID_HZ = 30
CONTRACT_TRANSITION_MACRO_HZ = 10
PRODUCTION_TRANSITION_COMMIT_HZ = "UNVERIFIED"
POLICY_REQUEST_TRIGGER = "event_driven_low_watermark"
POLICY_REQUEST_HZ_MEASURED = "UNVERIFIED"
POLICY_INFERENCE_10HZ_REQUIRED = False
PRODUCTION_SAFE_INFERENCE_REFRESH_RATE = "UNVERIFIED"

ACTION_SLOT_FIFO_PRESENT = False
H50_ACTIONS_CACHED = 50
MAX_SELECTIONS_PER_ADOPTED_CHUNK = 8
SELECTED_INDEX_POLICY = "rational_time_based_sparse_selection"
SELECTED_INDEX_PHASE = "ceil_from_t_ref_ns_on_rational_30hz_grid"

CURRENT_LOW_WATERMARK_DISPATCHES = 4
CURRENT_LOW_WATERMARK_COVERAGE_NS = 400_000_000
G7_CONCURRENT_MAX_SERVICE_LATENCY_NS = 443_161_677
CURRENT_LOW_WATERMARK_APPROVED = False

OBSERVATION_STATE_NORMALIZATION_ONCE = "code-audited"
OBSERVATION_WRENCH_NORMALIZATION_ONCE = "code-audited"
ACTION_DELTA_DENORMALIZATION_ONCE = "code-audited"
RECORDED_LIVE_ACCEPTED_MACRO_NORMALIZATION_ONCE = "UNVERIFIED"

RUNTIME_THREAD_OWNERSHIP = "single_owner_event_loop_only"
RUNTIME_CLOCK_DOMAIN_BOUND = True
CROSS_CLOCK_TIMESTAMP_REJECTED = True

GRIPPER_NOOP_ACK_POLICY = "UNBOUND"
FULL_ACTION7_ACK_CLOSURE = False
PRODUCTION_INTEGRATION_BLOCKED_ON_GRIPPER_ACK = True

RUNTIME_LEDGER_PERSISTED = False
PRODUCTION_RUNTIME_LEDGER_RESUME = "UNVERIFIED"
G5_PRODUCTION_DURABLE_RESUME = "UNVERIFIED"


class SafetyDirective(str, Enum):
    HOLD = "hold"
    STOP = "stop"


class RuntimeSafetyViolation(RuntimeError):
    def __init__(self, reason: str, directive: SafetyDirective) -> None:
        super().__init__(reason)
        self.reason = reason
        self.directive = directive


def rational_h50_index(t_ref_ns: int, selection_ns: int) -> int:
    """Ceiling selection on a fixed t_ref anchor without floating-point phase drift."""

    if t_ref_ns <= 0 or selection_ns < t_ref_ns:
        raise RuntimeSafetyViolation(
            "STAGE3_H50_SELECTION_TIME_INVALID", SafetyDirective.HOLD
        )
    age_ns = selection_ns - t_ref_ns
    return (
        age_ns * H50_MODEL_TIMEBASE_HZ + NANOSECONDS_PER_SECOND - 1
    ) // NANOSECONDS_PER_SECOND


def _absolute7(values: tuple[float, ...]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != 7 or not all(math.isfinite(value) for value in result):
        raise RuntimeSafetyViolation(
            "STAGE3_POST_ADAPTER_ABSOLUTE7_INVALID", SafetyDirective.HOLD
        )
    return result


@dataclass(frozen=True)
class RuntimeSafetyLimits:
    max_chunk_age_ns: int
    max_selected_index: int
    max_dispatch_count: int
    refresh_worst_case_service_ns: int
    refresh_additional_headroom_ns: int
    pose_ack_deadline_ns: int
    gripper_ack_deadline_ns: int

    def __post_init__(self) -> None:
        positive = (
            self.max_chunk_age_ns,
            self.refresh_worst_case_service_ns,
            self.refresh_additional_headroom_ns,
            self.pose_ack_deadline_ns,
            self.gripper_ack_deadline_ns,
        )
        if any(isinstance(value, bool) or value <= 0 for value in positive):
            raise ValueError("STAGE3_RUNTIME_SAFETY_LIMIT_NONPOSITIVE")
        if not 0 <= self.max_selected_index < H50_ACTIONS_CACHED:
            raise ValueError("STAGE3_MAX_SELECTED_INDEX_INVALID")
        if not 0 < self.max_dispatch_count <= MAX_SELECTIONS_PER_ADOPTED_CHUNK:
            raise ValueError("STAGE3_MAX_DISPATCH_COUNT_INVALID")

    @property
    def refresh_required_coverage_ns(self) -> int:
        return self.refresh_worst_case_service_ns + self.refresh_additional_headroom_ns


@dataclass(frozen=True)
class ChunkRequestIdentity:
    request_id: str
    chunk_id: str
    proposal_id: str
    policy_revision: str
    policy_epoch: int
    takeover_generation: int
    reset_generation: int
    request_clock_domain_id: str
    t_ref_clock_domain_id: str
    t_ref_ns: int

    def validate(self) -> "ChunkRequestIdentity":
        if not all(
            (
                self.request_id,
                self.chunk_id,
                self.proposal_id,
                self.policy_revision,
                self.request_clock_domain_id,
                self.t_ref_clock_domain_id,
            )
        ):
            raise ValueError("STAGE3_RUNTIME_IDENTITY_EMPTY")
        if self.request_clock_domain_id != self.t_ref_clock_domain_id:
            raise RuntimeSafetyViolation(
                "STAGE3_REQUEST_T_REF_CLOCK_MISMATCH", SafetyDirective.HOLD
            )
        counters = (
            self.policy_epoch,
            self.takeover_generation,
            self.reset_generation,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counters
        ) or self.t_ref_ns <= 0:
            raise ValueError("STAGE3_RUNTIME_GENERATION_OR_TIME_INVALID")
        return self


@dataclass(frozen=True)
class ChunkResultIdentity(ChunkRequestIdentity):
    result_id: str
    result_clock_domain_id: str

    def validate(self) -> "ChunkResultIdentity":
        super().validate()
        if not self.result_id or not self.result_clock_domain_id:
            raise ValueError("STAGE3_RESULT_ID_EMPTY")
        if self.result_clock_domain_id != self.request_clock_domain_id:
            raise RuntimeSafetyViolation(
                "STAGE3_RESULT_CLOCK_MISMATCH", SafetyDirective.HOLD
            )
        return self

    def request_identity(self) -> ChunkRequestIdentity:
        return ChunkRequestIdentity(
            request_id=self.request_id,
            chunk_id=self.chunk_id,
            proposal_id=self.proposal_id,
            policy_revision=self.policy_revision,
            policy_epoch=self.policy_epoch,
            takeover_generation=self.takeover_generation,
            reset_generation=self.reset_generation,
            request_clock_domain_id=self.request_clock_domain_id,
            t_ref_clock_domain_id=self.t_ref_clock_domain_id,
            t_ref_ns=self.t_ref_ns,
        )


@dataclass(frozen=True)
class RefreshAssessment:
    selected_index: int
    remaining_time_ns: int
    required_coverage_ns: int
    refresh_due: bool
    service_headroom_exhausted: bool


@dataclass(frozen=True)
class SelectionLedgerEntry:
    request_id: str
    result_id: str
    chunk_id: str
    proposal_id: str
    policy_revision: str
    policy_epoch: int
    takeover_generation: int
    reset_generation: int
    request_clock_domain_id: str
    result_clock_domain_id: str
    t_ref_clock_domain_id: str
    t_ref_ns: int
    dispatch_sequence: int
    selected_index: int
    selection_ns: int
    selection_clock_domain_id: str
    dispatch_ns: int
    dispatch_clock_domain_id: str
    selected_post_adapter_absolute7: tuple[float, ...]
    pose_command_id: str
    gripper_command_id: str
    pose_ack_id: str | None = None
    pose_ack_ns: int | None = None
    pose_ack_clock_domain_id: str | None = None
    pose_ack_accepted: bool | None = None
    gripper_ack_id: str | None = None
    gripper_ack_ns: int | None = None
    gripper_ack_clock_domain_id: str | None = None
    gripper_ack_accepted: bool | None = None
    status: str = "awaiting_ack"
    failure_reason: str | None = None

    def to_accepted_ack(self, *, clock_domain_id: str):
        """Expose only dual-ACK accepted post-adapter absolute7 to Stage-3."""

        if (
            self.status != "accepted"
            or self.pose_ack_accepted is not True
            or self.gripper_ack_accepted is not True
            or self.pose_ack_id != self.pose_command_id
            or self.gripper_ack_id != self.gripper_command_id
            or self.pose_ack_ns is None
            or self.gripper_ack_ns is None
            or not clock_domain_id
            or any(
                domain != clock_domain_id
                for domain in (
                    self.request_clock_domain_id,
                    self.result_clock_domain_id,
                    self.t_ref_clock_domain_id,
                    self.selection_clock_domain_id,
                    self.dispatch_clock_domain_id,
                    self.pose_ack_clock_domain_id,
                    self.gripper_ack_clock_domain_id,
                )
            )
        ):
            raise RuntimeSafetyViolation(
                "STAGE3_SELECTION_NOT_ACK_AUTHORITATIVE", SafetyDirective.STOP
            )
        from .transition import AcceptedAck

        return AcceptedAck(
            ack_id=self.pose_ack_id,
            receive_monotonic_ns=max(self.pose_ack_ns, self.gripper_ack_ns),
            accepted_absolute_action7=self.selected_post_adapter_absolute7,
            gripper_command_id=self.gripper_command_id,
            gripper_ack_command_id=self.gripper_ack_id,
            slot_owner="policy",
            accepted_action_source="policy",
            intervention=False,
        ).validate()


class RationalH50SelectionLedger:
    """Serial runtime ledger; no action-slot FIFO and no hardware dependency."""

    def __init__(
        self,
        limits: RuntimeSafetyLimits,
        *,
        policy_revision: str,
        clock_domain_id: str,
        policy_epoch: int = 0,
        takeover_generation: int = 0,
        reset_generation: int = 0,
    ) -> None:
        if not policy_revision or not clock_domain_id or min(
            policy_epoch, takeover_generation, reset_generation
        ) < 0:
            raise ValueError("STAGE3_RUNTIME_INITIAL_STATE_INVALID")
        self.limits = limits
        self.policy_revision = policy_revision
        self.clock_domain_id = clock_domain_id
        self.policy_epoch = policy_epoch
        self.takeover_generation = takeover_generation
        self.reset_generation = reset_generation
        self._owner_thread_id = threading.get_ident()
        self._entries: list[SelectionLedgerEntry] = []
        self._pinned_request: ChunkRequestIdentity | None = None
        self._active: ChunkResultIdentity | None = None
        self._pending_entry_index: int | None = None
        self._dispatch_count = 0
        self._last_selected_index: int | None = None
        self._last_dispatch_sequence: int | None = None
        self._seen_result_ids: set[str] = set()
        self._seen_pose_command_ids: set[str] = set()
        self._seen_gripper_command_ids: set[str] = set()
        self._stop_latched = False
        self._directive = SafetyDirective.HOLD

    @property
    def active_chunk(self) -> ChunkResultIdentity | None:
        self._assert_owner()
        return self._active

    @property
    def fail_closed_directive(self) -> SafetyDirective | None:
        self._assert_owner()
        if self._stop_latched:
            return SafetyDirective.STOP
        if self._active is None:
            return SafetyDirective.HOLD
        return self._directive

    @property
    def entries(self) -> tuple[SelectionLedgerEntry, ...]:
        self._assert_owner()
        return tuple(self._entries)

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            self._stop_latched = True
            self._directive = SafetyDirective.STOP
            raise RuntimeSafetyViolation(
                "STAGE3_RUNTIME_LEDGER_CROSS_THREAD_CALL", SafetyDirective.STOP
            )

    def _identity_is_current(self, identity: ChunkRequestIdentity) -> bool:
        return (
            identity.policy_revision == self.policy_revision
            and identity.policy_epoch == self.policy_epoch
            and identity.takeover_generation == self.takeover_generation
            and identity.reset_generation == self.reset_generation
            and identity.request_clock_domain_id == self.clock_domain_id
            and identity.t_ref_clock_domain_id == self.clock_domain_id
        )

    def _raise(self, reason: str, directive: SafetyDirective) -> None:
        self._directive = directive
        raise RuntimeSafetyViolation(reason, directive)

    def _quarantine_pending(self, reason: str) -> None:
        if self._pending_entry_index is not None:
            current = self._entries[self._pending_entry_index]
            self._entries[self._pending_entry_index] = replace(
                current, status="quarantined", failure_reason=reason
            )
            self._pending_entry_index = None

    def _stop(self, reason: str) -> None:
        self._quarantine_pending(reason)
        self._active = None
        self._pinned_request = None
        self._stop_latched = True
        self._raise(reason, SafetyDirective.STOP)

    def _flush(self, reason: str) -> SafetyDirective:
        if self._pending_entry_index is not None:
            self._quarantine_pending(reason)
            self._stop_latched = True
        self._active = None
        self._pinned_request = None
        self._dispatch_count = 0
        self._last_selected_index = None
        self._directive = (
            SafetyDirective.STOP if self._stop_latched else SafetyDirective.HOLD
        )
        return self._directive

    def pin_request(self, identity: ChunkRequestIdentity) -> None:
        self._assert_owner()
        identity.validate()
        if self._stop_latched:
            self._raise("STAGE3_STOP_LATCHED", SafetyDirective.STOP)
        if (
            identity.request_clock_domain_id != self.clock_domain_id
            or identity.t_ref_clock_domain_id != self.clock_domain_id
        ):
            self._raise("STAGE3_CROSS_CLOCK_REQUEST_REJECTED", SafetyDirective.HOLD)
        if not self._identity_is_current(identity):
            self._raise("STAGE3_STALE_REQUEST_GENERATION", SafetyDirective.HOLD)
        if self._pinned_request is not None:
            self._raise("STAGE3_REQUEST_ALREADY_PINNED", SafetyDirective.HOLD)
        if self._active is not None and identity.t_ref_ns < self._active.t_ref_ns:
            self._raise("STAGE3_REFRESH_T_REF_REGRESSION", SafetyDirective.HOLD)
        self._pinned_request = identity

    def adopt_result(
        self, identity: ChunkResultIdentity, *, actions_cached: int
    ) -> None:
        self._assert_owner()
        identity.validate()
        if self._stop_latched:
            self._raise("STAGE3_STOP_LATCHED", SafetyDirective.STOP)
        if actions_cached != H50_ACTIONS_CACHED:
            self._raise("STAGE3_H50_CACHE_SIZE_MISMATCH", SafetyDirective.HOLD)
        if identity.result_clock_domain_id != self.clock_domain_id:
            self._raise("STAGE3_CROSS_CLOCK_RESULT_REJECTED", SafetyDirective.HOLD)
        if not self._identity_is_current(identity):
            self._raise("STAGE3_STALE_RESULT_GENERATION", SafetyDirective.HOLD)
        if self._pinned_request is None or (
            identity.request_identity() != self._pinned_request
        ):
            self._raise("STAGE3_RESULT_REQUEST_BINDING_MISMATCH", SafetyDirective.HOLD)
        if identity.result_id in self._seen_result_ids:
            self._raise("STAGE3_RESULT_REPLAY", SafetyDirective.HOLD)
        if self._pending_entry_index is not None:
            self._stop("STAGE3_RESULT_ADOPTED_WITH_UNACKED_DISPATCH")
        self._seen_result_ids.add(identity.result_id)
        self._pinned_request = None
        self._active = identity
        self._dispatch_count = 0
        self._last_selected_index = None
        self._directive = None

    def refresh_assessment(
        self, selection_ns: int, *, selection_clock_domain_id: str
    ) -> RefreshAssessment:
        self._assert_owner()
        if selection_clock_domain_id != self.clock_domain_id:
            self._raise("STAGE3_CROSS_CLOCK_SELECTION_REJECTED", SafetyDirective.HOLD)
        if self._active is None:
            self._raise("STAGE3_NO_ADOPTED_CHUNK", SafetyDirective.HOLD)
        selected_index = rational_h50_index(self._active.t_ref_ns, selection_ns)
        remaining_slots = max(0, self.limits.max_selected_index - selected_index)
        remaining_time_ns = (
            remaining_slots * NANOSECONDS_PER_SECOND // H50_MODEL_TIMEBASE_HZ
        )
        required = self.limits.refresh_required_coverage_ns
        return RefreshAssessment(
            selected_index=selected_index,
            remaining_time_ns=remaining_time_ns,
            required_coverage_ns=required,
            refresh_due=remaining_time_ns <= required,
            service_headroom_exhausted=(
                remaining_time_ns <= self.limits.refresh_worst_case_service_ns
            ),
        )

    def begin_dispatch(
        self,
        *,
        dispatch_sequence: int,
        selection_ns: int,
        selection_clock_domain_id: str,
        dispatch_ns: int,
        dispatch_clock_domain_id: str,
        selected_post_adapter_absolute7: tuple[float, ...],
        pose_command_id: str,
        gripper_command_id: str,
    ) -> SelectionLedgerEntry:
        self._assert_owner()
        if self._stop_latched:
            self._raise("STAGE3_STOP_LATCHED", SafetyDirective.STOP)
        if self._active is None:
            self._raise("STAGE3_NO_SAFE_ACTION_HOLD", SafetyDirective.HOLD)
        if self._pending_entry_index is not None:
            self._stop("STAGE3_PREVIOUS_DISPATCH_ACK_INCOMPLETE")
        if dispatch_sequence < 0 or (
            self._last_dispatch_sequence is not None
            and dispatch_sequence != self._last_dispatch_sequence + 1
        ):
            self._raise("STAGE3_DISPATCH_SEQUENCE_INVALID", SafetyDirective.HOLD)
        if not pose_command_id or not gripper_command_id:
            self._raise("STAGE3_COMMAND_IDENTITY_MISSING", SafetyDirective.HOLD)
        if (
            pose_command_id in self._seen_pose_command_ids
            or gripper_command_id in self._seen_gripper_command_ids
        ):
            self._raise("STAGE3_COMMAND_IDENTITY_REUSED", SafetyDirective.HOLD)
        if (
            selection_clock_domain_id != self.clock_domain_id
            or dispatch_clock_domain_id != self.clock_domain_id
        ):
            self._raise("STAGE3_CROSS_CLOCK_DISPATCH_REJECTED", SafetyDirective.HOLD)
        if dispatch_ns < selection_ns:
            self._raise("STAGE3_DISPATCH_PRECEDES_SELECTION", SafetyDirective.HOLD)

        age_ns = selection_ns - self._active.t_ref_ns
        selected_index = rational_h50_index(self._active.t_ref_ns, selection_ns)
        if age_ns > self.limits.max_chunk_age_ns:
            self._flush("STAGE3_MAX_CHUNK_AGE_EXCEEDED")
            self._raise("STAGE3_MAX_CHUNK_AGE_EXCEEDED", SafetyDirective.HOLD)
        if selected_index > self.limits.max_selected_index:
            self._flush("STAGE3_MAX_SELECTED_INDEX_EXCEEDED")
            self._raise("STAGE3_MAX_SELECTED_INDEX_EXCEEDED", SafetyDirective.HOLD)
        if self._dispatch_count >= self.limits.max_dispatch_count:
            self._flush("STAGE3_MAX_DISPATCH_COUNT_EXCEEDED")
            self._raise("STAGE3_MAX_DISPATCH_COUNT_EXCEEDED", SafetyDirective.HOLD)
        if (
            self._last_selected_index is not None
            and selected_index <= self._last_selected_index
        ):
            self._flush("STAGE3_SELECTED_INDEX_NOT_STRICTLY_INCREASING")
            self._raise(
                "STAGE3_SELECTED_INDEX_NOT_STRICTLY_INCREASING",
                SafetyDirective.HOLD,
            )

        refresh = self.refresh_assessment(
            selection_ns,
            selection_clock_domain_id=selection_clock_domain_id,
        )
        if refresh.service_headroom_exhausted:
            self._flush("STAGE3_REFRESH_SERVICE_HEADROOM_EXHAUSTED")
            self._raise(
                "STAGE3_REFRESH_SERVICE_HEADROOM_EXHAUSTED", SafetyDirective.HOLD
            )
        if refresh.refresh_due and self._pinned_request is None:
            self._raise("STAGE3_REFRESH_REQUIRED_BEFORE_DISPATCH", SafetyDirective.HOLD)

        entry = SelectionLedgerEntry(
            request_id=self._active.request_id,
            result_id=self._active.result_id,
            chunk_id=self._active.chunk_id,
            proposal_id=self._active.proposal_id,
            policy_revision=self._active.policy_revision,
            policy_epoch=self._active.policy_epoch,
            takeover_generation=self._active.takeover_generation,
            reset_generation=self._active.reset_generation,
            request_clock_domain_id=self._active.request_clock_domain_id,
            result_clock_domain_id=self._active.result_clock_domain_id,
            t_ref_clock_domain_id=self._active.t_ref_clock_domain_id,
            t_ref_ns=self._active.t_ref_ns,
            dispatch_sequence=dispatch_sequence,
            selected_index=selected_index,
            selection_ns=selection_ns,
            selection_clock_domain_id=selection_clock_domain_id,
            dispatch_ns=dispatch_ns,
            dispatch_clock_domain_id=dispatch_clock_domain_id,
            selected_post_adapter_absolute7=_absolute7(
                selected_post_adapter_absolute7
            ),
            pose_command_id=pose_command_id,
            gripper_command_id=gripper_command_id,
        )
        self._entries.append(entry)
        self._pending_entry_index = len(self._entries) - 1
        self._dispatch_count += 1
        self._last_selected_index = selected_index
        self._last_dispatch_sequence = dispatch_sequence
        self._seen_pose_command_ids.add(pose_command_id)
        self._seen_gripper_command_ids.add(gripper_command_id)
        self._directive = SafetyDirective.HOLD
        return entry

    def _pending(self, dispatch_sequence: int) -> SelectionLedgerEntry:
        if self._pending_entry_index is None:
            self._raise("STAGE3_NO_PENDING_DISPATCH", SafetyDirective.STOP)
        entry = self._entries[self._pending_entry_index]
        if entry.dispatch_sequence != dispatch_sequence:
            self._stop("STAGE3_ACK_DISPATCH_SEQUENCE_MISMATCH")
        return entry

    def _entry_for_ack(
        self, dispatch_sequence: int, *, ack_kind: str
    ) -> tuple[int, SelectionLedgerEntry]:
        if self._stop_latched:
            self._raise("STAGE3_STOP_LATCHED", SafetyDirective.STOP)
        for index in range(len(self._entries) - 1, -1, -1):
            entry = self._entries[index]
            if entry.dispatch_sequence != dispatch_sequence:
                continue
            if (
                entry.policy_revision != self.policy_revision
                or entry.policy_epoch != self.policy_epoch
                or entry.takeover_generation != self.takeover_generation
                or entry.reset_generation != self.reset_generation
                or entry.request_clock_domain_id != self.clock_domain_id
                or entry.result_clock_domain_id != self.clock_domain_id
            ):
                self._stop(f"STAGE3_{ack_kind}_ACK_AFTER_FLUSH")
            if entry.status == "quarantined":
                self._stop(f"STAGE3_{ack_kind}_ACK_AFTER_FLUSH")
            if entry.status not in {"awaiting_ack", "accepted"}:
                self._stop(f"STAGE3_{ack_kind}_ACK_LIFECYCLE_INVALID")
            if (
                entry.status == "awaiting_ack"
                and index != self._pending_entry_index
            ):
                self._stop(f"STAGE3_{ack_kind}_ACK_NOT_CURRENT")
            return index, entry
        self._stop(f"STAGE3_{ack_kind}_ACK_BEFORE_COMMAND")

    def record_pose_ack(
        self,
        *,
        dispatch_sequence: int,
        ack_id: str,
        accepted: bool,
        ack_ns: int,
        ack_clock_domain_id: str,
    ) -> None:
        self._assert_owner()
        index, entry = self._entry_for_ack(dispatch_sequence, ack_kind="POSE")
        if entry.pose_ack_id is not None:
            if (
                ack_id == entry.pose_ack_id
                and accepted is entry.pose_ack_accepted
                and ack_ns == entry.pose_ack_ns
                and ack_clock_domain_id == entry.pose_ack_clock_domain_id
            ):
                return
            self._stop("STAGE3_DUPLICATE_POSE_ACK_CONFLICT")
        if ack_clock_domain_id != self.clock_domain_id:
            self._stop("STAGE3_CROSS_CLOCK_POSE_ACK_REJECTED")
        if ack_id != entry.pose_command_id:
            self._stop("STAGE3_POSE_ACK_ID_MISMATCH")
        if accepted is not True:
            self._stop("STAGE3_POSE_ACK_REJECTED")
        if (
            ack_ns < entry.dispatch_ns
            or ack_ns - entry.dispatch_ns > self.limits.pose_ack_deadline_ns
        ):
            self._stop("STAGE3_POSE_ACK_STALE_OR_NONCAUSAL")
        self._entries[index] = replace(
            entry,
            pose_ack_id=ack_id,
            pose_ack_ns=ack_ns,
            pose_ack_clock_domain_id=ack_clock_domain_id,
            pose_ack_accepted=True,
        )

    def record_gripper_ack(
        self,
        *,
        dispatch_sequence: int,
        ack_id: str,
        accepted: bool,
        ack_ns: int,
        ack_clock_domain_id: str,
    ) -> None:
        self._assert_owner()
        index, entry = self._entry_for_ack(dispatch_sequence, ack_kind="GRIPPER")
        if entry.gripper_ack_id is not None:
            if (
                ack_id == entry.gripper_ack_id
                and accepted is entry.gripper_ack_accepted
                and ack_ns == entry.gripper_ack_ns
                and ack_clock_domain_id == entry.gripper_ack_clock_domain_id
            ):
                return
            self._stop("STAGE3_DUPLICATE_GRIPPER_ACK_CONFLICT")
        if ack_clock_domain_id != self.clock_domain_id:
            self._stop("STAGE3_CROSS_CLOCK_GRIPPER_ACK_REJECTED")
        if ack_id != entry.gripper_command_id:
            self._stop("STAGE3_GRIPPER_ACK_ID_MISMATCH")
        if accepted is not True:
            self._stop("STAGE3_GRIPPER_ACK_REJECTED")
        if (
            ack_ns < entry.dispatch_ns
            or ack_ns - entry.dispatch_ns > self.limits.gripper_ack_deadline_ns
        ):
            self._stop("STAGE3_GRIPPER_ACK_STALE_OR_NONCAUSAL")
        self._entries[index] = replace(
            entry,
            gripper_ack_id=ack_id,
            gripper_ack_ns=ack_ns,
            gripper_ack_clock_domain_id=ack_clock_domain_id,
            gripper_ack_accepted=True,
        )

    def expire_missing_acks(
        self, now_ns: int, *, now_clock_domain_id: str
    ) -> None:
        self._assert_owner()
        if self._stop_latched:
            self._raise("STAGE3_STOP_LATCHED", SafetyDirective.STOP)
        if now_clock_domain_id != self.clock_domain_id:
            self._stop("STAGE3_CROSS_CLOCK_ACK_EXPIRY_REJECTED")
        if self._pending_entry_index is None:
            return
        entry = self._entries[self._pending_entry_index]
        if (
            entry.pose_ack_accepted is not True
            and now_ns - entry.dispatch_ns > self.limits.pose_ack_deadline_ns
        ):
            self._stop("STAGE3_POSE_ACK_MISSING")
        if (
            entry.gripper_ack_accepted is not True
            and now_ns - entry.dispatch_ns > self.limits.gripper_ack_deadline_ns
        ):
            self._stop("STAGE3_GRIPPER_ACK_MISSING")

    def commit_dispatch(self, dispatch_sequence: int) -> SelectionLedgerEntry:
        self._assert_owner()
        if self._stop_latched:
            self._raise("STAGE3_STOP_LATCHED", SafetyDirective.STOP)
        entry = self._pending(dispatch_sequence)
        if entry.pose_ack_accepted is not True:
            self._stop("STAGE3_POSE_ACK_MISSING")
        if entry.gripper_ack_accepted is not True:
            self._stop("STAGE3_GRIPPER_ACK_MISSING")
        accepted = replace(entry, status="accepted")
        self._entries[self._pending_entry_index] = accepted
        self._pending_entry_index = None
        self._directive = None
        return accepted

    def human_takeover_flush(
        self, *, takeover_generation: int, policy_epoch: int
    ) -> SafetyDirective:
        self._assert_owner()
        if (
            takeover_generation <= self.takeover_generation
            or policy_epoch <= self.policy_epoch
        ):
            self._raise("STAGE3_TAKEOVER_GENERATION_INVALID", SafetyDirective.STOP)
        self.takeover_generation = takeover_generation
        self.policy_epoch = policy_epoch
        return self._flush("STAGE3_HUMAN_TAKEOVER_FLUSH")

    def reset_home_flush(self, *, reset_generation: int) -> SafetyDirective:
        self._assert_owner()
        if reset_generation <= self.reset_generation:
            self._raise("STAGE3_RESET_GENERATION_INVALID", SafetyDirective.STOP)
        self.reset_generation = reset_generation
        self._quarantine_pending("STAGE3_RESET_HOME_FLUSH")
        self._stop_latched = False
        return self._flush("STAGE3_RESET_HOME_FLUSH")

    def policy_revision_flush(
        self, *, policy_revision: str, policy_epoch: int
    ) -> SafetyDirective:
        self._assert_owner()
        if not policy_revision or (
            policy_revision == self.policy_revision
            or policy_epoch <= self.policy_epoch
        ):
            self._raise("STAGE3_POLICY_REVISION_GENERATION_INVALID", SafetyDirective.STOP)
        self.policy_revision = policy_revision
        self.policy_epoch = policy_epoch
        return self._flush("STAGE3_POLICY_REVISION_FLUSH")


def project_acknowledged_runtime_macro(
    entries: Sequence[SelectionLedgerEntry],
    grid_monotonic_ns: Sequence[int],
    *,
    grid_clock_domain_id: str,
    max_ack_age_ms: float,
):
    """Project only ACK-authoritative post-adapter actions into Stage-3."""

    from .transition import causal_zoh_ack_macro

    return causal_zoh_ack_macro(
        [
            entry.to_accepted_ack(clock_domain_id=grid_clock_domain_id)
            for entry in entries
        ],
        grid_monotonic_ns,
        max_ack_age_ms=max_ack_age_ms,
    )
