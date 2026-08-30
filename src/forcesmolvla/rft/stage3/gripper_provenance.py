"""CPU-only gripper command provenance and held-target authority contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
from typing import Any


VALID_TERMINAL_OUTCOMES = frozenset({"reached", "stalled"})
INVALID_TERMINAL_OUTCOMES = frozenset(
    {"rejected", "send_error", "result_error", "not_reached"}
)
GRIPPER_NOOP_ACK_POLICY = "BOUND"
GRIPPER_FEEDBACK_FRESHNESS_BOUND = True
GRIPPER_TERMINAL_SEAL_REQUIRED = True
FULL_ACTION7_ACK_CLOSURE_PRODUCTION = False
PRODUCTION_INTEGRATION_BLOCKED_ON_GRIPPER_ACK = True


class GripperProvenanceError(RuntimeError):
    """A fail-closed gripper provenance violation."""


class GripperLifecycle(str, Enum):
    UNBOUND = "UNBOUND"
    COMMAND_PENDING = "COMMAND_PENDING"
    ACCEPTED_ACTIVE = "ACCEPTED_ACTIVE"
    TERMINAL_REACHED = "TERMINAL_REACHED"
    TERMINAL_STALLED = "TERMINAL_STALLED"
    INVALIDATED = "INVALIDATED"


class GripperAuthorityKind(str, Enum):
    NEW_COMMAND = "NEW_COMMAND"
    HELD_FROM_ACCEPTED_COMMAND = "HELD_FROM_ACCEPTED_COMMAND"


@dataclass(frozen=True)
class GripperGeneration:
    episode_id: str
    reset_generation: int
    takeover_generation: int
    policy_revision: str
    policy_epoch: int

    def validate(self) -> "GripperGeneration":
        if not self.episode_id or not self.policy_revision:
            raise GripperProvenanceError("GRIPPER_GENERATION_IDENTITY_MISSING")
        for value in (
            self.reset_generation,
            self.takeover_generation,
            self.policy_epoch,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GripperProvenanceError("GRIPPER_GENERATION_COUNTER_INVALID")
        return self


@dataclass(frozen=True)
class GripperFeedback:
    measured_width_m: float
    measured_state: str
    feedback_monotonic_ns: int
    clock_domain_id: str
    generation: GripperGeneration

    def validate(self) -> "GripperFeedback":
        self.generation.validate()
        if (
            not math.isfinite(self.measured_width_m)
            or not 0.0 <= self.measured_width_m <= 0.1
            or not self.measured_state
            or self.feedback_monotonic_ns <= 0
            or not self.clock_domain_id
        ):
            raise GripperProvenanceError("GRIPPER_FEEDBACK_INVALID")
        return self


@dataclass(frozen=True)
class GripperCommandRecord:
    local_goal_sequence: int
    requested_state: str
    requested_width_m: float
    started_monotonic_ns: int
    clock_domain_id: str
    generation: GripperGeneration
    lifecycle: GripperLifecycle = GripperLifecycle.COMMAND_PENDING
    action_goal_id: str | None = None
    accepted_monotonic_ns: int | None = None
    finished_monotonic_ns: int | None = None
    terminal_outcome: str | None = None
    invalidation_reason: str | None = None


@dataclass(frozen=True)
class GripperAuthorityEvidence:
    transition_id: str
    authority_kind: GripperAuthorityKind
    origin_local_goal_sequence: int
    origin_action_goal_id: str
    origin_accepted_monotonic_ns: int
    requested_state: str
    requested_width_m: float
    authority_monotonic_ns: int
    clock_domain_id: str
    generation: GripperGeneration
    feedback_width_m: float | None = None
    feedback_state: str | None = None
    feedback_monotonic_ns: int | None = None
    feedback_age_ns: int | None = None
    terminal_outcome: str | None = None
    terminal_finished_monotonic_ns: int | None = None
    terminal_sealed: bool = False

    def validate(self) -> "GripperAuthorityEvidence":
        self.generation.validate()
        if (
            not self.transition_id
            or self.origin_local_goal_sequence <= 0
            or not self.origin_action_goal_id
            or self.origin_accepted_monotonic_ns <= 0
            or not self.requested_state
            or not math.isfinite(self.requested_width_m)
            or not 0.0 <= self.requested_width_m <= 0.1
            or self.authority_monotonic_ns < self.origin_accepted_monotonic_ns
            or not self.clock_domain_id
        ):
            raise GripperProvenanceError("GRIPPER_AUTHORITY_IDENTITY_INVALID")
        feedback = (
            self.feedback_width_m,
            self.feedback_state,
            self.feedback_monotonic_ns,
            self.feedback_age_ns,
        )
        if self.authority_kind is GripperAuthorityKind.HELD_FROM_ACCEPTED_COMMAND:
            if any(value is None for value in feedback):
                raise GripperProvenanceError("GRIPPER_HELD_FEEDBACK_MISSING")
            if self.feedback_age_ns is None or self.feedback_age_ns < 0:
                raise GripperProvenanceError("GRIPPER_HELD_FEEDBACK_AGE_INVALID")
        elif any(value is not None for value in feedback):
            raise GripperProvenanceError("GRIPPER_NEW_COMMAND_FEEDBACK_AMBIGUOUS")
        if self.terminal_sealed:
            if (
                self.terminal_outcome not in VALID_TERMINAL_OUTCOMES
                or self.terminal_finished_monotonic_ns is None
                or self.terminal_finished_monotonic_ns
                < self.origin_accepted_monotonic_ns
            ):
                raise GripperProvenanceError("GRIPPER_TERMINAL_SEAL_INVALID")
        return self


@dataclass(frozen=True)
class PoseAcceptedAuthority:
    transition_id: str
    pose_command_id: str
    pose_ack_id: str
    pose_ack_monotonic_ns: int
    selected_post_adapter_tcp6: tuple[float, ...]
    declared_gripper_origin_action_goal_id: str
    clock_domain_id: str
    generation: GripperGeneration
    accepted: bool = True

    def validate(self) -> "PoseAcceptedAuthority":
        self.generation.validate()
        if (
            not self.transition_id
            or not self.pose_command_id
            or self.pose_ack_id != self.pose_command_id
            or self.pose_ack_monotonic_ns <= 0
            or not self.declared_gripper_origin_action_goal_id
            or not self.clock_domain_id
            or self.accepted is not True
            or len(self.selected_post_adapter_tcp6) != 6
            or not all(math.isfinite(value) for value in self.selected_post_adapter_tcp6)
        ):
            raise GripperProvenanceError("POSE_ACCEPTED_AUTHORITY_INVALID")
        return self


@dataclass(frozen=True)
class FullAction7Authority:
    transition_id: str
    accepted_absolute_action7: tuple[float, ...]
    pose: PoseAcceptedAuthority
    gripper: GripperAuthorityEvidence
    terminal_sealed_for_replay: bool

    def validate(self) -> "FullAction7Authority":
        self.pose.validate()
        self.gripper.validate()
        if (
            self.transition_id != self.pose.transition_id
            or self.transition_id != self.gripper.transition_id
            or self.pose.generation != self.gripper.generation
            or self.pose.clock_domain_id != self.gripper.clock_domain_id
            or self.pose.declared_gripper_origin_action_goal_id
            != self.gripper.origin_action_goal_id
            or len(self.accepted_absolute_action7) != 7
            or not all(math.isfinite(value) for value in self.accepted_absolute_action7)
            or self.terminal_sealed_for_replay is not self.gripper.terminal_sealed
        ):
            raise GripperProvenanceError("FULL_ACTION7_AUTHORITY_INVALID")
        return self


@dataclass(frozen=True)
class _StagedTransition:
    evidence: GripperAuthorityEvidence
    quarantined: bool = False
    quarantine_reason: str | None = None


class GripperProvenanceLedger:
    """Single-owner lifecycle ledger; held authority never invents a command/ACK."""

    def __init__(
        self,
        *,
        generation: GripperGeneration,
        clock_domain_id: str,
        max_feedback_age_ns: int,
        requested_width_tolerance_m: float = 1.0e-9,
    ) -> None:
        generation.validate()
        if (
            not clock_domain_id
            or max_feedback_age_ns <= 0
            or not math.isfinite(requested_width_tolerance_m)
            or requested_width_tolerance_m < 0.0
        ):
            raise GripperProvenanceError("GRIPPER_LEDGER_CONFIG_INVALID")
        self.generation = generation
        self.clock_domain_id = clock_domain_id
        self.max_feedback_age_ns = max_feedback_age_ns
        self.requested_width_tolerance_m = requested_width_tolerance_m
        self._owner_thread_id = threading.get_ident()
        self._commands: dict[int, GripperCommandRecord] = {}
        self._pending_sequence: int | None = None
        self._current_sequence: int | None = None
        self._transitions: dict[str, _StagedTransition] = {}
        self._episode_failure: str | None = None

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise GripperProvenanceError("GRIPPER_LEDGER_CROSS_THREAD_ACCESS")

    @property
    def lifecycle(self) -> GripperLifecycle:
        self._assert_owner()
        if self._pending_sequence is not None:
            return GripperLifecycle.COMMAND_PENDING
        if self._current_sequence is None:
            return GripperLifecycle.UNBOUND
        return self._commands[self._current_sequence].lifecycle

    @property
    def episode_failure(self) -> str | None:
        self._assert_owner()
        return self._episode_failure

    def command(self, local_goal_sequence: int) -> GripperCommandRecord:
        self._assert_owner()
        try:
            return self._commands[local_goal_sequence]
        except KeyError as error:
            raise GripperProvenanceError("GRIPPER_COMMAND_UNKNOWN") from error

    def transition_evidence(self, transition_id: str) -> GripperAuthorityEvidence:
        self._assert_owner()
        try:
            return self._transitions[transition_id].evidence
        except KeyError as error:
            raise GripperProvenanceError("GRIPPER_TRANSITION_UNKNOWN") from error

    def transition_quarantined(self, transition_id: str) -> bool:
        self._assert_owner()
        try:
            return self._transitions[transition_id].quarantined
        except KeyError as error:
            raise GripperProvenanceError("GRIPPER_TRANSITION_UNKNOWN") from error

    def eligible_for_replay(self, transition_id: str) -> bool:
        self._assert_owner()
        staged = self._transitions.get(transition_id)
        return bool(
            staged is not None
            and not staged.quarantined
            and staged.evidence.terminal_sealed
        )

    def _validate_event(self, generation: GripperGeneration, clock: str) -> None:
        generation.validate()
        if generation != self.generation:
            raise GripperProvenanceError("GRIPPER_EVENT_GENERATION_STALE")
        if clock != self.clock_domain_id:
            raise GripperProvenanceError("GRIPPER_EVENT_CLOCK_DOMAIN_MISMATCH")

    def _same_target(self, record: GripperCommandRecord, state: str, width: float) -> bool:
        return (
            record.requested_state == state
            and abs(record.requested_width_m - width)
            <= self.requested_width_tolerance_m
        )

    def _quarantine_dependencies(
        self, sequence: int, reason: str, *, include_sealed: bool = False
    ) -> None:
        for transition_id, staged in tuple(self._transitions.items()):
            if (
                staged.evidence.origin_local_goal_sequence == sequence
                and (include_sealed or not staged.evidence.terminal_sealed)
            ):
                self._transitions[transition_id] = replace(
                    staged, quarantined=True, quarantine_reason=reason
                )

    def _invalidate_sequence(
        self, sequence: int, reason: str, *, include_sealed: bool = False
    ) -> None:
        record = self._commands[sequence]
        self._commands[sequence] = replace(
            record,
            lifecycle=GripperLifecycle.INVALIDATED,
            invalidation_reason=reason,
        )
        self._quarantine_dependencies(
            sequence, reason, include_sealed=include_sealed
        )
        if self._pending_sequence == sequence:
            self._pending_sequence = None
        if self._current_sequence == sequence:
            self._current_sequence = None

    def begin_command(
        self,
        *,
        local_goal_sequence: int,
        requested_state: str,
        requested_width_m: float,
        started_monotonic_ns: int,
        generation: GripperGeneration,
        clock_domain_id: str,
    ) -> None:
        self._assert_owner()
        self._validate_event(generation, clock_domain_id)
        if (
            local_goal_sequence <= 0
            or local_goal_sequence in self._commands
            or not requested_state
            or not math.isfinite(requested_width_m)
            or not 0.0 <= requested_width_m <= 0.1
            or started_monotonic_ns <= 0
        ):
            raise GripperProvenanceError("GRIPPER_COMMAND_START_INVALID")
        if self._pending_sequence is not None:
            pending = self._pending_sequence
            self._invalidate_sequence(pending, "CONFLICTING_COMMAND_WHILE_PENDING")
            self._episode_failure = "CONFLICTING_COMMAND_WHILE_PENDING"
            raise GripperProvenanceError(self._episode_failure)
        if self._current_sequence is not None:
            current = self._commands[self._current_sequence]
            if not self._same_target(current, requested_state, requested_width_m):
                self._invalidate_sequence(
                    current.local_goal_sequence, "CONFLICTING_NEW_COMMAND"
                )
        self._commands[local_goal_sequence] = GripperCommandRecord(
            local_goal_sequence=local_goal_sequence,
            requested_state=requested_state,
            requested_width_m=requested_width_m,
            started_monotonic_ns=started_monotonic_ns,
            clock_domain_id=clock_domain_id,
            generation=generation,
        )
        self._pending_sequence = local_goal_sequence

    def accept_command(
        self,
        *,
        local_goal_sequence: int,
        action_goal_id: str,
        accepted_monotonic_ns: int,
        generation: GripperGeneration,
        clock_domain_id: str,
    ) -> None:
        self._assert_owner()
        self._validate_event(generation, clock_domain_id)
        record = self.command(local_goal_sequence)
        if record.lifecycle is not GripperLifecycle.COMMAND_PENDING:
            if (
                record.lifecycle is GripperLifecycle.ACCEPTED_ACTIVE
                and action_goal_id == record.action_goal_id
                and accepted_monotonic_ns == record.accepted_monotonic_ns
            ):
                return
            if record.lifecycle is not GripperLifecycle.INVALIDATED:
                self._invalidate_sequence(local_goal_sequence, "ACCEPT_EVENT_CONFLICT")
            raise GripperProvenanceError("GRIPPER_ACCEPT_EVENT_CONFLICT")
        if (
            self._pending_sequence != local_goal_sequence
            or not action_goal_id
            or accepted_monotonic_ns < record.started_monotonic_ns
        ):
            self._invalidate_sequence(local_goal_sequence, "ACCEPT_EVENT_INVALID")
            raise GripperProvenanceError("GRIPPER_ACCEPT_EVENT_INVALID")
        if self._current_sequence is not None:
            self._invalidate_sequence(self._current_sequence, "COMMAND_SUPERSEDED")
        self._commands[local_goal_sequence] = replace(
            record,
            lifecycle=GripperLifecycle.ACCEPTED_ACTIVE,
            action_goal_id=action_goal_id,
            accepted_monotonic_ns=accepted_monotonic_ns,
        )
        self._pending_sequence = None
        self._current_sequence = local_goal_sequence

    def record_terminal(
        self,
        *,
        local_goal_sequence: int,
        action_goal_id: str | None,
        outcome: str,
        finished_monotonic_ns: int,
        generation: GripperGeneration,
        clock_domain_id: str,
    ) -> None:
        self._assert_owner()
        self._validate_event(generation, clock_domain_id)
        record = self.command(local_goal_sequence)
        if record.lifecycle in {
            GripperLifecycle.TERMINAL_REACHED,
            GripperLifecycle.TERMINAL_STALLED,
            GripperLifecycle.INVALIDATED,
        }:
            if record.lifecycle is not GripperLifecycle.INVALIDATED:
                self._invalidate_sequence(
                    local_goal_sequence,
                    "DUPLICATE_TERMINAL_EVENT",
                    include_sealed=True,
                )
            self._episode_failure = "LATE_DUPLICATE_OR_CONFLICTING_TERMINAL"
            raise GripperProvenanceError(self._episode_failure)
        if outcome not in VALID_TERMINAL_OUTCOMES:
            reason = (
                f"TERMINAL_{outcome.upper()}"
                if outcome in INVALID_TERMINAL_OUTCOMES
                else "TERMINAL_OUTCOME_UNKNOWN"
            )
            self._invalidate_sequence(
                local_goal_sequence, reason, include_sealed=True
            )
            self._episode_failure = reason
            raise GripperProvenanceError(reason)
        if (
            record.lifecycle is not GripperLifecycle.ACCEPTED_ACTIVE
            or action_goal_id != record.action_goal_id
            or record.accepted_monotonic_ns is None
            or finished_monotonic_ns < record.accepted_monotonic_ns
        ):
            self._invalidate_sequence(
                local_goal_sequence,
                "TERMINAL_EVENT_INVALID",
                include_sealed=True,
            )
            self._episode_failure = "TERMINAL_EVENT_INVALID"
            raise GripperProvenanceError(self._episode_failure)
        lifecycle = (
            GripperLifecycle.TERMINAL_REACHED
            if outcome == "reached"
            else GripperLifecycle.TERMINAL_STALLED
        )
        self._commands[local_goal_sequence] = replace(
            record,
            lifecycle=lifecycle,
            finished_monotonic_ns=finished_monotonic_ns,
            terminal_outcome=outcome,
        )

    def _stage_authority(
        self,
        *,
        transition_id: str,
        authority_kind: GripperAuthorityKind,
        authority_monotonic_ns: int,
        feedback: GripperFeedback | None,
    ) -> GripperAuthorityEvidence:
        if not transition_id or transition_id in self._transitions:
            raise GripperProvenanceError("GRIPPER_TRANSITION_ID_INVALID")
        if self._pending_sequence is not None or self._current_sequence is None:
            raise GripperProvenanceError("GRIPPER_ACCEPTED_LEASE_UNAVAILABLE")
        record = self._commands[self._current_sequence]
        if record.lifecycle not in {
            GripperLifecycle.ACCEPTED_ACTIVE,
            GripperLifecycle.TERMINAL_REACHED,
            GripperLifecycle.TERMINAL_STALLED,
        }:
            raise GripperProvenanceError("GRIPPER_ACCEPTED_LEASE_INVALID")
        assert record.action_goal_id is not None
        assert record.accepted_monotonic_ns is not None
        if authority_monotonic_ns < record.accepted_monotonic_ns:
            raise GripperProvenanceError("GRIPPER_AUTHORITY_PRECEDES_ACCEPTANCE")
        feedback_fields: dict[str, Any] = {}
        if authority_kind is GripperAuthorityKind.HELD_FROM_ACCEPTED_COMMAND:
            if feedback is None:
                raise GripperProvenanceError("GRIPPER_HELD_FEEDBACK_MISSING")
            feedback.validate()
            self._validate_event(feedback.generation, feedback.clock_domain_id)
            age_ns = authority_monotonic_ns - feedback.feedback_monotonic_ns
            if (
                feedback.feedback_monotonic_ns < record.accepted_monotonic_ns
                or age_ns < 0
                or age_ns > self.max_feedback_age_ns
            ):
                raise GripperProvenanceError("GRIPPER_HELD_FEEDBACK_STALE")
            feedback_fields = {
                "feedback_width_m": feedback.measured_width_m,
                "feedback_state": feedback.measured_state,
                "feedback_monotonic_ns": feedback.feedback_monotonic_ns,
                "feedback_age_ns": age_ns,
            }
        elif feedback is not None:
            raise GripperProvenanceError("GRIPPER_NEW_COMMAND_FEEDBACK_AMBIGUOUS")
        evidence = GripperAuthorityEvidence(
            transition_id=transition_id,
            authority_kind=authority_kind,
            origin_local_goal_sequence=record.local_goal_sequence,
            origin_action_goal_id=record.action_goal_id,
            origin_accepted_monotonic_ns=record.accepted_monotonic_ns,
            requested_state=record.requested_state,
            requested_width_m=record.requested_width_m,
            authority_monotonic_ns=authority_monotonic_ns,
            clock_domain_id=self.clock_domain_id,
            generation=self.generation,
            terminal_outcome=record.terminal_outcome,
            terminal_finished_monotonic_ns=record.finished_monotonic_ns,
            **feedback_fields,
        ).validate()
        self._transitions[transition_id] = _StagedTransition(evidence=evidence)
        return evidence

    def new_command_authority(
        self, *, transition_id: str, authority_monotonic_ns: int
    ) -> GripperAuthorityEvidence:
        self._assert_owner()
        return self._stage_authority(
            transition_id=transition_id,
            authority_kind=GripperAuthorityKind.NEW_COMMAND,
            authority_monotonic_ns=authority_monotonic_ns,
            feedback=None,
        )

    def held_authority(
        self,
        *,
        transition_id: str,
        requested_state: str,
        requested_width_m: float,
        authority_monotonic_ns: int,
        feedback: GripperFeedback,
    ) -> GripperAuthorityEvidence:
        self._assert_owner()
        if self._current_sequence is None:
            raise GripperProvenanceError("GRIPPER_HELD_WITHOUT_ACCEPTED_ORIGIN")
        record = self._commands[self._current_sequence]
        if not self._same_target(record, requested_state, requested_width_m):
            raise GripperProvenanceError("GRIPPER_HELD_TARGET_CONFLICT")
        return self._stage_authority(
            transition_id=transition_id,
            authority_kind=GripperAuthorityKind.HELD_FROM_ACCEPTED_COMMAND,
            authority_monotonic_ns=authority_monotonic_ns,
            feedback=feedback,
        )

    def seal_episode(self) -> tuple[GripperAuthorityEvidence, ...]:
        self._assert_owner()
        if self._episode_failure is not None or self._pending_sequence is not None:
            raise GripperProvenanceError("GRIPPER_EPISODE_TERMINAL_SEAL_BLOCKED")
        current = [
            (transition_id, staged)
            for transition_id, staged in self._transitions.items()
            if staged.evidence.generation == self.generation
        ]
        if not current:
            raise GripperProvenanceError("GRIPPER_EPISODE_HAS_NO_AUTHORITY")
        sealed: list[GripperAuthorityEvidence] = []
        for transition_id, staged in current:
            if staged.quarantined:
                raise GripperProvenanceError("GRIPPER_QUARANTINED_TRANSITION_AT_SEAL")
            command = self._commands[staged.evidence.origin_local_goal_sequence]
            if (
                command.lifecycle
                not in {
                    GripperLifecycle.TERMINAL_REACHED,
                    GripperLifecycle.TERMINAL_STALLED,
                }
                or command.terminal_outcome not in VALID_TERMINAL_OUTCOMES
                or command.finished_monotonic_ns is None
            ):
                raise GripperProvenanceError("GRIPPER_TERMINAL_PAIRING_INCOMPLETE")
            evidence = replace(
                staged.evidence,
                terminal_outcome=command.terminal_outcome,
                terminal_finished_monotonic_ns=command.finished_monotonic_ns,
                terminal_sealed=True,
            ).validate()
            self._transitions[transition_id] = replace(staged, evidence=evidence)
            sealed.append(evidence)
        return tuple(sealed)

    def _generation_boundary(
        self, *, new_generation: GripperGeneration, reason: str
    ) -> None:
        self._assert_owner()
        new_generation.validate()
        if new_generation == self.generation:
            raise GripperProvenanceError("GRIPPER_GENERATION_DID_NOT_CHANGE")
        for sequence, record in tuple(self._commands.items()):
            if (
                record.generation == self.generation
                and record.lifecycle is not GripperLifecycle.INVALIDATED
            ):
                self._invalidate_sequence(sequence, reason)
        old_episode = self.generation.episode_id
        self.generation = new_generation
        self._pending_sequence = None
        self._current_sequence = None
        if new_generation.episode_id != old_episode:
            self._episode_failure = None

    def reset_home(self, *, new_generation: GripperGeneration) -> None:
        if new_generation.reset_generation <= self.generation.reset_generation:
            raise GripperProvenanceError("GRIPPER_RESET_GENERATION_NOT_ADVANCED")
        self._generation_boundary(new_generation=new_generation, reason="RESET_HOME")

    def human_takeover(self, *, new_generation: GripperGeneration) -> None:
        if new_generation.takeover_generation <= self.generation.takeover_generation:
            raise GripperProvenanceError("GRIPPER_TAKEOVER_GENERATION_NOT_ADVANCED")
        self._generation_boundary(new_generation=new_generation, reason="HUMAN_TAKEOVER")

    def policy_revision(self, *, new_generation: GripperGeneration) -> None:
        if (
            new_generation.policy_revision == self.generation.policy_revision
            and new_generation.policy_epoch <= self.generation.policy_epoch
        ):
            raise GripperProvenanceError("GRIPPER_POLICY_REVISION_NOT_ADVANCED")
        self._generation_boundary(new_generation=new_generation, reason="POLICY_REVISION")

    def episode_change(self, *, new_generation: GripperGeneration) -> None:
        if new_generation.episode_id == self.generation.episode_id:
            raise GripperProvenanceError("GRIPPER_EPISODE_DID_NOT_CHANGE")
        self._generation_boundary(new_generation=new_generation, reason="EPISODE_CHANGE")


def pose_authority_from_g7c1_entry(
    entry: Any,
    *,
    transition_id: str,
    episode_id: str,
) -> PoseAcceptedAuthority:
    """Read pose-only authority without weakening gripper ACK checks."""

    generation = GripperGeneration(
        episode_id=episode_id,
        reset_generation=entry.reset_generation,
        takeover_generation=entry.takeover_generation,
        policy_revision=entry.policy_revision,
        policy_epoch=entry.policy_epoch,
    ).validate()
    if (
        entry.pose_ack_accepted is not True
        or entry.pose_ack_id != entry.pose_command_id
        or entry.pose_ack_ns is None
        or entry.pose_ack_clock_domain_id != entry.dispatch_clock_domain_id
        or entry.pose_ack_ns < entry.dispatch_ns
        or len(entry.selected_post_adapter_absolute7) != 7
    ):
        raise GripperProvenanceError("G7C1_POSE_ACK_NOT_AUTHORITATIVE")
    return PoseAcceptedAuthority(
        transition_id=transition_id,
        pose_command_id=entry.pose_command_id,
        pose_ack_id=entry.pose_ack_id,
        pose_ack_monotonic_ns=entry.pose_ack_ns,
        selected_post_adapter_tcp6=tuple(entry.selected_post_adapter_absolute7[:6]),
        declared_gripper_origin_action_goal_id=entry.gripper_command_id,
        clock_domain_id=entry.dispatch_clock_domain_id,
        generation=generation,
    ).validate()


def close_full_action7_authority(
    *,
    pose: PoseAcceptedAuthority,
    selected_post_adapter_tcp6: tuple[float, ...],
    selected_gripper_width_m: float,
    gripper: GripperAuthorityEvidence,
    requested_width_tolerance_m: float = 1.0e-9,
) -> FullAction7Authority:
    """Bind Pose ACK and real gripper provenance; value equality alone is insufficient."""

    pose.validate()
    gripper.validate()
    tcp6 = tuple(float(value) for value in selected_post_adapter_tcp6)
    if (
        tcp6 != pose.selected_post_adapter_tcp6
        or not math.isfinite(selected_gripper_width_m)
        or abs(selected_gripper_width_m - gripper.requested_width_m)
        > requested_width_tolerance_m
    ):
        raise GripperProvenanceError("FULL_ACTION7_SELECTED_ACTION_MISMATCH")
    return FullAction7Authority(
        transition_id=pose.transition_id,
        accepted_absolute_action7=tcp6 + (float(selected_gripper_width_m),),
        pose=pose,
        gripper=gripper,
        terminal_sealed_for_replay=gripper.terminal_sealed,
    ).validate()
