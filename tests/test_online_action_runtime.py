from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from forcesmolvla.rft.online.action_runtime import (
    ACTION_DELTA_DENORMALIZATION_ONCE,
    ACTION_SLOT_FIFO_PRESENT,
    CONTRACT_TRANSITION_MACRO_HZ,
    CONTROLLER_INTERNAL_SERVO_HZ,
    CURRENT_LOW_WATERMARK_APPROVED,
    CURRENT_LOW_WATERMARK_COVERAGE_NS,
    CONCURRENT_MAX_SERVICE_LATENCY_NS,
    PRODUCTION_DURABLE_RESUME,
    GRIPPER_NOOP_ACK_POLICY,
    H50_ACTIONS_CACHED,
    H50_MODEL_TIMEBASE_HZ,
    MAX_SELECTIONS_PER_ADOPTED_CHUNK,
    OBSERVATION_STATE_NORMALIZATION_ONCE,
    OBSERVATION_WRENCH_NORMALIZATION_ONCE,
    POLICY_INFERENCE_10HZ_REQUIRED,
    POLICY_REQUEST_HZ_MEASURED,
    POLICY_REQUEST_TRIGGER,
    POSE_REFERENCE_DISPATCH_HZ,
    PRODUCTION_SAFE_INFERENCE_REFRESH_RATE,
    PRODUCTION_INTEGRATION_BLOCKED_ON_GRIPPER_ACK,
    PRODUCTION_RUNTIME_LEDGER_RESUME,
    PRODUCTION_TRANSITION_COMMIT_HZ,
    RECORDED_LIVE_ACCEPTED_MACRO_NORMALIZATION_ONCE,
    RUNTIME_CLOCK_DOMAIN_BOUND,
    RUNTIME_LEDGER_PERSISTED,
    RUNTIME_THREAD_OWNERSHIP,
    SELECTED_INDEX_POLICY,
    ONLINE_REPLAY_PROJECTION_GRID_HZ,
    ChunkRequestIdentity,
    ChunkResultIdentity,
    RationalH50SelectionLedger,
    RuntimeSafetyLimits,
    RuntimeSafetyViolation,
    SafetyDirective,
    CROSS_CLOCK_TIMESTAMP_REJECTED,
    FULL_ACTION7_ACK_CLOSURE,
    project_acknowledged_runtime_macro,
    rational_h50_index,
)


ROOT = Path(__file__).parents[1]
T_REF_NS = 1_000_000_000
CLOCK_DOMAIN_ID = "same-host-monotonic-v1"
POST_ADAPTER_ABSOLUTE7 = (0.4, 0.0, 0.3, 0.0, 0.1, 0.0, 0.085)


def limits(**overrides: int) -> RuntimeSafetyLimits:
    values = {
        "max_chunk_age_ns": 1_500_000_000,
        "max_selected_index": 49,
        "max_dispatch_count": 8,
        "refresh_worst_case_service_ns": CONCURRENT_MAX_SERVICE_LATENCY_NS,
        "refresh_additional_headroom_ns": 50_000_000,
        "pose_ack_deadline_ns": 20_000_000,
        "gripper_ack_deadline_ns": 30_000_000,
    }
    values.update(overrides)
    return RuntimeSafetyLimits(**values)


def request(
    *,
    request_id: str = "request-0",
    chunk_id: str = "chunk-0",
    proposal_id: str = "proposal-0",
    revision: str = "revision-0",
    epoch: int = 0,
    takeover_generation: int = 0,
    reset_generation: int = 0,
    request_clock_domain_id: str = CLOCK_DOMAIN_ID,
    t_ref_clock_domain_id: str = CLOCK_DOMAIN_ID,
    t_ref_ns: int = T_REF_NS,
) -> ChunkRequestIdentity:
    return ChunkRequestIdentity(
        request_id=request_id,
        chunk_id=chunk_id,
        proposal_id=proposal_id,
        policy_revision=revision,
        policy_epoch=epoch,
        takeover_generation=takeover_generation,
        reset_generation=reset_generation,
        request_clock_domain_id=request_clock_domain_id,
        t_ref_clock_domain_id=t_ref_clock_domain_id,
        t_ref_ns=t_ref_ns,
    )


def result(
    identity: ChunkRequestIdentity,
    result_id: str = "result-0",
    *,
    result_clock_domain_id: str = CLOCK_DOMAIN_ID,
) -> ChunkResultIdentity:
    return ChunkResultIdentity(
        **identity.__dict__,
        result_id=result_id,
        result_clock_domain_id=result_clock_domain_id,
    )


def ledger(**limit_overrides: int) -> RationalH50SelectionLedger:
    return RationalH50SelectionLedger(
        limits(**limit_overrides),
        policy_revision="revision-0",
        clock_domain_id=CLOCK_DOMAIN_ID,
    )


def adopt(
    value: RationalH50SelectionLedger, identity: ChunkRequestIdentity | None = None
) -> ChunkResultIdentity:
    pinned = request() if identity is None else identity
    adopted = result(pinned, f"result-{pinned.request_id}")
    value.pin_request(pinned)
    value.adopt_result(adopted, actions_cached=H50_ACTIONS_CACHED)
    return adopted


def selection_ns(index: int) -> int:
    return T_REF_NS + index * 1_000_000_000 // H50_MODEL_TIMEBASE_HZ


def begin_dispatch(
    value: RationalH50SelectionLedger,
    *,
    dispatch_sequence: int,
    selected_at_ns: int,
    selected_post_adapter_absolute7: tuple[float, ...] = POST_ADAPTER_ABSOLUTE7,
    pose_command_id: str,
    gripper_command_id: str,
    selection_clock_domain_id: str = CLOCK_DOMAIN_ID,
    dispatch_ns: int | None = None,
    dispatch_clock_domain_id: str = CLOCK_DOMAIN_ID,
):
    return value.begin_dispatch(
        dispatch_sequence=dispatch_sequence,
        selection_ns=selected_at_ns,
        selection_clock_domain_id=selection_clock_domain_id,
        dispatch_ns=selected_at_ns if dispatch_ns is None else dispatch_ns,
        dispatch_clock_domain_id=dispatch_clock_domain_id,
        selected_post_adapter_absolute7=selected_post_adapter_absolute7,
        pose_command_id=pose_command_id,
        gripper_command_id=gripper_command_id,
    )


def accepted_dispatch(
    value: RationalH50SelectionLedger, *, index: int, sequence: int
):
    selected_at = selection_ns(index)
    entry = begin_dispatch(
        value,
        dispatch_sequence=sequence,
        selected_at_ns=selected_at,
        pose_command_id=f"pose-{sequence}",
        gripper_command_id=f"gripper-{sequence}",
    )
    value.record_pose_ack(
        dispatch_sequence=sequence,
        ack_id=f"pose-{sequence}",
        accepted=True,
        ack_ns=selected_at + 1_000_000,
        ack_clock_domain_id=CLOCK_DOMAIN_ID,
    )
    value.record_gripper_ack(
        dispatch_sequence=sequence,
        ack_id=f"gripper-{sequence}",
        accepted=True,
        ack_ns=selected_at + 2_000_000,
        ack_clock_domain_id=CLOCK_DOMAIN_ID,
    )
    return entry, value.commit_dispatch(sequence)


def test_single_owner_event_loop_rejects_cross_thread_call_and_latches_stop() -> None:
    value = ledger()
    adopt(value)
    failures: list[RuntimeSafetyViolation] = []

    def cross_thread_call() -> None:
        try:
            value.refresh_assessment(
                selection_ns(3), selection_clock_domain_id=CLOCK_DOMAIN_ID
            )
        except RuntimeSafetyViolation as error:
            failures.append(error)

    thread = threading.Thread(target=cross_thread_call)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(failures) == 1
    assert failures[0].reason == "ONLINE_REPLAY_RUNTIME_LEDGER_CROSS_THREAD_CALL"
    assert failures[0].directive is SafetyDirective.STOP
    assert value.fail_closed_directive is SafetyDirective.STOP
    with pytest.raises(RuntimeSafetyViolation, match="STOP_LATCHED"):
        value.pin_request(
            request(
                request_id="request-after-cross-thread",
                chunk_id="chunk-after-cross-thread",
                proposal_id="proposal-after-cross-thread",
            )
        )


def test_runtime_clock_domain_is_required_for_every_comparable_timestamp() -> None:
    mismatched_request = ledger()
    with pytest.raises(RuntimeSafetyViolation, match="REQUEST_T_REF_CLOCK_MISMATCH"):
        mismatched_request.pin_request(
            request(t_ref_clock_domain_id="different-monotonic-clock")
        )

    cross_clock_request = ledger()
    with pytest.raises(RuntimeSafetyViolation, match="CROSS_CLOCK_REQUEST"):
        cross_clock_request.pin_request(
            request(
                request_clock_domain_id="different-monotonic-clock",
                t_ref_clock_domain_id="different-monotonic-clock",
            )
        )

    cross_clock_result = ledger()
    pinned = request()
    cross_clock_result.pin_request(pinned)
    with pytest.raises(RuntimeSafetyViolation, match="RESULT_CLOCK_MISMATCH"):
        cross_clock_result.adopt_result(
            result(pinned, result_clock_domain_id="different-monotonic-clock"),
            actions_cached=H50_ACTIONS_CACHED,
        )

    cross_clock_selection = ledger()
    adopt(cross_clock_selection)
    with pytest.raises(RuntimeSafetyViolation, match="CROSS_CLOCK_DISPATCH"):
        begin_dispatch(
            cross_clock_selection,
            dispatch_sequence=0,
            selected_at_ns=selection_ns(3),
            selection_clock_domain_id="different-monotonic-clock",
            pose_command_id="pose-clock",
            gripper_command_id="gripper-clock",
        )

    cross_clock_pose_ack = ledger()
    adopt(cross_clock_pose_ack)
    begin_dispatch(
        cross_clock_pose_ack,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-clock",
        gripper_command_id="gripper-clock",
    )
    with pytest.raises(RuntimeSafetyViolation, match="CROSS_CLOCK_POSE_ACK"):
        cross_clock_pose_ack.record_pose_ack(
            dispatch_sequence=0,
            ack_id="pose-clock",
            accepted=True,
            ack_ns=selection_ns(3) + 1,
            ack_clock_domain_id="different-monotonic-clock",
        )

    cross_clock_gripper_ack = ledger()
    adopt(cross_clock_gripper_ack)
    begin_dispatch(
        cross_clock_gripper_ack,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-clock",
        gripper_command_id="gripper-clock",
    )
    with pytest.raises(RuntimeSafetyViolation, match="CROSS_CLOCK_GRIPPER_ACK"):
        cross_clock_gripper_ack.record_gripper_ack(
            dispatch_sequence=0,
            ack_id="gripper-clock",
            accepted=True,
            ack_ns=selection_ns(3) + 1,
            ack_clock_domain_id="different-monotonic-clock",
        )

    cross_clock_macro = ledger()
    adopt(cross_clock_macro)
    _, committed = accepted_dispatch(cross_clock_macro, index=3, sequence=0)
    with pytest.raises(RuntimeSafetyViolation, match="NOT_ACK_AUTHORITATIVE"):
        project_acknowledged_runtime_macro(
            [committed],
            (1_133_333_333, 1_166_666_667, 1_200_000_000),
            grid_clock_domain_id="different-monotonic-clock",
            max_ack_age_ms=100.0,
        )


def test_rational_selection_fault_injection_accepts_index_steps_2_3_4() -> None:
    value = ledger()
    adopt(value)
    committed = []
    for sequence, index in enumerate((3, 5, 8, 12)):
        _, entry = accepted_dispatch(value, index=index, sequence=sequence)
        committed.append(entry)
    assert [entry.selected_index for entry in committed] == [3, 5, 8, 12]
    assert [
        right.selected_index - left.selected_index
        for left, right in zip(committed, committed[1:])
    ] == [2, 3, 4]
    assert all(
        entry.selected_post_adapter_absolute7 == POST_ADAPTER_ABSOLUTE7
        for entry in committed
    )
    macro = project_acknowledged_runtime_macro(
        committed,
        (1_433_333_333, 1_466_666_667, 1_500_000_000),
        grid_clock_domain_id=CLOCK_DOMAIN_ID,
        max_ack_age_ms=100.0,
    )
    assert macro.ack_ids == ("pose-3", "pose-3", "pose-3")


def test_fixed_anchor_rejects_future_phase_and_repeated_index() -> None:
    assert rational_h50_index(T_REF_NS, selection_ns(3)) == 3
    with pytest.raises(RuntimeSafetyViolation, match="SELECTION_TIME_INVALID"):
        rational_h50_index(T_REF_NS, T_REF_NS - 1)

    value = ledger()
    adopt(value)
    first_selection_ns = selection_ns(3) - 3
    begin_dispatch(
        value,
        dispatch_sequence=0,
        selected_at_ns=first_selection_ns,
        pose_command_id="pose-0",
        gripper_command_id="gripper-0",
    )
    value.record_pose_ack(
        dispatch_sequence=0,
        ack_id="pose-0",
        accepted=True,
        ack_ns=first_selection_ns + 1,
        ack_clock_domain_id=CLOCK_DOMAIN_ID,
    )
    value.record_gripper_ack(
        dispatch_sequence=0,
        ack_id="gripper-0",
        accepted=True,
        ack_ns=first_selection_ns + 2,
        ack_clock_domain_id=CLOCK_DOMAIN_ID,
    )
    value.commit_dispatch(0)
    with pytest.raises(RuntimeSafetyViolation, match="NOT_STRICTLY_INCREASING") as error:
        begin_dispatch(
            value,
            dispatch_sequence=1,
            selected_at_ns=selection_ns(3),
            pose_command_id="pose-1",
            gripper_command_id="gripper-1",
        )
    assert error.value.directive is SafetyDirective.HOLD
    assert value.active_chunk is None


def test_chunk_age_selected_index_and_dispatch_count_fail_closed() -> None:
    too_old = ledger(max_chunk_age_ns=100_000_000)
    adopt(too_old)
    with pytest.raises(RuntimeSafetyViolation, match="MAX_CHUNK_AGE"):
        begin_dispatch(
            too_old,
            dispatch_sequence=0,
            selected_at_ns=selection_ns(4),
            pose_command_id="pose-age",
            gripper_command_id="gripper-age",
        )

    too_far = ledger(max_selected_index=5)
    adopt(too_far)
    with pytest.raises(RuntimeSafetyViolation, match="MAX_SELECTED_INDEX"):
        begin_dispatch(
            too_far,
            dispatch_sequence=0,
            selected_at_ns=selection_ns(6),
            pose_command_id="pose-index",
            gripper_command_id="gripper-index",
        )

    too_many = ledger(max_dispatch_count=2)
    adopt(too_many)
    accepted_dispatch(too_many, index=1, sequence=0)
    accepted_dispatch(too_many, index=2, sequence=1)
    with pytest.raises(RuntimeSafetyViolation, match="MAX_DISPATCH_COUNT"):
        begin_dispatch(
            too_many,
            dispatch_sequence=2,
            selected_at_ns=selection_ns(3),
            pose_command_id="pose-count",
            gripper_command_id="gripper-count",
        )


def test_low_watermark_is_time_based_and_current_400ms_is_not_approved() -> None:
    assert CURRENT_LOW_WATERMARK_COVERAGE_NS == 400_000_000
    assert CONCURRENT_MAX_SERVICE_LATENCY_NS == 443_161_677
    assert CURRENT_LOW_WATERMARK_COVERAGE_NS < CONCURRENT_MAX_SERVICE_LATENCY_NS
    assert CURRENT_LOW_WATERMARK_APPROVED is False

    value = ledger()
    adopt(value)
    assessment = value.refresh_assessment(
        selection_ns(37), selection_clock_domain_id=CLOCK_DOMAIN_ID
    )
    assert assessment.remaining_time_ns == 400_000_000
    assert assessment.refresh_due
    assert assessment.service_headroom_exhausted
    with pytest.raises(RuntimeSafetyViolation, match="HEADROOM_EXHAUSTED") as error:
        begin_dispatch(
            value,
            dispatch_sequence=0,
            selected_at_ns=selection_ns(37),
            pose_command_id="pose-headroom",
            gripper_command_id="gripper-headroom",
        )
    assert error.value.directive is SafetyDirective.HOLD


def test_refresh_must_be_pinned_before_entering_additional_headroom() -> None:
    value = ledger()
    adopt(value)
    assessment = value.refresh_assessment(
        selection_ns(35), selection_clock_domain_id=CLOCK_DOMAIN_ID
    )
    assert assessment.refresh_due
    assert not assessment.service_headroom_exhausted
    with pytest.raises(RuntimeSafetyViolation, match="REFRESH_REQUIRED"):
        begin_dispatch(
            value,
            dispatch_sequence=0,
            selected_at_ns=selection_ns(35),
            pose_command_id="pose-refresh",
            gripper_command_id="gripper-refresh",
        )
    value.pin_request(
        request(
            request_id="request-1",
            chunk_id="chunk-1",
            proposal_id="proposal-1",
            t_ref_ns=selection_ns(35),
        )
    )
    accepted_dispatch(value, index=35, sequence=0)


def test_stale_result_and_takeover_reset_revision_generations_are_rejected() -> None:
    value = ledger()
    old_request = request()
    adopt(value, old_request)
    assert value.human_takeover_flush(takeover_generation=1, policy_epoch=1) is SafetyDirective.HOLD
    with pytest.raises(RuntimeSafetyViolation, match="STALE_RESULT_GENERATION"):
        value.adopt_result(
            result(old_request, "late-after-takeover"),
            actions_cached=H50_ACTIONS_CACHED,
        )
    with pytest.raises(RuntimeSafetyViolation, match="STALE_REQUEST_GENERATION"):
        value.pin_request(old_request)

    takeover_request = request(
        request_id="request-takeover",
        chunk_id="chunk-takeover",
        proposal_id="proposal-takeover",
        epoch=1,
        takeover_generation=1,
    )
    adopt(value, takeover_request)
    assert value.reset_home_flush(reset_generation=1) is SafetyDirective.HOLD
    with pytest.raises(RuntimeSafetyViolation, match="STALE_RESULT_GENERATION"):
        value.adopt_result(
            result(takeover_request, "late-after-reset"),
            actions_cached=H50_ACTIONS_CACHED,
        )

    reset_request = request(
        request_id="request-reset",
        chunk_id="chunk-reset",
        proposal_id="proposal-reset",
        epoch=1,
        takeover_generation=1,
        reset_generation=1,
    )
    adopt(value, reset_request)
    assert value.policy_revision_flush(
        policy_revision="revision-1", policy_epoch=2
    ) is SafetyDirective.HOLD
    with pytest.raises(RuntimeSafetyViolation, match="STALE_REQUEST_GENERATION"):
        value.pin_request(reset_request)


def test_pose_and_gripper_ack_fail_closed_on_missing_rejected_or_mismatch() -> None:
    missing_pose = ledger()
    adopt(missing_pose)
    begin_dispatch(
        missing_pose,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-0",
        gripper_command_id="gripper-0",
    )
    with pytest.raises(RuntimeSafetyViolation, match="POSE_ACK_MISSING") as error:
        missing_pose.commit_dispatch(0)
    assert error.value.directive is SafetyDirective.STOP
    assert missing_pose.fail_closed_directive is SafetyDirective.STOP

    rejected_pose = ledger()
    adopt(rejected_pose)
    begin_dispatch(
        rejected_pose,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-0",
        gripper_command_id="gripper-0",
    )
    with pytest.raises(RuntimeSafetyViolation, match="POSE_ACK_REJECTED"):
        rejected_pose.record_pose_ack(
            dispatch_sequence=0,
            ack_id="pose-0",
            accepted=False,
            ack_ns=selection_ns(3) + 1,
            ack_clock_domain_id=CLOCK_DOMAIN_ID,
        )

    missing_gripper = ledger()
    adopt(missing_gripper)
    begin_dispatch(
        missing_gripper,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-0",
        gripper_command_id="gripper-0",
    )
    missing_gripper.record_pose_ack(
        dispatch_sequence=0,
        ack_id="pose-0",
        accepted=True,
        ack_ns=selection_ns(3) + 1,
        ack_clock_domain_id=CLOCK_DOMAIN_ID,
    )
    with pytest.raises(RuntimeSafetyViolation, match="GRIPPER_ACK_MISSING"):
        missing_gripper.commit_dispatch(0)

    mismatched_gripper = ledger()
    adopt(mismatched_gripper)
    begin_dispatch(
        mismatched_gripper,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-0",
        gripper_command_id="gripper-0",
    )
    mismatched_gripper.record_pose_ack(
        dispatch_sequence=0,
        ack_id="pose-0",
        accepted=True,
        ack_ns=selection_ns(3) + 1,
        ack_clock_domain_id=CLOCK_DOMAIN_ID,
    )
    with pytest.raises(RuntimeSafetyViolation, match="GRIPPER_ACK_ID_MISMATCH"):
        mismatched_gripper.record_gripper_ack(
            dispatch_sequence=0,
            ack_id="wrong-gripper",
            accepted=True,
            ack_ns=selection_ns(3) + 2,
            ack_clock_domain_id=CLOCK_DOMAIN_ID,
        )

    timed_out = ledger()
    adopt(timed_out)
    begin_dispatch(
        timed_out,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-0",
        gripper_command_id="gripper-0",
    )
    with pytest.raises(RuntimeSafetyViolation, match="POSE_ACK_MISSING"):
        timed_out.expire_missing_acks(
            selection_ns(3) + 20_000_001,
            now_clock_domain_id=CLOCK_DOMAIN_ID,
        )


def test_duplicate_ack_is_idempotent_only_for_the_exact_same_payload() -> None:
    value = ledger()
    adopt(value)
    begin_dispatch(
        value,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-0",
        gripper_command_id="gripper-0",
    )
    pose_payload = {
        "dispatch_sequence": 0,
        "ack_id": "pose-0",
        "accepted": True,
        "ack_ns": selection_ns(3) + 1,
        "ack_clock_domain_id": CLOCK_DOMAIN_ID,
    }
    gripper_payload = {
        "dispatch_sequence": 0,
        "ack_id": "gripper-0",
        "accepted": True,
        "ack_ns": selection_ns(3) + 2,
        "ack_clock_domain_id": CLOCK_DOMAIN_ID,
    }
    value.record_pose_ack(**pose_payload)
    value.record_pose_ack(**pose_payload)
    value.record_gripper_ack(**gripper_payload)
    value.record_gripper_ack(**gripper_payload)
    value.commit_dispatch(0)
    value.record_pose_ack(**pose_payload)
    value.record_gripper_ack(**gripper_payload)
    assert value.entries[-1].status == "accepted"

    conflict = ledger()
    adopt(conflict)
    begin_dispatch(
        conflict,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-0",
        gripper_command_id="gripper-0",
    )
    conflict.record_pose_ack(**pose_payload)
    with pytest.raises(RuntimeSafetyViolation, match="DUPLICATE_POSE_ACK_CONFLICT"):
        conflict.record_pose_ack(**{**pose_payload, "ack_ns": selection_ns(3) + 2})
    assert conflict.fail_closed_directive is SafetyDirective.STOP


def test_ack_before_command_after_deadline_and_after_stop_latch_is_rejected() -> None:
    before_command = ledger()
    adopt(before_command)
    with pytest.raises(RuntimeSafetyViolation, match="POSE_ACK_BEFORE_COMMAND"):
        before_command.record_pose_ack(
            dispatch_sequence=0,
            ack_id="pose-0",
            accepted=True,
            ack_ns=selection_ns(3),
            ack_clock_domain_id=CLOCK_DOMAIN_ID,
        )
    assert before_command.fail_closed_directive is SafetyDirective.STOP
    with pytest.raises(RuntimeSafetyViolation, match="STOP_LATCHED"):
        before_command.commit_dispatch(0)
    with pytest.raises(RuntimeSafetyViolation, match="STOP_LATCHED"):
        before_command.record_gripper_ack(
            dispatch_sequence=0,
            ack_id="gripper-0",
            accepted=True,
            ack_ns=selection_ns(3),
            ack_clock_domain_id=CLOCK_DOMAIN_ID,
        )

    after_deadline = ledger()
    adopt(after_deadline)
    begin_dispatch(
        after_deadline,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-0",
        gripper_command_id="gripper-0",
    )
    with pytest.raises(RuntimeSafetyViolation, match="POSE_ACK_STALE_OR_NONCAUSAL"):
        after_deadline.record_pose_ack(
            dispatch_sequence=0,
            ack_id="pose-0",
            accepted=True,
            ack_ns=selection_ns(3) + 20_000_001,
            ack_clock_domain_id=CLOCK_DOMAIN_ID,
        )


@pytest.mark.parametrize("flush_kind", ("takeover", "reset", "revision"))
def test_ack_after_generation_flush_is_rejected(flush_kind: str) -> None:
    value = ledger()
    adopt(value)
    accepted_dispatch(value, index=3, sequence=0)
    if flush_kind == "takeover":
        value.human_takeover_flush(takeover_generation=1, policy_epoch=1)
    elif flush_kind == "reset":
        value.reset_home_flush(reset_generation=1)
    else:
        value.policy_revision_flush(policy_revision="revision-1", policy_epoch=1)
    with pytest.raises(RuntimeSafetyViolation, match="POSE_ACK_AFTER_FLUSH"):
        value.record_pose_ack(
            dispatch_sequence=0,
            ack_id="pose-0",
            accepted=True,
            ack_ns=selection_ns(3) + 1_000_000,
            ack_clock_domain_id=CLOCK_DOMAIN_ID,
        )
    assert value.fail_closed_directive is SafetyDirective.STOP


def test_partial_ack_cannot_create_accepted_action_or_transition() -> None:
    value = ledger()
    adopt(value)
    begin_dispatch(
        value,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-0",
        gripper_command_id="gripper-0",
    )
    value.record_pose_ack(
        dispatch_sequence=0,
        ack_id="pose-0",
        accepted=True,
        ack_ns=selection_ns(3) + 1,
        ack_clock_domain_id=CLOCK_DOMAIN_ID,
    )
    partial = value.entries[-1]
    with pytest.raises(RuntimeSafetyViolation, match="NOT_ACK_AUTHORITATIVE"):
        partial.to_accepted_ack(clock_domain_id=CLOCK_DOMAIN_ID)
    with pytest.raises(RuntimeSafetyViolation, match="NOT_ACK_AUTHORITATIVE"):
        project_acknowledged_runtime_macro(
            [partial],
            (1_133_333_333, 1_166_666_667, 1_200_000_000),
            grid_clock_domain_id=CLOCK_DOMAIN_ID,
            max_ack_age_ms=100.0,
        )


def test_only_dual_ack_post_adapter_absolute7_becomes_transition_authority() -> None:
    unacked_ledger = ledger()
    adopt(unacked_ledger)
    unacked = begin_dispatch(
        unacked_ledger,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-unacked",
        gripper_command_id="gripper-unacked",
    )
    with pytest.raises(RuntimeSafetyViolation, match="NOT_ACK_AUTHORITATIVE"):
        project_acknowledged_runtime_macro(
            [unacked],
            (1_133_333_333, 1_166_666_667, 1_200_000_000),
            grid_clock_domain_id=CLOCK_DOMAIN_ID,
            max_ack_age_ms=100.0,
        )

    value = ledger()
    adopt(value)
    _, committed = accepted_dispatch(value, index=3, sequence=0)
    assert (
        committed.request_id,
        committed.result_id,
        committed.chunk_id,
        committed.proposal_id,
        committed.policy_revision,
        committed.policy_epoch,
        committed.takeover_generation,
        committed.reset_generation,
        committed.t_ref_ns,
        committed.dispatch_sequence,
        committed.selected_index,
        committed.pose_command_id,
        committed.pose_ack_id,
        committed.gripper_command_id,
        committed.gripper_ack_id,
    ) == (
        "request-0",
        "result-request-0",
        "chunk-0",
        "proposal-0",
        "revision-0",
        0,
        0,
        0,
        T_REF_NS,
        0,
        3,
        "pose-0",
        "pose-0",
        "gripper-0",
        "gripper-0",
    )
    accepted_ack = committed.to_accepted_ack(clock_domain_id=CLOCK_DOMAIN_ID)
    assert accepted_ack.accepted_absolute_action7 == POST_ADAPTER_ABSOLUTE7
    macro = project_acknowledged_runtime_macro(
        [committed],
        (1_133_333_333, 1_166_666_667, 1_200_000_000),
        grid_clock_domain_id=CLOCK_DOMAIN_ID,
        max_ack_age_ms=100.0,
    )
    np.testing.assert_array_equal(
        macro.accepted_absolute_action_k7,
        np.asarray([POST_ADAPTER_ABSOLUTE7] * 3),
    )


def test_no_safe_action_is_explicit_hold_and_unknown_command_outcome_is_stop() -> None:
    value = ledger()
    assert value.fail_closed_directive is SafetyDirective.HOLD
    with pytest.raises(RuntimeSafetyViolation, match="NO_SAFE_ACTION_HOLD") as error:
        begin_dispatch(
            value,
            dispatch_sequence=0,
            selected_at_ns=selection_ns(1),
            pose_command_id="pose-none",
            gripper_command_id="gripper-none",
        )
    assert error.value.directive is SafetyDirective.HOLD

    adopt(value)
    begin_dispatch(
        value,
        dispatch_sequence=0,
        selected_at_ns=selection_ns(3),
        pose_command_id="pose-0",
        gripper_command_id="gripper-0",
    )
    assert value.human_takeover_flush(
        takeover_generation=1, policy_epoch=1
    ) is SafetyDirective.STOP
    assert value.fail_closed_directive is SafetyDirective.STOP
    assert value.reset_home_flush(reset_generation=1) is SafetyDirective.HOLD
    assert value.fail_closed_directive is SafetyDirective.HOLD


def test_online_import_has_no_ros_network_robot_or_cuda_side_effects() -> None:
    script = r'''
import importlib.abc
import json
import sys
import threading

blocked_roots = {
    "action_msgs", "control_msgs", "franka", "franka_msgs", "geometry_msgs",
    "rclpy", "rospy", "sensor_msgs",
}

class BlockRobotImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname.split(".", 1)[0] in blocked_roots:
            raise RuntimeError(f"FORBIDDEN_ROBOT_IMPORT:{fullname}")
        return None

def audit(event, args):
    del args
    if event.startswith("socket."):
        raise RuntimeError(f"FORBIDDEN_NETWORK_SIDE_EFFECT:{event}")
    if event in {
        "os.exec", "os.posix_spawn", "os.system", "pty.spawn", "subprocess.Popen"
    }:
        raise RuntimeError(f"FORBIDDEN_PROCESS_SIDE_EFFECT:{event}")

sys.meta_path.insert(0, BlockRobotImports())
sys.addaudithook(audit)
threads_before = tuple(thread.ident for thread in threading.enumerate())
import forcesmolvla.rft.online  # noqa: F401
import torch
threads_after = tuple(thread.ident for thread in threading.enumerate())
assert threads_after == threads_before
assert not torch.cuda.is_initialized()
print(json.dumps({
    "cuda_initialized": torch.cuda.is_initialized(),
    "network_side_effect": False,
    "robot_import": False,
    "thread_side_effect": False,
}, sort_keys=True))
'''
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "src:vendor/lerobot/src",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "cuda_initialized": False,
        "network_side_effect": False,
        "robot_import": False,
        "thread_side_effect": False,
    }
