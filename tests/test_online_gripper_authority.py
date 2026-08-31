from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from forcesmolvla.rft.online.gripper_authority import (
    FULL_ACTION7_ACK_CLOSURE_PRODUCTION,
    GRIPPER_FEEDBACK_FRESHNESS_BOUND,
    GRIPPER_NOOP_ACK_POLICY,
    GRIPPER_TERMINAL_SEAL_REQUIRED,
    INVALID_TERMINAL_OUTCOMES,
    PRODUCTION_INTEGRATION_BLOCKED_ON_GRIPPER_ACK,
    VALID_TERMINAL_OUTCOMES,
    GripperAuthorityKind,
    GripperFeedback,
    GripperGeneration,
    GripperLifecycle,
    GripperProvenanceError,
    GripperProvenanceLedger,
    close_full_action7_authority,
    pose_authority_from_accepted_reference,
)
from forcesmolvla.rft.online.action_runtime import (
    ChunkRequestIdentity,
    ChunkResultIdentity,
    RationalH50SelectionLedger,
    RuntimeSafetyLimits,
    RuntimeSafetyViolation,
)


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs/online_replay_gripper_authority.v1.development.json"
CLOCK = "upper_host_monotonic"


def _generation(**changes) -> GripperGeneration:
    values = {
        "episode_id": "episode-A",
        "reset_generation": 0,
        "takeover_generation": 0,
        "policy_revision": "revision-A",
        "policy_epoch": 3,
    }
    values.update(changes)
    return GripperGeneration(**values)


def _ledger(generation: GripperGeneration | None = None) -> GripperProvenanceLedger:
    return GripperProvenanceLedger(
        generation=generation or _generation(),
        clock_domain_id=CLOCK,
        max_feedback_age_ns=100_000_000,
    )


def _accept(
    ledger: GripperProvenanceLedger,
    *,
    sequence: int = 1,
    state: str = "CLOSED",
    width: float = 0.0,
    started_ns: int = 1_000_000_000,
    accepted_ns: int = 1_010_000_000,
    goal_id: str = "real-ros-goal-1",
    generation: GripperGeneration | None = None,
) -> None:
    generation = generation or ledger.generation
    ledger.begin_command(
        local_goal_sequence=sequence,
        requested_state=state,
        requested_width_m=width,
        started_monotonic_ns=started_ns,
        generation=generation,
        clock_domain_id=CLOCK,
    )
    ledger.accept_command(
        local_goal_sequence=sequence,
        action_goal_id=goal_id,
        accepted_monotonic_ns=accepted_ns,
        generation=generation,
        clock_domain_id=CLOCK,
    )


def _feedback(
    ledger: GripperProvenanceLedger,
    *,
    timestamp_ns: int = 1_020_000_000,
    width: float = 0.047,
    state: str = "INTERMEDIATE",
) -> GripperFeedback:
    return GripperFeedback(
        measured_width_m=width,
        measured_state=state,
        feedback_monotonic_ns=timestamp_ns,
        clock_domain_id=CLOCK,
        generation=ledger.generation,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_contract_binds_current_production_sources_and_fields():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["authority_kinds"] == [
        "NEW_COMMAND",
        "HELD_FROM_ACCEPTED_COMMAND",
    ]
    assert config["lifecycle"] == [state.value for state in GripperLifecycle]
    assert set(config["terminal_contract"]["valid_outcomes_from_current_recorder_quality_gate"]) == VALID_TERMINAL_OUTCOMES
    assert set(config["terminal_contract"]["invalid_outcomes"]) == INVALID_TERMINAL_OUTCOMES
    assert config["clock_contract"]["max_feedback_age_ms"] == 100.0
    assert config["held_authority_contract"]["new_command_or_ack_identity_synthesized"] is False

    source_audit = config["current_production_source_audit"]
    for name, binding in source_audit.items():
        path = Path(name)
        assert path.is_file()
        assert _sha256(path) == binding["sha256"]

    spacemouse = Path(
        "/home/rlc123/fr3_client_ws/scripts/record_franka_spacemouse_publisher.py"
    ).read_text(encoding="utf-8")
    for field in (
        "local_goal_sequence",
        "action_goal_id",
        "requested_state",
        "started_monotonic_ns",
        "accepted_monotonic_ns",
    ):
        assert field in spacemouse
    assert "result.reached_goal" in spacemouse
    assert "result.stalled" in spacemouse

    forcevla = Path(
        "/home/rlc123/fr3_client_ws/scripts/record_franka_forcevla.py"
    ).read_text(encoding="utf-8")
    assert 'if outcome in {"reached", "stalled"}' in forcevla
    assert "_gripper_goal_integrity_reason" in forcevla
    assert "_gripper_state_integrity_reason" in forcevla

    hilserl = Path(
        "/home/rlc123/fr3_client_ws/scripts/record_franka_hilserl_impedance.py"
    ).read_text(encoding="utf-8")
    assert "forcevla._gripper_goal_integrity_reason" in hilserl
    assert "forcevla._gripper_state_integrity_reason" in hilserl

    deploy = Path(
        "/home/rlc123/fr3_client_ws/scripts/deploy_forcesmolvla.py"
    ).read_text(encoding="utf-8")
    loop = deploy[deploy.index("def run_async_policy_loop("):deploy.index("def wait_observation(")]
    assert loop.index("validate_exact_controller_ack(") < loop.index("controller._send_gripper_goal(")
    assert "desired_close != controller.gripper_closed" in loop
    assert "not controller.gripper_goal_active" in loop
    assert "_gripper_result" not in loop

    binding = config["current_record_field_binding"]
    assert "token" in binding["gripper_target.jsonl"]
    assert "episode_id" in binding["absent_from_gripper_event_streams"]
    assert "reset_generation" in binding["absent_from_gripper_event_streams"]
    assert "takeover_generation" in binding["absent_from_gripper_event_streams"]
    assert "policy_revision" in binding["absent_from_gripper_event_streams"]


def test_new_command_authority_binds_real_goal_acceptance_without_fake_ack():
    ledger = _ledger()
    assert ledger.lifecycle is GripperLifecycle.UNBOUND
    ledger.begin_command(
        local_goal_sequence=7,
        requested_state="CLOSED",
        requested_width_m=0.0,
        started_monotonic_ns=1_000_000_000,
        generation=ledger.generation,
        clock_domain_id=CLOCK,
    )
    assert ledger.lifecycle is GripperLifecycle.COMMAND_PENDING
    ledger.accept_command(
        local_goal_sequence=7,
        action_goal_id="actual-action-goal-uuid",
        accepted_monotonic_ns=1_010_000_000,
        generation=ledger.generation,
        clock_domain_id=CLOCK,
    )
    assert ledger.lifecycle is GripperLifecycle.ACCEPTED_ACTIVE
    evidence = ledger.new_command_authority(
        transition_id="transition-new", authority_monotonic_ns=1_011_000_000
    )
    assert evidence.authority_kind is GripperAuthorityKind.NEW_COMMAND
    assert evidence.origin_local_goal_sequence == 7
    assert evidence.origin_action_goal_id == "actual-action-goal-uuid"
    assert evidence.origin_accepted_monotonic_ns == 1_010_000_000
    assert evidence.feedback_monotonic_ns is None
    assert not hasattr(evidence, "gripper_ack_id")
    assert ledger.eligible_for_replay("transition-new") is False


def test_held_authority_reuses_origin_and_requires_fresh_feedback():
    ledger = _ledger()
    _accept(ledger)
    feedback = _feedback(ledger)
    evidence = ledger.held_authority(
        transition_id="transition-held",
        requested_state="CLOSED",
        requested_width_m=0.0,
        authority_monotonic_ns=1_025_000_000,
        feedback=feedback,
    )
    assert evidence.authority_kind is GripperAuthorityKind.HELD_FROM_ACCEPTED_COMMAND
    assert evidence.origin_local_goal_sequence == 1
    assert evidence.origin_action_goal_id == "real-ros-goal-1"
    assert evidence.feedback_width_m == 0.047
    assert evidence.feedback_state == "INTERMEDIATE"
    assert evidence.feedback_age_ns == 5_000_000
    assert not hasattr(evidence, "new_command_id")
    assert not hasattr(evidence, "gripper_ack_id")

    with pytest.raises(GripperProvenanceError, match="STALE"):
        ledger.held_authority(
            transition_id="transition-stale",
            requested_state="CLOSED",
            requested_width_m=0.0,
            authority_monotonic_ns=1_120_000_001,
            feedback=feedback,
        )
    with pytest.raises(GripperProvenanceError, match="TARGET_CONFLICT"):
        ledger.held_authority(
            transition_id="transition-conflict",
            requested_state="OPEN",
            requested_width_m=0.085,
            authority_monotonic_ns=1_025_000_000,
            feedback=feedback,
        )


def test_held_authority_rejects_cross_clock_generation_future_and_pending():
    ledger = _ledger()
    _accept(ledger)
    with pytest.raises(GripperProvenanceError, match="CLOCK_DOMAIN"):
        ledger.held_authority(
            transition_id="cross-clock",
            requested_state="CLOSED",
            requested_width_m=0.0,
            authority_monotonic_ns=1_025_000_000,
            feedback=replace(_feedback(ledger), clock_domain_id="ros-time"),
        )
    with pytest.raises(GripperProvenanceError, match="GENERATION_STALE"):
        ledger.held_authority(
            transition_id="cross-generation",
            requested_state="CLOSED",
            requested_width_m=0.0,
            authority_monotonic_ns=1_025_000_000,
            feedback=replace(
                _feedback(ledger), generation=replace(ledger.generation, policy_epoch=4)
            ),
        )
    with pytest.raises(GripperProvenanceError, match="STALE"):
        ledger.held_authority(
            transition_id="future-feedback",
            requested_state="CLOSED",
            requested_width_m=0.0,
            authority_monotonic_ns=1_019_999_999,
            feedback=_feedback(ledger),
        )

    ledger.begin_command(
        local_goal_sequence=2,
        requested_state="CLOSED",
        requested_width_m=0.0,
        started_monotonic_ns=1_030_000_000,
        generation=ledger.generation,
        clock_domain_id=CLOCK,
    )
    with pytest.raises(GripperProvenanceError, match="LEASE_UNAVAILABLE"):
        ledger.held_authority(
            transition_id="pending-command",
            requested_state="CLOSED",
            requested_width_m=0.0,
            authority_monotonic_ns=1_031_000_000,
            feedback=replace(_feedback(ledger), feedback_monotonic_ns=1_031_000_000),
        )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("reached", GripperLifecycle.TERMINAL_REACHED),
        ("stalled", GripperLifecycle.TERMINAL_STALLED),
    ],
)
def test_current_recorder_terminal_quality_gate_allows_reached_and_stalled(
    outcome, expected
):
    ledger = _ledger()
    _accept(ledger)
    ledger.new_command_authority(
        transition_id=f"transition-{outcome}", authority_monotonic_ns=1_020_000_000
    )
    with pytest.raises(GripperProvenanceError, match="PAIRING_INCOMPLETE"):
        ledger.seal_episode()
    ledger.record_terminal(
        local_goal_sequence=1,
        action_goal_id="real-ros-goal-1",
        outcome=outcome,
        finished_monotonic_ns=1_100_000_000,
        generation=ledger.generation,
        clock_domain_id=CLOCK,
    )
    assert ledger.lifecycle is expected
    sealed = ledger.seal_episode()
    assert sealed[0].terminal_outcome == outcome
    assert sealed[0].terminal_sealed is True
    assert ledger.eligible_for_replay(f"transition-{outcome}") is True


@pytest.mark.parametrize("outcome", sorted(INVALID_TERMINAL_OUTCOMES))
def test_failed_terminal_invalidates_and_quarantines_all_dependencies(outcome):
    ledger = _ledger()
    _accept(ledger)
    ledger.new_command_authority(
        transition_id="dependent-new", authority_monotonic_ns=1_020_000_000
    )
    ledger.held_authority(
        transition_id="dependent-held",
        requested_state="CLOSED",
        requested_width_m=0.0,
        authority_monotonic_ns=1_025_000_000,
        feedback=_feedback(ledger),
    )
    with pytest.raises(GripperProvenanceError, match="TERMINAL"):
        ledger.record_terminal(
            local_goal_sequence=1,
            action_goal_id="real-ros-goal-1",
            outcome=outcome,
            finished_monotonic_ns=1_100_000_000,
            generation=ledger.generation,
            clock_domain_id=CLOCK,
        )
    assert ledger.command(1).lifecycle is GripperLifecycle.INVALIDATED
    assert ledger.transition_quarantined("dependent-new") is True
    assert ledger.transition_quarantined("dependent-held") is True
    assert ledger.eligible_for_replay("dependent-new") is False
    with pytest.raises(GripperProvenanceError, match="SEAL_BLOCKED"):
        ledger.seal_episode()


def test_rejected_before_acceptance_is_invalidated_and_cannot_authorize():
    ledger = _ledger()
    ledger.begin_command(
        local_goal_sequence=1,
        requested_state="CLOSED",
        requested_width_m=0.0,
        started_monotonic_ns=1_000_000_000,
        generation=ledger.generation,
        clock_domain_id=CLOCK,
    )
    with pytest.raises(GripperProvenanceError, match="TERMINAL_REJECTED"):
        ledger.record_terminal(
            local_goal_sequence=1,
            action_goal_id=None,
            outcome="rejected",
            finished_monotonic_ns=1_010_000_000,
            generation=ledger.generation,
            clock_domain_id=CLOCK,
        )
    assert ledger.command(1).lifecycle is GripperLifecycle.INVALIDATED
    with pytest.raises(GripperProvenanceError, match="LEASE_UNAVAILABLE"):
        ledger.new_command_authority(
            transition_id="never-accepted", authority_monotonic_ns=1_020_000_000
        )


@pytest.mark.parametrize("duplicate_outcome", ["reached", "stalled"])
def test_duplicate_or_conflicting_terminal_after_seal_revokes_replay_authority(
    duplicate_outcome,
):
    ledger = _ledger()
    _accept(ledger)
    ledger.new_command_authority(
        transition_id="sealed-before-duplicate", authority_monotonic_ns=1_020_000_000
    )
    ledger.record_terminal(
        local_goal_sequence=1,
        action_goal_id="real-ros-goal-1",
        outcome="reached",
        finished_monotonic_ns=1_100_000_000,
        generation=ledger.generation,
        clock_domain_id=CLOCK,
    )
    ledger.seal_episode()
    assert ledger.eligible_for_replay("sealed-before-duplicate") is True
    with pytest.raises(GripperProvenanceError, match="LATE_DUPLICATE"):
        ledger.record_terminal(
            local_goal_sequence=1,
            action_goal_id="real-ros-goal-1",
            outcome=duplicate_outcome,
            finished_monotonic_ns=1_100_000_000,
            generation=ledger.generation,
            clock_domain_id=CLOCK,
        )
    assert ledger.command(1).lifecycle is GripperLifecycle.INVALIDATED
    assert ledger.transition_quarantined("sealed-before-duplicate") is True
    assert ledger.eligible_for_replay("sealed-before-duplicate") is False


@pytest.mark.parametrize("boundary", ["reset", "takeover", "revision", "episode"])
def test_generation_boundaries_invalidate_lease_and_unsealed_transitions(boundary):
    ledger = _ledger()
    _accept(ledger)
    ledger.new_command_authority(
        transition_id=f"before-{boundary}", authority_monotonic_ns=1_020_000_000
    )
    if boundary == "reset":
        updated = replace(ledger.generation, reset_generation=1)
        ledger.reset_home(new_generation=updated)
    elif boundary == "takeover":
        updated = replace(ledger.generation, takeover_generation=1)
        ledger.human_takeover(new_generation=updated)
    elif boundary == "revision":
        updated = replace(
            ledger.generation, policy_revision="revision-B", policy_epoch=4
        )
        ledger.policy_revision(new_generation=updated)
    else:
        updated = replace(ledger.generation, episode_id="episode-B")
        ledger.episode_change(new_generation=updated)
    assert ledger.command(1).lifecycle is GripperLifecycle.INVALIDATED
    assert ledger.lifecycle is GripperLifecycle.UNBOUND
    assert ledger.transition_quarantined(f"before-{boundary}") is True
    assert ledger.eligible_for_replay(f"before-{boundary}") is False
    with pytest.raises(GripperProvenanceError, match="WITHOUT_ACCEPTED_ORIGIN"):
        ledger.held_authority(
            transition_id=f"held-after-{boundary}",
            requested_state="CLOSED",
            requested_width_m=0.0,
            authority_monotonic_ns=1_030_000_000,
            feedback=GripperFeedback(
                measured_width_m=0.047,
                measured_state="INTERMEDIATE",
                feedback_monotonic_ns=1_030_000_000,
                clock_domain_id=CLOCK,
                generation=updated,
            ),
        )


def test_conflicting_new_command_terminates_old_lease_authority():
    ledger = _ledger()
    _accept(ledger)
    ledger.held_authority(
        transition_id="old-held",
        requested_state="CLOSED",
        requested_width_m=0.0,
        authority_monotonic_ns=1_025_000_000,
        feedback=_feedback(ledger),
    )
    ledger.begin_command(
        local_goal_sequence=2,
        requested_state="OPEN",
        requested_width_m=0.085,
        started_monotonic_ns=1_030_000_000,
        generation=ledger.generation,
        clock_domain_id=CLOCK,
    )
    assert ledger.command(1).lifecycle is GripperLifecycle.INVALIDATED
    assert ledger.transition_quarantined("old-held") is True
    assert ledger.lifecycle is GripperLifecycle.COMMAND_PENDING
    ledger.accept_command(
        local_goal_sequence=2,
        action_goal_id="actual-open-goal",
        accepted_monotonic_ns=1_031_000_000,
        generation=ledger.generation,
        clock_domain_id=CLOCK,
    )
    assert ledger.lifecycle is GripperLifecycle.ACCEPTED_ACTIVE


def test_late_terminal_after_episode_boundary_fails_closed():
    ledger = _ledger()
    original = ledger.generation
    _accept(ledger)
    ledger.episode_change(new_generation=replace(original, episode_id="episode-B"))
    with pytest.raises(GripperProvenanceError, match="GENERATION_STALE"):
        ledger.record_terminal(
            local_goal_sequence=1,
            action_goal_id="real-ros-goal-1",
            outcome="reached",
            finished_monotonic_ns=1_100_000_000,
            generation=original,
            clock_domain_id=CLOCK,
        )


def _runtime_pose_entry(*, gripper_goal_id: str):
    limits = RuntimeSafetyLimits(
        max_chunk_age_ns=2_000_000_000,
        max_selected_index=49,
        max_dispatch_count=8,
        refresh_worst_case_service_ns=450_000_000,
        refresh_additional_headroom_ns=100_000_000,
        pose_ack_deadline_ns=20_000_000,
        gripper_ack_deadline_ns=100_000_000,
    )
    runtime = RationalH50SelectionLedger(
        limits,
        policy_revision="revision-A",
        clock_domain_id=CLOCK,
        policy_epoch=3,
    )
    request = ChunkRequestIdentity(
        request_id="request-1",
        chunk_id="chunk-1",
        proposal_id="proposal-1",
        policy_revision="revision-A",
        policy_epoch=3,
        takeover_generation=0,
        reset_generation=0,
        request_clock_domain_id=CLOCK,
        t_ref_clock_domain_id=CLOCK,
        t_ref_ns=1_000_000_000,
    )
    runtime.pin_request(request)
    runtime.adopt_result(
        ChunkResultIdentity(
            **request.__dict__, result_id="result-1", result_clock_domain_id=CLOCK
        ),
        actions_cached=50,
    )
    runtime.begin_dispatch(
        dispatch_sequence=1,
        selection_ns=1_100_000_000,
        selection_clock_domain_id=CLOCK,
        dispatch_ns=1_100_000_000,
        dispatch_clock_domain_id=CLOCK,
        selected_post_adapter_absolute7=(0.5, 0.0, 0.2, 0.1, -0.2, 0.3, 0.0),
        pose_command_id="pose-command-1",
        gripper_command_id=gripper_goal_id,
    )
    runtime.record_pose_ack(
        dispatch_sequence=1,
        ack_id="pose-command-1",
        accepted=True,
        ack_ns=1_101_000_000,
        ack_clock_domain_id=CLOCK,
    )
    return runtime, runtime.entries[0]


@pytest.mark.parametrize(
    "authority_kind", [GripperAuthorityKind.NEW_COMMAND, GripperAuthorityKind.HELD_FROM_ACCEPTED_COMMAND]
)
def test_g7c1_pose_ack_plus_gripper_authority_closes_cpu_action7_without_lowering_ack(
    authority_kind,
):
    gripper = _ledger()
    _accept(
        gripper,
        accepted_ns=1_050_000_000,
        goal_id="real-origin-action-goal",
    )
    if authority_kind is GripperAuthorityKind.NEW_COMMAND:
        evidence = gripper.new_command_authority(
            transition_id="runtime-transition", authority_monotonic_ns=1_102_000_000
        )
    else:
        evidence = gripper.held_authority(
            transition_id="runtime-transition",
            requested_state="CLOSED",
            requested_width_m=0.0,
            authority_monotonic_ns=1_102_000_000,
            feedback=_feedback(gripper, timestamp_ns=1_101_500_000),
        )
    runtime, entry = _runtime_pose_entry(
        gripper_goal_id="real-origin-action-goal"
    )
    with pytest.raises(RuntimeSafetyViolation, match="NOT_ACK_AUTHORITATIVE"):
        entry.to_accepted_ack(clock_domain_id=CLOCK)
    pose = pose_authority_from_accepted_reference(
        entry, transition_id="runtime-transition", episode_id="episode-A"
    )
    staged = close_full_action7_authority(
        pose=pose,
        selected_post_adapter_tcp6=(0.5, 0.0, 0.2, 0.1, -0.2, 0.3),
        selected_gripper_width_m=0.0,
        gripper=evidence,
    )
    assert staged.accepted_absolute_action7 == entry.selected_post_adapter_absolute7
    assert staged.terminal_sealed_for_replay is False

    gripper.record_terminal(
        local_goal_sequence=1,
        action_goal_id="real-origin-action-goal",
        outcome="stalled",
        finished_monotonic_ns=1_150_000_000,
        generation=gripper.generation,
        clock_domain_id=CLOCK,
    )
    sealed_evidence = gripper.seal_episode()[0]
    sealed = close_full_action7_authority(
        pose=pose,
        selected_post_adapter_tcp6=pose.selected_post_adapter_tcp6,
        selected_gripper_width_m=0.0,
        gripper=sealed_evidence,
    )
    assert sealed.terminal_sealed_for_replay is True
    assert runtime.entries[0].gripper_ack_id is None


def test_full_action7_rejects_value_only_or_cross_generation_claims():
    gripper = _ledger()
    _accept(gripper)
    evidence = gripper.held_authority(
        transition_id="transition-1",
        requested_state="CLOSED",
        requested_width_m=0.0,
        authority_monotonic_ns=1_025_000_000,
        feedback=_feedback(gripper),
    )
    _, entry = _runtime_pose_entry(gripper_goal_id="wrong-origin")
    pose = pose_authority_from_accepted_reference(
        entry, transition_id="transition-1", episode_id="episode-A"
    )
    with pytest.raises(GripperProvenanceError, match="FULL_ACTION7_AUTHORITY_INVALID"):
        close_full_action7_authority(
            pose=pose,
            selected_post_adapter_tcp6=pose.selected_post_adapter_tcp6,
            selected_gripper_width_m=0.0,
            gripper=evidence,
        )
    with pytest.raises(GripperProvenanceError, match="SELECTED_ACTION_MISMATCH"):
        close_full_action7_authority(
            pose=replace(
                pose,
                declared_gripper_origin_action_goal_id="real-ros-goal-1",
            ),
            selected_post_adapter_tcp6=(0.4,) + pose.selected_post_adapter_tcp6[1:],
            selected_gripper_width_m=0.0,
            gripper=evidence,
        )


def _measured_state(width_m: float) -> str:
    if width_m >= 0.080:
        return "OPEN"
    if width_m <= 0.005:
        return "CLOSED"
    return "INTERMEDIATE"


def test_recorded_offline_episode_pairs_real_goals_held_feedback_and_terminal_seal():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    trace = config["recorded_offline_trace"]
    episode = Path(trace["episode_dir"])
    result_path = episode / "episode_result.json"
    target_path = episode / "streams/gripper_target.jsonl"
    status_path = episode / "streams/gripper_goal_status.jsonl"
    state_path = episode / "streams/gripper_state.jsonl"
    assert _sha256(result_path) == trace["episode_result_sha256"]
    assert _sha256(target_path) == trace["gripper_target_sha256"]
    assert _sha256(status_path) == trace["gripper_goal_status_sha256"]
    assert _sha256(state_path) == trace["gripper_state_sha256"]

    result = json.loads(result_path.read_text(encoding="utf-8"))
    targets = [json.loads(line) for line in target_path.read_text().splitlines() if line]
    statuses = [json.loads(line) for line in status_path.read_text().splitlines() if line]
    states = [json.loads(line) for line in state_path.read_text().splitlines() if line]
    assert result["saved"] is True
    assert result["fatal_reason"] is None
    assert len(targets) == len(statuses) == trace["goal_count"] == 2
    assert len(states) == trace["state_count"] == 14312
    assert {item["outcome"] for item in statuses} == VALID_TERMINAL_OUTCOMES
    diagnostics = result["native_stream_quality"]["gripper_state"]["goal_aware_validation"]
    assert diagnostics["violations"] == []
    assert all(
        goal["accepted_state_delta_ms"]
        <= config["clock_contract"]["max_feedback_age_ms"]
        for goal in diagnostics["goals"]
    )

    target = targets[0]
    status = statuses[0]
    generation = _generation(
        episode_id="task1/episode_000017",
        policy_revision="recorded-offline-no-policy",
        policy_epoch=0,
    )
    ledger = _ledger(generation)
    ledger.begin_command(
        local_goal_sequence=target["local_goal_sequence"],
        requested_state=target["requested_state"],
        requested_width_m=target["target_width_m"],
        started_monotonic_ns=target["started_monotonic_ns"],
        generation=generation,
        clock_domain_id=CLOCK,
    )
    ledger.accept_command(
        local_goal_sequence=target["local_goal_sequence"],
        action_goal_id=target["action_goal_id"],
        accepted_monotonic_ns=target["accepted_monotonic_ns"],
        generation=generation,
        clock_domain_id=CLOCK,
    )
    new_evidence = ledger.new_command_authority(
        transition_id="recorded-new",
        authority_monotonic_ns=target["accepted_monotonic_ns"] + 1,
    )
    assert new_evidence.origin_action_goal_id == target["action_goal_id"]
    ledger.record_terminal(
        local_goal_sequence=status["local_goal_sequence"],
        action_goal_id=status["action_goal_id"],
        outcome=status["outcome"],
        finished_monotonic_ns=status["finished_monotonic_ns"],
        generation=generation,
        clock_domain_id=CLOCK,
    )
    feedback_record = next(
        item
        for item in states
        if item["receive_monotonic_ns"] >= status["finished_monotonic_ns"]
    )
    held = ledger.held_authority(
        transition_id="recorded-held",
        requested_state=target["requested_state"],
        requested_width_m=target["target_width_m"],
        authority_monotonic_ns=feedback_record["receive_monotonic_ns"],
        feedback=GripperFeedback(
            measured_width_m=feedback_record["width_m"],
            measured_state=_measured_state(feedback_record["width_m"]),
            feedback_monotonic_ns=feedback_record["receive_monotonic_ns"],
            clock_domain_id=CLOCK,
            generation=generation,
        ),
    )
    assert held.origin_action_goal_id == target["action_goal_id"]
    assert held.feedback_age_ns == 0
    sealed = ledger.seal_episode()
    assert {item.transition_id for item in sealed} == {"recorded-new", "recorded-held"}
    assert all(item.terminal_outcome == "stalled" for item in sealed)
    assert ledger.eligible_for_replay("recorded-new") is True
    assert ledger.eligible_for_replay("recorded-held") is True

    next_episode = replace(generation, episode_id="task1/episode_000018")
    ledger.episode_change(new_generation=next_episode)
    assert ledger.lifecycle is GripperLifecycle.UNBOUND
    with pytest.raises(GripperProvenanceError, match="WITHOUT_ACCEPTED_ORIGIN"):
        ledger.held_authority(
            transition_id="cross-episode-held",
            requested_state=target["requested_state"],
            requested_width_m=target["target_width_m"],
            authority_monotonic_ns=feedback_record["receive_monotonic_ns"] + 1,
            feedback=replace(
                _feedback(_ledger(next_episode)),
                feedback_monotonic_ns=feedback_record["receive_monotonic_ns"],
                generation=next_episode,
            ),
        )


def test_single_owner_and_import_are_cpu_only_without_ros_network_or_cuda():
    ledger = _ledger()
    errors = []

    def cross_thread():
        try:
            _ = ledger.lifecycle
        except Exception as error:  # noqa: BLE001 - assertion captures exact type below
            errors.append(error)

    thread = threading.Thread(target=cross_thread)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], GripperProvenanceError)
    assert "CROSS_THREAD" in str(errors[0])

    script = """
import importlib.abc, sys, threading
before = {t.ident for t in threading.enumerate()}
class BlockRobotImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname.split('.', 1)[0] in {'rclpy', 'rospy'}:
            raise AssertionError('forbidden import: ' + fullname)
        return None
def audit(event, args):
    del args
    if event.startswith('socket.'):
        raise AssertionError('network side effect: ' + event)
    if event in {
        'os.exec', 'os.posix_spawn', 'os.system', 'pty.spawn', 'subprocess.Popen'
    }:
        raise AssertionError('process side effect: ' + event)
sys.meta_path.insert(0, BlockRobotImports())
sys.addaudithook(audit)
import forcesmolvla.rft.online.gripper_authority
import torch
after = {t.ident for t in threading.enumerate()}
assert before == after
assert 'CUDA_VISIBLE_DEVICES' in __import__('os').environ
assert not torch.cuda.is_initialized()
"""
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    assert GRIPPER_NOOP_ACK_POLICY == "BOUND"
    assert GRIPPER_FEEDBACK_FRESHNESS_BOUND is True
    assert GRIPPER_TERMINAL_SEAL_REQUIRED is True
    assert FULL_ACTION7_ACK_CLOSURE_PRODUCTION is False
    assert PRODUCTION_INTEGRATION_BLOCKED_ON_GRIPPER_ACK is True
