from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from forcesmolvla.rft.stage3.loopback import (
    canonical_report_sha256,
    recorded_fixture_blocked_report,
    rational_grid_for_macro,
    run_synthetic_loopback,
    validate_loopback_report,
)
from tools.run_stage3_recorded_loopback import main as loopback_cli_main


@pytest.fixture(scope="module")
def synthetic_report() -> dict:
    # The loopback must remain valid even if every Stage-2 Cal-QL entry point is
    # fail-fast.  Stage-3 online TD has no call edge to either function.
    with (
        patch("forcesmolvla.rft.losses.evaluate_calql_candidates", side_effect=AssertionError),
        patch("forcesmolvla.rft.losses.compute_calql_penalty", side_effect=AssertionError),
    ):
        return run_synthetic_loopback(seed=20260828)


def test_end_to_end_synthetic_H50_ack_K7_seal_replay_learner_staged_revision(
    synthetic_report: dict,
) -> None:
    report = validate_loopback_report(synthetic_report)
    assert report["fixture_kind"] == "synthetic_tool_test"
    assert report["collection"]["action_horizon"] == 50
    assert report["collection"]["accepted_slots_per_decision"] == 3
    assert report["collection"]["episode_sealed_before_commit"]
    assert report["replay"]["R_online_membership_count"] == 100
    assert report["learner"]["critic_gradient_steps"] == 2
    assert report["policy_revision"]["staged"]
    assert not report["formal_gate_passed"]


def test_rational_30hz_grid_has_fixed_10hz_anchor_phase() -> None:
    for macro_index in (0, 1, 17, 1000):
        grid = rational_grid_for_macro(macro_index)
        indices = tuple((tick * 30 + 500_000_000) // 1_000_000_000 for tick in grid)
        assert indices[0] % 3 == 0
        assert indices == tuple(range(indices[0], indices[0] + 3))
        assert grid == tuple((index * 1_000_000_000 + 15) // 30 for index in indices)


def test_99_unique_R_blocks_and_100_unique_R_unlocks(synthetic_report: dict) -> None:
    gate = synthetic_report["training_gate"]
    assert gate["training_starts_unique_R"] == 100
    assert gate["blocked_at_99"]
    assert gate["unlocked_at_100"]


def test_exact_R_D_50_50_and_intervention_dual_membership(
    synthetic_report: dict,
) -> None:
    replay = synthetic_report["replay"]
    learner = synthetic_report["learner"]
    assert replay["mixed_replay_ratio"] == "50_50"
    assert learner["batch_R_count"] == learner["batch_D_count"]
    assert replay["intervention_dual_membership"]
    assert replay["canonical_payload_copies_per_uid"] == 1
    assert replay["independent_offline_demonstration"]
    assert learner["autonomous_fm_contribution"] == 0.0
    assert learner["expert_feature_count"] > 0


def test_two_critic_one_actor_and_two_polyak_updates(synthetic_report: dict) -> None:
    learner = synthetic_report["learner"]
    assert learner["critic_gradient_steps"] == 2
    assert learner["actor_gradient_steps"] == 1
    assert learner["target_polyak_updates"] == 2
    assert learner["critic_only_actor_unchanged"]
    assert learner["actor_optimizer_critics_unchanged"]


def test_calql_monkeypatch_to_raise_remains_uncalled(synthetic_report: dict) -> None:
    learner = synthetic_report["learner"]
    assert learner["online_critic_is_pure_td"]
    assert learner["calql_online_call_count"] == 0
    assert learner["cql_penalty_call_count"] == 0
    assert learner["random_candidate_call_count"] == 0
    assert learner["mc_return_read_count"] == 0
    assert learner["td_target_uses_target_twin_q_min"]


def test_actioncontract_v2_q_gradient_and_test_optimizer_ownership(
    synthetic_report: dict,
) -> None:
    learner = synthetic_report["learner"]
    assert learner["actor_guidance_uses_current_min_twin_q"]
    assert learner["tcp6_q_gradient_nonzero"]
    assert learner["gripper_q_gradient_exact_zero"]
    assert learner["post_K_q_gradient_exact_zero"]
    assert learner["zero_expert_graph_connected_finite"]
    assert learner["optimizer_kind"] == "test_only_sgd"
    assert learner["optimizer_excludes_vision_smolvlm_state_prefix"]
    assert synthetic_report["deferred"]["CROSS_STAGE_OPTIMIZER_REBUILT"] == "NOT_RUN"


def test_zero_credit_backpressure(synthetic_report: dict) -> None:
    gate = synthetic_report["training_gate"]
    assert gate["test_only_credit_configuration"]["production_policy"] is False
    assert gate["credits_after_cycle"] == 0
    assert gate["zero_credit_backpressure"]


def test_duplicate_commit_idempotence_and_conflicting_digest_rejection(
    synthetic_report: dict,
) -> None:
    faults = synthetic_report["fault_injection"]
    assert faults["duplicate_same_uid_same_digest_noop"]
    assert faults["conflicting_digest_fail_closed"]


def test_human_takeover_invalidates_stale_policy_chunk(
    synthetic_report: dict,
) -> None:
    assert synthetic_report["fault_injection"][
        "human_takeover_invalidated_stale_policy"
    ]
    assert synthetic_report["replay"]["intervention_dual_membership"]


def test_partial_missing_rejected_and_stale_ack_quarantine_without_replay(
    synthetic_report: dict,
) -> None:
    faults = synthetic_report["fault_injection"]
    assert faults["partial_macro_quarantined"]
    assert faults["missing_ack_quarantined"]
    assert faults["rejected_ack_quarantined"]
    assert faults["stale_ack_quarantined"]
    assert faults["quarantined_fault_replay_commit_count"] == 0


def test_revision_staged_but_never_activated(synthetic_report: dict) -> None:
    revision = synthetic_report["policy_revision"]
    assert revision["staged"]
    assert not revision["activated"]
    assert revision["active_revision_id"] == "fake-active-r0"
    assert revision["episode_activation_blocked"]
    assert revision["inflight_activation_blocked"]
    assert not revision["publisher_connected"]
    assert not revision["deployment_directory_written"]


def test_frozen_normalizer_applied_exactly_once_per_accepted_macro(
    synthetic_report: dict,
) -> None:
    collection = synthetic_report["collection"]
    assert collection["normalizer_application_count"] == collection["successful_macro_count"]
    assert collection["gripper_command_ack_identity"]


def test_two_identical_seeded_runs_have_same_canonical_report_digest(
    synthetic_report: dict,
) -> None:
    repeated = run_synthetic_loopback(seed=20260828)
    assert repeated == synthetic_report
    assert repeated["canonical_report_sha256"] == canonical_report_sha256(repeated)


def test_recorded_live_missing_fixture_returns_schema_valid_BLOCKED(tmp_path: Path) -> None:
    missing = tmp_path / "recorded-live.json"
    report = recorded_fixture_blocked_report(missing)
    assert validate_loopback_report(report) == report
    assert report["tool_status"] == "BLOCKED"
    assert report["blocked_reason"] == "RECORDED_LIVE_FIXTURE_MISSING"
    assert not report["formal_gate_passed"]


def test_cli_writes_schema_valid_recorded_live_BLOCKED_report(tmp_path: Path) -> None:
    missing = tmp_path / "missing-recorded-live.json"
    output = tmp_path / "report.json"
    exit_code = loopback_cli_main(
        [
            "--fixture-kind", "recorded_live",
            "--fixture", str(missing),
            "--output", str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert validate_loopback_report(report) == report
    assert report["tool_status"] == "BLOCKED"
