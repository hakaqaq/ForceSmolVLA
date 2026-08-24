import copy
import json
from pathlib import Path

import pytest

from forcesmolvla.shadow import (
    ShadowProtocol,
    ShadowResolution,
    arbitrate_shadow_candidates,
    build_shadow_record_artifact,
    evaluate_shadow_candidate,
    replay_shadow_record_artifact,
    resolve_shadow_artifacts,
)
from preflight_p9_offline_replay import (
    _episode_clock_diagnostics,
    _load_scope_amendment,
    _source_stamp_to_host_monotonic,
)
from train_task2_full_gpu import (
    _bind_task2_fixture_provenance,
    _final_checkpoint_due,
)


ROOT = Path(__file__).parents[1]
RULES = ROOT / "tests/fixtures/shadow_safety_thresholds.test_only.yaml"
CLOCK = ROOT / "tests/fixtures/shadow_clock_map.test_only.json"
SCHEMA = ROOT / "schemas/rulespec.schema.json"
PROTOCOL = ShadowProtocol.from_dict(
    json.loads((ROOT / "configs/p9_shadow_replay.development.json").read_text())["protocol"]
)
DIGEST = "a" * 64


def resolution(mode="test_only", clock=CLOCK):
    return resolve_shadow_artifacts(
        mode=mode,
        rules_path=RULES,
        schema_path=SCHEMA,
        clock_map_path=clock,
        test_fixture_root=ROOT / "tests/fixtures",
    )


def source(generation: int, ready_after_ref_ns: int, *, invalid_workspace=False):
    t_ref_sensor = 1_000_000_000 + generation * 100_000_000
    t_ref_controller = t_ref_sensor + 1_000_000_000
    t_ready_controller = t_ref_controller + ready_after_ref_ns
    target = [20.0 if invalid_workspace else 0.5 + generation * 0.001, 0.0, 0.2, 0.0, 0.2, 0.0, 0.05]
    chunk = [target[:] for _ in range(50)]
    context = {
        "policy_generation": generation,
        "raw_state_snapshot": [target],
        "t_ref_ns": [t_ref_controller],
        "tau0_ns": [t_ref_controller],
        "clock_domain_id": ["test_controller_monotonic_ns"],
        "episode_id": ["synthetic"],
        "session_id": ["test_only"],
        "sample_id": [f"sample-{generation}"],
        "chunk_id": [f"chunk-{generation}"],
        "action_valid_mask": [[True] * 50],
        "suffix_valid_mask": [[True] * 50],
        "calibration_bundle_hash": [DIGEST],
        "wrench_geometry_spec_hash": [DIGEST],
        "normalizer_hash": [DIGEST],
        "calibration_mapping_hash_or_none": [None],
        "wrench_geometry_valid": [True],
        "runtime_artifact_compatible": [True],
        "selected_provenance": [{"fixture": True}],
    }
    return {
        "generation": generation,
        "policy_tick_index": generation,
        "sensor_clock_domain": "test_sensor_monotonic_ns",
        "gpu_clock_domain": "test_gpu_monotonic_ns",
        "t_ref_sensor_ns": t_ref_sensor,
        "t_ready_gpu_ns": t_ready_controller - 2_000_000_000,
        "transport_ns": 5_000_000,
        "tau0_controller_ns": t_ref_controller,
        "observation_timestamps_sensor_ns": {
            "camera1": t_ref_sensor - 10_000_000,
            "camera2": t_ref_sensor - 9_000_000,
            "state": t_ref_sensor - 8_000_000,
            "wrench": t_ref_sensor - 7_000_000,
        },
        "raw_state7": target,
        "normalized_delta7_chunk": [[0.0] * 7 for _ in range(50)],
        "absolute_action7_chunk": chunk,
        "action_valid_mask": [True] * 50,
        "runtime_artifact_compatible": True,
        "wrench_geometry_valid": True,
        "chunk_context": context,
        "calibration_id": "synthetic",
        "calibration_bundle_hash": DIGEST,
        "normalizer_hash": DIGEST,
        "wrench_geometry_spec_hash": DIGEST,
        "raw_and_filter_timestamps": {"raw_ns": t_ref_sensor - 7_000_000, "filter_ns": t_ref_sensor - 7_000_000},
        "camera": {
            "camera1": {"id": "D435-third-person", "timestamp_ns": t_ref_sensor - 10_000_000, "sha256": DIGEST},
            "camera2": {"id": "D405-wrist", "timestamp_ns": t_ref_sensor - 9_000_000, "sha256": DIGEST},
        },
        "prompt": {"text_sha256": DIGEST, "token_sha256": DIGEST},
        "noise": {"seed": 42, "tensor_sha256": DIGEST},
    }


def candidates(*, invalid_new=False):
    resolved = resolution()
    assert resolved.valid
    sources = [source(0, 90_000_000), source(1, 20_000_000, invalid_workspace=invalid_new)]
    outcomes = [
        evaluate_shadow_candidate(item, resolution=resolved, protocol=PROTOCOL)
        for item in sources
    ]
    return resolved, sources, outcomes


def test_shadow_tick_schedule_and_latest_generation_wins():
    resolved, sources, outcomes = candidates()
    dispatch, run = arbitrate_shadow_candidates(
        sources,
        outcomes,
        rules=resolved.rules,
        protocol=PROTOCOL,
        run_end_controller_ns=2_300_000_000,
    )
    assert outcomes[0]["timing"]["j"] == 3
    assert outcomes[1]["timing"]["j"] == 1
    assert dispatch[0]["actual_dispatched_indices"] == [3]
    assert dispatch[0]["cancelled_indices"] == [4, 5]
    assert dispatch[1]["actual_dispatched_indices"] == [1, 2, 3]
    assert run["arrival_monotonic_strict"]
    assert run["intervals_nonnegative_nonoverlap"]
    assert run["run_valid"]


def test_invalid_new_candidate_does_not_supersede_valid_plan():
    resolved, sources, outcomes = candidates(invalid_new=True)
    assert not outcomes[1]["candidate_valid"]
    assert "SHADOW_WORKSPACE_INVALID" in outcomes[1]["candidate_reasons"]
    dispatch, _run = arbitrate_shadow_candidates(
        sources,
        outcomes,
        rules=resolved.rules,
        protocol=PROTOCOL,
        run_end_controller_ns=2_267_000_000,
    )
    assert dispatch[0]["actual_dispatched_indices"] == [3, 4, 5]
    assert dispatch[0]["cancelled_indices"] == []
    assert dispatch[1]["actual_dispatched_indices"] == []


def test_production_missing_or_test_only_clock_map_fails_candidate_closed():
    missing = resolve_shadow_artifacts(
        mode="production",
        rules_path=RULES,
        schema_path=SCHEMA,
        clock_map_path=None,
        test_fixture_root=ROOT / "tests/fixtures",
    )
    assert not missing.valid
    assert "SHADOW_CLOCK_MAP_MISSING" in missing.reasons
    outcome = evaluate_shadow_candidate(source(0, 20_000_000), resolution=missing, protocol=PROTOCOL)
    assert not outcome["candidate_valid"]

    test_assets = resolution(mode="production")
    assert not test_assets.valid
    assert any("PRODUCTION" in reason or "RuleSpec" in reason for reason in test_assets.reasons)


def test_stale_and_mismatched_clock_maps_fail_candidate_closed():
    resolved = resolution()
    stale_map = copy.deepcopy(resolved.clock_map)
    stale_map["valid_until_controller_ns"] = 1
    stale_map["max_age_ns"] = 1
    stale = ShadowResolution(
        mode="production",
        valid=True,
        reasons=(),
        rules=resolved.rules,
        rules_sha256=resolved.rules_sha256,
        clock_map=stale_map,
        clock_map_sha256=DIGEST,
    )
    stale_outcome = evaluate_shadow_candidate(source(0, 20_000_000), resolution=stale, protocol=PROTOCOL)
    assert not stale_outcome["candidate_valid"]
    assert "SHADOW_CLOCK_MAP_STALE" in stale_outcome["candidate_reasons"]

    mismatch_source = source(0, 20_000_000)
    mismatch_source["gpu_clock_domain"] = "wrong_gpu_clock"
    mismatch = evaluate_shadow_candidate(mismatch_source, resolution=resolved, protocol=PROTOCOL)
    assert not mismatch["candidate_valid"]
    assert "SHADOW_CLOCK_DOMAIN_MISMATCH:gpu_to_controller" in mismatch["candidate_reasons"]


def test_record_replay_is_exact_and_hash_tamper_fails():
    resolved, sources, _outcomes = candidates()
    artifact = build_shadow_record_artifact(
        sources,
        resolution=resolved,
        protocol=PROTOCOL,
        run_end_controller_ns=2_300_000_000,
        artifact_hashes={"source_binding_sha256": DIGEST, "resolved_config_sha256": DIGEST},
    )
    replay = replay_shadow_record_artifact(artifact)
    assert replay["replay_exact"]
    assert replay["robot_actions_sent"] == 0
    assert all(record["production_shadow"] is False for record in artifact["records"])
    assert all(record["native_queue"]["used"] is False for record in artifact["records"])

    tampered = copy.deepcopy(artifact)
    tampered["records"][0]["source"]["transport_ns"] += 1
    with pytest.raises(RuntimeError, match="ARTIFACT_HASH_MISMATCH"):
        replay_shadow_record_artifact(tampered)


def test_terminal_hold_overrun_rejects_run():
    resolved = resolution()
    sources = [source(0, 20_000_000)]
    outcomes = [evaluate_shadow_candidate(sources[0], resolution=resolved, protocol=PROTOCOL)]
    last_arrival = outcomes[0]["timing"]["planned_arrival_ns"][-1]
    _dispatch, run = arbitrate_shadow_candidates(
        sources,
        outcomes,
        rules=resolved.rules,
        protocol=PROTOCOL,
        run_end_controller_ns=last_arrival + PROTOCOL.max_hold_extension_ns + 1,
    )
    assert run["hold_overrun"]
    assert not run["run_valid"]
    assert "SHADOW_HOLD_OVERRUN" in run["run_reasons"]


def test_p9_module_has_no_ros_or_robot_send_path():
    text = (ROOT / "src/forcesmolvla/shadow.py").read_text(encoding="utf-8").lower()
    for forbidden in ("import rclpy", "import rospy", "franky", "send_goal", "publish("):
        assert forbidden not in text


def test_p9_contract_binds_current_task2_p8_and_keeps_formal_unapproved():
    config = json.loads((ROOT / "configs/p9_shadow_replay.development.json").read_text())
    assert config["input_profile_revision"] == "v4.2-p9-task2-user-confirmed-2026-08-21"
    assert config["allowed_inputs"] == [
        "datasets/task2_lerobotv3",
        "golden_fixtures",
        "tests/fixtures",
    ]
    assert config["dataset"] == "datasets/task2_lerobotv3"
    assert config["checkpoint"] == (
        "outputs/development/p8_v4_2_r4_checkpoint_seed42_step000001"
    )
    amendment, data_scope = _load_scope_amendment(ROOT, config)
    assert amendment["scope"] == "P9_only"
    assert amendment["supersedes_visible_p9_dataset"] == "task1_v4_1"
    assert amendment["p4_p8_rerun_required"] is False
    assert data_scope["session_provenance"]["explicit_physical_session_id"] is None
    assert data_scope["session_provenance"]["legacy_fixture_session_id_status"] == (
        "invalid_legacy_metadata_for_task2"
    )
    assert data_scope["training_budget"]["target_samples"] == 40_000
    assert data_scope["training_budget"]["derived_optimizer_updates"] == 10_000
    assert data_scope["training_budget"]["checkpoint_policy"] == "final_update_only"
    prerequisite = config["p8_prerequisite"]
    assert prerequisite["gate_result"]["sha256"] == (
        "27fd7846c380875a5969d8e54e919508a202e4e75a3be6a9e24af9cafd46ca24"
    )
    assert prerequisite["required_exact_resume"] is True
    assert prerequisite["required_p9_started"] is False
    assert prerequisite["required_robot_actions_sent"] == 0
    assert config["formal_signature_algorithm"] is None
    assert config["formal_key_id"] is None
    assert config["formal_approver"] is None
    assert config["detached_signature"] is None
    assert config["approval"] is None


def test_p9_scheduler_index_uses_candidate_time_not_apply_time():
    config = json.loads((ROOT / "configs/p9_shadow_replay.development.json").read_text())
    amendment, _data_scope = _load_scope_amendment(ROOT, config)
    semantics = amendment["scheduler_index_semantics"]
    tau0_ns = 1_000_000_000
    t_candidate_ns = tau0_ns + semantics["observed_candidate_minus_tau0_ns"]
    assert semantics["observed_unrounded_index"] == "6.15"
    assert PROTOCOL.chunk_index(t_candidate_ns, tau0_ns) == semantics["expected_j"] == 7
    assert semantics["t_apply_based_formula"] is False


def test_task2_long_sft_fixture_replaces_legacy_task1_session_metadata():
    data_scope = json.loads(
        (ROOT / "configs/task2_development_data_scope.json").read_text()
    )
    fixture = {
        "chunk_context": {
            "session_id": ["task1_within_session", "task1_within_session"],
            "selected_provenance": [{"fixture_position": 0}, {"fixture_position": 1}],
        },
        "chunk_context_sha256": "stale",
    }
    _bind_task2_fixture_provenance(fixture, data_scope)
    expected = data_scope["session_provenance"]["collection_scope_id"]
    assert fixture["chunk_context"]["session_id"] == [expected, expected]
    assert "task1_within_session" not in fixture["chunk_context"]["session_id"]
    assert all(
        item["physical_session_id"] is None
        for item in fixture["chunk_context"]["selected_provenance"]
    )
    assert len(fixture["chunk_context_sha256"]) == 64


def test_task2_long_sft_saves_only_final_checkpoint():
    assert not _final_checkpoint_due(1, 10_000)
    assert not _final_checkpoint_due(500, 10_000)
    assert not _final_checkpoint_due(9_999, 10_000)
    assert _final_checkpoint_due(10_000, 10_000)


def test_p9_uses_bound_integer_episode_clock_map_for_device_source_stamps():
    conversion = {
        "episodes": [
            {
                "output_episode_index": 7,
                "diagnostics": {
                    "clock_map_id": "sha256:" + "a" * 64,
                    "clock_map_sha256": "a" * 64,
                    "clock_offset_ns": -1_786_970_000_390_251_520,
                },
            }
        ]
    }
    diagnostics = _episode_clock_diagnostics(conversion, 7)
    source_ns = 1_787_136_207_722_195_163
    assert _source_stamp_to_host_monotonic(source_ns, diagnostics) == 166_207_331_943_643
    with pytest.raises(RuntimeError, match="NOT_UNIQUE"):
        _episode_clock_diagnostics(conversion, 8)
