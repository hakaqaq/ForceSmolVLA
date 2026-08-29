from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator
import numpy as np
import pytest
import yaml


ROOT = Path(__file__).parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
g7 = importlib.import_module("benchmark_stage3_actor_learner_coexistence_gpu")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/stage3_gpu_coexistence.v1.development.yaml").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def measured_report() -> dict:
    return json.loads(
        (ROOT / "artifacts/development/stage3/stage3_gpu_coexistence.v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_config_freezes_full_workload_and_safety_scope(config: dict) -> None:
    validated = g7.validate_config(config)
    assert validated["learner"] == {
        "config": "configs/stage3_gpu_preflight.v1.development.yaml",
        "critic_batch_size": 64,
        "actor_batch_size": 24,
        "flow_subbatch": 4,
        "critic_updates_per_cycle": 2,
        "actor_updates_per_cycle": 1,
        "polyak_updates_per_cycle": 2,
        "warmup_cycles": 1,
        "measured_cycles": 3,
        "real_online_R_used": False,
        "R_source": "synthetic_preflight_R_only",
    }
    assert validated["safety"] == {
        "network_server_authorized": False,
        "robot_execution_authorized": False,
        "replay_write_authorized": False,
        "checkpoint_writeback_authorized": False,
        "policy_publication_authorized": False,
        "g8_and_later": "NOT_RUN",
    }
    assert validated["time_sliced"] == {
        "inter_phase_gap_ms": 1000,
        "inter_phase_gap_is_execution_budget": False,
        "topology": "cold_process_swap",
        "resident_time_slicing": "NOT_RUN",
        "synthetic_reset_window": True,
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("learner", "critic_batch_size"), 16),
        (("learner", "actor_batch_size"), 8),
        (("learner", "flow_subbatch"), 2),
        (("deadline", "inference_deadline_source"), "100ms"),
        (("deadline", "deadline_equals_macro_period_assumed"), True),
        (("safety", "network_server_authorized"), True),
        (("time_sliced", "inter_phase_gap_is_execution_budget"), True),
        (("time_sliced", "topology"), "resident"),
    ],
)
def test_config_rejects_degraded_or_expanded_scope(
    config: dict, path: tuple[str, str], value: object,
) -> None:
    changed = deepcopy(config)
    changed[path[0]][path[1]] = value
    with pytest.raises(g7.G7PError):
        g7.validate_config(changed)


def test_source_audit_binds_real_symbols_and_leaves_approval_fields_unbound() -> None:
    audit = g7.source_audit()
    assert audit["temporal_contract"] == {
        "source": "configs/stage3_transition_contract.v1.development.json:/temporal",
        "grid_hz": 30,
        "macro_hz": 10,
        "H": 50,
        "K": 3,
        "macro_period_ms": 100,
        "execution_source": "src/forcesmolvla/rft/stage3/transition.py:causal_zoh_ack_macro",
        "grid_source": "src/forcesmolvla/temporal.py:controller_reference_grid",
    }
    assert audit["INFERENCE_DEADLINE_SOURCE"] == "UNBOUND"
    assert audit["QUEUE_LOW_WATERMARK_SOURCE"] == "UNBOUND"
    assert audit["HOLD_POLICY_SOURCE"] == "UNBOUND"
    assert audit["DEADLINE_EQ_MACRO_PERIOD_ASSUMED"] is False
    assert audit["decoded_image_cache"]["production_bounded"] is False
    assert audit["learner_workload"]["orchestrator"].endswith(":run_gpu_preflight")


def test_benchmark_only_decoded_cache_is_byte_bounded_lru() -> None:
    cache = g7.BoundedDecodedImageCache(max_bytes=8)
    calls = []

    def decode(encoded: dict) -> np.ndarray:
        calls.append(encoded["data"])
        return np.frombuffer(encoded["data"].encode(), dtype=np.uint8).copy()

    first = {"data": "aaaa"}
    second = {"data": "bbbbbb"}
    assert cache.get_or_decode(first, decode).tolist() == [97] * 4
    assert cache.get_or_decode(first, decode).tolist() == [97] * 4
    cache.get_or_decode(second, decode)
    report = cache.report()
    assert report["max_bytes"] == 8
    assert report["hits"] == 1 and report["misses"] == 2
    assert report["evictions"] == 1
    assert report["current_bytes"] <= report["max_bytes"]
    assert calls == ["aaaa", "bbbbbb"]


def test_request_summary_never_calls_macro_period_an_approved_deadline() -> None:
    records = [
        {
            "result_ready_ns": 200_000_000 + index * 100_000_000,
            "release_actual_ns": index * 100_000_000,
            "release_jitter_ms": float(index),
            "end_to_end_ms": 200.0,
            "preprocessing_ms": 5.0,
            "queue_wait_ms": 2.0,
            "gpu_service_ms": 190.0,
            "benchmark_macro_lateness_ms": 100.0,
            "queue_depth_at_consume": index,
        }
        for index in range(3)
    ]
    summary = g7.summarize_requests(records, formal_minimum=1000)
    assert summary["p99_status"] == "PROVISIONAL_INSUFFICIENT_SAMPLES"
    assert summary["approved_deadline"] == {
        "source": "UNBOUND", "deadline_ms": None, "miss_count": None,
        "miss_rate": None, "maximum_consecutive_misses": None,
    }
    assert summary["benchmark_macro_period_reference"]["not_an_approved_deadline"] is True
    assert summary["simulated_hold"]["simulation_only"] is True


def _inference_result(digest: str, values: list[list[float]]) -> dict:
    return {
        "status": "PASS",
        "trials": [{
            "requests": [{
                "action_digest": digest, "shape": [50, 7], "dtype": "torch.float32",
                "finite": True, "action_values": values,
            }],
        }],
    }


def test_fixed_action_comparison_reports_exact_parity() -> None:
    values = [[0.0] * 7 for _ in range(50)]
    reference = {"worker": _inference_result("a" * 64, values)}
    concurrent = {"inference": _inference_result("a" * 64, values)}
    result = g7.compare_action_semantics(reference, concurrent)
    assert result["status"] == "PASS"
    assert result["max_absolute_error"] == result["max_relative_error"] == 0.0


def test_fixed_action_comparison_reports_first_difference_without_tolerance_growth() -> None:
    left = [[0.0] * 7 for _ in range(50)]
    right = deepcopy(left)
    right[3][2] = 0.25
    reference = {"worker": _inference_result("a" * 64, left)}
    concurrent = {"inference": _inference_result("b" * 64, right)}
    result = g7.compare_action_semantics(reference, concurrent)
    assert result["status"] == "FAIL"
    assert result["first_different_request"] == 0
    assert result["max_absolute_error"] == pytest.approx(0.25)
    assert result["tolerance_expanded"] is False


def test_current_g7_closure_is_separate_from_historical_g6_closure() -> None:
    closure = g7.current_source_closure()
    assert closure["historical_g6_closure_comparison_required"] is False
    assert any(row["path"].endswith("stage3_gpu_coexistence.v1.development.yaml") for row in closure["records"])
    assert len(closure["sha256"]) == 64


def test_schema_accepts_all_mode_statuses() -> None:
    schema = json.loads(
        (ROOT / "schemas/stage3_gpu_coexistence_report.v1.schema.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": "forcesmolvla_stage3_gpu_coexistence_report.v1",
        "tool_status": "PASS",
        "baseline": {}, "source_audit": {}, "gpu_preflight": {}, "bindings": {},
        "modes": {
            "inference_only": {"status": "PASS"},
            "learner_only": {"status": "PASS"},
            "concurrent": {"status": "FAIL"},
            "episode_time_sliced": {"status": "PASS"},
            "separate_device": {"status": "NOT_RUN"},
        },
        "comparisons": {},
        "semantic_audit": {
            "G7A_SEMANTIC_AUDIT": "PASS", "GPU_RERUN": False,
            "GPU_OPTIMIZER_STEP_RERUN": False, "G7A_RAW_COUNTS_CONSISTENT": True,
            "G7A_DROP_RATES_RECOMPUTED": True, "G7A_QUEUE_STABILITY_CLASSIFIED": True,
            "G7A_TIME_SLICED_TIMESTAMPS_SUFFICIENT": False,
            "G7A_QUIESCENT_WINDOW_SEMANTICS": "extended",
            "G7A_QUIESCENT_WINDOW_BUDGET_MET": False,
            "G7A_MINIMUM_MEASURED_QUIESCENT_WINDOW_MS": 15000.0,
            "request_accounting": {"modes": {}, "totals": {}},
            "queue_stability": {
                "QUEUE_MEMORY_BOUNDED_BY_CAP": True,
                "QUEUE_LOAD_STABLE_AT_10HZ": False, "NO_DROP_10HZ": False,
            },
            "time_sliced": {
                "configured_quiescent_window_ms": 1000,
                "configured_value_role": "inter_phase_gap",
                "control_flow_semantics": "extended",
                "g7p_time_sliced_semantics": "UNVERIFIED",
                "timestamps_sufficient": False, "quiescent_window_budget_met": False,
                "minimum_measured_quiescent_window_required_ms": 15000.0,
                "cycle_trials": [{}, {}, {}],
            },
            "memory": {
                "SHORT_RUN_MEMORY_SAFETY": "PASS", "OOM_COUNT": 0,
                "ALLOCATION_FAILURE_COUNT": 0, "SUSTAINED_MEMORY_LEAK_TEST": "NOT_RUN",
                "SUSTAINED_THERMAL_STABILITY": "UNVERIFIED",
                "PRODUCTION_DECODED_CACHE_BOUNDED": False,
            },
            "formal_conclusion": {
                "resource_coexistence": "PASS", "numerical_action_parity": "PASS",
                "synthetic_10hz_scheduling": "FAIL",
                "production_cadence_deadline": "UNBOUND", "production_topology": "NOT_APPROVED",
            },
            "G7P_EVIDENCE_FREEZE_ALLOWED": False,
        },
        "gates": {
            "G7P_BENCHMARK_EXECUTION": "PASS",
            "G7P_RESULT_SEMANTICS": "PASS_MEANS_MEASUREMENT_COMPLETED_ONLY",
            "G7P_100MS_SYNTHETIC_GRID_FEASIBLE": False,
            "G7P_INFERENCE_ONLY_10HZ_SUSTAINABLE": False,
            "G7P_CONCURRENT_10HZ_SUSTAINABLE": False,
            "G7P_TIME_SLICED_10HZ_SUSTAINABLE": False,
            "PRODUCTION_REQUEST_CADENCE_VALIDATED": False,
            "PRODUCTION_COLD_PROCESS_SWAP_APPROVED": False,
            "G7P_EVIDENCE_FREEZE_ALLOWED": False,
        },
        "safety": {}, "final_checks": {},
        "canonical_report_sha256": "0" * 64,
    }
    Draft202012Validator(schema).validate(report)


def test_existing_raw_semantic_audit_recomputes_counts_and_is_deterministic(
    config: dict, measured_report: dict,
) -> None:
    first = g7.semantic_consistency_audit(measured_report, config)
    second = g7.semantic_consistency_audit(measured_report, config)
    accounting = first["semantic_audit"]["request_accounting"]
    observed = {
        name: (item["scheduled"], item["completed"], item["dropped"])
        for name, item in accounting["modes"].items()
    }
    assert {name: observed[name] for name in ("inference_only", "concurrent")} == {
        "inference_only": (1002, 617, 385),
        "concurrent": (600, 317, 283),
    }
    assert observed["time_sliced_episode"][0] == 90
    assert observed["time_sliced_resume"][0] == 30
    totals = accounting["totals"]
    assert totals["scheduled"] == totals["completed"] + totals["dropped"]
    assert totals["drop_rate"] == pytest.approx(totals["dropped"] / totals["scheduled"])
    assert all(
        item["p99_status"] == "PROVISIONAL_INSUFFICIENT_SAMPLES"
        for item in accounting["modes"].values()
    )
    assert first["canonical_report_sha256"] == second["canonical_report_sha256"]
    assert first["gates"]["G7P_RESULT_SEMANTICS"] == "PASS_MEANS_MEASUREMENT_COMPLETED_ONLY"
    assert first["gates"]["PRODUCTION_COLD_PROCESS_SWAP_APPROVED"] is False
    assert first["gates"]["G7P_100MS_SYNTHETIC_GRID_FEASIBLE"] is False


def test_existing_raw_queue_and_time_sliced_semantics_are_fail_closed(
    config: dict, measured_report: dict,
) -> None:
    report = g7.semantic_consistency_audit(measured_report, config)
    audit = report["semantic_audit"]
    for name in ("inference_only", "concurrent"):
        queue_audit = audit["request_accounting"]["modes"][name]["queue"]
        assert queue_audit["reached_capacity_in_every_trial"] is True
        assert queue_audit["drops_occur_only_at_capacity"] is True
        assert queue_audit["queue_wait_dominates_p50_end_to_end"] is True
        assert queue_audit["queue_load_stable_at_10hz"] is False
    time_sliced = audit["time_sliced"]
    if "g7b_timestamp_instrumentation" in report:
        assert time_sliced["control_flow_semantics"] == "gap_not_budget"
        assert time_sliced["timestamps_sufficient"] is True
        assert time_sliced["g7p_time_sliced_semantics"] == "VERIFIED_COLD_PROCESS_SWAP_ONLY"
        assert audit["G7P_EVIDENCE_FREEZE_ALLOWED"] is True
    else:
        assert time_sliced["configured_value_role"].startswith("inter_phase_gap")
        assert time_sliced["control_flow_semantics"] == "extended"
        assert time_sliced["timestamps_sufficient"] is False
        assert time_sliced["g7p_time_sliced_semantics"] == "UNVERIFIED"
        assert time_sliced["learner_cycles_completed_within_first_1000ms"] == 0
        assert time_sliced["minimum_measured_quiescent_window_required_ms"] == pytest.approx(
            15392.991911037825
        )
        assert audit["G7P_EVIDENCE_FREEZE_ALLOWED"] is False


def test_tool_has_worker_only_cuda_imports_and_no_network_server_entrypoint() -> None:
    source = (ROOT / "tools/benchmark_stage3_actor_learner_coexistence_gpu.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in top_level_imports
    assert "ThreadingHTTPServer" not in source
    assert "serve_forever" not in source
    assert "rclpy" not in source and "rospy" not in source
    assert "run_gpu_preflight" in source


def test_separate_device_is_explicitly_not_run_in_schema_contract(config: dict) -> None:
    assert config["environment"]["expected_cuda_visible_devices"] == "0"
    assert config["environment"]["visible_cuda_device_index"] == 0
    assert config["environment"]["physical_cuda_device_index"] == 0


def _valid_g7b_trace() -> dict:
    order = [
        "episode_last_release_ns", "episode_queue_drained_ns", "episode_worker_exit_ns",
        "pre_learner_gap_start_ns", "pre_learner_gap_end_ns",
        "learner_spawn_requested_ns", "learner_process_spawn_ns", "learner_model_ready_ns",
        "learner_warmup_cycle_start_ns", "learner_warmup_cycle_end_ns",
        "learner_measured_cycle_1_start_ns", "learner_measured_cycle_1_end_ns",
        "learner_measured_cycle_2_start_ns", "learner_measured_cycle_2_end_ns",
        "learner_measured_cycle_3_start_ns", "learner_measured_cycle_3_end_ns",
        "learner_worker_exit_ns", "pre_resume_gap_start_ns", "pre_resume_gap_end_ns",
        "resume_requested_ns", "resume_process_spawn_ns", "resume_model_ready_ns",
        "resume_first_request_release_ns", "resume_first_service_start_ns",
        "resume_first_service_end_ns", "resume_first_result_ready_ns",
        "resume_queue_drained_ns", "resume_worker_exit_ns",
    ]
    trace = {name: (index + 1) * 2_000_000_000 for index, name in enumerate(order)}
    trace.update({
        "clock_source": "CLOCK_MONOTONIC",
        "linux_same_boot_cross_process_comparable": True,
        "cross_process_clock_scope": "same Linux boot CLOCK_MONOTONIC",
        "TIME_SLICED_TOPOLOGY": "cold_process_swap",
        "RESIDENT_TIME_SLICING": "NOT_RUN",
        "REAL_RESET_HOME_WINDOW_USED": False,
        "INTER_PHASE_GAP_IS_EXECUTION_BUDGET": False,
    })
    return trace


@pytest.mark.parametrize("missing", g7.G7B_REQUIRED_TIMESTAMPS)
def test_g7b_missing_each_required_timestamp_fails(missing: str) -> None:
    trace = _valid_g7b_trace()
    del trace[missing]
    with pytest.raises(g7.G7PError, match=f"G7B_TIMESTAMP_MISSING:{missing}"):
        g7.validate_g7b_timestamp_trace(trace, inter_phase_gap_ms=1000)


def test_g7b_out_of_order_timestamp_fails() -> None:
    trace = _valid_g7b_trace()
    trace["resume_model_ready_ns"] = trace["resume_process_spawn_ns"] - 1
    with pytest.raises(g7.G7PError, match="G7B_TIMESTAMP_ORDER"):
        g7.validate_g7b_timestamp_trace(trace, inter_phase_gap_ms=1000)


def test_g7b_negative_duration_fails() -> None:
    with pytest.raises(g7.G7PError, match="G7B_NEGATIVE_DURATION"):
        g7._ms(1, 2)


def test_g7b_cycle_overlap_fails() -> None:
    trace = _valid_g7b_trace()
    trace["learner_measured_cycle_2_start_ns"] = trace["learner_measured_cycle_1_end_ns"] - 1
    with pytest.raises(g7.G7PError):
        g7.validate_g7b_timestamp_trace(trace, inter_phase_gap_ms=1000)


def test_g7b_gap_mislabeled_as_budget_fails() -> None:
    trace = _valid_g7b_trace()
    trace["INTER_PHASE_GAP_IS_EXECUTION_BUDGET"] = True
    with pytest.raises(g7.G7PError, match="G7B_GAP_IS_BUDGET"):
        g7.validate_g7b_timestamp_trace(trace, inter_phase_gap_ms=1000)


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("TIME_SLICED_TOPOLOGY", "resident", "G7B_TOPOLOGY"),
        ("RESIDENT_TIME_SLICING", "PASS", "G7B_RESIDENT"),
    ],
)
def test_g7b_cold_swap_cannot_be_mislabeled_resident(
    key: str, value: object, error: str,
) -> None:
    trace = _valid_g7b_trace()
    trace[key] = value
    with pytest.raises(g7.G7PError, match=error):
        g7.validate_g7b_timestamp_trace(trace, inter_phase_gap_ms=1000)


def test_g7b_valid_trace_and_derived_durations_are_exactly_recomputable() -> None:
    trace = _valid_g7b_trace()
    derived = g7.validate_g7b_timestamp_trace(trace, inter_phase_gap_ms=1000)
    assert derived["measured_cycle_ms"] == [2000.0, 2000.0, 2000.0]
    assert derived["measured_cycles_total_ms"] == 6000.0
    assert derived["FULL_MEASURED_LEARNER_PHASE_MS"] == derived["learner_phase_total_ms"]
    assert derived["FULL_MEASURED_LEARNER_PHASE_MS"] > derived["measured_cycles_total_ms"]
    assert derived["resume_first_inference_service_ms"] == 2000.0
    assert derived["full_policy_unavailability_ms"] == (
        trace["resume_first_result_ready_ns"] - trace["episode_queue_drained_ns"]
    ) / 1e6
    assert derived["PRODUCTION_REQUIRED_RESET_HOME_WINDOW_MS"] == "UNVERIFIED"


def test_g7b_base_component_digest_drift_fails(measured_report: dict) -> None:
    base = deepcopy(measured_report)
    if "g7b_timestamp_instrumentation" in base:
        expected = base["g7b_timestamp_instrumentation"]["base_component_digests"]
    else:
        expected = g7.g7b_component_digests(base)
    candidate = deepcopy(base)
    candidate["modes"]["inference_only"]["status"] = "FAIL"
    with pytest.raises(g7.G7PError, match="G7B_BASE_COMPONENT_DRIFT:inference_only"):
        g7.verify_g7b_base_components(base, candidate, expected)


def test_g7b_targeted_entrypoint_calls_only_time_sliced_mode() -> None:
    source = (ROOT / "tools/benchmark_stage3_actor_learner_coexistence_gpu.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_targeted_time_sliced_rerun"
    )
    calls = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "run_time_sliced" in calls
    assert not calls.intersection({"run_inference_only", "run_learner_only", "run_concurrent", "run_benchmark"})
