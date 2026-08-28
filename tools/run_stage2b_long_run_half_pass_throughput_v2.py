#!/usr/bin/env python3
"""Coordinator for Candidate-B exact resume and C64/C96/C128 repeats."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).parents[1].resolve()
WORKER = ROOT / "tools/run_stage2b_long_run_half_pass_worker_throughput_v2.py"
CONFIG = ROOT / "configs/stage2b_long_run_half_pass_throughput_v2.development.yaml"
OUTPUT = ROOT / "artifacts/development/stage2/throughput_v2_long_run"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0 and np.isfinite(array).all(), "THROUGHPUT_V2_AGGREGATE_INVALID")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "range": float(array.max() - array.min()),
    }


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT / 'tools'}:{ROOT}",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })
    return value


def run_worker(
    *, result: Path, critic_batch: int, cycles: int, warmup: int = 0,
    start_cycle: int = 0, resume: Path | None = None,
    checkpoint: Path | None = None, trace: bool = False,
) -> dict:
    require(not result.exists(), f"THROUGHPUT_V2_APPEND_ONLY_RESULT_EXISTS:{result}")
    command = [
        sys.executable, str(WORKER), "--result", str(result),
        "--critic-batch", str(critic_batch), "--cycles", str(cycles),
        "--warmup-cycles", str(warmup), "--start-cycle", str(start_cycle),
    ]
    if resume is not None:
        command.extend(("--resume-checkpoint", str(resume)))
    if checkpoint is not None:
        command.extend(("--checkpoint-out", str(checkpoint)))
    if trace:
        command.append("--trace")
    completed = subprocess.run(command, cwd=ROOT, env=environment(), check=False)
    require(completed.returncode == 0 and result.is_file(), f"THROUGHPUT_V2_WORKER_FAILED:{result}:{completed.returncode}")
    return json.loads(result.read_text())


def exact_resume() -> dict:
    root = OUTPUT / "exact_resume"
    root.mkdir(parents=True, exist_ok=True)
    branch_a = run_worker(
        result=root / "branch_a_continuous_2cycles.json",
        critic_batch=128, cycles=2, trace=True,
    )
    recovery = root / "branch_b_cycle1_recovery"
    branch_b1 = run_worker(
        result=root / "branch_b_cycle1.json",
        critic_batch=128, cycles=1, checkpoint=recovery, trace=True,
    )
    branch_b2 = run_worker(
        result=root / "branch_b_resumed_cycle2.json",
        critic_batch=128, cycles=1, start_cycle=1, resume=recovery, trace=True,
    )
    a1 = branch_a["records"][0]
    a2 = branch_a["records"][1]
    b1 = branch_b1["records"][0]
    b2 = branch_b2["records"][0]
    comparisons = {
        "cycle1_training_state": (
            a1["cycle_training_state"]["training_state_digest"]
            == branch_b1["training_state"]["training_state_digest"]
        ),
        "cycle1_trace": a1["trace"]["digest"] == b1["trace"]["digest"],
        "cycle1_flow_noise_actions": (
            a1["captured_flow_noise_and_actions"] == b1["captured_flow_noise_and_actions"]
        ),
        "cycle2_training_state": (
            branch_a["training_state"]["training_state_digest"]
            == branch_b2["training_state"]["training_state_digest"]
        ),
        "cycle2_trace": a2["trace"]["digest"] == b2["trace"]["digest"],
        "cycle2_flow_noise_actions": (
            a2["captured_flow_noise_and_actions"] == b2["captured_flow_noise_and_actions"]
        ),
        "cycle2_row_membership": (
            [item["row_identity_audit"] for item in a2["critic_updates"]]
            == [item["row_identity_audit"] for item in b2["critic_updates"]]
            and a2["actor_batch_identity_sha256"] == b2["actor_batch_identity_sha256"]
        ),
        "cycle2_loss_tensors": (
            [item["loss"] for item in a2["critic_updates"]]
            == [item["loss"] for item in b2["critic_updates"]]
            and a2["actor_update"]["loss"] == b2["actor_update"]["loss"]
        ),
        "cycle2_post_step_parameters": a2["post_critic_step_states"] == b2["post_critic_step_states"],
        "checkpoint_save_rng_side_effect_free": bool(branch_b1["checkpoint"]["save_side_effect_free"]),
        "different_fresh_process_pid": len({branch_a["pid"], branch_b1["pid"], branch_b2["pid"]}) == 3,
    }
    result = {
        "schema_version": "forcesmolvla_stage2b_throughput_v2_exact_resume.v1",
        "status": "pass" if all(comparisons.values()) else "fail",
        "comparison": "bitwise_canonical_rtol0_atol0_equal_nan_false",
        "comparisons": comparisons,
        "branch_pids": {
            "a": branch_a["pid"], "b1": branch_b1["pid"], "b2": branch_b2["pid"],
        },
        "cycle1_digest": branch_b1["training_state"]["training_state_digest"],
        "cycle2_digest": branch_b2["training_state"]["training_state_digest"],
        "cache_hit_miss_excluded_from_training_state": True,
        "recovery_checkpoint": recovery.relative_to(ROOT).as_posix(),
        "recovery_checkpoint_role": "exact_resume_preflight_only_not_training_parent",
    }
    atomic_json(root / "canonical_parity_report.json", result)
    require(result["status"] == "pass", "THROUGHPUT_V2_EXACT_RESUME_FAILED")
    return result


def aggregate(critic_batch: int, repeats: list[dict]) -> dict:
    cycles = [
        float(record["cycle_seconds"])
        for repeat in repeats for record in repeat["records"] if not record["warmup"]
    ]
    actor_tps = [float(repeat["actor_transitions_per_second"]) for repeat in repeats]
    td_tps = [float(repeat["critic_td_transitions_per_second"]) for repeat in repeats]
    calql_tps = [float(repeat["critic_calql_transitions_per_second"]) for repeat in repeats]
    cycle_mean = statistics.mean(cycles)
    return {
        "critic_batch": critic_batch,
        "actor_batch": 24,
        "repeat_count": len(repeats),
        "fresh_process_count": len({repeat["pid"] for repeat in repeats}),
        "warmup_cycles_per_repeat": 1,
        "measured_cycles_per_repeat": 3,
        "cycle_seconds": describe(cycles),
        "actor_transitions_per_second": describe(actor_tps),
        "critic_td_transitions_per_second": describe(td_tps),
        "critic_calql_transitions_per_second": describe(calql_tps),
        "joint_cycles_per_hour_from_mean_cycle": 3600.0 / cycle_mean,
        "gpu_utilization_repeat_means": describe([
            float(repeat["gpu_utilization_percent"]["mean"]) for repeat in repeats
        ]),
        "gpu_utilization_repeat_p95": describe([
            float(repeat["gpu_utilization_percent"]["p95"]) for repeat in repeats
        ]),
        "gpu_power_repeat_means": describe([
            float(repeat["gpu_power_watts"]["mean"]) for repeat in repeats
        ]),
        "peak_allocated_bytes": max(repeat["peak_allocated_bytes"] for repeat in repeats),
        "peak_reserved_bytes": max(repeat["peak_reserved_bytes"] for repeat in repeats),
        "peak_cpu_rss_bytes": max(repeat["peak_cpu_rss_bytes"] for repeat in repeats),
        "cache_hit_rate": describe([float(repeat["cache"]["cache_hit_rate"]) for repeat in repeats]),
        "cache_evictions": describe([float(repeat["cache"]["cache_evictions"]) for repeat in repeats]),
        "cold_start_initialization_seconds": describe([
            float(repeat["cold_start_initialization_seconds"]) for repeat in repeats
        ]),
        "steady_state_data_time_seconds": describe([
            float(record["actor_data_loading_seconds"])
            + sum(float(item["timing"]["data_loading"]) for item in record["critic_updates"])
            for repeat in repeats for record in repeat["records"] if not record["warmup"]
        ]),
        "prefix_prefill_count": sum(repeat["prefix_prefill_count"] for repeat in repeats),
        "flow_call_count": sum(repeat["flow_call_count"] for repeat in repeats),
        "euler_velocity_evaluation_count": sum(
            repeat["euler_velocity_evaluation_count"] for repeat in repeats
        ),
        "td_row_membership_per_cycle": 2 * critic_batch,
        "calql_row_membership_per_cycle": 2 * critic_batch,
        "total_critic_row_membership_per_cycle": 4 * critic_batch,
        "all_losses_gradients_finite": all(repeat["all_losses_and_gradients_finite"] for repeat in repeats),
        "action_contract_v2": all(repeat["action_contract_v2"] for repeat in repeats),
        "frozen_parameter_hash_unchanged": all(repeat["frozen_parameter_hash_unchanged"] for repeat in repeats),
        "result_paths": [],
    }


def benchmark_batches() -> dict:
    root = OUTPUT / "formal_repeats"
    root.mkdir(parents=True, exist_ok=True)
    aggregates = []
    for critic_batch in (64, 96, 128):
        repeats = []
        for repeat in range(1, 4):
            path = root / f"candidate_b_actor24_critic{critic_batch}_repeat{repeat}.json"
            value = run_worker(
                result=path, critic_batch=critic_batch, cycles=4, warmup=1
            )
            repeats.append(value)
        value = aggregate(critic_batch, repeats)
        value["result_paths"] = [
            (root / f"candidate_b_actor24_critic{critic_batch}_repeat{repeat}.json")
            .relative_to(ROOT).as_posix()
            for repeat in range(1, 4)
        ]
        aggregates.append(value)
        atomic_json(root / f"aggregate_actor24_critic{critic_batch}.json", value)
    valid = [
        item for item in aggregates
        if item["all_losses_gradients_finite"]
        and item["action_contract_v2"]
        and item["frozen_parameter_hash_unchanged"]
        and item["peak_reserved_bytes"] <= int(24 * 1024**3 * 0.85)
    ]
    require(valid, "THROUGHPUT_V2_NO_VALID_CRITIC_BATCH")
    selected = min(valid, key=lambda item: item["cycle_seconds"]["mean"])
    cycles_per_hour = float(selected["joint_cycles_per_hour_from_mean_cycle"])
    result = {
        "schema_version": "forcesmolvla_stage2b_throughput_v2_formal_repeats.v1",
        "status": "pass",
        "selection_objective": "minimum_fixed_actor_pass_wall_clock_under_contract",
        "aggregates": aggregates,
        "candidate_b_formal_repeats": next(
            item for item in aggregates if item["critic_batch"] == 128
        ),
        "selected_critic_batch": selected["critic_batch"],
        "selected_actor_batch": 24,
        "selected_flow_inference_subbatch": 4,
        "selected_cycles_per_hour": cycles_per_hour,
        "projected_runtime_hours": {
            "0.5_actor_pass_210_cycles": 210.0 / cycles_per_hour,
            "1.0_actor_pass_420_cycles": 420.0 / cycles_per_hour,
            "2.0_actor_pass_840_cycles": 840.0 / cycles_per_hour,
        },
        "long_run_started": False,
    }
    atomic_json(root / "batch_selection.json", result)
    return result


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    require(config["authorization"] == "integration_preflight_only_no_long_run", "THROUGHPUT_V2_COORDINATOR_AUTH")
    cache_path = OUTPUT / "cache_210_cycle_preflight.json"
    require(cache_path.is_file(), "THROUGHPUT_V2_CACHE_PREFLIGHT_MISSING")
    cache = json.loads(cache_path.read_text())
    require(cache["status"] == "pass" and cache["cycles_simulated"] == 210, "THROUGHPUT_V2_CACHE_PREFLIGHT_INVALID")
    exact = exact_resume()
    benchmark = benchmark_batches()
    result = {
        "schema_version": "forcesmolvla_stage2b_throughput_v2_integration.v1",
        "status": "pass",
        "config_path": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG),
        "bounded_cache_preflight": cache,
        "exact_resume": exact,
        "formal_repeats": benchmark,
        "candidate_b_definition": {
            "status": "fastest_currently_measured_semantically_valid_candidate_before_formal_retest",
            "global_optimum_claimed": False,
        },
        "old_cycle105_checkpoint_allowed_as_parent": False,
        "long_run_recipe_proposed": True,
        "long_run_authorized": False,
        "long_run_started": False,
        "robot_execution_authorized": False,
    }
    atomic_json(OUTPUT / "integration_summary.json", result)
    print(json.dumps({
        "status": result["status"],
        "exact_resume": exact["status"],
        "selected_critic_batch": benchmark["selected_critic_batch"],
        "cycles_per_hour": benchmark["selected_cycles_per_hour"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
