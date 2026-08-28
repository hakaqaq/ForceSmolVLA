#!/usr/bin/env python3
"""Build compact append-only throughput-v2 evidence from measured candidates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[1].resolve()
SUMMARY = ROOT / "artifacts/development/stage2/throughput_v2/summary.v2.json"
TIMING = ROOT / "artifacts/development/stage2/stage2_throughput_v2_timing_audit.v2.json"
INTERRUPTED = ROOT / "artifacts/development/stage2/s2_stage2b_interrupted_pilot_replay_abandoned.v1.json"
SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v28_throughput_v2.json"
ARTIFACT = ROOT / "artifacts/development/stage2/s2_stage2_throughput_v2_preflight.v1.json"
REPORT = ROOT / "docs/stage2_throughput_v2_report.v1.md"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "file_size": path.stat().st_size,
    }


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def candidate_compact(item: dict[str, Any], equivalence: dict[str, Any], baseline_mean: float) -> dict[str, Any]:
    return {
        "candidate_id": item["candidate"]["id"],
        "status": item["status"],
        "configuration": item["candidate"],
        "cycle_seconds": item["seconds_per_cycle"],
        "actor_transitions_per_second": item["actor_transitions_per_second"],
        "critic_transitions_per_second": item["critic_transitions_per_second"],
        "joint_cycles_per_hour": item["joint_cycles_per_hour"],
        "speedup_vs_baseline_mean": baseline_mean / item["seconds_per_cycle"]["mean"],
        "gpu_utilization_percent": item["gpu_utilization_percent"],
        "gpu_power_watts": item["gpu_power_watts"],
        "peak_allocated_bytes": item["peak_allocated_bytes"],
        "peak_reserved_bytes": item["peak_reserved_bytes"],
        "pipeline": item["pipeline"],
        "warmup_trace": {
            "path": item["warmup_trace_path"],
            "sha256": item["warmup_trace_sha256"],
        },
        "equivalence": equivalence,
        "frozen_parameter_hash_unchanged": item["frozen_parameter_hash_unchanged"],
        "all_losses_and_gradients_finite": item["all_losses_and_gradients_finite"],
        "action_contract_v2": item["action_contract_v2"],
        "internal_path_call_audit": item["internal_path_call_audit"],
        "public_inference": item["public_inference"],
        "parameter_updates": item["parameter_updates"],
        "training_checkpoint_created": item["training_checkpoint_created"],
        "access_audit": item["access_audit"],
    }


def main() -> None:
    require(not ARTIFACT.exists() and not REPORT.exists(), "THROUGHPUT_V2_ARTIFACT_APPEND_ONLY")
    summary = json.loads(SUMMARY.read_text())
    timing = json.loads(TIMING.read_text())
    interrupted = json.loads(INTERRUPTED.read_text())
    source = json.loads(SOURCE.read_text())
    require(summary["status"] == "pass", "THROUGHPUT_V2_SUMMARY_FAILED")
    require(timing["status"] == "pass", "THROUGHPUT_V2_TIMING_AUDIT_FAILED")
    require(
        interrupted["status"]
        == "valid_interrupted_long_run_pilot_without_cycle136_checkpoint",
        "THROUGHPUT_V2_INTERRUPTED_STATUS",
    )
    for entry in source["files"]:
        require(sha256_file(ROOT / entry["relative_path"]) == entry["sha256"], f"THROUGHPUT_V2_SOURCE_DRIFT:{entry['relative_path']}")
    candidates = {item["candidate"]["id"]: item for item in summary["candidates"]}
    selected_id = summary["recommended_candidate"]
    selected = candidates[selected_id]
    baseline = candidates["baseline_current_implementation"]
    cycle_seconds = float(selected["seconds_per_cycle"]["mean"])
    cycles = {"0.5_actor_pass": 210, "1.0_actor_pass": 420, "2.0_actor_passes": 840}
    projections = {
        name: {
            "joint_cycles": count,
            "runtime_hours_from_mean_steady_state": count * cycle_seconds / 3600.0,
            "actor_transition_exposure": count * 24,
            "critic_transition_exposure": count * 2 * 128,
        }
        for name, count in cycles.items()
    }
    compact = [
        candidate_compact(
            item,
            summary["equivalence"][item["candidate"]["id"]],
            float(baseline["seconds_per_cycle"]["mean"]),
        )
        for item in summary["candidates"]
        if item["status"] == "pass"
    ]
    artifact = {
        "schema_version": "forcesmolvla_stage2_throughput_v2_preflight.v1",
        "status": "pass",
        "authorization": "benchmark_only_no_training_checkpoint",
        "interrupted_pilot": binding(INTERRUPTED),
        "timing_audit": binding(TIMING),
        "raw_summary": binding(SUMMARY),
        "source_manifest": binding(SOURCE),
        "source_files_sha256": source["files_sha256"],
        "fixed_training_semantics": {
            "actor_batch": 24, "critic_batch": 128,
            "critic_to_actor": "2:1", "horizon": 50, "flow_steps": 10,
            "calql_candidates_per_source": 2, "eta": 3.0, "beta": 1.0,
            "action_contract": "v2", "trainability": "frozen_vlm_force_action_trainable",
        },
        "candidates": compact,
        "recommended_candidate": selected_id,
        "recommended_cycle_seconds_mean": cycle_seconds,
        "recommended_joint_cycles_per_hour": float(selected["joint_cycles_per_hour"]),
        "recommended_actor_transitions_per_second": float(selected["actor_transitions_per_second"]),
        "recommended_critic_transitions_per_second": float(selected["critic_transitions_per_second"]),
        "measured_speedup_vs_current_baseline": float(baseline["seconds_per_cycle"]["mean"]) / cycle_seconds,
        "projected_actor_pass_runtime": projections,
        "projection_basis": "mean_measured_steady_state_cycle_throughput",
        "cold_cache_initialization_reported_separately": True,
        "measurement_scope": "user_requested_fast_screen_1_warmup_plus_1_measured_cycle",
        "single_measured_cycle_has_no_statistical_p95_claim": True,
        "current_run_status": "valid_interrupted_long_run_pilot",
        "current_checkpoint_status": "audit_only_cycle105_latest; no_cycle136_checkpoint",
        "auto_resume": False,
        "throughput_v2_authorized": True,
        "throughput_v2_long_run": False,
        "throughput_v2_training_checkpoint": False,
        "restart_0_5_pass_authorized": False,
        "auto_extend_to_1_0_pass": False,
        "long_run_extension_authorized": False,
        "robot_execution_authorized": False,
    }
    atomic_text(ARTIFACT, json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = [
        "# Stage-2 throughput-v2 benchmark",
        "",
        "本轮仅执行临时 benchmark updates。所有候选从同一 G7-A-r2 父状态、相同样本顺序和 RNG 启动；候选参数均丢弃，没有训练 checkpoint，也没有恢复或启动 long-run。",
        "",
        "## 结果",
        "",
        "| Candidate | mean cycle (s) | Actor trans/s | Critic trans/s | cycles/h | speedup | peak reserved (GiB) | equivalence |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in compact:
        lines.append(
            f"| {item['candidate_id']} | {item['cycle_seconds']['mean']:.3f} | "
            f"{item['actor_transitions_per_second']:.4f} | {item['critic_transitions_per_second']:.4f} | "
            f"{item['joint_cycles_per_hour']:.3f} | {item['speedup_vs_baseline_mean']:.3f}× | "
            f"{item['peak_reserved_bytes'] / 2**30:.2f} | {item['equivalence']['classification']} |"
        )
    lines.extend([
        "",
        f"推荐：`{selected_id}`。选择依据是通过数值等价、finite、ActionContract-v2、冻结 hash 和公共接口检查后的最高 mean steady-state throughput，不是最大显存占用。",
        "",
        "## Actor transition-pass 预算（mean steady-state）",
        "",
        "| Budget | cycles | projected hours | Actor exposure | Critic exposure |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, value in projections.items():
        lines.append(
            f"| {name} | {value['joint_cycles']} | {value['runtime_hours_from_mean_steady_state']:.2f} | "
            f"{value['actor_transition_exposure']} | {value['critic_transition_exposure']} |"
        )
    lines.extend([
        "",
        "启动/冷 cache 成本单独报告，未计入 steady-state cycle throughput。按用户要求，本轮每候选仅运行 1 warm-up + 1 measured cycle，因此用于快速筛选，不对 P95 或跨重复波动作统计性主张。",
        "",
        "Action 等价性在真实 ActionContract-v2 Critic K×7 域比较：TCP6 使用预先声明的 bf16 容差，binary gripper endpoint 必须 exact。未投影 H×7 raw Flow 只作为诊断，不作为 Critic 输入等价性的错误门槛。B8/B16/E 因 TCP 超差或 gripper endpoint 翻转被拒绝，即使吞吐更高也不推荐。",
        "",
        "```text",
        "CURRENT_RUN_STATUS = valid_interrupted_long_run_pilot",
        "CURRENT_CHECKPOINT_STATUS = audit_only_cycle105_latest; no_cycle136_checkpoint",
        "AUTO_RESUME = no",
        "THROUGHPUT_V2_AUTHORIZED = yes",
        "THROUGHPUT_V2_LONG_RUN = no",
        "THROUGHPUT_V2_TRAINING_CHECKPOINT = no",
        "RESTART_0_5_PASS_AUTHORIZED = no",
        "AUTO_EXTEND_TO_1_0_PASS = no",
        "LONG_RUN_EXTENSION_AUTHORIZED = no",
        "ROBOT_EXECUTION_AUTHORIZED = false",
        "```",
    ])
    atomic_text(REPORT, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
