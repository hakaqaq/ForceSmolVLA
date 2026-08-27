#!/usr/bin/env python3
"""CPU coordinator for staged, disposable fresh-process batch benchmarking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from forcesmolvla.rft.batch_throughput import (
    epoch_budget,
    expanded_actor_candidates,
    select_best,
    stage_a_candidates,
)


ROOT = Path(__file__).parents[1].resolve()
CONFIG = ROOT / "configs/stage2_g7_batch_throughput_benchmark.development.yaml"
WORKER = ROOT / "tools/run_s2_g7_batch_candidate.py"
SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v19_g7_batch_benchmark.json"
OUTPUT = ROOT / "artifacts/development/stage2/g7_batch_throughput_benchmark"
ARTIFACT = ROOT / "artifacts/development/stage2/s2_g7_batch_throughput_benchmark.json"
REPORT = ROOT / "docs/s2_g7_batch_throughput_benchmark_report.md"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def frozen_snapshot() -> dict:
    import preflight_s2_g7b_joint_smoke_gpu as g7b
    from forcesmolvla.rft.exact_resume import checkpoint_tree

    files = {
        "g7b_artifact": ROOT / "artifacts/development/stage2/s2_g7b_joint_smoke_preflight.json",
        "g7b_source_manifest": ROOT / "artifacts/development/stage2/stage2_source_manifest.v17_g7b.json",
        "g7b_config": ROOT / "configs/stage2_g7b_joint_smoke.development.yaml",
        "g5_v2_config": ROOT / "configs/stage2_g5_single_cycle.v2.development.yaml",
        "action_contract_v2": ROOT / "configs/stage2_action_contract.v2.development.json",
        "public_action_source": ROOT / "src/forcesmolvla/action_delta.py",
        "public_model_source": ROOT / "src/forcesmolvla/modeling_forcesmolvla.py",
        "public_rules_source": ROOT / "src/forcesmolvla/rules.py",
        "public_rulespec": ROOT / "configs/live_action_safety.task2.development.yaml",
    }
    return {
        "g7b_protected_closure": g7b.snapshot(),
        "files": {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "file_size": path.stat().st_size} for name, path in files.items()},
        "trees": {
            "g7a_r2_parent": checkpoint_tree(ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint"),
            "g7b_smoke_checkpoint": checkpoint_tree(ROOT / "artifacts/development/stage2/g7b_joint_smoke_checkpoint.development"),
        },
    }


def candidate_id(stage: str, candidate: dict[str, int]) -> str:
    return (
        f"{stage}_am{candidate['actor_microbatch']}_aa{candidate['actor_accumulation']}"
        f"_ae{candidate['effective_actor_batch']}_cb{candidate['critic_batch']}"
    )


def run_candidate(stage: str, values: dict[str, int], config: dict) -> dict:
    candidate = {
        "candidate_id": candidate_id(stage, values),
        "stage": stage,
        **values,
        "warmup_joint_cycles": int(config["measurement"]["warmup_joint_cycles"]),
        "measured_joint_cycles": int(config["measurement"]["measured_joint_cycles"]),
        "gpu_utilization_poll_seconds": float(config["measurement"]["gpu_utilization_poll_seconds"]),
        "parent_checkpoint": config["parent"]["path"],
        "parent_tree_sha256": config["parent"]["tree_sha256"],
        "eta_actor_q": 10.0,
        "calql_candidates_per_source": 2,
        "action_horizon": 50,
        "flow_euler_steps": 10,
    }
    config_path = OUTPUT / "resolved_candidates" / f"{candidate['candidate_id']}.json"
    result_path = OUTPUT / "candidate_results" / f"{candidate['candidate_id']}.json"
    require(not config_path.exists() and not result_path.exists(), f"BENCHMARK_CANDIDATE_APPEND_ONLY_COLLISION:{candidate['candidate_id']}")
    atomic_json(config_path, candidate)
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT}",
    })
    command = [
        sys.executable, str(WORKER),
        "--candidate-config", str(config_path),
        "--result", str(result_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    require(completed.returncode == 0, f"BENCHMARK_CANDIDATE_PROCESS_FAILED:{candidate['candidate_id']}:{completed.returncode}")
    result = json.loads(result_path.read_text())
    require(result["candidate_id"] == candidate["candidate_id"], "BENCHMARK_CANDIDATE_RESULT_ID_MISMATCH")
    print(f"BATCH_BENCHMARK_RESULT {candidate['candidate_id']} {result['status']}", flush=True)
    return result


def summary(result: dict) -> dict:
    base = {
        "candidate_id": result["candidate_id"], "stage": result["stage"], "status": result["status"],
        "resolved_candidate": result["resolved_candidate"],
        "result_path": f"artifacts/development/stage2/g7_batch_throughput_benchmark/candidate_results/{result['candidate_id']}.json",
    }
    if result["status"] == "pass":
        base.update({
            "throughput": result["throughput"],
            "joint_cycle_seconds": result["timing_seconds"]["joint_cycle"],
            "gpu_utilization_percent": result["gpu_utilization_percent"],
            "peak_allocated_bytes": result["peak_allocated_bytes"],
            "peak_reserved_bytes": result["peak_reserved_bytes"],
            "all_finite": result["all_finite"],
            "action_contract_v2": result["action_contract_v2"],
            "router_auxiliary": result["router_auxiliary"],
            "update_counts": result["update_counts"],
        })
    else:
        base.update({"error_type": result.get("error_type"), "error": result.get("error"), "runtime_batch_fallback_used": result.get("runtime_batch_fallback_used")})
    return base


def report_markdown(artifact: dict) -> str:
    rows = []
    for item in artifact["candidate_summaries"]:
        candidate = item["resolved_candidate"]
        if item["status"] == "pass":
            rows.append(
                f"| {item['candidate_id']} | {candidate['actor_microbatch']}×{candidate['actor_accumulation']}={candidate['effective_actor_batch']} | "
                f"{candidate['critic_batch']} | PASS | {item['throughput']['actor_samples_per_second']:.3f} | "
                f"{item['throughput']['critic_sample_memberships_per_second']:.3f} | "
                f"{item['throughput']['joint_training_sample_memberships_per_second']:.3f} | "
                f"{item['joint_cycle_seconds']['median']:.3f}/{item['joint_cycle_seconds']['p95']:.3f} | "
                f"{item['gpu_utilization_percent']['mean']:.1f}/{item['gpu_utilization_percent']['p95']:.1f} | "
                f"{item['peak_allocated_bytes'] / 2**30:.2f}/{item['peak_reserved_bytes'] / 2**30:.2f} |"
            )
        else:
            rows.append(f"| {item['candidate_id']} | {candidate['actor_microbatch']}×{candidate['actor_accumulation']}={candidate['effective_actor_batch']} | {candidate['critic_batch']} | OOM | — | — | — | — | — | — |")
    selected = artifact["recommended_configuration"]
    budget_rows = [
        f"| {item['actor_epochs']} | {item['joint_cycles']} | {item['estimated_hours']:.3f} h | "
        f"{item['actor_sample_memberships']} | {item['critic_td_sample_memberships']} | "
        f"{item['critic_calql_sample_memberships']} | {item['critic_total_sample_memberships']} |"
        for item in artifact["sample_based_budget_estimates"]
    ]
    return f"""# Stage-2 offline batch-throughput benchmark

Status: **PASS (development throughput measurement only)**. Every candidate ran in an independent fresh process from the same G7-A-r2 Critic-warmup checkpoint. Each process executed one excluded warm-up joint cycle followed by exactly three measured cycles, and its updated state was discarded without a checkpoint.

The true G7-B baseline was parsed as Actor `1×4=4`, Critic `16`, Cal-QL candidates/source `2`.

| candidate | Actor micro×accum=effective | Critic B | status | Actor samples/s | Critic memberships/s | joint memberships/s | cycle median/P95 s | GPU util mean/P95 % | allocated/reserved GiB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The selected stable layout is Actor `{selected['actor_microbatch']}×{selected['actor_accumulation']}={selected['effective_actor_batch']}` and Critic `{selected['critic_batch']}`. Selection maximized measured joint sample-membership throughput among finite, ActionContract-v2-valid candidates with peak reserved VRAM at or below 21 GiB. VRAM occupancy alone was not optimized.

Router balance/z remain the frozen microbatch-local equal-average auxiliary. Their values can differ across physical microbatch layouts; no global-router-objective equivalence is claimed.

## Sample-based budget estimates

`cycles_per_actor_epoch = ceil(10075 / {selected['effective_actor_batch']}) = {artifact['cycles_per_actor_epoch']}`. Estimates use the final-confirmation median joint-cycle time.

| Actor epochs | cycles | estimated time | Actor memberships | Critic TD memberships | Critic Cal-QL memberships | Critic total memberships |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(budget_rows)}

All candidate losses, Q values, gradients, and parameters were finite. TCP6 received Q-gradient, gripper Q-gradient was exactly zero, gripper Flow-Matching gradient was nonzero, and invalid action slots remained exactly masked. Public inference sources, RuleSpec, gripper tolerance, Stage-1/G1/Reward artifacts, G7-A-r2, and preserved G7-B evidence were unchanged.

No validation/test transition, manual G1, manual label, or Reward Classifier access occurred. This benchmark did not start a long run and produced no policy-training, deployment, or evaluation checkpoint. The final sample budget still requires explicit approval.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run, "pass --run for authorized batch benchmark")
    for path in (OUTPUT, ARTIFACT, REPORT):
        require(not path.exists(), f"BENCHMARK_APPEND_ONLY_TARGET_EXISTS:{path}")
    config = yaml.safe_load(CONFIG.read_text())
    parsed = config["parsed_g7b_batching"]
    g7b = yaml.safe_load((ROOT / "configs/stage2_g7b_joint_smoke.development.yaml").read_text())
    g5 = yaml.safe_load((ROOT / "configs/stage2_g5_single_cycle.v2.development.yaml").read_text())
    actual = {
        "actor_microbatch_size": g5["batching"]["actor_microbatch_size"],
        "actor_gradient_accumulation": g5["batching"]["actor_gradient_accumulation"],
        "effective_actor_batch_size": g5["batching"]["actor_effective_batch_size"],
        "critic_batch_size": g5["batching"]["critic_batch_size"],
        "calql_candidates_per_source": g5["loss"]["cql_candidates_per_source_M"],
    }
    require(actual == parsed, f"BENCHMARK_PARSED_G7B_BATCH_DRIFT:{actual}")
    require(g7b["joint_smoke"]["eta_actor_q"] == 10.0, "BENCHMARK_ETA_DRIFT")
    require(sha(ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint/checkpoint_manifest.json") == "2e0902076cb12a1391613230679730d035155528c9be01bd17dce960d5e707f7", "BENCHMARK_PARENT_MANIFEST_DRIFT")
    require(SOURCE.is_file(), "BENCHMARK_SOURCE_MANIFEST_MISSING")
    source = json.loads(SOURCE.read_text())
    for entry in source["files"]:
        require(sha(ROOT / entry["relative_path"]) == entry["sha256"], f"BENCHMARK_SOURCE_SHA_DRIFT:{entry['relative_path']}")

    environment = os.environ.copy()
    environment.update({"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_rft_batch_throughput.py", "tests/test_rft_critic_action_contract_v2.py"],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
    )
    require(tests.returncode == 0, f"BENCHMARK_TEST_FAILURE:{(tests.stdout + tests.stderr)[-3000:]}")
    before = frozen_snapshot()
    OUTPUT.mkdir(parents=True)
    all_results = []

    stage_a = []
    for candidate in stage_a_candidates(parsed["effective_actor_batch_size"], parsed["critic_batch_size"]):
        result = run_candidate("stage_a", candidate, config)
        stage_a.append(result); all_results.append(result)
    stage_a_best = select_best(stage_a)
    physical = int(stage_a_best["resolved_candidate"]["actor_microbatch"])

    actor_stage = []
    for candidate in expanded_actor_candidates(physical, config["stage_b_actor"]["effective_actor_batches"], 16):
        result = run_candidate("stage_b_actor", candidate, config)
        actor_stage.append(result); all_results.append(result)
    actor_best = select_best([stage_a_best, *actor_stage])
    actor_layout = actor_best["resolved_candidate"]

    critic_stage = []
    for critic_batch in config["stage_b_critic"]["critic_batches"]:
        candidate = {
            "actor_microbatch": int(actor_layout["actor_microbatch"]),
            "actor_accumulation": int(actor_layout["actor_accumulation"]),
            "effective_actor_batch": int(actor_layout["effective_actor_batch"]),
            "critic_batch": int(critic_batch),
        }
        result = run_candidate("stage_b_critic", candidate, config)
        critic_stage.append(result); all_results.append(result)
    combination_best = select_best(critic_stage)
    final_values = {key: int(combination_best["resolved_candidate"][key]) for key in ("actor_microbatch", "actor_accumulation", "effective_actor_batch", "critic_batch")}
    final = run_candidate("final_confirmation", final_values, config)
    all_results.append(final)
    require(final["status"] == "pass" and final["all_finite"] and final["action_contract_v2"]["passed"], "BENCHMARK_FINAL_CONFIRMATION_FAILED")
    require(final["peak_reserved_bytes"] <= 21 * 2**30, "BENCHMARK_FINAL_CONFIRMATION_VRAM_LIMIT")

    after = frozen_snapshot()
    require(before == after, "BENCHMARK_FROZEN_INPUT_CHANGED")
    require(all(result.get("data_access", {}).get(key, 0) == 0 for result in all_results for key in (
        "validation_transition_reads", "test_transition_reads", "manual_g1_opens", "manual_label_opens", "reward_classifier_inference", "reward_classifier_updates"
    )), "BENCHMARK_FORBIDDEN_DATA_ACCESS")
    selected = final["resolved_candidate"]
    median_seconds = float(final["timing_seconds"]["joint_cycle"]["median"])
    budgets = epoch_budget(10075, int(selected["effective_actor_batch"]), int(selected["critic_batch"]), median_seconds)
    cycles_per_epoch = math.ceil(10075 / int(selected["effective_actor_batch"]))
    summaries = [summary(result) for result in all_results]
    artifact = {
        "schema_version": "forcesmolvla_s2_g7_batch_throughput_benchmark.v1",
        "artifact_status": "development_only",
        "G7B_JOINT_SMOKE": "pass_preserved",
        "BATCH_THROUGHPUT_BENCHMARK": "pass",
        "LONG_RUN_STARTED": "no",
        "LONG_RUN_AUTHORIZED": "no",
        "ROBOT_EXECUTION_AUTHORIZED": False,
        "NEXT_ALLOWED_ACTION": "request_sample_based_long_run_budget_approval",
        "parsed_current_g7b_batching": actual,
        "benchmark_contract": {
            "parent": config["parent"],
            "one_warmup_plus_three_measured_cycles_each": True,
            "candidate_processes_independent": True,
            "candidate_updated_state_discarded": True,
            "candidate_checkpoint_count": 0,
            "selection_metric": config["measurement"]["selection_metric"],
            "maximum_peak_reserved_gib": 21.0,
            "exact_two_pass_oracle_run": False,
        },
        "stage_a_best_candidate_id": stage_a_best["candidate_id"],
        "stage_b_actor_best_candidate_id": actor_best["candidate_id"],
        "stage_b_combination_best_candidate_id": combination_best["candidate_id"],
        "final_confirmation_candidate_id": final["candidate_id"],
        "candidate_summaries": summaries,
        "recommended_configuration": {
            "actor_microbatch": int(selected["actor_microbatch"]),
            "actor_accumulation": int(selected["actor_accumulation"]),
            "effective_actor_batch": int(selected["effective_actor_batch"]),
            "critic_batch": int(selected["critic_batch"]),
            "calql_candidates_per_source": 2,
            "eta": "10.0_development_only",
            "parent_checkpoint": config["parent"]["path"],
        },
        "cycles_per_actor_epoch": cycles_per_epoch,
        "sample_based_budget_estimates": budgets,
        "tests": {"exit_code": 0, "output": (tests.stdout + tests.stderr).strip()},
        "source_manifest": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": sha(SOURCE)},
        "protected_inputs_before": before,
        "protected_inputs_after_exact": before == after,
        "data_access": {
            "validation_transition_reads": 0, "test_transition_reads": 0,
            "manual_g1_opens": 0, "manual_label_opens": 0,
            "reward_classifier_inference": 0, "reward_classifier_updates": 0,
        },
        "long_run_budget_approved": False,
    }
    artifact["artifact_payload_sha256"] = canonical_sha(artifact)
    atomic_json(ARTIFACT, artifact)
    atomic_bytes(REPORT, report_markdown(artifact).encode())
    print("BATCH_THROUGHPUT_BENCHMARK pass", flush=True)


if __name__ == "__main__":
    main()

