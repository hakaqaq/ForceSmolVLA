#!/usr/bin/env python3
"""Coordinate the authorized 210-cycle Frozen-VLM Stage-2B run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).parents[1].resolve()
WORKER = ROOT / "tools/run_stage2b_long_run_half_pass_worker.py"
SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v21_stage2b_long_run_half_pass.json"
OUTPUT = ROOT / "artifacts/development/stage2/stage2b_long_run_half_pass"
CHECKPOINTS = ROOT / "artifacts/development/stage2/stage2b_long_run_half_pass_checkpoints"
ARTIFACT = ROOT / "artifacts/development/stage2/s2_stage2b_long_run_half_pass.json"
REPORT = ROOT / "docs/stage2b_long_run_half_pass_report.md"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def snapshot() -> dict:
    import preflight_s2_g7b_joint_smoke_gpu as g7b

    base = g7b.snapshot()
    files = {
        "stage2b_config": ROOT / "configs/stage2b_long_run_half_pass.development.yaml",
        "stage2b_source_manifest": SOURCE,
        "trainability_contract": ROOT / "configs/stage2_trainability_contract.v2.development.json",
        "action_contract_v2": ROOT / "configs/stage2_action_contract.v2.development.json",
        "batch_scaling_report": ROOT / "artifacts/development/stage2/batch_scaling_report.json",
        "batch_scaling_source_manifest": ROOT / "artifacts/development/stage2/stage2_source_manifest.v20_trainability_batch_scaling.json",
    }
    return {
        "g7b_protected": base,
        "files": {
            name: {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "sha256": sha(path), "file_size": path.stat().st_size,
            }
            for name, path in files.items()
        },
    }


def run_worker(*, mode: str, result: Path, start: int = 0, end: int = 0, protected: Path | None = None) -> subprocess.CompletedProcess:
    command = [sys.executable, str(WORKER), "--mode", mode, "--result", str(result)]
    if mode == "segment":
        command.extend(("--start-cycle", str(start), "--end-cycle", str(end), "--protected", str(protected)))
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT}",
    })
    return subprocess.run(command, cwd=ROOT, env=environment, check=False)


def stats(values: list[float]) -> dict:
    return {
        "count": len(values), "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": sorted(values)[min(len(values) - 1, int(0.95 * len(values)))],
        "minimum": min(values), "maximum": max(values),
    }


def report_markdown(artifact: dict) -> str:
    train = artifact["training"]
    cycles = train["cycles"]
    windows = []
    for start in range(0, 210, 35):
        local = cycles[start:start + 35]
        critic = [item["loss"]["L_critic"] for cycle in local for item in cycle["critic_updates"]]
        fm = [cycle["actor_update"]["loss"]["flow_matching"] for cycle in local]
        q = [cycle["actor_update"]["loss"]["actor_q_min_twin"] for cycle in local]
        total = [cycle["actor_update"]["loss"]["weighted_total"] for cycle in local]
        windows.append(
            f"| {start + 1}–{start + len(local)} | {statistics.fmean(critic):.6g} | "
            f"{statistics.fmean(fm):.6g} | {statistics.fmean(q):.6g} | "
            f"{statistics.fmean(total):.6g} |"
        )
    validation_rows = []
    for item in train["validation_diagnostics"]:
        validation_rows.append(
            f"| {item['cycle']} | {item['td_mse']['q1']:.6g}/{item['td_mse']['q2']:.6g} | "
            f"{item['calql_conservative']['q1']:.6g}/{item['calql_conservative']['q2']:.6g} | "
            f"{item['q_vs_mc_return']['mae']:.6g} | {item['q_vs_mc_return']['rmse']:.6g} | "
            f"{item['q_vs_mc_return']['spearman']:.6g} |"
        )
    gradient_rows = []
    for item in train["gradient_scale_diagnostics"]:
        global_metric = item["metrics"]["global"]
        gradient_rows.append(
            f"| {item['cycle']} | {global_metric['raw_q_over_fm']:.6g} | "
            f"{global_metric['weighted_eta3_q_over_beta1_fm']:.6g} | "
            f"{global_metric['cosine_similarity']:.6g} | {item['tcp6_q_gradient_norm']:.6g} | "
            f"{item['gripper_q_gradient_max_abs']:.1f} | {item['gripper_fm_gradient_norm']:.6g} |"
        )
    final = artifact["final_checkpoint"]
    return f"""# Stage-2B Frozen-VLM 0.5 Actor-pass long-run

Status: **{artifact['LONG_RUN_COMPLETED'].upper()}**. The run started from G7-A-r2, never loaded G7-B, and stopped exactly after 210 joint cycles.

## Training curve

| cycles | mean Critic loss | mean FM loss | mean Actor-Q loss | mean total Actor loss |
|---:|---:|---:|---:|---:|
{chr(10).join(windows)}

The JSON artifact and `progress.jsonl` retain all 210 per-cycle records.

## Read-only validation

| cycle | TD MSE Q1/Q2 | Cal-QL Q1/Q2 | Q-vs-return MAE | RMSE | Spearman |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(validation_rows)}

Validation ran only at cycles 0/105/210 and did not select a checkpoint or alter training state.

## Frozen-VLM gradient scale

| cycle | raw Q/FM | weighted eta=3 | cosine | TCP6 Q grad | gripper Q grad | gripper FM grad |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(gradient_rows)}

Every training cycle preserved nonzero TCP6 Q-gradient, exact-zero gripper Q-gradient, nonzero gripper FM-gradient, detached prefix, one Force K/V projection, and unchanged frozen hashes.

## Runtime and checkpoint

- Training-body runtime: {train['runtime']['training_body_seconds'] / 3600:.3f} h.
- Mean throughput: {train['runtime']['cycles_per_hour']:.3f} cycles/hour.
- Peak allocated/reserved: {train['runtime']['peak_allocated_bytes'] / 2**30:.2f}/{train['runtime']['peak_reserved_bytes'] / 2**30:.2f} GiB.
- Final checkpoint: `{final['relative_path']}`.
- Final checkpoint tree SHA-256: `{final['tree_sha256']}`.
- Cycle-105 fresh-process resume and cycle-210 strict-load audits: pass.

## Limits

No test data, manual G1, manual labels, or Reward Classifier was accessed. This checkpoint is for recovery and offline evaluation only. It is not authorized for deployment or robot execution. Additional training beyond 0.5 Actor pass remains unauthorized.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run, "pass --run for the authorized 210-cycle Stage-2B run")
    for path in (OUTPUT, CHECKPOINTS, ARTIFACT, REPORT):
        require(not path.exists(), f"STAGE2B_APPEND_ONLY_TARGET_EXISTS:{path}")
    from forcesmolvla.rft.source_manifest import validate_stage2_source_manifest

    validate_stage2_source_manifest(ROOT, SOURCE)
    environment = os.environ.copy()
    environment.update({"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"})
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_stage2b_long_run_half_pass.py"],
        cwd=ROOT, env=environment, capture_output=True, text=True,
    )
    require(tests.returncode == 0, f"STAGE2B_TEST_FAILURE:{tests.stdout}{tests.stderr}")
    before = snapshot()
    work = Path(tempfile.mkdtemp(prefix="stage2b-half-pass-", dir="/tmp"))
    protected = work / "protected.json"
    protected.write_text(json.dumps(before, indent=2, sort_keys=True) + "\n")

    segments = []
    for start, end in ((0, 105), (105, 210)):
        result = work / f"segment_{start}_{end}.json"
        completed = run_worker(
            mode="segment", result=result, start=start, end=end, protected=protected
        )
        if completed.returncode != 0:
            atomic_json(ARTIFACT, {
                "schema_version": "forcesmolvla_stage2b_long_run_half_pass.v1",
                "status": "stopped", "failure_segment": [start, end],
                "worker_returncode": completed.returncode,
                "LONG_RUN_STARTED": "yes", "LONG_RUN_COMPLETED": "stopped",
                "ADDITIONAL_LONG_RUN_AUTHORIZED": "no",
                "DEPLOYMENT_CHECKPOINT_AUTHORIZED": "no",
                "ROBOT_EXECUTION_AUTHORIZED": False,
            })
            raise RuntimeError(f"STAGE2B_SEGMENT_FAILED:{start}:{end}:{completed.returncode}")
        segments.append(json.loads(result.read_text()))

    verify_path = work / "strict_load.json"
    verify_completed = run_worker(mode="verify", result=verify_path)
    require(verify_completed.returncode == 0, f"STAGE2B_STRICT_LOAD_FAILED:{verify_completed.returncode}")
    verify = json.loads(verify_path.read_text())
    after = snapshot()
    require(before == after, "STAGE2B_PROTECTED_INPUT_CHANGED")
    progress = [json.loads(line) for line in (OUTPUT / "progress.jsonl").read_text().splitlines()]
    require([item["cycle"] for item in progress] == list(range(211)), "STAGE2B_PROGRESS_SEQUENCE")
    cycles = segments[0]["cycles"] + segments[1]["cycles"]
    require(len(cycles) == 210 and [item["cycle"] for item in cycles] == list(range(1, 211)), "STAGE2B_CYCLE_SEQUENCE")
    boundary = segments[0]["boundary_audits"] + segments[1]["boundary_audits"]
    gradients = segments[0]["gradient_scale_diagnostics"] + segments[1]["gradient_scale_diagnostics"]
    validations = segments[0]["validation_diagnostics"] + segments[1]["validation_diagnostics"]
    require([item["cycle"] for item in boundary] == [0, 105, 210], "STAGE2B_BOUNDARY_SEQUENCE")
    require([item["cycle"] for item in gradients] == [0, 105, 210], "STAGE2B_GRADIENT_SEQUENCE")
    require([item["cycle"] for item in validations] == [0, 105, 210], "STAGE2B_VALIDATION_SEQUENCE")
    training_seconds = sum(item["runtime"]["training_body_seconds"] for item in segments)
    peak_allocated = max(item["runtime"]["peak_allocated_bytes"] for item in segments)
    peak_reserved = max(item["runtime"]["peak_reserved_bytes"] for item in segments)
    from forcesmolvla.rft.exact_resume import checkpoint_tree

    checkpoint_bindings = {
        path.name: checkpoint_tree(path)
        for path in sorted(CHECKPOINTS.iterdir()) if path.is_dir()
    }
    final_path = CHECKPOINTS / "milestone_cycle_000210"
    final_tree = checkpoint_bindings[final_path.name]
    train = {
        "cycles": cycles, "boundary_audits": boundary,
        "gradient_scale_diagnostics": gradients,
        "validation_diagnostics": validations,
        "checkpoint_events": segments[0]["checkpoint_events"] + segments[1]["checkpoint_events"],
        "runtime": {
            "training_body_seconds": training_seconds,
            "cycles_per_hour": 210 / training_seconds * 3600,
            "actor_transitions_per_second": 5040 / training_seconds,
            "critic_transitions_per_second": 53760 / training_seconds,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "segments": [item["runtime"] for item in segments],
        },
        "data_access": {
            "train_actor_transition_exposure": 5040,
            "train_critic_td_transition_exposure": 53760,
            "train_calql_transition_exposure": 53760,
            "validation_transition_reads": 1205 * 3,
            "test_transition_reads": 0, "manual_g1_opens": 0,
            "manual_label_opens": 0, "reward_classifier_inference": 0,
            "reward_classifier_updates": 0,
        },
    }
    artifact = {
        "schema_version": "forcesmolvla_stage2b_long_run_half_pass.v1",
        "status": "pass", "artifact_scope": "development_offline_RFT_only",
        "STAGE2_LONG_RUN_BUDGET": "0.5_actor_pass",
        "LONG_RUN_AUTHORIZED": "yes_for_210_cycles_only",
        "LONG_RUN_STARTED": "yes", "LONG_RUN_COMPLETED": "pass",
        "ADDITIONAL_LONG_RUN_AUTHORIZED": "no",
        "DEPLOYMENT_CHECKPOINT_AUTHORIZED": "no",
        "ROBOT_EXECUTION_AUTHORIZED": False,
        "parent": {
            "path": "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint",
            "g7b_smoke_checkpoint_used": False,
            "parent_critic_optimizer_updates": 256,
            "parent_actor_optimizer_updates": 0,
        },
        "resolved_counts": {
            "joint_cycles": 210, "actor_optimizer_updates": 210,
            "critic_optimizer_updates": 420,
            "polyak_updates_per_target": 420,
            "actor_transition_exposure": 5040,
            "critic_transition_exposure": 53760,
        },
        "training": train,
        "resume_audit": {
            "cycle_105_fresh_process": segments[1]["resume_audit"],
            "cycle_210_strict_load": verify,
            "passed": True,
        },
        "checkpoints": checkpoint_bindings,
        "final_checkpoint": {
            "relative_path": final_path.relative_to(ROOT).as_posix(),
            "tree_sha256": final_tree["tree_sha256"],
            "checkpoint_manifest_sha256": sha(final_path / "checkpoint_manifest.json"),
        },
        "source_manifest": {
            "relative_path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": sha(SOURCE),
        },
        "protected_inputs_unchanged": before == after,
        "tests": {"exit_code": 0, "output": (tests.stdout + tests.stderr).strip()},
        "continue_to_1_actor_pass_recommendation": "review_required_not_authorized",
    }
    artifact["artifact_payload_sha256"] = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    atomic_json(OUTPUT / "segment_0_105_result.json", segments[0])
    atomic_json(OUTPUT / "segment_105_210_result.json", segments[1])
    atomic_json(OUTPUT / "strict_load_result.json", verify)
    atomic_json(ARTIFACT, artifact)
    atomic(REPORT, report_markdown(artifact).encode())
    print("STAGE2B_LONG_RUN_HALF_PASS pass", flush=True)


if __name__ == "__main__":
    main()
