#!/usr/bin/env python3
"""Recover the authorized Stage-2B run from its complete cycle-105 boundary."""

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

import run_stage2b_long_run_half_pass as base


ROOT = Path(__file__).parents[1].resolve()
WORKER = ROOT / "tools/run_stage2b_long_run_half_pass_worker_v5.py"
AUDITOR = ROOT / "tools/audit_stage2b_long_run_recovery_boundary_v5.py"
SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v25_stage2b_long_run_recovery.json"
OUTPUT = ROOT / "artifacts/development/stage2/stage2b_long_run_half_pass"
CHECKPOINTS = ROOT / "artifacts/development/stage2/stage2b_long_run_half_pass_checkpoints"
HISTORICAL_FAILURE = ROOT / "artifacts/development/stage2/s2_stage2b_long_run_half_pass.v4.json"
ARTIFACT = ROOT / "artifacts/development/stage2/s2_stage2b_long_run_half_pass.recovered.v5.json"
REPORT = ROOT / "docs/stage2b_long_run_half_pass_report.recovered.v5.md"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT / 'tools'}:{ROOT}",
    })
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    require(completed.returncode == 0, f"STAGE2B_RECOVERY_CHILD_FAILED:{command}:{completed.returncode}")


def compact_cycle(progress: dict) -> dict:
    return {
        "cycle": progress["cycle"],
        "critic_loss": progress["critic_loss"],
        "fm_loss": progress["fm_loss"],
        "actor_q_loss": progress["actor_q_loss"],
        "actor_total_loss": progress["actor_total_loss"],
        "cycle_seconds": progress["cycle_seconds"],
    }


def markdown(artifact: dict) -> str:
    validation = artifact["training"]["validation_diagnostics"]
    gradients = artifact["training"]["gradient_scale_diagnostics"]
    validation_rows = "\n".join(
        f"| {item['cycle']} | {item['td_mse']['q1']:.6g}/{item['td_mse']['q2']:.6g} | "
        f"{item['q_vs_mc_return']['mae']:.6g} | {item['q_vs_mc_return']['rmse']:.6g} | "
        f"{item['q_vs_mc_return']['spearman']:.6g} |"
        for item in validation
    )
    gradient_rows = "\n".join(
        f"| {item['cycle']} | {item['metrics']['global']['raw_q_over_fm']:.6g} | "
        f"{item['metrics']['global']['weighted_eta3_q_over_beta1_fm']:.6g} | "
        f"{item['metrics']['global']['cosine_similarity']:.6g} |"
        for item in gradients
    )
    runtime = artifact["training"]["runtime"]
    final = artifact["final_checkpoint"]
    return f"""# Stage-2B Frozen-VLM 0.5 Actor-pass recovered long-run

Status: **PASS**. Cycles 1–105 were preserved from the original process. A post-checkpoint report-field mismatch stopped v4 only after the complete cycle-105 checkpoint; v5 replayed zero-update boundary diagnostics and resumed in a fresh process at cycle 106. No training cycle was repeated.

## Validation

| cycle | TD MSE Q1/Q2 | Q/MC MAE | RMSE | Spearman |
|---:|---:|---:|---:|---:|
{validation_rows}

## Gradient scale

| cycle | raw Q/FM | weighted eta=3 | cosine |
|---:|---:|---:|---:|
{gradient_rows}

The JSON artifact retains a continuous 210-cycle compact loss curve. Detailed per-substep tensors are retained for cycles 106–210; cycles 1–105 retain the fsynced core loss/latency progress because their in-memory verbose report was lost after checkpoint serialization.

## Runtime and checkpoint

- Training-body runtime: {runtime['training_body_seconds'] / 3600:.3f} h.
- Mean throughput: {runtime['cycles_per_hour']:.3f} cycles/hour.
- Peak allocated/reserved in the resumed same-shape segment: {runtime['peak_allocated_bytes'] / 2**30:.2f}/{runtime['peak_reserved_bytes'] / 2**30:.2f} GiB.
- Final checkpoint: `{final['relative_path']}`.
- Final checkpoint tree SHA-256: `{final['tree_sha256']}`.
- Cycle-105 fresh-process exact resume and cycle-210 strict load: pass.

Actual validation reads were 6,025: the original cycles 0/105, zero-update recovery audits for 0/105, and final cycle 210. Test, manual G1, manual labels, and Reward Classifier access remained zero.

This is a development offline checkpoint only. Additional long-run, deployment, online HIL, ROS/RTC, and robot execution remain unauthorized.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run, "pass --run for the authorized cycle-105 recovery")
    require(not ARTIFACT.exists() and not REPORT.exists(), "STAGE2B_RECOVERY_APPEND_ONLY_TARGET_EXISTS")
    failure = json.loads(HISTORICAL_FAILURE.read_text())
    require(failure["status"] == "stopped" and failure["failure_segment"] == [0, 105], "STAGE2B_RECOVERY_HISTORICAL_FAILURE")

    from forcesmolvla.rft.exact_resume import checkpoint_tree
    from forcesmolvla.rft.g7_long_run import validate_cycle_checkpoint
    from forcesmolvla.rft.source_manifest import validate_stage2_source_manifest

    validate_stage2_source_manifest(ROOT, SOURCE)
    validate_cycle_checkpoint(CHECKPOINTS / "milestone_cycle_000000", expected_cycle=0)
    midpoint_manifest = validate_cycle_checkpoint(CHECKPOINTS / "milestone_cycle_000105", expected_cycle=105)
    progress_before = [json.loads(line) for line in (OUTPUT / "progress.jsonl").read_text().splitlines()]
    require([item["cycle"] for item in progress_before] == list(range(106)), "STAGE2B_RECOVERY_PROGRESS_0_105")
    require(progress_before[-1]["validation"] and progress_before[-1]["checkpoint"], "STAGE2B_RECOVERY_MIDPOINT_FLAGS")

    work = Path(tempfile.mkdtemp(prefix="stage2b-recovery-v5-", dir="/tmp"))
    audit_paths = {cycle: work / f"audit_{cycle}.json" for cycle in (0, 105)}
    for cycle, result in audit_paths.items():
        run([sys.executable, str(AUDITOR), "--cycle", str(cycle), "--result", str(result)])

    protected = work / "protected.json"
    protected.write_text(json.dumps(base.snapshot(), indent=2, sort_keys=True) + "\n")
    segment_path = work / "segment_105_210.json"
    run([
        sys.executable, str(WORKER), "--mode", "segment",
        "--start-cycle", "105", "--end-cycle", "210",
        "--protected", str(protected), "--result", str(segment_path),
    ])
    verify_path = work / "strict_load_210.json"
    run([sys.executable, str(WORKER), "--mode", "verify", "--result", str(verify_path)])

    audits = {cycle: json.loads(path.read_text()) for cycle, path in audit_paths.items()}
    segment = json.loads(segment_path.read_text())
    verify = json.loads(verify_path.read_text())
    progress = [json.loads(line) for line in (OUTPUT / "progress.jsonl").read_text().splitlines()]
    require([item["cycle"] for item in progress] == list(range(211)), "STAGE2B_RECOVERY_PROGRESS_0_210")
    require([item["cycle"] for item in segment["cycles"]] == list(range(106, 211)), "STAGE2B_RECOVERY_SEGMENT_SEQUENCE")
    require(segment["resume_audit"] is not None, "STAGE2B_RECOVERY_RESUME_AUDIT_MISSING")

    compact = [compact_cycle(item) for item in progress[1:]]
    training_seconds = sum(item["cycle_seconds"] for item in compact)
    final_path = CHECKPOINTS / "milestone_cycle_000210"
    final_tree = checkpoint_tree(final_path)
    artifact = {
        "schema_version": "forcesmolvla_stage2b_long_run_half_pass.recovered.v5",
        "status": "pass",
        "artifact_scope": "development_offline_RFT_only",
        "STAGE2_LONG_RUN_BUDGET": "0.5_actor_pass",
        "LONG_RUN_AUTHORIZED": "yes_for_210_cycles_only",
        "LONG_RUN_STARTED": "yes",
        "LONG_RUN_COMPLETED": "pass",
        "ADDITIONAL_LONG_RUN_AUTHORIZED": "no",
        "DEPLOYMENT_CHECKPOINT_AUTHORIZED": "no",
        "ROBOT_EXECUTION_AUTHORIZED": False,
        "historical_v4_stop_preserved": {
            "relative_path": HISTORICAL_FAILURE.relative_to(ROOT).as_posix(),
            "sha256": sha(HISTORICAL_FAILURE),
            "failure_scope": "post_checkpoint_report_field_compatibility",
            "training_or_numeric_failure": False,
        },
        "recovery": {
            "midpoint_checkpoint_manifest_payload_sha256": midpoint_manifest["manifest_payload_sha256"],
            "first_resumed_cycle": 106,
            "training_cycles_replayed": 0,
            "boundary_audit_replays": [0, 105],
            "boundary_audit_optimizer_updates": 0,
            "cycle_105_fresh_process": segment["resume_audit"],
            "cycle_210_strict_load": verify,
            "passed": True,
        },
        "resolved_counts": {
            "joint_cycles": 210,
            "actor_optimizer_updates": 210,
            "critic_optimizer_updates": 420,
            "polyak_updates_per_target": 420,
            "actor_transition_exposure": 5040,
            "critic_transition_exposure": 53760,
            "actor_transition_passes": 5040 / 10075,
        },
        "training": {
            "compact_cycle_curve": compact,
            "detailed_resumed_cycles": segment["cycles"],
            "boundary_audits": [
                audits[0]["boundary_audit"], audits[105]["boundary_audit"],
                segment["boundary_audits"][0],
            ],
            "gradient_scale_diagnostics": [
                audits[0]["gradient_scale_diagnostic"],
                audits[105]["gradient_scale_diagnostic"],
                segment["gradient_scale_diagnostics"][0],
            ],
            "validation_diagnostics": [
                audits[0]["validation_diagnostic"], audits[105]["validation_diagnostic"],
                segment["validation_diagnostics"][0],
            ],
            "runtime": {
                "training_body_seconds": training_seconds,
                "cycles_per_hour": 210 / training_seconds * 3600,
                "actor_transitions_per_second": 5040 / training_seconds,
                "critic_transitions_per_second": 53760 / training_seconds,
                "peak_allocated_bytes": segment["runtime"]["peak_allocated_bytes"],
                "peak_reserved_bytes": segment["runtime"]["peak_reserved_bytes"],
                "peak_scope": "cycles_106_210_same_shape_resume_segment",
            },
            "data_access": {
                "train_actor_transition_exposure": 5040,
                "train_critic_td_transition_exposure": 53760,
                "train_calql_transition_exposure": 53760,
                "validation_transition_reads": 6025,
                "test_transition_reads": 0,
                "manual_g1_opens": 0,
                "manual_label_opens": 0,
                "reward_classifier_inference": 0,
                "reward_classifier_updates": 0,
            },
        },
        "final_checkpoint": {
            "relative_path": final_path.relative_to(ROOT).as_posix(),
            "tree_sha256": final_tree["tree_sha256"],
            "checkpoint_manifest_sha256": sha(final_path / "checkpoint_manifest.json"),
        },
        "source_manifest": {
            "relative_path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": sha(SOURCE),
        },
        "continue_to_1_actor_pass_recommendation": "review_required_not_authorized",
    }
    artifact["artifact_payload_sha256"] = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    base.atomic_json(OUTPUT / "recovered_boundary_cycle_000000.json", audits[0])
    base.atomic_json(OUTPUT / "recovered_boundary_cycle_000105.json", audits[105])
    base.atomic_json(OUTPUT / "segment_105_210_result.recovered.v5.json", segment)
    base.atomic_json(OUTPUT / "strict_load_result.recovered.v5.json", verify)
    base.atomic_json(ARTIFACT, artifact)
    base.atomic(REPORT, markdown(artifact).encode())
    print("STAGE2B_LONG_RUN_HALF_PASS_RECOVERED_V5 pass", flush=True)


if __name__ == "__main__":
    main()
