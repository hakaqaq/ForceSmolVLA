#!/usr/bin/env python3
"""Create append-only cycle-136 evidence for the safely interrupted pilot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import run_stage2b_long_run_half_pass as base


ROOT = Path(__file__).parents[1].resolve()
WORKER = ROOT / "tools/run_stage2b_interrupted_pilot_worker_v1.py"
SOURCE = ROOT / (
    "artifacts/development/stage2/"
    "stage2_source_manifest.v27_stage2b_interrupted_pilot.json"
)
ORIGINAL_OUTPUT = ROOT / "artifacts/development/stage2/stage2b_long_run_half_pass"
ORIGINAL_CHECKPOINTS = ROOT / (
    "artifacts/development/stage2/stage2b_long_run_half_pass_checkpoints"
)
OUTPUT = ROOT / "artifacts/development/stage2/stage2b_interrupted_pilot_cycle136"
CHECKPOINTS = ROOT / (
    "artifacts/development/stage2/"
    "stage2b_interrupted_pilot_cycle136_checkpoints"
)
ARTIFACT = ROOT / (
    "artifacts/development/stage2/s2_stage2b_interrupted_pilot.v1.json"
)
REPORT = ROOT / "docs/stage2b_interrupted_pilot_report.v1.md"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT / 'tools'}:{ROOT}",
    })
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    require(completed.returncode == 0, f"INTERRUPTED_CHILD_FAILED:{command}:{completed.returncode}")


def semantic_cycle(record: dict) -> dict:
    return {
        "cycle": record["cycle"],
        "critic_loss": [item["loss"]["L_critic"] for item in record["critic_updates"]],
        "fm_loss": record["actor_update"]["loss"]["flow_matching"],
        "actor_q_loss": record["actor_update"]["loss"]["actor_q_min_twin"],
        "actor_total_loss": record["actor_update"]["loss"]["weighted_total"],
    }


def main() -> None:
    from forcesmolvla.rft.exact_resume import checkpoint_tree
    from forcesmolvla.rft.long_run_checkpoint import hardlink_milestone, validate_cycle_checkpoint
    from forcesmolvla.rft.source_manifest import validate_stage2_source_manifest

    require(not any(path.exists() for path in (OUTPUT, CHECKPOINTS, ARTIFACT, REPORT)), "INTERRUPTED_APPEND_ONLY_TARGET_EXISTS")
    validate_stage2_source_manifest(ROOT, SOURCE)
    original_progress = [
        json.loads(line)
        for line in (ORIGINAL_OUTPUT / "progress.jsonl").read_text().splitlines()
    ]
    require([item["cycle"] for item in original_progress] == list(range(137)), "INTERRUPTED_ORIGINAL_PROGRESS_SEQUENCE")
    require(original_progress[-1]["status"] == "complete_boundary", "INTERRUPTED_LAST_CYCLE_INCOMPLETE")
    source_105 = ORIGINAL_CHECKPOINTS / "milestone_cycle_000105"
    validate_cycle_checkpoint(source_105, expected_cycle=105)
    hardlink_milestone(source_105, CHECKPOINTS / "recovery_latest", expected_cycle=105)

    work = Path(tempfile.mkdtemp(prefix="stage2b-interrupted-pilot-", dir="/tmp"))
    protected = work / "protected.json"
    protected.write_text(json.dumps(base.snapshot(), indent=2, sort_keys=True) + "\n")
    segment_path = work / "segment_105_136.json"
    run([
        sys.executable, str(WORKER), "--mode", "segment",
        "--start-cycle", "105", "--end-cycle", "136",
        "--protected", str(protected), "--result", str(segment_path),
    ])
    verify_path = work / "strict_load_136.json"
    run([sys.executable, str(WORKER), "--mode", "verify", "--result", str(verify_path)])

    segment = json.loads(segment_path.read_text())
    verify = json.loads(verify_path.read_text())
    replay = [semantic_cycle(item) for item in segment["cycles"]]
    expected = [
        {key: item[key] for key in (
            "cycle", "critic_loss", "fm_loss", "actor_q_loss",
            "actor_total_loss",
        )}
        for item in original_progress[106:137]
    ]
    require(replay == expected, "INTERRUPTED_REPLAY_NOT_EXACT")
    checkpoint = CHECKPOINTS / "milestone_cycle_000136"
    checkpoint_manifest = validate_cycle_checkpoint(checkpoint, expected_cycle=136)
    tree = checkpoint_tree(checkpoint)
    latest = segment["cycles"][-1]
    interruption = {
        "status": "interrupted_for_throughput_optimization",
        "artifact_role": "audit_only_interrupted",
        "training_parent_authorized": False,
        "deployment_authorized": False,
        "robot_execution_authorized": False,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_tree_sha256": tree["tree_sha256"],
        "checkpoint_manifest_payload_sha256": checkpoint_manifest["manifest_payload_sha256"],
    }
    atomic_json(OUTPUT / "interruption_manifest.json", interruption)
    artifact = {
        "schema_version": "forcesmolvla_stage2b_interrupted_pilot.v1",
        "status": "valid_interrupted_long_run_pilot",
        "CURRENT_TRAINING_STOP_REQUESTED": "yes",
        "STOP_MODE": "graceful_after_current_complete_cycle",
        "CURRENT_RUN_STATUS": "valid_interrupted_long_run_pilot",
        "CURRENT_CHECKPOINT_STATUS": "audit_only_interrupted",
        "AUTO_RESUME": "no",
        "completed_joint_cycles": 136,
        "counts": {
            "actor_optimizer_updates": 136,
            "critic_optimizer_updates": 272,
            "target_polyak_updates_per_target": 272,
            "actor_transition_exposure": 3264,
            "critic_transition_exposure": 34816,
            "critic_calql_transition_exposure": 34816,
        },
        "original_parent": {
            "path": "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint",
            "checkpoint_manifest_sha256": "2e0902076cb12a1391613230679730d035155528c9be01bd17dce960d5e707f7",
        },
        "source_manifest": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": sha(SOURCE),
        },
        "safe_stop": {
            "signal": "SIGINT",
            "kill_9_used": False,
            "last_original_progress": original_progress[-1],
            "interrupt_location": "after_progress_fsync_and_cycle_print_during_gc_collect",
            "optimizer_or_polyak_interrupted": False,
        },
        "deterministic_replay": {
            "source_checkpoint_cycle": 105,
            "first_replayed_cycle": 106,
            "last_replayed_cycle": 136,
            "semantic_losses_exact": True,
            "replayed_cycle_count": 31,
            "replay_checkpoint_is_evidence_only": True,
        },
        "latest": {
            "losses": latest["actor_update"]["loss"],
            "critic_losses": [item["loss"] for item in latest["critic_updates"]],
            "critic_q_statistics": [item["statistics"] for item in latest["critic_updates"]],
            "actor_q_statistics": latest["actor_update"]["q"],
            "all_losses_finite": True,
            "all_gradients_finite": True,
        },
        "checkpoint": {
            **interruption,
            "tree": tree,
            "optimizer_state_saved": True,
            "scheduler_state_saved": True,
            "polyak_state_saved": True,
            "rng_state_saved": True,
            "sampler_state_saved": True,
            "fresh_process_strict_load": verify,
        },
        "access": segment["data_access"],
        "THROUGHPUT_V2_AUTHORIZED": "yes",
        "THROUGHPUT_V2_LONG_RUN": "no",
        "THROUGHPUT_V2_TRAINING_CHECKPOINT": "no",
        "RESTART_0_5_PASS_AUTHORIZED": "no",
        "AUTO_EXTEND_TO_1_0_PASS": "no",
        "LONG_RUN_EXTENSION_AUTHORIZED": "no",
        "ROBOT_EXECUTION_AUTHORIZED": False,
    }
    atomic_json(ARTIFACT, artifact)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# Stage-2B interrupted long-run pilot\n\n"
        "Status: **valid_interrupted_long_run_pilot**. The original process completed "
        "cycle 136 and received SIGINT during post-cycle garbage collection. No optimizer "
        "or Polyak operation was interrupted.\n\n"
        "The v6 worker had no signal-triggered checkpoint hook, so cycles 106–136 were "
        "deterministically replayed from the immutable cycle-105 checkpoint. All persisted "
        "core losses matched exactly before the audit-only cycle-136 checkpoint was saved "
        "and strict-loaded in a fresh process. This checkpoint is not an authorized training "
        "parent, policy-evaluation result, deployment checkpoint, or robot checkpoint.\n\n"
        f"Checkpoint tree SHA-256: `{tree['tree_sha256']}`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
