#!/usr/bin/env python3
"""CPU coordinator for the append-only G7-B development joint smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).parents[1].resolve()
WORKER = ROOT / "tools/run_s2_g7b_worker.py"
SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v17_g7b.json"
OUTPUT = ROOT / "artifacts/development/stage2/g7b_joint_smoke"
CHECKPOINT = ROOT / "artifacts/development/stage2/g7b_joint_smoke_checkpoint.development"
ARTIFACT = ROOT / "artifacts/development/stage2/s2_g7b_joint_smoke_preflight.json"
REPORT = ROOT / "docs/s2_g7b_joint_smoke_report.md"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def snapshot() -> dict:
    from forcesmolvla.rft.exact_resume import checkpoint_tree
    import preflight_s2_g7a_r2_critic_warmup_gpu as r2

    base = r2._protected_snapshot_r2()
    files = {
        "g7a_r2_artifact": ROOT / "artifacts/development/stage2/s2_g7a_r2_critic_warmup_preflight.json",
        "g7a_r2_report": ROOT / "docs/s2_g7a_r2_critic_warmup_report.md",
        "g7b_config": ROOT / "configs/stage2_g7b_joint_smoke.development.yaml",
        "g7b_source_manifest": SOURCE,
        "public_action_source": ROOT / "src/forcesmolvla/action_delta.py",
        "public_model_source": ROOT / "src/forcesmolvla/modeling_forcesmolvla.py",
        "public_rules_source": ROOT / "src/forcesmolvla/rules.py",
        "public_rulespec": ROOT / "configs/live_action_safety.task2.development.yaml",
    }
    return {
        "g7a_r2_protected": base,
        "files": {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "file_size": path.stat().st_size} for name, path in files.items()},
        "trees": {"g7a_r2_parent_checkpoint": checkpoint_tree(ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint")},
    }


def run_worker(mode: str, result: Path, protected: Path | None = None) -> dict:
    command = [sys.executable, str(WORKER), "--mode", mode, "--checkpoint", str(CHECKPOINT), "--result", str(result)]
    if protected is not None:
        command.extend(("--protected", str(protected)))
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT}",
    })
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, check=False)
    require(completed.returncode == 0, f"G7B_{mode.upper()}_WORKER_FAILED:{completed.returncode}")
    return json.loads(result.read_text())


def report_markdown(artifact: dict) -> str:
    train = artifact["train"]
    rows = []
    for cycle in train["cycles"]:
        c1, c2 = cycle["critic_updates"]
        actor = cycle["actor_update"]
        rows.append(
            f"| {cycle['cycle']} | {c1['loss']['L_critic']:.6g} / {c2['loss']['L_critic']:.6g} | "
            f"{c1['loss']['L_TD_Q1']:.5g}/{c1['loss']['L_TD_Q2']:.5g} → {c2['loss']['L_TD_Q1']:.5g}/{c2['loss']['L_TD_Q2']:.5g} | "
            f"{c1['loss']['L_CalQL_Q1']:.5g}/{c1['loss']['L_CalQL_Q2']:.5g} → {c2['loss']['L_CalQL_Q1']:.5g}/{c2['loss']['L_CalQL_Q2']:.5g} | "
            f"{actor['loss']['L_FM_window']:.6g} | {actor['loss']['L_actor_Q_window']:.6g} | {actor['loss']['weighted_actor_total']:.6g} | {cycle['cycle_latency_seconds']:.2f} |"
        )
    gradient = train["gradient_scale_summary"]
    return """# Stage-2 G7-B development joint-smoke report

Status: **PASS (development mechanics only)**. The frozen G7-A-r2 Critic warm-up checkpoint was loaded at update 256; no G5/G6 smoke checkpoint was used. Exactly eight joint cycles ran, each with two Critic updates followed by one Actor update.

| cycle | Critic loss #1/#2 | TD Q1/Q2 #1→#2 | Cal-QL Q1/Q2 #1→#2 | FM | Actor-Q | Actor total | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + f"""

## Gradient and action contract

Unweighted `||g_Q||/||g_FM||` median/P95/max: `{gradient['raw']['median']:.6g}` / `{gradient['raw']['p95']:.6g}` / `{gradient['raw']['maximum']:.6g}`. With the smoke-only eta=10, the weighted values are `{gradient['weighted_eta10']['median']:.6g}` / `{gradient['weighted_eta10']['p95']:.6g}` / `{gradient['weighted_eta10']['maximum']:.6g}`. Gradient-cosine median/P95/max: `{gradient['cosine']['median']:.6g}` / `{gradient['cosine']['p95']:.6g}` / `{gradient['cosine']['maximum']:.6g}`.

Every cycle had nonzero TCP6 Q-gradient, exactly zero gripper Q-gradient, and nonzero Flow-Matching gripper gradient. The v2 total-binary internal projection remained separate from public execution authorization. The raw out-of-public-tolerance rate is a distribution diagnostic only; it neither clipped nor resampled Critic candidates. Fixed train observations and fixed noise were used for normalized TCP drift/binary-gripper diagnostics. `predict_action_chunk()` completed before training and after every cycle under the unchanged public RuleSpec.

## Ownership, access, and limits

Actor, Q1/Q2, and both Polyak targets changed in their authorized substeps; frozen ResNet backbones did not. The final atomic checkpoint passed a fresh-process strict model/optimizer/scheduler/sampler/RNG load. Validation/test transitions, manual G1, manual labels, and Reward Classifier inference/updates were all zero.

This run does not authorize a long run, policy evaluation, export, online HIL, ROS/RTC, or robot execution. Eta=10.0 is approved only for these eight development cycles. The all-success dataset and reward-model training overlap remain unchanged, so no policy-improvement, recovery, unbiased-evaluation, or deployment claim is made.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run, "pass --run for authorized G7-B")
    for path in (OUTPUT, CHECKPOINT, ARTIFACT, REPORT):
        require(not path.exists(), f"G7B_APPEND_ONLY_TARGET_EXISTS:{path}")
    from forcesmolvla.rft.g7b import verify_g7b_source_manifest
    verify_g7b_source_manifest(ROOT, SOURCE)
    environment = os.environ.copy(); environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"; environment["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_rft_g7b.py", "tests/test_rft_critic_action_contract_v2.py", "tests/test_rft_losses.py"], cwd=ROOT, env=environment, capture_output=True, text=True)
    require(tests.returncode == 0, f"G7B_TEST_FAILURE:{(tests.stdout + tests.stderr)[-3000:]}")
    before = snapshot()
    work = Path(tempfile.mkdtemp(prefix="g7b-", dir="/tmp"))
    protected = work / "protected.json"
    protected.write_text(json.dumps(before, indent=2, sort_keys=True) + "\n")
    train = run_worker("train", work / "train.json", protected)
    verify = run_worker("verify", work / "verify.json")
    after = snapshot()
    require(before == after, "G7B_FROZEN_INPUT_SHA_CHANGED")
    require(train["counters"] == {
        "joint_cycles": 8, "critic_optimizer_updates": 16, "actor_optimizer_updates": 8,
        "q1_target_polyak_updates": 16, "q2_target_polyak_updates": 16,
        "critic_scheduler_steps": 16, "actor_scheduler_steps": 8, "actor_target_updates": 0,
    }, "G7B_FINAL_COUNTERS_INVALID")
    require(verify["strict_model_load"] and verify["critic_optimizer_step"] == 272 and verify["actor_optimizer_step"] == 8, "G7B_FRESH_LOAD_FAILED")
    require(train["data_access"] == {
        "train_transitions_available": 10075, "validation_transition_reads": 0,
        "test_transition_reads": 0, "manual_g1_opens": 0, "manual_label_opens": 0,
        "reward_classifier_inference": 0, "reward_classifier_updates": 0,
    }, "G7B_DATA_ACCESS_INVALID")
    OUTPUT.mkdir(parents=True)
    atomic_write(OUTPUT / "train_worker_result.json", (json.dumps(train, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    atomic_write(OUTPUT / "strict_load_result.json", (json.dumps(verify, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    artifact = {
        "schema_version": "forcesmolvla_s2_g7b_joint_smoke_preflight.v1",
        "artifact_status": "development_only", "G7B_JOINT_SMOKE": "pass",
        "ETA_USED": "10.0_development_only", "CRITIC_UPDATES": 16,
        "ACTOR_UPDATES": 8, "POLYAK_UPDATES_PER_TARGET": 16,
        "LONG_RUN_AUTHORIZED": "no", "ROBOT_EXECUTION_AUTHORIZED": False,
        "NEXT_ALLOWED_ACTION": "request_long_run_recipe_and_eta_approval",
        "source_manifest": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": sha(SOURCE)},
        "checkpoint": {"path": CHECKPOINT.relative_to(ROOT).as_posix(), "manifest_sha256": sha(CHECKPOINT / "checkpoint_manifest.json")},
        "tests": {"exit_code": 0, "output": (tests.stdout + tests.stderr).strip()},
        "train": train, "fresh_process_strict_load": verify,
        "protected_inputs_before": before, "protected_inputs_after_exact": before == after,
        "research_limits": {"all_success_demos": True, "reward_model_training_overlap": True, "unbiased_policy_evaluation": False},
    }
    artifact["artifact_payload_sha256"] = hashlib.sha256(json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    atomic_write(ARTIFACT, (json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    atomic_write(REPORT, report_markdown(artifact).encode())
    print("G7B_JOINT_SMOKE pass")


if __name__ == "__main__":
    main()
