#!/usr/bin/env python3
"""CPU coordinator for the 256-cycle G7 development long-run stage 1."""

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


ROOT = Path(__file__).parents[1].resolve()
WORKER = ROOT / "tools/run_s2_g7_long_run_worker.py"
SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v18_g7_long_run.json"
OUTPUT = ROOT / "artifacts/development/stage2/g7_long_run_stage1"
CHECKPOINTS = ROOT / "artifacts/development/stage2/g7_long_run_stage1_checkpoints"
ARTIFACT = ROOT / "artifacts/development/stage2/s2_g7_long_run_stage1.json"
REPORT = ROOT / "docs/s2_g7_long_run_stage1_report.md"


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
    import preflight_s2_g7b_joint_smoke_gpu as g7b

    base = g7b.snapshot()
    files = {
        "long_config": ROOT / "configs/stage2_g7_long_run_stage1.development.yaml",
        "long_source_manifest": SOURCE,
        "g7b_artifact": ROOT / "artifacts/development/stage2/s2_g7b_joint_smoke_preflight.json",
        "g7b_report": ROOT / "docs/s2_g7b_joint_smoke_report.md",
        "g7a_fixed_diagnostics": ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup/fixed_diagnostics.pt",
    }
    return {
        "g7b_protected": base,
        "files": {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "file_size": path.stat().st_size} for name, path in files.items()},
        "trees": {"g7b_smoke_checkpoint_preserved_not_parent": checkpoint_tree(ROOT / "artifacts/development/stage2/g7b_joint_smoke_checkpoint.development")},
    }


def run_worker(mode: str, result: Path, protected: Path | None = None) -> dict:
    command = [sys.executable, str(WORKER), "--mode", mode, "--result", str(result)]
    if protected is not None:
        command.extend(("--protected", str(protected)))
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT}",
    })
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    require(completed.returncode == 0, f"G7_LONG_{mode.upper()}_WORKER_FAILED:{completed.returncode}")
    return json.loads(result.read_text())


def describe(values: list[float]) -> str:
    return f"median={statistics.median(values):.6g}, min={min(values):.6g}, max={max(values):.6g}"


def report_markdown(artifact: dict) -> str:
    train = artifact["train"]
    windows = []
    for start in range(0, 256, 32):
        cycles = train["cycles"][start:start + 32]
        windows.append(
            f"| {start + 1}–{start + 32} | "
            f"{describe([value for cycle in cycles for value in [item['loss']['L_critic'] for item in cycle['critic_updates']]])} | "
            f"{describe([cycle['actor_update']['loss']['L_FM_window'] for cycle in cycles])} | "
            f"{describe([cycle['actor_update']['loss']['L_actor_Q_window'] for cycle in cycles])} | "
            f"{describe([cycle['actor_update']['loss']['weighted_actor_total'] for cycle in cycles])} | "
            f"{describe([cycle['gradient_scale']['weighted_eta_q_over_beta_fm'] for cycle in cycles])} |"
        )
    validation = []
    for item in train["validation_diagnostics"]:
        validation.append(
            f"| {item['cycle']} | {item['td_mse']['q1']:.6g}/{item['td_mse']['q2']:.6g} | "
            f"{item['calql_conservative']['q1']:.6g}/{item['calql_conservative']['q2']:.6g} | "
            f"{item['q_vs_mc_return']['mae']:.6g} | {item['q_vs_mc_return']['rmse']:.6g} | "
            f"{item['q_vs_mc_return']['spearman']:.6g} |"
        )
    gradient = train["gradient_scale_summary"]
    runtime = train["runtime"]
    return """# Stage-2 G7 development long-run stage 1

Status: **PASS (development learning dynamics only)**. Training started from the frozen G7-A-r2 Critic-warmup checkpoint: Actor optimizer step 0 and Critic optimizer step 256. The G7-B smoke checkpoint was preserved but was not loaded.

Exactly 256 complete joint cycles ran: 512 Critic optimizer/Polyak updates and 256 Actor optimizer updates. Eta remained 10.0 and beta remained 1.0 throughout.

## Training curves by fixed 32-cycle window

| cycles | Critic loss | FM loss | Actor-Q loss | total Actor loss | weighted gradient ratio |
|---:|---|---|---|---|---|
""" + "\n".join(windows) + """

The complete per-cycle loss, Q/target-Q distribution, action-drift, gradient, timing, sample-count, and row-identity evidence is stored in the JSON artifact and `progress.jsonl`.

## Read-only validation diagnostics

| cycle | TD MSE Q1/Q2 | Cal-QL Q1/Q2 | Q-vs-return MAE | RMSE | Spearman |
|---:|---:|---:|---:|---:|---:|
""" + "\n".join(validation) + f"""

Validation was evaluated only at the predeclared cycles 0/64/128/256 with `no_grad`; it did not select a checkpoint, stop training, or update parameters/RNG.

## Gradient/action/runtime audit

Weighted `||eta*g_Q||/||beta*g_FM||` median/P95/max: `{gradient['weighted_eta10']['median']:.6g}` / `{gradient['weighted_eta10']['p95']:.6g}` / `{gradient['weighted_eta10']['maximum']:.6g}`. Raw ratio median/P95/max: `{gradient['raw']['median']:.6g}` / `{gradient['raw']['p95']:.6g}` / `{gradient['raw']['maximum']:.6g}`. Gradient-cosine median/P95/max: `{gradient['cosine']['median']:.6g}` / `{gradient['cosine']['p95']:.6g}` / `{gradient['cosine']['maximum']:.6g}`.

TCP6 Q-gradient was nonzero, gripper Q-gradient exactly zero, and gripper FM-gradient nonzero for every cycle. Public `predict_action_chunk()` success was `{sum(item['success'] for item in train['public_predict_diagnostics'])}/{len(train['public_predict_diagnostics'])}`. Maximum fixed-observation normalized TCP drift was `{max(item.get('normalized_tcp_drift_l2', 0.0) for item in train['action_diagnostics']):.6g}`; maximum binary-gripper change rate was `{max(item.get('binary_gripper_change_rate', 0.0) for item in train['action_diagnostics']):.6g}`. Internal raw-gripper out-of-public-tolerance rate was `{train['action_contract_v2']['internal_raw_gripper_out_of_public_tolerance_rate']:.6g}` and remained diagnostic-only.

Training-body runtime was `{runtime['training_body_seconds']:.1f}` seconds. Peak CUDA allocated/reserved memory was `{runtime['peak_allocated_bytes']}` / `{runtime['peak_reserved_bytes']}` bytes.

## Checkpoint and limits

Recovery checkpoints were atomically replaced every 32 cycles. Append-only milestones 0/64/128/256 were retained, and the cycle-256 milestone passed fresh-process strict model/optimizer/scheduler/sampler/RNG loading.

Test transitions/images, manual G1, manual labels, and Reward Classifier inference/updates were zero. Stage-1, detector-G1, public inference, normalizer, ActionContract-v2, G7-A-r2, and G7-B evidence remained unchanged.

Eta=10.0 remains development-only. This result does not authorize policy evaluation, online HIL, ROS/RTC, deployment, or robot execution. The next allowed action is `review_256_cycle_learning_dynamics_and_freeze_final_budget`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run, "pass --run for authorized G7 long-run stage 1")
    for path in (OUTPUT, CHECKPOINTS, ARTIFACT, REPORT):
        require(not path.exists(), f"G7_LONG_APPEND_ONLY_TARGET_EXISTS:{path}")
    from forcesmolvla.rft.source_manifest import validate_stage2_source_manifest
    source = validate_stage2_source_manifest(ROOT, SOURCE)
    require(source["scope"] == "G7_development_long_run_stage1_ActionContract_v2", "G7_LONG_SOURCE_SCOPE")
    environment = os.environ.copy(); environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"; environment["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_rft_g7_long_run.py", "tests/test_rft_g7b.py", "tests/test_rft_critic_action_contract_v2.py"], cwd=ROOT, env=environment, capture_output=True, text=True)
    require(tests.returncode == 0, f"G7_LONG_TEST_FAILURE:{(tests.stdout + tests.stderr)[-3000:]}")
    before = snapshot()
    work = Path(tempfile.mkdtemp(prefix="g7-long-", dir="/tmp"))
    protected = work / "protected.json"
    protected.write_text(json.dumps(before, indent=2, sort_keys=True) + "\n")
    train = run_worker("train", work / "train.json", protected)
    verify = run_worker("verify", work / "verify.json")
    after = snapshot()
    require(before == after, "G7_LONG_FROZEN_INPUT_SHA_CHANGED")
    expected = {
        "joint_cycles": 256, "critic_optimizer_updates": 512,
        "actor_optimizer_updates": 256, "q1_target_polyak_updates": 512,
        "q2_target_polyak_updates": 512, "critic_scheduler_steps": 512,
        "actor_scheduler_steps": 256, "actor_target_updates": 0,
    }
    require(train["counters"] == expected and verify["counters"] == expected, "G7_LONG_FINAL_COUNTERS")
    require(verify["strict_model_load"] and verify["critic_optimizer_step"] == 768 and verify["actor_optimizer_step"] == 256, "G7_LONG_FRESH_LOAD")
    progress = [json.loads(line) for line in (OUTPUT / "progress.jsonl").read_text().splitlines()]
    require([item["cycle"] for item in progress] == list(range(257)), "G7_LONG_PROGRESS_SEQUENCE")
    atomic_write(OUTPUT / "train_result.json", (json.dumps(train, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    atomic_write(OUTPUT / "strict_load_result.json", (json.dumps(verify, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    from forcesmolvla.rft.exact_resume import checkpoint_tree
    checkpoint_bindings = {
        path.name: checkpoint_tree(path)
        for path in sorted(CHECKPOINTS.iterdir()) if path.is_dir()
    }
    artifact = {
        "schema_version": "forcesmolvla_s2_g7_long_run_stage1.v1",
        "artifact_status": "development_only", "LONG_RUN_STAGE1": "256_joint_cycles",
        "status": "pass", "ETA": "10.0_development_only",
        "CRITIC_UPDATES": 512, "ACTOR_UPDATES": 256,
        "POLYAK_UPDATES_PER_TARGET": 512,
        "ROBOT_EXECUTION_AUTHORIZED": False,
        "NEXT_ALLOWED_ACTION": "review_256_cycle_learning_dynamics_and_freeze_final_budget",
        "source_manifest": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": sha(SOURCE)},
        "checkpoints": checkpoint_bindings, "tests": {"exit_code": 0, "output": (tests.stdout + tests.stderr).strip()},
        "train": train, "fresh_process_strict_load": verify,
        "protected_inputs_before": before, "protected_inputs_after_exact": before == after,
        "research_limits": {"all_success_demos": True, "reward_model_training_overlap": True, "unbiased_policy_evaluation": False, "eta_paper_final": False, "eta_online_deployment": False},
    }
    artifact["artifact_payload_sha256"] = hashlib.sha256(json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    atomic_write(ARTIFACT, (json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    atomic_write(REPORT, report_markdown(artifact).encode())
    print("G7_LONG_RUN_STAGE1 pass")


if __name__ == "__main__":
    main()
