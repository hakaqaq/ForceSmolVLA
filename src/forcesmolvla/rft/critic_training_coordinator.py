#!/usr/bin/env python3
"""CPU-only Twin-Q warmup coordinator with fresh strict-load verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/twin_q_critic_warmup.development.yaml"
WORKER_MODULE = "forcesmolvla.rft.critic_training"
OUTPUT = ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup"
CHECKPOINT = ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path), "file_size": path.stat().st_size,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def protected_snapshot() -> dict:
    from forcesmolvla.rft.exact_resume import checkpoint_tree
    from forcesmolvla.rft.training_cycle_runtime import protected_snapshot

    training_inputs = protected_snapshot()
    files = {
        "critic_warmup_config": CONFIG,
        "critic_worker": ROOT / "src/forcesmolvla/rft/critic_training.py",
    }
    return {
        "training_inputs": training_inputs,
        "files": {name: binding(path) for name, path in files.items()},
        "trees": {
            "parent_actor_checkpoint": training_inputs[
                "parent_actor_checkpoint_tree"
            ],
            "offline_dataset": training_inputs["dataset_storage_tree"],
        },
    }


def wait_process(process: subprocess.Popen, log_path: Path, label: str, timeout: float = 14400) -> None:
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait()
        raise RuntimeError(f"CRITIC_WARMUP_{label}_TIMEOUT")
    if return_code:
        raise RuntimeError(
            f"CRITIC_WARMUP_{label}_FAILED:{return_code}\n"
            f"{log_path.read_text(errors='replace')[-12000:]}"
        )


def worker_environment() -> dict:
    environment = os.environ.copy()
    environment.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HOME": "/tmp/forcesmolvla_critic_warmup_hf_cache",
        "HF_DATASETS_CACHE": "/tmp/forcesmolvla_critic_warmup_hf_cache/datasets",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT / 'tools'}",
    })
    return environment


def report_markdown(artifact: dict) -> str:
    warmup = artifact["warmup"]
    evaluation = warmup["evaluation"]
    gradient = warmup["gradient_scale"]
    rows = []
    for update in ("update_0", "update_256"):
        for split in ("train_probe", "validation"):
            value = evaluation[update][split]
            rows.append(
                f"| {update.removeprefix('update_')} | {split} | {value['row_count']} | "
                f"{value['td_mse']['twin_mean']:.6g} | {value['calql_conservative']['twin_mean']:.6g} | "
                f"{value['total_critic_loss']:.6g} | {value['q_vs_mc_return']['mae']:.6g} | "
                f"{value['q_vs_mc_return']['spearman']} |"
            )
    candidates = gradient["candidates_with_median_in_reference_band"]
    return f"""# Stage-2 G7-A Critic-only warm-up report

Status: `G7A_CRITIC_WARMUP_MECHANICS = pass`.

The worker started from the frozen Stage-1 r5 Actor and fresh G2 seed-0 Twin-Q; no G5/G6 checkpoint training state was loaded. Exactly 256 Critic optimizer/scheduler steps and 256 Polyak updates per target ran. Actor optimizer/scheduler/update counts remained zero, and Actor parameters plus floating buffers matched r5 bitwise before and after.

| Update | Dataset | Rows | TD MSE | Cal-QL term | Total critic loss | Q/MC MAE | Spearman |
|---:|---|---:|---:|---:|---:|---:|---:|
{"\n".join(rows)}

The fixed 32-batch train-only scale probe measured median raw `||g_Q||/||g_FM|| = {gradient['global']['raw_q_over_fm']['median']:.6g}` with p10/p90 `{gradient['global']['raw_q_over_fm']['p10']:.6g}/{gradient['global']['raw_q_over_fm']['p90']:.6g}` and maximum `{gradient['global']['raw_q_over_fm']['maximum']:.6g}`. Median cosine similarity was `{gradient['global']['cosine_similarity']['median']:.6g}`. Measurement-only eta candidates whose median weighted ratio fell in `[0.01, 0.10]`: `{candidates}`. No eta was selected or approved, and no Actor update occurred.

All fixed train/validation row IDs, Flow noises, timesteps, and proposals were frozen before update 0. Validation was evaluated only at updates 0 and 256 and was not used for search, early stopping, or checkpoint selection. Test transition/image reads, manual G1/label reads, and Reward Classifier inference/updates were zero.

The checkpoint is `DEVELOPMENT_G7A_CRITIC_WARMUP_ONLY`, `NOT_FOR_DEPLOYMENT`, `NOT_FOR_POLICY_EVALUATION`, `NOT_AN_APPROVED_LONG_TRAIN_PARENT`, and `APPROVED_ONLY_FOR_G7B_IF_EXPLICITLY_AUTHORIZED`. A second fresh process strictly loaded it without an update or sampler draw.

All demonstrations are successes; Reward Classifier training overlaps the RL train episodes; unbiased policy evaluation is false. G7-A establishes only Critic warm-up numerical behavior and Q-gradient scale. It does not demonstrate policy improvement, failure recovery, OOD conservatism, or reward-model generalization.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run, "pass --run for authorized G7-A")
    require(not torch.cuda.is_initialized(), "G7A_COORDINATOR_MUST_NOT_CREATE_CUDA_CONTEXT")
    require(CONFIG.is_file() and SOURCE_MANIFEST.is_file(), "G7A_CONFIG_OR_SOURCE_MANIFEST_MISSING")
    for target in (OUTPUT, CHECKPOINT, ARTIFACT, REPORT):
        require(not target.exists(), f"G7A_APPEND_ONLY_TARGET_EXISTS:{target}")

    from forcesmolvla.rft.exact_resume import checkpoint_tree
    from forcesmolvla.rft.critic_warmup_checkpoint import (
        CRITIC_WARMUP_COUNTERS,
        validate_critic_warmup_checkpoint,
        verify_source_manifest,
    )

    verify_source_manifest(ROOT, SOURCE_MANIFEST)
    tests = run_tests()
    before = protected_snapshot()
    work = Path(tempfile.mkdtemp(prefix=".g7a_work.", dir=OUTPUT.parent))
    protected_path = work / "protected_before.json"
    atomic_json(protected_path, before)
    temp_checkpoint = work / "g7a_critic_warmup_checkpoint.development"
    fixed_path = work / "fixed_diagnostics.pt"
    warmup_result_path = work / "warmup_result.json"
    verify_result_path = work / "fresh_load_result.json"
    environment = worker_environment()
    try:
        warmup_log = work / "warmup_worker.log"
        with warmup_log.open("w", encoding="utf-8") as log:
            warmup_process = subprocess.Popen([
                sys.executable, "-m", WORKER_MODULE, "--mode", "warmup",
                "--checkpoint", str(temp_checkpoint), "--result", str(warmup_result_path),
                "--fixed-diagnostics", str(fixed_path),
                "--protected-snapshot", str(protected_path),
            ], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_process(warmup_process, warmup_log, "WARMUP")
        warmup = json.loads(warmup_result_path.read_text())
        require(warmup["counters"] == CRITIC_WARMUP_COUNTERS, "G7A_WARMUP_COUNTERS_INVALID")
        warmup_pid = warmup["environment"]["pid"]

        verify_log = work / "fresh_load_worker.log"
        with verify_log.open("w", encoding="utf-8") as log:
            verify_process = subprocess.Popen([
                sys.executable, "-m", WORKER_MODULE, "--mode", "verify",
                "--checkpoint", str(temp_checkpoint), "--result", str(verify_result_path),
            ], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_process(verify_process, verify_log, "FRESH_LOAD")
        fresh_load = json.loads(verify_result_path.read_text())
        require(fresh_load["strict_model_load"] and fresh_load["parameter_updates"] == 0, "G7A_FRESH_LOAD_INVALID")
        require(warmup_pid != fresh_load["environment"]["pid"], "G7A_FRESH_LOAD_PID_REUSED")
        validate_critic_warmup_checkpoint(temp_checkpoint)
        os.replace(temp_checkpoint, CHECKPOINT)
        os.replace(work, OUTPUT)
    except BaseException:
        failure = OUTPUT.parent / f"g7a_failed_{os.getpid()}"
        if work.exists():
            os.replace(work, failure)
        raise

    after = protected_snapshot()
    require(before == after, "G7A_FROZEN_INPUT_MUTATION")
    require(not torch.cuda.is_initialized(), "G7A_COORDINATOR_CREATED_CUDA_CONTEXT")
    warmup = json.loads((OUTPUT / "warmup_result.json").read_text())
    fresh_load = json.loads((OUTPUT / "fresh_load_result.json").read_text())
    gradient = warmup["gradient_scale"]
    acceptance = {
        "critic_optimizer_updates_256": warmup["counters"]["critic_optimizer_updates"] == 256,
        "critic_scheduler_steps_256": warmup["counters"]["critic_scheduler_steps"] == 256,
        "polyak_updates_each_target_256": warmup["counters"]["q1_target_polyak_updates"] == warmup["counters"]["q2_target_polyak_updates"] == 256,
        "actor_optimizer_scheduler_updates_zero": warmup["counters"]["actor_optimizer_updates"] == warmup["counters"]["actor_scheduler_steps"] == 0,
        "actor_parameters_and_buffers_bitwise_unchanged": warmup["state"]["actor"]["state_initial"] == warmup["state"]["actor"]["state_final"],
        "q1_q2_independent_and_updated": warmup["state"]["q1_q2_independent"],
        "targets_only_polyak": warmup["state"]["targets_changed_only_by_256_polyak_calls"],
        "frozen_backbones_unchanged": warmup["state"]["backbone_initial"] == warmup["state"]["backbone_final"],
        "terminal_mask_padding_k3x7_contract": all(item["terminal_next_actor_and_target_q_calls"] == 0 for item in warmup["warmup_updates"]),
        "all_values_finite": all(item["gradient"]["finite_before_and_after"] for item in warmup["warmup_updates"]),
        "sampler_rng_counter_checkpoint_self_consistent": warmup["checkpoint_save_rng_unchanged"],
        "fresh_process_strict_checkpoint_load": fresh_load["strict_model_load"] and fresh_load["rng_restored_last"],
        "frozen_sha_unchanged": before == after,
        "forbidden_data_access_zero": warmup["data_access_audit"]["test_transition_reads"] == warmup["data_access_audit"]["manual_g1_files_opened"] == warmup["data_access_audit"]["manual_label_files_opened"] == 0,
        "reward_classifier_access_zero": warmup["data_access_audit"]["reward_classifier_inference_calls"] == warmup["data_access_audit"]["reward_classifier_optimizer_updates"] == 0,
        "tcp6_q_gradient_nonzero": gradient["tcp6_q_gradient_nonzero_all_probes"],
        "gripper_q_gradient_zero": gradient["gripper_q_gradient_exact_zero_all_probes"],
        "gripper_fm_gradient_nonzero": gradient["gripper_flow_matching_gradient_nonzero_all_probes"],
        "validation_fixed_two_timepoints_only": set(warmup["evaluation"]) == {"update_0", "update_256"},
        "g7b_not_started": True,
    }
    require(all(acceptance.values()), f"G7A_ACCEPTANCE_FAILED:{acceptance}")
    artifact = {
        "schema_version": "forcesmolvla_s2_g7a_critic_warmup_preflight.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "G7A_CRITIC_WARMUP_MECHANICS": "pass",
        "CRITIC_WARMUP_UPDATES": 256,
        "ACTOR_UPDATES": 0,
        "CRITIC_NUMERICALLY_STABLE": "yes",
        "Q_GUIDANCE_SCALE_MEASURED": "yes",
        "ETA_G7B_APPROVED": "no", "G7B_STARTED": "no",
        "LONG_RUN_AUTHORIZED": "no", "ROBOT_EXECUTION_AUTHORIZED": False,
        "NEXT_ALLOWED_ACTION": "request_G7B_eta_and_joint_smoke_approval",
        "ALL_SUCCESS_DEMOS": True, "REWARD_MODEL_TRAINING_OVERLAP": True,
        "UNBIASED_POLICY_EVALUATION": False,
        "tests": tests, "protected_before": before, "protected_after": after,
        "warmup": warmup, "fresh_load": fresh_load,
        "acceptance": acceptance,
        "checkpoint": checkpoint_tree(CHECKPOINT),
        "output_tree": checkpoint_tree(OUTPUT),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "limits": warmup["research_limits"],
    }
    atomic_text(REPORT, report_markdown(artifact))
    artifact["report_sha256"] = sha256_file(REPORT)
    artifact["artifact_payload_sha256"] = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    atomic_json(ARTIFACT, artifact)
    print(json.dumps({
        "status": "pass", "warmup_pid": warmup["environment"]["pid"],
        "fresh_load_pid": fresh_load["environment"]["pid"],
        "checkpoint_tree_sha256": artifact["checkpoint"]["tree_sha256"],
        "median_raw_q_over_fm": gradient["global"]["raw_q_over_fm"]["median"],
        "eta_candidates_in_reference_band": gradient["candidates_with_median_in_reference_band"],
    }, sort_keys=True))


def production_main() -> None:
    """Run the canonical warmup and strict-load verification without reports."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run, "pass --run for authorized Twin-Q warmup")
    require(not torch.cuda.is_initialized(), "G7A_COORDINATOR_MUST_NOT_CREATE_CUDA_CONTEXT")
    require(CONFIG.is_file(), "G7A_CONFIG_MISSING")
    for target in (OUTPUT, CHECKPOINT):
        require(not target.exists(), f"G7A_APPEND_ONLY_TARGET_EXISTS:{target}")

    from forcesmolvla.rft.critic_warmup_checkpoint import (
        CRITIC_WARMUP_COUNTERS,
        validate_critic_warmup_checkpoint,
    )

    before = protected_snapshot()
    work = Path(tempfile.mkdtemp(prefix=".g7a_work.", dir=OUTPUT.parent))
    protected_path = work / "protected_before.json"
    atomic_json(protected_path, before)
    temp_checkpoint = work / "g7a_r2_critic_warmup_checkpoint"
    fixed_path = work / "fixed_diagnostics.pt"
    warmup_result_path = work / "warmup_result.json"
    verify_result_path = work / "fresh_load_result.json"
    environment = worker_environment()
    try:
        warmup_log = work / "warmup_worker.log"
        with warmup_log.open("w", encoding="utf-8") as log:
            warmup_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    WORKER_MODULE,
                    "--mode",
                    "warmup",
                    "--checkpoint",
                    str(temp_checkpoint),
                    "--result",
                    str(warmup_result_path),
                    "--fixed-diagnostics",
                    str(fixed_path),
                    "--protected-snapshot",
                    str(protected_path),
                ],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_process(warmup_process, warmup_log, "WARMUP")
        warmup = json.loads(warmup_result_path.read_text(encoding="utf-8"))
        require(warmup["counters"] == CRITIC_WARMUP_COUNTERS, "G7A_WARMUP_COUNTERS_INVALID")

        verify_log = work / "fresh_load_worker.log"
        with verify_log.open("w", encoding="utf-8") as log:
            verify_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    WORKER_MODULE,
                    "--mode",
                    "verify",
                    "--checkpoint",
                    str(temp_checkpoint),
                    "--result",
                    str(verify_result_path),
                ],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_process(verify_process, verify_log, "FRESH_LOAD")
        fresh_load = json.loads(verify_result_path.read_text(encoding="utf-8"))
        require(
            fresh_load["strict_model_load"]
            and fresh_load["parameter_updates"] == 0,
            "G7A_FRESH_LOAD_INVALID",
        )
        validate_critic_warmup_checkpoint(temp_checkpoint)
        os.replace(temp_checkpoint, CHECKPOINT)
        os.replace(work, OUTPUT)
    except BaseException:
        failure = OUTPUT.parent / f"g7a_failed_{os.getpid()}"
        if work.exists():
            os.replace(work, failure)
        raise

    require(before == protected_snapshot(), "G7A_FROZEN_INPUT_MUTATION")
    require(not torch.cuda.is_initialized(), "G7A_COORDINATOR_CREATED_CUDA_CONTEXT")
    print(
        json.dumps(
            {
                "status": "pass",
                "critic_optimizer_updates": 256,
                "actor_optimizer_updates": 0,
                "checkpoint": str(CHECKPOINT),
                "strict_load": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    production_main()
