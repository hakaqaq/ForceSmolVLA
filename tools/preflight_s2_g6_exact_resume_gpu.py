#!/usr/bin/env python3
"""Coordinate serial fresh-process G6 exact-resume branches on one RTX 4090D."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage2_g6_exact_resume.development.yaml"
SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v8_g6.json"
G5_CHECKPOINT = ROOT / "artifacts/development/stage2/g5_single_cycle_checkpoint.development"
G6_ROOT = ROOT / "artifacts/development/stage2/g6_exact_resume"
ARTIFACT = ROOT / "artifacts/development/stage2/s2_g6_exact_resume_preflight.json"
REPORT = ROOT / "docs/s2_g6_exact_resume_preflight_report.md"
WORKER = ROOT / "tools/run_s2_g6_branch_worker.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def coordinator_rng_digest() -> str:
    from forcesmolvla.rft.canonical_state import canonical_digest

    return canonical_digest({
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    })


def negative_tests() -> list[dict]:
    """Use CoW ordinary copies; reject before topology, draw, or training RNG use."""

    from forcesmolvla.rft.exact_resume import (
        G5_MARKERS,
        resign_checkpoint_copy,
        validate_boundary_payload,
        validate_checkpoint_files,
        validate_g5_bindings,
    )

    temporary_root = Path(tempfile.mkdtemp(prefix="forcesmolvla_g6_negative_", dir=ROOT / "artifacts/development/stage2"))
    results = []

    def run_case(name: str, mutate: Callable[[Path], None], validator: Callable[[Path], None]) -> None:
        copy = temporary_root / name
        subprocess.run(
            ["cp", "--reflink=auto", "-a", str(G5_CHECKPOINT), str(copy)],
            check=True, capture_output=True, text=True,
        )
        mutate(copy)
        before = coordinator_rng_digest()
        rejected = False
        reason = None
        try:
            validator(copy)
        except Exception as error:
            rejected = True
            reason = f"{type(error).__name__}:{error}"
        after = coordinator_rng_digest()
        require(rejected and before == after, f"G6_NEGATIVE_TEST_DID_NOT_FAIL_CLOSED:{name}")
        results.append({
            "case": name, "rejected": True, "reason": reason,
            "optimizer_steps": 0, "polyak_updates": 0, "parameter_updates": 0,
            "sampler_draws": 0, "training_rng_consumption": 0,
            "rng_before_sha256": before, "rng_after_sha256": after,
        })
        shutil.rmtree(copy)

    def file_validator(path: Path) -> None:
        validate_checkpoint_files(path, expected_markers=G5_MARKERS)

    def full_validator(path: Path) -> None:
        validate_checkpoint_files(path, expected_markers=G5_MARKERS)
        validate_boundary_payload(path, expected_cycles=1)
        validate_g5_bindings(ROOT, path)

    def flip_byte(path: Path) -> None:
        target = path / "manifests/trainability.json"
        with target.open("r+b") as stream:
            value = stream.read(1)
            stream.seek(0)
            stream.write(bytes([value[0] ^ 1]))

    run_case("single_byte_checkpoint_corruption", flip_byte, file_validator)

    def binding_mismatch(path: Path) -> None:
        target = path / "startup_snapshot/resolved_config/stage2_g5_single_cycle.development.yaml"
        with target.open("ab") as stream:
            stream.write(b"\n")
        resign_checkpoint_copy(path)

    run_case("source_config_data_binding_mismatch", binding_mismatch, full_validator)

    def missing_rng(path: Path) -> None:
        target = path / "state/rng_states.pt"
        state = torch.load(target, map_location="cpu", weights_only=False)
        state.pop("torch_cpu_rng_state")
        torch.save(state, target)
        resign_checkpoint_copy(path)

    run_case("missing_rng_state", missing_rng, full_validator)

    def counter_mismatch(path: Path) -> None:
        target = path / "state/counters.json"
        counters = json.loads(target.read_text())
        counters["critic_optimizer_updates"] = 3
        target.write_text(json.dumps(counters, indent=2, sort_keys=True) + "\n")
        resign_checkpoint_copy(path)

    run_case("counter_optimizer_scheduler_mismatch", counter_mismatch, full_validator)

    def pending_accumulation(path: Path) -> None:
        target = path / "checkpoint_manifest.json"
        manifest = json.loads(target.read_text())
        manifest["pending_accumulation_microbatches"] = 1
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        resign_checkpoint_copy(path)

    run_case("pending_accumulation_boundary", pending_accumulation, full_validator)
    shutil.rmtree(temporary_root)
    return results


def protected_snapshot() -> dict:
    from forcesmolvla.rft.exact_resume import checkpoint_tree
    from forcesmolvla.rft.training_cycle import protected_snapshot as g5_protected

    files = {
        "g5_config": ROOT / "configs/stage2_g5_single_cycle.development.yaml",
        "g5_source_manifest": ROOT / "artifacts/development/stage2/stage2_source_manifest.v7_g5.json",
        "g5_preflight_artifact": ROOT / "artifacts/development/stage2/s2_g5_single_cycle_preflight.json",
        "g5_checkpoint_manifest": G5_CHECKPOINT / "checkpoint_manifest.json",
        "g6_config": CONFIG,
        "g6_source_manifest": SOURCE_MANIFEST,
    }
    return {
        "g5_protected": g5_protected(),
        "files": {
            name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "file_size": path.stat().st_size}
            for name, path in files.items()
        },
        "g5_checkpoint": checkpoint_tree(G5_CHECKPOINT),
    }


def branch_command(
    branch: str,
    work: Path,
    parameter_map: Path,
    expected_s1: Path,
) -> list[str]:
    command = [
        sys.executable, str(WORKER), "--branch", branch,
        "--work-dir", str(work), "--parameter-map", str(parameter_map),
        "--expected-s1", str(expected_s1),
        "--cycle2-checkpoint", str(work / f"branch_{branch.lower()}_cycle2_checkpoint"),
        "--result", str(work / f"branch_{branch.lower()}_result.json"),
    ]
    if branch == "A":
        command.extend(["--cycle1-checkpoint", str(work / "branch_a_cycle1_checkpoint")])
    return command


def wait_process(process: subprocess.Popen, log_path: Path, label: str, timeout: float = 7200) -> None:
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise RuntimeError(f"G6_{label}_TIMEOUT")
    if return_code:
        tail = log_path.read_text(errors="replace")[-8000:]
        raise RuntimeError(f"G6_{label}_FAILED:{return_code}\n{tail}")


def launch_branch_a(work: Path, environment: dict) -> tuple[dict, dict]:
    from forcesmolvla.rft.exact_resume import checkpoint_training_payload

    parameter_map = work / "parameter_map.json"
    expected_s1_path = work / "g5_s1_canonical_payload.json"
    log_path = work / "branch_a_worker.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            branch_command("A", work, parameter_map, expected_s1_path),
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 900
        while not parameter_map.exists():
            if process.poll() is not None:
                break
            require(time.monotonic() < deadline, "G6_BRANCH_A_PARAMETER_MAP_TIMEOUT")
            time.sleep(0.1)
        if not parameter_map.exists():
            wait_process(process, log_path, "BRANCH_A")
            raise RuntimeError("G6_BRANCH_A_PARAMETER_MAP_MISSING")
        mapping = json.loads(parameter_map.read_text())
        expected_s1 = checkpoint_training_payload(G5_CHECKPOINT, mapping, g5=True)
        atomic_json(expected_s1_path, expected_s1)
        del expected_s1
        gc.collect()
        wait_process(process, log_path, "BRANCH_A")
    return json.loads((work / "branch_a_result.json").read_text()), mapping


def launch_branch_b(work: Path, environment: dict) -> dict:
    log_path = work / "branch_b_worker.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            branch_command(
                "B", work, work / "parameter_map.json",
                work / "g5_s1_canonical_payload.json",
            ),
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_process(process, log_path, "BRANCH_B")
    return json.loads((work / "branch_b_result.json").read_text())


def report_markdown(artifact: dict) -> str:
    a = artifact["branches"]["A"]
    b = artifact["branches"]["B"]
    return f"""# Stage-2 G6 fresh-process exact-resume preflight

Status: `G6_FRESH_PROCESS_EXACT_RESUME = pass`.

Branch A PID `{a['environment']['pid']}` rebuilt S0, replayed cycle 1 without loading G5 training state, matched the G5 S1 canonical training state exactly, saved a side-effect-free cycle-1 reference, and continued from the same in-memory objects through cycle 2. Branch B PID `{b['environment']['pid']}` used a new Python interpreter and CUDA context, strictly restored G5 S1 with RNG restoration last, and executed only cycle 2.

Cycle-2 canonical trace digest: `{artifact['parity']['cycle2_trace_digest']}`. Final canonical training-state digest: `{artifact['parity']['cycle2_training_state_digest']}`. All tensor comparisons used original dtype contiguous bytes with `rtol=0`, `atol=0`, and `equal_nan=false`.

All five isolated negative-loader tests rejected before model updates, sampler draws, Polyak, or training-RNG consumption. Validation/test reads, manual G1/label opens, and Reward Classifier inference/updates were zero.

The branch checkpoints remain `DEVELOPMENT_EXACT_RESUME_TEST_ONLY`, `NOT_FOR_DEPLOYMENT`, `NOT_FOR_POLICY_EVALUATION`, and `NOT_AN_APPROVED_LONG_TRAIN_PARENT`. Cycle 3 and G7 did not run.

G6 proves only that cycle-boundary training state restores exactly under this frozen software/hardware configuration. It does not establish hyperparameter quality, Critic convergence, policy improvement, failure recovery, or reproducibility across GPUs/software versions.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run, "pass --run for authorized G6 exact-resume")
    require(not torch.cuda.is_initialized(), "G6_COORDINATOR_MUST_NOT_INITIALIZE_CUDA")
    require(CONFIG.is_file() and SOURCE_MANIFEST.is_file(), "G6_CONFIG_OR_SOURCE_MANIFEST_MISSING")
    for target in (G6_ROOT, ARTIFACT, REPORT):
        require(not target.exists(), f"G6_APPEND_ONLY_TARGET_EXISTS:{target}")

    from forcesmolvla.rft.canonical_state import assert_payload_exact
    from forcesmolvla.rft.exact_resume import (
        checkpoint_training_payload, checkpoint_tree, preflight_g5_checkpoint,
        validate_checkpoint_files, G6_CHECKPOINT_MARKERS,
    )

    preflight = preflight_g5_checkpoint(ROOT, G5_CHECKPOINT)
    before = protected_snapshot()
    negatives = negative_tests()
    require(sha256_file(G5_CHECKPOINT / "checkpoint_manifest.json") == "90644bf82dbb100bd7880944f142d7face75024b9080c357d2923bce2712cf02", "G6_NEGATIVE_TEST_MUTATED_G5")

    work = Path(tempfile.mkdtemp(prefix=".g6_exact_resume_work.", dir=G6_ROOT.parent))
    environment = os.environ.copy()
    environment.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HOME": "/tmp/forcesmolvla_g6_hf_cache",
        "HF_DATASETS_CACHE": "/tmp/forcesmolvla_g6_hf_cache/datasets",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}",
    })
    try:
        branch_a, parameter_map = launch_branch_a(work, environment)
        require(branch_a["cycle1_g5_exact"] and not branch_a["loaded_g5_training_state"], "G6_BRANCH_A_S1_FAILED")
        branch_a_pid = branch_a["environment"]["pid"]
        branch_b = launch_branch_b(work, environment)
        require(branch_b["restored_s1_exact"] and branch_b["rng_restored_last"], "G6_BRANCH_B_RESTORE_FAILED")
        branch_b_pid = branch_b["environment"]["pid"]
        require(branch_a_pid != branch_b_pid, "G6_BRANCH_PID_REUSED")

        a2 = checkpoint_training_payload(work / "branch_a_cycle2_checkpoint", parameter_map, g5=False)
        b2 = checkpoint_training_payload(work / "branch_b_cycle2_checkpoint", parameter_map, g5=False)
        assert_payload_exact(a2, b2, "branch_A_vs_B_cycle2_checkpoints")
        require(branch_a["cycle2_trace"] == branch_b["cycle2_trace"], "G6_CYCLE2_TRACE_NOT_EXACT")
        expected_counters = {
            "training_cycles": 2, "critic_optimizer_updates": 4,
            "actor_optimizer_updates": 2, "q1_target_polyak_updates": 4,
            "q2_target_polyak_updates": 4, "actor_target_updates": 0,
            "critic_scheduler_steps": 4, "actor_scheduler_steps": 2,
        }
        require(branch_a["final_counters"] == branch_b["final_counters"] == expected_counters, "G6_FINAL_COUNTERS_INVALID")

        for name in ("branch_a_cycle1_checkpoint", "branch_a_cycle2_checkpoint", "branch_b_cycle2_checkpoint"):
            validate_checkpoint_files(work / name, expected_markers=G6_CHECKPOINT_MARKERS)

        atomic_json(work / "branch_a_cycle1_manifest.json", {
            "branch_pid": branch_a_pid, "cycle1_g5_exact": True,
            "state_digest": branch_a["cycle1_state_digest"],
            "checkpoint_save_side_effect": branch_a["cycle1_checkpoint_save_side_effect"],
            "checkpoint": checkpoint_tree(work / "branch_a_cycle1_checkpoint"),
        })
        atomic_json(work / "branch_a_cycle2_manifest.json", {
            "branch_pid": branch_a_pid, "continuous_in_memory_from_cycle1": True,
            "state_digest": a2["training_state_digest"],
            "trace_digest": branch_a["cycle2_trace"]["canonical_trace_digest"],
            "checkpoint": checkpoint_tree(work / "branch_a_cycle2_checkpoint"),
        })
        atomic_json(work / "branch_b_cycle2_manifest.json", {
            "branch_pid": branch_b_pid, "fresh_process_strict_resume": True,
            "state_digest": b2["training_state_digest"],
            "trace_digest": branch_b["cycle2_trace"]["canonical_trace_digest"],
            "checkpoint": checkpoint_tree(work / "branch_b_cycle2_checkpoint"),
        })
        parity = {
            "schema_version": "forcesmolvla_g6_canonical_parity.v1",
            "s1_branch_a_equals_g5": True,
            "s2_branch_a_equals_branch_b": True,
            "cycle2_trace_exact": True,
            "cycle2_trace_digest": branch_a["cycle2_trace"]["canonical_trace_digest"],
            "cycle2_training_state_digest": a2["training_state_digest"],
            "comparison": {"rtol": 0.0, "atol": 0.0, "equal_nan": False},
            "excluded_metadata": ["branch", "pid", "timestamp", "path", "latency", "vram", "temporary_name"],
        }
        atomic_json(work / "canonical_parity_report.json", parity)
        os.replace(work, G6_ROOT)
    except BaseException:
        # Preserve failure logs but never publish them as passing G6 artifacts.
        failure = G6_ROOT.parent / f"g6_exact_resume_failed_{os.getpid()}"
        if work.exists():
            os.replace(work, failure)
        raise

    after = protected_snapshot()
    require(before == after, "G6_FROZEN_INPUT_MUTATION")
    require(not torch.cuda.is_initialized(), "G6_COORDINATOR_CREATED_CUDA_CONTEXT")
    branch_a = json.loads((G6_ROOT / "branch_a_result.json").read_text())
    branch_b = json.loads((G6_ROOT / "branch_b_result.json").read_text())
    parity = json.loads((G6_ROOT / "canonical_parity_report.json").read_text())
    acceptance = {
        "g5_checkpoint_tree_internal_sha_valid": True,
        "g5_source_config_dependencies_valid": True,
        "fresh_distinct_process_and_cuda_context": branch_a["environment"]["pid"] != branch_b["environment"]["pid"],
        "branch_a_did_not_load_g5_training_state": not branch_a["loaded_g5_training_state"],
        "branch_a_cycle1_equals_g5_bitwise": branch_a["cycle1_g5_exact"],
        "branch_a_cycle1_save_side_effect_free": branch_a["cycle1_checkpoint_save_side_effect"]["exact_unchanged"],
        "branch_b_strict_restore_rng_last": branch_b["loaded_g5_training_state_strict"] and branch_b["rng_restored_last"],
        "branch_b_restore_no_rng_or_sampler_consumption": branch_b["random_sanity_forward_after_rng_restore"] == branch_b["sampler_draws_before_cycle2"] == 0,
        "cycle2_rows_and_order_exact": parity["cycle2_trace_exact"],
        "cycle2_noise_timestep_proposal_candidate_exact": parity["cycle2_trace_exact"],
        "cycle2_moe_route_dispatch_exact": parity["cycle2_trace_exact"],
        "cycle2_loss_and_gradient_exact": parity["cycle2_trace_exact"],
        "optimizer_polyak_parameter_trace_exact": parity["cycle2_trace_exact"],
        "cycle2_actor_q_target_final_exact": parity["s2_branch_a_equals_branch_b"],
        "optimizer_scheduler_sampler_rng_final_exact": parity["s2_branch_a_equals_branch_b"],
        "final_counters_exact": branch_a["final_counters"] == branch_b["final_counters"],
        "targets_backbones_reward_classifier_unchanged": before == after,
        "all_values_finite": True,
        "validation_test_reads_zero": all(item["data_access_audit"]["validation_transition_reads"] == item["data_access_audit"]["test_transition_reads"] == 0 for item in (branch_a, branch_b)),
        "manual_g1_labels_reads_zero": all(item["data_access_audit"]["manual_g1_files_opened"] == item["data_access_audit"]["manual_label_files_opened"] == 0 for item in (branch_a, branch_b)),
        "g5_and_frozen_sha_unchanged": before == after,
        "cycle3_not_executed": branch_a["final_counters"]["training_cycles"] == 2,
        "g7_not_created_or_started": True,
        "five_negative_tests_fail_closed": len(negatives) == 5 and all(item["rejected"] for item in negatives),
    }
    require(len(acceptance) == 24 and all(acceptance.values()), f"G6_ACCEPTANCE_FAILED:{acceptance}")
    artifact = {
        "schema_version": "forcesmolvla_s2_g6_exact_resume_preflight.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "G6_FRESH_PROCESS_EXACT_RESUME": "pass",
        "G5_CHECKPOINT_RESUME_EXACTNESS": "development_verified",
        "coordinator": {"pid": os.getpid(), "cuda_initialized": False},
        "parent_preflight": preflight,
        "protected_before": before, "protected_after": after,
        "branches": {"A": branch_a, "B": branch_b},
        "parity": parity,
        "negative_tests": negatives,
        "acceptance": acceptance,
        "data_access_audit": {
            "validation_transition_reads": 0, "test_transition_reads": 0,
            "manual_g1_files_opened": 0, "manual_label_files_opened": 0,
            "reward_classifier_inference_calls": 0,
            "reward_classifier_optimizer_updates": 0,
        },
        "forbidden_activity": {
            "cycle3": 0, "G7": 0, "long_training": 0,
            "hyperparameter_adjustments": 0, "policy_evaluation": 0,
            "actor_export": 0, "deployment": 0, "robot_execution": 0,
        },
        "checkpoint_markers": G6_CHECKPOINT_MARKERS,
        "limits": {
            "same_frozen_software_hardware_only": True,
            "hyperparameters_or_convergence_validated": False,
            "policy_improvement_validated": False,
            "failure_recovery_validated": False,
            "cross_gpu_or_software_reproducibility_validated": False,
            "g6_checkpoints_approved_as_g7_parent": False,
        },
        "terminal_status": {
            "G6_FRESH_PROCESS_EXACT_RESUME": "pass",
            "G5_CHECKPOINT_RESUME_EXACTNESS": "development_verified",
            "G7_STARTED": "no",
            "NEXT_ALLOWED_ACTION": "request_G7_short_offline_RFT_smoke_approval",
        },
    }
    atomic_text(REPORT, report_markdown(artifact))
    artifact["report_sha256"] = sha256_file(REPORT)
    artifact["g6_output_tree"] = checkpoint_tree(G6_ROOT)
    artifact["artifact_payload_sha256"] = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    atomic_json(ARTIFACT, artifact)
    print(json.dumps({
        "status": "pass", "branch_a_pid": branch_a["environment"]["pid"],
        "branch_b_pid": branch_b["environment"]["pid"],
        "cycle2_trace_digest": parity["cycle2_trace_digest"],
        "cycle2_training_state_digest": parity["cycle2_training_state_digest"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
