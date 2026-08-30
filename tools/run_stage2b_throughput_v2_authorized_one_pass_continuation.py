#!/usr/bin/env python3
"""Strict cycle-210 to cycle-420 continuation using the frozen v2 worker."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).parents[1].resolve()
CONFIG = ROOT / "configs/stage2b_long_run_one_pass_throughput_v2.authorized.yaml"
OUTPUT = ROOT / "artifacts/development/stage2/stage2b_throughput_v2_one_pass_continuation.v1"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
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
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def memory() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        values[name] = int(value.split()[0]) * 1024
    return {
        "mem_total_bytes": values["MemTotal"],
        "mem_available_bytes": values["MemAvailable"],
        "swap_free_bytes": values["SwapFree"],
    }


def process_rss(pid: int) -> int | None:
    path = Path(f"/proc/{pid}/status")
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


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


def main() -> None:
    from forcesmolvla.rft.exact_resume import checkpoint_tree
    from forcesmolvla.rft.long_run_checkpoint import validate_cycle_checkpoint

    config = yaml.safe_load(CONFIG.read_text())
    require(config["authorization"] == "yes_for_420_total_cycles", "ONE_PASS_NOT_AUTHORIZED")
    require(config["recipe"]["target_cycles"] == 420, "TARGET_CYCLE_DRIFT")
    require(config["recipe"]["continuation_start_cycle"] == 210, "RESUME_CYCLE_DRIFT")
    require(config["recipe"]["continuation_cycles"] == 210, "CONTINUATION_CYCLE_DRIFT")
    worker = ROOT / config["runtime"]["verified_worker"]
    require(sha256_file(worker) == config["runtime"]["verified_worker_sha256"], "VERIFIED_WORKER_SHA_DRIFT")
    half = config["half_pass_run"]
    require(sha256_file(ROOT / half["launcher"]) == half["launcher_sha256"], "HALF_PASS_LAUNCHER_SHA_DRIFT")
    require(sha256_file(ROOT / half["authorization_config"]) == half["authorization_config_sha256"], "HALF_PASS_CONFIG_SHA_DRIFT")
    checkpoint210 = ROOT / half["checkpoint"]
    boundary = json.loads((ROOT / half["boundary_audit"]).read_text())
    summary = json.loads((ROOT / half["completion_summary"]).read_text())
    require(boundary["cycle"] == 210, "CYCLE210_BOUNDARY_MISSING")
    require(boundary["frozen_parameter_hash_unchanged"], "CYCLE210_FROZEN_HASH_FAILED")
    require(boundary["action_contract_v2"], "CYCLE210_ACTION_CONTRACT_FAILED")
    require(boundary["all_losses_and_gradients_finite"], "CYCLE210_FINITE_GATE_FAILED")
    require(summary["status"] == "pass" and summary["completed_cycles"] == 210, "HALF_PASS_INCOMPLETE")
    require(summary["actor_exposure"] == 5040, "HALF_PASS_ACTOR_EXPOSURE_DRIFT")
    validate_cycle_checkpoint(checkpoint210, expected_cycle=210)
    checkpoint210_tree = checkpoint_tree(checkpoint210)
    startup_memory = memory()
    required = int(config["memory_gate"]["minimum_mem_available_gib"] * 1024**3)
    require(startup_memory["mem_available_bytes"] >= required, "INSUFFICIENT_MEM_AVAILABLE")
    require(not OUTPUT.exists(), "ONE_PASS_CONTINUATION_OUTPUT_EXISTS")
    OUTPUT.mkdir(parents=True)
    atomic_json(OUTPUT / "cycle210_resume_startup_audit.json", {
        "authorization_config": CONFIG.relative_to(ROOT).as_posix(),
        "authorization_config_sha256": sha256_file(CONFIG),
        "verified_worker": worker.relative_to(ROOT).as_posix(),
        "verified_worker_sha256": sha256_file(worker),
        "checkpoint210": checkpoint210.relative_to(ROOT).as_posix(),
        "checkpoint210_tree": checkpoint210_tree,
        "checkpoint210_boundary": boundary,
        "startup_memory": startup_memory,
        "required_mem_available_bytes": required,
        "memory_gate_pass": True,
        "strict_resume_cycle": 210,
        "first_post_resume_cycle": 211,
        "target_cycle": 420,
    })
    result = OUTPUT / "segment_cycle_210_420.json"
    checkpoint420 = OUTPUT / "checkpoint_cycle_000420"
    command = [
        sys.executable, str(worker), "--result", str(result),
        "--critic-batch", "64", "--start-cycle", "210",
        "--cycles", "210", "--warmup-cycles", "0",
        "--resume-checkpoint", str(checkpoint210),
        "--checkpoint-out", str(checkpoint420),
    ]
    process = subprocess.Popen(command, cwd=ROOT, env=environment())
    samples = []
    hard_rss = int(config["memory_gate"]["hard_process_rss_limit_gib"] * 1024**3)
    hard_available = int(config["memory_gate"]["hard_minimum_mem_available_during_run_gib"] * 1024**3)
    failure = None
    while process.poll() is None:
        snapshot = memory()
        rss = process_rss(process.pid)
        samples.append({"monotonic_seconds": time.monotonic(), "process_rss_bytes": rss, **snapshot})
        if rss is not None and rss > hard_rss:
            failure = "PROCESS_RSS_HARD_LIMIT"
        elif snapshot["mem_available_bytes"] < hard_available:
            failure = "MEM_AVAILABLE_HARD_LIMIT"
        if failure:
            atomic_json(OUTPUT / "fault_cycle_210_420.json", {
                "failure": failure, "pid": process.pid, "samples": samples,
                "parameter_changes_attempted": False,
            })
            process.terminate()
            break
        time.sleep(5)
    returncode = process.wait()
    monitor = {
        "segment": [210, 420], "pid": process.pid, "returncode": returncode,
        "failure": failure, "sample_count": len(samples),
        "peak_process_rss_bytes": max((item["process_rss_bytes"] or 0 for item in samples), default=0),
        "minimum_mem_available_bytes": min((item["mem_available_bytes"] for item in samples), default=0),
        "samples": samples,
    }
    atomic_json(OUTPUT / "memory_monitor_cycle_210_420.json", monitor)
    require(returncode == 0 and failure is None and result.is_file(), f"ONE_PASS_SEGMENT_FAILED:{returncode}:{failure}")
    payload = json.loads(result.read_text())
    require(
        payload["status"] == "pass"
        and payload["start_cycle"] == 210
        and payload["end_cycle"] == 420
        and payload["critic_batch"] == 64
        and payload["actor_batch"] == 24
        and payload["resume_audit"] is not None
        and payload["all_losses_and_gradients_finite"]
        and payload["action_contract_v2"]
        and payload["frozen_parameter_hash_unchanged"],
        "ONE_PASS_FINAL_AUDIT_FAILED",
    )
    validate_cycle_checkpoint(checkpoint420, expected_cycle=420)
    checkpoint420_tree = checkpoint_tree(checkpoint420)
    atomic_json(OUTPUT / "authorized_boundary_cycle_420.json", {
        "cycle": 420,
        "checkpoint": checkpoint420.relative_to(ROOT).as_posix(),
        "checkpoint_tree": checkpoint420_tree,
        "checkpoint_manifest_sha256": sha256_file(checkpoint420 / "checkpoint_manifest.json"),
        "frozen_parameter_hash_unchanged": True,
        "action_contract_v2": True,
        "all_losses_and_gradients_finite": True,
        "candidate_checkpoint_only_not_deployment": True,
        "auto_continue_beyond_cycle420": False,
    })
    atomic_json(OUTPUT / "training_complete_raw_summary.json", {
        "status": "pass",
        "completed_cycles": 420,
        "resume_boundary_cycle": 210,
        "first_post_resume_cycle": 211,
        "checkpoint210": checkpoint210.relative_to(ROOT).as_posix(),
        "checkpoint210_tree_sha256": checkpoint210_tree["tree_sha256"],
        "checkpoint420": checkpoint420.relative_to(ROOT).as_posix(),
        "checkpoint420_tree_sha256": checkpoint420_tree["tree_sha256"],
        "actor_exposure": 10080,
        "td_row_membership": 53760,
        "calql_row_membership": 53760,
        "auto_continue_beyond_cycle420": False,
        "validation_reads": 0,
        "test_reads": 0,
        "rollout_executed": False,
        "robot_execution_authorized": False,
    })
    print("STAGE2B_THROUGHPUT_V2_ONE_PASS_COMPLETE cycles=420", flush=True)


if __name__ == "__main__":
    main()
