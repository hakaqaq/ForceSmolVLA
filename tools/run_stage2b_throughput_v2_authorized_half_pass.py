#!/usr/bin/env python3
"""Thin authorization/monitor around the already verified throughput-v2 worker."""

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
CONFIG = ROOT / "configs/stage2b_long_run_half_pass_throughput_v2.authorized.yaml"
OUTPUT = ROOT / "artifacts/development/stage2/stage2b_throughput_v2_half_pass_run.v1"


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


def run_segment(
    *, worker: Path, start: int, end: int, result: Path,
    checkpoint: Path, resume: Path | None, config: dict,
) -> dict:
    command = [
        sys.executable, str(worker), "--result", str(result),
        "--critic-batch", "64", "--start-cycle", str(start),
        "--cycles", str(end - start), "--warmup-cycles", "0",
        "--checkpoint-out", str(checkpoint),
    ]
    if resume is not None:
        command.extend(("--resume-checkpoint", str(resume)))
    process = subprocess.Popen(command, cwd=ROOT, env=environment())
    samples = []
    hard_rss = int(config["memory_gate"]["hard_process_rss_limit_gib"] * 1024**3)
    hard_available = int(
        config["memory_gate"]["hard_minimum_mem_available_during_run_gib"] * 1024**3
    )
    failure = None
    while process.poll() is None:
        snapshot = memory()
        rss = process_rss(process.pid)
        samples.append({
            "monotonic_seconds": time.monotonic(),
            "process_rss_bytes": rss,
            **snapshot,
        })
        if rss is not None and rss > hard_rss:
            failure = "PROCESS_RSS_HARD_LIMIT"
        elif snapshot["mem_available_bytes"] < hard_available:
            failure = "MEM_AVAILABLE_HARD_LIMIT"
        if failure:
            atomic_json(OUTPUT / f"fault_cycle_{start}_{end}.json", {
                "failure": failure, "pid": process.pid, "samples": samples,
                "parameter_changes_attempted": False,
            })
            process.terminate()
            break
        time.sleep(5)
    returncode = process.wait()
    monitor = {
        "segment": [start, end],
        "pid": process.pid,
        "returncode": returncode,
        "failure": failure,
        "sample_count": len(samples),
        "peak_process_rss_bytes": max(
            (item["process_rss_bytes"] or 0 for item in samples), default=0
        ),
        "minimum_mem_available_bytes": min(
            (item["mem_available_bytes"] for item in samples), default=0
        ),
        "samples": samples,
    }
    atomic_json(OUTPUT / f"memory_monitor_cycle_{start}_{end}.json", monitor)
    require(returncode == 0 and failure is None and result.is_file(), f"AUTHORIZED_SEGMENT_FAILED:{start}:{end}:{returncode}:{failure}")
    payload = json.loads(result.read_text())
    require(
        payload["status"] == "pass"
        and payload["start_cycle"] == start
        and payload["end_cycle"] == end
        and payload["critic_batch"] == 64
        and payload["actor_batch"] == 24
        and payload["checkpoint"] is not None
        and payload["all_losses_and_gradients_finite"]
        and payload["action_contract_v2"]
        and payload["frozen_parameter_hash_unchanged"],
        f"AUTHORIZED_SEGMENT_AUDIT_FAILED:{start}:{end}",
    )
    from forcesmolvla.rft.g7_long_run import validate_cycle_checkpoint

    validate_cycle_checkpoint(checkpoint, expected_cycle=end)
    atomic_json(OUTPUT / f"authorized_boundary_cycle_{end}.json", {
        "cycle": end,
        "verified_worker_result": result.relative_to(ROOT).as_posix(),
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json"),
        "frozen_parameter_hash_unchanged": True,
        "action_contract_v2": True,
        "all_losses_and_gradients_finite": True,
        "training_execution_authorized": True,
        "candidate_checkpoint_only_not_deployment": True,
    })
    return payload


def main() -> None:
    from forcesmolvla.rft.exact_resume import checkpoint_tree

    config = yaml.safe_load(CONFIG.read_text())
    require(config["authorization"] == "yes_for_210_cycles_only", "LONG_RUN_NOT_AUTHORIZED")
    worker = ROOT / config["runtime"]["verified_worker"]
    require(sha256_file(worker) == config["runtime"]["verified_worker_sha256"], "VERIFIED_WORKER_SHA_DRIFT")
    parent = ROOT / config["parent"]["path"]
    parent_tree = checkpoint_tree(parent)
    require(parent_tree["tree_sha256"] == config["parent"]["tree_sha256"], "PARENT_TREE_SHA_DRIFT")
    startup_memory = memory()
    required = int(config["memory_gate"]["minimum_mem_available_gib"] * 1024**3)
    require(startup_memory["mem_available_bytes"] >= required, "INSUFFICIENT_MEM_AVAILABLE")
    require(not OUTPUT.exists(), "AUTHORIZED_LONG_RUN_OUTPUT_EXISTS")
    OUTPUT.mkdir(parents=True)
    atomic_json(OUTPUT / "cycle0_startup_audit.json", {
        "authorization_config": CONFIG.relative_to(ROOT).as_posix(),
        "authorization_config_sha256": sha256_file(CONFIG),
        "verified_worker": worker.relative_to(ROOT).as_posix(),
        "verified_worker_sha256": sha256_file(worker),
        "parent": parent.relative_to(ROOT).as_posix(),
        "parent_tree": parent_tree,
        "startup_memory": startup_memory,
        "required_mem_available_bytes": required,
        "memory_gate_pass": True,
        "fresh_parent_only": True,
        "C128_EQUIVALENT_INFRA_SPEEDUP": "2.20x",
        "C64_TOTAL_WALL_CLOCK_SPEEDUP": "4.28x",
        "C64_SEMANTICALLY_IDENTICAL_TO_C128": "no",
        "C64_STATUS": "approved_long_run_hyperparameter",
        "long_run_authorized_cycles": 210,
        "auto_continue_to_1_pass": False,
    })
    cycle105 = OUTPUT / "checkpoint_cycle_000105"
    segment1 = run_segment(
        worker=worker, start=0, end=105,
        result=OUTPUT / "segment_cycle_000_105.json",
        checkpoint=cycle105, resume=None, config=config,
    )
    cycle210 = OUTPUT / "checkpoint_cycle_000210"
    segment2 = run_segment(
        worker=worker, start=105, end=210,
        result=OUTPUT / "segment_cycle_105_210.json",
        checkpoint=cycle210, resume=cycle105, config=config,
    )
    atomic_json(OUTPUT / "training_complete_raw_summary.json", {
        "status": "pass",
        "completed_cycles": 210,
        "segment1_training_state_digest": segment1["training_state"]["training_state_digest"],
        "segment2_training_state_digest": segment2["training_state"]["training_state_digest"],
        "cycle105_checkpoint": cycle105.relative_to(ROOT).as_posix(),
        "cycle210_checkpoint": cycle210.relative_to(ROOT).as_posix(),
        "actor_exposure": 5040,
        "td_row_membership": 26880,
        "calql_row_membership": 26880,
        "auto_continue_to_1_pass": False,
        "validation_reads": 0,
        "test_reads": 0,
        "robot_execution_authorized": False,
    })
    print("STAGE2B_THROUGHPUT_V2_HALF_PASS_COMPLETE cycles=210", flush=True)


if __name__ == "__main__":
    main()
