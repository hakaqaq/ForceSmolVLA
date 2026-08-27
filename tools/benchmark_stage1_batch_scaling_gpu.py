#!/usr/bin/env python3
"""Append-only Stage-1 full-model physical-batch throughput benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).parents[1].resolve()
CONFIG = ROOT / "configs/stage1_batch_scaling.development.yaml"
R5 = ROOT / "outputs/development/task2_lerobotv3_full_sft_10k_r5/checkpoints/step_010000"
DATASET = ROOT / "datasets/task2_lerobotv3"
OUTPUT = ROOT / "artifacts/development/stage2/batch_scaling/stage1"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0 and np.isfinite(array).all(), "STAGE1_BENCHMARK_STAT_INVALID")
    return {
        "count": int(array.size), "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)), "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()), "maximum": float(array.max()),
        "range": float(array.max() - array.min()),
    }


class GpuTelemetry:
    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.utilization: list[float] = []
        self.power: list[float] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self):
        def collect() -> None:
            while not self.stop.is_set():
                completed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=False,
                )
                if completed.returncode == 0 and completed.stdout.strip():
                    try:
                        util, power = completed.stdout.strip().splitlines()[0].split(",", 1)
                        self.utilization.append(float(util.strip())); self.power.append(float(power.strip()))
                    except ValueError:
                        pass
                self.stop.wait(self.interval)

        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


def configure_runtime() -> torch.device:
    require(torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0), "STAGE1_BENCHMARK_RTX4090D_REQUIRED")
    require(os.environ.get("PYTHONHASHSEED") == "42", "STAGE1_BENCHMARK_PYTHONHASHSEED")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "STAGE1_BENCHMARK_CUBLAS")
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda:0")


def worker(candidate: dict) -> dict:
    from forcesmolvla.dataset_v3 import load_dataset_split
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.router_training import MoEMicrobatch, build_p7_optimizer_and_scheduler, single_pass_optimizer_update
    from forcesmolvla.training_data import load_runtime_artifacts, prepare_training_sample
    from preflight_p7_two_pass_gpu import _make_batch

    device = configure_runtime()
    batch_size = int(candidate["physical_batch_size"])
    repeat = int(candidate["repeat"])
    config = yaml.safe_load(CONFIG.read_text())
    require(batch_size in config["benchmark"]["mandatory_physical_batches"] + config["benchmark"]["conditional_physical_batches"], "STAGE1_BENCHMARK_BATCH_NOT_APPROVED")
    policy = ForceSmolVLAPolicy.from_pretrained(
        R5, local_files_only=True, force_download=False, strict=True, artifact_use="development"
    ).to(device)
    require(all(parameter.requires_grad for parameter in policy.parameters()), "STAGE1_BENCHMARK_FULL_TRAINABILITY_REQUIRED")
    optimizer, scheduler, ownership = build_p7_optimizer_and_scheduler(policy, derived_optimizer_updates=10000)
    conversion = json.loads((DATASET / "conversion_manifest.json").read_text())
    repo_id = conversion["repo_id"]
    dataset = load_dataset_split(
        DATASET, repo_id=repo_id, split_name="train", artifact_use="development",
        delta_timestamps={"action": [index / 30 for index in range(50)]},
    )
    runtime = load_runtime_artifacts(
        DATASET,
        calibration_bundle_path=ROOT / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=ROOT / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=ROOT / "artifacts/development/action_delta_spec.json",
        expected_repo_id=repo_id,
    )
    generator = random.Random(42)
    sequence = [generator.randrange(len(dataset)) for _ in range(21 * 32)]
    windows = [sequence[index * 32:index * 32 + batch_size] for index in range(21)]

    def prepare(index: int) -> dict:
        return prepare_training_sample(dataset[index], runtime.normalizer)

    reports = []
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="stage1-batch-benchmark") as pool:
        futures = [pool.submit(prepare, index) for index in windows[0]]
        telemetry = None
        for update in range(21):
            update_started = time.perf_counter()
            wait_started = time.perf_counter()
            prepared = [future.result() for future in futures]
            data_wait = time.perf_counter() - wait_started
            if update < 20:
                futures = [pool.submit(prepare, index) for index in windows[update + 1]]
            batch_started = time.perf_counter()
            batch = _make_batch(policy, prepared, device)
            noise = torch.randn(batch_size, 50, 7, device=device, dtype=torch.float32)
            timestep = policy.model.sample_time(batch_size, device)
            microbatch = MoEMicrobatch(batch=batch, noise7=noise, time=timestep, identity=f"stage1-b{batch_size}-repeat{repeat}-update{update}")
            torch.cuda.synchronize()
            batch_prepare = time.perf_counter() - batch_started
            if update == 1:
                torch.cuda.reset_peak_memory_stats(device)
                telemetry = GpuTelemetry().__enter__()
            train_started = time.perf_counter()
            policy.train(True)
            report = single_pass_optimizer_update(policy, microbatch, optimizer, scheduler=scheduler, grad_clip_norm=10.0)
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - train_started
            update_seconds = time.perf_counter() - update_started
            require(all(math.isfinite(float(report[key])) for key in (
                "backward_flow_sum", "backward_balance_sum", "backward_z_sum", "backward_total_sum", "gradient_norm_before_clip"
            )), "STAGE1_BENCHMARK_NONFINITE_LOSS_OR_GRADIENT")
            if update > 0:
                reports.append({
                    "timed_update": update,
                    "sample_indices": windows[update],
                    "data_loading_seconds": data_wait,
                    "batch_prepare_host_to_device_seconds": batch_prepare,
                    "training_forward_backward_optimizer_seconds": train_seconds,
                    "update_seconds": update_seconds,
                    "loss": {name: float(report[key]) for name, key in (
                        ("flow", "backward_flow_sum"), ("balance", "backward_balance_sum"),
                        ("z", "backward_z_sum"), ("total", "backward_total_sum"),
                    )},
                    "gradient_norm_before_clip": float(report["gradient_norm_before_clip"]),
                })
            del batch, microbatch, prepared
        require(telemetry is not None, "STAGE1_BENCHMARK_TELEMETRY_NOT_STARTED")
        telemetry.__exit__(None, None, None)

    require(len(reports) == 20, "STAGE1_BENCHMARK_TIMED_UPDATE_COUNT")
    require(all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()), "STAGE1_BENCHMARK_NONFINITE_PARAMETER")
    elapsed = sum(item["update_seconds"] for item in reports)
    return {
        "schema_version": "forcesmolvla_stage1_batch_candidate.v1",
        "status": "pass", "candidate_id": candidate["candidate_id"],
        "physical_batch_size": batch_size, "repeat": repeat, "pid": os.getpid(),
        "parent_checkpoint": R5.relative_to(ROOT).as_posix(),
        "same_fixed_initial_weights": True, "fixed_data_order_seed": 42,
        "warmup_updates": 1, "timed_updates": 20,
        "optimizer_updates_total_discarded": 21,
        "samples_per_second": 20 * batch_size / elapsed,
        "seconds_per_update": describe([item["update_seconds"] for item in reports]),
        "data_loading_seconds": describe([item["data_loading_seconds"] for item in reports]),
        "batch_prepare_host_to_device_seconds": describe([item["batch_prepare_host_to_device_seconds"] for item in reports]),
        "training_forward_backward_optimizer_seconds": describe([item["training_forward_backward_optimizer_seconds"] for item in reports]),
        "gpu_utilization_percent": describe(telemetry.utilization or [0.0]),
        "gpu_power_watts": describe(telemetry.power or [0.0]),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "total_gpu_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "all_finite": True, "contract_valid": True,
        "all_parameters_trainable": True, "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
        "optimizer_ownership": ownership,
        "lr_sequence": [float(optimizer.param_groups[0]["lr"])],
        "linear_lr_scaling_used": False,
        "timed_update_records": reports,
        "checkpoint_created": False, "candidate_state_discarded": True,
        "validation_reads": 0, "test_reads": 0,
        "robot_execution_authorized": False,
    }


def run_worker(candidate: dict, result_path: Path) -> dict:
    config_path = result_path.with_suffix(".config.json")
    atomic_json(config_path, candidate)
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT}",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    })
    completed = subprocess.run(
        [sys.executable, __file__, "--worker", "--candidate", str(config_path), "--result", str(result_path)],
        cwd=ROOT, env=environment, check=False,
    )
    require(completed.returncode == 0, f"STAGE1_BENCHMARK_WORKER_FAILED:{candidate['candidate_id']}:{completed.returncode}")
    return json.loads(result_path.read_text())


def coordinator() -> None:
    from forcesmolvla.rft.batch_scaling import aggregate_repeats, select_by_samples_per_second

    require(not OUTPUT.exists(), "STAGE1_BATCH_SCALING_OUTPUT_EXISTS")
    OUTPUT.mkdir(parents=True)
    config = yaml.safe_load(CONFIG.read_text())
    total_memory = int(subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]) * 1024 * 1024
    aggregates, raw_results = [], []

    def run_batch(batch: int) -> dict:
        repeats = []
        for repeat in range(1, 4):
            candidate = {"candidate_id": f"stage1_b{batch}_repeat{repeat}", "physical_batch_size": batch, "repeat": repeat}
            result_path = OUTPUT / "candidate_results" / f"{candidate['candidate_id']}.json"
            try:
                result = run_worker(candidate, result_path)
            except RuntimeError as error:
                if "WORKER_FAILED" not in str(error):
                    raise
                require(result_path.is_file(), f"STAGE1_BENCHMARK_WORKER_NO_RESULT:{candidate['candidate_id']}")
                result = json.loads(result_path.read_text())
            result["result_path"] = result_path.relative_to(ROOT).as_posix()
            repeats.append(result); raw_results.append(result)
            print(f"STAGE1_BATCH_RESULT B{batch} repeat={repeat} {result['status']}", flush=True)
        aggregate = aggregate_repeats(batch, repeats, "samples_per_second")
        aggregates.append(aggregate)
        return aggregate

    mandatory = [run_batch(int(batch)) for batch in config["benchmark"]["mandatory_physical_batches"]]
    b8, b16 = mandatory[1], mandatory[2]
    should_expand = bool(
        b16.get("all_pass") and b16.get("peak_reserved_bytes", total_memory + 1) <= 0.85 * total_memory
        and b16["samples_per_second"]["median"] >= 1.05 * b8["samples_per_second"]["median"]
    )
    if should_expand:
        for batch in config["benchmark"]["conditional_physical_batches"]:
            run_batch(int(batch))
    selected = select_by_samples_per_second(
        aggregates, "samples_per_second", total_memory_bytes=total_memory,
        maximum_fraction=0.85, minimum_gain=0.05,
    )
    summary = {
        "schema_version": "forcesmolvla_stage1_batch_scaling_summary.v1",
        "status": "pass", "config": CONFIG.relative_to(ROOT).as_posix(),
        "parent_checkpoint": R5.relative_to(ROOT).as_posix(),
        "candidate_aggregates": aggregates,
        "conditional_batches_executed": should_expand,
        "recommended_physical_batch": int(selected["physical_batch_size"]),
        "recommended_samples_per_second": float(selected["samples_per_second"]["median"]),
        "projected_40000_sample_runtime_seconds": 40000.0 / float(selected["samples_per_second"]["median"]),
        "total_gpu_memory_bytes": total_memory,
        "selection_rule": "valid_samples_per_second_with_5pct_incremental_gain_and_85pct_vram_limit",
        "checkpoint_count": 0, "stage1_retraining_performed": False,
    }
    atomic_json(OUTPUT / "stage1_summary.json", summary)
    print("STAGE1_BATCH_SCALING complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        require(args.candidate is not None and args.result is not None and not args.result.exists(), "STAGE1_WORKER_ARGUMENTS_INVALID")
        candidate = json.loads(args.candidate.read_text())
        try:
            result = worker(candidate)
        except BaseException as error:
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
            if not is_oom:
                raise
            result = {
                "schema_version": "forcesmolvla_stage1_batch_candidate.v1", "status": "oom",
                "candidate_id": candidate["candidate_id"], "physical_batch_size": candidate["physical_batch_size"],
                "repeat": candidate["repeat"], "error": str(error), "all_finite": False,
                "contract_valid": False, "checkpoint_created": False, "candidate_state_discarded": True,
            }
        atomic_json(args.result, result)
    else:
        require(args.run, "pass --run")
        coordinator()


if __name__ == "__main__":
    main()

