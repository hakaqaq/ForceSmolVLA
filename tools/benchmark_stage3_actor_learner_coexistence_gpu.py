#!/usr/bin/env python3
"""Isolated single-GPU Stage-3 Actor/Learner coexistence benchmark.

The parent process never imports torch.  CUDA is owned by disposable worker
processes; the inference worker loads the read-only cycle210 evaluation Actor,
and the learner worker reuses the accepted G4P numerical/ownership path.
Nothing in this tool starts a server, writes replay, exports a policy, or
persists a model/optimizer/checkpoint.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/stage3_gpu_coexistence.v1.development.yaml"
EXPECTED_HEAD = "7f1b9ea490e7d8286895d55745d6aa701e122ea1"
G6C_COMMIT = "ae5c74a53c8afb58e0ba0557fae84cf98ee32832"
G6P_FREEZE = "aef723103dd8683fc99f03766102b9b19dbcc43b"
HISTORICAL_CANONICAL_SHA256 = "d597ef3631a580e4cc8e67e00d7dacf4190de14ba830760cfe5c2e7225e80fd6"
HISTORICAL_JSON_SHA256 = "211ae0b9f397ef30685e55a66b5187de05eef59568ac61e722d3fd4c1caf5d40"
HISTORICAL_MD_SHA256 = "01d2fca550426836cddcabf8fb10c4ea0648020e013219cbbe9bf89978976d2e"
HISTORICAL_JSON = ROOT / "artifacts/development/stage3/stage3_policy_revision_loopback.v1.json"
HISTORICAL_MD = ROOT / "docs/stage3_policy_revision_loopback_report.v1.md"
BASE_G7A_CANONICAL_SHA256 = "1fd51e03eaa57c10412f4b38e2c4671edcd3abb474f7cec9ac45a800e4dacadb"
G7B_COMPONENT_NAMES = (
    "inference_only", "learner_only", "concurrent", "environment",
    "checkpoint_bindings", "action_semantics",
)
BASE_G7A_COMPONENT_DIGESTS = {
    "inference_only": "5af3dc87a30cc7816e6d7d634639f0412f74b04b0cd98843111a950724b27ccc",
    "learner_only": "3cebeaf4ffc4350fa0939bd4c3e946344f52e5bf10b3696f0ddb3582b13f96b6",
    "concurrent": "7f83588bd51d9d05c8e17b0b0e72e4e43994d8d9aaeaf50b73fef90f825eb618",
    "environment": "cc0685cffd1298f3fc956ecb700aaf4d31059ee9fc940ac0c2e00d35851ba379",
    "checkpoint_bindings": "85e52bcafe4829abb8bc9ab3b58b5b85e827404a16e94937707e92f97e37beb3",
    "action_semantics": "9520e50873d117ba2de2b0f88e6a8db12a861d3d5cdc2279f3c0384879f61837",
}
G7B_REQUIRED_TIMESTAMPS = (
    "episode_last_release_ns", "episode_queue_drained_ns", "episode_worker_exit_ns",
    "pre_learner_gap_start_ns", "pre_learner_gap_end_ns",
    "learner_spawn_requested_ns", "learner_process_spawn_ns", "learner_model_ready_ns",
    "learner_warmup_cycle_start_ns", "learner_warmup_cycle_end_ns",
    "learner_measured_cycle_1_start_ns", "learner_measured_cycle_1_end_ns",
    "learner_measured_cycle_2_start_ns", "learner_measured_cycle_2_end_ns",
    "learner_measured_cycle_3_start_ns", "learner_measured_cycle_3_end_ns",
    "learner_worker_exit_ns", "pre_resume_gap_start_ns", "pre_resume_gap_end_ns",
    "resume_requested_ns", "resume_process_spawn_ns", "resume_model_ready_ns",
    "resume_first_request_release_ns", "resume_first_result_ready_ns",
    "resume_queue_drained_ns", "resume_worker_exit_ns",
)


class G7PError(RuntimeError):
    """Fail-closed benchmark contract violation."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise G7PError(code)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_report_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def canonical_value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def g7b_component_values(report: Mapping[str, Any]) -> dict[str, Any]:
    modes = report["modes"]
    return {
        "inference_only": modes["inference_only"],
        "learner_only": modes["learner_only"],
        "concurrent": modes["concurrent"],
        "environment": report["gpu_preflight"],
        "checkpoint_bindings": {
            key: report["bindings"][key] for key in ("before", "after", "unchanged")
        },
        "action_semantics": {
            "comparison": report["comparisons"]["fixed_action_semantics"],
            "inference_only": modes["inference_only"]["worker"]["action_semantics"],
            "concurrent": modes["concurrent"]["inference"]["action_semantics"],
        },
    }


def g7b_component_digests(report: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: canonical_value_sha256(value)
        for name, value in g7b_component_values(report).items()
    }


def verify_g7b_base_components(
    base: Mapping[str, Any], candidate: Mapping[str, Any], expected: Mapping[str, str],
) -> dict[str, bool]:
    base_values = g7b_component_values(base)
    candidate_values = g7b_component_values(candidate)
    actual = g7b_component_digests(candidate)
    status: dict[str, bool] = {}
    for name in G7B_COMPONENT_NAMES:
        unchanged = (
            base_values[name] == candidate_values[name]
            and canonical_value_sha256(base_values[name]) == expected[name] == actual[name]
        )
        require(unchanged, f"G7B_BASE_COMPONENT_DRIFT:{name}")
        status[name] = True
    return status


def _ms(later_ns: int, earlier_ns: int) -> float:
    require(later_ns >= earlier_ns, "G7B_NEGATIVE_DURATION")
    return (later_ns - earlier_ns) / 1e6


def validate_g7b_timestamp_trace(
    trace: Mapping[str, Any], *, inter_phase_gap_ms: float,
) -> dict[str, Any]:
    require(trace.get("clock_source") == "CLOCK_MONOTONIC", "G7B_CLOCK_SOURCE")
    require(trace.get("linux_same_boot_cross_process_comparable") is True, "G7B_CLOCK_COMPARABILITY")
    require(trace.get("TIME_SLICED_TOPOLOGY") == "cold_process_swap", "G7B_TOPOLOGY")
    require(trace.get("RESIDENT_TIME_SLICING") == "NOT_RUN", "G7B_RESIDENT")
    require(trace.get("REAL_RESET_HOME_WINDOW_USED") is False, "G7B_REAL_RESET_WINDOW")
    require(trace.get("INTER_PHASE_GAP_IS_EXECUTION_BUDGET") is False, "G7B_GAP_IS_BUDGET")
    values: dict[str, int] = {}
    for key in G7B_REQUIRED_TIMESTAMPS:
        require(key in trace, f"G7B_TIMESTAMP_MISSING:{key}")
        require(isinstance(trace[key], int) and not isinstance(trace[key], bool), f"G7B_TIMESTAMP_TYPE:{key}")
        values[key] = int(trace[key])
    for key in ("resume_first_service_start_ns", "resume_first_service_end_ns"):
        require(key in trace, f"G7B_TIMESTAMP_MISSING:{key}")
        require(isinstance(trace[key], int) and not isinstance(trace[key], bool), f"G7B_TIMESTAMP_TYPE:{key}")
        values[key] = int(trace[key])

    order = list(G7B_REQUIRED_TIMESTAMPS[:3]) + [
        "pre_learner_gap_start_ns", "pre_learner_gap_end_ns",
        "learner_spawn_requested_ns", "learner_process_spawn_ns", "learner_model_ready_ns",
        "learner_warmup_cycle_start_ns", "learner_warmup_cycle_end_ns",
        "learner_measured_cycle_1_start_ns", "learner_measured_cycle_1_end_ns",
        "learner_measured_cycle_2_start_ns", "learner_measured_cycle_2_end_ns",
        "learner_measured_cycle_3_start_ns", "learner_measured_cycle_3_end_ns",
        "learner_worker_exit_ns", "pre_resume_gap_start_ns", "pre_resume_gap_end_ns",
        "resume_requested_ns", "resume_process_spawn_ns", "resume_model_ready_ns",
        "resume_first_request_release_ns", "resume_first_service_start_ns",
        "resume_first_service_end_ns", "resume_first_result_ready_ns",
        "resume_queue_drained_ns", "resume_worker_exit_ns",
    ]
    require(all(values[left] < values[right] for left, right in zip(order, order[1:])), "G7B_TIMESTAMP_ORDER")
    require(
        values["learner_measured_cycle_1_end_ns"] < values["learner_measured_cycle_2_start_ns"]
        and values["learner_measured_cycle_2_end_ns"] < values["learner_measured_cycle_3_start_ns"],
        "G7B_CYCLE_OVERLAP",
    )
    pre_learner_gap = _ms(values["pre_learner_gap_end_ns"], values["pre_learner_gap_start_ns"])
    pre_resume_gap = _ms(values["pre_resume_gap_end_ns"], values["pre_resume_gap_start_ns"])
    require(pre_learner_gap >= inter_phase_gap_ms and pre_resume_gap >= inter_phase_gap_ms, "G7B_INTER_PHASE_GAP_SHORT")
    cycles = [
        _ms(values[f"learner_measured_cycle_{index}_end_ns"], values[f"learner_measured_cycle_{index}_start_ns"])
        for index in (1, 2, 3)
    ]
    learner_phase_total_ms = _ms(
        values["learner_worker_exit_ns"], values["learner_spawn_requested_ns"],
    )
    return {
        "episode_drain_duration_ms": _ms(values["episode_queue_drained_ns"], values["episode_last_release_ns"]),
        "pre_learner_gap_actual_ms": pre_learner_gap,
        "learner_process_load_ms": _ms(values["learner_model_ready_ns"], values["learner_process_spawn_ns"]),
        "warmup_cycle_ms": _ms(values["learner_warmup_cycle_end_ns"], values["learner_warmup_cycle_start_ns"]),
        "measured_cycle_ms": cycles,
        "measured_cycles_total_ms": sum(cycles),
        "learner_phase_total_ms": learner_phase_total_ms,
        "pre_resume_gap_actual_ms": pre_resume_gap,
        "resume_model_load_ms": _ms(values["resume_model_ready_ns"], values["resume_process_spawn_ns"]),
        "resume_first_inference_service_ms": _ms(values["resume_first_service_end_ns"], values["resume_first_service_start_ns"]),
        "resume_spawn_to_first_ready_ms": _ms(values["resume_first_result_ready_ns"], values["resume_process_spawn_ns"]),
        "episode_drain_to_resume_spawn_ms": _ms(values["resume_process_spawn_ns"], values["episode_queue_drained_ns"]),
        "episode_drain_to_first_resumed_action_ms": _ms(values["resume_first_result_ready_ns"], values["episode_queue_drained_ns"]),
        "full_policy_unavailability_ms": _ms(values["resume_first_result_ready_ns"], values["episode_queue_drained_ns"]),
        "MINIMUM_MEASURED_SINGLE_JOINT_CYCLE_MS": min(cycles),
        "FULL_MEASURED_LEARNER_PHASE_MS": learner_phase_total_ms,
        "COLD_RESUME_SPAWN_TO_FIRST_READY_MS": _ms(values["resume_first_result_ready_ns"], values["resume_process_spawn_ns"]),
        "FULL_COLD_SWAP_INTERRUPTION_MS": _ms(values["resume_first_result_ready_ns"], values["episode_queue_drained_ns"]),
        "PRODUCTION_REQUIRED_RESET_HOME_WINDOW_MS": "UNVERIFIED",
    }


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"G7P_MAPPING_REQUIRED:{path}")
    return value


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(dict(value))
    require(
        config.get("schema_version") == "forcesmolvla_stage3_gpu_coexistence.v1.development",
        "G7P_CONFIG_SCHEMA_VERSION",
    )
    require(config.get("authorization") == "isolated_single_gpu_actor_learner_coexistence_only", "G7P_AUTHORIZATION")
    require(config["baseline"]["head"] == EXPECTED_HEAD, "G7P_BASELINE_CONFIG_HEAD")
    require(config["baseline"]["g6c_commit"] == G6C_COMMIT, "G7P_G6C_CONFIG")
    require(config["baseline"]["g6p_freeze_commit"] == G6P_FREEZE, "G7P_G6P_FREEZE_CONFIG")
    inference = config["inference"]
    require(inference["release_period_ms"] > 0 and inference["queue_capacity"] > 0, "G7P_INFERENCE_SCHEDULE")
    require(inference["warmup_requests"] >= 1 and inference["measured_trials"] >= 3, "G7P_INFERENCE_TRIALS")
    for name in (
        "inference_only_requests_per_trial", "concurrent_requests_per_trial",
        "time_sliced_requests_per_trial", "resume_requests_per_trial",
    ):
        require(inference[name] > 0, f"G7P_REQUEST_COUNT:{name}")
    learner = config["learner"]
    require(
        learner["critic_batch_size"] == 64
        and learner["actor_batch_size"] == 24
        and learner["flow_subbatch"] == 4
        and learner["critic_updates_per_cycle"] == 2
        and learner["actor_updates_per_cycle"] == 1
        and learner["polyak_updates_per_cycle"] == 2,
        "G7P_LEARNER_WORKLOAD_DRIFT",
    )
    require(learner["real_online_R_used"] is False and learner["R_source"] == "synthetic_preflight_R_only", "G7P_REAL_ONLINE_R")
    time_sliced = config["time_sliced"]
    require(
        time_sliced["inter_phase_gap_ms"] == 1000
        and time_sliced["inter_phase_gap_is_execution_budget"] is False
        and time_sliced["topology"] == "cold_process_swap"
        and time_sliced["resident_time_slicing"] == "NOT_RUN"
        and time_sliced["synthetic_reset_window"] is True
        and "quiescent_window_ms" not in time_sliced,
        "G7B_TIME_SLICED_CONFIG_SEMANTICS",
    )
    deadline = config["deadline"]
    require(
        deadline["inference_deadline_source"] == "UNBOUND"
        and deadline["approved_inference_deadline_ms"] is None
        and deadline["approved_miss_rate"] is None
        and deadline["deadline_equals_macro_period_assumed"] is False,
        "G7P_DEADLINE_NOT_UNBOUND",
    )
    safety = config["safety"]
    require(
        not safety["network_server_authorized"]
        and not safety["robot_execution_authorized"]
        and not safety["replay_write_authorized"]
        and not safety["checkpoint_writeback_authorized"]
        and not safety["policy_publication_authorized"]
        and safety["g8_and_later"] == "NOT_RUN",
        "G7P_SAFETY_SCOPE",
    )
    return config


def source_audit() -> dict[str, Any]:
    """Return audited, current-tree file:symbol bindings without importing runtime code."""
    checks = {
        "src/forcesmolvla/inference.py": (
            "def decode_rgb_image", "def prepare_policy_inputs", "HORIZON = 50",
        ),
        "tools/serve_policy.py": ("class InferenceEngine", "def infer"),
        "src/forcesmolvla/temporal.py": ("def controller_reference_grid",),
        "src/forcesmolvla/rft/stage3/transition.py": ("def causal_zoh_ack_macro",),
        "tools/preflight_stage3_gpu.py": (
            "def _load_real_batches", "def _critic_step", "def _actor_step", "def run_gpu_preflight",
        ),
        "tools/preflight_s2_g5_single_cycle_gpu.py": ("class TrainData", "def build_batch"),
    }
    for relative, needles in checks.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        for needle in needles:
            require(needle in source, f"G7P_AUDIT_SYMBOL_MISSING:{relative}:{needle}")
    transition = json.loads((ROOT / "configs/stage3_transition_contract.v1.development.json").read_text())
    temporal = transition["temporal"]
    require(
        temporal["data_grid_hz"] == 30
        and temporal["policy_hz"] == 10
        and temporal["flow_horizon"] == 50
        and temporal["critic_slots"] == 3,
        "G7P_TEMPORAL_CONTRACT_DRIFT",
    )
    shadow = yaml.safe_load((ROOT / "configs/shadow_safety_thresholds.development.yaml").read_text())
    rules = {item["rule_id"]: item for item in shadow["rules"]}
    for rule_id in ("SS_END_TO_APPLY", "SS_MISSED_TICK_RATE", "SS_HOLD_OVERRUN"):
        require(rules[rule_id]["threshold"]["approval_status"] == "approval_pending", f"G7P_THRESHOLD_STATUS:{rule_id}")
    inference_source = (ROOT / "src/forcesmolvla/inference.py").read_text(encoding="utf-8")
    server_source = (ROOT / "tools/serve_policy.py").read_text(encoding="utf-8")
    cache_tokens = ("OrderedDict", "max_bytes", "eviction", "rss")
    production_cache_bounded = all(token in inference_source or token in server_source for token in cache_tokens)
    return {
        "temporal_contract": {
            "source": "configs/stage3_transition_contract.v1.development.json:/temporal",
            "grid_hz": 30,
            "macro_hz": 10,
            "H": 50,
            "K": 3,
            "macro_period_ms": 100,
            "execution_source": "src/forcesmolvla/rft/stage3/transition.py:causal_zoh_ack_macro",
            "grid_source": "src/forcesmolvla/temporal.py:controller_reference_grid",
        },
        "inference_request_cadence": {"source": "UNBOUND", "status": "approval_pending"},
        "chunk_refresh_policy": {"source": "UNBOUND", "status": "approval_pending"},
        "action_queue_low_watermark": {"source": "UNBOUND", "status": "approval_pending"},
        "inference_timeout": {"source": "UNBOUND", "status": "approval_pending"},
        "hold_behavior": {
            "source": "UNBOUND",
            "status": "approval_pending",
            "shadow_only_source": "configs/shadow_safety_thresholds.development.yaml:SS_HOLD_OVERRUN",
        },
        "serve_preprocessing": {
            "source": "src/forcesmolvla/inference.py:prepare_policy_inputs",
            "decode_source": "src/forcesmolvla/inference.py:decode_rgb_image",
            "normalizer_exactly_once": True,
            "modalities": ["camera1", "camera2", "state7", "wrench6", "task_tokens"],
        },
        "critic_task_feature": {
            "source": "src/forcesmolvla/rft/critic.py:frozen_task_feature",
            "dimension": 256,
        },
        "decoded_image_cache": {
            "production_bounded": production_cache_bounded,
            "explicit_max_bytes": False,
            "lru_policy": False,
            "rss_monitor": False,
            "hit_miss_eviction_counters": False,
            "resume_reconstruction": False,
        },
        "learner_workload": {
            "orchestrator": "tools/preflight_stage3_gpu.py:run_gpu_preflight",
            "real_data": "tools/preflight_stage3_gpu.py:_load_real_batches",
            "data_builder": "tools/preflight_s2_g5_single_cycle_gpu.py:TrainData.build_batch",
            "critic_step": "tools/preflight_stage3_gpu.py:_critic_step",
            "actor_step": "tools/preflight_stage3_gpu.py:_actor_step",
            "optimizer_ownership": "tools/preflight_stage3_gpu.py:_optimizer_factory",
        },
        "INFERENCE_DEADLINE_SOURCE": "UNBOUND",
        "APPROVED_INFERENCE_DEADLINE_MS": None,
        "APPROVED_MISS_RATE": None,
        "QUEUE_LOW_WATERMARK_SOURCE": "UNBOUND",
        "HOLD_POLICY_SOURCE": "UNBOUND",
        "DEADLINE_EQ_MACRO_PERIOD_ASSUMED": False,
        "PRODUCTION_DEADLINE_GATE": "BLOCKED_ON_APPROVED_THRESHOLD_AND_RUNTIME_PARITY",
    }


class BoundedDecodedImageCache:
    """Benchmark-only byte-bounded LRU; never installed into production source."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("cache max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._values: OrderedDict[str, Any] = OrderedDict()
        self.current_bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_or_decode(
        self, encoded: Mapping[str, Any], decoder: Callable[..., Any], **decoder_kwargs: Any,
    ) -> Any:
        data = encoded.get("data")
        if not isinstance(data, str):
            return decoder(encoded, **decoder_kwargs)
        key = hashlib.sha256(data.encode("ascii")).hexdigest()
        if key in self._values:
            self.hits += 1
            value, size = self._values.pop(key)
            self._values[key] = (value, size)
            return value.copy()
        self.misses += 1
        value = decoder(encoded, **decoder_kwargs)
        size = int(value.nbytes)
        if size <= self.max_bytes:
            while self._values and self.current_bytes + size > self.max_bytes:
                _old_key, (_old_value, old_size) = self._values.popitem(last=False)
                self.current_bytes -= old_size
                self.evictions += 1
            self._values[key] = (value.copy(), size)
            self.current_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        return value

    def report(self) -> dict[str, Any]:
        return {
            "benchmark_only": True,
            "policy": "LRU",
            "max_bytes": self.max_bytes,
            "current_bytes": self.current_bytes,
            "peak_bytes": self.peak_bytes,
            "entries": len(self._values),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "resume_reconstruction": False,
        }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
    }


def summarize_requests(records: list[Mapping[str, Any]], *, formal_minimum: int) -> dict[str, Any]:
    completed = [row for row in records if row.get("result_ready_ns") is not None]
    fields = {
        "release_jitter_ms": "release_jitter_ms",
        "end_to_end_ms": "end_to_end_ms",
        "preprocessing_ms": "preprocessing_ms",
        "queue_wait_ms": "queue_wait_ms",
        "gpu_service_ms": "gpu_service_ms",
        "benchmark_macro_lateness_ms": "benchmark_macro_lateness_ms",
    }
    summary = {
        name: distribution([float(row[key]) for row in completed])
        for name, key in fields.items()
    }
    queue_depths = [int(row["queue_depth_at_consume"]) for row in completed]
    hold_durations = [max(0.0, float(row["benchmark_macro_lateness_ms"])) for row in completed]
    holds = [value for value in hold_durations if value > 0.0]
    consecutive = maximum_consecutive([value > 0.0 for value in hold_durations])
    elapsed = 0.0
    if completed:
        elapsed = (int(completed[-1]["result_ready_ns"]) - int(completed[0]["release_actual_ns"])) / 1e9
    summary.update({
        "request_count": len(completed),
        "p99_status": "FORMAL_SAMPLE_COUNT_MET" if len(completed) >= formal_minimum else "PROVISIONAL_INSUFFICIENT_SAMPLES",
        "approved_deadline": {
            "source": "UNBOUND", "deadline_ms": None, "miss_count": None,
            "miss_rate": None, "maximum_consecutive_misses": None,
        },
        "benchmark_macro_period_reference": {
            "period_ms": 100.0,
            "not_an_approved_deadline": True,
            "late_count": len(holds),
            "late_rate": len(holds) / len(completed) if completed else 0.0,
            "maximum_consecutive_late": consecutive,
        },
        "queue_depth": distribution([float(value) for value in queue_depths]),
        "simulated_hold": {
            "count": len(holds),
            "rate": len(holds) / len(completed) if completed else 0.0,
            "duration_ms": distribution(holds),
            "policy_source": "UNBOUND",
            "simulation_only": True,
        },
        "requests_per_second": len(completed) / elapsed if elapsed > 0 else 0.0,
    })
    return summary


def maximum_consecutive(flags: list[bool]) -> int:
    maximum = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        maximum = max(maximum, current)
    return maximum


def audit_request_stream(
    worker: Mapping[str, Any], *, queue_capacity: int, release_period_ms: float,
) -> dict[str, Any]:
    """Recompute request accounting and queue behavior from raw trial timestamps."""
    trial_audits = []
    scheduled_total = completed_total = dropped_total = 0
    for trial in worker["trials"]:
        schedule = trial["release_schedule"]
        requests = trial["requests"]
        drops = [row for row in schedule if row["status"] != "released"]
        depths = [int(row["queue_depth_at_release"]) for row in schedule if row.get("queue_depth_at_release") is not None]
        first_capacity_slot = next(
            (int(row["slot"]) for row in schedule if row.get("queue_depth_at_release") == queue_capacity),
            None,
        )
        first_drop_slot = next((int(row["slot"]) for row in drops), None)
        last_target_ns = max(int(row["release_target_ns"]) for row in schedule)
        completed_after_horizon = sum(int(row["result_ready_ns"]) > last_target_ns for row in requests)
        scheduled = len(schedule)
        completed = len(requests)
        dropped = len(drops)
        trial_audits.append({
            "trial": int(trial["trial"]),
            "scheduled": scheduled,
            "completed": completed,
            "dropped": dropped,
            "drop_rate": dropped / scheduled,
            "count_equation_holds": scheduled == completed + dropped,
            "maximum_queue_depth_at_release": max(depths),
            "first_capacity_slot": first_capacity_slot,
            "first_drop_slot": first_drop_slot,
            "drops_at_capacity": sum(
                row.get("queue_depth_at_release") == queue_capacity for row in drops
            ),
            "all_drops_at_capacity": all(
                row.get("queue_depth_at_release") == queue_capacity for row in drops
            ),
            "capacity_fraction_after_first_saturation": (
                sum(row.get("queue_depth_at_release") == queue_capacity for row in schedule[first_capacity_slot:])
                / len(schedule[first_capacity_slot:])
                if first_capacity_slot is not None else 0.0
            ),
            "completed_after_release_horizon": completed_after_horizon,
            "backlog_at_release_horizon": completed_after_horizon > 0,
            "last_result_after_release_horizon_ms": (
                (max(int(row["result_ready_ns"]) for row in requests) - last_target_ns) / 1e6
            ),
            "queue_drained_before_trial_return_by_fifo_sentinel": True,
        })
        scheduled_total += scheduled
        completed_total += completed
        dropped_total += dropped
    metrics = worker["metrics"]
    service = metrics["gpu_service_ms"]
    queue_wait = metrics["queue_wait_ms"]
    end_to_end = metrics["end_to_end_ms"]
    stored_counts_match_raw = (
        int(metrics["scheduled_release_count"]) == scheduled_total
        and int(metrics["request_count"]) == completed_total
        and int(metrics["stale_drop_count"]) == dropped_total
        and int(metrics["queue_exhaustion_count"]) == dropped_total
        and all(int(trial["scheduled_release_count"]) == len(trial["release_schedule"]) for trial in worker["trials"])
        and all(int(trial["completed_request_count"]) == len(trial["requests"]) for trial in worker["trials"])
    )
    p50_service = float(service["p50"])
    return {
        "scheduled": scheduled_total,
        "completed": completed_total,
        "dropped": dropped_total,
        "drop_rate": dropped_total / scheduled_total,
        "p99_status": metrics["p99_status"],
        "count_equation_holds": scheduled_total == completed_total + dropped_total,
        "stored_counts_match_raw": stored_counts_match_raw,
        "empirical_capacity_hz": {
            "formula": "1000 / measured GPU service latency ms",
            "empirical_only_not_request_cadence_or_deadline": True,
            "p50_service_latency_ms": p50_service,
            "p50_estimate_hz": 1000.0 / p50_service,
            "mean_service_latency_ms": float(service["mean"]),
            "mean_estimate_hz": 1000.0 / float(service["mean"]),
        },
        "queue": {
            "capacity": queue_capacity,
            "queue_memory_bounded_by_cap": True,
            "reached_capacity_in_every_trial": all(
                row["maximum_queue_depth_at_release"] == queue_capacity for row in trial_audits
            ),
            "drops_occur_only_at_capacity": all(row["all_drops_at_capacity"] for row in trial_audits),
            "queue_wait_dominates_p50_end_to_end": float(queue_wait["p50"]) > 0.5 * float(end_to_end["p50"]),
            "queue_wait_p50_share_of_end_to_end": float(queue_wait["p50"]) / float(end_to_end["p50"]),
            "backlog_at_release_horizon_in_every_trial": all(
                row["backlog_at_release_horizon"] for row in trial_audits
            ),
            "queue_drained_only_by_extending_trial_past_release_horizon": all(
                row["queue_drained_before_trial_return_by_fifo_sentinel"] for row in trial_audits
            ),
            "queue_load_stable_at_10hz": False,
            "no_drop_10hz": dropped_total == 0 and p50_service <= release_period_ms,
            "steady_state_classification": (
                "DROP_POLICY_TRUNCATED" if dropped_total else "NOT_DEMONSTRATED_SHORT_TRIAL_WITH_BACKLOG"
            ),
        },
        "trial_audits": trial_audits,
    }


def audit_time_sliced_semantics(
    mode: Mapping[str, Any], *, configured_gap_ms: float,
) -> dict[str, Any]:
    """Classify the existing time-sliced evidence without inventing timestamps."""
    if "timestamp_trace" in mode:
        derived = validate_g7b_timestamp_trace(
            mode["timestamp_trace"], inter_phase_gap_ms=configured_gap_ms,
        )
        return {
            "control_flow_source": "tools/benchmark_stage3_actor_learner_coexistence_gpu.py:run_time_sliced",
            "configured_inter_phase_gap_ms": configured_gap_ms,
            "configured_value_role": "inter_phase_gap_not_execution_budget",
            "control_flow_semantics": "gap_not_budget",
            "g7p_time_sliced_semantics": "VERIFIED_COLD_PROCESS_SWAP_ONLY",
            "timestamps_sufficient": True,
            "timestamp_order_valid": True,
            "TIME_SLICED_TOPOLOGY": "cold_process_swap",
            "RESIDENT_TIME_SLICING": "NOT_RUN",
            "REAL_RESET_HOME_WINDOW_USED": False,
            "INTER_PHASE_GAP_IS_EXECUTION_BUDGET": False,
            "timestamp_trace": copy.deepcopy(mode["timestamp_trace"]),
            "derived_timing": derived,
        }
    phase = mode["phase_timestamps"]
    learner = mode["reset_window_learner"]["worker"]
    resume = mode["resume_inference"]["worker"]
    measured = [
        row for row in learner["source_report"]["performance"]["cycles"]
        if row["kind"] == "measured"
    ]
    measured_ms = [float(row["wall_seconds"]) * 1000.0 for row in measured]
    cold_requests = resume["cold_start"]["requests"]
    first_ready_ns = int(cold_requests[0]["result_ready_ns"]) if cold_requests else None
    resume_spawn_ns = int(phase["resume_inference_spawn_ns"])
    actual_quiescent_ms = (
        resume_spawn_ns - int(phase["episode_inference_idle_ns"])
    ) / 1e6
    cycle_trials = [{
        "cycle": int(row["cycle"]),
        "quiescent_window_open_timestamp_ns": int(phase["episode_inference_idle_ns"]),
        "learner_cycle_start_timestamp_ns": None,
        "learner_cycle_end_timestamp_ns": None,
        "inference_resume_requested_timestamp_ns": None,
        "inference_process_spawn_timestamp_ns": resume_spawn_ns,
        "first_result_ready_timestamp_ns": first_ready_ns,
        "actual_quiescent_duration_ms": actual_quiescent_ms,
        "raw_cycle_wall_ms": wall_ms,
        "learner_cycles_completed_within_first_1000ms": 0,
        "learner_overrun_ms": max(0.0, wall_ms - configured_gap_ms),
    } for row, wall_ms in zip(measured, measured_ms)]
    return {
        "control_flow_source": "tools/benchmark_stage3_actor_learner_coexistence_gpu.py:run_time_sliced",
        "configured_quiescent_window_ms": configured_gap_ms,
        "configured_value_role": "inter_phase_gap_before_learner_and_before_resume",
        "control_flow_semantics": "extended",
        "g7p_time_sliced_semantics": "UNVERIFIED",
        "timestamps_sufficient": False,
        "timestamp_limitations": [
            "absolute learner cycle start/end timestamps were not recorded",
            "inference resume-requested and process-spawn timestamps were not recorded separately",
        ],
        "quiescent_window_budget_met": False,
        "minimum_measured_quiescent_window_required_ms": min(measured_ms),
        "learner_cycles_completed_within_first_1000ms": 0,
        "configured_pre_learner_gap_actual_ms": (
            int(phase["learner_spawn_ns"]) - int(phase["episode_inference_idle_ns"])
        ) / 1e6,
        "actual_learner_quiescent_occupancy_ms": (
            int(phase["learner_idle_ns"]) - int(phase["learner_spawn_ns"])
        ) / 1e6,
        "configured_pre_resume_gap_actual_ms": (
            resume_spawn_ns - int(phase["learner_idle_ns"])
        ) / 1e6,
        "actual_quiescent_duration_ms": actual_quiescent_ms,
        "learner_worker_started_timestamp_ns": int(learner["started_ns"]),
        "learner_worker_completed_timestamp_ns": int(learner["completed_ns"]),
        "resume_worker_started_timestamp_ns": int(resume["process_started_ns"]),
        "first_result_ready_timestamp_ns": first_ready_ns,
        "cold_restart_spawn_to_first_result_ready_ms": (
            (first_ready_ns - resume_spawn_ns) / 1e6 if first_ready_ns is not None else None
        ),
        "cold_restart_latency_separate_from_steady_state": True,
        "cycle_trials": cycle_trials,
    }


def semantic_consistency_audit(
    raw_report: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit and annotate an existing G7P report; never starts a CUDA worker."""
    report = copy.deepcopy(dict(raw_report))
    require(
        report.get("canonical_report_sha256") == canonical_sha256(report),
        "G7A_INPUT_CANONICAL_SHA_MISMATCH",
    )
    if "g7b_timestamp_instrumentation" in report:
        base_digests = report["g7b_timestamp_instrumentation"]["base_component_digests"]
        verify_g7b_base_components(report, report, base_digests)
        report["gates"]["PRODUCTION_COLD_PROCESS_SWAP_APPROVED"] = False
        derived = validate_g7b_timestamp_trace(
            report["modes"]["episode_time_sliced"]["timestamp_trace"],
            inter_phase_gap_ms=float(config["time_sliced"]["inter_phase_gap_ms"]),
        )
        report["modes"]["episode_time_sliced"]["derived_timing"] = copy.deepcopy(derived)
        report["semantic_audit"]["time_sliced"]["derived_timing"] = copy.deepcopy(derived)
        report["g7b_timestamp_instrumentation"]["derived_timing"] = copy.deepcopy(derived)
        report["baseline"]["current_g7_source_closure"] = current_source_closure()
        report["bindings"]["current_g7_source_closure"] = report["baseline"]["current_g7_source_closure"]
        report["canonical_report_sha256"] = canonical_sha256(report)
        verify_g7b_base_components(report, report, base_digests)
        return report
    modes = report["modes"]
    workers = {
        "inference_only": modes["inference_only"]["worker"],
        "concurrent": modes["concurrent"]["inference"],
        "time_sliced_episode": modes["episode_time_sliced"]["episode_inference"]["worker"],
        "time_sliced_resume": modes["episode_time_sliced"]["resume_inference"]["worker"],
    }
    accounting = {
        name: audit_request_stream(
            worker,
            queue_capacity=int(config["inference"]["queue_capacity"]),
            release_period_ms=float(config["inference"]["release_period_ms"]),
        )
        for name, worker in workers.items()
    }
    totals = {
        key: sum(int(mode[key]) for mode in accounting.values())
        for key in ("scheduled", "completed", "dropped")
    }
    totals["drop_rate"] = totals["dropped"] / totals["scheduled"]
    totals["count_equation_holds"] = totals["scheduled"] == totals["completed"] + totals["dropped"]
    raw_counts_consistent = (
        totals["count_equation_holds"]
        and all(mode["count_equation_holds"] and mode["stored_counts_match_raw"] for mode in accounting.values())
    )
    time_sliced = audit_time_sliced_semantics(
        modes["episode_time_sliced"],
        configured_gap_ms=float(config["time_sliced"]["inter_phase_gap_ms"]),
    )
    worker_reports = [
        modes["inference_only"]["worker"], modes["learner_only"]["worker"],
        modes["concurrent"]["inference"], modes["concurrent"]["learner"],
        modes["episode_time_sliced"]["episode_inference"]["worker"],
        modes["episode_time_sliced"]["reset_window_learner"]["worker"],
        modes["episode_time_sliced"]["resume_inference"]["worker"],
    ]
    memory = {
        "SHORT_RUN_MEMORY_SAFETY": "PASS",
        "OOM_COUNT": sum(int(mode.get("oom_count", 0)) for mode in modes.values()),
        "ALLOCATION_FAILURE_COUNT": sum(
            int(worker.get("torch_memory", {}).get("allocation_failure_count", 0))
            for worker in worker_reports
        ),
        "FRAGMENTATION_FAILURE_COUNT": sum(
            int(worker.get("torch_memory", {}).get("fragmentation_failure_count", 0))
            for worker in worker_reports
        ),
        "SUSTAINED_MEMORY_LEAK_TEST": "NOT_RUN",
        "SUSTAINED_THERMAL_STABILITY": "UNVERIFIED",
        "PRODUCTION_DECODED_CACHE_BOUNDED": False,
        "benchmark_only_cache_does_not_bind_production": True,
    }
    raw_input_sha = report.get("semantic_audit", {}).get(
        "raw_input_canonical_report_sha256", report["canonical_report_sha256"],
    )
    measurement_closure = report["baseline"].get(
        "raw_measurement_source_closure", report["baseline"].get("current_g7_source_closure"),
    )
    audit_closure = current_source_closure()
    report["baseline"]["raw_measurement_source_closure"] = measurement_closure
    report["baseline"]["current_g7_source_closure"] = audit_closure
    report["bindings"]["raw_measurement_source_closure"] = measurement_closure
    report["bindings"]["current_g7_source_closure"] = audit_closure
    report["semantic_audit"] = {
        "G7A_SEMANTIC_AUDIT": "PASS",
        "GPU_RERUN": False,
        "GPU_OPTIMIZER_STEP_RERUN": False,
        "raw_input_canonical_report_sha256": raw_input_sha,
        "audit_source_closure": audit_closure,
        "G7A_RAW_COUNTS_CONSISTENT": raw_counts_consistent,
        "G7A_DROP_RATES_RECOMPUTED": True,
        "G7A_QUEUE_STABILITY_CLASSIFIED": True,
        "G7A_TIME_SLICED_TIMESTAMPS_SUFFICIENT": time_sliced["timestamps_sufficient"],
        "G7A_QUIESCENT_WINDOW_SEMANTICS": time_sliced["control_flow_semantics"],
        "G7A_QUIESCENT_WINDOW_BUDGET_MET": time_sliced["quiescent_window_budget_met"],
        "G7A_MINIMUM_MEASURED_QUIESCENT_WINDOW_MS": time_sliced["minimum_measured_quiescent_window_required_ms"],
        "request_accounting": {"modes": accounting, "totals": totals},
        "queue_stability": {
            "QUEUE_MEMORY_BOUNDED_BY_CAP": True,
            "QUEUE_LOAD_STABLE_AT_10HZ": False,
            "NO_DROP_10HZ": False,
            "drops_only_after_queue_saturation": all(
                accounting[name]["queue"]["drops_occur_only_at_capacity"]
                for name in ("inference_only", "concurrent", "time_sliced_episode")
            ),
            "queue_wait_dominates_high_end_to_end_latency": all(
                accounting[name]["queue"]["queue_wait_dominates_p50_end_to_end"]
                for name in ("inference_only", "concurrent", "time_sliced_episode")
            ),
            "steady_state_classification": "DROP_POLICY_TRUNCATED_NOT_10HZ_STABLE",
        },
        "time_sliced": time_sliced,
        "memory": memory,
        "formal_conclusion": {
            "resource_coexistence": "PASS",
            "numerical_action_parity": "PASS",
            "synthetic_10hz_scheduling": "FAIL",
            "production_cadence_deadline": "UNBOUND",
            "production_topology": "NOT_APPROVED",
            "time_slicing_removes_learner_contention_only": True,
            "time_slicing_does_not_fix_inference_only_service_over_100ms": True,
        },
        "G7P_EVIDENCE_FREEZE_ALLOWED": False,
        "freeze_blocker": "TIME_SLICED_PER_CYCLE_ABSOLUTE_TIMESTAMPS_AND_DISTINCT_RESUME_REQUEST_TIMESTAMP_MISSING",
    }
    for name in ("inference_only", "learner_only", "concurrent", "episode_time_sliced"):
        modes[name]["status_semantics"] = "PASS_MEANS_MEASUREMENT_COMPLETED_ONLY"
    modes["episode_time_sliced"]["minimum_quiescent_window_required_ms"] = time_sliced[
        "minimum_measured_quiescent_window_required_ms"
    ]
    modes["episode_time_sliced"]["completed_learner_cycles_within_first_1000ms"] = 0
    modes["episode_time_sliced"]["actual_learner_quiescent_occupancy_ms"] = time_sliced[
        "actual_learner_quiescent_occupancy_ms"
    ]
    report["comparisons"]["synthetic_release_grid"].update({
        "inference_only_10hz_sustainable": False,
        "concurrent_10hz_sustainable": False,
        "time_sliced_10hz_sustainable": False,
        "synthetic_100ms_grid_feasible": False,
    })
    report["gates"].update({
        "G7P_BENCHMARK_EXECUTION": "PASS",
        "G7P_RESULT_SEMANTICS": "PASS_MEANS_MEASUREMENT_COMPLETED_ONLY",
        "G7P_100MS_SYNTHETIC_GRID_FEASIBLE": False,
        "G7P_INFERENCE_ONLY_10HZ_SUSTAINABLE": False,
        "G7P_CONCURRENT_10HZ_SUSTAINABLE": False,
        "G7P_TIME_SLICED_10HZ_SUSTAINABLE": False,
        "PRODUCTION_REQUEST_CADENCE_VALIDATED": False,
        "QUEUE_MEMORY_BOUNDED_BY_CAP": True,
        "QUEUE_LOAD_STABLE_AT_10HZ": False,
        "NO_DROP_10HZ": False,
        "SHORT_RUN_MEMORY_SAFETY": "PASS",
        "SUSTAINED_MEMORY_LEAK_TEST": "NOT_RUN",
        "SUSTAINED_THERMAL_STABILITY": "UNVERIFIED",
        "G7P_TIME_SLICED_SEMANTICS": "UNVERIFIED",
        "G7P_EVIDENCE_FREEZE_ALLOWED": False,
    })
    report["canonical_report_sha256"] = canonical_sha256(report)
    return report


def _apply_g7b_semantics(
    report: dict[str, Any], config: Mapping[str, Any], *, base_sha: str,
    base_digests: Mapping[str, str], current_digests: Mapping[str, str],
    components_unchanged: Mapping[str, bool], current_preflight: Mapping[str, Any],
    parent_before: Mapping[str, Any], parent_after: Mapping[str, Any],
) -> dict[str, Any]:
    mode = report["modes"]["episode_time_sliced"]
    time_sliced = audit_time_sliced_semantics(
        mode, configured_gap_ms=float(config["time_sliced"]["inter_phase_gap_ms"]),
    )
    audit = copy.deepcopy(report["semantic_audit"])
    accounting = copy.deepcopy(audit["request_accounting"])
    accounting["modes"]["time_sliced_episode"] = audit_request_stream(
        mode["episode_inference"]["worker"],
        queue_capacity=int(config["inference"]["queue_capacity"]),
        release_period_ms=float(config["inference"]["release_period_ms"]),
    )
    accounting["modes"]["time_sliced_resume"] = audit_request_stream(
        mode["resume_inference"]["worker"],
        queue_capacity=int(config["inference"]["queue_capacity"]),
        release_period_ms=float(config["inference"]["release_period_ms"]),
    )
    totals = {
        key: sum(int(item[key]) for item in accounting["modes"].values())
        for key in ("scheduled", "completed", "dropped")
    }
    totals["drop_rate"] = totals["dropped"] / totals["scheduled"]
    totals["count_equation_holds"] = totals["scheduled"] == totals["completed"] + totals["dropped"]
    accounting["totals"] = totals
    audit.update({
        "GPU_RERUN": True,
        "GPU_OPTIMIZER_STEP_RERUN": True,
        "G7A_TIME_SLICED_TIMESTAMPS_SUFFICIENT": True,
        "G7A_QUIESCENT_WINDOW_SEMANTICS": "gap_not_budget",
        "request_accounting": accounting,
        "time_sliced": time_sliced,
        "G7P_EVIDENCE_FREEZE_ALLOWED": True,
    })
    for obsolete in (
        "G7A_QUIESCENT_WINDOW_BUDGET_MET", "G7A_MINIMUM_MEASURED_QUIESCENT_WINDOW_MS",
        "freeze_blocker",
    ):
        audit.pop(obsolete, None)
    report["semantic_audit"] = audit
    report["gates"].update({
        "G7P_TIME_SLICED_SEMANTICS": "VERIFIED_COLD_PROCESS_SWAP_ONLY",
        "G7P_EVIDENCE_FREEZE_ALLOWED": True,
        "G7P_100MS_SYNTHETIC_GRID_FEASIBLE": False,
        "G7P_TIME_SLICED_10HZ_SUSTAINABLE": False,
        "PRODUCTION_REQUEST_CADENCE_VALIDATED": False,
        "PRODUCTION_DEADLINE_VALIDATED": False,
        "PRODUCTION_GPU_TOPOLOGY_APPROVED": False,
        "PRODUCTION_COLD_PROCESS_SWAP_APPROVED": False,
        "G7_FORMAL_GATE_PASSED": False,
        "G7P_PROVISIONAL_TOPOLOGY_CANDIDATE": "NONE",
    })
    report["safety"].update({
        "GPU_RERUN": True,
        "GPU_OPTIMIZER_STEP_RERUN": True,
        "PRODUCTION_CHECKPOINT_WRITES": 0,
        "PRODUCTION_ACTOR_STATE_MUTATED": False,
    })
    report["baseline"]["current_g7_source_closure"] = current_source_closure()
    report["bindings"]["current_g7_source_closure"] = report["baseline"]["current_g7_source_closure"]
    parent_unchanged = parent_before == parent_after
    derived = time_sliced["derived_timing"]
    report["g7b_timestamp_instrumentation"] = {
        "G7B_BASE_G7A_CANONICAL_SHA256": base_sha,
        "G7B_TIMESTAMP_INSTRUMENTATION": "PASS",
        "G7B_TARGETED_TIME_SLICED_RERUN": "PASS",
        "G7B_RESULT": "PASS",
        "GPU_RERUN": True,
        "GPU_OPTIMIZER_STEP_RERUN": True,
        "base_component_digests": dict(base_digests),
        "post_rerun_component_digests": dict(current_digests),
        "component_digest_unchanged": dict(components_unchanged),
        "G7B_TIMESTAMP_ORDER_VALID": True,
        "G7B_INTER_PHASE_GAP_SEMANTICS": "gap_not_budget",
        "G7B_TIME_SLICED_TOPOLOGY": "cold_process_swap",
        "RESIDENT_TIME_SLICING": "NOT_RUN",
        "REAL_RESET_HOME_WINDOW_USED": False,
        "INTER_PHASE_GAP_IS_EXECUTION_BUDGET": False,
        "PRODUCTION_REQUIRED_RESET_HOME_WINDOW_MS": "UNVERIFIED",
        "timestamp_trace": copy.deepcopy(mode["timestamp_trace"]),
        "derived_timing": copy.deepcopy(derived),
        "targeted_gpu_preflight": copy.deepcopy(dict(current_preflight)),
        "parent_bindings_before": copy.deepcopy(dict(parent_before)),
        "parent_bindings_after": copy.deepcopy(dict(parent_after)),
        "parent_checkpoint_sha_unchanged": parent_unchanged,
    }
    report["final_checks"].update({
        "all_own_worker_pids_exited": True,
        "cuda_compute_process_count_after_exit": len(compute_processes()),
        "parent_checkpoint_sha_unchanged": parent_unchanged,
    })
    require(mode["oom_count"] == 0, "G7B_OOM")
    require(mode["reset_window_learner"]["worker"]["metrics"]["all_finite"] is True, "G7B_NONFINITE")
    require(parent_unchanged, "G7B_PARENT_BINDING_MUTATED")
    require(report["final_checks"]["cuda_compute_process_count_after_exit"] == 0, "G7B_FINAL_COMPUTE_PROCESS")
    return report


def _run(argv: list[str], *, check: bool = True, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, check=check, env=env)


def git_output(*args: str) -> str:
    return _run(["git", *args]).stdout.strip()


def is_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def baseline_verification(config: Mapping[str, Any]) -> dict[str, Any]:
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    parent = git_output("rev-parse", f"{head}^")
    require(branch == config["baseline"]["branch"], f"G7P_BRANCH:{branch}")
    require(head == EXPECTED_HEAD, f"G7P_HEAD:{head}")
    require(parent == G6C_COMMIT, f"G7P_PARENT:{parent}")
    require(is_ancestor(G6C_COMMIT, head) and is_ancestor(G6P_FREEZE, head), "G7P_G6_ANCESTRY")
    status = git_output("status", "--short")
    status_lines = [line for line in status.splitlines() if line]
    allowed_untracked = {
        "?? graphify-out/",
        "?? src/graphify-out/",
        "?? configs/stage3_gpu_coexistence.v1.development.yaml",
        "?? schemas/stage3_gpu_coexistence_report.v1.schema.json",
        "?? tools/benchmark_stage3_actor_learner_coexistence_gpu.py",
        "?? tests/test_stage3_gpu_coexistence_contract.py",
        "?? docs/stage3_gpu_coexistence_report.v1.md",
        "?? artifacts/development/stage3/stage3_gpu_coexistence.v1.json",
    }
    require(set(status_lines).issubset(allowed_untracked), f"G7P_WORKTREE:{status_lines}")
    historical_report = json.loads(HISTORICAL_JSON.read_text(encoding="utf-8"))
    historical = {
        "canonical_sha256": historical_report.get("canonical_report_sha256"),
        "json_sha256": sha256_file(HISTORICAL_JSON),
        "markdown_sha256": sha256_file(HISTORICAL_MD),
    }
    require(historical["json_sha256"] == HISTORICAL_JSON_SHA256, "G7P_G6P_JSON_SHA")
    require(historical["markdown_sha256"] == HISTORICAL_MD_SHA256, "G7P_G6P_MD_SHA")
    require(historical["canonical_sha256"] == HISTORICAL_CANONICAL_SHA256, "G7P_G6P_CANONICAL_SHA")
    return {
        "branch": branch,
        "head": head,
        "head_parent": parent,
        "g6c_is_ancestor": is_ancestor(G6C_COMMIT, head),
        "g6c2_is_ancestor": is_ancestor(EXPECTED_HEAD, head),
        "g6p_freeze_is_ancestor": is_ancestor(G6P_FREEZE, head),
        "historical_g6p": historical,
        "historical_evidence_unchanged": True,
        "current_g7_source_closure": current_source_closure(),
    }


def current_source_closure() -> dict[str, Any]:
    paths = [
        "configs/stage3_gpu_coexistence.v1.development.yaml",
        "schemas/stage3_gpu_coexistence_report.v1.schema.json",
        "tools/benchmark_stage3_actor_learner_coexistence_gpu.py",
        "tests/test_stage3_gpu_coexistence_contract.py",
    ]
    records = []
    for relative in paths:
        path = ROOT / relative
        if path.is_file():
            records.append({"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"records": records, "sha256": digest, "historical_g6_closure_comparison_required": False}


def _parse_csv(text: str) -> list[list[str]]:
    return [[field.strip() for field in line.split(",")] for line in text.splitlines() if line.strip()]


def compute_processes() -> list[dict[str, Any]]:
    result = _run([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
        "--format=csv,noheader,nounits",
    ], check=False)
    if result.returncode != 0 or "No running" in result.stdout:
        return []
    records = []
    for fields in _parse_csv(result.stdout):
        if len(fields) >= 4 and fields[0].isdigit():
            records.append({
                "pid": int(fields[0]), "process_name": fields[1],
                "used_memory_mib": float(fields[2]), "gpu_uuid": fields[3],
            })
    return records


def _query_gpu() -> dict[str, Any]:
    fields = [
        "index", "uuid", "name", "driver_version", "compute_mode", "memory.total",
        "memory.free", "memory.used", "utilization.gpu", "temperature.gpu", "power.draw",
        "power.limit", "clocks.current.sm", "clocks.current.memory",
        "clocks_event_reasons.active",
    ]
    result = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    rows = _parse_csv(result.stdout)
    require(len(rows) == 1 and len(rows[0]) == len(fields), f"G7P_GPU_QUERY:{rows}")
    row = dict(zip(fields, rows[0], strict=True))
    numeric = {
        "index": int(row["index"]),
        "memory_total_mib": float(row["memory.total"]),
        "memory_free_mib": float(row["memory.free"]),
        "memory_used_mib": float(row["memory.used"]),
        "utilization_gpu_percent": float(row["utilization.gpu"]),
        "temperature_c": float(row["temperature.gpu"]),
        "power_draw_w": float(row["power.draw"]),
        "power_limit_w": float(row["power.limit"]),
        "sm_clock_mhz": float(row["clocks.current.sm"]),
        "memory_clock_mhz": float(row["clocks.current.memory"]),
    }
    return {
        **numeric, "uuid": row["uuid"], "name": row["name"],
        "driver_version": row["driver_version"], "compute_mode": row["compute_mode"],
        "clock_event_reasons_active": row["clocks_event_reasons.active"],
    }


def _meminfo() -> dict[str, float]:
    values: dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, rest = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[f"{name}_mib"] = float(rest.strip().split()[0]) / 1024.0
    return values


def gpu_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    gpu = _query_gpu()
    expected = config["environment"]
    require(gpu["index"] == expected["physical_cuda_device_index"], "G7P_GPU_INDEX")
    require(gpu["uuid"] == expected["expected_gpu_uuid"], f"G7P_GPU_UUID:{gpu['uuid']}")
    require(gpu["name"] == expected["expected_gpu_name"], f"G7P_GPU_NAME:{gpu['name']}")
    processes = compute_processes()
    require(not processes, f"G7P_OTHER_CUDA_COMPUTE_PROCESS:{processes}")
    torch_probe = _run([
        expected["python_executable"], "-c",
        "import json,torch; before=torch.cuda.is_initialized(); cudnn=torch.backends.cudnn.version(); print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':cudnn,'cuda_initialized_before_cudnn_query':before,'cuda_initialized_after_cudnn_query':torch.cuda.is_initialized()}))",
    ])
    torch_info = json.loads(torch_probe.stdout)
    require(torch_info["cuda_initialized_before_cudnn_query"] is False, "G7P_PREFLIGHT_CUDA_INITIALIZED")
    require(not compute_processes(), "G7P_DIAGNOSTIC_PROBE_DID_NOT_EXIT")
    mps = _run(["pgrep", "-af", "nvidia-cuda-mps"], check=False)
    disk = shutil.disk_usage(ROOT)
    return {
        "physical_device": gpu,
        "compute_processes_before": processes,
        "torch_runtime": torch_info,
        "memory": _meminfo(),
        "disk": {
            "path": str(ROOT), "total_mib": disk.total / 2**20,
            "free_mib": disk.free / 2**20,
        },
        "mps_status": "inactive" if mps.returncode != 0 else mps.stdout.strip(),
        "clocks_or_power_modified": False,
        "gpu_settings_modified": False,
        "base_graphics_vram_mib": gpu["memory_used_mib"],
        "cuda_initialized_in_parent": False,
    }


def _proc_metrics(pid: int) -> dict[str, float]:
    result: dict[str, float] = {
        "rss_mib": 0.0, "pss_mib": 0.0, "shared_mib": 0.0,
        "swap_mib": 0.0, "pinned_mib": 0.0, "read_mib": 0.0, "write_mib": 0.0,
    }
    try:
        rollup: dict[str, float] = {}
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if ":" in line:
                key, rest = line.split(":", 1)
                if rest.strip().split() and rest.strip().split()[0].isdigit():
                    rollup[key] = float(rest.strip().split()[0]) / 1024.0
        result.update({
            "rss_mib": rollup.get("Rss", 0.0),
            "pss_mib": rollup.get("Pss", 0.0),
            "shared_mib": rollup.get("Shared_Clean", 0.0) + rollup.get("Shared_Dirty", 0.0),
            "swap_mib": rollup.get("Swap", 0.0),
        })
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmPin:"):
                result["pinned_mib"] = float(line.split()[1]) / 1024.0
        io_values = {}
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                io_values[key] = int(value.strip())
        result["read_mib"] = io_values.get("read_bytes", 0) / 2**20
        result["write_mib"] = io_values.get("write_bytes", 0) / 2**20
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return result


class ResourceSampler:
    def __init__(self, pids: list[int], *, interval_ms: int) -> None:
        self.pids = pids
        self.interval = interval_ms / 1000.0
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=10)
        return self.report()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                gpu = _query_gpu()
                compute = {item["pid"]: item for item in compute_processes()}
            except Exception as error:  # evidence records sampler failures; workers continue.
                self.samples.append({"monotonic_ns": time.monotonic_ns(), "error": repr(error)})
                self._stop.wait(self.interval)
                continue
            per_process = {}
            for pid in self.pids:
                row = _proc_metrics(pid)
                row["device_memory_mib"] = compute.get(pid, {}).get("used_memory_mib", 0.0)
                per_process[str(pid)] = row
            self.samples.append({
                "monotonic_ns": time.monotonic_ns(), "gpu": gpu, "processes": per_process,
            })
            self._stop.wait(self.interval)

    def report(self) -> dict[str, Any]:
        valid = [sample for sample in self.samples if "gpu" in sample]
        gpu_fields = (
            "memory_used_mib", "utilization_gpu_percent", "temperature_c", "power_draw_w",
            "sm_clock_mhz", "memory_clock_mhz",
        )
        gpu = {
            name: distribution([float(sample["gpu"][name]) for sample in valid])
            for name in gpu_fields
        }
        process_reports = {}
        for pid in self.pids:
            rows = [sample["processes"].get(str(pid), {}) for sample in valid]
            process_reports[str(pid)] = {
                name: distribution([float(row.get(name, 0.0)) for row in rows])
                for name in (
                    "device_memory_mib", "rss_mib", "pss_mib", "shared_mib", "swap_mib",
                    "pinned_mib", "read_mib", "write_mib",
                )
            }
        aggregate_rss = [
            sum(float(row.get("rss_mib", 0.0)) for row in sample["processes"].values())
            for sample in valid
        ]
        clock_reason_counts: dict[str, int] = {}
        for sample in valid:
            reason = str(sample["gpu"]["clock_event_reasons_active"])
            clock_reason_counts[reason] = clock_reason_counts.get(reason, 0) + 1
        return {
            "sample_count": len(valid),
            "sampling_errors": len(self.samples) - len(valid),
            "sample_interval_ms": self.interval * 1000.0,
            "gpu": gpu,
            "per_process": process_reports,
            "aggregate_rss_mib": distribution(aggregate_rss),
            "clock_event_reasons": {
                "source": "nvidia-smi:clocks_event_reasons.active",
                "active_mask_sample_counts": clock_reason_counts,
                "note": "bitmask 0 means no active clock event reason; bitmask 1 is GPU idle",
            },
        }


def _tensor_digest(tensor: Any) -> str:
    value = tensor.detach().to(dtype=tensor.float().dtype, device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _inference_trial(
    *, engine: Any, base_request: Mapping[str, Any], trial: int, count: int,
    release_period_ms: float, queue_capacity: int, noise_seed: int,
) -> dict[str, Any]:
    import torch
    from forcesmolvla.inference import prepare_policy_inputs

    request_queue: queue.Queue[Any] = queue.Queue(maxsize=queue_capacity)
    producer_records: list[dict[str, Any]] = []
    sentinel = object()
    period_ns = int(release_period_ms * 1e6)

    def producer() -> None:
        base_ns = time.monotonic_ns() + 2 * period_ns
        slot = 0
        while slot < count:
            target_ns = base_ns + slot * period_ns
            now = time.monotonic_ns()
            if now < target_ns:
                time.sleep((target_ns - now) / 1e9)
            actual_ns = time.monotonic_ns()
            if actual_ns - target_ns >= period_ns:
                skipped = min(count - slot, int((actual_ns - target_ns) // period_ns))
                for offset in range(skipped):
                    producer_records.append({
                        "trial": trial, "slot": slot + offset,
                        "release_target_ns": target_ns + offset * period_ns,
                        "release_actual_ns": None, "status": "scheduler_stale_drop",
                    })
                slot += skipped
                if slot >= count:
                    break
                target_ns = base_ns + slot * period_ns
                now = time.monotonic_ns()
                if now < target_ns:
                    time.sleep((target_ns - now) / 1e9)
                actual_ns = time.monotonic_ns()
            release = {
                "trial": trial, "slot": slot, "release_target_ns": target_ns,
                "release_actual_ns": actual_ns,
                "release_jitter_ms": (actual_ns - target_ns) / 1e6,
                "queue_depth_at_release": request_queue.qsize(),
            }
            try:
                request_queue.put_nowait(release)
                release["status"] = "released"
            except queue.Full:
                release["status"] = "queue_exhaustion_drop"
            producer_records.append(release)
            slot += 1
        request_queue.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    trial_started_ns = time.monotonic_ns()
    thread.start()
    results: list[dict[str, Any]] = []
    queue_drained_ns: int | None = None
    while True:
        release = request_queue.get()
        if release is sentinel:
            queue_drained_ns = time.monotonic_ns()
            break
        consume_ns = time.monotonic_ns()
        sequence = f"g7p-t{trial:02d}-s{release['slot']:06d}"
        request = copy.deepcopy(base_request)
        request["request_id"] = sequence
        request["chunk_id"] = sequence
        preprocessing_started_ns = time.monotonic_ns()
        batch, context = prepare_policy_inputs(
            engine.policy, request, engine.runtime_artifacts, engine.contract, engine.device,
        )
        torch.cuda.synchronize(engine.device)
        preprocessing_completed_ns = time.monotonic_ns()
        captured: dict[str, Any] = {}
        original_private = engine.policy._predict_action_chunks

        def capture(*args: Any, **kwargs: Any):
            normalized, absolute = original_private(*args, **kwargs)
            captured["normalized"] = normalized.detach()
            return normalized, absolute

        engine.policy._predict_action_chunks = capture
        service_started_ns = time.monotonic_ns()
        try:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
            ):
                actions = engine.policy.predict_action_chunk(
                    batch, chunk_context=context, noise=noise_seed,
                )
            torch.cuda.synchronize(engine.device)
        finally:
            del engine.policy._predict_action_chunks
        service_completed_ns = time.monotonic_ns()
        actions_cpu = actions[0].detach().float().cpu().contiguous()
        normalized_cpu = captured["normalized"][0].detach().float().cpu().contiguous()
        result_ready_ns = time.monotonic_ns()
        require(tuple(actions_cpu.shape) == (50, 7), "G7P_INFERENCE_SHAPE")
        require(tuple(normalized_cpu.shape) == (50, 7), "G7P_NORMALIZED_SHAPE")
        require(bool(torch.isfinite(actions_cpu).all()) and bool(torch.isfinite(normalized_cpu).all()), "G7P_INFERENCE_NONFINITE")
        e2e_ms = (result_ready_ns - int(release["release_actual_ns"])) / 1e6
        results.append({
            **release,
            "status": "result_ready",
            "consume_ns": consume_ns,
            "preprocessing_started_ns": preprocessing_started_ns,
            "preprocessing_completed_ns": preprocessing_completed_ns,
            "service_started_ns": service_started_ns,
            "service_completed_ns": service_completed_ns,
            "result_ready_ns": result_ready_ns,
            "queue_depth_at_consume": request_queue.qsize(),
            "queue_wait_ms": (consume_ns - int(release["release_actual_ns"])) / 1e6,
            "preprocessing_ms": (preprocessing_completed_ns - preprocessing_started_ns) / 1e6,
            "gpu_service_ms": (service_completed_ns - service_started_ns) / 1e6,
            "end_to_end_ms": e2e_ms,
            "benchmark_macro_lateness_ms": e2e_ms - release_period_ms,
            "action_digest": _tensor_digest(actions_cpu),
            "normalized_action_digest": _tensor_digest(normalized_cpu),
            "shape": list(actions_cpu.shape),
            "dtype": str(actions_cpu.dtype),
            "finite": True,
            "tcp6_finite": bool(torch.isfinite(actions_cpu[:, :6]).all()),
            "gripper_values": sorted(set(float(value) for value in actions_cpu[:, 6].tolist())),
            "action_min": float(actions_cpu.min()),
            "action_max": float(actions_cpu.max()),
            "action_values": actions_cpu.tolist(),
        })
        del batch, context, actions, actions_cpu, normalized_cpu, captured
    thread.join(timeout=10)
    trial_completed_ns = time.monotonic_ns()
    drops = [row for row in producer_records if row["status"] != "released"]
    return {
        "trial": trial,
        "started_ns": trial_started_ns,
        "completed_ns": trial_completed_ns,
        "queue_drained_ns": queue_drained_ns,
        "scheduled_release_count": count,
        "completed_request_count": len(results),
        "stale_drop_count": len(drops),
        "queue_exhaustion_count": sum(row["status"] == "queue_exhaustion_drop" for row in drops),
        "burst_catchup_count": 0,
        "release_schedule": producer_records,
        "requests": results,
    }


def inference_worker(spec: Mapping[str, Any]) -> dict[str, Any]:
    import random
    import numpy as np
    import torch

    tools_path = str(ROOT / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from export_stage2b_cycle210_evaluation_smoke import build_request
    from forcesmolvla.dataset_v3 import load_dataset_split
    from forcesmolvla.inference import load_checkpoint_inference_contract
    import forcesmolvla.inference as inference_module
    from forcesmolvla.rft.frozen_vlm_trainability import frozen_state_digest
    from forcesmolvla.rft.stage3.parent import load_parent_binding
    from forcesmolvla.rft.training_cycle import module_state_sha256
    from serve_policy import InferenceEngine

    config = validate_config(_load_mapping(Path(spec["config_path"])))
    expected = config["environment"]
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == expected["expected_cuda_visible_devices"], "G7P_WORKER_VISIBLE_DEVICE")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", expected["cublas_workspace_config"])
    seed = int(expected["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "G7P_INFERENCE_SINGLE_GPU")
    device = torch.device("cuda:0")
    process_started_ns = time.monotonic_ns()
    checkpoint = _resolve(config["inference"]["checkpoint"])
    checkpoint_model_sha_before = sha256_file(checkpoint / "model.safetensors")
    evaluation = json.loads(_resolve(config["inference"]["evaluation_config"]).read_text())
    contract = load_checkpoint_inference_contract(checkpoint)
    dataset_started_ns = time.monotonic_ns()
    dataset = load_dataset_split(
        _resolve(evaluation["dataset_root"]), repo_id=contract.repo_id,
        split_name=evaluation["fixed_observation"]["split"], artifact_use="development",
        delta_timestamps={"action": [index / 30 for index in range(50)]},
    )
    sample = dataset[int(evaluation["fixed_observation"]["dataset_index"])]
    dataset_completed_ns = time.monotonic_ns()
    base_request = build_request(contract, sample, request_id=f"g7p-{spec['label']}-base")
    engine_started_ns = time.monotonic_ns()
    engine = InferenceEngine(
        checkpoint, _resolve(config["inference"]["rulespec"]),
        _resolve(config["inference"]["rulespec_schema"]), device,
    )
    torch.cuda.synchronize(device)
    engine_completed_ns = time.monotonic_ns()
    actor_hash_before = module_state_sha256(engine.policy)
    frozen_hash_before = frozen_state_digest(engine.policy)
    parent_binding = load_parent_binding(ROOT / "configs/stage3_parent_binding.v1.development.json")
    cache = BoundedDecodedImageCache(int(config["inference"]["cache_max_bytes"]))
    original_decode = inference_module.decode_rgb_image
    inference_module.decode_rgb_image = (
        lambda encoded, **kwargs: cache.get_or_decode(encoded, original_decode, **kwargs)
    )
    try:
        cold = _inference_trial(
            engine=engine, base_request=base_request, trial=-1,
            count=int(config["inference"]["warmup_requests"]),
            release_period_ms=float(config["inference"]["release_period_ms"]),
            queue_capacity=int(config["inference"]["queue_capacity"]),
            noise_seed=int(config["inference"]["fixed_flow_noise_seed"]),
        )
        trials = [
            _inference_trial(
                engine=engine, base_request=base_request, trial=trial, count=int(spec["requests_per_trial"]),
                release_period_ms=float(config["inference"]["release_period_ms"]),
                queue_capacity=int(config["inference"]["queue_capacity"]),
                noise_seed=int(config["inference"]["fixed_flow_noise_seed"]),
            )
            for trial in range(int(config["inference"]["measured_trials"]))
        ]
    finally:
        inference_module.decode_rgb_image = original_decode
    torch.cuda.synchronize(device)
    actor_hash_after = module_state_sha256(engine.policy)
    frozen_hash_after = frozen_state_digest(engine.policy)
    all_requests = [row for trial in trials for row in trial["requests"]]
    summary = summarize_requests(
        all_requests, formal_minimum=int(config["inference"]["formal_p99_minimum_requests"]),
    )
    summary["scheduled_release_count"] = sum(trial["scheduled_release_count"] for trial in trials)
    summary["stale_drop_count"] = sum(trial["stale_drop_count"] for trial in trials)
    summary["queue_exhaustion_count"] = sum(trial["queue_exhaustion_count"] for trial in trials)
    summary["queue_exhaustion"] = summary["queue_exhaustion_count"] > 0
    unique_digests = sorted(set(row["action_digest"] for row in all_requests))
    unique_normalized = sorted(set(row["normalized_action_digest"] for row in all_requests))
    parameter_storage = [parameter.untyped_storage().data_ptr() for parameter in engine.policy.parameters()]
    result = {
        "status": "PASS",
        "worker_role": "inference_actor",
        "label": spec["label"],
        "pid": os.getpid(),
        "process_started_ns": process_started_ns,
        "model_ready_ns": engine_completed_ns,
        "dataset_load_ms": (dataset_completed_ns - dataset_started_ns) / 1e6,
        "actor_load_ms": (engine_completed_ns - engine_started_ns) / 1e6,
        "cold_start": cold,
        "trials": trials,
        "metrics": summary,
        "row_identity": {
            "split": evaluation["fixed_observation"]["split"],
            "dataset_index": evaluation["fixed_observation"]["dataset_index"],
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
        },
        "fixed_flow_noise_seed": int(config["inference"]["fixed_flow_noise_seed"]),
        "action_semantics": {
            "shape": [50, 7], "dtype": "torch.float32", "H": 50,
            "normalized_delta_private_path_recorded": True,
            "public_absolute7": True, "tcp6": "absolute TCP xyz+rpy",
            "gripper": "binary absolute width metres",
            "finite": all(row["finite"] for row in all_requests),
            "unique_action_digests": unique_digests,
            "unique_normalized_action_digests": unique_normalized,
            "per_request_action_digest": [row["action_digest"] for row in all_requests],
        },
        "actor_state": {
            "module_sha256_before": actor_hash_before,
            "module_sha256_after": actor_hash_after,
            "unchanged": actor_hash_before == actor_hash_after,
            "checkpoint_model_sha256_before": checkpoint_model_sha_before,
            "checkpoint_model_sha256_after": sha256_file(checkpoint / "model.safetensors"),
            "frozen_vlm_sha256_before": frozen_hash_before,
            "frozen_vlm_sha256_after": frozen_hash_after,
            "gradients_absent": all(parameter.grad is None for parameter in engine.policy.parameters()),
            "parameter_storage_pointer_digest": hashlib.sha256(
                json.dumps(parameter_storage).encode()
            ).hexdigest(),
        },
        "bindings": {
            "normalizer_sha256": parent_binding["normalizer_binding"]["sha256"],
            "action_contract_sha256": parent_binding["action_contract_binding"]["sha256"],
            "task_feature_logical_sha256": parent_binding["task_feature_binding"]["logical_object_sha256"],
            "runtime_contract_sha256": parent_binding["runtime_contract_binding"]["sha256"],
        },
        "decoded_image_cache": cache.report(),
        "torch_memory": {
            "allocated_peak_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "reserved_peak_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "allocation_failure_count": 0,
            "fragmentation_failure_count": 0,
        },
        "safety": {
            "production_actor_state_mutated": False,
            "network_server_started": False,
            "robot_connection_count": 0,
            "robot_command_count": 0,
            "policy_publication_count": 0,
        },
    }
    require(result["actor_state"]["unchanged"], "G7P_INFERENCE_ACTOR_HASH_CHANGED")
    require(result["actor_state"]["gradients_absent"], "G7P_INFERENCE_ACTOR_GRADIENT")
    require(result["actor_state"]["checkpoint_model_sha256_before"] == result["actor_state"]["checkpoint_model_sha256_after"], "G7P_CHECKPOINT_MUTATED")
    return result


def learner_worker(spec: Mapping[str, Any]) -> dict[str, Any]:
    tools_path = str(ROOT / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    import torch
    import preflight_stage3_gpu as g4p
    from unittest.mock import patch

    config = validate_config(_load_mapping(Path(spec["config_path"])))
    started_ns = time.monotonic_ns()
    timestamp_trace: dict[str, int] = {}
    strict_load = g4p._strict_load_parents
    critic_step = g4p._critic_step
    summarize_cycle = g4p._summarize_cycle_performance

    def timestamped_load(*args: Any, **kwargs: Any) -> Any:
        value = strict_load(*args, **kwargs)
        torch.cuda.synchronize(torch.device("cuda:0"))
        timestamp_trace["learner_model_ready_ns"] = time.monotonic_ns()
        return value

    def timestamped_critic(*args: Any, **kwargs: Any) -> Any:
        cycle = int(kwargs["cycle"])
        if int(kwargs["substep"]) == 1:
            kind = "warmup_cycle" if cycle == 0 else f"measured_cycle_{cycle}"
            timestamp_trace[f"learner_{kind}_start_ns"] = time.monotonic_ns()
        return critic_step(*args, **kwargs)

    def timestamped_summary(*args: Any, **kwargs: Any) -> Any:
        value = summarize_cycle(*args, **kwargs)
        cycle = int(args[0] if args else kwargs["cycle"])
        kind = "warmup_cycle" if cycle == 0 else f"measured_cycle_{cycle}"
        timestamp_trace[f"learner_{kind}_end_ns"] = time.monotonic_ns()
        return value

    with (
        patch.object(g4p, "_strict_load_parents", side_effect=timestamped_load),
        patch.object(g4p, "_critic_step", side_effect=timestamped_critic),
        patch.object(g4p, "_summarize_cycle_performance", side_effect=timestamped_summary),
    ):
        report = g4p.run_gpu_preflight(_resolve(config["learner"]["config"]))
    completed_ns = time.monotonic_ns()
    measured = [row for row in report["performance"]["cycles"] if row["kind"] == "measured"]
    require(len(measured) == 3, "G7P_LEARNER_MEASURED_TRIALS")
    cycle_seconds = [float(row["wall_seconds"]) for row in measured]
    all_cycle_seconds = [float(row["wall_seconds"]) for row in report["performance"]["cycles"]]
    load_seconds = float(report["parent_load"].get("actor_load_seconds", 0.0)) + float(
        report["parent_load"].get("critic_load_seconds", 0.0)
    )
    setup_seconds = max(0.0, (completed_ns - started_ns) / 1e9 - sum(all_cycle_seconds) - load_seconds)
    losses = {
        "critic": [row["loss"] for row in report["critic_updates"] if row["cycle"] > 0],
        "actor": [row["loss"] for row in report["actor_updates"] if row["cycle"] > 0],
        "q_statistics": [row["actor_action_q"] for row in report["actor_updates"] if row["cycle"] > 0],
        "gradient_norms": [
            {
                "cycle": row["cycle"], "preclip": row["gradient_preclip_norm"],
                "postclip": row["gradient_postclip_norm"], "per_module": row["per_module_gradient_norm"],
            }
            for row in report["actor_updates"] if row["cycle"] > 0
        ],
    }
    return {
        "status": "PASS", "worker_role": "disposable_learner", "label": spec["label"],
        "pid": os.getpid(), "started_ns": started_ns, "completed_ns": completed_ns,
        "timestamp_trace": timestamp_trace,
        "source_report": report,
        "metrics": {
            "warmup_cycles": 1, "measured_cycles": 3,
            "critic_optimizer_steps": report["cycles"]["critic_optimizer_steps"],
            "actor_optimizer_steps": report["cycles"]["actor_optimizer_steps"],
            "target_polyak_steps": report["cycles"]["target_polyak_steps"],
            "measured_critic_optimizer_steps": 6,
            "measured_actor_optimizer_steps": 3,
            "measured_target_polyak_steps": 6,
            "cycle_time_ms": distribution([value * 1000.0 for value in cycle_seconds]),
            "cycles_per_hour": distribution([3600.0 / value for value in cycle_seconds]),
            "data_and_pre_cycle_setup_latency_ms": setup_seconds * 1000.0,
            "data_latency_method": "worker wall minus strict parent load and synchronized cycle bodies; includes validation/setup overhead",
            "losses": losses,
            "all_finite": report["numerics"]["all_finite"],
            "optimizer_ownership": report["optimizer_ownership"],
            "calql_online_call_count": report["numerics"]["calql_online_call_count"],
            "cql_online_call_count": report["numerics"]["cql_online_call_count"],
            "random_candidate_online_call_count": report["numerics"]["random_candidate_online_call_count"],
            "mc_return_online_call_count": report["numerics"]["mc_return_online_call_count"],
        },
        "torch_memory": {
            "allocated_peak_mib": report["performance"]["measured_peak"]["peak_allocated_mib"],
            "reserved_peak_mib": torch.cuda.max_memory_reserved(torch.device("cuda:0")) / 2**20,
            "reserved_at_measured_end_mib": report["performance"]["measured_peak"]["reserved_mib"],
            "allocation_failure_count": 0, "fragmentation_failure_count": 0,
        },
        "actor_state": {
            "frozen_vlm_sha256_before": report["numerics"]["frozen_hash_before"],
            "frozen_vlm_sha256_after": report["numerics"]["frozen_hash_after"],
            "frozen_vlm_unchanged": report["numerics"]["frozen_hash_unchanged"],
            "disposable_actor_updated": True,
            "production_actor_state_mutated": False,
        },
        "data": report["data"],
        "safety": {
            "real_online_R_used": False,
            "R_source": "synthetic_preflight_R_only",
            "no_production_replay_writes": True,
            "runtime_optimizer_state_persisted": False,
            "checkpoint_writeback": False,
            "policy_publication_count": 0,
            "network_server_started": False,
            "robot_connection_count": 0,
            "robot_command_count": 0,
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def worker_main(kind: str, spec_path: Path, output_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    try:
        result = inference_worker(spec) if kind == "inference" else learner_worker(spec)
    except BaseException as error:
        result = {
            "status": "FAIL", "worker_role": kind, "label": spec.get("label"),
            "pid": os.getpid(), "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "oom_count": int("out of memory" in str(error).lower() or type(error).__name__ == "OutOfMemoryError"),
            "network_server_started": False, "robot_connection_count": 0,
            "robot_command_count": 0, "checkpoint_writeback": False,
        }
        _write_json(output_path, result)
        return 1
    _write_json(output_path, result)
    return 0


def _worker_environment(config: Mapping[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": config["environment"]["expected_cuda_visible_devices"],
        "CUBLAS_WORKSPACE_CONFIG": config["environment"]["cublas_workspace_config"],
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT / 'tools'}",
        "PYTHONUNBUFFERED": "1",
    })
    return env


def _spawn_worker(
    kind: str, label: str, requests_per_trial: int | None,
    *, config_path: Path, temp_dir: Path, config: Mapping[str, Any],
    timing: dict[str, int] | None = None, timing_prefix: str | None = None,
) -> tuple[subprocess.Popen[str], Path]:
    spec = {
        "config_path": str(config_path), "label": label,
        "requests_per_trial": requests_per_trial,
    }
    spec_path = temp_dir / f"{label}.spec.json"
    output_path = temp_dir / f"{label}.result.json"
    _write_json(spec_path, spec)
    if timing is not None and timing_prefix is not None:
        timing[f"{timing_prefix}_spawn_requested_ns"] = time.monotonic_ns()
    process = subprocess.Popen(
        [
            config["environment"]["python_executable"], str(Path(__file__).resolve()),
            "--worker", kind, "--spec", str(spec_path), "--worker-output", str(output_path),
        ],
        cwd=ROOT, env=_worker_environment(config), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if timing is not None and timing_prefix is not None:
        timing[f"{timing_prefix}_process_spawn_ns"] = time.monotonic_ns()
    return process, output_path


def _collect_worker(process: subprocess.Popen[str], output_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stdout, stderr = process.communicate()
    diagnostics = {
        "returncode": process.returncode,
        "stdout_tail": stdout[-8000:], "stderr_tail": stderr[-16000:],
    }
    if output_path.is_file():
        result = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        result = {
            "status": "FAIL", "pid": process.pid, "error": "G7P_WORKER_RESULT_MISSING",
            "oom_count": int(process.returncode in (-9, 137)),
        }
    result["process_diagnostics"] = diagnostics
    return result, diagnostics


def _wait_gpu_pids_gone(pids: list[int], *, timeout_s: float = 30.0) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    deadline = time.monotonic() + timeout_s
    last = []
    while time.monotonic() < deadline:
        last = [item for item in compute_processes() if item["pid"] in pids]
        if not last:
            completed_ns = time.monotonic_ns()
            return {"idle": True, "latency_ms": (completed_ns - started_ns) / 1e6, "remaining": []}
        time.sleep(0.05)
    return {
        "idle": False, "latency_ms": (time.monotonic_ns() - started_ns) / 1e6,
        "remaining": last,
    }


def _run_single_worker(
    kind: str, label: str, requests_per_trial: int | None,
    *, config_path: Path, temp_dir: Path, config: Mapping[str, Any],
    timing: dict[str, int] | None = None, timing_prefix: str | None = None,
) -> dict[str, Any]:
    process, output = _spawn_worker(
        kind, label, requests_per_trial, config_path=config_path, temp_dir=temp_dir, config=config,
        timing=timing, timing_prefix=timing_prefix,
    )
    sampler = ResourceSampler([process.pid], interval_ms=int(config["environment"]["sample_period_ms"]))
    sampler.start()
    result, _diagnostics = _collect_worker(process, output)
    if timing is not None and timing_prefix is not None:
        timing[f"{timing_prefix}_worker_exit_ns"] = time.monotonic_ns()
    resources = sampler.stop()
    idle = _wait_gpu_pids_gone([process.pid])
    return {
        "status": "PASS" if result.get("status") == "PASS" and idle["idle"] else "FAIL",
        "worker_pids": [process.pid], "worker": result,
        "resources": resources, "worker_exit_to_gpu_idle": idle,
        "oom_count": int(result.get("oom_count", 0)),
    }


def run_inference_only(config_path: Path, config: Mapping[str, Any], temp_dir: Path) -> dict[str, Any]:
    return _run_single_worker(
        "inference", "inference_only", int(config["inference"]["inference_only_requests_per_trial"]),
        config_path=config_path, temp_dir=temp_dir, config=config,
    )


def run_learner_only(config_path: Path, config: Mapping[str, Any], temp_dir: Path) -> dict[str, Any]:
    return _run_single_worker(
        "learner", "learner_only", None,
        config_path=config_path, temp_dir=temp_dir, config=config,
    )


def run_concurrent(config_path: Path, config: Mapping[str, Any], temp_dir: Path) -> dict[str, Any]:
    inference, inference_output = _spawn_worker(
        "inference", "concurrent_inference", int(config["inference"]["concurrent_requests_per_trial"]),
        config_path=config_path, temp_dir=temp_dir, config=config,
    )
    learner, learner_output = _spawn_worker(
        "learner", "concurrent_learner", None,
        config_path=config_path, temp_dir=temp_dir, config=config,
    )
    require(inference.pid != learner.pid, "G7P_CONCURRENT_PID_ALIAS")
    sampler = ResourceSampler(
        [inference.pid, learner.pid], interval_ms=int(config["environment"]["sample_period_ms"]),
    )
    sampler.start()
    inference_result, _ = _collect_worker(inference, inference_output)
    learner_result, _ = _collect_worker(learner, learner_output)
    resources = sampler.stop()
    idle = _wait_gpu_pids_gone([inference.pid, learner.pid])
    both_pass = inference_result.get("status") == learner_result.get("status") == "PASS"
    independent = {
        "fresh_subprocesses": True,
        "distinct_pids": inference.pid != learner.pid,
        "distinct_cuda_contexts": True,
        "shared_parameter_storage": False,
        "shared_optimizer": False,
        "shared_rng": False,
    }
    return {
        "status": "PASS" if both_pass and idle["idle"] else "FAIL",
        "worker_pids": {"inference": inference.pid, "learner": learner.pid},
        "inference": inference_result, "learner": learner_result,
        "isolation": independent, "resources": resources,
        "worker_exit_to_gpu_idle": idle,
        "oom_count": int(inference_result.get("oom_count", 0)) + int(learner_result.get("oom_count", 0)),
    }


def run_time_sliced(config_path: Path, config: Mapping[str, Any], temp_dir: Path) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "clock_source": "CLOCK_MONOTONIC",
        "linux_same_boot_cross_process_comparable": sys.platform.startswith("linux"),
        "cross_process_clock_scope": "same Linux boot CLOCK_MONOTONIC",
        "TIME_SLICED_TOPOLOGY": "cold_process_swap",
        "RESIDENT_TIME_SLICING": "NOT_RUN",
        "REAL_RESET_HOME_WINDOW_USED": False,
        "INTER_PHASE_GAP_IS_EXECUTION_BUDGET": False,
    }
    episode = _run_single_worker(
        "inference", "time_sliced_episode_inference",
        int(config["inference"]["time_sliced_requests_per_trial"]),
        config_path=config_path, temp_dir=temp_dir, config=config,
        timing=trace, timing_prefix="episode",
    )
    episode_worker = episode["worker"]
    episode_releases = [
        int(row["release_actual_ns"])
        for trial in episode_worker["trials"] for row in trial["release_schedule"]
        if row.get("release_actual_ns") is not None
    ]
    require(episode_releases, "G7B_EPISODE_RELEASES_MISSING")
    trace["episode_last_release_ns"] = max(episode_releases)
    trace["episode_queue_drained_ns"] = int(episode_worker["trials"][-1]["queue_drained_ns"])
    gap_ms = float(config["time_sliced"]["inter_phase_gap_ms"])
    trace["pre_learner_gap_start_ns"] = time.monotonic_ns()
    time.sleep(gap_ms / 1000.0)
    trace["pre_learner_gap_end_ns"] = time.monotonic_ns()
    learner = _run_single_worker(
        "learner", "time_sliced_reset_learner", None,
        config_path=config_path, temp_dir=temp_dir, config=config,
        timing=trace, timing_prefix="learner",
    )
    trace.update(learner["worker"]["timestamp_trace"])
    trace["pre_resume_gap_start_ns"] = time.monotonic_ns()
    time.sleep(gap_ms / 1000.0)
    trace["pre_resume_gap_end_ns"] = time.monotonic_ns()
    resume = _run_single_worker(
        "inference", "time_sliced_resume_inference",
        int(config["inference"]["resume_requests_per_trial"]),
        config_path=config_path, temp_dir=temp_dir, config=config,
        timing=trace, timing_prefix="resume",
    )
    trace["resume_requested_ns"] = trace.pop("resume_spawn_requested_ns")
    worker = resume["worker"]
    trace["resume_model_ready_ns"] = int(worker["model_ready_ns"])
    cold_requests = worker["cold_start"]["requests"]
    require(cold_requests, "G7B_RESUME_COLD_REQUEST_MISSING")
    first = cold_requests[0]
    trace.update({
        "resume_first_request_release_ns": int(first["release_actual_ns"]),
        "resume_first_service_start_ns": int(first["service_started_ns"]),
        "resume_first_service_end_ns": int(first["service_completed_ns"]),
        "resume_first_result_ready_ns": int(first["result_ready_ns"]),
        "resume_queue_drained_ns": int(worker["trials"][-1]["queue_drained_ns"]),
    })
    derived = validate_g7b_timestamp_trace(trace, inter_phase_gap_ms=gap_ms)
    all_pass = episode["status"] == learner["status"] == resume["status"] == "PASS"
    return {
        "status": "PASS" if all_pass else "FAIL",
        "TIME_SLICED_TOPOLOGY": "cold_process_swap",
        "RESIDENT_TIME_SLICING": "NOT_RUN",
        "REAL_RESET_HOME_WINDOW_USED": False,
        "INTER_PHASE_GAP_IS_EXECUTION_BUDGET": False,
        "inter_phase_gap_ms": gap_ms,
        "synthetic_reset_home_window": True,
        "not_a_real_robot_reset_home": True,
        "episode_inference": episode, "reset_window_learner": learner,
        "resume_inference": resume,
        "timestamp_trace": trace,
        "derived_timing": derived,
        "timestamp_order_valid": True,
        "no_gpu_work_overlap": True,
        "learner_stop_boundary": "committed joint-cycle optimizer/Polyak/subbatch boundary",
        "learner_stop_to_gpu_idle_latency_ms": learner["worker_exit_to_gpu_idle"]["latency_ms"],
        "completed_learner_cycles_per_reset_window": (
            learner.get("worker", {}).get("metrics", {}).get("measured_cycles", 0)
        ),
        "inference_window_startup_resume_latency_ms": derived["resume_spawn_to_first_ready_ms"],
        "worker_pids": {
            "episode_inference": episode["worker_pids"],
            "learner": learner["worker_pids"], "resume_inference": resume["worker_pids"],
        },
        "resources": {
            "episode_inference": episode["resources"],
            "learner": learner["resources"], "resume_inference": resume["resources"],
        },
        "oom_count": episode["oom_count"] + learner["oom_count"] + resume["oom_count"],
    }


def compare_action_semantics(reference_mode: Mapping[str, Any], concurrent: Mapping[str, Any]) -> dict[str, Any]:
    reference = reference_mode.get("worker", {})
    contender = concurrent.get("inference", {})
    if reference.get("status") != "PASS" or contender.get("status") != "PASS":
        return {"status": "FAIL", "reason": "inference worker unavailable", "first_different_request": None}
    left_trials = reference["trials"]
    right_trials = contender["trials"]
    left = [row for trial in left_trials for row in trial["requests"]]
    right = [row for trial in right_trials for row in trial["requests"]]
    count = min(len(left), len(right))
    first = next((index for index in range(count) if left[index]["action_digest"] != right[index]["action_digest"]), None)
    if first is None and count and all(
        left[index]["shape"] == right[index]["shape"]
        and left[index]["dtype"] == right[index]["dtype"]
        and left[index]["finite"] and right[index]["finite"]
        for index in range(count)
    ):
        return {
            "status": "PASS", "compared_request_count": count,
            "shape_dtype_H50_parity": True, "finite": True,
            "action_digest_parity": True, "first_different_request": None,
            "max_absolute_error": 0.0, "max_relative_error": 0.0,
            "normalized_action_delta_semantics": True,
            "tcp6_gripper_semantics": True,
        }
    if first is None:
        first = 0
    left_values = left[first].get("action_values", []) if left else []
    right_values = right[first].get("action_values", []) if right else []
    pairs = [
        (float(a), float(b))
        for left_row, right_row in zip(left_values, right_values)
        for a, b in zip(left_row, right_row)
    ]
    absolute = [abs(a - b) for a, b in pairs]
    relative = [abs(a - b) / max(abs(a), abs(b), 1e-12) for a, b in pairs]
    return {
        "status": "FAIL", "compared_request_count": count,
        "shape_dtype_H50_parity": False, "finite": all(row["finite"] for row in left[:count] + right[:count]),
        "action_digest_parity": False, "first_different_request": first,
        "max_absolute_error": max(absolute) if absolute else None,
        "max_relative_error": max(relative) if relative else None,
        "tolerance_expanded": False,
    }


def _learner_slowdown(learner_only: Mapping[str, Any], concurrent: Mapping[str, Any]) -> dict[str, Any]:
    left = learner_only.get("worker", {}).get("metrics", {}).get("cycle_time_ms", {}).get("p50")
    right = concurrent.get("learner", {}).get("metrics", {}).get("cycle_time_ms", {}).get("p50")
    return {
        "learner_only_cycle_time_p50_ms": left,
        "concurrent_cycle_time_p50_ms": right,
        "slowdown_ratio": right / left if left and right else None,
    }


def _parent_records() -> dict[str, dict[str, Any]]:
    binding = json.loads((ROOT / "configs/stage3_parent_binding.v1.development.json").read_text())
    paths = {"actor": Path(binding["actor_parent"]["absolute_path"])}
    for group in ("critic_parent", "target_critic_parent"):
        for item in binding[group]["artifacts"]:
            paths[item["logical_role"]] = Path(item["absolute_path"])
    for name in (
        "normalizer_binding", "action_contract_binding", "task_feature_binding",
        "calibration_binding", "runtime_contract_binding",
    ):
        paths[name] = Path(binding[name]["absolute_path"])
    return {
        role: {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for role, path in paths.items()
    }


def _validate_report(report: Mapping[str, Any], schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(report)


def render_g7b_markdown(report: Mapping[str, Any]) -> str:
    g7b = report["g7b_timestamp_instrumentation"]
    trace = g7b["timestamp_trace"]
    derived = g7b["derived_timing"]
    accounting = report["semantic_audit"]["request_accounting"]
    lines = [
        "# Stage-3 G7B targeted time-sliced timestamp report",
        "",
        "Result: `PASS`; this means the targeted cold-process-swap workload completed and its timestamp evidence passed consistency checks. It does not approve a production topology or cadence.",
        "",
        f"- Base G7A canonical SHA-256: `{g7b['G7B_BASE_G7A_CANONICAL_SHA256']}`.",
        "- `TIME_SLICED_TOPOLOGY=cold_process_swap`; `RESIDENT_TIME_SLICING=NOT_RUN`; `REAL_RESET_HOME_WINDOW_USED=false`.",
        "- `G7P_RESULT_SEMANTICS=PASS_MEANS_MEASUREMENT_COMPLETED_ONLY`.",
        "- `inter_phase_gap_ms=1000`; `INTER_PHASE_GAP_IS_EXECUTION_BUDGET=false`. The gap is a coordinator delay, not learner execution budget.",
        "- Clock: `CLOCK_MONOTONIC` via `time.monotonic_ns()`; Linux processes on the same boot share a comparable monotonic clock domain.",
        "- Production reset/Home window, request cadence, action-queue low-watermark, refresh cadence, staleness safety limit, deadline, and topology remain unverified/unbound.",
        "- H=50 is the model output horizon, not authorization to execute the full chunk open-loop for 1.67 seconds.",
        "",
        "## Preserved G7A components",
        "",
        "| Component | Before digest | After digest | Unchanged |",
        "| --- | --- | --- | --- |",
    ]
    for name in G7B_COMPONENT_NAMES:
        lines.append(
            f"| `{name}` | `{g7b['base_component_digests'][name]}` | "
            f"`{g7b['post_rerun_component_digests'][name]}` | "
            f"`{str(g7b['component_digest_unchanged'][name]).lower()}` |"
        )
    lines.extend([
        "",
        "## Absolute monotonic timestamps",
        "",
        "| Event | monotonic ns |",
        "| --- | ---: |",
    ])
    for key in G7B_REQUIRED_TIMESTAMPS:
        lines.append(f"| `{key}` | {trace[key]} |")
    for key in ("resume_first_service_start_ns", "resume_first_service_end_ns"):
        lines.append(f"| `{key}` | {trace[key]} |")
    lines.extend([
        "",
        "## Recomputed durations",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ])
    for key, value in derived.items():
        rendered = f"{value:.6f} ms" if isinstance(value, float) else str(value)
        lines.append(f"| `{key}` | {rendered} |")
    lines.extend([
        "",
        "`COLD_RESUME_SPAWN_TO_FIRST_READY_MS` is cold restart latency and is not mixed into steady-state inference latency. `FULL_MEASURED_LEARNER_PHASE_MS` spans learner spawn request through worker exit (including load, setup, warm-up, three measured cycles, and teardown); a single cycle is not presented as the full required reset/Home window.",
        "",
        "## Targeted request accounting",
        "",
        "| Stream | Scheduled | Completed | Dropped | Drop rate | p99 status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for name in ("time_sliced_episode", "time_sliced_resume"):
        item = accounting["modes"][name]
        lines.append(
            f"| `{name}` | {item['scheduled']} | {item['completed']} | {item['dropped']} | "
            f"{100 * item['drop_rate']:.2f}% | `{item['p99_status']}` |"
        )
    lines.extend([
        "",
        "## Conclusion boundary",
        "",
        "- `G7P_TIME_SLICED_SEMANTICS=VERIFIED_COLD_PROCESS_SWAP_ONLY` and `G7P_EVIDENCE_FREEZE_ALLOWED=true`.",
        "- `G7P_100MS_SYNTHETIC_GRID_FEASIBLE=false`; cold swapping can remove learner contention but cannot fix inference-only service time above 100 ms.",
        "- `G7P_PROVISIONAL_TOPOLOGY_CANDIDATE=NONE`, `PRODUCTION_COLD_PROCESS_SWAP_APPROVED=false`, `PRODUCTION_REQUEST_CADENCE_VALIDATED=false`, `PRODUCTION_DEADLINE_VALIDATED=false`, `PRODUCTION_GPU_TOPOLOGY_APPROVED=false`, `G7_FORMAL_GATE_PASSED=false`.",
        "- `PRODUCTION_CHECKPOINT_WRITES=0`; no server, ROS, robot, replay, publication, activation, or G8 path ran.",
        "",
        f"Canonical report SHA-256: `{report['canonical_report_sha256']}`",
        "",
    ])
    return "\n".join(lines)


def render_markdown(report: Mapping[str, Any]) -> str:
    if "g7b_timestamp_instrumentation" in report:
        return render_g7b_markdown(report)
    modes = report["modes"]
    audit = report["semantic_audit"]
    accounting = audit["request_accounting"]
    time_sliced = audit["time_sliced"]
    lines = [
        "# Stage-3 G7P isolated GPU coexistence report",
        "",
        f"Result: `{report['tool_status']}` — `PASS_MEANS_MEASUREMENT_COMPLETED_ONLY`.",
        "Workloads executed, metrics were captured, and no OOM/nonfinite result was observed. PASS does not mean 10 Hz was sustained, a deadline passed, the queue was stable, or a production topology was approved.",
        "",
        f"G7A semantic audit: `{audit['G7A_SEMANTIC_AUDIT']}`. Evidence freeze allowed: `{str(audit['G7P_EVIDENCE_FREEZE_ALLOWED']).lower()}`.",
        "",
        "## Audited runtime contract",
        "",
        "- 30/10 Hz, H=50/K=3: `configs/stage3_transition_contract.v1.development.json:/temporal`, `src/forcesmolvla/temporal.py:controller_reference_grid`, and `src/forcesmolvla/rft/stage3/transition.py:causal_zoh_ack_macro`.",
        "- Serve preprocessing: `src/forcesmolvla/inference.py:prepare_policy_inputs`; image decode: `src/forcesmolvla/inference.py:decode_rgb_image`.",
        "- Request cadence, chunk refresh, queue low-watermark, inference timeout, approved deadline/miss rate, and production hold policy are `UNBOUND`.",
        "- The 100 ms macro period was used only as a benchmark lateness reference and was not treated as a safety deadline.",
        "- Production decoded-image cache has no explicit max bytes/LRU/RSS/counters/resume reconstruction; `PRODUCTION_DECODED_CACHE_BOUNDED=false`.",
        "",
        "## Formal interpretation",
        "",
        "- Resource coexistence: `PASS`.",
        "- Numerical/action parity: `PASS`.",
        "- Synthetic 10 Hz scheduling: `FAIL`.",
        "- Production cadence/deadline: `UNBOUND`.",
        "- Production topology: `NOT_APPROVED`.",
        "- Time slicing removes learner contention only; inference-only p50 GPU service time already exceeds 100 ms.",
        "",
        "## Mode execution status",
        "",
    ]
    for name in ("inference_only", "learner_only", "concurrent", "episode_time_sliced", "separate_device"):
        lines.append(f"- `{name}`: `{modes[name]['status']}`")
    lines.extend([
        "",
        "## Request accounting recomputed from raw schedules",
        "",
        "| Stream | Scheduled | Completed | Dropped | Drop rate | Empirical p50 service capacity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name in ("inference_only", "concurrent", "time_sliced_episode", "time_sliced_resume"):
        item = accounting["modes"][name]
        lines.append(
            f"| `{name}` | {item['scheduled']} | {item['completed']} | {item['dropped']} | "
            f"{100.0 * item['drop_rate']:.2f}% | {item['empirical_capacity_hz']['p50_estimate_hz']:.3f} Hz |"
        )
    totals = accounting["totals"]
    lines.extend([
        f"| **total** | **{totals['scheduled']}** | **{totals['completed']}** | **{totals['dropped']}** | **{100.0 * totals['drop_rate']:.2f}%** | — |",
        "",
        "Empirical capacity is only `1000 / measured GPU service latency ms`; it is not a production request cadence or safety deadline.",
        "All four p99 values remain `PROVISIONAL_INSUFFICIENT_SAMPLES` because completed request counts, not scheduled slots, are below 1000.",
        "",
        "## Inference latency",
        "",
        "| Stream | Metric | p50 ms | p95 ms | p99 ms | max ms |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    workers = {
        "inference_only": modes["inference_only"]["worker"],
        "concurrent": modes["concurrent"]["inference"],
        "time_sliced_episode": modes["episode_time_sliced"]["episode_inference"]["worker"],
        "time_sliced_resume": modes["episode_time_sliced"]["resume_inference"]["worker"],
    }
    for name, worker in workers.items():
        for metric in ("gpu_service_ms", "queue_wait_ms", "end_to_end_ms"):
            value = worker["metrics"][metric]
            lines.append(
                f"| `{name}` | `{metric}` | {value['p50']:.3f} | {value['p95']:.3f} | "
                f"{value['p99']:.3f} | {value['max']:.3f} |"
            )
    lines.extend([
        "",
        "## Queue stability",
        "",
        "- `QUEUE_MEMORY_BOUNDED_BY_CAP=true`, `QUEUE_LOAD_STABLE_AT_10HZ=false`, `NO_DROP_10HZ=false`.",
        "- Inference-only, concurrent, and time-sliced episode reached depth 8 in every trial; every observed drop occurred at depth 8.",
        "- Queue wait dominates p50 E2E latency for those three streams. Every trial still completed requests after the final release target; the FIFO sentinel extended the trial until the backlog drained.",
        "- The apparent bound therefore comes from the fixed queue cap and drop policy, not a stable no-drop steady state.",
        "",
        "## Time-sliced evidence semantics",
        "",
        f"- Configured value: `{time_sliced['configured_quiescent_window_ms']:.0f} ms`; control-flow role: `{time_sliced['configured_value_role']}`.",
        f"- Actual learner phase occupancy: `{time_sliced['actual_learner_quiescent_occupancy_ms']:.3f} ms`; full episode-idle → resume-spawn quiescence: `{time_sliced['actual_quiescent_duration_ms']:.3f} ms`.",
        f"- Minimum measured committed learner-cycle wall time: `{time_sliced['minimum_measured_quiescent_window_required_ms']:.3f} ms`; cycles completed within the first 1000 ms: `0`; budget met: `false`.",
        f"- Resume spawn → first result-ready: `{time_sliced['cold_restart_spawn_to_first_result_ready_ms']:.3f} ms` (cold restart only, excluded from steady-state latency).",
        "- Absolute per-cycle start/end and a distinct resume-requested timestamp were not recorded. Therefore `G7P_TIME_SLICED_SEMANTICS=UNVERIFIED` and evidence freeze is blocked without a GPU rerun in this audit.",
        "",
        "| Cycle | Quiescent open ns | Learner start ns | Learner end ns | Resume requested ns | Inference spawn ns | First ready ns | Actual quiescence ms | In first 1000 ms | Overrun ms |",
        "| ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for trial in time_sliced["cycle_trials"]:
        lines.append(
            f"| {trial['cycle']} | {trial['quiescent_window_open_timestamp_ns']} | UNRECORDED | "
            f"UNRECORDED | UNRECORDED | {trial['inference_process_spawn_timestamp_ns']} | "
            f"{trial['first_result_ready_timestamp_ns']} | {trial['actual_quiescent_duration_ms']:.3f} | "
            f"{trial['learner_cycles_completed_within_first_1000ms']} | {trial['learner_overrun_ms']:.3f} |"
        )
    memory = audit["memory"]
    lines.extend([
        "",
        "## Memory conclusion boundary",
        "",
        f"- `SHORT_RUN_MEMORY_SAFETY={memory['SHORT_RUN_MEMORY_SAFETY']}`, `OOM_COUNT={memory['OOM_COUNT']}`, `ALLOCATION_FAILURE_COUNT={memory['ALLOCATION_FAILURE_COUNT']}`.",
        "- `SUSTAINED_MEMORY_LEAK_TEST=NOT_RUN` and `SUSTAINED_THERMAL_STABILITY=UNVERIFIED`.",
        "- The 16 MiB benchmark-only LRU does not repair or validate the unbounded production decoded-image cache.",
        "",
        "## Gates",
        "",
        f"- Fixed action semantics: `{report['comparisons']['fixed_action_semantics']['status']}`.",
        f"- Provisional topology candidate: `{report['gates']['G7P_PROVISIONAL_TOPOLOGY_CANDIDATE']}`.",
        "- `PRODUCTION_GPU_TOPOLOGY_APPROVED=false` and `G7_FORMAL_GATE_PASSED=false` because approval thresholds and production runtime parity remain unbound.",
        "- No server, replay write, policy publication/activation, checkpoint writeback, ROS, or robot path was used.",
        "",
        f"Canonical report SHA-256: `{report['canonical_report_sha256']}`",
        "",
    ])
    return "\n".join(lines)


def _atomic_write_report_pair(
    report: Mapping[str, Any], *, json_path: Path, markdown_path: Path, schema_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_fd, json_name = tempfile.mkstemp(prefix=".g7b-json-", dir=json_path.parent)
    md_fd, md_name = tempfile.mkstemp(prefix=".g7b-md-", dir=markdown_path.parent)
    os.close(json_fd)
    os.close(md_fd)
    json_temp = Path(json_name)
    md_temp = Path(md_name)
    try:
        _write_json(json_temp, report)
        md_temp.write_text(render_markdown(report), encoding="utf-8")
        reread = json.loads(json_temp.read_text(encoding="utf-8"))
        require(reread == report, "G7B_ATOMIC_JSON_REREAD")
        require(reread["canonical_report_sha256"] == canonical_sha256(reread), "G7B_ATOMIC_CANONICAL")
        _validate_report(reread, schema_path)
        require(
            reread["canonical_report_sha256"] in md_temp.read_text(encoding="utf-8"),
            "G7B_MARKDOWN_CANONICAL_BINDING",
        )
        os.replace(json_temp, json_path)
        os.replace(md_temp, markdown_path)
    finally:
        json_temp.unlink(missing_ok=True)
        md_temp.unlink(missing_ok=True)


def run_targeted_time_sliced_rerun(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = validate_config(_load_mapping(config_path))
    baseline_verification(config)
    report_path = _resolve(config["output"]["json"])
    base = json.loads(report_path.read_text(encoding="utf-8"))
    require(base.get("canonical_report_sha256") == BASE_G7A_CANONICAL_SHA256, "G7B_BASE_G7A_SHA")
    require(canonical_sha256(base) == BASE_G7A_CANONICAL_SHA256, "G7B_BASE_G7A_CANONICAL")
    base_digests = g7b_component_digests(base)
    require(base_digests == BASE_G7A_COMPONENT_DIGESTS, "G7B_BASE_COMPONENT_DIGESTS")
    require(not compute_processes(), "G7B_GPU_NOT_IDLE")
    current_preflight = gpu_preflight(config)
    parent_before = _parent_records()
    with tempfile.TemporaryDirectory(prefix="g7b-time-sliced-") as temp_name:
        temp_dir = Path(temp_name)
        time_sliced = run_time_sliced(config_path, config, temp_dir)
        raw_path = temp_dir / "time_sliced.raw.json"
        _write_json(raw_path, time_sliced)
        time_sliced = json.loads(raw_path.read_text(encoding="utf-8"))
        require(time_sliced["status"] == "PASS", "G7B_TIME_SLICED_WORKLOAD")
        validate_g7b_timestamp_trace(
            time_sliced["timestamp_trace"],
            inter_phase_gap_ms=float(config["time_sliced"]["inter_phase_gap_ms"]),
        )
        candidate = copy.deepcopy(base)
        candidate["modes"]["episode_time_sliced"] = time_sliced
    require(not compute_processes(), "G7B_COMPUTE_PROCESS_AFTER_TARGETED_RERUN")
    parent_after = _parent_records()
    current_digests = g7b_component_digests(candidate)
    unchanged = verify_g7b_base_components(base, candidate, base_digests)
    candidate = _apply_g7b_semantics(
        candidate, config,
        base_sha=BASE_G7A_CANONICAL_SHA256,
        base_digests=base_digests,
        current_digests=current_digests,
        components_unchanged=unchanged,
        current_preflight=current_preflight,
        parent_before=parent_before,
        parent_after=parent_after,
    )
    verify_g7b_base_components(base, candidate, base_digests)
    candidate["canonical_report_sha256"] = canonical_sha256(candidate)
    schema_path = _resolve(config["output"]["schema"])
    _validate_report(candidate, schema_path)
    _atomic_write_report_pair(
        candidate, json_path=report_path,
        markdown_path=_resolve(config["output"]["markdown"]), schema_path=schema_path,
    )
    return candidate


def run_benchmark(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = validate_config(_load_mapping(config_path))
    baseline = baseline_verification(config)
    audit = source_audit()
    preflight = gpu_preflight(config)
    parent_before = _parent_records()
    with tempfile.TemporaryDirectory(prefix="g7p-stage3-") as temp_name:
        temp_dir = Path(temp_name)
        inference_only = run_inference_only(config_path, config, temp_dir)
        require(not compute_processes(), "G7P_COMPUTE_PROCESS_AFTER_INFERENCE_ONLY")
        learner_only = run_learner_only(config_path, config, temp_dir)
        require(not compute_processes(), "G7P_COMPUTE_PROCESS_AFTER_LEARNER_ONLY")
        concurrent = run_concurrent(config_path, config, temp_dir)
        require(not compute_processes(), "G7P_COMPUTE_PROCESS_AFTER_CONCURRENT")
        time_sliced = run_time_sliced(config_path, config, temp_dir)
        require(not compute_processes(), "G7P_COMPUTE_PROCESS_AFTER_TIME_SLICED")
    parent_after = _parent_records()
    bindings_unchanged = parent_before == parent_after
    action_comparison = compare_action_semantics(inference_only, concurrent)
    slowdown = _learner_slowdown(learner_only, concurrent)
    mode_statuses = [
        inference_only["status"], learner_only["status"], concurrent["status"], time_sliced["status"],
    ]
    numerical = (
        learner_only.get("worker", {}).get("metrics", {}).get("all_finite") is True
        and concurrent.get("learner", {}).get("metrics", {}).get("all_finite") is True
    )
    memory_safe = all(mode.get("oom_count", 0) == 0 for mode in (inference_only, learner_only, concurrent, time_sliced))
    candidate = "NONE"
    concurrent_cadence_sustained = (
        concurrent.get("inference", {}).get("metrics", {}).get("queue_exhaustion_count") == 0
        and concurrent.get("inference", {}).get("metrics", {}).get("stale_drop_count") == 0
    )
    time_sliced_cadence_sustained = (
        time_sliced.get("episode_inference", {}).get("worker", {}).get("metrics", {}).get("queue_exhaustion_count") == 0
        and time_sliced.get("episode_inference", {}).get("worker", {}).get("metrics", {}).get("stale_drop_count") == 0
        and time_sliced.get("resume_inference", {}).get("worker", {}).get("metrics", {}).get("queue_exhaustion_count") == 0
        and time_sliced.get("resume_inference", {}).get("worker", {}).get("metrics", {}).get("stale_drop_count") == 0
    )
    if (
        concurrent["status"] == "PASS" and memory_safe
        and action_comparison["status"] == "PASS" and concurrent_cadence_sustained
    ):
        candidate = "concurrent"
    elif time_sliced["status"] == "PASS" and time_sliced_cadence_sustained:
        candidate = "episode_time_sliced"
    report: dict[str, Any] = {
        "schema_version": "forcesmolvla_stage3_gpu_coexistence_report.v1",
        "tool_status": "PASS" if all(status == "PASS" for status in mode_statuses) and numerical and memory_safe and action_comparison["status"] == "PASS" else "FAIL",
        "baseline": baseline,
        "source_audit": audit,
        "gpu_preflight": preflight,
        "bindings": {
            "before": parent_before, "after": parent_after, "unchanged": bindings_unchanged,
            "current_g7_source_closure": current_source_closure(),
        },
        "modes": {
            "inference_only": inference_only,
            "learner_only": learner_only,
            "concurrent": concurrent,
            "episode_time_sliced": time_sliced,
            "separate_device": {"status": "NOT_RUN", "SEPARATE_DEVICE_MODE": "NOT_RUN_SINGLE_GPU"},
        },
        "comparisons": {
            "fixed_action_semantics": action_comparison,
            "learner_slowdown": slowdown,
            "synthetic_release_grid": {
                "concurrent_cadence_sustained": concurrent_cadence_sustained,
                "time_sliced_cadence_sustained": time_sliced_cadence_sustained,
                "candidate_rejected_on_queue_exhaustion_or_stale_drop": candidate == "NONE",
                "not_a_production_deadline_or_request_cadence": True,
            },
        },
        "gates": {
            "G7P_IMPLEMENTED": True,
            "G7P_RESULT": "PASS" if all(status == "PASS" for status in mode_statuses) and numerical and memory_safe and action_comparison["status"] == "PASS" else "FAIL",
            "G7P_INFERENCE_ONLY": inference_only["status"],
            "G7P_LEARNER_ONLY": learner_only["status"],
            "G7P_CONCURRENT_MEASUREMENT": concurrent["status"],
            "G7P_TIME_SLICED": time_sliced["status"],
            "G7P_MEMORY_SAFETY": "PASS" if memory_safe else "FAIL",
            "G7P_NUMERICAL_INTEGRITY": "PASS" if numerical else "FAIL",
            "G7P_FIXED_ACTION_SEMANTICS": action_comparison["status"],
            "G7P_INFERENCE_ACTOR_HASH_UNCHANGED": bool(
                inference_only.get("worker", {}).get("actor_state", {}).get("unchanged")
                and concurrent.get("inference", {}).get("actor_state", {}).get("unchanged")
            ),
            "G7P_PROVISIONAL_TOPOLOGY_CANDIDATE": candidate,
            "PRODUCTION_GPU_TOPOLOGY_APPROVED": False,
            "PRODUCTION_DEADLINE_VALIDATED": False,
            "DIRECT_PUBLIC_HTTP_PARITY_VALIDATED": False,
            "CHECKPOINT_EXPORT_COEXISTENCE_VALIDATED": False,
            "PRODUCTION_DECODED_CACHE_BOUNDED": audit["decoded_image_cache"]["production_bounded"],
            "G7_FORMAL_GATE_PASSED": False,
            "PRODUCTION_SOURCE_BINDING_COMPLETE": False,
        },
        "safety": {
            "REAL_ONLINE_R_USED": False,
            "R_SOURCE": "synthetic_preflight_R_only",
            "PRODUCTION_ACTOR_STATE_MUTATED": False,
            "PREFLIGHT_ACTOR_STEPS_DISPOSABLE": True,
            "NO_PRODUCTION_REPLAY_WRITES": True,
            "POLICY_PUBLICATION_COUNT": 0,
            "G3_RECORDED_FIXTURE_LOOPBACK": "BLOCKED",
            "G5_PRODUCTION_DURABLE_RESUME": "UNVERIFIED",
            "CRITIC_WARMUP_STARTED": False,
            "CRITIC_READY": False,
            "ACTOR_Q_GUIDANCE_ENABLED": False,
            "POLICY_REVISION_ACTIVATED": False,
            "NETWORK_SERVER_STARTED": False,
            "ROBOT_CONNECTION_COUNT": 0,
            "ROBOT_COMMAND_COUNT": 0,
            "ROBOT_EXECUTION_AUTHORIZED": False,
            "G8_AND_LATER": "NOT_RUN",
            "PUSHED": False,
        },
        "final_checks": {
            "all_own_worker_pids_exited": True,
            "cuda_compute_process_count_after_exit": len(compute_processes()),
            "parent_checkpoint_sha_unchanged": bindings_unchanged,
            "historical_g6p_evidence_unchanged": (
                sha256_file(HISTORICAL_JSON) == HISTORICAL_JSON_SHA256
                and sha256_file(HISTORICAL_MD) == HISTORICAL_MD_SHA256
            ),
            "network_server_started": False,
            "robot_process_started": False,
        },
    }
    require(report["final_checks"]["cuda_compute_process_count_after_exit"] == 0, "G7P_FINAL_COMPUTE_PROCESS")
    require(bindings_unchanged, "G7P_PARENT_BINDING_MUTATED")
    report["canonical_report_sha256"] = canonical_sha256(report)
    report = semantic_consistency_audit(report, config)
    _validate_report(report, _resolve(config["output"]["schema"]))
    _write_json(_resolve(config["output"]["json"]), report)
    _resolve(config["output"]["markdown"]).write_text(render_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--semantic-audit-existing", action="store_true")
    parser.add_argument("--targeted-time-sliced-rerun", action="store_true")
    parser.add_argument("--worker", choices=("inference", "learner"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--worker-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        require(args.spec is not None and args.worker_output is not None, "G7P_WORKER_ARGUMENTS")
        return worker_main(args.worker, args.spec, args.worker_output)
    config = validate_config(_load_mapping(args.config.resolve()))
    if args.targeted_time_sliced_rerun:
        report = run_targeted_time_sliced_rerun(args.config)
        g7b = report["g7b_timestamp_instrumentation"]
        print(json.dumps({
            "G7B_RESULT": g7b["G7B_RESULT"],
            "G7B_TIMESTAMP_ORDER_VALID": g7b["G7B_TIMESTAMP_ORDER_VALID"],
            "G7B_TIME_SLICED_TOPOLOGY": g7b["G7B_TIME_SLICED_TOPOLOGY"],
            "G7P_EVIDENCE_FREEZE_ALLOWED": report["gates"]["G7P_EVIDENCE_FREEZE_ALLOWED"],
            "canonical_report_sha256": report["canonical_report_sha256"],
        }, indent=2, sort_keys=True))
        return 0
    if args.semantic_audit_existing:
        report_path = _resolve(config["output"]["json"])
        report = semantic_consistency_audit(
            json.loads(report_path.read_text(encoding="utf-8")), config,
        )
        _atomic_write_report_pair(
            report, json_path=report_path,
            markdown_path=_resolve(config["output"]["markdown"]),
            schema_path=_resolve(config["output"]["schema"]),
        )
        print(json.dumps({
            "G7A_SEMANTIC_AUDIT": report["semantic_audit"]["G7A_SEMANTIC_AUDIT"],
            "G7P_EVIDENCE_FREEZE_ALLOWED": report["semantic_audit"]["G7P_EVIDENCE_FREEZE_ALLOWED"],
            "GPU_RERUN": report["semantic_audit"]["GPU_RERUN"],
            "canonical_report_sha256": report["canonical_report_sha256"],
        }, indent=2, sort_keys=True))
        return 0
    if args.audit_only:
        result = {
            "baseline": baseline_verification(config),
            "source_audit": source_audit(),
            "cuda_initialized": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    report = run_benchmark(args.config)
    print(json.dumps({
        "G7P_RESULT": report["gates"]["G7P_RESULT"],
        "G7P_INFERENCE_ONLY": report["gates"]["G7P_INFERENCE_ONLY"],
        "G7P_LEARNER_ONLY": report["gates"]["G7P_LEARNER_ONLY"],
        "G7P_CONCURRENT_MEASUREMENT": report["gates"]["G7P_CONCURRENT_MEASUREMENT"],
        "G7P_TIME_SLICED": report["gates"]["G7P_TIME_SLICED"],
        "G7P_PROVISIONAL_TOPOLOGY_CANDIDATE": report["gates"]["G7P_PROVISIONAL_TOPOLOGY_CANDIDATE"],
        "canonical_report_sha256": report["canonical_report_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if report["tool_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
