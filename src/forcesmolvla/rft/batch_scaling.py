"""Pure aggregation, selection, and sample-exposure math for batch scaling."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("BATCH_SCALING_DISTRIBUTION_INVALID")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "range": float(array.max() - array.min()),
        "p95": float(np.quantile(array, 0.95)),
    }


def aggregate_repeats(batch: int, repeats: list[Mapping[str, Any]], throughput_key: str) -> dict[str, Any]:
    if len(repeats) < 3 or any(int(item["physical_batch_size"]) != batch for item in repeats):
        raise ValueError("BATCH_SCALING_REPEAT_SET_INVALID")
    passed = [item for item in repeats if item.get("status") == "pass"]
    result = {
        "physical_batch_size": batch,
        "repeat_count": len(repeats),
        "pass_count": len(passed),
        "oom_count": sum(item.get("status") == "oom" for item in repeats),
        "all_pass": len(passed) == len(repeats),
        "all_finite": bool(passed) and all(item.get("all_finite") is True for item in passed),
        "all_contract_valid": bool(passed) and all(item.get("contract_valid") is True for item in passed),
        "repeat_result_paths": [item["result_path"] for item in repeats],
    }
    if passed:
        result.update({
            throughput_key: distribution(float(item[throughput_key]) for item in passed),
            "seconds_per_update": distribution(float(item["seconds_per_update"]["median"]) for item in passed),
            "peak_allocated_bytes": max(int(item["peak_allocated_bytes"]) for item in passed),
            "peak_reserved_bytes": max(int(item["peak_reserved_bytes"]) for item in passed),
            "gpu_utilization_percent": distribution(float(item["gpu_utilization_percent"]["mean"]) for item in passed),
            "gpu_power_watts": distribution(float(item["gpu_power_watts"]["mean"]) for item in passed),
        })
    return result


def eligible(candidate: Mapping[str, Any], *, total_memory_bytes: int, maximum_fraction: float = 0.85) -> bool:
    return bool(
        candidate.get("all_pass")
        and candidate.get("all_finite")
        and candidate.get("all_contract_valid")
        and int(candidate.get("peak_reserved_bytes", total_memory_bytes + 1)) <= int(total_memory_bytes * maximum_fraction)
    )


def select_by_samples_per_second(
    candidates: Iterable[Mapping[str, Any]], throughput_key: str, *,
    total_memory_bytes: int, maximum_fraction: float = 0.85,
    minimum_gain: float = 0.05,
) -> Mapping[str, Any]:
    ordered = sorted(candidates, key=lambda item: int(item["physical_batch_size"]))
    valid = [item for item in ordered if eligible(item, total_memory_bytes=total_memory_bytes, maximum_fraction=maximum_fraction)]
    if not valid:
        raise RuntimeError("NO_ELIGIBLE_BATCH_SCALING_CANDIDATE")
    selected = valid[0]
    previous = valid[0]
    for candidate in valid[1:]:
        previous_rate = float(previous[throughput_key]["median"])
        current_rate = float(candidate[throughput_key]["median"])
        candidate["incremental_gain_over_previous"] = current_rate / previous_rate - 1.0
        if current_rate >= previous_rate * (1.0 + minimum_gain):
            selected = candidate
        previous = candidate
    return selected


def actor_pass_table(train_transitions: int = 10075) -> list[dict[str, int]]:
    result = []
    for batch in (4, 8, 16, 24, 32, 48, 64):
        one = math.ceil(train_transitions / batch)
        result.append({
            "effective_actor_batch": batch,
            "half_pass_updates": math.ceil(0.5 * train_transitions / batch),
            "one_pass_updates": one,
            "two_pass_updates": math.ceil(2.0 * train_transitions / batch),
        })
    return result
