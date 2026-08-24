#!/usr/bin/env python3
"""Audit causal TCP-pose age at each raw wrench sample.

This tool is read-only.  It uses source timestamps from the two ROS streams and
selects the newest pose satisfying ``t_pose <= t_wrench`` (causal ZOH).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_timestamps(path: Path) -> tuple[list[int], int, int]:
    stamps: list[int] = []
    nonfinite = 0
    malformed = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
                stamp = item["source_stamp_ns"]
                if not isinstance(stamp, int) or not math.isfinite(float(stamp)):
                    nonfinite += 1
                    continue
                stamps.append(stamp)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                malformed += 1
    return stamps, nonfinite, malformed


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    rank = (len(sorted_values) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    fraction = rank - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def audit_episode(episode_dir: Path) -> tuple[dict[str, Any], list[float]]:
    stream_dir = episode_dir / "streams"
    pose_stamps, pose_nonfinite, pose_malformed = _read_timestamps(
        stream_dir / "measured_tcp_pose.jsonl"
    )
    wrench_stamps, wrench_nonfinite, wrench_malformed = _read_timestamps(
        stream_dir / "wrench_notch_sensor.jsonl"
    )

    pose_monotonic_violations = sum(
        current <= previous for previous, current in zip(pose_stamps, pose_stamps[1:])
    )
    wrench_monotonic_violations = sum(
        current <= previous for previous, current in zip(wrench_stamps, wrench_stamps[1:])
    )

    ages_ms: list[float] = []
    without_prior_pose = 0
    pose_index = -1
    for wrench_stamp in wrench_stamps:
        while pose_index + 1 < len(pose_stamps) and pose_stamps[pose_index + 1] <= wrench_stamp:
            pose_index += 1
        if pose_index < 0:
            without_prior_pose += 1
            continue
        age_ms = (wrench_stamp - pose_stamps[pose_index]) / 1_000_000.0
        if age_ms < 0:
            raise AssertionError("causal selection produced a negative pose age")
        ages_ms.append(age_ms)

    sorted_ages = sorted(ages_ms)
    summary = {
        "episode": episode_dir.name,
        "pose_samples": len(pose_stamps),
        "wrench_samples": len(wrench_stamps),
        "matched_wrench_samples": len(ages_ms),
        "wrench_samples_without_prior_pose": without_prior_pose,
        "pose_source_timestamp_nonfinite": pose_nonfinite,
        "wrench_source_timestamp_nonfinite": wrench_nonfinite,
        "pose_json_malformed": pose_malformed,
        "wrench_json_malformed": wrench_malformed,
        "pose_timestamp_monotonic_violations": pose_monotonic_violations,
        "wrench_timestamp_monotonic_violations": wrench_monotonic_violations,
        "pose_age_ms": {
            "p50": _percentile(sorted_ages, 50.0),
            "p95": _percentile(sorted_ages, 95.0),
            "p99": _percentile(sorted_ages, 99.0),
            "max": sorted_ages[-1] if sorted_ages else None,
        },
    }
    return summary, ages_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    args = parser.parse_args()

    episode_dirs = sorted((args.raw_root / "episodes").glob("episode_*"))
    episode_summaries: list[dict[str, Any]] = []
    all_ages_ms: list[float] = []
    for episode_dir in episode_dirs:
        summary, ages_ms = audit_episode(episode_dir)
        episode_summaries.append(summary)
        all_ages_ms.extend(ages_ms)

    sorted_ages = sorted(all_ages_ms)
    result = {
        "contract": "forcesmolvla-v4.1-available-sensor",
        "selection": "latest measured_tcp_pose.source_stamp_ns <= wrench_notch_sensor.source_stamp_ns",
        "interpolation": "causal-zoh-only",
        "raw_root": str(args.raw_root.resolve()),
        "episodes": len(episode_summaries),
        "totals": {
            "pose_samples": sum(item["pose_samples"] for item in episode_summaries),
            "wrench_samples": sum(item["wrench_samples"] for item in episode_summaries),
            "matched_wrench_samples": len(sorted_ages),
            "wrench_samples_without_prior_pose": sum(
                item["wrench_samples_without_prior_pose"] for item in episode_summaries
            ),
            "pose_timestamp_monotonic_violations": sum(
                item["pose_timestamp_monotonic_violations"] for item in episode_summaries
            ),
            "wrench_timestamp_monotonic_violations": sum(
                item["wrench_timestamp_monotonic_violations"] for item in episode_summaries
            ),
        },
        "pose_age_ms": {
            "p50": _percentile(sorted_ages, 50.0),
            "p95": _percentile(sorted_ages, 95.0),
            "p99": _percentile(sorted_ages, 99.0),
            "max": sorted_ages[-1] if sorted_ages else None,
        },
        "per_episode": episode_summaries,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
