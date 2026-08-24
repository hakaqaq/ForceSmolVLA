#!/usr/bin/env python3
"""Audit causal reference_ack to accepted_reference pose/gripper association."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    args = parser.parse_args()
    position_errors = []
    angle_errors = []
    signed_deltas_ms = []
    rejected = 0
    for episode in sorted((args.raw_root / "episodes").glob("episode_*")):
        streams = episode / "streams"
        references = read_jsonl(streams / "accepted_reference.jsonl")
        acknowledgements = read_jsonl(streams / "reference_ack.jsonl")
        reference_times = np.asarray(
            [item["accepted_receive_monotonic_ns"] for item in references], dtype=np.int64
        )
        if np.any(np.diff(reference_times) <= 0):
            raise ValueError(f"nonmonotonic accepted_reference: {episode.name}")
        for acknowledgement in acknowledgements:
            if acknowledgement["payload"]["accepted"] is not True:
                rejected += 1
                continue
            ack_time = int(acknowledgement["receive_monotonic_ns"])
            index = int(np.searchsorted(reference_times, ack_time, side="right")) - 1
            if index < 0:
                raise ValueError(f"ack has no causal accepted_reference: {episode.name}")
            accepted_pose = acknowledgement["payload"]["accepted_pose"]
            reference_pose = references[index]["pose"]
            position_errors.append(float(np.linalg.norm(
                np.asarray(accepted_pose["position_m"]) - np.asarray(reference_pose["position_m"])
            )))
            first = np.asarray(accepted_pose["quaternion_xyzw"], dtype=np.float64)
            second = np.asarray(reference_pose["quaternion_xyzw"], dtype=np.float64)
            first /= np.linalg.norm(first)
            second /= np.linalg.norm(second)
            angle_errors.append(float(2 * np.arccos(np.clip(abs(np.dot(first, second)), -1, 1))))
            signed_deltas_ms.append((int(reference_times[index]) - ack_time) / 1_000_000.0)
            width = float(references[index]["target_gripper_width_m"])
            if not np.isfinite(width) or not 0 <= width <= 0.1:
                raise ValueError("associated target_gripper_width_m is invalid")
    print(json.dumps({
        "status": "development_only_measurement",
        "association_candidate": (
            "for each accepted reference_ack, select latest accepted_reference."
            "accepted_receive_monotonic_ns <= reference_ack.receive_monotonic_ns; "
            "require accepted pose equality, take pose and target_gripper_width_m from that record"
        ),
        "accepted_acknowledgements": len(position_errors),
        "rejected_acknowledgements": rejected,
        "position_error_m": stats(position_errors),
        "quaternion_geodesic_error_rad": stats(angle_errors),
        "reference_minus_ack_ms": stats(signed_deltas_ms),
        "future_reference_associations": int(np.sum(np.asarray(signed_deltas_ms) > 0)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
