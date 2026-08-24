#!/usr/bin/env python3
"""Read-only causal camera-age audit on the proposed controller-clock grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forcesmolvla.temporal import controller_reference_grid, select_latest_causal


def read_field(path: Path, field: str) -> np.ndarray:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(int(json.loads(line)[field]))
    result = np.asarray(values, dtype=np.int64)
    if len(result) > 1 and np.any(np.diff(result) <= 0):
        raise ValueError(f"nonmonotonic {field}: {path}")
    return result


def stats(values: np.ndarray) -> dict:
    return {
        "count": int(len(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    args = parser.parse_args()
    all_external_age = []
    all_wrist_age = []
    all_skew = []
    episodes = []
    for episode in sorted((args.raw_root / "episodes").glob("episode_*")):
        streams = episode / "streams"
        external = read_field(streams / "external_camera.jsonl", "receive_monotonic_ns")
        wrist = read_field(streams / "wrist_camera.jsonl", "receive_monotonic_ns")
        acknowledgements = read_field(streams / "reference_ack.jsonl", "receive_monotonic_ns")
        result = json.loads((episode / "episode_result.json").read_text())
        end_ns = min(int(result["finished_monotonic_ns"]), int(external[-1]), int(wrist[-1]))
        grid = controller_reference_grid(
            session_start_ack_ns=int(acknowledgements[0]),
            episode_end_ns=end_ns,
            fps=30,
        )
        # A deliberately nonbinding bound lets the audit measure observed maxima.
        external_selected = select_latest_causal(external, grid, max_age_ms=1000.0)
        wrist_selected = select_latest_causal(wrist, grid, max_age_ms=1000.0)
        valid = (external_selected.source_indices >= 0) & (wrist_selected.source_indices >= 0)
        external_age = external_selected.age_ms[valid]
        wrist_age = wrist_selected.age_ms[valid]
        skew = np.abs(
            external[external_selected.source_indices[valid]]
            - wrist[wrist_selected.source_indices[valid]]
        ) / 1_000_000.0
        all_external_age.append(external_age)
        all_wrist_age.append(wrist_age)
        all_skew.append(skew)
        episodes.append({
            "episode": episode.name,
            "grid_ticks": int(len(grid)),
            "causal_two_camera_ticks": int(valid.sum()),
            "external_age_ms": stats(external_age),
            "wrist_age_ms": stats(wrist_age),
            "intercamera_skew_ms": stats(skew),
        })
    external_age = np.concatenate(all_external_age)
    wrist_age = np.concatenate(all_wrist_age)
    skew = np.concatenate(all_skew)
    print(json.dumps({
        "status": "development_only_measurement",
        "raw_root": str(args.raw_root.resolve()),
        "camera_selection": "latest receive_monotonic_ns <= t_ref",
        "grid_anchor_candidate": "first reference_ack.receive_monotonic_ns per episode",
        "grid_phase": "global zero-phase rational 30 Hz",
        "future_camera_reads": "forbidden",
        "episodes": episodes,
        "aggregate": {
            "external_age_ms": stats(external_age),
            "wrist_age_ms": stats(wrist_age),
            "intercamera_skew_ms": stats(skew),
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
