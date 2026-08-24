#!/usr/bin/env python3
"""Direct-raw causal geometry smoke; deliberately stops before filtering/resampling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from forcesmolvla.geometry import (
    StaticWrenchCalibration,
    calibrated_tcp_wrench_conditioned_on_measured_tcp_pose,
)
from forcesmolvla.temporal import match_measured_tcp_pose_causal_zoh


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-pose-age-ms", type=float, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    episode = args.raw_root / "episodes" / f"episode_{args.episode:06d}"
    poses = read_jsonl(episode / "streams" / "measured_tcp_pose.jsonl")
    wrenches = read_jsonl(episode / "streams" / "wrench_notch_sensor.jsonl")[: args.limit]
    pose_stamps = np.asarray([item["source_stamp_ns"] for item in poses], dtype=np.int64)
    wrench_stamps = np.asarray([item["source_stamp_ns"] for item in wrenches], dtype=np.int64)
    matches = match_measured_tcp_pose_causal_zoh(
        pose_stamps, wrench_stamps, max_pose_age_ms=args.max_pose_age_ms
    )
    bundle = json.loads(
        (Path(__file__).parents[1] / "configs" / "calibration_bundle.development.json").read_text()
    )
    calibration = StaticWrenchCalibration(
        calibration_id=bundle["calibration_id"],
        translation_tcp_sensor_m=bundle["static_transform_tcp_sensor"]["translation_m"],
        quaternion_tcp_sensor_xyzw=bundle["static_transform_tcp_sensor"]["quaternion_xyzw"],
        sensor_bias6=bundle["sensor_bias6"],
        wrench_sign6=bundle["wrench_sign6"],
        downstream_mass_kg=bundle["downstream_mass_kg"],
        downstream_com_sensor_m=bundle["downstream_com_sensor_m"],
        gravity_base_m_s2=bundle["gravity_base_m_s2"],
    )
    outputs = []
    selected_pose_stamps = []
    selected_wrench_stamps = []
    for index, valid in enumerate(matches.valid):
        if not valid:
            continue
        pose_index = int(matches.pose_indices[index])
        pose = poses[pose_index]["pose"]
        wrench = wrenches[index]
        result = calibrated_tcp_wrench_conditioned_on_measured_tcp_pose(
            wrench["force_xyz_n_torque_xyz_nm"],
            pose["position_m"],
            pose["quaternion_xyzw"],
            calibration,
        )
        outputs.append(result.wrench_base_at_tcp6)
        selected_pose_stamps.append(pose_stamps[pose_index])
        selected_wrench_stamps.append(wrench_stamps[index])
    output = np.asarray(outputs, dtype="<f8")
    selected_pose_stamps = np.asarray(selected_pose_stamps, dtype=np.int64)
    selected_wrench_stamps = np.asarray(selected_wrench_stamps, dtype=np.int64)
    if np.any(selected_pose_stamps > selected_wrench_stamps):
        raise AssertionError("future pose selected")
    if not np.all(np.isfinite(output)):
        raise AssertionError("nonfinite calibrated wrench")
    print(json.dumps({
        "status": "development_only_smoke_pass",
        "raw_root": str(args.raw_root.resolve()),
        "episode": episode.name,
        "raw_wrench_records_examined": len(wrenches),
        "valid_geometry_records": len(output),
        "invalid_geometry_records": int((~matches.valid).sum()),
        "max_pose_age_ms_candidate": args.max_pose_age_ms,
        "maximum_selected_pose_age_ms": float(np.nanmax(matches.pose_age_ms[matches.valid])),
        "future_pose_count": 0,
        "wording": "calibrated TCP wrench conditioned on measured TCP pose",
        "calibration_id": calibration.calibration_id,
        "output_float64_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "filter_or_resample_applied": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
