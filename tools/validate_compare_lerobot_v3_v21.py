#!/usr/bin/env python3
"""Validate ForceSmolVLA task1 v3 and compare it with legacy ForceVLA v2.1."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from forcesmolvla.raw_to_lerobot_v3 import source_tree_manifest


V3_NUMERIC_COLUMNS = [
    "observation.state",
    "observation.wrench",
    "action",
    "provenance.tuple_host_monotonic_ns",
    "provenance.state_pose_source_stamp_ns",
    "provenance.state_pose_age_ms",
    "provenance.camera1_receive_monotonic_ns",
    "provenance.camera1_age_ms",
    "provenance.camera2_receive_monotonic_ns",
    "provenance.camera2_age_ms",
    "provenance.intercamera_skew_ms",
    "provenance.gripper_source_stamp_ns",
    "provenance.pose_source_stamp_ns",
    "provenance.pose_age_ms",
    "provenance.wrench_raw_source_stamp_ns",
    "provenance.wrench_filter_output_stamp_ns",
    "provenance.action_ack_receive_monotonic_ns",
    "provenance.action_ack_age_ms",
    "provenance.calibration_index",
    "provenance.validity_bits",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_list(table, name: str, width: int) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=np.float64).reshape(-1, width)


def scalar(table, name: str, dtype=None) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=dtype)


def describe(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(len(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def per_dimension_abs_difference(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    difference = np.abs(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64))
    return {
        "aggregate": describe(difference),
        "per_dimension": [describe(difference[:, index]) for index in range(difference.shape[1])],
        "exact_row_fraction": float(np.mean(np.all(difference == 0, axis=1))),
    }


def nearest_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    insertion = np.searchsorted(source_ns, target_ns, side="left")
    right = np.minimum(insertion, len(source_ns) - 1)
    left = np.maximum(insertion - 1, 0)
    choose_right = np.abs(source_ns[right] - target_ns) < np.abs(source_ns[left] - target_ns)
    indices = np.where(choose_right, right, left)
    return indices, np.abs(source_ns[indices] - target_ns)


def decode_image(entry: dict[str, Any]) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(entry["bytes"])).convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--v21-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rehash-source", action="store_true")
    args = parser.parse_args()

    v3_info = load_json(args.v3_root / "meta/info.json")
    v2_info = load_json(args.v21_root / "meta/info.json")
    v3_manifest = load_json(args.v3_root / "conversion_manifest.json")
    v2_manifest = load_json(args.v21_root / "conversion_manifest.json")
    split = load_json(args.v3_root / "split_manifest.json")
    normalizer = load_json(args.v3_root / "normalizer_manifest.json")

    errors: list[str] = []
    warnings: list[str] = []
    if v3_info.get("codebase_version") != "v3.0":
        errors.append("V3_CODEBASE_VERSION_MISMATCH")
    if v3_info.get("total_episodes") != 50:
        errors.append("V3_EPISODE_COUNT_MISMATCH")
    if v3_manifest.get("artifact_status") != "development_only":
        errors.append("V3_DEVELOPMENT_STATUS_MISSING")
    if v3_manifest.get("formal_ready") is not False:
        errors.append("V3_FORMAL_READY_MUST_BE_FALSE")

    split_sets = {name: set(split[name]) for name in ("train", "val", "test")}
    if (
        split_sets["train"] & split_sets["val"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["val"] & split_sets["test"]
    ):
        errors.append("SPLIT_NOT_DISJOINT")
    if tuple(map(len, (split_sets["train"], split_sets["val"], split_sets["test"]))) != (
        40,
        5,
        5,
    ):
        errors.append("SPLIT_COUNT_MISMATCH")
    normalizer_episode_sets = {
        name: set(payload["fit_episode_ids"])
        for name, payload in normalizer["features"].items()
    }
    if any(value != split_sets["train"] for value in normalizer_episode_sets.values()):
        errors.append("NORMALIZER_NOT_TRAIN_ONLY")
    if v3_info.get("splits") == {"train": "0:50"}:
        warnings.append(
            "LEROBOT_STORAGE_META_EXPOSES_ALL_EPISODES_AS_TRAIN; "
            "ForceSmolVLA loader must enforce split_manifest.json"
        )

    v2_entries = {entry["raw_episode"]: entry for entry in v2_manifest["episodes"]}
    v3_entries = {entry["raw_episode_id"]: entry for entry in v3_manifest["episodes"]}
    if set(v2_entries) != set(v3_entries):
        errors.append("RAW_EPISODE_MAPPING_MISMATCH")

    all_state: list[np.ndarray] = []
    all_wrench: list[np.ndarray] = []
    all_action: list[np.ndarray] = []
    all_pose_age: list[np.ndarray] = []
    all_state_pose_age: list[np.ndarray] = []
    all_camera1_age: list[np.ndarray] = []
    all_camera2_age: list[np.ndarray] = []
    all_camera_skew: list[np.ndarray] = []
    all_action_age: list[np.ndarray] = []
    aligned_v3_state: list[np.ndarray] = []
    aligned_v2_state: list[np.ndarray] = []
    aligned_v3_wrench: list[np.ndarray] = []
    aligned_v2_wrench: list[np.ndarray] = []
    aligned_v3_action: list[np.ndarray] = []
    aligned_v2_action: list[np.ndarray] = []
    alignment_error_ns: list[np.ndarray] = []
    per_episode: list[dict[str, Any]] = []
    expected_global_index = 0

    v3_data_files = sorted((args.v3_root / "data").rglob("file-*.parquet"))
    v2_data_files = sorted((args.v21_root / "data").rglob("episode_*.parquet"))
    if len(v3_data_files) != 50 or len(v2_data_files) != 50:
        errors.append("PARQUET_FILE_COUNT_MISMATCH")

    for episode_index in range(50):
        episode_id = f"episode_{episode_index:06d}"
        v3_path = args.v3_root / "data/chunk-000" / f"file-{episode_index:03d}.parquet"
        v2_path = args.v21_root / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
        v3_table = pq.read_table(v3_path, columns=V3_NUMERIC_COLUMNS)
        v2_table = pq.read_table(v2_path, columns=["observation.state", "action"])
        row_count = len(v3_table)
        expected_rows = int(v3_entries[episode_id]["frames"])
        if row_count != expected_rows:
            errors.append(f"{episode_id}:V3_ROW_COUNT_MISMATCH")

        state = fixed_list(v3_table, "observation.state", 7)
        wrench = fixed_list(v3_table, "observation.wrench", 6)
        action = fixed_list(v3_table, "action", 7)
        all_state.append(state)
        all_wrench.append(wrench)
        all_action.append(action)
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(wrench)) or not np.all(
            np.isfinite(action)
        ):
            errors.append(f"{episode_id}:NONFINITE_FEATURE")
        if np.any((state[:, 6] < 0) | (state[:, 6] > 0.1)) or np.any(
            (action[:, 6] < 0) | (action[:, 6] > 0.1)
        ):
            errors.append(f"{episode_id}:GRIPPER_RANGE")

        tuple_ns = scalar(v3_table, "provenance.tuple_host_monotonic_ns", np.int64)
        state_pose_age = scalar(v3_table, "provenance.state_pose_age_ms", np.float64)
        camera1_ns = scalar(v3_table, "provenance.camera1_receive_monotonic_ns", np.int64)
        camera1_age = scalar(v3_table, "provenance.camera1_age_ms", np.float64)
        camera2_ns = scalar(v3_table, "provenance.camera2_receive_monotonic_ns", np.int64)
        camera2_age = scalar(v3_table, "provenance.camera2_age_ms", np.float64)
        camera_skew = scalar(v3_table, "provenance.intercamera_skew_ms", np.float64)
        pose_ns = scalar(v3_table, "provenance.pose_source_stamp_ns", np.int64)
        pose_age = scalar(v3_table, "provenance.pose_age_ms", np.float64)
        wrench_raw_ns = scalar(v3_table, "provenance.wrench_raw_source_stamp_ns", np.int64)
        wrench_filter_ns = scalar(
            v3_table, "provenance.wrench_filter_output_stamp_ns", np.int64
        )
        ack_ns = scalar(v3_table, "provenance.action_ack_receive_monotonic_ns", np.int64)
        ack_age = scalar(v3_table, "provenance.action_ack_age_ms", np.float64)
        calibration_index = scalar(v3_table, "provenance.calibration_index", np.int64)
        validity = scalar(v3_table, "provenance.validity_bits", np.int64)
        frame_index = scalar(v3_table, "frame_index", np.int64)
        output_episode = scalar(v3_table, "episode_index", np.int64)
        global_index = scalar(v3_table, "index", np.int64)
        timestamp = scalar(v3_table, "timestamp", np.float64)

        if np.any(np.diff(tuple_ns) <= 0) or not np.all(np.isin(np.diff(tuple_ns), [33333333, 33333334])):
            errors.append(f"{episode_id}:GRID_NOT_RATIONAL_30HZ")
        if not np.array_equal(frame_index, np.arange(row_count)):
            errors.append(f"{episode_id}:FRAME_INDEX_NOT_CONTIGUOUS")
        if not np.all(output_episode == episode_index):
            errors.append(f"{episode_id}:EPISODE_INDEX_MISMATCH")
        if not np.array_equal(global_index, np.arange(expected_global_index, expected_global_index + row_count)):
            errors.append(f"{episode_id}:GLOBAL_INDEX_NOT_CONTIGUOUS")
        expected_global_index += row_count
        if not np.allclose(timestamp, frame_index / 30.0, atol=2e-6, rtol=0):
            errors.append(f"{episode_id}:TIMESTAMP_MISMATCH")
        if np.any(camera1_ns > tuple_ns) or np.any(camera2_ns > tuple_ns) or np.any(ack_ns > tuple_ns):
            errors.append(f"{episode_id}:FUTURE_HOST_SAMPLE")
        if np.any(pose_ns > wrench_raw_ns):
            errors.append(f"{episode_id}:FUTURE_GEOMETRY_POSE")
        if not np.array_equal(wrench_raw_ns, wrench_filter_ns):
            errors.append(f"{episode_id}:FILTER_TIMESTAMP_CHANGED")
        if not np.allclose(camera1_age, (tuple_ns - camera1_ns) / 1e6, atol=1e-4):
            errors.append(f"{episode_id}:CAMERA1_AGE_MISMATCH")
        if not np.allclose(camera2_age, (tuple_ns - camera2_ns) / 1e6, atol=1e-4):
            errors.append(f"{episode_id}:CAMERA2_AGE_MISMATCH")
        if not np.allclose(camera_skew, np.abs(camera1_ns - camera2_ns) / 1e6, atol=1e-4):
            errors.append(f"{episode_id}:CAMERA_SKEW_MISMATCH")
        if not np.allclose(pose_age, (wrench_raw_ns - pose_ns) / 1e6, atol=1e-4):
            errors.append(f"{episode_id}:POSE_AGE_MISMATCH")
        if not np.allclose(ack_age, (tuple_ns - ack_ns) / 1e6, atol=1e-4):
            errors.append(f"{episode_id}:ACK_AGE_MISMATCH")
        if np.any(pose_age > 12.0) or np.any(state_pose_age > 12.0):
            errors.append(f"{episode_id}:POSE_CANDIDATE_THRESHOLD")
        if np.any(camera1_age > 34.0) or np.any(camera2_age > 34.0):
            errors.append(f"{episode_id}:CAMERA_AGE_CANDIDATE_THRESHOLD")
        if np.any(camera_skew > 33.0):
            errors.append(f"{episode_id}:CAMERA_SKEW_CANDIDATE_THRESHOLD")
        if not np.all(calibration_index == 0) or not np.all(validity == 255):
            errors.append(f"{episode_id}:VALIDITY_OR_CALIBRATION")

        all_pose_age.append(pose_age)
        all_state_pose_age.append(state_pose_age)
        all_camera1_age.append(camera1_age)
        all_camera2_age.append(camera2_age)
        all_camera_skew.append(camera_skew)
        all_action_age.append(ack_age)

        v2_state = fixed_list(v2_table, "observation.state", 13)
        v2_action = fixed_list(v2_table, "action", 7)
        v2_start_ns = int(v2_entries[episode_id]["alignment"]["timeline_start_monotonic_ns"])
        v2_host_ns = v2_start_ns + np.arange(len(v2_state), dtype=np.int64) * 33333333
        nearest, time_error = nearest_indices(v2_host_ns, tuple_ns)
        valid_alignment = time_error <= 16_666_667
        if not np.all(valid_alignment):
            errors.append(f"{episode_id}:NO_NEAREST_V21_FRAME")
        nearest = nearest[valid_alignment]
        aligned_v3_state.append(state[valid_alignment])
        aligned_v2_state.append(v2_state[nearest, :7])
        aligned_v3_wrench.append(wrench[valid_alignment])
        aligned_v2_wrench.append(v2_state[nearest, 7:13])
        aligned_v3_action.append(action[valid_alignment])
        aligned_v2_action.append(v2_action[nearest])
        alignment_error_ns.append(time_error[valid_alignment])
        per_episode.append(
            {
                "episode": episode_id,
                "v21_frames": int(len(v2_state)),
                "v3_frames": row_count,
                "frame_delta_v3_minus_v21": row_count - int(len(v2_state)),
                "nearest_timeline_error_ms_max": float(np.max(time_error)) / 1e6,
            }
        )

    state_array = np.concatenate(all_state)
    wrench_array = np.concatenate(all_wrench)
    action_array = np.concatenate(all_action)
    v3_state_aligned = np.concatenate(aligned_v3_state)
    v2_state_aligned = np.concatenate(aligned_v2_state)
    v3_wrench_aligned = np.concatenate(aligned_v3_wrench)
    v2_wrench_aligned = np.concatenate(aligned_v2_wrench)
    v3_action_aligned = np.concatenate(aligned_v3_action)
    v2_action_aligned = np.concatenate(aligned_v2_action)

    image_comparison: dict[str, Any] = {}
    sample_episodes = [0, 12, 25, 37, 49]
    for v3_name, v2_name in (
        ("observation.images.camera1", "observation.image"),
        ("observation.images.camera2", "observation.wrist_image"),
    ):
        exact = 0
        pixel_abs: list[np.ndarray] = []
        compared = 0
        for episode_index in sample_episodes:
            episode_id = f"episode_{episode_index:06d}"
            v3_path = args.v3_root / "data/chunk-000" / f"file-{episode_index:03d}.parquet"
            v2_path = args.v21_root / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
            v3_table = pq.read_table(
                v3_path, columns=[v3_name, "provenance.tuple_host_monotonic_ns"]
            )
            v2_table = pq.read_table(v2_path, columns=[v2_name])
            tuple_ns = scalar(v3_table, "provenance.tuple_host_monotonic_ns", np.int64)
            v2_start_ns = int(v2_entries[episode_id]["alignment"]["timeline_start_monotonic_ns"])
            v2_host_ns = v2_start_ns + np.arange(len(v2_table), dtype=np.int64) * 33333333
            positions = np.unique(np.linspace(0, len(v3_table) - 1, 5, dtype=int))
            nearest, _ = nearest_indices(v2_host_ns, tuple_ns[positions])
            for v3_index, v2_index in zip(positions, nearest, strict=True):
                first = decode_image(v3_table[v3_name][int(v3_index)].as_py())
                second = decode_image(v2_table[v2_name][int(v2_index)].as_py())
                difference = np.abs(first.astype(np.int16) - second.astype(np.int16))
                pixel_abs.append(difference.reshape(-1))
                exact += int(np.array_equal(first, second))
                compared += 1
        values = np.concatenate(pixel_abs)
        image_comparison[v3_name] = {
            "sampled_frame_pairs": compared,
            "exact_frame_fraction": exact / compared,
            "pixel_abs_difference": describe(values),
        }

    source_verification: dict[str, Any] = {
        "manifest_root_sha256": v3_manifest["raw_source_tree_sha256"],
        "full_rehash_performed": args.rehash_source,
    }
    if args.rehash_source:
        _, rehashed = source_tree_manifest(args.raw_root, progress_every=10000)
        source_verification["rehashed_root_sha256"] = rehashed
        source_verification["matches"] = rehashed == v3_manifest["raw_source_tree_sha256"]
        if not source_verification["matches"]:
            errors.append("RAW_SOURCE_TREE_HASH_MISMATCH")

    report = {
        "status": "pass" if not errors else "fail",
        "artifact_status": "development_only",
        "v3_root": str(args.v3_root.resolve()),
        "v21_root": str(args.v21_root.resolve()),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "v3_validation": {
            "episodes": v3_info["total_episodes"],
            "frames": v3_info["total_frames"],
            "all_numeric_features_finite": not any("NONFINITE" in error for error in errors),
            "causal_provenance_checks": not any(
                "FUTURE" in error or "AGE_MISMATCH" in error for error in errors
            ),
            "state7_range": {"min": state_array.min(axis=0).tolist(), "max": state_array.max(axis=0).tolist()},
            "wrench6_range": {"min": wrench_array.min(axis=0).tolist(), "max": wrench_array.max(axis=0).tolist()},
            "action7_range": {"min": action_array.min(axis=0).tolist(), "max": action_array.max(axis=0).tolist()},
            "pose_age_ms": describe(np.concatenate(all_pose_age)),
            "state_pose_age_ms": describe(np.concatenate(all_state_pose_age)),
            "camera1_age_ms": describe(np.concatenate(all_camera1_age)),
            "camera2_age_ms": describe(np.concatenate(all_camera2_age)),
            "intercamera_skew_ms": describe(np.concatenate(all_camera_skew)),
            "action_ack_age_ms": describe(np.concatenate(all_action_age)),
            "split_counts": {name: len(value) for name, value in split_sets.items()},
            "normalizer_train_only": not any("NORMALIZER" in error for error in errors),
        },
        "source_verification": source_verification,
        "comparison": {
            "only_format_version_difference": False,
            "reasons_not_only_version": [
                "v2.1 stores float64 state13=[state7,wrench6]; v3 stores separate float32 state7 and wrench6",
                "v2.1 uses nearest cameras, interpolated pose/gripper, and linearly regularized/interpolated wrench; v3 uses causal latest-only selection and no future interpolation",
                "v3 resets the causal filter per episode and excludes 250 warm-up samples",
                "v3 actions are acknowledgement-associated before ZOH and include action provenance",
                "v3 uses a global zero-phase rational 30 Hz grid and has additional provenance fields",
                "episode frame counts differ",
            ],
            "v21_frames": v2_info["total_frames"],
            "v3_frames": v3_info["total_frames"],
            "frame_delta_v3_minus_v21": v3_info["total_frames"] - v2_info["total_frames"],
            "nearest_timeline_error_ms": describe(np.concatenate(alignment_error_ns) / 1e6),
            "state7_abs_difference": per_dimension_abs_difference(v3_state_aligned, v2_state_aligned),
            "wrench6_abs_difference": per_dimension_abs_difference(
                v3_wrench_aligned, v2_wrench_aligned
            ),
            "action7_abs_difference": per_dimension_abs_difference(v3_action_aligned, v2_action_aligned),
            "image_sample_comparison": image_comparison,
            "per_episode_frame_counts": per_episode,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "errors": report["errors"],
        "warnings": report["warnings"],
        "v3_frames": v3_info["total_frames"],
        "v21_frames": v2_info["total_frames"],
        "report": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
