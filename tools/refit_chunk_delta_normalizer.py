#!/usr/bin/env python3
"""Refit an existing development v3 dataset to the ForceVLA chunk-delta contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from forcesmolvla.normalizer import (
    CartesianNormalizerBundle,
    build_action_target_population,
)
from forcesmolvla.split import EpisodeSplit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    project_root = Path(__file__).parents[1].resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "datasets/task2_lerobotv3",
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    normalizer_path = dataset_root / "normalizer_manifest.json"
    conversion_path = dataset_root / "conversion_manifest.json"
    split_path = dataset_root / "split_manifest.json"
    old_normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    if (
        conversion.get("artifact_status") != "development_only"
        or conversion.get("formal_ready") is not False
    ):
        raise RuntimeError("refit is restricted to development-only datasets")
    if old_normalizer.get("action_target_population", {}).get("status") == "pass":
        raise RuntimeError("dataset already has a bound action-target population")

    split = EpisodeSplit(
        tuple(split_payload["train"]),
        tuple(split_payload["val"]),
        tuple(split_payload["test"]),
    )
    split.assert_disjoint()
    mapping = {
        entry["raw_episode_id"]: int(entry["output_episode_index"])
        for entry in conversion["episodes"]
    }
    episodes = []
    state_rows = []
    wrench_rows = []
    sample_episode_ids: list[str] = []
    for raw_episode_id in split.train:
        output_index = mapping[raw_episode_id]
        parquet_path = (
            dataset_root / "data/chunk-000" / f"file-{output_index:03d}.parquet"
        )
        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "observation.wrench", "action", "episode_index"],
        )
        state7 = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        wrench6 = np.asarray(table["observation.wrench"].to_pylist(), dtype=np.float64)
        action7 = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        if (
            state7.shape != action7.shape
            or state7.ndim != 2
            or state7.shape[1] != 7
            or wrench6.shape != (len(state7), 6)
            or not np.all(episode_indices == output_index)
        ):
            raise RuntimeError(f"unexpected parquet contract: {parquet_path}")
        episodes.append((raw_episode_id, state7, action7))
        state_rows.append(state7)
        wrench_rows.append(wrench6)
        sample_episode_ids.extend((raw_episode_id,) * len(state7))

    action_target_population = build_action_target_population(episodes)
    delta_rows = action_target_population.action_target7
    delta_episode_ids = action_target_population.episode_ids
    bundle = CartesianNormalizerBundle.fit(
        state7=np.concatenate(state_rows),
        wrench6=np.concatenate(wrench_rows),
        delta_action7=delta_rows,
        sample_episode_ids=sample_episode_ids,
        delta_action_episode_ids=delta_episode_ids,
        split=split,
        split_sha256=old_normalizer["split_sha256"],
        calibration_bundle_sha256=old_normalizer["calibration_bundle_sha256"],
        wrench_geometry_spec_sha256=old_normalizer["wrench_geometry_spec_sha256"],
        action_target_population=action_target_population.manifest(
            split_sha256=old_normalizer["split_sha256"],
            builder_source_sha256=_sha256(
                project_root / "src/forcesmolvla/normalizer.py"
            ),
        ),
    )
    new_normalizer = bundle.manifest()
    for feature in ("state7", "wrench6"):
        for statistic in ("mean", "std"):
            np.testing.assert_allclose(
                new_normalizer["features"][feature][statistic],
                old_normalizer["features"][feature][statistic],
                rtol=0,
                atol=0,
            )

    old_normalizer_sha256 = _sha256(normalizer_path)
    old_conversion_sha256 = _sha256(conversion_path)
    backup_normalizer = dataset_root / "normalizer_manifest.same_frame_v1.backup.json"
    backup_conversion = dataset_root / "conversion_manifest.same_frame_v1.backup.json"
    if not backup_normalizer.exists() and not backup_conversion.exists():
        shutil.copy2(normalizer_path, backup_normalizer)
        shutil.copy2(conversion_path, backup_conversion)
    elif not backup_normalizer.exists() or not backup_conversion.exists():
        raise FileExistsError("normalizer refit backup pair is incomplete")

    conversion["normalizer_stats_sha256"] = new_normalizer["normalizer_stats_sha256"]
    conversion["normalizer_fit_contract"] = new_normalizer["fit_contract"]
    conversion["normalizer_refit"] = {
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "reason": "replace same-frame delta statistics with valid H=50 chunk-relative training targets",
        "source_normalizer_manifest_sha256": old_normalizer_sha256,
        "source_conversion_manifest_sha256": old_conversion_sha256,
        "tool": "tools/refit_chunk_delta_normalizer.py",
        "tool_sha256": _sha256(Path(__file__)),
        "normalizer_source_sha256": _sha256(
            project_root / "src/forcesmolvla/normalizer.py"
        ),
        "train_frame_count": len(sample_episode_ids),
        "valid_chunk_target_count": len(delta_episode_ids),
        "action_target_population_manifest_sha256": new_normalizer[
            "action_target_population"
        ]["population_manifest_sha256"],
    }
    _write_json_atomic(normalizer_path, new_normalizer)
    _write_json_atomic(conversion_path, conversion)
    print(
        json.dumps(
            {
                "status": "pass",
                "acceptance_status": "development_only",
                "normalizer_manifest_sha256": _sha256(normalizer_path),
                "conversion_manifest_sha256": _sha256(conversion_path),
                "normalizer_stats_sha256": new_normalizer["normalizer_stats_sha256"],
                "old_delta_std": old_normalizer["features"]["delta_action7"]["std"],
                "new_delta_std": new_normalizer["features"]["delta_action7"]["std"],
                "valid_chunk_target_count": len(delta_episode_ids),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
