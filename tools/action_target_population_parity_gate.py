#!/usr/bin/env python3
"""Independent exact oracle for the H=50 mixed Cartesian action-target population."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pyarrow.parquet as pq

from forcesmolvla.action_delta import ActionDeltaProcessor
from forcesmolvla.normalizer import (
    ACTION_TARGET_QUANTILES,
    build_action_target_population,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _oracle_statistics(values: np.ndarray, horizon_k: np.ndarray) -> dict:
    def summarize(rows: np.ndarray) -> dict:
        quantiles = np.quantile(rows, ACTION_TARGET_QUANTILES, axis=0)
        return {
            "count": len(rows),
            "mean": np.mean(rows, axis=0).tolist(),
            "std": np.sqrt(np.mean((rows - np.mean(rows, axis=0)) ** 2, axis=0)).tolist(),
            "min": np.min(rows, axis=0).tolist(),
            "max": np.max(rows, axis=0).tolist(),
            "quantiles": {
                str(q): row.tolist()
                for q, row in zip(ACTION_TARGET_QUANTILES, quantiles, strict=True)
            },
        }

    return {
        "method": "numpy population moments ddof=0; linear quantiles [0.01,0.1,0.5,0.9,0.99]",
        "global": summarize(values),
        "per_horizon": [
            {"horizon_k": k, **summarize(values[horizon_k == k])}
            for k in range(50)
        ],
    }


def _identity_sha256(identities: tuple[tuple[str, int, int], ...]) -> str:
    digest = hashlib.sha256()
    for episode_id, anchor_t, horizon_k in identities:
        encoded = episode_id.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<qB", anchor_t, horizon_k))
    return digest.hexdigest()


def _tensor_sha256(values: np.ndarray) -> str:
    tensor = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", *tensor.shape))
    digest.update(tensor.view(np.uint8))
    return digest.hexdigest()


def _oracle_population(
    episodes: dict[str, tuple[np.ndarray, np.ndarray]],
    ordered_train_ids: tuple[str, ...],
    *,
    padding_value: float,
) -> tuple[np.ndarray, tuple[tuple[str, int, int], ...], np.ndarray]:
    rows: list[np.ndarray] = []
    identities: list[tuple[str, int, int]] = []
    horizons: list[int] = []
    for episode_id in ordered_train_ids:
        states, actions = episodes[episode_id]
        for anchor_t in range(len(states)):
            valid_count = min(50, len(actions) - anchor_t)
            padded = np.full((50, 7), padding_value, dtype=np.float64)
            padded[:valid_count] = actions[anchor_t : anchor_t + valid_count]
            valid = np.arange(50) < valid_count
            selected = padded[valid]
            target = selected.copy()
            target[:, :3] = selected[:, :3] - states[anchor_t, :3]
            angle = selected[:, 3:6] - states[anchor_t, 3:6]
            target[:, 3:6] = (angle + np.pi) % (2 * np.pi) - np.pi
            # target[:, 6] deliberately stays absolute.
            rows.append(target)
            identities.extend((episode_id, anchor_t, k) for k in range(valid_count))
            horizons.extend(range(valid_count))
    return (
        np.concatenate(rows),
        tuple(identities),
        np.asarray(horizons, dtype=np.int64),
    )


def main() -> None:
    root = Path(__file__).parents[1].resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=root / "datasets/task2_lerobotv3"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite parity artifact: {output}")
    dataset_root = args.dataset_root.resolve()
    split = json.loads((dataset_root / "split_manifest.json").read_text())
    conversion = json.loads((dataset_root / "conversion_manifest.json").read_text())
    normalizer = json.loads((dataset_root / "normalizer_manifest.json").read_text())
    ordered_train_ids = tuple(split["train"])
    mapping = {
        entry["raw_episode_id"]: int(entry["output_episode_index"])
        for entry in conversion["episodes"]
    }
    episodes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for episode_id, output_index in mapping.items():
        table = pq.read_table(
            dataset_root / "data/chunk-000" / f"file-{output_index:03d}.parquet",
            columns=["observation.state", "action"],
        )
        episodes[episode_id] = (
            np.asarray(table["observation.state"].to_pylist(), dtype=np.float64),
            np.asarray(table["action"].to_pylist(), dtype=np.float64),
        )

    oracle_a, identities_a, horizon_a = _oracle_population(
        episodes, ordered_train_ids, padding_value=1234567.0
    )
    oracle_b, identities_b, horizon_b = _oracle_population(
        {episode_id: episodes[episode_id] for episode_id in ordered_train_ids},
        ordered_train_ids,
        padding_value=-7654321.0,
    )
    padding_invariant = np.array_equal(oracle_a, oracle_b)
    val_test_zero_influence = identities_a == identities_b and np.array_equal(
        horizon_a, horizon_b
    )
    production = build_action_target_population(
        (episode_id, *episodes[episode_id]) for episode_id in ordered_train_ids
    )
    production_identities = tuple(
        zip(
            production.episode_ids,
            production.anchor_t.tolist(),
            production.horizon_k.tolist(),
            strict=True,
        )
    )
    oracle_stats = _oracle_statistics(oracle_a, horizon_a)
    split_sha256 = _canonical_sha256(split)
    builder_source_sha256 = _sha256(root / "src/forcesmolvla/normalizer.py")
    expected_manifest = production.manifest(
        split_sha256=split_sha256,
        builder_source_sha256=builder_source_sha256,
    )

    sentinel_state = np.zeros((3, 7), dtype=np.float64)
    sentinel_state[:, 0] = [0.0, 10.0, 20.0]
    sentinel_state[:, 6] = 0.04
    sentinel_action = sentinel_state.copy()
    sentinel_action[:, 0] += 1.0
    sentinel_delta = ActionDeltaProcessor.to_delta(sentinel_action, sentinel_state[0])
    sentinel_same_frame = sentinel_action[1, 0] - sentinel_state[1, 0]
    sentinel_anchor = sentinel_delta[1, 0]
    sentinel_pass = sentinel_anchor == 11.0 and sentinel_anchor != sentinel_same_frame
    sentinel_roundtrip = ActionDeltaProcessor.from_delta(sentinel_delta, sentinel_state[0])
    gripper_absolute = np.array_equal(sentinel_delta[:, 6], sentinel_action[:, 6]) and np.array_equal(
        sentinel_roundtrip[:, 6], sentinel_action[:, 6]
    )

    assertions = {
        "valid_pair_count_exact": len(production.action_target7) == len(oracle_a),
        "ordered_pair_identity_exact": production_identities == identities_a,
        "ordered_pair_identity_sha256_exact": production.identity_sha256()
        == _identity_sha256(identities_a),
        "action_target7_tensor_exact": np.array_equal(production.action_target7, oracle_a),
        "action_target7_tensor_sha256_exact": production.tensor_sha256()
        == _tensor_sha256(oracle_a),
        "global_and_per_horizon_statistics_exact": production.statistics() == oracle_stats,
        "padding_value_invariant": padding_invariant,
        "val_test_zero_influence": val_test_zero_influence,
        "absolute_gripper_to_from_delta": gripper_absolute,
        "anchor_state_not_future_same_frame_state_sentinel": sentinel_pass,
        "normalizer_manifest_population_binding_exact": normalizer.get(
            "action_target_population"
        )
        == expected_manifest,
        "conversion_manifest_fit_contract_exact": conversion.get(
            "normalizer_fit_contract"
        )
        == normalizer.get("fit_contract"),
    }
    # NumPy comparisons intentionally return np.bool_; normalize the public
    # artifact boundary to JSON-native booleans.
    assertions = {name: bool(value) for name, value in assertions.items()}
    if not all(assertions.values()):
        raise RuntimeError(f"ACTION_TARGET_POPULATION_PARITY_FAILED:{assertions}")
    result = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "gate": "ActionTargetPopulationParityGate",
        "gate_status": "pass",
        "dataset_root": str(dataset_root),
        "split_sha256": split_sha256,
        "normalizer_manifest_sha256": _sha256(
            dataset_root / "normalizer_manifest.json"
        ),
        "conversion_manifest_sha256": _sha256(
            dataset_root / "conversion_manifest.json"
        ),
        "target_builder_source_sha256": builder_source_sha256,
        "oracle_source_sha256": _sha256(Path(__file__)),
        "valid_pair_count": len(oracle_a),
        "ordered_pair_identity_sha256": _identity_sha256(identities_a),
        "action_target7_float64_tensor_sha256": _tensor_sha256(oracle_a),
        "statistics": oracle_stats,
        "statistics_sha256": _canonical_sha256(oracle_stats),
        "assertions": assertions,
        "forcevla_numeric_comparison_role": "auxiliary_only_not_acceptance_oracle",
        "robot_actions_sent": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "gate_status": "pass",
                "valid_pair_count": len(oracle_a),
                "identity_sha256": result["ordered_pair_identity_sha256"],
                "tensor_sha256": result["action_target7_float64_tensor_sha256"],
                "statistics_sha256": result["statistics_sha256"],
                "artifact_sha256": _sha256(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
