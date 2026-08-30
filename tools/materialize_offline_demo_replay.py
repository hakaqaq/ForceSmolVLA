#!/usr/bin/env python3
"""Materialize offline demo replay from a frozen reward-label sidecar."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from forcesmolvla.rft.offline_transitions import (
    ANCHOR_STRIDE,
    EXECUTED_ACTION_SLOTS,
    GAMMA,
    HORIZON,
    PROVENANCE_KEYS,
    OrderedTensorDigest,
    canonical_sha256,
    dataset_tree_sha256,
    iter_episode_transitions,
    sha256_file,
    validate_action_contract,
    validate_outcome_labels,
    validate_reward_spec,
)
from forcesmolvla.rft.source_manifest import stage2_source_manifest_binding
from forcesmolvla.training_data import load_runtime_artifacts


ROOT = Path(__file__).parents[1].resolve()
DATA_COLUMNS = (
    "observation.state",
    "observation.wrench",
    "action",
    "frame_index",
    "episode_index",
    "index",
    *PROVENANCE_KEYS,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _binding(path: Path, *, relative_to: Path = ROOT) -> dict:
    return {
        "relative_path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "file_size": path.stat().st_size,
    }


def _tree_progress(phase: str):
    def report(index: int, total: int, relative: str) -> None:
        print(f"G1_DATA_TREE_{phase}:{index}/{total}:{relative}", flush=True)

    return report


def _episode_metadata(dataset_root: Path):
    import pyarrow.parquet as pq

    path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    columns = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    return {row["episode_index"]: row for row in pq.read_table(path, columns=columns).to_pylist()}


def _read_episode_arrays(path: Path) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(DATA_COLUMNS))
    arrays = {}
    for name in DATA_COLUMNS:
        dtype = None
        if name in {"observation.state", "observation.wrench", "action"}:
            dtype = np.float64
        elif name in {"frame_index", "episode_index", "index"} or name.endswith("_ns"):
            dtype = np.int64
        arrays[name] = np.asarray(table[name].to_pylist(), dtype=dtype)
    return arrays


def _transition_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("transition_index", pa.int64()),
            ("raw_episode_id", pa.string()),
            ("output_episode_index", pa.int32()),
            ("split", pa.string()),
            ("source_data_relative_path", pa.string()),
            ("anchor_row_index", pa.int32()),
            ("next_row_index", pa.int32()),
            ("anchor_frame_index", pa.int32()),
            ("next_frame_index", pa.int32()),
            ("terminal_frame_index", pa.int32()),
            ("anchor_global_index", pa.int64()),
            ("next_global_index", pa.int64()),
            ("critic_action_k7", pa.list_(pa.float32(), 21)),
            ("actor_q_gradient_mask_k7", pa.list_(pa.bool_(), 21)),
            ("reward", pa.float32()),
            ("terminated", pa.bool_()),
            ("truncated", pa.bool_()),
            ("timeout", pa.bool_()),
            ("bootstrap_mask", pa.float32()),
            ("discount", pa.float32()),
            ("mc_return", pa.float64()),
        ]
    )


def _build(args, temporary_root: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset_root = args.dataset_root.resolve()
    before_tree = dataset_tree_sha256(
        dataset_root, progress=_tree_progress("BEFORE")
    )
    conversion_path = dataset_root / "conversion_manifest.json"
    split_path = dataset_root / "split_manifest.json"
    normalizer_path = dataset_root / "normalizer_manifest.json"
    conversion = _load_json(conversion_path)
    split = _load_json(split_path)
    outcomes = _load_json(args.reward_labels.resolve())
    reward_spec = _load_json(args.reward_spec.resolve())
    action_contract = _load_json(args.action_contract.resolve())
    validate_reward_spec(reward_spec)
    if reward_spec["real_g1_generation_permitted"] is not True:
        raise RuntimeError(
            "G1_REAL_BUILD_BLOCKED_PENDING_EXTERNAL_FROZEN_REWARD_LABEL_APPROVAL"
        )
    validate_action_contract(action_contract)
    metadata = _episode_metadata(dataset_root)
    lengths = {index: int(row["length"]) for index, row in metadata.items()}
    labels = validate_outcome_labels(
        outcomes,
        conversion_episodes=conversion["episodes"],
        episode_lengths=lengths,
    )
    if conversion.get("split") != split:
        raise RuntimeError("G1_CONVERSION_SPLIT_MANIFEST_MISMATCH")
    normalizer_binding = action_contract["frozen_normalizer_manifest"]
    if (
        normalizer_binding["path"]
        != normalizer_path.relative_to(ROOT).as_posix()
        or normalizer_binding["sha256"] != sha256_file(normalizer_path)
    ):
        raise RuntimeError("G1_ACTION_CONTRACT_NORMALIZER_BINDING_MISMATCH")

    runtime = load_runtime_artifacts(
        dataset_root,
        calibration_bundle_path=ROOT / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=ROOT / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=ROOT / "artifacts/development/action_delta_spec.json",
        expected_repo_id=conversion["repo_id"],
    )
    info = _load_json(dataset_root / "meta/info.json")
    if info.get("fps") != 30 or info.get("total_episodes") != 47:
        raise RuntimeError("G1_LEROBOT_METADATA_DRIFT")

    digests = {
        name: OrderedTensorDigest()
        for name in (
            "absolute_action_chunk_h50",
            "delta_action_chunk_h50",
            "normalized_action_chunk_h50",
            "action_valid_mask_h50",
            "action_feature_mask_h50x7",
            "critic_action_k3x7",
        )
    }
    rows = []
    per_episode_counts = Counter()
    for label in labels:
        meta = metadata[label["output_episode_index"]]
        source_relative = info["data_path"].format(
            chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"]
        )
        arrays = _read_episode_arrays(dataset_root / source_relative)
        conversion_entry = conversion["episodes"][label["output_episode_index"]]
        episode_rows = []
        for prepared in iter_episode_transitions(
            arrays=arrays,
            outcome=label,
            normalizer=runtime.normalizer,
            source_data_relative_path=source_relative,
            task=conversion_entry["task"],
        ):
            row = {"transition_index": len(rows), **prepared.row}
            identity = (
                f"{row['raw_episode_id']}/anchor={row['anchor_frame_index']}"
            )
            for name, value in (
                ("absolute_action_chunk_h50", prepared.absolute_action_chunk),
                ("delta_action_chunk_h50", prepared.delta_action_chunk),
                ("normalized_action_chunk_h50", prepared.normalized_action_chunk),
                ("action_valid_mask_h50", prepared.action_valid_mask),
                ("action_feature_mask_h50x7", prepared.action_feature_mask),
                (
                    "critic_action_k3x7",
                    np.asarray(row["critic_action_k7"], dtype=np.float32).reshape(3, 7),
                ),
            ):
                digests[name].update(identity, value)
            rows.append(row)
            episode_rows.append(row)
        if sum(row["terminated"] for row in episode_rows) != 1:
            raise RuntimeError("G1_EPISODE_TERMINAL_TRANSITION_COUNT_INVALID")
        if sum(row["reward"] == 1.0 for row in episode_rows) != 1:
            raise RuntimeError("G1_EPISODE_TERMINAL_REWARD_COUNT_INVALID")
        if any(row["anchor_frame_index"] >= row["next_frame_index"] for row in episode_rows):
            raise RuntimeError("G1_TERMINAL_SELF_LOOP_OR_REVERSED_TRANSITION")
        per_episode_counts[label["raw_episode_id"]] = len(episode_rows)
        print(
            f"G1_EPISODE:{label['output_episode_index'] + 1}/47:"
            f"{label['raw_episode_id']}:transitions={len(episode_rows)}",
            flush=True,
        )

    table = pa.Table.from_pylist(rows, schema=_transition_schema())
    parquet_path = temporary_root / "transition_index.parquet"
    pq.write_table(table, parquet_path, compression="zstd", row_group_size=8192)
    after_tree = dataset_tree_sha256(
        dataset_root, progress=_tree_progress("AFTER")
    )
    if before_tree != after_tree:
        raise RuntimeError("G1_V3_DATA_TREE_MUTATED")

    split_counts = Counter(row["split"] for row in rows)
    terminal_count = sum(row["terminated"] for row in rows)
    terminal_reward_count = sum(row["reward"] == 1.0 for row in rows)
    discount_identity = all(
        abs(row["discount"] - GAMMA * row["bootstrap_mask"]) < 1e-15
        for row in rows
    )
    source_manifest = stage2_source_manifest_binding(
        ROOT, args.source_manifest.resolve()
    )
    upstream = {
        "conversion_manifest": _binding(conversion_path),
        "split_manifest": _binding(split_path),
        "normalizer_manifest": _binding(normalizer_path),
        "lerobot_info": _binding(dataset_root / "meta/info.json"),
        "lerobot_episode_metadata": _binding(
            dataset_root / "meta/episodes/chunk-000/file-000.parquet"
        ),
        "action_delta_spec": _binding(ROOT / "artifacts/development/action_delta_spec.json"),
        "action_delta_source": _binding(ROOT / "src/forcesmolvla/action_delta.py"),
    }
    bindings = {
        "episode_frame_reward_labels": _binding(args.reward_labels.resolve()),
        "reward_spec": _binding(args.reward_spec.resolve()),
        "action_contract": _binding(args.action_contract.resolve()),
        "builder_source": _binding(Path(__file__).resolve()),
        "transition_source": _binding(
            ROOT / "src/forcesmolvla/rft/offline_transitions.py"
        ),
        "stage2_source_manifest": source_manifest,
    }
    acceptance = {
        "v3_data_tree_sha_before_after_exact": before_tree == after_tree,
        "all_47_attested_episodes_exactly_once": set(per_episode_counts)
        == {label["raw_episode_id"] for label in labels}
        and len(per_episode_counts) == 47,
        "split_matches_stage1": all(
            label["raw_episode_id"] in split[label["split"]] for label in labels
        ),
        "one_terminal_transition_per_episode": terminal_count == 47,
        "one_terminal_reward_per_episode": terminal_reward_count == 47,
        "no_cross_episode_or_terminal_self_loop": all(
            row["anchor_frame_index"] < row["next_frame_index"]
            <= row["terminal_frame_index"]
            for row in rows
        ),
        "action_next_t_excluded": all(
            row["next_frame_index"] - row["anchor_frame_index"]
            == EXECUTED_ACTION_SLOTS
            for row in rows
        ),
        "critic_action_shape_k3x7": table.schema.field("critic_action_k7").type.list_size
        == 21,
        "actor_q_gradient_mask_tcp_only": all(
            row["actor_q_gradient_mask_k7"]
            == [
                dimension < 6
                for slot in range(3)
                for dimension in range(7)
            ]
            for row in rows
        ),
        "discount_identity_exact": discount_identity,
        "no_images_or_observation_payload_copied": not any(
            "image" in name or name.startswith("observation.") for name in table.column_names
        ),
        "stage1_action_transform_and_normalizer_reused": True,
        "dataset_future_action_as_td_bootstrap_absent": "td_bootstrap_action"
        not in table.column_names,
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"G1_ACCEPTANCE_FAILED:{acceptance}")
    manifest = {
        "schema_version": "1.0",
        "gate": "S2-G1",
        "gate_status": "pass",
        "artifact_status": "development_only",
        "formal_eligible": False,
        "source_dataset": "datasets/task2_lerobotv3",
        "source_dataset_tree_before": before_tree,
        "source_dataset_tree_after": after_tree,
        "transition_index": {
            "relative_path": "transition_index.parquet",
            "sha256": sha256_file(parquet_path),
            "file_size": parquet_path.stat().st_size,
            "row_count": table.num_rows,
            "schema": str(table.schema),
        },
        "episode_semantics": {
            "episode_count": 47,
            "task_outcome": "success",
            "label_source": "retrospective_operator_attestation",
            "terminal_source": "external_frozen_episode_frame_reward_sidecar",
            "first_causally_confirmed_success_claimed": False,
        },
        "temporal_contract": {
            "f_data_hz": 30,
            "f_policy_hz": 10,
            "horizon": HORIZON,
            "executed_action_slots": EXECUTED_ACTION_SLOTS,
            "critic_action_shape": [EXECUTED_ACTION_SLOTS, 7],
            "partial_action_interface": False,
            "terminal_derived_action_mask": False,
            "anchor_stride_frames": ANCHOR_STRIDE,
            "terminal_self_loop": False,
        },
        "reward_contract": {
            "gamma_per_policy_decision": GAMMA,
            "terminal_reward": 1.0,
            "nonterminal_reward": 0.0,
            "terminal_discount": 0.0,
            "nonterminal_discount": GAMMA,
            "description": "development endpoint reward, not exact task-completion-time reward",
        },
        "bootstrap_policy_contract": "deferred_to_G4",
        "dataset_future_action_as_td_bootstrap": "forbidden",
        "td_target_implemented": False,
        "statistics": {
            "episode_count": 47,
            "frame_count": sum(lengths.values()),
            "transition_count": len(rows),
            "split_transition_counts": dict(sorted(split_counts.items())),
            "terminal_transition_count": terminal_count,
            "terminal_reward_count": terminal_reward_count,
            "per_episode_transition_counts_sha256": canonical_sha256(
                dict(sorted(per_episode_counts.items()))
            ),
        },
        "stage1_action_parity": {
            "owner": "forcesmolvla.training_data.prepare_training_sample",
            "absolute_to_delta_owner": "forcesmolvla.action_delta.ActionDeltaProcessor",
            "normalizer_owner": "forcesmolvla.CartesianNormalizerBundle",
            "all_transitions_checked": True,
            "ordered_digests": {
                name: digest.record() for name, digest in sorted(digests.items())
            },
        },
        "upstream_bindings": upstream,
        "stage2_bindings": bindings,
        "acceptance": acceptance,
        "critic_created": False,
        "target_critic_created": False,
        "target_actor_created": False,
        "optimizer_created": False,
        "training_loop_created": False,
        "robot_actions_sent": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    (temporary_root / "rl_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "datasets/task2_lerobotv3"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "datasets/task2_forcerft_rl_v1"
    )
    parser.add_argument("--reward-labels", type=Path, required=True)
    parser.add_argument(
        "--reward-spec",
        type=Path,
        default=ROOT / "configs/stage2_reward_spec.development.yaml",
    )
    parser.add_argument(
        "--action-contract",
        type=Path,
        default=ROOT / "configs/stage2_action_contract.development.json",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "artifacts/development/stage2/stage2_source_manifest.v4.json",
    )
    args = parser.parse_args()
    reward_spec = _load_json(args.reward_spec.resolve())
    validate_reward_spec(reward_spec)
    if reward_spec["real_g1_generation_permitted"] is not True:
        raise RuntimeError(
            "G1_REAL_BUILD_BLOCKED_PENDING_EXTERNAL_FROZEN_REWARD_LABEL_APPROVAL"
        )
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite G1 output: {output_root}")
    if dataset_root == output_root or dataset_root in output_root.parents:
        raise RuntimeError("G1_OUTPUT_MUST_BE_OUTSIDE_TASK2_LEROBOTV3")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        manifest = _build(args, temporary_root)
        os.rename(temporary_root, output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "gate": manifest["gate"],
                "gate_status": manifest["gate_status"],
                "output_root": str(output_root),
                "transition_count": manifest["statistics"]["transition_count"],
                "v3_tree_sha256": manifest["source_dataset_tree_after"]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
