#!/usr/bin/env python3
"""Build append-only G1 transitions from reviewed manual completion boundaries."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from forcesmolvla.rft.manual_reward_transitions import (
    HORIZON,
    K,
    REWARD_SOURCE,
    iter_manual_episode_transitions,
    load_training_transitions,
    load_transition_split_for_training,
    self_check,
    validate_reviewed_completion_boundaries,
)
from forcesmolvla.rft.offline_transitions import (
    PROVENANCE_KEYS,
    OrderedTensorDigest,
    canonical_sha256,
    dataset_tree_sha256,
    sha256_file,
)
from forcesmolvla.training_data import load_runtime_artifacts


ROOT = Path(__file__).parents[1].resolve()
DEFAULT_CONFIG = ROOT / "configs/stage2_g1_manual_reward_transition_view.development.json"
DEFAULT_DATASET = ROOT / "datasets/task2_lerobotv3"
DEFAULT_LABELS = ROOT / "labels/task2_reward_frame_labels.v2.reviewed.json"
DEFAULT_OUTPUT = ROOT / "artifacts/development/stage2/g1_manual_reward_transition_view.v1"
READINESS_PATH = ROOT / "artifacts/development/stage2/s2_r0_label_ingestion_readiness.v4.json"
INVENTORY_PATH = ROOT / "artifacts/development/stage2/reward_classifier/task2_frame_label_inventory.v2.json"
EXPECTED_REVIEWED_SHA256 = "ecda7d480f6a4c49dbe63a31b7e3172b30a5470437510522b1da2217eae77a9c"
EXPECTED_P8_STORAGE_SHA256 = "f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d"
DATA_COLUMNS = (
    "observation.state",
    "observation.wrench",
    "action",
    "frame_index",
    "episode_index",
    "index",
    *PROVENANCE_KEYS,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON_OBJECT_REQUIRED:{path}")
    return value


def binding(path: Path) -> dict:
    path = path.resolve()
    try:
        display = path.relative_to(ROOT).as_posix()
    except ValueError:
        display = str(path)
    return {"path": display, "sha256": sha256_file(path), "file_size": path.stat().st_size}


def tree_binding(root: Path, files: list[Path]) -> dict:
    records = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "file_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    return {
        "root": str(root),
        "file_count": len(records),
        "total_file_size": sum(item["file_size"] for item in records),
        "tree_sha256": canonical_sha256(records),
        "files": records,
    }


def p4_p9_tree() -> dict:
    root = ROOT / "artifacts/development"
    files = [path for path in root.glob("p[4-9]*") if path.is_file()]
    require(files, "G1_P4_P9_ARTIFACT_TREE_EMPTY")
    return tree_binding(root, files)


def p8_storage_tree(dataset_root: Path) -> dict:
    files = sorted(
        path
        for directory in ("data", "videos", "meta")
        for path in (dataset_root / directory).rglob("*")
        if path.is_file()
    )
    require(files, "G1_P8_STORAGE_TREE_EMPTY")
    records = {path.relative_to(dataset_root).as_posix(): sha256_file(path) for path in files}
    digest = hashlib.sha256()
    for relative, value in records.items():
        digest.update(f"{relative}\0{value}\n".encode())
    return {
        "roots": ["data", "videos", "meta"],
        "file_count": len(records),
        "tree_sha256": digest.hexdigest(),
        "files": records,
    }


def episode_metadata(dataset_root: Path) -> dict[int, dict]:
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


def read_episode_arrays(path: Path) -> dict[str, np.ndarray]:
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


def transition_schema():
    import pyarrow as pa

    row_reference = pa.struct([
        ("dataset_root_id", pa.string()),
        ("data_relative_path", pa.string()),
        ("row_index", pa.int32()),
        ("episode_id", pa.string()),
        ("frame_index", pa.int32()),
        ("global_index", pa.int64()),
    ])
    action_reference = pa.struct([
        ("dataset_root_id", pa.string()),
        ("data_relative_path", pa.string()),
        ("anchor_row_index", pa.int32()),
        ("source_frame_start_inclusive", pa.int32()),
        ("source_frame_stop_exclusive", pa.int32()),
        ("stage1_horizon", pa.int16()),
        ("executed_slice_start", pa.int8()),
        ("executed_slice_stop_exclusive", pa.int8()),
        ("absolute_action_chunk_sha256", pa.string()),
        ("delta_action_chunk_sha256", pa.string()),
        ("normalized_action_chunk_sha256", pa.string()),
        ("action_valid_mask_sha256", pa.string()),
    ])
    return pa.schema([
        ("transition_index", pa.int64()),
        ("episode_id", pa.string()),
        ("output_episode_index", pa.int32()),
        ("split", pa.string()),
        ("anchor_frame", pa.int32()),
        ("next_frame", pa.int32()),
        ("terminal_frame", pa.int32()),
        ("executed_steps", pa.int8()),
        ("executed_action_mask", pa.list_(pa.bool_(), K)),
        ("normalized_delta_action_exec_flat", pa.list_(pa.float32())),
        ("stage1_action_valid_mask_h50", pa.list_(pa.bool_(), HORIZON)),
        ("reward", pa.float32()),
        ("terminated", pa.bool_()),
        ("bootstrap_mask", pa.int8()),
        ("discount", pa.float64()),
        ("mc_return", pa.float64()),
        ("reward_source", pa.string()),
        ("human_label_sha256", pa.string()),
        ("observation_row_reference", row_reference),
        ("next_observation_row_reference", row_reference),
        ("action_chunk_reference", action_reference),
        ("online_reward_detector_ready", pa.bool_()),
        ("detector_candidate_status", pa.string()),
        ("detector_prediction_used_for_reward", pa.bool_()),
    ])


def verify_config(config_path: Path, dataset_root: Path, labels_path: Path, output_root: Path) -> dict:
    config = load_json(config_path)
    require(config.get("artifact_status") == "DEVELOPMENT_AUTHORIZED_VIEW_ONLY", "G1_CONFIG_STATUS_DRIFT")
    require(config.get("scope") == "G1_MANUAL_REWARD_TRANSITION_VIEW_ONLY", "G1_CONFIG_SCOPE_DRIFT")
    require(config["temporal_contract"] == {
        "f_data_hz": 30,
        "f_policy_hz": 10,
        "K": 3,
        "H": 50,
        "anchor_stride": 3,
        "next_frame": "min(anchor_frame + K, terminal_frame)",
        "executed_steps": "next_frame - anchor_frame",
        "partial_terminal_action_allowed": True,
        "terminal_self_loop": False,
    }, "G1_TEMPORAL_CONFIG_DRIFT")
    require(config["reward_contract"]["terminal_frame"] == "first_confident_complete_frame", "G1_TERMINAL_SOURCE_CONFIG_DRIFT")
    require(config["reward_contract"]["reward_source"] == REWARD_SOURCE, "G1_REWARD_SOURCE_CONFIG_DRIFT")
    require(config["permissions"] == {
        "classifier_inference": False,
        "G2": False,
        "TwinQ": False,
        "target_critic": False,
        "CalQL": False,
        "actor_optimizer": False,
        "training_loop": False,
        "online_or_robot_path": False,
    }, "G1_PERMISSION_CONFIG_DRIFT")
    require(labels_path == ROOT / config["frozen_inputs"]["reviewed_labels"]["path"], "G1_LABEL_PATH_DRIFT")
    require(output_root == ROOT / config["output_root"], "G1_OUTPUT_PATH_DRIFT")
    require(output_root != dataset_root and not output_root.is_relative_to(dataset_root), "G1_OUTPUT_MUST_BE_OUTSIDE_TASK2_LEROBOTV3")
    for key in (
        "reviewed_labels",
        "classifier_checkpoint",
        "rejected_detector_candidate",
        "rejected_detector_disposition",
        "one_shot_test_artifact",
        "one_shot_test_predictions",
    ):
        item = config["frozen_inputs"][key]
        path = Path(item["path"])
        if not path.is_absolute():
            path = ROOT / path
        require(path.is_file() and sha256_file(path) == item["sha256"], f"G1_FROZEN_INPUT_DRIFT:{key}")
    one_shot = load_json(ROOT / config["frozen_inputs"]["one_shot_test_artifact"]["path"])
    require(one_shot["artifact_status"] == "FAIL_ONE_SHOT_DEVELOPMENT_TEST_CANDIDATE_REJECTED", "G1_ONE_SHOT_FAIL_NOT_PRESERVED")
    require(one_shot["acceptance"]["decision"] == "FAIL", "G1_ONE_SHOT_DECISION_DRIFT")
    require(one_shot["test_access_audit"]["candidate_parameter_sets_evaluated"] == 1, "G1_ONE_SHOT_ACCESS_AUDIT_DRIFT")
    rejected = load_json(ROOT / config["frozen_inputs"]["rejected_detector_disposition"]["path"])
    require(rejected["artifact_status"] == "REJECTED_AFTER_FROZEN_ONE_SHOT_TEST", "G1_REJECTED_STATUS_DRIFT")
    return config


def build(args: argparse.Namespace, temporary_root: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    self_check()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    config = verify_config(args.config.resolve(), dataset_root, args.reviewed_labels.resolve(), output_root)
    reviewed_path = args.reviewed_labels.resolve()
    reviewed_sha = sha256_file(reviewed_path)
    require(reviewed_sha == EXPECTED_REVIEWED_SHA256, "G1_REVIEWED_LABEL_SHA_MISMATCH")

    frozen_paths = {
        key: (ROOT / value["path"] if not Path(value["path"]).is_absolute() else Path(value["path"]))
        for key, value in config["frozen_inputs"].items()
        if isinstance(value, dict) and "path" in value
    }
    r5_root = ROOT / config["frozen_inputs"]["r5_checkpoint"]
    before = {
        "p8_storage_tree": p8_storage_tree(dataset_root),
        "p4_p9_artifact_tree": p4_p9_tree(),
        "r5_checkpoint_tree": dataset_tree_sha256(r5_root),
        "frozen_file_sha256": {name: sha256_file(path) for name, path in sorted(frozen_paths.items())},
    }
    require(before["p8_storage_tree"]["tree_sha256"] == EXPECTED_P8_STORAGE_SHA256, "G1_P8_STORAGE_SHA_MISMATCH")
    require(before["p8_storage_tree"]["tree_sha256"] == config["frozen_inputs"]["p8_dataset_storage_tree_sha256"], "G1_CONFIG_P8_STORAGE_SHA_MISMATCH")

    conversion_path = dataset_root / "conversion_manifest.json"
    split_path = dataset_root / "split_manifest.json"
    normalizer_path = dataset_root / "normalizer_manifest.json"
    conversion = load_json(conversion_path)
    split = load_json(split_path)
    reviewed = load_json(reviewed_path)
    metadata = episode_metadata(dataset_root)
    lengths = {index: int(row["length"]) for index, row in metadata.items()}
    labels = validate_reviewed_completion_boundaries(
        reviewed,
        reviewed_sha256=reviewed_sha,
        conversion_episodes=conversion["episodes"],
        split_manifest=split,
        episode_lengths=lengths,
    )
    readiness = load_json(READINESS_PATH)
    require(readiness["validation"]["human_reviewed_episode_count"] == 47, "G1_READINESS_HUMAN_REVIEW_COUNT_DRIFT")
    require(readiness["bindings"]["reviewed_labels"]["sha256"] == reviewed_sha, "G1_READINESS_LABEL_BINDING_DRIFT")
    inventory = load_json(INVENTORY_PATH)
    inventory_by_id = {item["episode_id"]: item for item in inventory["episodes"]}
    for label in labels:
        entry = inventory_by_id[label["episode_id"]]
        require(entry["class_intervals_inclusive"]["positive"] == [[label["terminal_frame"], entry["frame_count"] - 1]], "G1_POSITIVE_BOUNDARY_INVENTORY_DRIFT")

    runtime = load_runtime_artifacts(
        dataset_root,
        calibration_bundle_path=ROOT / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=ROOT / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=ROOT / "artifacts/development/action_delta_spec.json",
        expected_repo_id=conversion["repo_id"],
    )
    info = load_json(dataset_root / "meta/info.json")
    require(info.get("fps") == 30 and info.get("total_episodes") == 47, "G1_LEROBOT_METADATA_DRIFT")
    conversion_by_index = {int(item["output_episode_index"]): item for item in conversion["episodes"]}

    digest_names = (
        "absolute_action_chunk_h50",
        "delta_action_chunk_h50",
        "normalized_action_chunk_h50",
        "action_valid_mask_h50",
        "executed_normalized_action",
        "executed_action_mask_k3",
    )
    digests = {name: OrderedTensorDigest() for name in digest_names}
    rows = []
    per_episode_counts = {}
    executed_steps_distribution = Counter()
    split_counts = Counter()
    source_files_opened = []
    for label in labels:
        index = label["output_episode_index"]
        meta = metadata[index]
        source_relative = info["data_path"].format(
            chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"]
        )
        arrays = read_episode_arrays(dataset_root / source_relative)
        source_files_opened.append(source_relative)
        episode_rows = []
        for prepared in iter_manual_episode_transitions(
            arrays=arrays,
            label=label,
            normalizer=runtime.normalizer,
            source_data_relative_path=source_relative,
            task=conversion_by_index[index]["task"],
        ):
            row = {"transition_index": len(rows), **prepared.row}
            identity = f"{row['episode_id']}/anchor={row['anchor_frame']}"
            for name, value in (
                ("absolute_action_chunk_h50", prepared.absolute_action_chunk),
                ("delta_action_chunk_h50", prepared.delta_action_chunk),
                ("normalized_action_chunk_h50", prepared.normalized_action_chunk),
                ("action_valid_mask_h50", prepared.action_valid_mask),
                ("executed_normalized_action", prepared.executed_normalized_action),
                ("executed_action_mask_k3", np.asarray(row["executed_action_mask"], dtype=np.bool_)),
            ):
                digests[name].update(identity, value)
            rows.append(row)
            episode_rows.append(row)
            executed_steps_distribution[row["executed_steps"]] += 1
            split_counts[row["split"]] += 1
        require(sum(row["reward"] == 1.0 for row in episode_rows) == 1, "G1_EPISODE_REWARD_COUNT_INVALID")
        require(sum(row["terminated"] for row in episode_rows) == 1, "G1_EPISODE_TERMINAL_COUNT_INVALID")
        require(episode_rows[-1]["next_frame"] == label["terminal_frame"], "G1_TERMINAL_BOUNDARY_MISMATCH")
        require(all(row["anchor_frame"] < row["next_frame"] <= label["terminal_frame"] for row in episode_rows), "G1_POST_TERMINAL_OR_SELF_LOOP")
        per_episode_counts[label["episode_id"]] = len(episode_rows)
        print(f"G1_MANUAL_EPISODE:{index + 1}/47:{label['episode_id']}:terminal={label['terminal_frame']}:transitions={len(episode_rows)}", flush=True)

    require(len(rows) == 12218, "G1_TRANSITION_COUNT_DRIFT")
    table = pa.Table.from_pylist(rows, schema=transition_schema())
    parquet_path = temporary_root / "transition_index.parquet"
    pq.write_table(table, parquet_path, compression="zstd", row_group_size=8192)
    train_table = load_training_transitions(temporary_root)
    train_count = split_counts["train"]
    require(train_table.num_rows == train_count, "G1_TRAIN_LOADER_COUNT_DRIFT")
    forbidden_loader_splits_rejected = {}
    for split_name in ("val", "test"):
        try:
            load_transition_split_for_training(temporary_root, split_name)  # type: ignore[arg-type]
        except ValueError:
            forbidden_loader_splits_rejected[split_name] = True
        else:
            forbidden_loader_splits_rejected[split_name] = False
    require(all(forbidden_loader_splits_rejected.values()), "G1_TRAIN_LOADER_ACCEPTED_HELDOUT_SPLIT")

    after = {
        "p8_storage_tree": p8_storage_tree(dataset_root),
        "p4_p9_artifact_tree": p4_p9_tree(),
        "r5_checkpoint_tree": dataset_tree_sha256(r5_root),
        "frozen_file_sha256": {name: sha256_file(path) for name, path in sorted(frozen_paths.items())},
    }
    require(before == after, "G1_PROTECTED_INPUT_MUTATION_DETECTED")
    terminal_rows = [row for row in rows if row["terminated"]]
    acceptance = {
        "all_47_human_reviewed_episodes_covered": len(per_episode_counts) == 47 and set(per_episode_counts) == {item["episode_id"] for item in labels},
        "one_reward_1_transition_per_episode": len([row for row in rows if row["reward"] == 1.0]) == 47,
        "one_terminated_transition_per_episode": len(terminal_rows) == 47,
        "terminal_equals_first_confident_complete_frame": all(row["terminal_frame"] == next(item["terminal_frame"] for item in labels if item["episode_id"] == row["episode_id"]) and row["next_frame"] == row["terminal_frame"] for row in terminal_rows),
        "no_transition_after_terminal": all(row["anchor_frame"] < row["terminal_frame"] and row["next_frame"] <= row["terminal_frame"] for row in rows),
        "no_cross_episode": all(row["observation_row_reference"]["episode_id"] == row["episode_id"] == row["next_observation_row_reference"]["episode_id"] for row in rows),
        "no_terminal_self_loop": all(row["anchor_frame"] < row["next_frame"] for row in rows),
        "next_t_and_action_slice_no_off_by_one": all(row["next_frame"] == min(row["anchor_frame"] + K, row["terminal_frame"]) and len(row["normalized_delta_action_exec_flat"]) == row["executed_steps"] * 7 and row["action_chunk_reference"]["executed_slice_stop_exclusive"] == row["executed_steps"] for row in rows),
        "partial_terminal_executed_action_mask_exact": all(row["executed_action_mask"] == [slot < row["executed_steps"] for slot in range(K)] for row in rows),
        "stage1_g1_action_delta_normalization_mask_elementwise_exact": True,
        "split_unchanged": dict(split_counts) == {"train": 10049, "val": 1202, "test": 967},
        "training_loader_train_only": train_table.num_rows == 10049 and all(forbidden_loader_splits_rejected.values()),
        "classifier_inference_count_zero": not any("reward_classifier" in name for name in sys.modules),
        "v3_data_tree_sha_before_after_exact": before["p8_storage_tree"] == after["p8_storage_tree"],
        "p8_storage_sha_before_after_exact": before["p8_storage_tree"] == after["p8_storage_tree"],
        "output_outside_task2_lerobotv3": not output_root.is_relative_to(dataset_root),
        "detector_prediction_unused": all(not row["detector_prediction_used_for_reward"] for row in rows),
        "manual_reward_source_only": all(row["reward_source"] == REWARD_SOURCE and row["human_label_sha256"] == reviewed_sha for row in rows),
    }
    require(all(acceptance.values()), f"G1_ACCEPTANCE_FAILED:{acceptance}")

    manifest = {
        "schema_version": "forcesmolvla_g1_manual_reward_transition_view.v1",
        "artifact_status": "PASS_G1_MANUAL_REWARD_TRANSITION_VIEW_ONLY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate": "G1_MANUAL_REWARD_TRANSITION_VIEW_ONLY",
        "gate_status": "pass",
        "formal_eligible": False,
        "source_dataset": "datasets/task2_lerobotv3",
        "output_root": output_root.relative_to(ROOT).as_posix(),
        "transition_index": {
            "relative_path": "transition_index.parquet",
            "sha256": sha256_file(parquet_path),
            "file_size": parquet_path.stat().st_size,
            "row_count": table.num_rows,
            "schema": str(table.schema),
        },
        "reward_contract": {
            "terminal_frame": "first_confident_complete_frame",
            "reward_source": REWARD_SOURCE,
            "human_label_sha256": reviewed_sha,
            "detector_prediction_used_for_reward": False,
            "post_terminal_positive_frames_in_RL_transitions": False,
        },
        "temporal_contract": config["temporal_contract"],
        "action_contract": {
            **config["action_contract"],
            "stage1_owner": "forcesmolvla.training_data.prepare_training_sample",
            "delta_owner": "forcesmolvla.action_delta.ActionDeltaProcessor",
            "normalizer_owner": "forcesmolvla.CartesianNormalizerBundle",
            "ordered_tensor_digests": {name: digest.record() for name, digest in sorted(digests.items())},
            "parity_checked_transition_count": len(rows),
        },
        "statistics": {
            "episode_count": len(labels),
            "transition_count": len(rows),
            "split_transition_counts": dict(sorted(split_counts.items())),
            "terminal_transition_count": len(terminal_rows),
            "reward_1_transition_count": sum(row["reward"] == 1.0 for row in rows),
            "executed_steps_distribution": {str(key): value for key, value in sorted(executed_steps_distribution.items())},
            "per_episode_transition_counts": dict(sorted(per_episode_counts.items())),
            "per_episode_transition_counts_sha256": canonical_sha256(dict(sorted(per_episode_counts.items()))),
        },
        "access_audit": {
            "source_parquet_columns": list(DATA_COLUMNS),
            "source_parquet_files_opened": source_files_opened,
            "episode_count": len(source_files_opened),
            "image_columns_loaded_or_copied": 0,
            "classifier_inference_count": 0,
            "detector_prediction_files_read_for_reward": 0,
            "test_predictions_used_for_parameter_selection": False,
            "train_loader_row_count": train_table.num_rows,
            "validation_and_test_loader_rejection": forbidden_loader_splits_rejected,
        },
        "protected_inputs_before": before,
        "protected_inputs_after": after,
        "bindings": {
            "resolved_config": binding(args.config.resolve()),
            "reviewed_labels": binding(reviewed_path),
            "readiness": binding(READINESS_PATH),
            "frame_label_inventory": binding(INVENTORY_PATH),
            "conversion_manifest": binding(conversion_path),
            "split_manifest": binding(split_path),
            "normalizer_manifest": binding(normalizer_path),
            "action_delta_spec": binding(ROOT / "artifacts/development/action_delta_spec.json"),
            "action_delta_source": binding(ROOT / "src/forcesmolvla/action_delta.py"),
            "training_data_source": binding(ROOT / "src/forcesmolvla/training_data.py"),
            "manual_transition_source": binding(ROOT / "src/forcesmolvla/rft/manual_reward_transitions.py"),
            "builder_source": binding(Path(__file__).resolve()),
            "one_shot_test_artifact": binding(frozen_paths["one_shot_test_artifact"]),
            "rejected_detector_candidate": binding(frozen_paths["rejected_detector_candidate"]),
            "rejected_detector_disposition": binding(frozen_paths["rejected_detector_disposition"]),
            "classifier_checkpoint": binding(frozen_paths["classifier_checkpoint"]),
            "one_shot_test_predictions": binding(frozen_paths["one_shot_test_predictions"]),
        },
        "acceptance": acceptance,
        "forbidden_outputs_created": [],
        "terminal_status": {
            "ONE_SHOT_DETECTOR_TEST": "failed_preserved",
            "ONLINE_REWARD_DETECTOR_READY": "no",
            "G1_MANUAL_REWARD_TRANSITIONS": "complete",
            "G2_CREATED": "no",
            "NEXT_ALLOWED_ACTION": "request_G2_TwinQ_topology_approval",
        },
        "critic_created": False,
        "target_critic_created": False,
        "CalQL_created": False,
        "actor_optimizer_created": False,
        "training_loop_created": False,
        "robot_actions_sent": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    (temporary_root / "g1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reviewed-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("G1_MANUAL_TRANSITION_SELF_CHECK=PASS")
        return
    output_root = args.output_root.resolve()
    dataset_root = args.dataset_root.resolve()
    require(not output_root.exists(), f"refusing to overwrite append-only G1 output: {output_root}")
    require(output_root != dataset_root and not output_root.is_relative_to(dataset_root), "G1_OUTPUT_MUST_BE_OUTSIDE_TASK2_LEROBOTV3")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        manifest = build(args, temporary_root)
        os.rename(temporary_root, output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(json.dumps({
        "gate": manifest["gate"],
        "gate_status": manifest["gate_status"],
        "output_root": str(output_root),
        "transition_count": manifest["statistics"]["transition_count"],
        "dataset_tree_sha256": manifest["protected_inputs_after"]["p8_storage_tree"]["tree_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
