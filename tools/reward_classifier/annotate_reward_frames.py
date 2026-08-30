#!/usr/bin/env python3
"""Validate reviewed task2 frame labels and write append-only R0 readiness artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forcesmolvla.dataset_binding import dataset_storage_binding  # noqa: E402


EXPECTED_P8_DATASET_SHA256 = "f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d"
CLASS_NAMES = ("positive", "ordinary_negative", "hard_negative", "ambiguous")
TRAINABLE_CLASS_NAMES = ("positive", "ordinary_negative", "hard_negative")
INTERVAL_FIELDS = {
    "ordinary_negative": "ordinary_negative_intervals",
    "hard_negative": "hard_negative_intervals",
    "ambiguous": "ambiguous_intervals",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "file_size": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_optional_review_timestamp(episode: dict[str, Any]) -> bool:
    if "review_timestamp" not in episode or episode["review_timestamp"] is None:
        return False
    value = episode["review_timestamp"]
    require(isinstance(value, str) and value, "REVIEW_TIMESTAMP_TYPE_INVALID")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("REVIEW_TIMESTAMP_ISO8601_INVALID") from error
    require(
        parsed.tzinfo is not None and parsed.utcoffset() is not None,
        "REVIEW_TIMESTAMP_TZ_MISSING",
    )
    return True


def split_lookup(split_manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, set[str]]]:
    split_sets: dict[str, set[str]] = {}
    lookup: dict[str, str] = {}
    for split in ("train", "val", "test"):
        values = split_manifest.get(split)
        require(
            isinstance(values, list) and all(isinstance(x, str) for x in values),
            "SPLIT_SCHEMA_INVALID",
        )
        require(len(values) == len(set(values)), f"SPLIT_DUPLICATE_EPISODE:{split}")
        split_sets[split] = set(values)
        for episode_id in values:
            require(episode_id not in lookup, f"EPISODE_LEAKAGE:{episode_id}")
            lookup[episode_id] = split
    require(
        {split: len(values) for split, values in split_sets.items()}
        == {"train": 38, "val": 5, "test": 4},
        "SPLIT_EPISODE_COUNTS_INVALID",
    )
    return lookup, split_sets


def validate_review_schema(
    reviewed: dict[str, Any], template: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    require(
        reviewed.get("schema_version") == "force_rft_task2_reward_frame_labels.v2",
        "REVIEW_SCHEMA_VERSION_INVALID",
    )
    require(set(reviewed) == set(template), "REVIEW_TOP_LEVEL_KEYS_INVALID")
    for key in set(template) - {"episodes"}:
        require(reviewed[key] == template[key], f"REVIEW_TOP_LEVEL_CONTRACT_DRIFT:{key}")

    reviewed_episodes = reviewed.get("episodes")
    template_episodes = template.get("episodes")
    require(
        reviewed.get("episode_count") == 47
        and isinstance(reviewed_episodes, list)
        and len(reviewed_episodes) == 47
        and isinstance(template_episodes, list)
        and len(template_episodes) == 47,
        "REVIEW_EPISODE_COUNT_INVALID",
    )
    template_by_id = {episode["episode_id"]: episode for episode in template_episodes}
    require(len(template_by_id) == 47, "TEMPLATE_EPISODE_IDS_INVALID")
    required_keys = set(template_episodes[0]) - {"review_timestamp"}
    allowed_keys = required_keys | {"review_timestamp"}
    fixed_keys = {
        "episode_id",
        "output_episode_index",
        "split",
        "task_outcome_context",
        "outcome_source",
        "outcome_is_not_a_frame_label",
    }
    by_id: dict[str, dict[str, Any]] = {}
    timestamp_count = 0
    for episode in reviewed_episodes:
        require(isinstance(episode, dict), "REVIEW_EPISODE_OBJECT_REQUIRED")
        require(required_keys <= set(episode) <= allowed_keys, "REVIEW_EPISODE_KEYS_INVALID")
        episode_id = episode.get("episode_id")
        require(
            isinstance(episode_id, str) and episode_id in template_by_id,
            "REVIEW_EPISODE_ID_INVALID",
        )
        require(episode_id not in by_id, f"REVIEW_EPISODE_DUPLICATE:{episode_id}")
        for key in fixed_keys:
            require(
                episode.get(key) == template_by_id[episode_id].get(key),
                f"REVIEW_EPISODE_CONTRACT_DRIFT:{episode_id}:{key}",
            )
        require(
            episode.get("manual_review_status") == "human_reviewed",
            f"EPISODE_NOT_HUMAN_REVIEWED:{episode_id}",
        )
        require(episode.get("positive_available") is True, f"POSITIVE_UNAVAILABLE:{episode_id}")
        require(episode.get("completion_visible") is True, f"COMPLETION_NOT_VISIBLE:{episode_id}")
        require(episode.get("completion_stable") is True, f"COMPLETION_NOT_STABLE:{episode_id}")
        require(
            isinstance(episode.get("reviewer_id"), str) and episode["reviewer_id"],
            f"REVIEWER_ID_INVALID:{episode_id}",
        )
        require(
            isinstance(episode.get("confidence"), str) and episode["confidence"],
            f"CONFIDENCE_INVALID:{episode_id}",
        )
        require(
            episode.get("notes") is None or isinstance(episode["notes"], str),
            f"NOTES_INVALID:{episode_id}",
        )
        timestamp_count += validate_optional_review_timestamp(episode)
        by_id[episode_id] = episode
    return by_id, {
        "schema_valid": True,
        "manual_audit_complete": len(by_id) == 47,
        "human_reviewed_episode_count": len(by_id),
        "review_timestamp_optional": True,
        "review_timestamp_autofilled": False,
        "review_timestamp_present_and_valid_count": timestamp_count,
        "review_timestamp_absent_or_null_count": 47 - timestamp_count,
        "source_top_level_manual_audit_complete": reviewed.get("manual_audit_complete"),
        "source_top_level_artifact_status": reviewed.get("artifact_status"),
    }


def classify_episode(
    episode: dict[str, Any], frame_count: int
) -> tuple[list[str], dict[str, list[list[int]]]]:
    episode_id = episode["episode_id"]
    first_positive = episode.get("first_confident_complete_frame")
    last_incomplete = episode.get("last_confident_incomplete_frame")
    require(
        is_integer(first_positive) and 0 <= first_positive < frame_count,
        f"FIRST_POSITIVE_INVALID:{episode_id}",
    )
    require(last_incomplete == first_positive - 1, f"LAST_INCOMPLETE_INVALID:{episode_id}")

    normalized: dict[str, list[list[int]]] = {}
    owner: list[str | None] = [None] * frame_count
    for class_name, field in INTERVAL_FIELDS.items():
        intervals = episode.get(field)
        require(isinstance(intervals, list), f"INTERVAL_LIST_INVALID:{episode_id}:{field}")
        normalized[class_name] = []
        for interval in intervals:
            require(
                isinstance(interval, list)
                and len(interval) == 2
                and all(is_integer(value) for value in interval),
                f"INTERVAL_SCHEMA_INVALID:{episode_id}:{field}",
            )
            start, end = interval
            require(
                0 <= start <= end < frame_count,
                f"INTERVAL_BOUNDS_INVALID:{episode_id}:{field}",
            )
            normalized[class_name].append([start, end])
            for frame_index in range(start, end + 1):
                require(owner[frame_index] is None, f"OVERLAPPING_FRAME:{episode_id}:{frame_index}")
                owner[frame_index] = class_name

    normalized["positive"] = [[first_positive, frame_count - 1]]
    for frame_index in range(first_positive, frame_count):
        require(owner[frame_index] is None, f"POSITIVE_OVERLAP:{episode_id}:{frame_index}")
        owner[frame_index] = "positive"
    missing = [index for index, value in enumerate(owner) if value is None]
    if missing:
        raise RuntimeError(f"UNLABELED_FRAMES:{episode_id}:{missing[0]}-{missing[-1]}")
    require(
        all(value == "positive" for value in owner[first_positive:]),
        f"POSITIVE_NOT_CONTINUOUS_TO_EPISODE_END:{episode_id}",
    )
    return [value for value in owner if value is not None], normalized


def validate_upstream(
    args: argparse.Namespace, dataset_storage: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    protocol = args.protocol.resolve()
    template = args.template.resolve()
    bundle_manifest_path = args.review_bundle.resolve() / "bundle_manifest.json"
    task_audit_path = args.task_semantics_audit.resolve()
    split_path = args.dataset_root.resolve() / "split_manifest.json"
    conversion_path = args.dataset_root.resolve() / "conversion_manifest.json"
    info_path = args.dataset_root.resolve() / "meta/info.json"
    review_index_path = args.review_bundle.resolve() / "review_index.json"
    p8_binding_path = args.p8_binding.resolve()

    task_audit = load_json(task_audit_path)
    bundle = load_json(bundle_manifest_path)
    p8_binding = load_json(p8_binding_path)
    require(task_audit.get("task_semantics_audit") == "pass", "TASK_SEMANTICS_AUDIT_NOT_PASS")
    require(task_audit.get("semantic_equivalence") is True, "TASK_SEMANTICS_NOT_EQUIVALENT")
    require(
        task_audit["active_v2"]["protocol_sha256"] == sha256(protocol),
        "V2_PROTOCOL_SHA_MISMATCH",
    )
    require(
        task_audit["active_v2"]["template_sha256"] == sha256(template),
        "V2_TEMPLATE_SHA_MISMATCH",
    )
    require(
        task_audit["historical_v1"]["review_bundle_manifest_sha256"]
        == sha256(bundle_manifest_path),
        "REVIEW_BUNDLE_SHA_MISMATCH",
    )
    require(
        task_audit["frozen_stage1_evidence"]["dataset_tree_sha256"]
        == EXPECTED_P8_DATASET_SHA256,
        "TASK_AUDIT_P8_SHA_MISMATCH",
    )
    require(
        dataset_storage["tree_sha256"] == EXPECTED_P8_DATASET_SHA256,
        "P8_DATASET_STORAGE_SHA_MISMATCH",
    )
    require(
        p8_binding["dataset"]["storage_tree"] == dataset_storage,
        "P8_FROZEN_STORAGE_BINDING_MISMATCH",
    )
    require(bundle.get("dataset_unchanged") is True, "REVIEW_BUNDLE_DATASET_CHANGED")
    require(
        bundle["dataset_storage_before"] == dataset_storage,
        "REVIEW_BUNDLE_DATASET_BEFORE_MISMATCH",
    )
    require(
        bundle["dataset_storage_after"] == dataset_storage,
        "REVIEW_BUNDLE_DATASET_AFTER_MISMATCH",
    )
    require(
        bundle["upstream_manifests"]["split_manifest.json"]["sha256"] == sha256(split_path)
        and p8_binding["dataset"]["manifest_files"]["split_manifest.json"] == sha256(split_path),
        "SPLIT_MANIFEST_SHA_MISMATCH",
    )
    require(
        task_audit["frozen_stage1_evidence"]["conversion_manifest_sha256"]
        == sha256(conversion_path),
        "CONVERSION_MANIFEST_SHA_MISMATCH",
    )
    for relative, expected in bundle["bundle_files_before_manifest"].items():
        path = args.review_bundle.resolve() / relative
        require(
            path.stat().st_size == expected["file_size"]
            and sha256(path) == expected["sha256"],
            f"REVIEW_BUNDLE_FILE_DRIFT:{relative}",
        )

    return {
        "reviewed_labels": binding(args.reviewed_labels),
        "ingestion_validator": binding(Path(__file__)),
        "v2_protocol": binding(protocol),
        "v2_template": binding(template),
        "review_bundle_manifest": binding(bundle_manifest_path),
        "review_index": binding(review_index_path),
        "task_semantics_audit": binding(task_audit_path),
        "split_manifest": binding(split_path),
        "conversion_manifest": binding(conversion_path),
        "dataset_info": binding(info_path),
        "p8_source_binding": binding(p8_binding_path),
        "p8_dataset_hash_implementation": binding(ROOT / "src/forcesmolvla/dataset_binding.py"),
    }


def build_inventory(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if not args.validate_only:
        for output in (args.inventory_output.resolve(), args.readiness_output.resolve()):
            if output.exists():
                raise FileExistsError(f"refusing to overwrite append-only artifact: {output}")

    reviewed = load_json(args.reviewed_labels.resolve())
    template = load_json(args.template.resolve())
    reviewed_by_id, schema_validation = validate_review_schema(reviewed, template)
    dataset_storage = dataset_storage_binding(args.dataset_root.resolve())
    bindings = validate_upstream(args, dataset_storage)
    bindings["p8_dataset_storage"] = {
        "dataset_root": args.dataset_root.resolve().relative_to(ROOT).as_posix(),
        "hash_algorithm": (
            "sha256(concat(sorted(relative_path + NUL + file_sha256 + LF)))"
        ),
        "included_roots": dataset_storage["roots"],
        "file_count": dataset_storage["file_count"],
        "tree_sha256": dataset_storage["tree_sha256"],
    }

    split_manifest = load_json(args.dataset_root.resolve() / "split_manifest.json")
    conversion = load_json(args.dataset_root.resolve() / "conversion_manifest.json")
    info = load_json(args.dataset_root.resolve() / "meta/info.json")
    review_index = load_json(args.review_bundle.resolve() / "review_index.json")
    split_by_episode, split_sets = split_lookup(split_manifest)
    require(
        info.get("total_episodes") == 47 and info.get("total_frames") == 38_639,
        "DATASET_INVENTORY_INVALID",
    )
    require(
        review_index.get("episode_count") == 47
        and review_index.get("frame_count") == 38_639,
        "REVIEW_INDEX_INVENTORY_INVALID",
    )

    conversion_by_id = {episode["raw_episode_id"]: episode for episode in conversion["episodes"]}
    review_index_by_id = {episode["episode_id"]: episode for episode in review_index["episodes"]}
    require(
        len(conversion_by_id) == len(review_index_by_id) == 47
        and set(reviewed_by_id)
        == set(conversion_by_id)
        == set(review_index_by_id)
        == set(split_by_episode),
        "EPISODE_INVENTORY_MISMATCH",
    )

    stats = {
        split: {
            "episode_count": 0,
            "frame_count": 0,
            "classes": {
                class_name: {"frame_count": 0, "episode_count": 0}
                for class_name in CLASS_NAMES
            },
        }
        for split in ("train", "validation", "test")
    }
    split_rows: dict[str, set[tuple[str, int]]] = {split: set() for split in stats}
    split_global_indices: dict[str, set[int]] = {split: set() for split in stats}
    all_global_indices: set[int] = set()
    inventory_digest = hashlib.sha256()
    inventory_episodes = []

    for expected_output, conversion_episode in enumerate(conversion["episodes"]):
        episode_id = conversion_episode["raw_episode_id"]
        label = reviewed_by_id[episode_id]
        review_meta = review_index_by_id[episode_id]
        source_split = split_by_episode[episode_id]
        output_split = "validation" if source_split == "val" else source_split
        require(
            conversion_episode["output_episode_index"] == expected_output,
            f"OUTPUT_EPISODE_INDEX_INVALID:{episode_id}",
        )
        require(
            label["output_episode_index"] == expected_output,
            f"LABEL_OUTPUT_INDEX_INVALID:{episode_id}",
        )
        require(
            label["split"] == source_split == conversion_episode["split"],
            f"EPISODE_SPLIT_MISMATCH:{episode_id}",
        )

        chunk_index, file_index = divmod(expected_output, info["chunks_size"])
        relative = info["data_path"].format(
            chunk_index=chunk_index,
            file_index=file_index,
            episode_chunk=chunk_index,
        )
        table = pq.read_table(
            args.dataset_root.resolve() / relative,
            columns=["frame_index", "episode_index", "index"],
        ).to_pydict()
        frames = table["frame_index"]
        episode_indices = table["episode_index"]
        global_indices = table["index"]
        frame_count = conversion_episode["frames"]
        require(len(frames) == frame_count, f"EPISODE_FRAME_COUNT_MISMATCH:{episode_id}")
        require(frames == list(range(frame_count)), f"FRAME_INDEX_INVALID:{episode_id}")
        require(
            set(episode_indices) == {expected_output},
            f"PARQUET_EPISODE_INDEX_INVALID:{episode_id}",
        )
        require(
            len(global_indices) == len(set(global_indices)),
            f"DUPLICATE_GLOBAL_INDEX:{episode_id}",
        )
        require(
            review_meta["output_episode_index"] == expected_output
            and review_meta["split"] == source_split
            and review_meta["parquet_relative_path"] == relative
            and review_meta["frame_indices"] == frames
            and review_meta["dataset_global_indices"] == global_indices,
            f"REVIEW_INDEX_ROW_IDENTITY_MISMATCH:{episode_id}",
        )

        labels, intervals = classify_episode(label, frame_count)
        class_counts = Counter(labels)
        episode_row_digest = hashlib.sha256()
        episode_label_digest = hashlib.sha256()
        for frame_index, (global_index, class_name) in enumerate(
            zip(global_indices, labels, strict=True)
        ):
            row_identity = (relative, frame_index)
            require(
                row_identity not in split_rows[output_split],
                f"DUPLICATE_ROW_IDENTITY:{episode_id}:{frame_index}",
            )
            require(global_index not in all_global_indices, f"GLOBAL_ROW_LEAKAGE:{global_index}")
            split_rows[output_split].add(row_identity)
            split_global_indices[output_split].add(global_index)
            all_global_indices.add(global_index)
            row_line = f"{relative}\0{frame_index}\0{global_index}\n".encode()
            label_line = f"{episode_id}\0{frame_index}\0{class_name}\n".encode()
            episode_row_digest.update(row_line)
            episode_label_digest.update(label_line)
            inventory_digest.update(row_line)
            inventory_digest.update(label_line)

        split_stats = stats[output_split]
        split_stats["episode_count"] += 1
        split_stats["frame_count"] += frame_count
        for class_name in CLASS_NAMES:
            count = class_counts[class_name]
            split_stats["classes"][class_name]["frame_count"] += count
            split_stats["classes"][class_name]["episode_count"] += int(count > 0)
        inventory_episodes.append(
            {
                "episode_id": episode_id,
                "output_episode_index": expected_output,
                "split": output_split,
                "source_split_name": source_split,
                "frame_count": frame_count,
                "source_data_relative_path": relative,
                "row_identity": {
                    "frame_index_range_inclusive": [0, frame_count - 1],
                    "dataset_global_index_range_inclusive": [
                        min(global_indices),
                        max(global_indices),
                    ],
                    "sha256": episode_row_digest.hexdigest(),
                },
                "class_intervals_inclusive": intervals,
                "class_frame_counts": {
                    class_name: class_counts[class_name] for class_name in CLASS_NAMES
                },
                "frame_label_sha256": episode_label_digest.hexdigest(),
            }
        )

    require(
        all_global_indices == set(range(info["total_frames"])),
        "GLOBAL_ROW_INVENTORY_NOT_CONTIGUOUS",
    )
    episode_intersections = {
        "train_validation": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["val"] & split_sets["test"]),
    }
    row_intersections = {
        "train_validation": len(split_rows["train"] & split_rows["validation"]),
        "train_test": len(split_rows["train"] & split_rows["test"]),
        "validation_test": len(split_rows["validation"] & split_rows["test"]),
    }
    global_index_intersections = {
        "train_validation": len(split_global_indices["train"] & split_global_indices["validation"]),
        "train_test": len(split_global_indices["train"] & split_global_indices["test"]),
        "validation_test": len(split_global_indices["validation"] & split_global_indices["test"]),
    }
    require(not any(episode_intersections.values()), "EPISODE_LEAKAGE")
    require(not any(row_intersections.values()), "ROW_IDENTITY_LEAKAGE")
    require(not any(global_index_intersections.values()), "GLOBAL_INDEX_LEAKAGE")
    require(
        sum(value["frame_count"] for value in stats.values()) == 38_639,
        "TOTAL_FRAME_COUNT_INVALID",
    )
    for split_stats in stats.values():
        split_stats["trainable_frame_count"] = sum(
            split_stats["classes"][name]["frame_count"] for name in TRAINABLE_CLASS_NAMES
        )
        split_stats["ignored_frame_count"] = split_stats["classes"]["ambiguous"]["frame_count"]
        split_stats["required_trainable_classes_present"] = all(
            split_stats["classes"][name]["frame_count"] > 0 for name in TRAINABLE_CLASS_NAMES
        )
        require(
            split_stats["required_trainable_classes_present"],
            "SPLIT_CLASS_COVERAGE_INCOMPLETE",
        )

    leakage = {
        "episode_leakage": False,
        "episode_intersection_counts": episode_intersections,
        "row_leakage": False,
        "row_identity_intersection_counts": row_intersections,
        "global_index_intersection_counts": global_index_intersections,
        "unique_row_identity_count": sum(len(values) for values in split_rows.values()),
        "unique_global_index_count": len(all_global_indices),
    }
    validation = {
        **schema_validation,
        "intervals_valid": True,
        "positive_continuous_to_episode_end": True,
        "overlapping_frame_count": 0,
        "unlabeled_frame_count": 0,
        "episode_leakage": False,
        "row_leakage": False,
        "p8_dataset_storage_exact_match": True,
    }
    inventory = {
        "schema_version": "force_rft_task2_frame_label_inventory.v2",
        "artifact_status": "PASS_APPEND_ONLY_VALIDATED_LABEL_INGESTION",
        "operation": "R0_LABEL_INGESTION_AND_READINESS_ARTIFACT",
        "bindings": bindings,
        "label_semantics": {
            "interval_convention": "inclusive",
            "positive_interval": "first_confident_complete_frame through episode final frame",
            "review_timestamp": (
                "optional; absent/null accepted; present value must be timezone-aware ISO-8601"
            ),
            "review_timestamp_autofill": False,
            "ambiguous_disposition": "excluded_from_training_threshold_selection_and_metrics",
        },
        "validation": validation,
        "leakage_checks": leakage,
        "class_statistics": stats,
        "frame_inventory_sha256": inventory_digest.hexdigest(),
        "episodes": inventory_episodes,
        "scope_boundaries": {
            "reward_classifier_training_performed": False,
            "optimizer_or_checkpoint_created": False,
            "human_labels_modified": False,
            "reward_or_terminal_created": False,
            "g1_or_g2_created": False,
            "twin_q_cal_ql_or_actor_training_performed": False,
        },
    }
    summary = {
        "REVIEWED_LABEL_SHA256": bindings["reviewed_labels"]["sha256"],
        "MANUAL_AUDIT_COMPLETE": "yes",
        "SCHEMA_VALID": "yes",
        "INTERVALS_VALID": "yes",
        "OVERLAPPING_FRAME_COUNT": 0,
        "UNLABELED_FRAME_COUNT": 0,
        "TRAIN_CLASS_COVERAGE": "yes",
        "VALIDATION_CLASS_COVERAGE": "yes",
        "TEST_CLASS_COVERAGE": "yes",
        "EPISODE_LEAKAGE": "no",
        "ROW_LEAKAGE": "no",
        "INGESTION_ARTIFACT_CREATED": "yes",
        "READINESS_ARTIFACT_CREATED": "yes",
        "DEVELOPMENT_R0_TRAINING_DATA_READY": "yes",
        "FORMAL_INDEPENDENT_HELDOUT_READY": "no",
        "R0_TRAINING_AUTHORIZED": "no",
        "NEXT_ALLOWED_ACTION": "request_R0_classifier_training_approval",
    }
    return inventory, summary


def write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
            stream.write(json_bytes(payload))
            temporary = Path(stream.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviewed-labels",
        type=Path,
        default=ROOT / "labels/task2_reward_frame_labels.v2.reviewed.json",
    )
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/task2_lerobotv3")
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/task2_reward_labeling_protocol.v2.md"
    )
    parser.add_argument(
        "--template", type=Path, default=ROOT / "labels/task2_reward_frame_labels.v2.template.json"
    )
    parser.add_argument(
        "--review-bundle",
        type=Path,
        default=ROOT / "artifacts/development/stage2/task2_reward_review_bundle_v1",
    )
    parser.add_argument(
        "--task-semantics-audit",
        type=Path,
        default=ROOT / "artifacts/development/stage2/task_semantics_audit.v4.json",
    )
    parser.add_argument(
        "--p8-binding",
        type=Path,
        default=ROOT / "artifacts/development/p8_v4_2_r7_source_binding.json",
    )
    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=ROOT
        / "artifacts/development/stage2/reward_classifier/task2_frame_label_inventory.v2.json",
    )
    parser.add_argument(
        "--readiness-output",
        type=Path,
        default=ROOT / "artifacts/development/stage2/s2_r0_label_ingestion_readiness.v4.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    inventory, summary = build_inventory(args)
    if args.validate_only:
        summary["INGESTION_ARTIFACT_CREATED"] = "no"
        summary["READINESS_ARTIFACT_CREATED"] = "no"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    inventory_bytes = json_bytes(inventory)
    inventory_sha = hashlib.sha256(inventory_bytes).hexdigest()
    readiness = {
        "schema_version": "force_rft_s2_r0_label_ingestion_readiness.v4",
        "artifact_status": "PASS_DEVELOPMENT_R0_TRAINING_DATA_READY",
        "operation": "R0_LABEL_INGESTION_AND_READINESS_ARTIFACT",
        "bindings": {
            **inventory["bindings"],
            "frame_label_inventory": {
                "path": args.inventory_output.resolve().relative_to(ROOT).as_posix(),
                "sha256": inventory_sha,
                "file_size": len(inventory_bytes),
            },
        },
        "validation": inventory["validation"],
        "leakage_checks": inventory["leakage_checks"],
        "class_statistics": inventory["class_statistics"],
        "readiness": {
            "DEVELOPMENT_R0_TRAINING_DATA_READY": "yes",
            "FORMAL_INDEPENDENT_HELDOUT_READY": "no",
            "R0_TRAINING_AUTHORIZED": "no",
            "NEXT_ALLOWED_ACTION": "request_R0_classifier_training_approval",
            "development_heldout": "episode_disjoint_within_task2",
            "formal_heldout": "independent_collection_run_not_available",
        },
        "scope_boundaries": inventory["scope_boundaries"],
    }

    created: list[Path] = []
    try:
        write_exclusive_json(args.inventory_output.resolve(), inventory)
        created.append(args.inventory_output.resolve())
        write_exclusive_json(args.readiness_output.resolve(), readiness)
        created.append(args.readiness_output.resolve())
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    require(
        sha256(args.inventory_output.resolve()) == inventory_sha,
        "INVENTORY_WRITE_SHA_MISMATCH",
    )
    for key, value in summary.items():
        print(f"{key} = {value}")


if __name__ == "__main__":
    main()
