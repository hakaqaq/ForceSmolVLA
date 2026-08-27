#!/usr/bin/env python3
"""Build a metadata-only manual reward-frame review bundle for task2.

The dataset is read-only.  No images are copied and every human judgment field
is emitted blank.  The bundle manifest is written last as the completeness
marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from preflight_p6_variants_gpu import _dataset_storage_binding  # noqa: E402


EXPECTED_P8_SHA = "f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d"
EXPECTED_MANIFESTS = {
    "conversion_manifest.json": "b1d625b892e9df763c5bb0ffbe1ef78996e50bec8e27da1fca2806906d74d477",
    "normalizer_manifest.json": "c053d6aadd9db1dd7e365afdb08ef020d10b990b2eec1a9103ffca5b1a1f6e7e",
    "split_manifest.json": "5e63e21d1daf47cfe51fe169497cc77f69bc42e7c919cce708cb2bb0de3dc8d7",
}
HUMAN_FIELDS = {
    "last_confident_incomplete_frame": None,
    "first_confident_complete_frame": None,
    "hard_negative_intervals": [],
    "ordinary_negative_intervals": [],
    "ambiguous_intervals": [],
    "completion_visible": None,
    "completion_stable": None,
    "positive_available": None,
    "reviewer_id": None,
    "review_timestamp": None,
    "confidence": None,
    "notes": None,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def split_lookup(split_manifest: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for episode_id in split_manifest[split]:
            if episode_id in lookup:
                raise RuntimeError(f"duplicate split episode: {episode_id}")
            lookup[episode_id] = split
    return lookup


def require_blank_labels(value: dict[str, Any]) -> None:
    for key, blank in HUMAN_FIELDS.items():
        if value[key] != blank:
            raise RuntimeError(f"programmatic label population forbidden: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/task2_lerobotv3")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=ROOT / "artifacts/development/stage2/task2_reward_review_bundle_v1",
    )
    parser.add_argument(
        "--label-template",
        type=Path,
        default=ROOT / "labels/task2_reward_frame_labels.v1.template.json",
    )
    args = parser.parse_args()

    if args.bundle_dir.exists() or args.label_template.exists():
        raise RuntimeError("review outputs already exist; use validation instead of overwriting")
    dataset_root = args.dataset_root.resolve()
    before = _dataset_storage_binding(dataset_root)
    if before["tree_sha256"] != EXPECTED_P8_SHA:
        raise RuntimeError("STAGE1_DATA_DRIFT")
    for name, expected in EXPECTED_MANIFESTS.items():
        if sha256(dataset_root / name) != expected:
            raise RuntimeError(f"STAGE1_MANIFEST_DRIFT: {name}")

    info = json.loads((dataset_root / "meta/info.json").read_text())
    conversion = json.loads((dataset_root / "conversion_manifest.json").read_text())
    split_manifest = json.loads((dataset_root / "split_manifest.json").read_text())
    split_by_episode = split_lookup(split_manifest)
    if info["fps"] != 30 or info["total_episodes"] != 47 or len(conversion["episodes"]) != 47:
        raise RuntimeError("task2 inventory prerequisite mismatch")
    if conversion["camera_order"] != ["camera1", "camera2"] or conversion["camera_roles"] != {
        "camera1": "D435 third-person",
        "camera2": "D405 wrist",
    }:
        raise RuntimeError("frozen dual-camera contract mismatch")

    episodes: list[dict[str, Any]] = []
    label_episodes: list[dict[str, Any]] = []
    total_frames = 0
    for expected_output, episode in enumerate(conversion["episodes"]):
        output_index = episode["output_episode_index"]
        episode_id = episode["raw_episode_id"]
        if output_index != expected_output or split_by_episode.get(episode_id) != episode["split"]:
            raise RuntimeError(f"episode mapping/split mismatch: {episode_id}")
        chunk_index, file_index = divmod(output_index, info["chunks_size"])
        relative_parquet = info["data_path"].format(
            chunk_index=chunk_index, file_index=file_index, episode_chunk=chunk_index
        )
        parquet_path = dataset_root / relative_parquet
        table = pq.read_table(
            parquet_path, columns=["timestamp", "frame_index", "episode_index", "index"]
        )
        values = table.to_pydict()
        frame_count = len(values["frame_index"])
        if frame_count != episode["frames"] or frame_count != episode["diagnostics"]["frames"]:
            raise RuntimeError(f"frame count mismatch: {episode_id}")
        if values["frame_index"] != list(range(frame_count)):
            raise RuntimeError(f"non-contiguous frame index: {episode_id}")
        if set(values["episode_index"]) != {output_index}:
            raise RuntimeError(f"output episode index mismatch: {episode_id}")
        if any(b <= a for a, b in zip(values["timestamp"], values["timestamp"][1:])):
            raise RuntimeError(f"non-monotonic timestamps: {episode_id}")

        episodes.append(
            {
                "episode_id": episode_id,
                "output_episode_index": output_index,
                "split": episode["split"],
                "task_text_from_conversion_manifest": episode["task"],
                "frame_count": frame_count,
                "fps": 30,
                "parquet_relative_path": relative_parquet,
                "frame_indices": values["frame_index"],
                "timestamps_seconds": values["timestamp"],
                "dataset_global_indices": values["index"],
                "source_row_reference_format": (
                    f"task2_lerobotv3/{relative_parquet}#row={{frame_index}}"
                ),
                "camera_row_identity": {
                    "D435 third-person": "observation.images.camera1@same_parquet_row",
                    "D405 wrist": "observation.images.camera2@same_parquet_row",
                },
                "candidate_10hz_frame_indices": list(range(0, frame_count, 3)),
                "detector_calibration_frame_indices": "all_30hz_frames",
            }
        )
        label_entry = {
            "episode_id": episode_id,
            "output_episode_index": output_index,
            "split": episode["split"],
            "task_outcome_context": "success",
            "outcome_source": "retrospective_operator_attestation",
            "outcome_is_not_a_frame_label": True,
            "manual_review_status": "unreviewed",
            **HUMAN_FIELDS,
        }
        require_blank_labels(label_entry)
        label_episodes.append(label_entry)
        total_frames += frame_count

    if set(split_by_episode) != {episode["episode_id"] for episode in episodes}:
        raise RuntimeError("split does not cover exactly the converted episodes")
    if total_frames != info["total_frames"]:
        raise RuntimeError("total frame count mismatch")

    review_index = {
        "schema_version": "force_rft_task2_reward_review_index.v1",
        "artifact_status": "MANUAL_REVIEW_MATERIALS_ONLY",
        "dataset_root_id": "task2_lerobotv3",
        "dataset_root_absolute_path": str(dataset_root),
        "fps": 30,
        "episode_count": len(episodes),
        "frame_count": total_frames,
        "camera_order": ["observation.images.camera1", "observation.images.camera2"],
        "camera_roles": conversion["camera_roles"],
        "image_storage": "embedded_png_bytes_in_parquet; served on demand; not copied",
        "episodes": episodes,
    }
    label_template = {
        "schema_version": "force_rft_task2_reward_frame_labels.v1",
        "artifact_status": "BLANK_MANUAL_LABEL_TEMPLATE",
        "programmatic_labels_generated": False,
        "episode_count": len(label_episodes),
        "label_definitions": {
            "positive": "physically complete insertion, not alignment/contact/partial insertion; remains complete in subsequent observable frames",
            "ordinary_negative": "clearly not yet aligned, contacting, or inserted",
            "hard_negative": "near/aligned/contacting/partially inserted but not complete",
            "ignore": "ambiguous boundary frame; excluded from training, threshold selection, and metrics",
        },
        "forbidden_inference_sources": [
            "saved=true", "episode_end", "last_valid_frame", "file_name", "episode_success_label"
        ],
        "episodes": label_episodes,
    }

    after = _dataset_storage_binding(dataset_root)
    if before != after:
        raise RuntimeError("STAGE1_DATA_CHANGED_DURING_REVIEW_INDEX_READ")

    args.bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="task2_reward_review_bundle_v1.", dir=args.bundle_dir.parent))
    try:
        atomic_json(staging / "review_index.json", review_index)
        shutil.copyfile(ROOT / "tools/reward_classifier/task2_label_ui.html", staging / "review_app.html")
        atomic_json(staging / "label_template.json", label_template)
        files = {
            path.relative_to(staging).as_posix(): {
                "file_size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": "force_rft_task2_reward_review_bundle.v1",
            "artifact_status": "REVIEW_MATERIALS_READY_MANUAL_AUDIT_INCOMPLETE",
            "dataset_storage_before": before,
            "dataset_storage_after": after,
            "dataset_unchanged": True,
            "dataset_hash_bridge": {
                "path": "artifacts/development/stage2/dataset_hash_bridge.v4.json",
                "sha256": sha256(ROOT / "artifacts/development/stage2/dataset_hash_bridge.v4.json"),
            },
            "upstream_manifests": {
                name: {"sha256": sha256(dataset_root / name)} for name in EXPECTED_MANIFESTS
            },
            "resnet_asset_manifest": {
                "path": "artifacts/development/stage2/reward_classifier/pretrained/resnet10_asset_manifest.v4.json",
                "sha256": sha256(
                    ROOT / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_asset_manifest.v4.json"
                ),
            },
            "source_files": {
                path.relative_to(ROOT).as_posix(): sha256(path)
                for path in (
                    ROOT / "tools/reward_classifier/build_task2_review_bundle.py",
                    ROOT / "tools/reward_classifier/serve_task2_label_ui.py",
                    ROOT / "tools/reward_classifier/task2_label_ui.html",
                    ROOT / "docs/task2_reward_labeling_protocol.md",
                    ROOT / "tests/test_s2_r0_data_audit.py",
                )
            },
            "episode_count": len(episodes),
            "frame_count": total_frames,
            "split_episode_counts": {
                split: sum(episode["split"] == split for episode in episodes)
                for split in ("train", "val", "test")
            },
            "manual_audit_complete": False,
            "programmatic_labels_generated": False,
            "images_copied": False,
            "reward_predictions_created": False,
            "reward_or_terminal_created": False,
            "development_heldout": "episode_disjoint_within_task2",
            "formal_heldout": "independent_collection_run",
            "classifier_data_readiness": {
                "positive_episode_count": None,
                "ordinary_negative_episode_count": None,
                "hard_negative_episode_count": None,
                "ignored_frame_count": None,
                "existing_task2_classifier_data_ready": False,
                "reason": "manual frame-level audit not complete",
            },
            "bundle_files_before_manifest": files,
        }
        atomic_json(staging / "bundle_manifest.json", manifest)
        staging.replace(args.bundle_dir)
        atomic_json(args.label_template, label_template)
    except Exception:
        # Leave the staging directory as explicit interruption evidence; never
        # overwrite a completed bundle or touch the source dataset.
        raise

    print(
        json.dumps(
            {
                "bundle": str(args.bundle_dir),
                "bundle_manifest_sha256": sha256(args.bundle_dir / "bundle_manifest.json"),
                "label_template": str(args.label_template),
                "label_template_sha256": sha256(args.label_template),
                "episodes": len(episodes),
                "frames": total_frames,
                "p8_before": before["tree_sha256"],
                "p8_after": after["tree_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
