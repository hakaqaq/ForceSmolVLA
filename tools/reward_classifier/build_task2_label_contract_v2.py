#!/usr/bin/env python3
"""Create the append-only task2 v2 label template and semantics audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forcesmolvla.dataset_binding import dataset_storage_binding  # noqa: E402


CANONICAL_TASK_PROMPT = "Pick up the purple ring and place it onto the red peg."
PHYSICAL_TASK_DESCRIPTION = (
    "Pick up the purple ring, align its center hole with the red peg, "
    "lower the ring over the peg, release it, and leave it stably supported "
    "by the red peg/base assembly."
)
EXPECTED_V1_PROTOCOL_SHA = "31213136e922fd827a364a232fcddd631db2d141c26cdb02231376f4aa004d79"
EXPECTED_V1_TEMPLATE_SHA = "bb4df9b118e138894b8c2bd0d9d6d128fa1c35f6240ffc74b03be8fceaf765fa"
EXPECTED_V1_BUNDLE_SHA = "13d5748091c2255054d94f09a233ceeaf7d096d61c7161598e701a5fddfc3442"
EXPECTED_P8_SHA = "f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d"
EXPECTED_R5_MODEL_SHA = "49248561be7043b38bfce60f200d8bf265e1b16b4b9553ccc6aa4c87241b762e"


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


def require_blank_episode(episode: dict[str, Any]) -> None:
    for key in (
        "last_confident_incomplete_frame",
        "first_confident_complete_frame",
        "completion_visible",
        "completion_stable",
        "positive_available",
        "reviewer_id",
        "review_timestamp",
        "confidence",
        "notes",
    ):
        if episode[key] is not None:
            raise RuntimeError(f"v1 template is not blank: {episode['episode_id']}:{key}")
    for key in ("hard_negative_intervals", "ordinary_negative_intervals", "ambiguous_intervals"):
        if episode[key] != []:
            raise RuntimeError(f"v1 template is not blank: {episode['episode_id']}:{key}")
    if episode["manual_review_status"] != "unreviewed":
        raise RuntimeError(f"v1 template is not blank: {episode['episode_id']}:status")


def main() -> None:
    v1_protocol = ROOT / "docs/task2_reward_labeling_protocol.md"
    v1_template = ROOT / "labels/task2_reward_frame_labels.v1.template.json"
    v1_bundle = ROOT / "artifacts/development/stage2/task2_reward_review_bundle_v1/bundle_manifest.json"
    v2_protocol = ROOT / "docs/task2_reward_labeling_protocol.v2.md"
    v2_template = ROOT / "labels/task2_reward_frame_labels.v2.template.json"
    audit_path = ROOT / "artifacts/development/stage2/task_semantics_audit.v4.json"
    dataset_root = ROOT / "datasets/task2_lerobotv3"
    r5_model = (
        ROOT
        / "outputs/development/task2_lerobotv3_full_sft_10k_r5/checkpoints/step_010000/model.safetensors"
    )

    if v2_template.exists() or audit_path.exists():
        raise RuntimeError("append-only v2 outputs already exist; refusing overwrite")
    expected = {
        v1_protocol: EXPECTED_V1_PROTOCOL_SHA,
        v1_template: EXPECTED_V1_TEMPLATE_SHA,
        v1_bundle: EXPECTED_V1_BUNDLE_SHA,
        r5_model: EXPECTED_R5_MODEL_SHA,
    }
    for path, value in expected.items():
        if sha256(path) != value:
            raise RuntimeError(f"frozen input SHA mismatch: {path}")
    if dataset_storage_binding(dataset_root)["tree_sha256"] != EXPECTED_P8_SHA:
        raise RuntimeError("STAGE1_DATA_DRIFT")

    protocol_text = v2_protocol.read_text()
    if CANONICAL_TASK_PROMPT not in protocol_text or PHYSICAL_TASK_DESCRIPTION not in protocol_text:
        raise RuntimeError("v2 protocol task semantics mismatch")
    conversion = json.loads((dataset_root / "conversion_manifest.json").read_text())
    task_texts = sorted({episode["task"] for episode in conversion["episodes"]})
    if task_texts != [CANONICAL_TASK_PROMPT]:
        raise RuntimeError(f"unexpected Stage-1 task text: {task_texts}")

    parent = json.loads(v1_template.read_text())
    if parent["episode_count"] != 47 or len(parent["episodes"]) != 47:
        raise RuntimeError("v1 episode inventory mismatch")
    for episode in parent["episodes"]:
        require_blank_episode(episode)

    template = {
        "schema_version": "force_rft_task2_reward_frame_labels.v2",
        "artifact_status": "BLANK_MANUAL_LABEL_TEMPLATE_V2",
        "programmatic_labels_generated": False,
        "manual_audit_complete": False,
        "canonical_task_prompt": CANONICAL_TASK_PROMPT,
        "physical_task_description": PHYSICAL_TASK_DESCRIPTION,
        "conversion_task_text_semantically_equivalent": True,
        "first_confident_complete_frame_definition": (
            "first frame at which both cameras support all four positive criteria simultaneously"
        ),
        "positive_all_of": [
            "red peg clearly passes through the purple ring center hole",
            "ring has left the gripper and is no longer gripper-supported",
            "ring rests stably on the red peg/base assembly",
            "subsequent observable frames show no slip, ejection, or re-grasp",
        ],
        "ordinary_negative_any_of": [
            "ring remains on the table",
            "ring has not been successfully grasped",
            "ring and peg are clearly separated",
            "ring is clearly offset from the peg",
            "robot is in ordinary transport",
        ],
        "hard_negative_any_of": [
            "ring is above the peg",
            "ring and peg are approximately coaxial",
            "ring contacts the peg",
            "ring is partially lowered over the peg",
            "ring appears placed but remains held or supported by the gripper",
            "ring was just released but is moving, tilted, or not demonstrably stable",
        ],
        "ambiguous_ignore_any_of": [
            "occlusion prevents confirming peg-through-center-hole",
            "gripper support cannot be determined",
            "the two cameras disagree",
            "ring stability cannot be determined",
        ],
        "ambiguous_frames_excluded_from": [
            "classifier_training", "threshold_selection", "metric_computation"
        ],
        "forbidden_inference_sources": [
            "saved=true",
            "episode_end",
            "last_valid_frame",
            "file_name",
            "episode_success_label",
            "example_video_timestamp",
        ],
        "example_video_semantics_only": {
            "duration_seconds_approx": 23.87,
            "fps": 30,
            "placement_and_release_seconds_approx": 22,
            "stable_terminal_seconds_after_approx": 23,
            "may_map_to_lerobot_frame": False,
        },
        "historical_parent": {
            "protocol_path": v1_protocol.relative_to(ROOT).as_posix(),
            "protocol_sha256": EXPECTED_V1_PROTOCOL_SHA,
            "template_path": v1_template.relative_to(ROOT).as_posix(),
            "template_sha256": EXPECTED_V1_TEMPLATE_SHA,
            "disposition": "HISTORICAL_INVALID_FOR_LABELING",
        },
        "episode_count": 47,
        "episodes": parent["episodes"],
    }
    atomic_json(v2_template, template)

    source_paths = [
        ROOT / "tools/reward_classifier/build_task2_label_contract_v2.py",
        ROOT / "tools/reward_classifier/serve_task2_label_ui.py",
        ROOT / "tools/reward_classifier/task2_label_ui.html",
        ROOT / "tests/test_offline_task_semantics.py",
    ]
    audit = {
        "schema_version": "force_rft_task_semantics_audit.v4",
        "artifact_status": "PASS_APPEND_ONLY_LABELING_PROTOCOL_REPAIR",
        "task_semantics_audit": "pass",
        "semantic_equivalence": True,
        "canonical_task_prompt": CANONICAL_TASK_PROMPT,
        "physical_task_description": PHYSICAL_TASK_DESCRIPTION,
        "conversion_manifest_task_texts": task_texts,
        "stage1_task_text_invalid": False,
        "r5_retraining_required": False,
        "stage1_data_reconversion_required": False,
        "scope_of_repair": "Reward Classifier labeling protocol/template/UI only",
        "historical_v1": {
            "protocol_path": v1_protocol.relative_to(ROOT).as_posix(),
            "protocol_sha256": sha256(v1_protocol),
            "template_path": v1_template.relative_to(ROOT).as_posix(),
            "template_sha256": sha256(v1_template),
            "review_bundle_manifest_path": v1_bundle.relative_to(ROOT).as_posix(),
            "review_bundle_manifest_sha256": sha256(v1_bundle),
            "disposition": "HISTORICAL_INVALID_FOR_LABELING",
            "bytes_modified": False,
        },
        "active_v2": {
            "protocol_path": v2_protocol.relative_to(ROOT).as_posix(),
            "protocol_sha256": sha256(v2_protocol),
            "template_path": v2_template.relative_to(ROOT).as_posix(),
            "template_sha256": sha256(v2_template),
            "ui_uses_v2": True,
        },
        "frozen_stage1_evidence": {
            "r5_model_path": r5_model.relative_to(ROOT).as_posix(),
            "r5_model_sha256": sha256(r5_model),
            "dataset_hash_algorithm": "P8 original _dataset_storage_binding",
            "dataset_tree_sha256": dataset_storage_binding(dataset_root)["tree_sha256"],
            "conversion_manifest_sha256": sha256(dataset_root / "conversion_manifest.json"),
            "stage1_data_changed": False,
        },
        "example_video_evidence_scope": "task_semantics_only_not_frame_alignment",
        "reward_classifier_training_authorized": False,
        "reward_or_terminal_created": False,
        "g1_created": False,
        "g2_created": False,
        "source_files": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in source_paths
        },
    }
    atomic_json(audit_path, audit)
    print(
        json.dumps(
            {
                "protocol_v2_sha256": sha256(v2_protocol),
                "template_v2_sha256": sha256(v2_template),
                "audit_sha256": sha256(audit_path),
                "p8_dataset_sha256": audit["frozen_stage1_evidence"]["dataset_tree_sha256"],
                "r5_model_sha256": audit["frozen_stage1_evidence"]["r5_model_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
