from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "artifacts/development/stage2/dataset_hash_bridge.v4.json"
ASSET = (
    ROOT
    / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_asset_manifest.v4.json"
)
BUNDLE = ROOT / "artifacts/development/stage2/task2_reward_review_bundle_v1"
TEMPLATE = ROOT / "labels/task2_reward_frame_labels.v1.template.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_dataset_hash_bridge_proves_identical_p8_payload() -> None:
    bridge = json.loads(BRIDGE.read_text())
    assert bridge["p8_original_implementation"]["tree_sha256"] == (
        "f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d"
    )
    assert bridge["r0_preparation_implementation"]["tree_sha256"] == (
        "daa3d3b876cddc25caa4effa1e7ac8c55e875738367304c4d51a18653118aa01"
    )
    proof = bridge["bridge_proof"]
    assert proof["common_payload_file_count"] == 51
    assert proof["common_records_match_path_size_sha256"] is True
    assert proof["r0_only_file_count"] == 6
    assert proof["p8_only_file_count"] == 0
    assert proof["stage1_data_drift"] is False


def test_resnet_safe_asset_round_trip_and_coverage() -> None:
    manifest = json.loads(ASSET.read_text())
    assert manifest["unsafe_pickle_asset"]["sha256"] == (
        "175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b"
    )
    assert manifest["repository_copy"]["classification"] == "truncated_committed_pickle"
    assert manifest["repository_copy"]["git_lfs_pointer"] is False
    assert manifest["parameter_key_count"] == 38
    coverage = manifest["expected_parameter_coverage"]
    assert coverage["coverage_fraction"] == 1.0
    assert coverage["backbone_exact"] is True
    assert coverage["forbidden_unexpected_paths"] == []
    assert len(coverage["allowed_unused_imagenet_output_head_paths"]) == 2
    assert manifest["parameter_tree_content_sha256_before"] == manifest[
        "parameter_tree_content_sha256_after"
    ]
    safe_path = ROOT / manifest["safe_asset"]["relative_path"]
    assert sha256(safe_path) == manifest["safe_asset"]["sha256"]
    with np.load(safe_path, allow_pickle=False) as archive:
        assert len(archive.files) == 38


def test_review_bundle_has_all_episodes_and_no_images_or_labels() -> None:
    index = json.loads((BUNDLE / "review_index.json").read_text())
    template = json.loads(TEMPLATE.read_text())
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text())
    assert index["episode_count"] == template["episode_count"] == 47
    assert index["frame_count"] == 38_639
    assert manifest["split_episode_counts"] == {"train": 38, "val": 5, "test": 4}
    assert manifest["dataset_unchanged"] is True
    assert manifest["images_copied"] is False
    assert manifest["manual_audit_complete"] is False
    assert manifest["programmatic_labels_generated"] is False
    assert manifest["classifier_data_readiness"]["existing_task2_classifier_data_ready"] is False
    assert (BUNDLE / "label_template.json").read_bytes() == TEMPLATE.read_bytes()
    assert not any(path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4"} for path in BUNDLE.rglob("*"))

    ids = [episode["episode_id"] for episode in index["episodes"]]
    assert len(ids) == len(set(ids)) == 47
    for episode, label in zip(index["episodes"], template["episodes"], strict=True):
        assert episode["episode_id"] == label["episode_id"]
        assert len(episode["frame_indices"]) == episode["frame_count"]
        assert len(episode["timestamps_seconds"]) == episode["frame_count"]
        assert episode["frame_indices"] == list(range(episode["frame_count"]))
        assert label["manual_review_status"] == "unreviewed"
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
            assert label[key] is None
        for key in ("hard_negative_intervals", "ordinary_negative_intervals", "ambiguous_intervals"):
            assert label[key] == []


def test_review_ui_contract_and_first_dual_camera_row() -> None:
    source = (ROOT / "tools/reward_classifier/serve_task2_label_ui.py").read_text()
    spec = importlib.util.spec_from_file_location("task2_label_server", ROOT / "tools/reward_classifier/serve_task2_label_ui.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index = module.load_index(BUNDLE)
    store = module.FrameStore(ROOT / "datasets/task2_lerobotv3", index)
    episode_id = index["episodes"][0]["episode_id"]
    camera1, type1 = store.image(episode_id, 0, "camera1")
    camera2, type2 = store.image(episode_id, 0, "camera2")
    assert type1 == type2 == "image/png"
    assert camera1.startswith(b"\x89PNG") and camera2.startswith(b"\x89PNG")
    assert camera1 != camera2
    assert "do_POST" in source and "read-only server" in source
    html = (BUNDLE / "review_app.html").read_text()
    assert "D435 third-person" in html and "D405 wrist" in html
    assert "1000 / 30" in html
    assert "saved=true" in html and "last_valid_frame" in html
