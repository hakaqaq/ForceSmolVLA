import json

import numpy as np
import pytest

from forcesmolvla.dataset_v3 import (
    create_dataset,
    load_dataset_split,
    split_episode_indices,
    storage_features,
)


def test_feature_contract_splits_state7_and_wrench6():
    features = storage_features()
    assert features["observation.state"]["shape"] == (7,)
    assert features["observation.wrench"]["shape"] == (6,)
    assert features["action"]["shape"] == (7,)
    assert tuple(key for key in features if key.startswith("observation.images.")) == (
        "observation.images.camera1",
        "observation.images.camera2",
    )


def test_lerobot_v3_two_camera_writer_smoke(tmp_path):
    root = tmp_path / "dataset"
    dataset = create_dataset(root, repo_id="local/task1_forcesmolvla_v4_1_smoke", height=8, width=8)
    frame = {
        "task": "fixture task",
        "observation.images.camera1": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.camera2": np.full((8, 8, 3), 127, dtype=np.uint8),
        "observation.state": np.arange(7, dtype=np.float32),
        "observation.wrench": np.arange(6, dtype=np.float32),
        "action": np.arange(7, dtype=np.float32),
        "provenance.tuple_host_monotonic_ns": np.array([110], dtype=np.int64),
        "provenance.state_pose_source_stamp_ns": np.array([100], dtype=np.int64),
        "provenance.state_pose_age_ms": np.array([5.0], dtype=np.float32),
        "provenance.camera1_receive_monotonic_ns": np.array([101], dtype=np.int64),
        "provenance.camera1_age_ms": np.array([9.0], dtype=np.float32),
        "provenance.camera2_receive_monotonic_ns": np.array([102], dtype=np.int64),
        "provenance.camera2_age_ms": np.array([8.0], dtype=np.float32),
        "provenance.intercamera_skew_ms": np.array([1.0], dtype=np.float32),
        "provenance.gripper_source_stamp_ns": np.array([103], dtype=np.int64),
        "provenance.pose_source_stamp_ns": np.array([100], dtype=np.int64),
        "provenance.pose_age_ms": np.array([5.0], dtype=np.float32),
        "provenance.wrench_raw_source_stamp_ns": np.array([105], dtype=np.int64),
        "provenance.wrench_filter_output_stamp_ns": np.array([105], dtype=np.int64),
        "provenance.action_ack_receive_monotonic_ns": np.array([104], dtype=np.int64),
        "provenance.action_ack_age_ms": np.array([6.0], dtype=np.float32),
        "provenance.calibration_index": np.array([0], dtype=np.int64),
        "provenance.validity_bits": np.array([0x3F], dtype=np.int64),
    }
    dataset.add_frame(frame)
    dataset.save_episode()
    dataset.finalize()
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"].startswith("v3.")
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 1
    assert (root / "data").is_dir()


def test_writer_refuses_existing_root(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_dataset(root, repo_id="local/will-not-write")


def test_split_loader_resolves_output_indices_and_rejects_formal_development_data(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "split_manifest.json").write_text(
        json.dumps(
            {
                "train": ["episode_a"],
                "val": ["episode_c"],
                "test": ["episode_b"],
            }
        )
    )
    (root / "conversion_manifest.json").write_text(
        json.dumps(
                {
                    "artifact_status": "development_only",
                    "formal_ready": False,
                    "repo_id": "local/fixture",
                    "split": {
                        "train": ["episode_a"],
                        "val": ["episode_c"],
                        "test": ["episode_b"],
                    },
                    "episodes": [
                    {"raw_episode_id": "episode_a", "output_episode_index": 2},
                    {"raw_episode_id": "episode_b", "output_episode_index": 0},
                    {"raw_episode_id": "episode_c", "output_episode_index": 1},
                ],
            }
        )
    )
    assert split_episode_indices(root, "train") == [2]
    assert split_episode_indices(root, "val") == [1]
    assert split_episode_indices(root, "test") == [0]
    with pytest.raises(ValueError, match="approved, formal-ready"):
        load_dataset_split(
            root,
            repo_id="local/fixture",
            split_name="train",
            artifact_use="formal",
        )
