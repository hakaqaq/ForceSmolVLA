from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from forcesmolvla.rft.detector_reward_transitions import load_training_transitions


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/task2_forcerft_offline_reward_transitions"


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def test_task2_reward_transition_dataset_has_final_names_and_no_sha_fields() -> None:
    assert sorted(path.name for path in DATASET.iterdir()) == [
        "dataset_manifest.json",
        "forcerft_offline_td_transitions.parquet",
        "reward_detector_frame_scores.parquet",
    ]
    manifest = json.loads((DATASET / "dataset_manifest.json").read_text())
    assert manifest["task_id"] == "task2"
    assert manifest["dataset_type"] == "forcerft_offline_reward_transitions"
    assert manifest["status"] == "final"
    assert not any("sha" in key.lower() for key in _keys(manifest))
    assert "sha" not in str(
        pq.read_schema(DATASET / "forcerft_offline_td_transitions.parquet")
    ).lower()
    assert "sha" not in str(
        pq.read_schema(DATASET / "reward_detector_frame_scores.parquet")
    ).lower()
    assert "stage1" not in str(
        pq.read_schema(DATASET / "forcerft_offline_td_transitions.parquet")
    ).lower()


def test_actor_and_critic_loader_reads_only_task2_train_rows() -> None:
    table = load_training_transitions(DATASET, task_id="task2")
    assert table.num_rows == 10075
    assert set(table.column("split").to_pylist()) == {"train"}
