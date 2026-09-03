from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels/task2_reward_frame_labels.json"
TRAINER = ROOT / "tools/reward_classifier/train_reward_classifier.py"


def _trainer_module():
    spec = importlib.util.spec_from_file_location("reward_classifier_training", TRAINER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task2_has_one_unambiguous_human_label_file() -> None:
    assert sorted(path.name for path in (ROOT / "labels").glob("task2*.json")) == [
        "task2_reward_frame_labels.json"
    ]
    value = json.loads(LABELS.read_text(encoding="utf-8"))
    assert value["schema_version"] == "force_rft_task2_reward_frame_labels.v2"
    assert value["episode_count"] == len(value["episodes"]) == 47
    assert all(
        episode["manual_review_status"] == "human_reviewed"
        for episode in value["episodes"]
    )


def test_reward_classifier_accepts_a_20_episode_reviewed_subset() -> None:
    module = _trainer_module()
    reviewed = json.loads(LABELS.read_text(encoding="utf-8"))
    selected = (
        [item for item in reviewed["episodes"] if item["split"] == "train"][:16]
        + [item for item in reviewed["episodes"] if item["split"] == "val"][:4]
    )
    reviewed = {**reviewed, "episode_count": len(selected), "episodes": selected}
    dataset = ROOT / "datasets/task2_lerobotv3"
    inventory = module._label_inventory(
        reviewed,
        json.loads((dataset / "split_manifest.json").read_text(encoding="utf-8")),
        json.loads((dataset / "conversion_manifest.json").read_text(encoding="utf-8")),
        json.loads((dataset / "meta/info.json").read_text(encoding="utf-8")),
    )
    assert len(inventory["episodes"]) == 20
    assert inventory["class_statistics"]["train"]["episode_count"] == 16
    assert inventory["class_statistics"]["validation"]["episode_count"] == 4
    assert "3775" not in TRAINER.read_text(encoding="utf-8")


def test_task2_reward_configs_use_final_task_scoped_names() -> None:
    for name in (
        "reward_classifier_training.json",
        "forcerft_offline_reward_transitions.json",
    ):
        value = json.loads((ROOT / "configs/tasks/task2" / name).read_text())
        text = json.dumps(value).lower()
        assert value["task_id"] == "task2"
        assert value["status"] == "final"
        assert not any(token in text for token in ("stage2", "development", "g1_"))
        assert "sha256" not in text
