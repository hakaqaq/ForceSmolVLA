from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/reward_classifier/label_reward_frames.py"


def _module():
    spec = importlib.util.spec_from_file_location("label_reward_frames", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task3_builds_a_16_train_4_validation_label_workspace(tmp_path: Path) -> None:
    module = _module()
    workspace = tmp_path / "task3/reward_labeling"
    template, protocol = module.build_workspace(
        task_id="task3",
        dataset_root=ROOT / "datasets/task3_lerobotv3",
        workspace=workspace,
        train_episodes=16,
        val_episodes=4,
    )
    labels = json.loads(template.read_text())
    assert labels["task_id"] == "task3"
    assert labels["episode_count"] == 20
    assert [item["split"] for item in labels["episodes"]].count("train") == 16
    assert [item["split"] for item in labels["episodes"]].count("val") == 4
    assert all(item["manual_review_status"] == "unreviewed" for item in labels["episodes"])
    assert all(
        not {"reviewer_id", "review_timestamp", "notes"} & set(item)
        for item in labels["episodes"]
    )
    assert "green plug" in protocol.read_text()
