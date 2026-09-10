import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_public_sft_configs_are_self_contained() -> None:
    for task_id in ("task2", "task3", "task4"):
        config = json.loads((ROOT / f"configs/train/{task_id}.json").read_text())
        assert config["schema_version"] == "2.0"
        assert config["training"] == {
            "target_samples": 40_000,
            "validation_interval_samples": 2_000,
            "checkpoint_interval_samples": 40_000,
        }
        assert "artifacts/development" not in json.dumps(config)
