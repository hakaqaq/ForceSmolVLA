import json
from pathlib import Path
import sys

import pytest

from train_forcesmolvla_sft import _load_config, parse_args


ROOT = Path(__file__).parents[1]


def test_training_entry_is_dataset_agnostic() -> None:
    source = (ROOT / "tools/train_forcesmolvla_sft.py").read_text(encoding="utf-8").lower()
    assert "task2" not in source
    assert not (ROOT / "tools/train_task2_full_gpu.py").exists()


def test_checkpoint_payloads_use_repository_config_not_generated_audit() -> None:
    source = (ROOT / "tools/train_forcesmolvla_sft.py").read_text(encoding="utf-8")
    assert '"manifests/action_delta_spec.json"' in source
    assert '"configs/action_delta_spec.json"' in source
    assert "artifacts/development" not in source


def test_cli_requires_only_dataset_and_experiment_config(monkeypatch) -> None:
    dataset = ROOT / "datasets/example_lerobotv3"
    config = ROOT / "configs/train/example.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_forcesmolvla_sft.py", "--dataset", str(dataset), "--config", str(config), "--task-id", "test_task"],
    )

    args = parse_args()

    assert args.dataset == dataset
    assert args.config == config
    assert args.resume is None


def test_experiment_name_is_not_special_cased(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs/train/task2.json").read_text())
    config["name"] = "another_dataset_full_sft"
    config["output_dir"] = "outputs/development/another_dataset_full_sft"
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    assert _load_config(path)["name"] == "another_dataset_full_sft"


def test_training_budget_must_align_with_effective_batch(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs/train/task2.json").read_text())
    config["training"]["target_samples"] = 3
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="TRAIN_CONFIG_INVALID"):
        _load_config(path)
