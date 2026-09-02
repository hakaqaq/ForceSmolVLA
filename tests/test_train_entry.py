import copy
import json
from pathlib import Path
import sys

from train_forcesmolvla_sft import _bind_fixture_provenance, _load_config, parse_args


ROOT = Path(__file__).parents[1]


def test_training_entry_is_dataset_agnostic() -> None:
    source = (ROOT / "tools/train_forcesmolvla_sft.py").read_text(encoding="utf-8").lower()
    assert "task2" not in source
    assert not (ROOT / "tools/train_task2_full_gpu.py").exists()


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


def test_config_can_reuse_architecture_gate_with_dataset_specific_parity(
    tmp_path: Path,
) -> None:
    config = json.loads((ROOT / "configs/train/task2.json").read_text())
    config["action_target_population_parity"] = (
        "artifacts/development/action_target_population_parity_task3.json"
    )
    config["training_readiness"]["reuse_validated_architecture_gate"] = True
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = _load_config(path)

    assert loaded["action_target_population_parity"].endswith("task3.json")
    assert loaded["training_readiness"]["reuse_validated_architecture_gate"] is True


def test_generic_fixture_without_legacy_session_binding_is_unchanged() -> None:
    fixture = {"chunk_context": {"session_id": ["session-a"]}}
    original = copy.deepcopy(fixture)

    _bind_fixture_provenance(fixture, {"session_provenance": {}})

    assert fixture == original
