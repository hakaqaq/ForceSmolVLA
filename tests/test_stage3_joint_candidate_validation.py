from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from forcesmolvla.checkpoint import export_development_actor_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_stage3_joint_candidate",
    ROOT / "tools/validate_stage3_joint_candidate.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_saved_td_trace_requires_all_twenty_steps() -> None:
    assert MODULE.extract_saved_td_losses({"metrics": {"td_losses": [1.0, 2.0]}}) is None
    result = MODULE.summarize_td_losses(None)
    assert result == {
        "TD_LOSSES": [],
        "TD_FIRST_5_MEDIAN": None,
        "TD_LAST_5_MEDIAN": None,
        "TD_TREND": "UNAVAILABLE:SAVED_20_STEP_TD_TRACE_MISSING",
    }


def test_saved_td_trace_reports_full_series_and_trend() -> None:
    values = [float(value) for value in range(20)]
    restored = MODULE.extract_saved_td_losses(
        {"step_metrics": {"critic_td_loss": values}}
    )
    summary = MODULE.summarize_td_losses(restored)
    assert summary["TD_LOSSES"] == values
    assert summary["TD_FIRST_5_MEDIAN"] == 2.0
    assert summary["TD_LAST_5_MEDIAN"] == 17.0
    assert summary["TD_TREND"]["continuous_growth"] is True
    assert summary["TD_TREND"]["longest_consecutive_growth"] == 19


def test_standard_actor_export_contains_strict_runtime_package(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([3.0]))

        def save_pretrained(self, destination: Path) -> None:
            destination.mkdir(parents=True, exist_ok=True)
            save_file({"weight": self.weight.detach()}, destination / "model.safetensors")
            (destination / "config.json").write_text("{}\n", encoding="utf-8")

    parent = tmp_path / "parent"
    payloads = [
        "config.json",
        "model.safetensors",
        "base_assets/smolvlm_constructor",
        "trainability_manifest.json",
        "manifests/action_delta_spec.json",
        "manifests/normalizer_manifest.json",
        "manifests/stage2_action_contract.v2.development.json",
        "manifests/calibration_bundle.development.json",
        "manifests/converter_runtime_spec.task2.development.json",
        "manifests/training_checkpoint_contract.development.json",
    ]
    for relative in payloads:
        target = parent / relative
        if relative == "base_assets/smolvlm_constructor":
            target.mkdir(parents=True)
            (target / "config.json").write_text("{}\n", encoding="utf-8")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}\n", encoding="utf-8")
    contract = {
        "schema_version": "forcesmolvla_evaluation_checkpoint_contract.v1",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "artifact_type": "forcesmolvla_training_checkpoint",
        "training_stage": "offline_full_finetune",
        "required_payloads": payloads,
    }
    (parent / "manifests/training_checkpoint_contract.development.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )

    destination = tmp_path / "export"
    manifest = export_development_actor_checkpoint(
        policy=TinyPolicy(),
        destination=destination,
        runtime_parent=parent,
        source_joint_checkpoint=tmp_path / "joint",
        candidate_revision_id="candidate-test",
        parent_binding_id="parent-test",
        published=True,
    )

    candidate = json.loads((destination / "candidate.json").read_text())
    assert candidate["published"] is True
    assert candidate["activated"] is False
    assert "candidate.json" in manifest["payloads"]
    assert (destination / "manifests/normalizer_manifest.json").is_file()
