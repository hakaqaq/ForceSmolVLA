import json
from pathlib import Path

import pytest
import torch

from forcesmolvla.checkpoint import (
    validate_force_artifact_manifest,
    validate_training_payload_contract,
    write_development_artifact_manifest,
)
from tools.export_stage2b_cycle210_evaluation_smoke import (
    tensor_record,
    tensor_state_record,
    validate_evaluation_export_scope,
)


def _evaluation_checkpoint(root: Path) -> None:
    required = [
        "config.json",
        "model.safetensors",
        "base_assets/smolvlm_constructor",
        "trainability_manifest.json",
    ]
    (root / "base_assets/smolvlm_constructor").mkdir(parents=True)
    for relative in required:
        path = root / relative
        if relative.endswith("smolvlm_constructor"):
            path.joinpath("config.json").write_text("{}", encoding="utf-8")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
    contract = {
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "artifact_type": "forcesmolvla_training_checkpoint",
        "training_stage": "offline_full_finetune",
        "artifact_purpose": "evaluation_smoke_only",
        "deployment_release": False,
        "training_parent_allowed": False,
        "online_update_allowed": False,
        "robot_execution_authorized": "false_pending_offline_parity",
        "strict_loader_container_only": True,
        "required_payloads": required,
    }
    path = root / "manifests/training_checkpoint_contract.development.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(contract), encoding="utf-8")


def test_evaluation_checkpoint_is_strict_and_training_state_free(tmp_path):
    _evaluation_checkpoint(tmp_path)
    validate_training_payload_contract(tmp_path)
    validate_evaluation_export_scope(tmp_path)
    write_development_artifact_manifest(
        tmp_path,
        artifact_type="forcesmolvla_training_checkpoint",
        metadata={"artifact_purpose": "evaluation_smoke_only"},
    )
    validate_force_artifact_manifest(tmp_path, artifact_use="development")

    forbidden = tmp_path / "optimizers/actor.pt"
    forbidden.parent.mkdir()
    forbidden.write_bytes(b"forbidden")
    with pytest.raises(RuntimeError, match="TRAINING_PAYLOAD_PRESENT"):
        validate_evaluation_export_scope(tmp_path)
    with pytest.raises(RuntimeError, match="PAYLOAD_HASH_OR_FILESET"):
        validate_force_artifact_manifest(tmp_path, artifact_use="development")


def test_evaluation_contract_fails_closed_on_scope_drift(tmp_path):
    _evaluation_checkpoint(tmp_path)
    path = tmp_path / "manifests/training_checkpoint_contract.development.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["training_parent_allowed"] = True
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SCOPE_MISMATCH"):
        validate_evaluation_export_scope(tmp_path)


def test_tensor_digest_accepts_scalar_state_entries():
    state = {"scalar": torch.tensor(1.0), "vector": torch.tensor([2.0])}
    assert tensor_state_record(state)["tensor_count"] == 2
    assert tensor_record(state["scalar"])["shape"] == []
