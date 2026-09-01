import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from forcesmolvla.checkpoint import (
    export_development_actor_checkpoint,
    load_sft_training_state,
    optimizer_state_sha256,
    parameter_trainability_manifest,
    resolve_local_force_checkpoint_dir,
    save_sft_training_state,
    validate_force_artifact_manifest,
    validate_training_payload_contract,
    write_development_artifact_manifest,
)
from forcesmolvla.configuration_forcesmolvla import OFFLINE_FULL_FINETUNE
from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
from forcesmolvla.router_training import SerializableUniformSampler


def _artifact(tmp_path: Path):
    (tmp_path / "payload.txt").write_text("bound\n", encoding="utf-8")
    return write_development_artifact_manifest(
        tmp_path,
        metadata={"training_stage": OFFLINE_FULL_FINETUNE},
    )


def test_development_manifest_hashes_exact_files_and_tamper_fails(tmp_path):
    manifest = _artifact(tmp_path)
    assert manifest["acceptance_status"] == "development_only"
    validate_force_artifact_manifest(tmp_path, artifact_use="development")
    (tmp_path / "payload.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PAYLOAD_HASH_OR_FILESET"):
        validate_force_artifact_manifest(tmp_path, artifact_use="development")


def test_development_manifest_accepts_training_checkpoint_type(tmp_path):
    (tmp_path / "payload.txt").write_text("training\n", encoding="utf-8")
    manifest = write_development_artifact_manifest(
        tmp_path,
        metadata={"optimizer_update": 1},
        artifact_type="forcesmolvla_training_checkpoint",
    )
    assert manifest["artifact_type"] == "forcesmolvla_training_checkpoint"
    validate_force_artifact_manifest(tmp_path, artifact_use="development")


def test_training_payload_contract_requires_bound_files_and_constructor(tmp_path):
    contract = {
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "artifact_type": "forcesmolvla_training_checkpoint",
        "training_stage": OFFLINE_FULL_FINETUNE,
        "required_payloads": ["model.safetensors", "base_assets/smolvlm_constructor"],
    }
    contract_path = tmp_path / "manifests/training_checkpoint_contract.development.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"model")
    constructor = tmp_path / "base_assets/smolvlm_constructor"
    constructor.mkdir(parents=True)
    (constructor / "config.json").write_text("{}", encoding="utf-8")
    assert validate_training_payload_contract(tmp_path) == contract
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="REQUIRED_PAYLOAD_MISSING"):
        validate_training_payload_contract(tmp_path)


def test_actor_checkpoint_export_does_not_duplicate_candidate_payload(tmp_path):
    parent = tmp_path / "parent"
    constructor = parent / "base_assets/smolvlm_constructor"
    constructor.mkdir(parents=True)
    (constructor / "config.json").write_text("{}", encoding="utf-8")
    (parent / "config.json").write_text("{}", encoding="utf-8")
    (parent / "model.safetensors").write_bytes(b"parent")
    (parent / "candidate.json").write_text("{}", encoding="utf-8")
    contract_path = parent / "manifests/training_checkpoint_contract.development.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps({
        "schema_version": "1.0",
        "training_stage": OFFLINE_FULL_FINETUNE,
        "required_payloads": [
            "config.json",
            "model.safetensors",
            "base_assets/smolvlm_constructor",
            "manifests/training_checkpoint_contract.development.json",
            "candidate.json",
        ],
    }), encoding="utf-8")

    class Policy:
        @staticmethod
        def save_pretrained(path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.safetensors").write_bytes(b"updated")
            (path / "config.json").write_text("{}", encoding="utf-8")

    destination = tmp_path / "export"
    export_development_actor_checkpoint(
        policy=Policy(),
        destination=destination,
        runtime_parent=parent,
        source_joint_checkpoint=tmp_path / "online-checkpoint",
        candidate_revision_id="online-actor-critic-cycle-000001",
        parent_binding_id="task2-offline-exact-resume",
        published=False,
    )

    exported = json.loads(
        (destination / "manifests/training_checkpoint_contract.development.json").read_text()
    )
    assert exported["required_payloads"].count("candidate.json") == 1
    validate_training_payload_contract(destination)


def test_manifest_rejects_extra_file_and_formal_use(tmp_path):
    _artifact(tmp_path)
    with pytest.raises(RuntimeError, match="SIGNATURE_OR_APPROVAL_MISSING"):
        validate_force_artifact_manifest(tmp_path, artifact_use="formal")
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PAYLOAD_HASH_OR_FILESET"):
        validate_force_artifact_manifest(tmp_path, artifact_use="development")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"force_download": True, "local_files_only": True, "strict": True},
        {"force_download": False, "local_files_only": False, "strict": True},
        {"force_download": False, "local_files_only": True, "strict": False},
    ],
)
def test_force_checkpoint_requires_local_strict_arguments(tmp_path, kwargs):
    with pytest.raises(RuntimeError, match="LOCAL_STRICT_ARGUMENTS"):
        resolve_local_force_checkpoint_dir(
            tmp_path,
            revision=None,
            config=None,
            **kwargs,
        )


def test_force_policy_from_pretrained_rejects_remote_identifier_before_hub_call():
    with pytest.raises(RuntimeError, match="LOCAL_DIRECTORY"):
        ForceSmolVLAPolicy.from_pretrained("owner/remote-repo")


def test_trainability_manifest_has_exact_names_shapes_and_hash():
    module = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
    module[1].bias.requires_grad_(False)
    payload = parameter_trainability_manifest(module)
    assert payload["total_parameters"] == 12
    assert payload["trainable_parameters"] == 10
    assert payload["frozen_parameters"] == 2
    assert payload["frozen_names"] == ["1.bias"]
    assert len(payload["entries_sha256"]) == 64


def test_optimizer_state_hash_accepts_bfloat16_tensor_bytes():
    optimizer_like = SimpleNamespace(
        state_dict=lambda: {"state": {0: {"exp_avg": torch.ones(3, dtype=torch.bfloat16)}}}
    )
    assert len(optimizer_state_sha256(optimizer_like)) == 64


class _Policy(torch.nn.Linear):
    def __init__(self, stage=OFFLINE_FULL_FINETUNE):
        super().__init__(2, 2)
        self.config = SimpleNamespace(training_stage=stage)


def test_resume_restores_optimizer_scheduler_rng_sampler_scaler_and_phase(tmp_path):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    policy = _Policy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    sampler = SerializableUniformSampler([1, 3, 5], seed=42)
    sampler.draw(4)
    policy(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    expected_optimizer_hash = optimizer_state_sha256(optimizer)
    save_sft_training_state(
        tmp_path,
        step=1,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        accumulation_phase=0,
        batch_size=4,
        gradient_accumulation_microbatches=1,
        resume_contract={
            "acceptance_status": "development_only",
            "prefetched_sample_indices": [5, 3, 1],
        },
    )
    expected_rng = (random.random(), float(np.random.rand()), torch.rand(2))
    expected_sampler = sampler.draw(6)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restored_policy = _Policy()
    restored_optimizer = torch.optim.AdamW(restored_policy.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda _step: 1.0
    )
    restored_scaler = torch.amp.GradScaler("cpu", enabled=False)
    restored_sampler = SerializableUniformSampler([1, 3, 5], seed=42)
    step, contract = load_sft_training_state(
        tmp_path,
        policy=restored_policy,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        sampler=restored_sampler,
        batch_size=4,
        gradient_accumulation_microbatches=1,
        expected_resume_contract={
            "acceptance_status": "development_only",
            "prefetched_sample_indices": [5, 3, 1],
        },
    )
    assert step == 1
    assert contract["accumulation_state"]["microbatch_index_within_window"] == 0
    assert contract["prefetched_sample_indices"] == [5, 3, 1]
    assert optimizer_state_sha256(restored_optimizer) == expected_optimizer_hash
    assert random.random() == expected_rng[0]
    assert float(np.random.rand()) == expected_rng[1]
    assert torch.equal(torch.rand(2), expected_rng[2])
    assert restored_sampler.draw(6) == expected_sampler


def test_resume_rejects_cross_training_stage_before_state_restore(tmp_path):
    policy = _Policy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    sampler = SerializableUniformSampler([1], seed=42)
    policy(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    save_sft_training_state(
        tmp_path,
        step=1,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        accumulation_phase=0,
        batch_size=4,
        gradient_accumulation_microbatches=1,
        resume_contract={"acceptance_status": "development_only"},
    )
    other = _Policy(stage="online_hil_vlm_frozen")
    with pytest.raises(RuntimeError, match="TRAINING_STAGE"):
        load_sft_training_state(
            tmp_path,
            policy=other,
            optimizer=torch.optim.AdamW(other.parameters(), lr=1e-3),
            scheduler=None,
            scaler=scaler,
            sampler=SerializableUniformSampler([1], seed=42),
            batch_size=4,
            gradient_accumulation_microbatches=1,
            expected_resume_contract={"acceptance_status": "development_only"},
        )


@pytest.mark.parametrize(
    ("batch_size", "microbatches", "expected_error"),
    [
        (2, 1, "P8_ACCUMULATION_STATE_INVALID"),
        (4, 8, "P8_ACCUMULATION_STATE_INVALID"),
    ],
)
def test_resume_rejects_batching_drift_before_optimizer_restore(
    tmp_path, batch_size, microbatches, expected_error
):
    policy = _Policy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    sampler = SerializableUniformSampler([1], seed=42)
    policy(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    save_sft_training_state(
        tmp_path,
        step=1,
        policy=policy,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        sampler=sampler,
        accumulation_phase=0,
        batch_size=4,
        gradient_accumulation_microbatches=1,
        resume_contract={"source_binding_sha256": "a" * 64},
    )
    restored = _Policy()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    before = optimizer_state_sha256(restored_optimizer)
    with pytest.raises(RuntimeError, match=expected_error):
        load_sft_training_state(
            tmp_path,
            policy=restored,
            optimizer=restored_optimizer,
            scheduler=None,
            scaler=scaler,
            sampler=SerializableUniformSampler([1], seed=42),
            batch_size=batch_size,
            gradient_accumulation_microbatches=microbatches,
            expected_resume_contract={"source_binding_sha256": "a" * 64},
        )
    assert optimizer_state_sha256(restored_optimizer) == before


def test_resume_rejects_source_binding_drift_before_optimizer_restore(tmp_path):
    policy = _Policy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    sampler = SerializableUniformSampler([1], seed=42)
    policy(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    save_sft_training_state(
        tmp_path,
        step=1,
        policy=policy,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        sampler=sampler,
        accumulation_phase=0,
        batch_size=4,
        gradient_accumulation_microbatches=1,
        resume_contract={"source_binding_sha256": "a" * 64},
    )
    restored = _Policy()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    before = optimizer_state_sha256(restored_optimizer)
    with pytest.raises(RuntimeError, match="P8_RESUME_CONTRACT_MISMATCH"):
        load_sft_training_state(
            tmp_path,
            policy=restored,
            optimizer=restored_optimizer,
            scheduler=None,
            scaler=scaler,
            sampler=SerializableUniformSampler([1], seed=42),
            batch_size=4,
            gradient_accumulation_microbatches=1,
            expected_resume_contract={"source_binding_sha256": "b" * 64},
        )
    assert optimizer_state_sha256(restored_optimizer) == before
