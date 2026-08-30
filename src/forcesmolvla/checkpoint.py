"""Strict, source-bound base initialization and complete local Force checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from safetensors.torch import load_file
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.common.train_utils import load_training_state, save_training_state

from .configuration_forcesmolvla import SMOLVLA_CARTESIAN7D, load_force_config
from .configuration_forcesmolvla import OFFLINE_FULL_FINETUNE
from .modeling_forcesmolvla import ForceSmolVLAPolicy


COMPILED_MODEL_PREFIX = "model._orig_mod."
RUNTIME_MODEL_PREFIX = "model."
ARTIFACT_MANIFEST = "artifact_manifest.json"
TRAINABILITY_MANIFEST = "trainability_manifest.json"
RESUME_CONTRACT = "training_state/resume_contract.json"
SCALER_STATE = "training_state/scaler_state.json"
SAMPLER_STATE = "training_state/sampler_state.json"
ACCUMULATION_STATE = "training_state/accumulation_state.json"
EMBEDDED_P8_CONTRACT = "manifests/p8_checkpoint_contract.development.json"
EMBEDDED_TRAINING_CONTRACT = "manifests/training_checkpoint_contract.development.json"
DEVELOPMENT_ARTIFACT_TYPES = frozenset(
    {"forcesmolvla_p8_checkpoint", "forcesmolvla_training_checkpoint"}
)

ALLOWED_DROPPED_NORMALIZER_KEYS = frozenset(
    f"{prefix}.{robot}.{stat}"
    for prefix in ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
    for robot in (
        "so100-blue_buffer_observation_state" if prefix == "normalize_inputs" else "so100-blue_buffer_action",
        "so100-red_buffer_observation_state" if prefix == "normalize_inputs" else "so100-red_buffer_action",
        "so100_buffer_observation_state" if prefix == "normalize_inputs" else "so100_buffer_action",
    )
    for stat in ("mean", "std")
)


def validate_resume_training_stage(
    *, checkpoint_stage: str, runtime_stage: str, restore_optimizer_state: bool
) -> None:
    if checkpoint_stage != runtime_stage and restore_optimizer_state:
        raise RuntimeError(
            "TRAINING_STAGE_OPTIMIZER_STATE_INCOMPATIBLE: rebuild optimizer/scheduler"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}\0".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict{")
            for key in sorted(value, key=lambda item: str(item)):
                update(str(key))
                update(value[key])
            digest.update(b"}")
        elif isinstance(value, (list, tuple)):
            digest.update(f"{type(value).__name__}[".encode())
            for item in value:
                update(item)
            digest.update(b"]")
        else:
            digest.update(f"{type(value).__name__}:{value!r}\0".encode())

    update(optimizer.state_dict())
    return digest.hexdigest()


def parameter_trainability_manifest(policy: torch.nn.Module) -> dict:
    entries = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
            "requires_grad": parameter.requires_grad,
        }
        for name, parameter in policy.named_parameters()
    ]
    trainable = [entry["name"] for entry in entries if entry["requires_grad"]]
    frozen = [entry["name"] for entry in entries if not entry["requires_grad"]]
    return {
        "schema_version": "1.0",
        "entries": entries,
        "entries_sha256": canonical_sha256(entries),
        "trainable_names": trainable,
        "frozen_names": frozen,
        "trainable_name_sha256": canonical_sha256(trainable),
        "frozen_name_sha256": canonical_sha256(frozen),
        "total_parameters": sum(entry["numel"] for entry in entries),
        "trainable_parameters": sum(
            entry["numel"] for entry in entries if entry["requires_grad"]
        ),
        "frozen_parameters": sum(
            entry["numel"] for entry in entries if not entry["requires_grad"]
        ),
    }


def write_trainability_manifest(policy: torch.nn.Module, checkpoint_dir: Path) -> dict:
    payload = parameter_trainability_manifest(policy)
    (checkpoint_dir / TRAINABILITY_MANIFEST).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def validate_trainability_manifest(policy: torch.nn.Module, checkpoint_dir: Path) -> dict:
    path = checkpoint_dir / TRAINABILITY_MANIFEST
    try:
        expected = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("FORCE_CHECKPOINT_TRAINABILITY_MANIFEST_INVALID") from error
    actual = parameter_trainability_manifest(policy)
    if actual != expected:
        raise RuntimeError("FORCE_CHECKPOINT_TRAINABILITY_MISMATCH")
    return actual


def _checkpoint_payloads(checkpoint_dir: Path) -> dict[str, dict]:
    payloads = {}
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"FORCE_CHECKPOINT_SYMLINK_FORBIDDEN: {path}")
        if not path.is_file() or path.name == ARTIFACT_MANIFEST:
            continue
        relative = path.relative_to(checkpoint_dir).as_posix()
        payloads[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return payloads


def write_development_artifact_manifest(
    checkpoint_dir: Path,
    *,
    metadata: dict,
    artifact_type: str = "forcesmolvla_p8_checkpoint",
) -> dict:
    if (checkpoint_dir / ARTIFACT_MANIFEST).exists():
        raise FileExistsError("refusing to overwrite artifact_manifest.json")
    if artifact_type not in DEVELOPMENT_ARTIFACT_TYPES:
        raise ValueError(f"unsupported development artifact type: {artifact_type!r}")
    payload = {
        "schema_version": "1.0",
        "artifact_type": artifact_type,
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "hash_algorithm": "sha256",
        "payloads": _checkpoint_payloads(checkpoint_dir),
        "metadata": metadata,
        "detached_signature": None,
        "approval": None,
        "formal_signature_fields": "unresolved_per_configs/approval_checklist.yaml",
    }
    (checkpoint_dir / ARTIFACT_MANIFEST).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def export_development_actor_checkpoint(
    *,
    policy: torch.nn.Module,
    destination: Path,
    runtime_parent: Path,
    source_joint_checkpoint: Path,
    candidate_revision_id: str,
    parent_binding_id: str,
    published: bool,
) -> dict[str, Any]:
    """Export Actor weights in the same strict package used by production loading."""

    destination = Path(destination)
    runtime_parent = Path(runtime_parent).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Actor export: {destination}")
    parent_contract = json.loads(
        (runtime_parent / EMBEDDED_TRAINING_CONTRACT).read_text(encoding="utf-8")
    )
    required = [str(value) for value in parent_contract["required_payloads"]]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.new.", dir=destination.parent)
    )
    try:
        policy.save_pretrained(temporary)
        shutil.copy2(runtime_parent / "config.json", temporary / "config.json")
        for relative in required:
            if relative in {
                "config.json",
                "model.safetensors",
                EMBEDDED_TRAINING_CONTRACT,
            }:
                continue
            source = runtime_parent / relative
            target = temporary / relative
            if source.is_dir():
                shutil.copytree(source, target)
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            else:
                raise FileNotFoundError(f"runtime parent payload missing: {source}")

        model_revision = sha256_file(temporary / "model.safetensors")
        candidate = {
            "revision_id": candidate_revision_id,
            "model_revision": model_revision,
            "state": "published" if published else "candidate",
            "published": published,
            "activated": False,
            "source_joint_checkpoint": str(Path(source_joint_checkpoint).resolve()),
            "parent_binding_id": parent_binding_id,
        }
        (temporary / "candidate.json").write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        contract = {
            "schema_version": parent_contract["schema_version"],
            "acceptance_status": "development_only",
            "formal_eligible": False,
            "artifact_type": "forcesmolvla_training_checkpoint",
            "training_stage": parent_contract["training_stage"],
            "strict_loader_container_only": True,
            "artifact_purpose": "stage3_development_candidate_actor",
            "deployment_release": published,
            "training_parent_allowed": False,
            "online_update_allowed": False,
            "robot_execution_authorized": False,
            "critic_exported": False,
            "optimizer_exported": False,
            "scheduler_exported": False,
            "rng_exported": False,
            "sampler_exported": False,
            "required_payloads": [*required, "candidate.json"],
            "runtime_parent": str(runtime_parent),
            "weight_source": str(
                Path(source_joint_checkpoint).resolve()
                / "candidate_policy/model.safetensors"
            ),
            "candidate_revision_id": candidate_revision_id,
            "model_revision": model_revision,
            "parent_binding": {
                "binding_id": parent_binding_id,
                "runtime_parent": str(runtime_parent),
            },
        }
        contract_path = temporary / EMBEDDED_TRAINING_CONTRACT
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = write_development_artifact_manifest(
            temporary,
            artifact_type="forcesmolvla_training_checkpoint",
            metadata={
                "artifact_purpose": "stage3_development_candidate_actor",
                "candidate_revision_id": candidate_revision_id,
                "model_revision": model_revision,
                "published": published,
                "activated": False,
                "source_joint_checkpoint": str(Path(source_joint_checkpoint).resolve()),
                "parent_binding_id": parent_binding_id,
                "strict_load": {"missing_keys": 0, "unexpected_keys": 0},
            },
        )
        validate_force_artifact_manifest(temporary, artifact_use="development")
        validate_training_payload_contract(temporary)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_force_artifact_manifest(
    checkpoint_dir: Path, *, artifact_use: str
) -> dict:
    try:
        manifest = json.loads(
            (checkpoint_dir / ARTIFACT_MANIFEST).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("FORCE_CHECKPOINT_ARTIFACT_MANIFEST_INVALID") from error
    if manifest.get("artifact_type") not in DEVELOPMENT_ARTIFACT_TYPES:
        raise RuntimeError("FORCE_CHECKPOINT_ARTIFACT_TYPE_MISMATCH")
    if artifact_use == "formal":
        if (
            manifest.get("acceptance_status") != "approved"
            or manifest.get("formal_eligible") is not True
            or not manifest.get("detached_signature")
            or not manifest.get("approval")
        ):
            raise RuntimeError("FORMAL_FORCE_CHECKPOINT_SIGNATURE_OR_APPROVAL_MISSING")
        raise RuntimeError("FORMAL_SIGNATURE_VERIFIER_NOT_CONFIGURED")
    if artifact_use != "development":
        raise ValueError("artifact_use must be 'development' or 'formal'")
    if (
        manifest.get("acceptance_status") != "development_only"
        or manifest.get("formal_eligible") is not False
        or manifest.get("detached_signature") is not None
        or manifest.get("approval") is not None
    ):
        raise RuntimeError("DEVELOPMENT_FORCE_CHECKPOINT_STATUS_MISMATCH")
    expected = manifest.get("payloads")
    if not isinstance(expected, dict) or expected != _checkpoint_payloads(checkpoint_dir):
        raise RuntimeError("FORCE_CHECKPOINT_PAYLOAD_HASH_OR_FILESET_MISMATCH")
    return manifest


def validate_sft_payload_contract(checkpoint_dir: Path) -> dict:
    """Validate the frozen P8 payload list embedded in a complete Force checkpoint."""
    try:
        contract = json.loads(
            (checkpoint_dir / EMBEDDED_P8_CONTRACT).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("FORCE_CHECKPOINT_P8_CONTRACT_INVALID") from error
    if (
        contract.get("acceptance_status") != "development_only"
        or contract.get("formal_eligible") is not False
        or contract.get("artifact_type") != "forcesmolvla_p8_checkpoint"
        or contract.get("training_stage") != OFFLINE_FULL_FINETUNE
    ):
        raise RuntimeError("FORCE_CHECKPOINT_P8_CONTRACT_STATUS_MISMATCH")
    required = contract.get("required_payloads")
    if not isinstance(required, list) or not required:
        raise RuntimeError("FORCE_CHECKPOINT_P8_REQUIRED_PAYLOAD_LIST_INVALID")
    required = [str(value) for value in required]
    if len(required) != len(set(required)):
        raise RuntimeError("FORCE_CHECKPOINT_P8_REQUIRED_PAYLOAD_LIST_DUPLICATE")
    extra_required = (
        EMBEDDED_P8_CONTRACT,
        "manifests/p8_resolved_config.json",
        "manifests/environment_manifest.json",
        "environment/conda-explicit.txt",
        "environment/conda-from-history.yml",
        "environment/pip-freeze.txt",
        "environment/requirements.lock",
        "parity_reference.json",
    )
    missing = [
        relative
        for relative in (*required, *extra_required)
        if not (checkpoint_dir / relative).exists()
    ]
    if missing:
        raise RuntimeError(f"FORCE_CHECKPOINT_REQUIRED_PAYLOAD_MISSING: {missing}")
    constructor = checkpoint_dir / "base_assets/smolvlm_constructor"
    constructor_files = [path for path in constructor.rglob("*") if path.is_file()]
    if not constructor_files:
        raise RuntimeError("FORCE_CHECKPOINT_EMBEDDED_BASE_ASSETS_EMPTY")
    return contract


def validate_training_payload_contract(checkpoint_dir: Path) -> dict:
    """Validate a complete development full-finetuning checkpoint payload."""
    try:
        contract = json.loads(
            (checkpoint_dir / EMBEDDED_TRAINING_CONTRACT).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("FORCE_TRAINING_CHECKPOINT_CONTRACT_INVALID") from error
    if (
        contract.get("acceptance_status") != "development_only"
        or contract.get("formal_eligible") is not False
        or contract.get("artifact_type") != "forcesmolvla_training_checkpoint"
        or contract.get("training_stage") != OFFLINE_FULL_FINETUNE
    ):
        raise RuntimeError("FORCE_TRAINING_CHECKPOINT_CONTRACT_STATUS_MISMATCH")
    required = contract.get("required_payloads")
    if not isinstance(required, list) or not required:
        raise RuntimeError("FORCE_TRAINING_CHECKPOINT_REQUIRED_PAYLOAD_LIST_INVALID")
    required = [str(value) for value in required]
    if len(required) != len(set(required)):
        raise RuntimeError("FORCE_TRAINING_CHECKPOINT_REQUIRED_PAYLOAD_LIST_DUPLICATE")
    missing = [relative for relative in required if not (checkpoint_dir / relative).exists()]
    if missing:
        raise RuntimeError(f"FORCE_CHECKPOINT_REQUIRED_PAYLOAD_MISSING: {missing}")
    constructor = checkpoint_dir / "base_assets/smolvlm_constructor"
    if not any(path.is_file() for path in constructor.rglob("*")):
        raise RuntimeError("FORCE_CHECKPOINT_EMBEDDED_BASE_ASSETS_EMPTY")
    return contract


def resolve_local_force_checkpoint_dir(
    pretrained_name_or_path: str | Path,
    *,
    force_download: bool,
    local_files_only: bool,
    strict: bool,
    revision: str | None,
    config,
) -> Path:
    if force_download or not local_files_only or not strict or revision is not None or config is not None:
        raise RuntimeError("FORCE_CHECKPOINT_LOCAL_STRICT_ARGUMENTS_REQUIRED")
    path = Path(pretrained_name_or_path).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("FORCE_CHECKPOINT_LOCAL_DIRECTORY_REQUIRED")
    return path.resolve()


def prepare_strict_force_config(checkpoint_dir: Path, *, artifact_use: str):
    manifest = validate_force_artifact_manifest(checkpoint_dir, artifact_use=artifact_use)
    if manifest["artifact_type"] == "forcesmolvla_p8_checkpoint":
        validate_sft_payload_contract(checkpoint_dir)
    elif manifest["artifact_type"] == "forcesmolvla_training_checkpoint":
        validate_training_payload_contract(checkpoint_dir)
    else:  # guarded above
        raise RuntimeError("FORCE_CHECKPOINT_ARTIFACT_TYPE_MISMATCH")
    try:
        raw = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("FORCE_CHECKPOINT_CONFIG_INVALID") from error
    if raw.get("type") != "force_smolvla":
        raise RuntimeError("FORCE_CHECKPOINT_BARE_OR_WRONG_CONFIG_TYPE")
    constructor = checkpoint_dir / "base_assets/smolvlm_constructor"
    if not constructor.is_dir():
        raise RuntimeError("FORCE_CHECKPOINT_EMBEDDED_BASE_ASSETS_MISSING")
    config = PreTrainedConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    from .configuration_forcesmolvla import ForceSmolVLAConfig

    if not isinstance(config, ForceSmolVLAConfig):
        raise RuntimeError("FORCE_CHECKPOINT_CONFIG_CLASS_MISMATCH")
    config.vlm_model_name = str(constructor.resolve())
    config.load_vlm_weights = False
    return config


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def save_sft_training_state(
    checkpoint_dir: Path,
    *,
    step: int,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    sampler,
    accumulation_phase: int,
    batch_size: int,
    gradient_accumulation_microbatches: int,
    resume_contract: dict,
) -> None:
    if accumulation_phase != 0:
        raise ValueError("P8 checkpoint must be saved on an optimizer-update boundary")
    if batch_size <= 0 or gradient_accumulation_microbatches <= 0:
        raise ValueError("checkpoint batching values must be positive")
    save_training_state(
        checkpoint_dir,
        step,
        optimizer,
        scheduler,
        num_processes=1,
        batch_size=batch_size,
    )
    training_state = checkpoint_dir / "training_state"
    scaler_payload = {"enabled": scaler.is_enabled(), "state": scaler.state_dict()}
    (checkpoint_dir / SCALER_STATE).write_text(
        json.dumps(_jsonable(scaler_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / SAMPLER_STATE).write_text(
        json.dumps(_jsonable(sampler.state_dict()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    accumulation = {
        "microbatch_index_within_window": accumulation_phase,
        "batch_size": batch_size,
        "gradient_accumulation_microbatches": gradient_accumulation_microbatches,
        "optimizer_update": step,
    }
    (checkpoint_dir / ACCUMULATION_STATE).write_text(
        json.dumps(accumulation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = {
        **resume_contract,
        "training_stage": policy.config.training_stage,
        "optimizer_state_sha256": optimizer_state_sha256(optimizer),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler_payload,
        "sampler_cursor": sampler.cursor,
        "accumulation_state": accumulation,
    }
    (checkpoint_dir / RESUME_CONTRACT).write_text(
        json.dumps(_jsonable(contract), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not training_state.is_dir():
        raise RuntimeError("P8_TRAINING_STATE_DIRECTORY_MISSING")


def load_sft_training_state(
    checkpoint_dir: Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    sampler,
    batch_size: int,
    gradient_accumulation_microbatches: int,
    expected_resume_contract: dict,
) -> tuple[int, dict]:
    contract = json.loads((checkpoint_dir / RESUME_CONTRACT).read_text(encoding="utf-8"))
    validate_resume_training_stage(
        checkpoint_stage=contract["training_stage"],
        runtime_stage=policy.config.training_stage,
        restore_optimizer_state=True,
    )
    for key, expected in expected_resume_contract.items():
        if contract.get(key) != _jsonable(expected):
            raise RuntimeError(f"P8_RESUME_CONTRACT_MISMATCH:{key}")
    accumulation = json.loads(
        (checkpoint_dir / ACCUMULATION_STATE).read_text(encoding="utf-8")
    )
    if (
        accumulation.get("microbatch_index_within_window") != 0
        or accumulation.get("batch_size") != batch_size
        or accumulation.get("gradient_accumulation_microbatches")
        != gradient_accumulation_microbatches
        or accumulation.get("optimizer_update") != contract.get("accumulation_state", {}).get(
            "optimizer_update"
        )
    ):
        raise RuntimeError("P8_ACCUMULATION_STATE_INVALID")
    step, optimizer, scheduler = load_training_state(
        checkpoint_dir, optimizer, scheduler, load_optimizer=True
    )
    scaler_payload = json.loads((checkpoint_dir / SCALER_STATE).read_text(encoding="utf-8"))
    if bool(scaler_payload["enabled"]) != scaler.is_enabled():
        raise RuntimeError("P8_SCALER_ENABLED_STATE_MISMATCH")
    scaler.load_state_dict(scaler_payload["state"])
    sampler_payload = json.loads((checkpoint_dir / SAMPLER_STATE).read_text(encoding="utf-8"))
    sampler_payload["eligible_indices"] = tuple(sampler_payload["eligible_indices"])
    sampler_payload["rng_state"] = _tuple_tree(sampler_payload["rng_state"])
    sampler.load_state_dict(sampler_payload)
    if accumulation.get("optimizer_update") != step:
        raise RuntimeError("P8_ACCUMULATION_STATE_INVALID")
    if optimizer_state_sha256(optimizer) != contract["optimizer_state_sha256"]:
        raise RuntimeError("P8_OPTIMIZER_STATE_HASH_MISMATCH")
    if scheduler is not None and scheduler.state_dict() != contract["scheduler_state"]:
        raise RuntimeError("P8_SCHEDULER_STATE_MISMATCH")
    if sampler.cursor != contract["sampler_cursor"]:
        raise RuntimeError("P8_SAMPLER_CURSOR_MISMATCH")
    return step, contract


@dataclass(frozen=True)
class BaseLoadReport:
    checkpoint: str
    source_tensor_count: int
    loaded_tensor_count: int
    dropped_tensor_count: int
    dropped_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    key_transform: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_frozen_base_state_dict(source: dict) -> tuple[dict, tuple[str, ...]]:
    """Apply the only two compatibility transforms allowed by the frozen binding."""
    source_keys = set(source)
    dropped = source_keys & ALLOWED_DROPPED_NORMALIZER_KEYS
    unknown_non_model = source_keys - dropped - {
        key for key in source_keys if key.startswith(COMPILED_MODEL_PREFIX)
    }
    missing_allowlisted = ALLOWED_DROPPED_NORMALIZER_KEYS - source_keys
    if unknown_non_model:
        raise RuntimeError(f"BASE_CHECKPOINT_UNKNOWN_KEYS: {sorted(unknown_non_model)}")
    if missing_allowlisted:
        raise RuntimeError(f"BASE_CHECKPOINT_ALLOWLIST_DRIFT: {sorted(missing_allowlisted)}")

    normalized = {}
    for key, tensor in source.items():
        if key in dropped:
            continue
        new_key = RUNTIME_MODEL_PREFIX + key[len(COMPILED_MODEL_PREFIX) :]
        if new_key in normalized:
            raise RuntimeError(f"BASE_CHECKPOINT_KEY_COLLISION: {new_key}")
        normalized[new_key] = tensor
    return normalized, tuple(sorted(dropped))


def load_frozen_base_weights_strict(
    policy: ForceSmolVLAPolicy, checkpoint_file: Path
) -> BaseLoadReport:
    source = load_file(str(checkpoint_file), device="cpu")
    normalized, dropped = normalize_frozen_base_state_dict(source)
    expected = set(policy.state_dict())
    provided = set(normalized)
    missing = tuple(sorted(expected - provided))
    unexpected = tuple(sorted(provided - expected))
    allowed_missing = policy.force_initialization_state_keys()
    if missing != allowed_missing or unexpected:
        raise RuntimeError(
            "BASE_CHECKPOINT_INCOMPATIBLE: "
            f"missing={missing}, allowed_missing={allowed_missing}, unexpected={unexpected}"
        )
    incompatible = policy.load_state_dict(normalized, strict=False)
    if tuple(sorted(incompatible.missing_keys)) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "BASE_CHECKPOINT_LOAD_RESULT_DRIFT: "
            f"missing={sorted(incompatible.missing_keys)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return BaseLoadReport(
        checkpoint=str(checkpoint_file.resolve()),
        source_tensor_count=len(source),
        loaded_tensor_count=len(normalized),
        dropped_tensor_count=len(dropped),
        dropped_keys=dropped,
        missing_keys=missing,
        unexpected_keys=unexpected,
        key_transform="replace leading model._orig_mod. with model. exactly once",
    )


def load_offline_base_policy(
    base_checkpoint: Path,
    constructor_assets: Path,
    *,
    device: str = "cpu",
    training_stage: str = OFFLINE_FULL_FINETUNE,
    force_variant: str = SMOLVLA_CARTESIAN7D,
    acceptance_status: str = "development_only",
    force_init_seed: int = 42,
) -> tuple[ForceSmolVLAPolicy, BaseLoadReport]:
    config = load_force_config(
        base_checkpoint,
        constructor_assets,
        device=device,
        training_stage=training_stage,
        force_variant=force_variant,
        acceptance_status=acceptance_status,
        force_init_seed=force_init_seed,
    )
    policy = ForceSmolVLAPolicy(config)
    report = load_frozen_base_weights_strict(policy, base_checkpoint / "model.safetensors")
    policy.to(device)
    policy.eval()
    return policy, report
