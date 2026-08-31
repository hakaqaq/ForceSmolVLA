"""Online-replay metadata validation and isolated exact-resume checkpoints."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from forcesmolvla.rft.canonical_state import (
    canonical_digest,
    canonicalize,
    module_record,
    optimizer_parameter_name_groups,
    optimizer_record,
    tensor_record,
)
from forcesmolvla.rft.training_cycle import ensure_all_gradients_none


ROOT = Path(__file__).parents[4]
SCHEMA_PATH = ROOT / "schemas/stage3_online_checkpoint.v1.schema.json"
EXACT_RESUME_SCHEMA_VERSION = "forcesmolvla_stage3_exact_resume_checkpoint.v1"
EXACT_RESUME_COMPLETION = "COMPLETED.json"
EXACT_RESUME_MANIFEST = "manifest.json"
EXACT_RESUME_MARKERS = {
    "artifact_status": "DISPOSABLE_G5P_EXACT_RESUME_PREFLIGHT",
    "parent_checkpoint_status": "NOT_A_PARENT_CHECKPOINT",
    "policy_revision_status": "NOT_A_PUBLISHABLE_POLICY_REVISION",
    "production_durable_resume_validated": False,
    "robot_execution_authorized": False,
}
_MODEL_NAMES = ("actor", "q1", "q2", "q1_target", "q2_target")


class OnlineCheckpointSchemaError(ValueError):
    pass


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_online_checkpoint_metadata(payload: Mapping) -> dict:
    value = deepcopy(dict(payload))
    errors = sorted(
        Draft202012Validator(_schema()).iter_errors(value),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        path = ".".join(str(item) for item in errors[0].absolute_path)
        raise OnlineCheckpointSchemaError(
            f"ONLINE_REPLAY_CHECKPOINT_SCHEMA:{path}:{errors[0].message}"
        )
    credits = value["credits"]
    if credits["available"] != credits["minted"] - credits["consumed"]:
        raise OnlineCheckpointSchemaError("ONLINE_REPLAY_CHECKPOINT_CREDIT_LEDGER_MISMATCH")
    counters = value["counters"]
    if counters["critic_updates"] != 2 * counters["learner_cycles"]:
        raise OnlineCheckpointSchemaError("ONLINE_REPLAY_CHECKPOINT_CRITIC_COUNTER_MISMATCH")
    if counters["actor_updates"] != counters["learner_cycles"]:
        raise OnlineCheckpointSchemaError("ONLINE_REPLAY_CHECKPOINT_ACTOR_COUNTER_MISMATCH")
    if counters["polyak_updates_per_target"] != counters["critic_updates"]:
        raise OnlineCheckpointSchemaError("ONLINE_REPLAY_CHECKPOINT_POLYAK_COUNTER_MISMATCH")
    return value


def cpu_round_trip_online_checkpoint(payload: Mapping) -> tuple[dict, bytes]:
    value = validate_online_checkpoint_metadata(payload)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    decoded = json.loads(encoded)
    validate_online_checkpoint_metadata(decoded)
    if decoded != value:
        raise OnlineCheckpointSchemaError("ONLINE_REPLAY_CHECKPOINT_CPU_ROUND_TRIP_MISMATCH")
    return decoded, encoded


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, _json_bytes(value))


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True) + [root]:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _manifest_entries(root: Path) -> list[dict[str, Any]]:
    excluded = {EXACT_RESUME_MANIFEST, EXACT_RESUME_COMPLETION}
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"G5P_JSON_STATE_UNSUPPORTED:{type(value).__name__}")


def _nested_tuple(value: Any) -> Any:
    return tuple(_nested_tuple(item) for item in value) if isinstance(value, list) else value


def capture_exact_rng_state(
    generators: Mapping[str, torch.Generator],
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "json": {
            "python_random_state": _json_safe(random.getstate()),
            "numpy_random_state": {
                "bit_generator": numpy_state[0],
                "keys": numpy_state[1].tolist(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
        },
        "tensors": {
            "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
            **{
                f"torch_cuda.{index}": state.detach().cpu().clone()
                for index, state in enumerate(torch.cuda.get_rng_state_all())
            },
            **{
                f"generator.{name}": generator.get_state().detach().cpu().clone()
                for name, generator in sorted(generators.items())
            },
        },
    }


def _rng_manifest(rng_state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "json": _json_safe(rng_state["json"]),
        "tensors": {
            name: tensor_record(value)
            for name, value in sorted(rng_state["tensors"].items())
        },
    }


def restore_exact_rng_state(
    rng_state: Mapping[str, Any], generators: Mapping[str, torch.Generator]
) -> None:
    tensors = rng_state["tensors"]
    expected_named = {f"generator.{name}" for name in generators}
    actual_named = {name for name in tensors if name.startswith("generator.")}
    if actual_named != expected_named:
        raise RuntimeError("G5P_NAMED_GENERATOR_STATE_MISMATCH")
    cuda_keys = sorted(
        (name for name in tensors if name.startswith("torch_cuda.")),
        key=lambda name: int(name.rsplit(".", 1)[1]),
    )
    if len(cuda_keys) != torch.cuda.device_count():
        raise RuntimeError("G5P_CUDA_RNG_DEVICE_COUNT_MISMATCH")
    for name, generator in generators.items():
        generator.set_state(tensors[f"generator.{name}"])
    numpy_state = rng_state["json"]["numpy_random_state"]
    random.setstate(_nested_tuple(rng_state["json"]["python_random_state"]))
    np.random.set_state((
        numpy_state["bit_generator"],
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(tensors["torch_cpu"])
    if cuda_keys:
        torch.cuda.set_rng_state_all([tensors[name] for name in cuda_keys])


def actor_frozen_state_digest(actor: torch.nn.Module) -> str:
    trainable = {name for name, parameter in actor.named_parameters() if parameter.requires_grad}
    frozen = {
        name: value
        for name, value in actor.state_dict().items()
        if name not in trainable
    }
    return module_record(frozen)["digest"]


def exact_resume_parameter_name_groups(
    modules: Mapping[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
) -> dict[str, list[list[str]]]:
    actor_names = dict(modules["actor"].named_parameters())
    critic_names = {
        **{
            f"q1.{name}": parameter
            for name, parameter in modules["q1"].named_parameters()
        },
        **{
            f"q2.{name}": parameter
            for name, parameter in modules["q2"].named_parameters()
        },
    }
    return {
        "actor": optimizer_parameter_name_groups(actor_optimizer, actor_names),
        "critic": optimizer_parameter_name_groups(critic_optimizer, critic_names),
    }


def _module_modes(modules: Mapping[str, torch.nn.Module]) -> dict[str, bool]:
    return {
        f"{owner}.{name or '<root>'}": bool(child.training)
        for owner, module in sorted(modules.items())
        for name, child in module.named_modules()
    }


def _validate_boundary(boundary: Mapping[str, Any]) -> None:
    required = {
        "episode_sealed": True,
        "active_episode": False,
        "request_in_flight": False,
        "partial_macro": False,
        "learner_update_committed": True,
        "pending_gradients": False,
        "pending_optimizer_steps": 0,
        "pending_accumulation_microbatches": 0,
    }
    if any(boundary.get(name) != value for name, value in required.items()):
        raise RuntimeError("G5P_CHECKPOINT_BOUNDARY_NOT_QUIESCENT")


def _validate_control_state(state: Mapping[str, Any]) -> None:
    _validate_boundary(state["boundary"])
    counters = state["counters"]
    cycles = int(counters["learner_cycles"])
    expected = {
        "learner_cycles": cycles,
        "critic_updates": 2 * cycles,
        "actor_updates": cycles,
        "polyak_updates_per_target": 2 * cycles,
        "publication_count": 0,
    }
    if dict(counters) != expected or cycles < 1:
        raise RuntimeError("G5P_COUNTER_DRIFT")
    credits = state["credits"]
    credited = credits["credited_uids"]
    if (
        credits["minted"] != len(credited) * credits["credits_per_transition"]
        or credits["available"] != credits["minted"] - credits["consumed"]
        or credits["consumed"] != cycles * credits["credits_per_joint_cycle"]
        or credits["available"] < 0
    ):
        raise RuntimeError("G5P_CREDIT_COUNTER_DRIFT")
    replay = state["replay"]
    index = replay["canonical_index"]
    if replay["canonical_index_sha256"] != canonical_digest(index):
        raise RuntimeError("G5P_REPLAY_CANONICAL_INDEX_DRIFT")
    if replay["R_watermark"] != len(replay["R_membership_uids"]):
        raise RuntimeError("G5P_REPLAY_R_WATERMARK_DRIFT")
    if replay["D_watermark"] != len(replay["D_membership_uids"]):
        raise RuntimeError("G5P_REPLAY_D_WATERMARK_DRIFT")
    if replay.get("episode_finalization_state") != "sealed":
        raise RuntimeError("G5P_UNSEALED_EPISODE")
    revision = state["revision"]
    if revision.get("pending_revision") is not None:
        raise RuntimeError("G5P_PENDING_REVISION_RESTORE_FORBIDDEN")
    if revision.get("episode_revision") is not None:
        raise RuntimeError("G5P_ACTIVE_EPISODE_RESTORE_FORBIDDEN")
    if revision.get("publication_count") != 0:
        raise RuntimeError("G5P_PUBLICATION_STATE_DRIFT")
    durable = state["durable_state"]
    expected_unsupported = {
        "production_wal": "UNSUPPORTED_IN_ISOLATED_G5P",
        "production_outbox": "UNSUPPORTED_IN_ISOLATED_G5P",
        "production_publication": "UNSUPPORTED_IN_ISOLATED_G5P",
    }
    if durable != expected_unsupported:
        raise RuntimeError("G5P_DURABLE_STATE_MARKER_DRIFT")


def online_exact_canonical_state(
    *,
    modules: Mapping[str, torch.nn.Module | Mapping[str, torch.Tensor]],
    actor_optimizer: torch.optim.Optimizer | Mapping[str, Any],
    critic_optimizer: torch.optim.Optimizer | Mapping[str, Any],
    parameter_name_groups: Mapping[str, Sequence[Sequence[str]]],
    scheduler_states: Mapping[str, Any],
    scaler_state: Any,
    rng_state: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
    replay_state: Mapping[str, Any],
    credit_state: Mapping[str, Any],
    counters: Mapping[str, Any],
    revision_state: Mapping[str, Any],
    durable_state: Mapping[str, Any],
    boundary: Mapping[str, Any],
    bindings: Mapping[str, Any],
    module_modes: Mapping[str, bool],
    frozen_actor_digest: str,
) -> dict[str, Any]:
    sections = {
        "models": {
            name: module_record(modules[name]) for name in _MODEL_NAMES
        },
        "optimizers": {
            "actor": optimizer_record(actor_optimizer, parameter_name_groups["actor"]),
            "critic": optimizer_record(critic_optimizer, parameter_name_groups["critic"]),
        },
        "schedulers": canonicalize(dict(scheduler_states)),
        "grad_scaler": canonicalize(scaler_state),
        "rng": canonicalize(_rng_manifest(rng_state)),
        "sampler": canonicalize(dict(sampler_state)),
        "replay": canonicalize(dict(replay_state)),
        "credits": canonicalize(dict(credit_state)),
        "counters": canonicalize(dict(counters)),
        "revision": canonicalize(dict(revision_state)),
        "durable_state": canonicalize(dict(durable_state)),
        "boundary": canonicalize(dict(boundary)),
        "bindings": canonicalize(dict(bindings)),
        "module_modes": canonicalize(dict(module_modes)),
        "actor_frozen_parent_digest": frozen_actor_digest,
    }
    section_digests = {
        name: canonical_digest(value) for name, value in sorted(sections.items())
    }
    return {
        "schema_version": EXACT_RESUME_SCHEMA_VERSION,
        "comparison": {"rtol": 0.0, "atol": 0.0, "equal_nan": False},
        "sections": sections,
        "section_digests": section_digests,
        "canonical_content_digest": canonical_digest(section_digests),
    }


def online_exact_live_state(
    *,
    modules: Mapping[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    generators: Mapping[str, torch.Generator],
    sampler_state: Mapping[str, Any],
    replay_state: Mapping[str, Any],
    credit_state: Mapping[str, Any],
    counters: Mapping[str, Any],
    revision_state: Mapping[str, Any],
    durable_state: Mapping[str, Any],
    boundary: Mapping[str, Any],
    bindings: Mapping[str, Any],
    scheduler_states: Mapping[str, Any] | None = None,
    scaler_state: Any = None,
) -> dict[str, Any]:
    scheduler_states = dict(scheduler_states or {"actor": None, "critic": None})
    return online_exact_canonical_state(
        modules=modules,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        parameter_name_groups=exact_resume_parameter_name_groups(
            modules, actor_optimizer, critic_optimizer
        ),
        scheduler_states=scheduler_states,
        scaler_state=scaler_state,
        rng_state=capture_exact_rng_state(generators),
        sampler_state=_json_safe(sampler_state),
        replay_state=_json_safe(replay_state),
        credit_state=_json_safe(credit_state),
        counters=_json_safe(counters),
        revision_state=_json_safe(revision_state),
        durable_state=_json_safe(durable_state),
        boundary=_json_safe(boundary),
        bindings=_json_safe(bindings),
        module_modes=_module_modes(modules),
        frozen_actor_digest=actor_frozen_state_digest(modules["actor"]),
    )


def _estimate_checkpoint_bytes(
    modules: Mapping[str, torch.nn.Module],
    optimizers: Sequence[torch.optim.Optimizer],
    rng_state: Mapping[str, Any],
) -> int:
    tensor_bytes = sum(
        value.numel() * value.element_size()
        for module in modules.values()
        for value in module.state_dict().values()
    )
    tensor_bytes += sum(
        value.numel() * value.element_size()
        for optimizer in optimizers
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    tensor_bytes += sum(
        value.numel() * value.element_size()
        for value in rng_state["tensors"].values()
    )
    return tensor_bytes + 64 * 1024 * 1024


def save_exact_resume_checkpoint(
    destination: Path,
    *,
    modules: Mapping[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    generators: Mapping[str, torch.Generator],
    sampler_state: Mapping[str, Any],
    replay_state: Mapping[str, Any],
    credit_state: Mapping[str, Any],
    counters: Mapping[str, Any],
    revision_state: Mapping[str, Any],
    bindings: Mapping[str, Any],
    boundary: Mapping[str, Any],
    scheduler_states: Mapping[str, Any] | None = None,
    scaler_state: Any = None,
    minimum_free_copies: int = 3,
) -> dict[str, Any]:
    """Publish a complete G5P checkpoint by same-filesystem atomic rename."""

    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"G5P_CHECKPOINT_EXISTS:{destination}")
    if set(modules) != set(_MODEL_NAMES):
        raise ValueError("G5P_MODEL_SET_INVALID")
    _validate_boundary(boundary)
    ensure_all_gradients_none(*(modules[name] for name in _MODEL_NAMES))
    scheduler_states = dict(scheduler_states or {"actor": None, "critic": None})
    if set(scheduler_states) != {"actor", "critic"}:
        raise ValueError("G5P_SCHEDULER_STATE_SCHEMA")
    durable_state = {
        "production_wal": "UNSUPPORTED_IN_ISOLATED_G5P",
        "production_outbox": "UNSUPPORTED_IN_ISOLATED_G5P",
        "production_publication": "UNSUPPORTED_IN_ISOLATED_G5P",
    }
    control = {
        "sampler": _json_safe(sampler_state),
        "replay": _json_safe(replay_state),
        "credits": _json_safe(credit_state),
        "counters": _json_safe(counters),
        "revision": _json_safe(revision_state),
        "durable_state": durable_state,
        "boundary": _json_safe(boundary),
        "bindings": _json_safe(bindings),
    }
    _validate_control_state(control)
    parameter_groups = exact_resume_parameter_name_groups(
        modules, actor_optimizer, critic_optimizer
    )
    rng_state = capture_exact_rng_state(generators)
    frozen_digest = actor_frozen_state_digest(modules["actor"])
    if bindings.get("actor_frozen_parent_digest") != frozen_digest:
        raise RuntimeError("G5P_FROZEN_ACTOR_BINDING_MISMATCH")
    modes = _module_modes(modules)
    canonical_state = online_exact_canonical_state(
        modules=modules,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        parameter_name_groups=parameter_groups,
        scheduler_states=scheduler_states,
        scaler_state=scaler_state,
        rng_state=rng_state,
        sampler_state=control["sampler"],
        replay_state=control["replay"],
        credit_state=control["credits"],
        counters=control["counters"],
        revision_state=control["revision"],
        durable_state=durable_state,
        boundary=control["boundary"],
        bindings=control["bindings"],
        module_modes=modes,
        frozen_actor_digest=frozen_digest,
    )
    metadata = {
        "schema_version": EXACT_RESUME_SCHEMA_VERSION,
        "actor_representation": "full_actor_state",
        "parameter_name_groups": parameter_groups,
        "scheduler_states": _json_safe(scheduler_states),
        "grad_scaler_state": _json_safe(scaler_state),
        "rng": _rng_manifest(rng_state),
        "module_modes": modes,
        **control,
        "canonical_state": canonical_state,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    estimate = _estimate_checkpoint_bytes(
        modules, (actor_optimizer, critic_optimizer), rng_state
    )
    free = shutil.disk_usage(destination.parent).free
    if minimum_free_copies < 3 or free < minimum_free_copies * estimate:
        raise RuntimeError(
            f"G5P_INSUFFICIENT_DISK_SPACE:free={free}:required={minimum_free_copies * estimate}"
        )
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ))
    try:
        for name in _MODEL_NAMES:
            state = {
                key: value.detach().cpu().contiguous().clone()
                for key, value in modules[name].state_dict().items()
            }
            path = temporary / "models" / f"{name}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_safetensors(state, path)
            _fsync_file(path)
            del state
        for name, optimizer in (
            ("actor", actor_optimizer), ("critic", critic_optimizer)
        ):
            path = temporary / "optimizers" / f"{name}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(optimizer.state_dict(), path)
            _fsync_file(path)
        rng_path = temporary / "state" / "rng.safetensors"
        rng_path.parent.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            {name: value.contiguous() for name, value in rng_state["tensors"].items()},
            rng_path,
        )
        _fsync_file(rng_path)
        _write_json(temporary / "metadata.json", metadata)
        entries = _manifest_entries(temporary)
        manifest = {
            "schema_version": EXACT_RESUME_SCHEMA_VERSION,
            **EXACT_RESUME_MARKERS,
            "canonical_content_digest": canonical_state["canonical_content_digest"],
            "estimated_checkpoint_bytes": estimate,
            "minimum_free_copies": minimum_free_copies,
            "files": entries,
            "files_digest": canonical_digest(entries),
        }
        _write_json(temporary / EXACT_RESUME_MANIFEST, manifest)
        _fsync_directories(temporary)
        completion = {
            "schema_version": EXACT_RESUME_SCHEMA_VERSION,
            "manifest_sha256": sha256_file(temporary / EXACT_RESUME_MANIFEST),
            "canonical_content_digest": canonical_state["canonical_content_digest"],
            "complete": True,
        }
        # The completion marker is intentionally the final file created.
        _write_json(temporary / EXACT_RESUME_COMPLETION, completion)
        _fsync_directories(temporary)
        os.replace(temporary, destination)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return {
            "manifest": manifest,
            "completion": completion,
            "checkpoint_size_bytes": sum(
                path.stat().st_size for path in destination.rglob("*") if path.is_file()
            ),
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_restricted_payloads(checkpoint: Path) -> dict[str, Any]:
    models = {
        name: load_safetensors(checkpoint / "models" / f"{name}.safetensors", device="cpu")
        for name in _MODEL_NAMES
    }
    optimizers = {
        name: torch.load(
            checkpoint / "optimizers" / f"{name}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for name in ("actor", "critic")
    }
    rng_tensors = load_safetensors(checkpoint / "state" / "rng.safetensors", device="cpu")
    return {"models": models, "optimizers": optimizers, "rng_tensors": rng_tensors}


def validate_exact_resume_checkpoint(checkpoint: Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    completion_path = checkpoint / EXACT_RESUME_COMPLETION
    if not completion_path.is_file():
        raise RuntimeError("G5P_CHECKPOINT_COMPLETION_MARKER_MISSING")
    manifest_path = checkpoint / EXACT_RESUME_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError("G5P_CHECKPOINT_MANIFEST_MISSING")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if completion != {
        "schema_version": EXACT_RESUME_SCHEMA_VERSION,
        "manifest_sha256": sha256_file(manifest_path),
        "canonical_content_digest": manifest.get("canonical_content_digest"),
        "complete": True,
    }:
        raise RuntimeError("G5P_COMPLETION_MARKER_INVALID")
    if manifest.get("schema_version") != EXACT_RESUME_SCHEMA_VERSION:
        raise RuntimeError("G5P_CHECKPOINT_SCHEMA_VERSION_MISMATCH")
    if any(manifest.get(name) != value for name, value in EXACT_RESUME_MARKERS.items()):
        raise RuntimeError("G5P_CHECKPOINT_MARKER_DRIFT")
    entries = _manifest_entries(checkpoint)
    if entries != manifest.get("files") or canonical_digest(entries) != manifest.get("files_digest"):
        raise RuntimeError("G5P_CHECKPOINT_FILE_SHA_MISMATCH")
    expected_files = {
        "metadata.json", "state/rng.safetensors",
        "optimizers/actor.pt", "optimizers/critic.pt",
        *{f"models/{name}.safetensors" for name in _MODEL_NAMES},
    }
    if {entry["relative_path"] for entry in entries} != expected_files:
        raise RuntimeError("G5P_CHECKPOINT_FILESET_MISMATCH")
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != EXACT_RESUME_SCHEMA_VERSION:
        raise RuntimeError("G5P_METADATA_SCHEMA_VERSION_MISMATCH")
    if metadata.get("actor_representation") != "full_actor_state":
        raise RuntimeError("G5P_ACTOR_REPRESENTATION_INVALID")
    control = {
        name: metadata[name]
        for name in (
            "sampler", "replay", "credits", "counters", "revision",
            "durable_state", "boundary", "bindings",
        )
    }
    _validate_control_state(control)
    payloads = _load_restricted_payloads(checkpoint)
    rng_state = {"json": metadata["rng"]["json"], "tensors": payloads["rng_tensors"]}
    if _rng_manifest(rng_state) != metadata["rng"]:
        raise RuntimeError("G5P_RNG_STATE_CORRUPTED_OR_OMITTED")
    groups = metadata["parameter_name_groups"]
    optimizer_records = {
        name: optimizer_record(payloads["optimizers"][name], groups[name])
        for name in ("actor", "critic")
    }
    for name, record in optimizer_records.items():
        if any(
            value == {"kind": "missing_optimizer_state"}
            for value in record["parameter_states"].values()
        ):
            raise RuntimeError(f"G5P_{name.upper()}_OPTIMIZER_STATE_MISSING")
    canonical_state = online_exact_canonical_state(
        modules=payloads["models"],
        actor_optimizer=payloads["optimizers"]["actor"],
        critic_optimizer=payloads["optimizers"]["critic"],
        parameter_name_groups=groups,
        scheduler_states=metadata["scheduler_states"],
        scaler_state=metadata["grad_scaler_state"],
        rng_state=rng_state,
        sampler_state=metadata["sampler"],
        replay_state=metadata["replay"],
        credit_state=metadata["credits"],
        counters=metadata["counters"],
        revision_state=metadata["revision"],
        durable_state=metadata["durable_state"],
        boundary=metadata["boundary"],
        bindings=metadata["bindings"],
        module_modes=metadata["module_modes"],
        frozen_actor_digest=metadata["bindings"]["actor_frozen_parent_digest"],
    )
    if canonical_state != metadata.get("canonical_state"):
        raise RuntimeError("G5P_CANONICAL_CONTENT_MISMATCH")
    if canonical_state["canonical_content_digest"] != manifest["canonical_content_digest"]:
        raise RuntimeError("G5P_CANONICAL_DIGEST_MISMATCH")
    return {
        "manifest": manifest,
        "completion": completion,
        "metadata": metadata,
        "payloads": payloads,
        "rng_state": rng_state,
    }


def strict_load_exact_resume_checkpoint(
    checkpoint: Path,
    *,
    modules: Mapping[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    generators: Mapping[str, torch.Generator],
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_exact_resume_checkpoint(checkpoint)
    metadata = validated["metadata"]
    if metadata["bindings"] != _json_safe(expected_bindings):
        raise RuntimeError("G5P_PARENT_CONFIG_SOURCE_BINDING_MISMATCH")
    parent_frozen = actor_frozen_state_digest(modules["actor"])
    if parent_frozen != metadata["bindings"]["actor_frozen_parent_digest"]:
        raise RuntimeError("G5P_FROZEN_PARENT_DIGEST_MISMATCH")
    actual_groups = exact_resume_parameter_name_groups(
        modules, actor_optimizer, critic_optimizer
    )
    if actual_groups != metadata["parameter_name_groups"]:
        raise RuntimeError("G5P_OPTIMIZER_PARAMETER_GROUP_MISMATCH")
    for name in _MODEL_NAMES:
        incompatible = modules[name].load_state_dict(
            validated["payloads"]["models"][name], strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"G5P_MODEL_STRICT_LOAD_FAILED:{name}")
    actor_optimizer.load_state_dict(validated["payloads"]["optimizers"]["actor"])
    critic_optimizer.load_state_dict(validated["payloads"]["optimizers"]["critic"])
    for owner, module in modules.items():
        for name, child in module.named_modules():
            key = f"{owner}.{name or '<root>'}"
            if key not in metadata["module_modes"]:
                raise RuntimeError("G5P_MODULE_MODE_STATE_MISSING")
            child.training = bool(metadata["module_modes"][key])
    ensure_all_gradients_none(*(modules[name] for name in _MODEL_NAMES))
    if actor_frozen_state_digest(modules["actor"]) != parent_frozen:
        raise RuntimeError("G5P_FROZEN_ACTOR_TENSOR_MISMATCH")
    # No random draw or model forward may follow this operation before resume.
    restore_exact_rng_state(validated["rng_state"], generators)
    return {
        "sampler": metadata["sampler"],
        "replay": metadata["replay"],
        "credits": metadata["credits"],
        "counters": metadata["counters"],
        "revision": metadata["revision"],
        "durable_state": metadata["durable_state"],
        "boundary": metadata["boundary"],
        "scheduler_states": metadata["scheduler_states"],
        "grad_scaler_state": metadata["grad_scaler_state"],
        "canonical_state": metadata["canonical_state"],
        "rng_restored_last": True,
    }


def resign_exact_resume_checkpoint_copy(checkpoint: Path) -> None:
    """Re-sign an isolated fault-injection copy without blessing its semantics."""

    checkpoint = Path(checkpoint)
    manifest_path = checkpoint / EXACT_RESUME_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = _manifest_entries(checkpoint)
    manifest["files_digest"] = canonical_digest(manifest["files"])
    temporary_manifest = manifest_path.with_name(f".{EXACT_RESUME_MANIFEST}.tmp")
    temporary_manifest.write_bytes(_json_bytes(manifest))
    os.replace(temporary_manifest, manifest_path)
    completion = {
        "schema_version": EXACT_RESUME_SCHEMA_VERSION,
        "manifest_sha256": sha256_file(manifest_path),
        "canonical_content_digest": manifest["canonical_content_digest"],
        "complete": True,
    }
    completion_path = checkpoint / EXACT_RESUME_COMPLETION
    temporary_completion = completion_path.with_name(f".{EXACT_RESUME_COMPLETION}.tmp")
    temporary_completion.write_bytes(_json_bytes(completion))
    os.replace(temporary_completion, completion_path)
