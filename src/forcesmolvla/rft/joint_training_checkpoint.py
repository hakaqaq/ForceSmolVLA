"""Append-only joint-training ownership and checkpoint contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from forcesmolvla.rft.exact_resume import directory_entries
from forcesmolvla.rft.source_manifest import validate_stage2_source_manifest
from forcesmolvla.rft.training_cycle import ensure_all_gradients_none


JOINT_TRAINING_CHECKPOINT_MARKERS = {
    "artifact_status": "DEVELOPMENT_G7B_JOINT_SMOKE_ONLY",
    "deployment_status": "NOT_FOR_DEPLOYMENT",
    "policy_evaluation_status": "NOT_FOR_POLICY_EVALUATION",
    "long_train_parent_status": "NOT_AN_APPROVED_LONG_TRAIN_PARENT",
    "robot_execution_authorized": False,
}
JOINT_TRAINING_COUNTERS = {
    "joint_cycles": 8,
    "critic_optimizer_updates": 16,
    "actor_optimizer_updates": 8,
    "q1_target_polyak_updates": 16,
    "q2_target_polyak_updates": 16,
    "critic_scheduler_steps": 16,
    "actor_scheduler_steps": 8,
    "actor_target_updates": 0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_joint_training_source_manifest(root: Path, manifest_path: Path) -> dict:
    payload = validate_stage2_source_manifest(root, manifest_path)
    if payload.get("scope") != "G7B_development_joint_smoke_ActionContract_v2":
        raise RuntimeError("G7B_SOURCE_SCOPE_INVALID")
    paths = [entry["relative_path"] for entry in payload["files"]]
    if any("manual_reward" in path or path.startswith("labels/") for path in paths):
        raise RuntimeError("G7B_MANUAL_SOURCE_IN_RUNTIME_CLOSURE")
    return payload


def build_actor_optimizer_scheduler(actor: nn.Module):
    """Build the frozen Stage-1 decay/no-decay AdamW without a second Critic optimizer."""
    from forcesmolvla.router_training import _no_decay_parameter_names

    named = dict(actor.named_parameters())
    if not named or not all(parameter.requires_grad for parameter in named.values()):
        raise RuntimeError("G7B_ACTOR_TRAINABILITY_INVALID")
    no_decay = _no_decay_parameter_names(actor)
    decay = set(named) - no_decay
    if decay & no_decay or decay | no_decay != set(named):
        raise RuntimeError("G7B_ACTOR_PARAMETER_PARTITION_INVALID")
    optimizer = torch.optim.AdamW(
        [
            {"params": [named[name] for name in sorted(decay)], "weight_decay": 1e-10},
            {"params": [named[name] for name in sorted(no_decay)], "weight_decay": 0.0},
        ],
        lr=1e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    owned = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(owned) != len(set(owned)) or set(owned) != {
        id(parameter) for parameter in actor.parameters() if parameter.requires_grad
    }:
        raise RuntimeError("G7B_ACTOR_OPTIMIZER_OWNERSHIP_INVALID")
    return optimizer, scheduler, {
        "type": "AdamW",
        "trainable_tensor_count": len(owned),
        "parameter_count": sum(parameter.numel() for parameter in named.values()),
        "decay_tensor_count": len(decay),
        "no_decay_tensor_count": len(no_decay),
    }


def describe_p95(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("G7B_STATISTIC_INPUT_INVALID")
    return {
        "count": int(array.size),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def validate_optimizer_step_sets(
    critic_steps: set[int], actor_steps: set[int], *, expected_actor_updates: int = 8
) -> None:
    """Validate global progress without rejecting sparsely routed Actor parameters."""
    if (
        critic_steps != {256 + 2 * expected_actor_updates}
        or not actor_steps
        or max(actor_steps) != expected_actor_updates
        or min(actor_steps) < 1
        or max(actor_steps) > expected_actor_updates
    ):
        raise RuntimeError(
            f"G7B_OPTIMIZER_COUNTER_DRIFT:{critic_steps}:{actor_steps}"
        )


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def save_joint_training_checkpoint(
    destination: Path,
    *,
    modules: Mapping[str, nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    actor_scheduler: Any,
    critic_scheduler: Any,
    counters: Mapping[str, int],
    parent_counters: Mapping[str, int],
    sampler_states: Mapping[str, Any],
    rng_states: Mapping[str, Any],
    ownership_manifest: Mapping[str, Any],
    protected_snapshot: Mapping[str, Any],
    startup_snapshot_bytes: Mapping[str, bytes],
) -> dict:
    destination = Path(destination).resolve()
    if destination.exists() or dict(counters) != JOINT_TRAINING_COUNTERS:
        raise RuntimeError("G7B_CHECKPOINT_TARGET_OR_COUNTER_INVALID")
    ensure_all_gradients_none(*modules.values())
    targets = (modules["q1_target"], modules["q2_target"])
    if any(target.training for target in targets) or any(
        parameter.requires_grad for target in targets for parameter in target.parameters()
    ):
        raise RuntimeError("G7B_CHECKPOINT_TARGET_OWNERSHIP_INVALID")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name, module in modules.items():
            _torch_save(temporary / f"models/{name}_state.pt", module.state_dict())
        for relative, value in (
            ("optimizers/actor_optimizer_state.pt", actor_optimizer.state_dict()),
            ("optimizers/critic_optimizer_state.pt", critic_optimizer.state_dict()),
            ("schedulers/actor_scheduler_state.pt", actor_scheduler.state_dict()),
            ("schedulers/critic_scheduler_state.pt", critic_scheduler.state_dict()),
            ("state/sampler_states.pt", dict(sampler_states)),
            ("state/rng_states.pt", dict(rng_states)),
        ):
            _torch_save(temporary / relative, value)
        for relative, value in (
            ("state/counters.json", dict(counters)),
            ("state/parent_counters.json", dict(parent_counters)),
            ("manifests/parameter_ownership.json", dict(ownership_manifest)),
            ("manifests/protected_snapshot.json", dict(protected_snapshot)),
        ):
            _write_json(temporary / relative, value)
        for relative, value in sorted(startup_snapshot_bytes.items()):
            target = Path(relative)
            if target.is_absolute() or ".." in target.parts:
                raise ValueError("G7B_STARTUP_SNAPSHOT_PATH_INVALID")
            _write_bytes(temporary / "startup_snapshot" / target, value)
        entries = directory_entries(temporary)
        manifest = {
            "schema_version": "forcesmolvla_g7b_joint_smoke_checkpoint.v1",
            **JOINT_TRAINING_CHECKPOINT_MARKERS,
            "complete_cycle_boundary": True,
            "pending_graph": False,
            "pending_accumulation": False,
            "pending_optimizer_step": False,
            "pending_polyak_update": False,
            "all_gradients_none": True,
            "parent_checkpoint": "g7a_r2_critic_warmup_checkpoint",
            "counters": dict(counters),
            "parent_counters": dict(parent_counters),
            "total_critic_optimizer_step": int(parent_counters["critic_optimizer_updates"]) + 16,
            "files": entries,
            "files_sha256": hashlib.sha256(
                json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        manifest["manifest_payload_sha256"] = _payload_sha256(manifest)
        _write_json(temporary / "checkpoint_manifest.json", manifest)
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(temporary, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_joint_training_checkpoint(checkpoint: Path) -> dict:
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    if any(manifest.get(key) != value for key, value in JOINT_TRAINING_CHECKPOINT_MARKERS.items()):
        raise RuntimeError("G7B_CHECKPOINT_MARKER_MISMATCH")
    if manifest.get("manifest_payload_sha256") != _payload_sha256(manifest):
        raise RuntimeError("G7B_CHECKPOINT_MANIFEST_PAYLOAD_MISMATCH")
    if manifest.get("counters") != JOINT_TRAINING_COUNTERS or not manifest.get("complete_cycle_boundary"):
        raise RuntimeError("G7B_CHECKPOINT_COUNTER_OR_BOUNDARY_INVALID")
    if any(manifest.get(key) for key in (
        "pending_graph", "pending_accumulation", "pending_optimizer_step", "pending_polyak_update"
    )):
        raise RuntimeError("G7B_CHECKPOINT_PENDING_WORK")
    entries = directory_entries(checkpoint)
    if entries != manifest.get("files"):
        raise RuntimeError("G7B_CHECKPOINT_INTERNAL_FILE_SHA_MISMATCH")
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != manifest.get("files_sha256"):
        raise RuntimeError("G7B_CHECKPOINT_FILE_DIGEST_MISMATCH")
    if json.loads((checkpoint / "state/counters.json").read_text()) != JOINT_TRAINING_COUNTERS:
        raise RuntimeError("G7B_CHECKPOINT_COUNTER_FILE_MISMATCH")
    return manifest
