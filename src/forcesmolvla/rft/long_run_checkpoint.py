"""Cycle-boundary checkpoint contract for development long runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import torch
from torch import nn

from forcesmolvla.rft.exact_resume import directory_entries
from forcesmolvla.rft.training_cycle import ensure_all_gradients_none


MARKERS = {
    "artifact_status": "DEVELOPMENT_LONG_RUN_STAGE1",
    "deployment_status": "NOT_FOR_DEPLOYMENT",
    "policy_evaluation_status": "NOT_FOR_POLICY_EVALUATION",
    "next_stage_parent_status": "REVIEW_REQUIRED_BEFORE_USE",
    "robot_execution_authorized": False,
}


def counters_for_cycle(cycle: int) -> dict[str, int]:
    if cycle < 0 or cycle > 256:
        raise ValueError("G7_LONG_RUN_CYCLE_OUT_OF_RANGE")
    return {
        "joint_cycles": cycle,
        "critic_optimizer_updates": 2 * cycle,
        "actor_optimizer_updates": cycle,
        "q1_target_polyak_updates": 2 * cycle,
        "q2_target_polyak_updates": 2 * cycle,
        "critic_scheduler_steps": 2 * cycle,
        "actor_scheduler_steps": cycle,
        "actor_target_updates": 0,
    }


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush(); os.fsync(stream.fileno())


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def save_cycle_checkpoint(
    destination: Path,
    *,
    cycle: int,
    modules: Mapping[str, nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    actor_scheduler: Any,
    critic_scheduler: Any,
    sampler_states: Mapping[str, Any],
    rng_states: Mapping[str, Any],
    ownership_manifest: Mapping[str, Any],
    protected_snapshot: Mapping[str, Any],
    startup_snapshot_bytes: Mapping[str, bytes],
    replace_rolling: bool,
) -> dict:
    destination = Path(destination).resolve()
    counters = counters_for_cycle(cycle)
    if destination.exists() and not replace_rolling:
        raise RuntimeError("G7_LONG_RUN_APPEND_ONLY_CHECKPOINT_EXISTS")
    ensure_all_gradients_none(*modules.values())
    targets = (modules["q1_target"], modules["q2_target"])
    if any(target.training for target in targets) or any(
        parameter.requires_grad for target in targets for parameter in target.parameters()
    ):
        raise RuntimeError("G7_LONG_RUN_TARGET_OWNERSHIP_INVALID")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.new.", dir=destination.parent))
    backup = destination.parent / f".{destination.name}.previous"
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
            ("state/counters.json", counters),
            ("manifests/parameter_ownership.json", dict(ownership_manifest)),
            ("manifests/protected_snapshot.json", dict(protected_snapshot)),
        ):
            _write_json(temporary / relative, value)
        for relative, value in sorted(startup_snapshot_bytes.items()):
            target = Path(relative)
            if target.is_absolute() or ".." in target.parts:
                raise ValueError("G7_LONG_RUN_STARTUP_SNAPSHOT_PATH_INVALID")
            target_path = temporary / "startup_snapshot" / target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("xb") as stream:
                stream.write(value); stream.flush(); os.fsync(stream.fileno())
        entries = directory_entries(temporary)
        manifest = {
            "schema_version": "forcesmolvla_g7_long_run_stage1_checkpoint.v1",
            **MARKERS,
            "complete_cycle_boundary": True,
            "pending_graph": False,
            "pending_accumulation": False,
            "pending_optimizer_step": False,
            "pending_polyak_update": False,
            "all_gradients_none": True,
            "parent_checkpoint": "g7a_r2_critic_warmup_checkpoint",
            "g7b_smoke_checkpoint_used_as_parent": False,
            "cycle": cycle,
            "counters": counters,
            "total_critic_optimizer_step": 256 + 2 * cycle,
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
        if destination.exists():
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(destination, backup)
        os.replace(temporary, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if not destination.exists() and backup.exists():
            os.replace(backup, destination)
        raise


def validate_cycle_checkpoint(checkpoint: Path, *, expected_cycle: int) -> dict:
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    if any(manifest.get(key) != value for key, value in MARKERS.items()):
        raise RuntimeError("G7_LONG_RUN_CHECKPOINT_MARKER_MISMATCH")
    if manifest.get("manifest_payload_sha256") != _payload_sha256(manifest):
        raise RuntimeError("G7_LONG_RUN_CHECKPOINT_MANIFEST_PAYLOAD_MISMATCH")
    if manifest.get("cycle") != expected_cycle or manifest.get("counters") != counters_for_cycle(expected_cycle):
        raise RuntimeError("G7_LONG_RUN_CHECKPOINT_COUNTER_MISMATCH")
    if not manifest.get("complete_cycle_boundary") or any(manifest.get(key) for key in (
        "pending_graph", "pending_accumulation", "pending_optimizer_step", "pending_polyak_update"
    )):
        raise RuntimeError("G7_LONG_RUN_CHECKPOINT_NOT_AT_CYCLE_BOUNDARY")
    entries = directory_entries(checkpoint)
    if entries != manifest.get("files"):
        raise RuntimeError("G7_LONG_RUN_CHECKPOINT_INTERNAL_SHA_MISMATCH")
    return manifest


def hardlink_milestone(source: Path, destination: Path, *, expected_cycle: int) -> dict:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    validate_cycle_checkpoint(source, expected_cycle=expected_cycle)
    if destination.exists():
        raise RuntimeError("G7_LONG_RUN_MILESTONE_EXISTS")
    temporary = destination.parent / f".{destination.name}.new"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary, copy_function=os.link)
    os.replace(temporary, destination)
    parent_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return validate_cycle_checkpoint(destination, expected_cycle=expected_cycle)
