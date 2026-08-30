"""Atomic, cycle-boundary-only ForceRFT training checkpoint writer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import torch

from forcesmolvla.rft.training_cycle import ensure_all_gradients_none


G5_CHECKPOINT_MARKERS = {
    "artifact_status": "DEVELOPMENT_SINGLE_CYCLE_ONLY",
    "deployment_status": "NOT_FOR_DEPLOYMENT",
    "policy_evaluation_status": "NOT_FOR_POLICY_EVALUATION",
    "long_train_parent_status": "NOT_AN_APPROVED_LONG_TRAIN_PARENT",
    "robot_execution_authorized": False,
    "resume_exactness_tested": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


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
    _fsync_file(path)


def _directory_manifest(root: Path) -> list[dict]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "file_size": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checkpoint_manifest.json"
    ]


def save_g5_cycle_checkpoint(
    destination: Path,
    *,
    actor,
    q1,
    q2,
    q1_target,
    q2_target,
    actor_optimizer,
    critic_optimizer,
    actor_scheduler,
    critic_scheduler,
    counters: Mapping[str, int],
    sampler_states: Mapping[str, dict],
    rng_states: Mapping[str, Any],
    startup_snapshot_bytes: Mapping[str, bytes],
    parameter_ownership_manifest: dict,
    trainability_manifest: dict,
    proposal_population_manifest: dict,
) -> dict:
    """Write one complete checkpoint by temp-dir + fsync + atomic rename."""

    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite G5 checkpoint: {destination}")
    expected = {
        "training_cycles": 1,
        "critic_optimizer_updates": 2,
        "actor_optimizer_updates": 1,
        "q1_target_polyak_updates": 2,
        "q2_target_polyak_updates": 2,
        "actor_target_updates": 0,
        "critic_scheduler_steps": 2,
        "actor_scheduler_steps": 1,
    }
    if dict(counters) != expected:
        raise RuntimeError(f"G5_CHECKPOINT_COUNTER_BOUNDARY_INVALID:{dict(counters)}")
    ensure_all_gradients_none(actor, q1, q2, q1_target, q2_target)
    if any(target.training for target in (q1_target, q2_target)) or any(
        parameter.requires_grad
        for target in (q1_target, q2_target)
        for parameter in target.parameters()
    ):
        raise RuntimeError("G5_CHECKPOINT_TARGET_OWNERSHIP_INVALID")
    if not startup_snapshot_bytes or any(not isinstance(value, bytes) for value in startup_snapshot_bytes.values()):
        raise ValueError("G5_STARTUP_SNAPSHOT_BYTES_INVALID")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        # torch.save preserves tied/shared tensors and the exact optimizer state.
        for name, value in (
            ("models/actor_state.pt", actor.state_dict()),
            ("models/q1_state.pt", q1.state_dict()),
            ("models/q2_state.pt", q2.state_dict()),
            ("models/q1_target_state.pt", q1_target.state_dict()),
            ("models/q2_target_state.pt", q2_target.state_dict()),
            ("optimizers/actor_optimizer_state.pt", actor_optimizer.state_dict()),
            ("optimizers/critic_optimizer_state.pt", critic_optimizer.state_dict()),
            ("schedulers/actor_scheduler_state.pt", actor_scheduler.state_dict()),
            ("schedulers/critic_scheduler_state.pt", critic_scheduler.state_dict()),
            ("state/sampler_states.pt", dict(sampler_states)),
            ("state/rng_states.pt", dict(rng_states)),
        ):
            _torch_save(temporary / name, value)
        _write_json(temporary / "state/counters.json", dict(counters))
        _write_json(temporary / "manifests/parameter_ownership.json", parameter_ownership_manifest)
        _write_json(temporary / "manifests/trainability.json", trainability_manifest)
        _write_json(temporary / "manifests/proposal_population.json", proposal_population_manifest)
        for relative, value in sorted(startup_snapshot_bytes.items()):
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("G5_STARTUP_SNAPSHOT_RELATIVE_PATH_INVALID")
            _write_bytes(temporary / "startup_snapshot" / path, value)

        entries = _directory_manifest(temporary)
        manifest = {
            "schema_version": "forcesmolvla_g5_single_cycle_checkpoint.v1",
            **G5_CHECKPOINT_MARKERS,
            "cycle_boundary_complete": True,
            "pending_graphs": 0,
            "pending_accumulation_microbatches": 0,
            "pending_optimizer_steps": 0,
            "all_gradients_none": True,
            "fresh_process_exact_resume_tested": False,
            "counters": dict(counters),
            "files": entries,
            "files_sha256": hashlib.sha256(
                json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        manifest["manifest_payload_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
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


def validate_g5_checkpoint(destination: Path) -> dict:
    destination = Path(destination).resolve()
    manifest = json.loads((destination / "checkpoint_manifest.json").read_text())
    if any(manifest.get(name) != value for name, value in G5_CHECKPOINT_MARKERS.items()):
        raise RuntimeError("G5_CHECKPOINT_MARKER_DRIFT")
    entries = _directory_manifest(destination)
    if entries != manifest["files"]:
        raise RuntimeError("G5_CHECKPOINT_FILE_MANIFEST_DRIFT")
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != manifest["files_sha256"]:
        raise RuntimeError("G5_CHECKPOINT_FILES_SHA_DRIFT")
    return manifest
