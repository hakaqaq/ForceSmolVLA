"""Reusable ForceRFT training and validation runtime primitives."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any


def _validate_task_id(task_id: str) -> str:
    task_id = task_id.strip()
    if not task_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in task_id
    ):
        raise ValueError("TASK_ID_INVALID")
    return task_id


def _resolve_task_root(
    repository_root: Path,
    *,
    task_id: str,
    selected_root: Path | None,
    default_relative: Path,
) -> Path:
    _validate_task_id(task_id)
    selected = repository_root / default_relative if selected_root is None else selected_root
    return Path(selected).expanduser().resolve()


def resolve_task_output_root(
    repository_root: Path,
    *,
    task_id: str,
    output_root: Path | None = None,
) -> Path:
    """Return the sole task-scoped training output root."""

    return _resolve_task_root(
        repository_root,
        task_id=task_id,
        selected_root=output_root,
        default_relative=Path("outputs") / task_id,
    )


def resolve_task_dataset_root(
    repository_root: Path,
    *,
    task_id: str,
    dataset_root: Path | None = None,
) -> Path:
    """Return the task's canonical LeRobot-v3 source dataset root."""

    return _resolve_task_root(
        repository_root,
        task_id=task_id,
        selected_root=dataset_root,
        default_relative=Path("datasets") / f"{task_id}_lerobotv3",
    )


def resolve_task_reward_transition_root(
    repository_root: Path,
    *,
    task_id: str,
    reward_transition_root: Path | None = None,
) -> Path:
    """Return the task's canonical offline reward-transition dataset root."""

    return _resolve_task_root(
        repository_root,
        task_id=task_id,
        selected_root=reward_transition_root,
        default_relative=(
            Path("datasets") / f"{task_id}_forcerft_offline_reward_transitions"
        ),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(file_sha256(path).encode() + b"\n")
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require_offline_environment() -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 required")
    socket.socket.connect = lambda self, address: (_ for _ in ()).throw(
        RuntimeError(f"NETWORK_ACCESS_FORBIDDEN: {address}")
    )


def build_training_batch(policy: Any, prepared_samples: list[dict], device: Any) -> dict:
    import numpy as np
    import torch

    from forceprior.configuration_forceprior import CAMERA1, CAMERA2
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    encoded = tokenizer(
        [sample["task"] + "\n" for sample in prepared_samples],
        padding="max_length",
        truncation=True,
        max_length=48,
        return_tensors="pt",
    )
    return {
        CAMERA1: torch.stack([sample["camera1"] for sample in prepared_samples]).to(device),
        CAMERA2: torch.stack([sample["camera2"] for sample in prepared_samples]).to(device),
        "observation.state": torch.from_numpy(
            np.stack([sample["state7"] for sample in prepared_samples])
        ).to(device),
        "observation.wrench": torch.from_numpy(
            np.stack([sample["wrench6"] for sample in prepared_samples])
        ).to(device),
        ACTION: torch.from_numpy(
            np.stack([sample["delta_action7"] for sample in prepared_samples])
        ).to(device),
        "action_valid_mask": torch.from_numpy(
            np.stack([sample["action_valid_mask"] for sample in prepared_samples])
        ).to(device),
        OBS_LANGUAGE_TOKENS: encoded["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: encoded["attention_mask"].to(
            device=device, dtype=torch.bool
        ),
    }


def validation_scalar(policy: Any, batch: dict, noise: Any, timestep: Any) -> float:
    import torch

    policy.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        losses, feature_mask, _router = policy.forward_single_pass_training_terms(
            batch, noise=noise, time=timestep
        )
        scalar = losses.sum() / feature_mask.sum()
    torch.cuda.synchronize()
    return float(scalar.cpu())


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().to(device="cpu", dtype=tensor.dtype).contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def build_validation_fixture(
    *,
    root: Path,
    dataset_root: Path,
    raw_samples: list[dict],
    batch: dict,
    noise: Any,
    timestep: Any,
) -> dict:
    import torch

    from forceprior.context import ChunkContext

    normalizer_sha = file_sha256(dataset_root / "normalizer_manifest.json")
    calibration_sha = file_sha256(root / "configs/calibration_bundle.json")
    geometry_sha = file_sha256(root / "configs/wrench_geometry_spec.json")
    action_mask = batch["action_valid_mask"].detach().cpu()
    tuples = [
        {
            "split": "val",
            "fixture_position": index,
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
        }
        for index, sample in enumerate(raw_samples)
    ]
    chunk = {
        "policy_generation": 0,
        "raw_state_snapshot": [
            torch.as_tensor(sample["observation.state"]).tolist() for sample in raw_samples
        ],
        "t_ref_ns": [int(round(float(sample["timestamp"]) * 1e9)) for sample in raw_samples],
        "tau0_ns": [int(round(float(sample["timestamp"]) * 1e9)) for sample in raw_samples],
        "clock_domain_id": ["lerobot_v3_episode_time"] * len(raw_samples),
        "episode_id": [f"episode_{int(sample['episode_index']):06d}" for sample in raw_samples],
        "session_id": ["task1_within_session"] * len(raw_samples),
        "sample_id": [
            f"episode_{int(sample['episode_index']):06d}/frame_{int(sample['frame_index']):06d}"
            for sample in raw_samples
        ],
        "chunk_id": [f"sft-validation-{index}" for index in range(len(raw_samples))],
        "action_valid_mask": action_mask.tolist(),
        "suffix_valid_mask": action_mask.tolist(),
        "calibration_bundle_hash": [calibration_sha] * len(raw_samples),
        "wrench_geometry_spec_hash": [geometry_sha] * len(raw_samples),
        "normalizer_hash": [normalizer_sha] * len(raw_samples),
        "calibration_mapping_hash_or_none": [None] * len(raw_samples),
        "wrench_geometry_valid": [True] * len(raw_samples),
        "runtime_artifact_compatible": [True] * len(raw_samples),
        "selected_provenance": tuples,
    }
    context = ChunkContext(
        policy_generation=0,
        raw_state_snapshot=torch.tensor(chunk["raw_state_snapshot"]),
        t_ref_ns=torch.tensor(chunk["t_ref_ns"], dtype=torch.int64),
        tau0_ns=torch.tensor(chunk["tau0_ns"], dtype=torch.int64),
        clock_domain_id=tuple(chunk["clock_domain_id"]),
        episode_id=tuple(chunk["episode_id"]),
        session_id=tuple(chunk["session_id"]),
        sample_id=tuple(chunk["sample_id"]),
        chunk_id=tuple(chunk["chunk_id"]),
        action_valid_mask=action_mask,
        suffix_valid_mask=action_mask,
        calibration_bundle_hash=tuple(chunk["calibration_bundle_hash"]),
        wrench_geometry_spec_hash=tuple(chunk["wrench_geometry_spec_hash"]),
        normalizer_hash=tuple(chunk["normalizer_hash"]),
        calibration_mapping_hash_or_none=tuple(chunk["calibration_mapping_hash_or_none"]),
        wrench_geometry_valid=torch.ones(len(raw_samples), dtype=torch.bool),
        runtime_artifact_compatible=torch.ones(len(raw_samples), dtype=torch.bool),
        selected_provenance=tuple(tuples),
    )
    context.validate(batch_size=len(raw_samples), horizon=50, policy_generation=0)
    return {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "mode": "fixed_validation_development",
        "tuple_list": tuples,
        "masks": {
            "action_valid_mask": action_mask.tolist(),
            "valid_feature_count": int(action_mask.sum()) * 7,
        },
        "epsilon7": {
            "dtype": "float32",
            "shape": list(noise.shape),
            "tensor": noise.detach().cpu().tolist(),
            "sha256": _tensor_sha256(noise),
        },
        "time": {
            "dtype": "float32",
            "shape": list(timestep.shape),
            "tensor": timestep.detach().cpu().tolist(),
            "sha256": _tensor_sha256(timestep),
        },
        "chunk_context": chunk,
        "chunk_context_sha256": canonical_sha256(chunk),
        "checkpoint_selection_use": "single_pass_global_valid_feature_token_weighted_L_flow_only",
        "validation_algorithm": "single_pass_batch_local",
        "detached_signature": None,
        "approval": None,
    }
