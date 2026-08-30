"""Critic warm-up statistics and checkpoint contract."""

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
from torch import Tensor, nn

from forcesmolvla.rft.exact_resume import directory_entries
from forcesmolvla.rft.training_cycle import ensure_all_gradients_none


CRITIC_WARMUP_CHECKPOINT_MARKERS = {
    "artifact_status": "DEVELOPMENT_G7A_CRITIC_WARMUP_ONLY",
    "deployment_status": "NOT_FOR_DEPLOYMENT",
    "policy_evaluation_status": "NOT_FOR_POLICY_EVALUATION",
    "long_train_parent_status": "NOT_AN_APPROVED_LONG_TRAIN_PARENT",
    "g7b_use_status": "APPROVED_ONLY_FOR_G7B_IF_EXPLICITLY_AUTHORIZED",
    "robot_execution_authorized": False,
}
CRITIC_WARMUP_COUNTERS = {
    "critic_optimizer_updates": 256,
    "critic_scheduler_steps": 256,
    "q1_target_polyak_updates": 256,
    "q2_target_polyak_updates": 256,
    "actor_optimizer_updates": 0,
    "actor_scheduler_steps": 0,
    "actor_target_updates": 0,
}


def verify_source_manifest(root: Path, manifest_path: Path) -> dict:
    """Fail closed on the immutable G7-A source/config closure."""
    payload = json.loads(Path(manifest_path).read_text())
    if payload.get("schema_version") != "forcesmolvla_stage2_source_manifest.v9_g7a":
        raise RuntimeError("G7A_SOURCE_MANIFEST_SCHEMA_INVALID")
    if payload.get("manual_g1_or_manual_label_in_runtime_closure") is not False:
        raise RuntimeError("G7A_MANUAL_SOURCE_IN_RUNTIME_CLOSURE")
    for name, record in payload.get("files", {}).items():
        path = Path(root) / record["path"]
        if not path.is_file():
            raise RuntimeError(f"G7A_SOURCE_FILE_MISSING:{name}")
        if path.stat().st_size != int(record["file_size"]):
            raise RuntimeError(f"G7A_SOURCE_FILE_SIZE_MISMATCH:{name}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"G7A_SOURCE_FILE_SHA_MISMATCH:{name}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_component_digests(module: nn.Module) -> dict:
    def digest(items) -> str:
        value = hashlib.sha256()
        for name, tensor in sorted(items):
            value.update(name.encode())
            value.update(tensor_sha256(tensor).encode())
        return value.hexdigest()

    parameters = list(module.named_parameters())
    floating_buffers = [
        (name, value) for name, value in module.named_buffers() if value.is_floating_point()
    ]
    other_buffers = [
        (name, value) for name, value in module.named_buffers() if not value.is_floating_point()
    ]
    return {
        "parameters_sha256": digest(parameters),
        "floating_buffers_sha256": digest(floating_buffers),
        "nonfloating_buffers_sha256": digest(other_buffers),
        "parameter_tensor_count": len(parameters),
        "floating_buffer_count": len(floating_buffers),
        "nonfloating_buffer_count": len(other_buffers),
    }


def describe(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "minimum": None, "p10": None,
                "median": None, "p90": None, "maximum": None}
    if not np.isfinite(array).all():
        raise FloatingPointError("G7A_NONFINITE_STATISTIC_INPUT")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(array.max()),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("G7A_SPEARMAN_SHAPE_MISMATCH")
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    value = float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])
    if not math.isfinite(value):
        raise FloatingPointError("G7A_SPEARMAN_NONFINITE")
    return value


def regression_metrics(q: Sequence[float], mc_return: Sequence[float]) -> dict:
    prediction = np.asarray(q, dtype=np.float64)
    target = np.asarray(mc_return, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("G7A_REGRESSION_SHAPE_MISMATCH")
    if prediction.size == 0:
        return {"count": 0, "mae": None, "rmse": None, "bias": None,
                "spearman": None}
    error = prediction - target
    return {
        "count": int(prediction.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "spearman": spearman_correlation(prediction, target),
    }


def policy_distance_bucket(distance: int) -> str:
    if distance == 0:
        return "0_terminal"
    if distance == 1:
        return "1"
    if distance <= 5:
        return "2_5"
    if distance <= 20:
        return "6_20"
    if distance <= 50:
        return "21_50"
    if distance <= 100:
        return "51_100"
    return "gt_100"


def grouped_regression(rows: Sequence[Mapping[str, Any]]) -> dict:
    groups: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        labels = (
            f"terminal={str(bool(row['terminated'])).lower()}",
            f"executed_steps={int(row['executed_steps'])}",
            f"distance={policy_distance_bucket(int(row['policy_decision_distance']))}",
        )
        for label in labels:
            group = groups.setdefault(label, {"q": [], "mc": []})
            group["q"].append(float(row["q_mean"]))
            group["mc"].append(float(row["mc_return"]))
    return {
        name: regression_metrics(value["q"], value["mc"])
        for name, value in sorted(groups.items())
    }


def select_fixed_critic_probe(
    rows: Sequence[Mapping[str, Any]], size: int, *, seed: int = 0
) -> list[int]:
    """Cover every rare terminal/partial-tail row, then evenly fill full macros."""

    mandatory = [
        index for index, row in enumerate(rows)
        if bool(row["terminated"]) or int(row["executed_steps"]) < 3
    ]
    if len(mandatory) > size:
        raise ValueError("G7A_TRAIN_PROBE_TOO_SMALL_FOR_RARE_ROWS")
    remaining = np.asarray(
        [index for index in range(len(rows)) if index not in set(mandatory)],
        dtype=np.int64,
    )
    np.random.default_rng(seed).shuffle(remaining)
    needed = size - len(mandatory)
    if needed:
        mandatory.extend(int(index) for index in remaining[:needed])
    result = sorted(set(mandatory))
    if len(result) != size:
        raise RuntimeError("G7A_TRAIN_PROBE_SELECTION_NOT_EXACT")
    return result


def actor_gradient_group(name: str) -> str:
    if name.startswith("model.vlm_with_expert.lm_expert."):
        return "action_expert"
    if name.startswith("model.vlm_with_expert."):
        return "vision_vlm"
    if name.startswith(("model.action_in_proj.", "model.action_out_proj.", "model.state_proj.")):
        return "action_io"
    if name.startswith("model.force_branch.force_mlp."):
        return "force_mlp"
    if name.startswith((
        "model.force_branch.segment_embedding.",
        "model.force_branch.fusion_position_embedding.",
        "model.force_branch.fusion_blocks.",
        "model.force_branch.guidance_projection.",
        "model.force_branch.refiner.experts.",
    )):
        return "fusion_moe"
    if name.startswith("model.force_adapter."):
        return "force_adapter"
    if name.startswith("model.force_branch.refiner.router."):
        return "router"
    return "unassigned"


def aggregate_gradient_probes(
    probes: Sequence[Mapping[str, Any]], eta_candidates: Sequence[float], band: Sequence[float]
) -> dict:
    if len(band) != 2 or band[0] > band[1]:
        raise ValueError("G7A_REFERENCE_RATIO_BAND_INVALID")

    def aggregate(items: Sequence[Mapping[str, float]]) -> dict:
        return {
            "raw_q_over_fm": describe([item["raw_q_over_fm"] for item in items]),
            "cosine_similarity": describe([item["cosine_similarity"] for item in items]),
            "fm_norm": describe([item["fm_norm"] for item in items]),
            "q_norm": describe([item["q_norm"] for item in items]),
        }

    global_result = aggregate([probe["global"] for probe in probes])
    modules = sorted({name for probe in probes for name in probe["modules"]})
    module_result = {
        name: aggregate([probe["modules"][name] for probe in probes])
        for name in modules
    }
    eta = {}
    candidates_in_band = []
    raw = [float(probe["global"]["raw_q_over_fm"]) for probe in probes]
    for candidate in eta_candidates:
        weighted = [float(candidate) * value for value in raw]
        summary = describe(weighted)
        in_band = bool(band[0] <= summary["median"] <= band[1])
        eta[str(candidate)] = {"weighted_ratio": summary, "median_in_reference_band": in_band}
        if in_band:
            candidates_in_band.append(float(candidate))
    for name in modules:
        module_raw = [
            float(probe["modules"][name]["raw_q_over_fm"]) for probe in probes
        ]
        module_result[name]["weighted_ratio_by_eta"] = {
            str(candidate): describe([
                float(candidate) * value for value in module_raw
            ])
            for candidate in eta_candidates
        }
    return {
        "probe_batch_count": len(probes),
        "global": global_result,
        "modules": module_result,
        "eta_candidates": eta,
        "reference_weighted_ratio_band": list(band),
        "candidates_with_median_in_reference_band": candidates_in_band,
        "eta_selected_or_approved": False,
    }


def _manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def save_critic_warmup_checkpoint(
    destination: Path,
    *,
    critics: Mapping[str, nn.Module],
    critic_optimizer: torch.optim.Optimizer,
    critic_scheduler: Any,
    counters: Mapping[str, int],
    sampler_states: Mapping[str, Any],
    rng_states: Mapping[str, Any],
    actor_binding: Mapping[str, Any],
    ownership_manifest: Mapping[str, Any],
    fixed_diagnostics_manifest: Mapping[str, Any],
    protected_snapshot: Mapping[str, Any],
    startup_snapshot_bytes: Mapping[str, bytes],
) -> dict:
    destination = Path(destination).resolve()
    if destination.exists() or dict(counters) != CRITIC_WARMUP_COUNTERS:
        raise RuntimeError("G7A_CHECKPOINT_TARGET_OR_COUNTER_INVALID")
    ensure_all_gradients_none(*critics.values())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name, module in critics.items():
            _torch_save(temporary / f"models/{name}_state.pt", module.state_dict())
        _torch_save(temporary / "optimizers/critic_optimizer_state.pt", critic_optimizer.state_dict())
        _torch_save(temporary / "schedulers/critic_scheduler_state.pt", critic_scheduler.state_dict())
        _torch_save(temporary / "state/sampler_states.pt", dict(sampler_states))
        _torch_save(temporary / "state/rng_states.pt", dict(rng_states))
        for relative, value in (
            ("state/counters.json", dict(counters)),
            ("manifests/actor_binding.json", dict(actor_binding)),
            ("manifests/parameter_ownership.json", dict(ownership_manifest)),
            ("manifests/fixed_diagnostics.json", dict(fixed_diagnostics_manifest)),
            ("manifests/protected_snapshot.json", dict(protected_snapshot)),
        ):
            _write_json(temporary / relative, value)
        for relative, value in sorted(startup_snapshot_bytes.items()):
            target = Path(relative)
            if target.is_absolute() or ".." in target.parts:
                raise ValueError("G7A_STARTUP_SNAPSHOT_PATH_INVALID")
            target_path = temporary / "startup_snapshot" / target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("xb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
        entries = directory_entries(temporary)
        manifest = {
            "schema_version": "forcesmolvla_g7a_critic_warmup_checkpoint.v1",
            **CRITIC_WARMUP_CHECKPOINT_MARKERS,
            "complete_update_boundary": True,
            "pending_optimizer_step": False,
            "pending_polyak_update": False,
            "pending_gradient": False,
            "actor_state_stored": False,
            "actor_bound_to_frozen_r5": True,
            "counters": dict(counters),
            "files": entries,
            "files_sha256": hashlib.sha256(
                json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        manifest["manifest_payload_sha256"] = _manifest_payload_sha256(manifest)
        _write_json(temporary / "checkpoint_manifest.json", manifest)
        fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, destination)
        fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_critic_warmup_checkpoint(checkpoint: Path) -> dict:
    checkpoint = Path(checkpoint)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if any(manifest.get(key) != value for key, value in CRITIC_WARMUP_CHECKPOINT_MARKERS.items()):
        raise RuntimeError("G7A_CHECKPOINT_MARKER_MISMATCH")
    if manifest.get("manifest_payload_sha256") != _manifest_payload_sha256(manifest):
        raise RuntimeError("G7A_CHECKPOINT_MANIFEST_PAYLOAD_MISMATCH")
    if manifest.get("counters") != CRITIC_WARMUP_COUNTERS or not manifest.get("complete_update_boundary"):
        raise RuntimeError("G7A_CHECKPOINT_COUNTER_OR_BOUNDARY_INVALID")
    if any(manifest.get(key) for key in (
        "pending_optimizer_step", "pending_polyak_update", "pending_gradient"
    )):
        raise RuntimeError("G7A_CHECKPOINT_PENDING_WORK")
    entries = directory_entries(checkpoint)
    if entries != manifest.get("files"):
        raise RuntimeError("G7A_CHECKPOINT_INTERNAL_FILE_SHA_MISMATCH")
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != manifest.get("files_sha256"):
        raise RuntimeError("G7A_CHECKPOINT_FILE_DIGEST_MISMATCH")
    counters = json.loads((checkpoint / "state/counters.json").read_text())
    if counters != CRITIC_WARMUP_COUNTERS:
        raise RuntimeError("G7A_CHECKPOINT_COUNTER_FILE_MISMATCH")
    return manifest
