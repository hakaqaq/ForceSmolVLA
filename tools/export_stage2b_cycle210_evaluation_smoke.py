#!/usr/bin/env python3
"""Export cycle-210 Actor weights and prove direct/public/HTTP smoke parity."""

from __future__ import annotations

import argparse
import base64
from dataclasses import replace
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import traceback
from typing import Any, Mapping
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
from safetensors.torch import load_file
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
CONFIG = ROOT / "configs/stage2b_cycle210_evaluation_smoke.development.json"
FROZEN_PREFIXES = ("model.vlm_with_expert.vlm.", "model.state_proj.")
CLIENT_ROOT = Path("/home/rlc123/fr3_client_ws")
CLIENT_SOURCE_FILES = (
    "scripts/deploy_forcesmolvla.py",
    "scripts/deploy_forcevla.py",
    "scripts/record_franka_hilserl_impedance.py",
    "scripts/hilserl_impedance_protocol.py",
    "scripts/record_franka_forcevla.py",
    "scripts/record_franka_forcevla_raw.py",
    "scripts/record_franka_spacemouse_publisher.py",
    "scripts/convert_franka_forcevla_raw_to_lerobot_v21.py",
)


def require(value: bool, code: str) -> None:
    if not value:
        raise RuntimeError(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"append-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"append-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def tree_record(root: Path) -> dict[str, Any]:
    records = []
    digest = hashlib.sha256()
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        sha = sha256_file(path)
        size = path.stat().st_size
        records.append({"relative_path": relative, "sha256": sha, "file_size": size})
        digest.update(f"{relative}\0{sha}\n".encode())
        total += size
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(records),
        "total_file_size": total,
        "files": records,
    }


def tensor_state_record(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    digest = hashlib.sha256()
    total = 0
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor), f"ACTOR_STATE_NON_TENSOR:{name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        total += tensor.numel()
    return {"sha256": digest.hexdigest(), "tensor_count": len(state), "numel": total}


def tensor_record(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "finite": bool(torch.isfinite(tensor).all()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def exact_tensor_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    a = left.detach().cpu().to(torch.float32).contiguous()
    b = right.detach().cpu().to(torch.float32).contiguous()
    require(a.shape == b.shape, "PARITY_SHAPE_MISMATCH")
    error = (a - b).abs()
    return {
        "exact": bool(torch.equal(a, b)),
        "max_abs_error": float(error.max()),
        "tcp6_max_abs_error": float(error[..., :6].max()),
        "raw_gripper_max_abs_error": float(error[..., 6].max()),
        "left": tensor_record(a),
        "right": tensor_record(b),
    }


def current_git() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": head, "worktree_dirty": dirty}


def copy_runtime_payloads(parent: Path, destination: Path, relative_paths: list[str]) -> None:
    shutil.copytree(parent / "base_assets/smolvlm_constructor", destination / "base_assets/smolvlm_constructor")
    shutil.copy2(parent / "trainability_manifest.json", destination / "trainability_manifest.json")
    for relative in relative_paths:
        source = parent / relative
        require(source.is_file(), f"R5_RUNTIME_PAYLOAD_MISSING:{relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def validate_evaluation_export_scope(checkpoint: Path) -> dict[str, Any]:
    """Enforce evaluation-only semantics around the unchanged strict-loader container."""

    contract_path = checkpoint / "manifests/training_checkpoint_contract.development.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {
        "artifact_type": "forcesmolvla_training_checkpoint",
        "artifact_purpose": "evaluation_smoke_only",
        "deployment_release": False,
        "training_parent_allowed": False,
        "online_update_allowed": False,
        "robot_execution_authorized": "false_pending_offline_parity",
        "strict_loader_container_only": True,
    }
    require(
        all(contract.get(key) == value for key, value in required.items()),
        "EVALUATION_EXPORT_SCOPE_MISMATCH",
    )
    forbidden_roots = ("training_state", "optimizers", "schedulers", "state")
    forbidden = [
        path.relative_to(checkpoint).as_posix()
        for root in forbidden_roots
        for path in (checkpoint / root).rglob("*")
        if path.is_file()
    ]
    require(not forbidden, f"EVALUATION_EXPORT_TRAINING_PAYLOAD_PRESENT:{forbidden}")
    return contract


def export_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    from forcesmolvla.checkpoint import (
        validate_force_artifact_manifest,
        validate_training_payload_contract,
        write_development_artifact_manifest,
    )
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.long_run_checkpoint import validate_cycle_checkpoint

    parent = ROOT / config["runtime_parent"]
    source_checkpoint = ROOT / config["source_checkpoint"]
    source_actor = ROOT / config["source_actor_state"]
    destination = ROOT / config["output_checkpoint"]
    require(not destination.exists(), "EVALUATION_EXPORT_APPEND_ONLY_DESTINATION_EXISTS")
    source_manifest = validate_cycle_checkpoint(source_checkpoint, expected_cycle=210)
    source_entry = next(
        item for item in source_manifest["files"] if item["relative_path"] == "models/actor_state.pt"
    )
    require(sha256_file(source_actor) == source_entry["sha256"], "CYCLE210_ACTOR_STATE_SHA_MISMATCH")
    source_tree = tree_record(source_checkpoint)

    actor_state = torch.load(source_actor, map_location="cpu", weights_only=True)
    require(isinstance(actor_state, Mapping), "CYCLE210_ACTOR_STATE_NOT_MAPPING")
    actor_state = dict(actor_state)
    actor_record = tensor_state_record(actor_state)

    policy = ForceSmolVLAPolicy.from_pretrained(
        parent, local_files_only=True, strict=True, artifact_use="development"
    )
    r5_state = policy.state_dict()
    require(set(actor_state) == set(r5_state), "CYCLE210_ACTOR_STATE_NOT_COMPLETE")
    frozen_names = sorted(name for name in actor_state if name.startswith(FROZEN_PREFIXES))
    require(frozen_names, "CYCLE210_FROZEN_VLM_KEYS_MISSING")
    frozen_mismatch = [
        name
        for name in frozen_names
        if not torch.equal(actor_state[name].cpu(), r5_state[name].cpu())
    ]
    require(not frozen_mismatch, f"CYCLE210_FROZEN_VLM_DIFFERS_FROM_R5:{frozen_mismatch[:3]}")
    incompatible = policy.load_state_dict(actor_state, strict=True)
    require(not incompatible.missing_keys, "CYCLE210_ACTOR_MISSING_KEYS")
    require(not incompatible.unexpected_keys, "CYCLE210_ACTOR_UNEXPECTED_KEYS")
    require(
        all(torch.equal(policy.state_dict()[name].cpu(), value.cpu()) for name, value in actor_state.items()),
        "CYCLE210_ACTOR_POST_LOAD_TENSOR_MISMATCH",
    )
    policy.eval()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.new.", dir=destination.parent))
    try:
        policy.save_pretrained(temporary)
        shutil.copy2(parent / "config.json", temporary / "config.json")
        copy_runtime_payloads(parent, temporary, list(config["runtime_payloads"]))
        provenance = temporary / "provenance"
        provenance.mkdir()
        shutil.copy2(source_checkpoint / "checkpoint_manifest.json", provenance / "cycle210_checkpoint_manifest.json")
        shutil.copy2(parent / "artifact_manifest.json", provenance / "r5_parent_artifact_manifest.json")
        shutil.copy2(ROOT / config["action_contract_v2"], temporary / "manifests/stage2_action_contract.v2.development.json")

        exported_state = load_file(str(temporary / "model.safetensors"), device="cpu")
        require(set(exported_state) == set(actor_state), "EXPORTED_SAFETENSORS_KEY_MISMATCH")
        export_mismatch = [
            name for name in actor_state if not torch.equal(exported_state[name], actor_state[name])
        ]
        require(not export_mismatch, f"EXPORTED_SAFETENSORS_TENSOR_MISMATCH:{export_mismatch[:3]}")
        required = [
            "config.json",
            "model.safetensors",
            "base_assets/smolvlm_constructor",
            "trainability_manifest.json",
            *config["runtime_payloads"],
            "manifests/stage2_action_contract.v2.development.json",
            "manifests/training_checkpoint_contract.development.json",
            "provenance/cycle210_checkpoint_manifest.json",
            "provenance/r5_parent_artifact_manifest.json",
        ]
        runtime_records = [
            {
                "relative_path": relative,
                "sha256": sha256_file(parent / relative),
                "file_size": (parent / relative).stat().st_size,
            }
            for relative in config["runtime_payloads"]
        ]
        contract = {
            "schema_version": "forcesmolvla_evaluation_checkpoint_contract.v1",
            "acceptance_status": "development_only",
            "formal_eligible": False,
            "artifact_type": "forcesmolvla_training_checkpoint",
            "training_stage": "offline_full_finetune",
            "strict_loader_container_only": True,
            **config["restrictions"],
            "artifact_purpose": "evaluation_smoke_only",
            "runtime_parent": config["runtime_parent"],
            "weight_source": config["source_actor_state"],
            "required_payloads": required,
            "source_bindings": {
                "cycle210_checkpoint_tree_sha256": source_tree["tree_sha256"],
                "cycle210_actor_state_sha256": sha256_file(source_actor),
                "r5_config_sha256": sha256_file(parent / "config.json"),
                "r5_artifact_manifest_sha256": sha256_file(parent / "artifact_manifest.json"),
                "r5_runtime_manifest_closure_sha256": canonical_sha256(runtime_records),
                "normalizer_sha256": sha256_file(parent / "manifests/normalizer_manifest.json"),
                "action_contract_v2_sha256": sha256_file(ROOT / config["action_contract_v2"]),
                "source_code": current_git(),
            },
        }
        atomic_json(temporary / "manifests/training_checkpoint_contract.development.json", contract)
        write_development_artifact_manifest(
            temporary,
            artifact_type="forcesmolvla_training_checkpoint",
            metadata={
                "artifact_purpose": "evaluation_smoke_only",
                **config["restrictions"],
                "cycle": 210,
                "strict_load": {"missing_keys": 0, "unexpected_keys": 0},
                "actor_state_coverage": {
                    "source_tensor_count": len(actor_state),
                    "loaded_tensor_count": len(actor_state),
                    "coverage_fraction": 1.0,
                    "full_actor_state": True,
                },
                "frozen_vlm_parent_parity": {
                    "tensor_count": len(frozen_names),
                    "exact": True,
                },
                "source_bindings": contract["source_bindings"],
            },
        )
        validate_force_artifact_manifest(temporary, artifact_use="development")
        validate_training_payload_contract(temporary)
        validate_evaluation_export_scope(temporary)
        os.replace(temporary, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        del policy, r5_state, actor_state

    strict_policy = ForceSmolVLAPolicy.from_pretrained(
        destination, local_files_only=True, strict=True, artifact_use="development"
    )
    strict_policy.eval()
    strict_state_record = tensor_state_record(strict_policy.state_dict())
    del strict_policy
    return {
        "path": config["output_checkpoint"],
        "tree": tree_record(destination),
        "artifact_manifest_sha256": sha256_file(destination / "artifact_manifest.json"),
        "evaluation_contract_sha256": sha256_file(
            destination / "manifests/training_checkpoint_contract.development.json"
        ),
        "model_safetensors_sha256": sha256_file(destination / "model.safetensors"),
        "config_sha256": sha256_file(destination / "config.json"),
        "cycle210_checkpoint_tree": source_tree,
        "cycle210_actor_state_sha256": sha256_file(source_actor),
        "actor_state": actor_record,
        "strict_reload_state": strict_state_record,
        "strict_load": {"missing_keys": 0, "unexpected_keys": 0, "pass": True},
        "actor_state_is_complete": True,
        "frozen_vlm_parent_parity": {"tensor_count": len(frozen_names), "exact": True},
        "training_payloads_exported": False,
    }


def validate_existing_export(config: dict[str, Any]) -> dict[str, Any]:
    """Strictly revalidate an append-only export before resuming parity."""

    from forcesmolvla.checkpoint import (
        validate_force_artifact_manifest,
        validate_training_payload_contract,
    )
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.long_run_checkpoint import validate_cycle_checkpoint

    parent = ROOT / config["runtime_parent"]
    source_checkpoint = ROOT / config["source_checkpoint"]
    source_actor = ROOT / config["source_actor_state"]
    destination = ROOT / config["output_checkpoint"]
    require(destination.is_dir(), "EXISTING_EVALUATION_EXPORT_MISSING")
    source_manifest = validate_cycle_checkpoint(source_checkpoint, expected_cycle=210)
    source_entry = next(
        item for item in source_manifest["files"] if item["relative_path"] == "models/actor_state.pt"
    )
    require(sha256_file(source_actor) == source_entry["sha256"], "CYCLE210_ACTOR_STATE_SHA_MISMATCH")
    source_tree = tree_record(source_checkpoint)
    validate_force_artifact_manifest(destination, artifact_use="development")
    contract = validate_training_payload_contract(destination)
    validate_evaluation_export_scope(destination)
    require(
        contract["source_bindings"]["cycle210_checkpoint_tree_sha256"]
        == source_tree["tree_sha256"],
        "EXISTING_EXPORT_CYCLE210_TREE_BINDING_MISMATCH",
    )
    require(
        contract["source_bindings"]["cycle210_actor_state_sha256"]
        == sha256_file(source_actor),
        "EXISTING_EXPORT_ACTOR_BINDING_MISMATCH",
    )

    actor_state = dict(torch.load(source_actor, map_location="cpu", weights_only=True))
    exported_state = load_file(str(destination / "model.safetensors"), device="cpu")
    require(set(exported_state) == set(actor_state), "EXISTING_EXPORT_KEY_MISMATCH")
    mismatch = [
        name for name in actor_state if not torch.equal(exported_state[name], actor_state[name])
    ]
    require(not mismatch, f"EXISTING_EXPORT_TENSOR_MISMATCH:{mismatch[:3]}")
    frozen_names = sorted(name for name in actor_state if name.startswith(FROZEN_PREFIXES))
    r5_state = load_file(str(parent / "model.safetensors"), device="cpu")
    frozen_mismatch = [
        name for name in frozen_names if not torch.equal(actor_state[name], r5_state[name])
    ]
    require(not frozen_mismatch, f"EXISTING_EXPORT_FROZEN_VLM_MISMATCH:{frozen_mismatch[:3]}")
    actor_record = tensor_state_record(actor_state)
    del exported_state, r5_state, actor_state

    strict_policy = ForceSmolVLAPolicy.from_pretrained(
        destination, local_files_only=True, strict=True, artifact_use="development"
    )
    strict_policy.eval()
    strict_state_record = tensor_state_record(strict_policy.state_dict())
    del strict_policy
    torch.cuda.empty_cache()
    return {
        "path": config["output_checkpoint"],
        "tree": tree_record(destination),
        "artifact_manifest_sha256": sha256_file(destination / "artifact_manifest.json"),
        "evaluation_contract_sha256": sha256_file(
            destination / "manifests/training_checkpoint_contract.development.json"
        ),
        "model_safetensors_sha256": sha256_file(destination / "model.safetensors"),
        "config_sha256": sha256_file(destination / "config.json"),
        "cycle210_checkpoint_tree": source_tree,
        "cycle210_actor_state_sha256": sha256_file(source_actor),
        "actor_state": actor_record,
        "strict_reload_state": strict_state_record,
        "strict_load": {"missing_keys": 0, "unexpected_keys": 0, "pass": True},
        "actor_state_is_complete": True,
        "frozen_vlm_parent_parity": {"tensor_count": len(frozen_names), "exact": True},
        "training_payloads_exported": False,
        "append_only_existing_export_revalidated": True,
    }


def encoded_image(tensor: torch.Tensor) -> dict[str, Any]:
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    require(image.shape == (480, 640, 3), f"OFFLINE_IMAGE_SHAPE:{image.shape}")
    return {
        "encoding": "raw-uint8-base64",
        "shape": [480, 640, 3],
        "data": base64.b64encode(image.tobytes()).decode("ascii"),
    }


def scalar(sample: dict[str, Any], name: str, cast):
    return cast(sample[name].detach().cpu().item())


def build_request(contract, sample: dict[str, Any], *, request_id: str) -> dict[str, Any]:
    from forcesmolvla.inference import CLOCK_DOMAIN, PROTOCOL_VERSION

    t_ref_ns = scalar(sample, "provenance.tuple_host_monotonic_ns", int)
    stored_state_pose_age_ms = scalar(sample, "provenance.state_pose_age_ms", float)
    pose_receive_ns = t_ref_ns - int(round(stored_state_pose_age_ms * 1.0e6))
    camera1_receive_ns = scalar(sample, "provenance.camera1_receive_monotonic_ns", int)
    camera2_receive_ns = scalar(sample, "provenance.camera2_receive_monotonic_ns", int)
    action_ack_receive_ns = scalar(sample, "provenance.action_ack_receive_monotonic_ns", int)
    pose_source_ns = scalar(sample, "provenance.pose_source_stamp_ns", int)
    wrench_source_ns = scalar(sample, "provenance.wrench_raw_source_stamp_ns", int)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "chunk_id": request_id,
        "client_hostname": socket.gethostname(),
        "clock_domain_id": CLOCK_DOMAIN,
        "dataset_repo_id": contract.repo_id,
        "tool_profile_sha256": contract.tool_profile_sha256,
        "calibration_id": contract.calibration_id,
        "task": str(sample["task"]),
        "state7": sample["observation.state"].detach().cpu().to(torch.float64).tolist(),
        "wrench6": sample["observation.wrench"].detach().cpu().to(torch.float64).tolist(),
        "camera1": encoded_image(sample["observation.images.camera1"]),
        "camera2": encoded_image(sample["observation.images.camera2"]),
        "provenance": {
            "t_ref_ns": t_ref_ns,
            "tau0_ns": t_ref_ns,
            "pose_receive_monotonic_ns": pose_receive_ns,
            "state_pose_age_ms": (t_ref_ns - pose_receive_ns) / 1.0e6,
            "camera1_receive_monotonic_ns": camera1_receive_ns,
            "camera1_age_ms": (t_ref_ns - camera1_receive_ns) / 1.0e6,
            "camera2_receive_monotonic_ns": camera2_receive_ns,
            "camera2_age_ms": (t_ref_ns - camera2_receive_ns) / 1.0e6,
            "intercamera_skew_ms": abs(camera1_receive_ns - camera2_receive_ns) / 1.0e6,
            "gripper_receive_monotonic_ns": min(t_ref_ns, action_ack_receive_ns),
            "wrench_receive_monotonic_ns": t_ref_ns,
            "geometry_pose_source_stamp_ns": pose_source_ns,
            "wrench_raw_source_stamp_ns": wrench_source_ns,
            "wrench_filter_output_stamp_ns": scalar(
                sample, "provenance.wrench_filter_output_stamp_ns", int
            ),
            "geometry_pose_age_ms": (wrench_source_ns - pose_source_ns) / 1.0e6,
            "filter_warmup_complete": True,
            "wrench_geometry_valid": True,
            "session_id": "task2-cycle210-evaluation-smoke",
        },
    }


def infer_direct(policy, request, runtime, contract, device, *, seed: int, invalid_tail: int = 0):
    from forcesmolvla.inference import prepare_policy_inputs

    batch, context = prepare_policy_inputs(policy, request, runtime, contract, device)
    if invalid_tail:
        mask = context.action_valid_mask.clone()
        mask[:, -invalid_tail:] = False
        context = replace(
            context,
            action_valid_mask=mask,
            suffix_valid_mask=mask.clone(),
            chunk_id=(request["chunk_id"] + "-masked",),
        )
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        raw, public = policy._predict_action_chunks(batch, chunk_context=context, noise=seed)
    return raw.detach(), public.detach(), context.action_valid_mask.detach()


def infer_public(policy, request, runtime, contract, device, *, seed: int):
    from forcesmolvla.inference import prepare_policy_inputs

    batch, context = prepare_policy_inputs(policy, request, runtime, contract, device)
    captured: dict[str, torch.Tensor] = {}
    original = policy._predict_action_chunks

    def capture(*args, **kwargs):
        raw, public = original(*args, **kwargs)
        captured["raw"] = raw.detach().clone()
        captured["public_private"] = public.detach().clone()
        return raw, public

    policy._predict_action_chunks = capture
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            public = policy.predict_action_chunk(batch, chunk_context=context, noise=seed)
    finally:
        del policy._predict_action_chunks
    require(torch.equal(public, captured["public_private"]), "PUBLIC_WRAPPER_CHANGED_PRIVATE_OUTPUT")
    return captured["raw"], public.detach(), context.action_valid_mask.detach()


def http_infer_fixed_noise(engine, request: dict[str, Any], *, seed: int):
    from serve_policy import RequestHandler

    captured: dict[str, torch.Tensor] = {}
    original_private = engine.policy._predict_action_chunks
    original_public = engine.policy.predict_action_chunk

    def capture_private(*args, **kwargs):
        raw, public = original_private(*args, **kwargs)
        captured["raw"] = raw.detach().clone()
        captured["public"] = public.detach().clone()
        return raw, public

    def fixed_public(*args, **kwargs):
        require("noise" not in kwargs, "HTTP_ENGINE_UNEXPECTED_NOISE_ARGUMENT")
        return original_public(*args, noise=seed, **kwargs)

    engine.policy._predict_action_chunks = capture_private
    engine.policy.predict_action_chunk = fixed_public
    server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
    server.engine = engine
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps(request, separators=(",", ":")).encode()
        http_request = Request(
            f"http://127.0.0.1:{server.server_port}/infer",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with build_opener(ProxyHandler({})).open(http_request, timeout=120) as response:
            require(response.status == 200, f"HTTP_STATUS:{response.status}")
            payload = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        del engine.policy._predict_action_chunks
        del engine.policy.predict_action_chunk
    actions = torch.tensor(payload["actions"], dtype=torch.float32, device=engine.device).unsqueeze(0)
    require(torch.equal(actions, captured["public"]), "HTTP_JSON_FLOAT32_ROUNDTRIP_MISMATCH")
    return captured["raw"], actions, payload


def reconstruct_public(policy, raw: torch.Tensor, context) -> torch.Tensor:
    from forcesmolvla.action_delta import ActionDeltaProcessor, decode_binary_gripper_width

    normalized = raw.detach().cpu().to(torch.float32).numpy().astype(np.float64)
    delta = policy._runtime_artifacts.normalizer.delta_action7.inverse(normalized)
    delta = decode_binary_gripper_width(delta)
    state = context.raw_state_snapshot.detach().cpu().numpy().astype(np.float64)
    absolute = ActionDeltaProcessor.from_delta(delta, state)
    mask = context.action_valid_mask.detach().cpu().numpy()
    policy._action_safety_profile.validate_chunk(absolute, mask, state)
    absolute = np.where(mask[..., None], absolute, 0.0)
    return torch.from_numpy(np.ascontiguousarray(absolute)).to(torch.float32)


def client_source_sha256() -> str:
    mapping = {relative: sha256_file(CLIENT_ROOT / relative) for relative in CLIENT_SOURCE_FILES}
    return hashlib.sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_client_response(response: dict, request: dict) -> dict[str, Any]:
    session = json.loads((CLIENT_ROOT / "datasets/task2/session.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="forcesmolvla_cycle210_client_") as directory:
        payload = Path(directory) / "exchange.json"
        payload.write_text(
            json.dumps({"response": response, "request": request, "workspace": session["workspace"]}),
            encoding="utf-8",
        )
        validator = Path(directory) / "validate.py"
        validator.write_text(
            "import json,sys\n"
            "sys.path.insert(0, '/home/rlc123/fr3_client_ws/scripts')\n"
            "import deploy_forcesmolvla as m\n"
            "p=json.load(open(sys.argv[1], encoding='utf-8'))\n"
            "a=m.validate_response(p['response'],p['request'],p['workspace'])\n"
            "print(json.dumps({'shape':list(a.shape),'dtype':str(a.dtype),'finite':bool(__import__('numpy').isfinite(a).all())}))\n",
            encoding="utf-8",
        )
        command = (
            "source /opt/ros/humble/setup.bash; "
            "source /home/rlc123/fr3_client_ws/install/setup.bash; "
            "source /home/rlc123/fr3_client_ws/.venv/bin/activate; "
            f"python {validator} {payload}"
        )
        completed = subprocess.run(
            ["/bin/bash", "-lc", command], capture_output=True, text=True, timeout=60
        )
        require(completed.returncode == 0, f"DEPLOY_CLIENT_RESPONSE_REJECTED:{completed.stderr}")
        return json.loads(completed.stdout.strip().splitlines()[-1])


def create_binding(config: dict[str, Any], *, model_sha256: str) -> dict[str, Any]:
    from serve_policy import load_deployment_binding, source_tree_sha256

    destination = ROOT / config["evaluation_binding"]
    require(not destination.exists(), "EVALUATION_BINDING_APPEND_ONLY_DESTINATION_EXISTS")
    old = json.loads(
        (ROOT / "artifacts/development/live/task2_r5_live_deployment_binding.json").read_text(
            encoding="utf-8"
        )
    )
    rulespec = ROOT / config["live_rulespec"]
    server_sha = source_tree_sha256(ROOT)
    client_sha = client_source_sha256()
    require(server_sha == old["server_source_sha256"], "PUBLIC_SERVER_SOURCE_CHANGED")
    require(client_sha == old["client_source_sha256"], "DEPLOY_CLIENT_SOURCE_CHANGED")
    require(
        sha256_file(rulespec) == old["rulespec_sha256"],
        "PUBLIC_RULESPEC_CHANGED",
    )
    binding = {
        "schema_version": "forcesmolvla-live-deployment-binding-v1",
        "artifact_status": "approved",
        "model_sha256": model_sha256,
        "rulespec_sha256": sha256_file(rulespec),
        "server_source_sha256": server_sha,
        "client_source_sha256": client_sha,
        "state_pose_max_age_ms": old["state_pose_max_age_ms"],
        "camera_max_age_ms": old["camera_max_age_ms"],
        "max_intercamera_skew_ms": old["max_intercamera_skew_ms"],
        "gripper_max_age_ms": old["gripper_max_age_ms"],
        "controller_ack_timeout_ms": old["controller_ack_timeout_ms"],
        "approval": {
            "status": "approved",
            "approval_id": "forcesmolvla-cycle210-evaluation-smoke-20260828-001",
            "approver_identity": "rlc123",
            "approver_role": "experiment_lead",
            "approved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
                "+00:00", "Z"
            ),
        },
    }
    atomic_json(destination, binding)
    digest = sha256_file(destination)
    loaded, loaded_sha = load_deployment_binding(
        destination,
        digest,
        model_sha256=model_sha256,
        rulespec_sha256=sha256_file(rulespec),
        server_source_sha256=server_sha,
    )
    require(loaded == binding and loaded_sha == digest, "EVALUATION_BINDING_STRICT_RELOAD_FAILED")
    return {
        "path": config["evaluation_binding"],
        "sha256": digest,
        "model_sha256": model_sha256,
        "rulespec_sha256": binding["rulespec_sha256"],
        "server_source_sha256": server_sha,
        "client_source_sha256": client_sha,
        "scope": "evaluation_smoke_only_pending_physical_confirmation",
        "deployment_release": False,
    }


def parity(config: dict[str, Any], export: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from forcesmolvla.dataset_v3 import load_dataset_split
    from forcesmolvla.inference import load_checkpoint_inference_contract, prepare_policy_inputs
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rules import load_and_validate_rulespec
    from forcesmolvla.training_data import load_checkpoint_runtime_artifacts
    from serve_policy import InferenceEngine, bind_policy_action_safety

    device = torch.device("cuda")
    exported = ROOT / config["output_checkpoint"]
    parent = ROOT / config["runtime_parent"]
    source_actor = ROOT / config["source_actor_state"]
    contract = load_checkpoint_inference_contract(exported)
    runtime = load_checkpoint_runtime_artifacts(exported)
    dataset = load_dataset_split(
        ROOT / config["dataset_root"],
        repo_id=contract.repo_id,
        split_name=config["fixed_observation"]["split"],
        artifact_use="development",
        delta_timestamps={"action": [index / 30 for index in range(50)]},
    )
    sample = dataset[int(config["fixed_observation"]["dataset_index"])]
    require(int(sample["episode_index"]) == 0, "FIXED_OBSERVATION_EPISODE_MISMATCH")
    require(int(sample["frame_index"]) == 0, "FIXED_OBSERVATION_FRAME_MISMATCH")
    seed = int(config["fixed_observation"]["flow_noise_seed"])
    invalid_tail = int(config["fixed_observation"]["invalid_tail_slots"])
    rules_path = ROOT / config["live_rulespec"]
    rules = load_and_validate_rulespec(
        rules_path, ROOT / config["rulespec_schema"], formal=False
    )

    source_policy = ForceSmolVLAPolicy.from_pretrained(
        parent, local_files_only=True, strict=True, artifact_use="development"
    )
    source_state = torch.load(source_actor, map_location="cpu", weights_only=True)
    incompatible = source_policy.load_state_dict(source_state, strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys, "SOURCE_DIRECT_STRICT_LOAD")
    source_policy.bind_runtime_artifacts(runtime)
    bind_policy_action_safety(
        source_policy,
        rules,
        rules_sha256=sha256_file(rules_path),
        approved_development_execution=True,
    )
    source_policy.to(device).eval()
    request_a = build_request(contract, sample, request_id="cycle210-direct")
    raw_a, public_a, mask_a = infer_direct(
        source_policy, request_a, runtime, contract, device, seed=seed
    )
    request_a_mask = build_request(contract, sample, request_id="cycle210-direct-mask")
    raw_a_mask, public_a_mask, invalid_mask_a = infer_direct(
        source_policy,
        request_a_mask,
        runtime,
        contract,
        device,
        seed=seed,
        invalid_tail=invalid_tail,
    )
    del source_policy, source_state
    torch.cuda.empty_cache()

    engine = InferenceEngine(
        exported,
        ROOT / config["offline_rulespec"],
        ROOT / config["rulespec_schema"],
        device,
    )
    bind_policy_action_safety(
        engine.policy,
        rules,
        rules_sha256=sha256_file(rules_path),
        approved_development_execution=True,
    )
    state_before = tensor_state_record(engine.policy.state_dict())
    request_b = build_request(contract, sample, request_id="cycle210-public")
    raw_b, public_b, mask_b = infer_public(
        engine.policy, request_b, runtime, contract, device, seed=seed
    )
    request_b_mask = build_request(contract, sample, request_id="cycle210-public-mask")
    raw_b_mask, public_b_mask, invalid_mask_b = infer_direct(
        engine.policy,
        request_b_mask,
        runtime,
        contract,
        device,
        seed=seed,
        invalid_tail=invalid_tail,
    )
    engine.policy.reset()
    request_c = build_request(contract, sample, request_id="cycle210-http")
    raw_c, public_c, http_response = http_infer_fixed_noise(engine, request_c, seed=seed)
    state_after = tensor_state_record(engine.policy.state_dict())
    require(state_before == state_after, "PUBLIC_HTTP_INFERENCE_MUTATED_MODEL_STATE")

    raw_ab = exact_tensor_comparison(raw_a, raw_b)
    raw_ac = exact_tensor_comparison(raw_a, raw_c)
    public_ab = exact_tensor_comparison(public_a, public_b)
    public_ac = exact_tensor_comparison(public_a, public_c)
    require(raw_ab["exact"] and raw_ac["exact"], "DIRECT_PUBLIC_HTTP_RAW_PARITY_FAILED")
    require(public_ab["exact"] and public_ac["exact"], "DIRECT_PUBLIC_HTTP_ACTION_PARITY_FAILED")
    require(torch.equal(mask_a, mask_b), "DIRECT_PUBLIC_MASK_MISMATCH")
    require(torch.equal(invalid_mask_a, invalid_mask_b), "INVALID_MASK_PARITY_FAILED")
    require(torch.equal(raw_a_mask, raw_b_mask), "INVALID_MASK_RAW_PARITY_FAILED")
    require(torch.equal(public_a_mask, public_b_mask), "INVALID_MASK_PUBLIC_PARITY_FAILED")
    require(bool(torch.all(public_a_mask[:, -invalid_tail:] == 0)), "INVALID_TAIL_NOT_ZERO")
    reconstruction_batch, reconstruction_context = prepare_policy_inputs(
        engine.policy,
        build_request(contract, sample, request_id="cycle210-reconstruct"),
        runtime,
        contract,
        device,
    )
    del reconstruction_batch
    reconstructed = reconstruct_public(engine.policy, raw_b, reconstruction_context)
    require(torch.equal(reconstructed, public_b.cpu()), "PUBLIC_ACTION_RECONSTRUCTION_MISMATCH")
    grippers = sorted(set(float(value) for value in public_b[0, :, 6].cpu().tolist()))
    float32_endpoints = {
        float(torch.tensor(0.0, dtype=torch.float32)),
        float(torch.tensor(0.085, dtype=torch.float32)),
    }
    require(set(grippers).issubset(float32_endpoints), "PUBLIC_GRIPPER_NOT_BINARY")
    client = validate_client_response(http_response, request_c)
    require(client == {"shape": [50, 7], "dtype": "float64", "finite": True}, "DEPLOY_CLIENT_CHUNK_RESULT")

    result = {
        "fixed_observation": {
            **config["fixed_observation"],
            "row_identity": {
                "episode_index": int(sample["episode_index"]),
                "frame_index": int(sample["frame_index"]),
                "index": int(sample["index"]),
            },
            "manual_label_read": False,
        },
        "model_runtime": {
            "device": str(device),
            "eval_mode": not engine.policy.training,
            "optimizer_loaded": False,
            "critic_loaded": False,
            "training_worker_loaded": False,
            "parameter_state_unchanged": True,
            "state_digest": state_before,
        },
        "direct_public_http": {
            "raw_direct_vs_public": raw_ab,
            "raw_direct_vs_http": raw_ac,
            "public_direct_vs_public": public_ab,
            "public_direct_vs_http": public_ac,
            "binary_gripper_endpoints_m_contract": [0.0, 0.085],
            "binary_gripper_observed_float32_values": grippers,
            "binary_gripper_exact": True,
            "full_valid_mask_exact": bool(torch.equal(mask_a, mask_b)),
            "invalid_tail_mask_exact": True,
            "invalid_tail_zero": True,
            "all_outputs_finite": all(
                bool(torch.isfinite(value).all())
                for value in (raw_a, raw_b, raw_c, public_a, public_b, public_c)
            ),
            "public_action_reconstruction_exact": True,
            "http_shape": list(public_c.shape[1:]),
            "http_tensor_dtype_after_float32_deserialization": str(public_c.dtype),
            "http_serialization_exact": True,
            "fixed_noise_injection": "test_harness_only_at_public_policy_noise_argument",
        },
        "action_contract_v2": {
            "status": "pass",
            "horizon": int(raw_a.shape[1]),
            "raw_action_dim": int(raw_a.shape[2]),
            "public_action_dim": int(public_a.shape[2]),
            "normalizer_sha256": sha256_file(exported / "manifests/normalizer_manifest.json"),
            "contract_sha256": sha256_file(ROOT / config["action_contract_v2"]),
            "tcp6_denormalization_and_limits": "pass_exact_reconstruction_and_rulespec",
            "binary_gripper_projection": "pass",
            "public_safety_thresholds_changed": False,
        },
        "serve_deploy": {
            "http_handler": "tools/serve_policy.py:RequestHandler",
            "client_validator": "/home/rlc123/fr3_client_ws/scripts/deploy_forcesmolvla.py:validate_response",
            "client_result": client,
            "robot_connected": False,
            "persistent_server_started": False,
        },
    }
    require(result["direct_public_http"]["all_outputs_finite"], "PARITY_OUTPUT_NONFINITE")
    binding = create_binding(config, model_sha256=export["model_safetensors_sha256"])
    return result, binding


def source_manifest(config: dict[str, Any], export: dict, binding: dict, result_path: Path, report_path: Path) -> dict:
    entries = []
    for relative, role in (
        ("src/forcesmolvla/checkpoint.py", "unchanged_strict_checkpoint_loader"),
        ("src/forcesmolvla/modeling_forcesmolvla.py", "public_actor_runtime"),
        ("src/forcesmolvla/inference.py", "public_input_runtime"),
        ("src/forcesmolvla/action_delta.py", "public_action_contract_runtime"),
        ("tools/serve_policy.py", "unchanged_http_inference_server"),
        ("tools/export_stage2b_cycle210_evaluation_smoke.py", "evaluation_export_and_parity"),
        ("tests/test_stage2b_cycle210_evaluation_checkpoint.py", "evaluation_checkpoint_regression"),
        ("configs/stage2b_cycle210_evaluation_smoke.development.json", "resolved_export_config"),
        (config["action_contract_v2"], "action_contract_v2"),
        (config["live_rulespec"], "unchanged_live_rulespec"),
        (config["output_checkpoint"] + "/artifact_manifest.json", "export_artifact_manifest"),
        (config["output_checkpoint"] + "/model.safetensors", "exported_cycle210_actor"),
        (config["evaluation_binding"], "evaluation_smoke_binding"),
        (result_path.relative_to(ROOT).as_posix(), "offline_parity_result"),
        (report_path.relative_to(ROOT).as_posix(), "offline_parity_report"),
    ):
        path = ROOT / relative
        entries.append(
            {
                "relative_path": relative,
                "artifact_role": role,
                "sha256": sha256_file(path),
                "file_size": path.stat().st_size,
            }
        )
    entries.sort(key=lambda item: item["relative_path"])
    external = [
        {
            "path": str(CLIENT_ROOT / relative),
            "artifact_role": "unchanged_deploy_client_source",
            "sha256": sha256_file(CLIENT_ROOT / relative),
            "file_size": (CLIENT_ROOT / relative).stat().st_size,
        }
        for relative in CLIENT_SOURCE_FILES
    ]
    return {
        "schema_version": "forcesmolvla_stage2_source_manifest.v31.cycle210_evaluation_smoke",
        "artifact_status": "development_only",
        "scope": "cycle210_evaluation_smoke_export_and_offline_three_path_parity",
        "source_code": current_git(),
        "files": entries,
        "files_sha256": canonical_sha256(entries),
        "external_client_files": external,
        "external_client_files_sha256": canonical_sha256(external),
        "export_tree_sha256": export["tree"]["tree_sha256"],
        "evaluation_binding_sha256": binding["sha256"],
        "deployment_release": False,
        "robot_execution_authorized": "false_pending_physical_confirmation",
    }


def report_text(result: dict[str, Any]) -> str:
    binding = result["evaluation_binding"]
    export = result["export"]
    parity_result = result["parity"]
    return f"""# Stage-2B cycle210 evaluation-smoke export

Status: **PASS**. The full cycle210 Actor state was strictly overlaid on the unchanged r5 runtime/config and exported without Critic, optimizer, scheduler, RNG, or sampler payloads.

- Export: `{export['path']}`
- Exported model SHA-256: `{export['model_safetensors_sha256']}`
- Cycle210 source Actor SHA-256: `{export['cycle210_actor_state_sha256']}`
- Strict load: missing keys 0, unexpected keys 0
- Frozen VLM/state-prefix tensors equal r5: exact ({export['frozen_vlm_parent_parity']['tensor_count']} tensors)
- Direct / exported-public / serve HTTP raw action parity: exact
- Direct / exported-public / serve HTTP public action parity: exact
- H=50, 7D, finite, binary gripper endpoints, invalid-tail masking: pass
- Existing deploy client accepted the complete HTTP action chunk: pass
- Evaluation-smoke binding: `{binding['path']}`
- Binding SHA-256: `{binding['sha256']}`

No robot connection, training, online update, persistent service, deployment release, or policy-performance claim occurred. The evaluation binding remains scoped to one supervised smoke rollout pending physical confirmation.

## Commands for the later approved physical smoke

Server:

```bash
cd /home/rlc123/ForceSmolVLA
python tools/serve_policy.py \\
  --host 127.0.0.1 \\
  --port 8000 \\
  --checkpoint /home/rlc123/ForceSmolVLA/{export['path']} \\
  --rulespec /home/rlc123/ForceSmolVLA/configs/live_action_safety.task2.development.yaml \\
  --allow-development-robot-execution \\
  --deployment-binding /home/rlc123/ForceSmolVLA/{binding['path']} \\
  --trusted-deployment-binding-sha256 {binding['sha256']}
```

Client (do not run until workspace/estop/operator confirmation):

```bash
source /opt/ros/humble/setup.bash
source /home/rlc123/fr3_client_ws/install/setup.bash
source /home/rlc123/fr3_client_ws/.venv/bin/activate
cd /home/rlc123/fr3_client_ws
python scripts/deploy_forcesmolvla.py \\
  --host 127.0.0.1 \\
  --port 8000 \\
  --allow-development-robot-execution \\
  --trusted-deployment-binding-sha256 {binding['sha256']} \\
  --execute \\
  --duration 120
```

Parity evidence digest: `{canonical_sha256(parity_result)}`.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["artifact_purpose"] == "evaluation_smoke_only", "EXPORT_CONFIG_SCOPE")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        require(os.environ.get(name) == "1", f"{name}_REQUIRED")
    require(torch.cuda.is_available(), "CUDA_REQUIRED_NO_CPU_FALLBACK")
    torch.use_deterministic_algorithms(True)
    protected_before = {
        "cycle210_checkpoint_tree_sha256": tree_record(
            ROOT / config["source_checkpoint"]
        )["tree_sha256"],
        "r5_model_sha256": sha256_file(
            ROOT / config["runtime_parent"] / "model.safetensors"
        ),
        "r5_config_sha256": sha256_file(
            ROOT / config["runtime_parent"] / "config.json"
        ),
        "r5_artifact_manifest_sha256": sha256_file(
            ROOT / config["runtime_parent"] / "artifact_manifest.json"
        ),
        "r5_normalizer_sha256": sha256_file(
            ROOT / config["runtime_parent"] / "manifests/normalizer_manifest.json"
        ),
        "action_contract_v2_sha256": sha256_file(ROOT / config["action_contract_v2"]),
        "serve_policy_sha256": sha256_file(ROOT / "tools/serve_policy.py"),
        "deploy_forcesmolvla_sha256": sha256_file(
            CLIENT_ROOT / "scripts/deploy_forcesmolvla.py"
        ),
    }
    export_path = ROOT / config["output_checkpoint"]
    export = (
        validate_existing_export(config)
        if export_path.exists()
        else export_checkpoint(config)
    )
    parity_result, binding = parity(config, export)
    protected_after = {
        "cycle210_checkpoint_tree_sha256": tree_record(
            ROOT / config["source_checkpoint"]
        )["tree_sha256"],
        "r5_model_sha256": sha256_file(
            ROOT / config["runtime_parent"] / "model.safetensors"
        ),
        "r5_config_sha256": sha256_file(
            ROOT / config["runtime_parent"] / "config.json"
        ),
        "r5_artifact_manifest_sha256": sha256_file(
            ROOT / config["runtime_parent"] / "artifact_manifest.json"
        ),
        "r5_normalizer_sha256": sha256_file(
            ROOT / config["runtime_parent"] / "manifests/normalizer_manifest.json"
        ),
        "action_contract_v2_sha256": sha256_file(ROOT / config["action_contract_v2"]),
        "serve_policy_sha256": sha256_file(ROOT / "tools/serve_policy.py"),
        "deploy_forcesmolvla_sha256": sha256_file(
            CLIENT_ROOT / "scripts/deploy_forcesmolvla.py"
        ),
    }
    require(protected_before == protected_after, "PROTECTED_INPUT_CHANGED_DURING_EXPORT")
    result = {
        "schema_version": "forcesmolvla_stage2b_cycle210_evaluation_smoke.v1",
        "artifact_status": "development_only",
        "artifact_purpose": "evaluation_smoke_only",
        "status": "pass",
        "CYCLE210_EVALUATION_EXPORT": "pass",
        "ACTOR_STATE_STRICT_LOAD": "pass",
        "DIRECT_PUBLIC_HTTP_PARITY": "pass",
        "ACTION_CONTRACT_V2": "pass",
        "EVALUATION_BINDING_CREATED": "yes",
        "DEPLOYMENT_RELEASE_AUTHORIZED": "no",
        "ROBOT_EXECUTION_AUTHORIZED": "false_pending_physical_confirmation",
        "TRAINING_STARTED": "no",
        "ONLINE_UPDATE_STARTED": "no",
        "export": export,
        "parity": parity_result,
        "evaluation_binding": binding,
        "protected_inputs": {
            "before": protected_before,
            "after": protected_after,
            "unchanged": True,
        },
        "source_code": current_git(),
    }
    result_path = ROOT / config["result_artifact"]
    report_path = ROOT / config["report"]
    atomic_json(result_path, result)
    atomic_text(report_path, report_text(result))
    manifest = source_manifest(config, export, binding, result_path, report_path)
    manifest_path = ROOT / config["source_manifest"]
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "export": export["path"],
                "model_sha256": export["model_safetensors_sha256"],
                "binding": binding["path"],
                "binding_sha256": binding["sha256"],
                "result_sha256": sha256_file(result_path),
                "report_sha256": sha256_file(report_path),
                "source_manifest_sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
