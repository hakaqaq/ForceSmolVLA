#!/usr/bin/env python3
"""Fresh-process, CUDA-only, network-denied P8 strict reload verifier."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import socket
import sys


def _require_offline() -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 required")


def _files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite cold-start result: {args.output}")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("PYTHONHASHSEED=42 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required")
    _require_offline()

    cache_roots = [
        Path(os.environ[name])
        for name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")
        if os.environ.get(name)
    ]
    cache_before = {str(root): _files(root) for root in cache_roots}
    if any(cache_before.values()):
        raise RuntimeError("P8_COLD_START_CACHE_NOT_EMPTY")

    network_attempts: list[str] = []
    hub_attempts: list[str] = []

    def deny_connect(self, address):
        network_attempts.append(repr(address))
        raise RuntimeError(f"NETWORK_ACCESS_FORBIDDEN: {address}")

    def deny_hub(*args, **kwargs):
        hub_attempts.append(repr((args, kwargs)))
        raise RuntimeError("HUGGINGFACE_HUB_API_FORBIDDEN")

    socket.socket.connect = deny_connect

    import huggingface_hub
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")
    if "4090 D" not in torch.cuda.get_device_name(0) and "4090D" not in torch.cuda.get_device_name(0):
        raise RuntimeError("P8_REQUIRES_RTX_4090D")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    huggingface_hub.hf_hub_download = deny_hub
    huggingface_hub.snapshot_download = deny_hub
    import lerobot.configs.policies as lerobot_config_module
    import lerobot.policies.pretrained as lerobot_policy_module

    lerobot_config_module.hf_hub_download = deny_hub
    lerobot_policy_module.hf_hub_download = deny_hub

    from forcesmolvla.checkpoint import (
        load_p8_training_state,
        optimizer_state_sha256,
        validate_force_artifact_manifest,
    )
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.router_training import SerializableUniformSampler
    from forcesmolvla.router_training import build_p7_optimizer_and_scheduler
    from forcesmolvla.training_runtime import file_sha256 as _sha256
    from preflight_p8_checkpoint_gpu import (
        _validate_contract,
        _validate_p8_source_binding,
    )
    from p8_checkpoint_common import compute_fixed_parity, load_fixed_validation_inputs

    checkpoint = args.checkpoint.resolve()
    root = Path(__file__).parents[1].resolve()
    formal_rejection = None
    try:
        validate_force_artifact_manifest(checkpoint, artifact_use="formal")
    except RuntimeError as error:
        formal_rejection = str(error)
    if formal_rejection != "FORMAL_FORCE_CHECKPOINT_SIGNATURE_OR_APPROVAL_MISSING":
        raise RuntimeError("P8_FORMAL_FAIL_CLOSED_BEHAVIOR_DRIFT")
    torch.cuda.reset_peak_memory_stats()
    policy = ForceSmolVLAPolicy.from_pretrained(
        checkpoint,
        local_files_only=True,
        force_download=False,
        strict=True,
        artifact_use="development",
    )
    if not all(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("P8_COLD_START_FROZEN_PARAMETER_DETECTED")
    optimizer, scheduler, optimizer_groups = build_p7_optimizer_and_scheduler(policy)
    embedded_binding = json.loads(
        (checkpoint / "manifests/p8_source_binding.json").read_text(encoding="utf-8")
    )
    embedded_contract = json.loads(
        (checkpoint / "manifests/p8_checkpoint_contract.development.json").read_text(
            encoding="utf-8"
        )
    )
    _validate_contract(embedded_contract)
    conversion = json.loads(
        (args.dataset_root.resolve() / "conversion_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    repo_id = conversion.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("P8_COLD_DATASET_REPO_ID_MISSING")
    _validate_p8_source_binding(
        root,
        embedded_binding,
        dataset_root=args.dataset_root.resolve(),
        repo_id=repo_id,
        contract=embedded_contract,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sampler_payload = json.loads(
        (checkpoint / "training_state/sampler_state.json").read_text(encoding="utf-8")
    )
    sampler = SerializableUniformSampler(
        sampler_payload["eligible_indices"], seed=int(sampler_payload["seed"])
    )
    step, resume_contract = load_p8_training_state(
        checkpoint,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        batch_size=4,
        gradient_accumulation_microbatches=1,
        expected_resume_contract={
            "source_binding_sha256": _sha256(
                checkpoint / "manifests/p8_source_binding.json"
            ),
            "checkpoint_contract_sha256": _sha256(
                checkpoint / "manifests/p8_checkpoint_contract.development.json"
            ),
            "optimizer_groups": optimizer_groups,
            "gate_contract_version": "v4.2-b4x1-single-pass-exact-resume",
            "training_update_algorithm": "single_pass_batch_local",
        },
    )
    expected_rng = resume_contract["expected_next_rng"]
    actual_rng = {
        "python": random.random(),
        "numpy": float(np.random.rand()),
        "torch_cpu": torch.rand(4).tolist(),
        "torch_cuda": torch.rand(4, device="cuda").cpu().tolist(),
    }
    if actual_rng != expected_rng:
        raise RuntimeError("P8_RNG_CONTINUATION_MISMATCH")
    actual_sampler = sampler.draw(len(resume_contract["expected_next_sampler_indices"]))
    if actual_sampler != resume_contract["expected_next_sampler_indices"]:
        raise RuntimeError("P8_SAMPLER_CONTINUATION_MISMATCH")

    fixture = json.loads(
        (checkpoint / "manifests/p7_validation_fixture.json").read_text(encoding="utf-8")
    )
    batch, _raw_samples, normalizer = load_fixed_validation_inputs(
        policy, args.dataset_root.resolve(), fixture, torch.device("cuda:0")
    )
    reference = json.loads((checkpoint / "parity_reference.json").read_text(encoding="utf-8"))
    actual_parity = compute_fixed_parity(policy, batch, normalizer, fixture)
    if actual_parity != reference["parity"]:
        differing_keys = sorted(
            key
            for key in set(actual_parity) | set(reference["parity"])
            if actual_parity.get(key) != reference["parity"].get(key)
        )
        diagnostic = {
            "schema_version": "1.0",
            "acceptance_status": "development_only",
            "formal_eligible": False,
            "gate": "P8_cold_start",
            "gate_status": "fail",
            "reason": "P8_COLD_START_PARITY_MISMATCH",
            "differing_keys": differing_keys,
            "reference": reference["parity"],
            "actual": actual_parity,
            "robot_actions_sent": 0,
            "p9_started": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError("P8_COLD_START_PARITY_MISMATCH")

    noise = torch.tensor(fixture["epsilon7"]["tensor"], dtype=torch.float32, device="cuda")
    timestep = torch.tensor(fixture["time"]["tensor"], dtype=torch.float32, device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        losses, feature_mask, _router = policy.forward_single_pass_training_terms(
            batch, noise=noise, time=timestep
        )
        validation_scalar = float((losses.sum() / feature_mask.sum()).cpu())
    expected_scalar = fixture["evaluation"]["after_development_update_L_flow_run_1"]
    if validation_scalar != expected_scalar:
        raise RuntimeError("P8_FIXED_VALIDATION_SCALAR_MISMATCH")

    cache_after = {str(root): _files(root) for root in cache_roots}
    unexpected_cache_files = []
    for root, files in cache_after.items():
        for relative in files:
            if not relative.startswith("datasets/"):
                unexpected_cache_files.append(f"{root}/{relative}")
    if unexpected_cache_files or network_attempts or hub_attempts:
        raise RuntimeError("P8_LOCAL_ONLY_RELOAD_CONTRACT_VIOLATED")
    local_dataset_cache_file_count = sum(len(files) for files in cache_after.values())
    result = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "gate": "P8_cold_start",
        "gate_status": "pass",
        "fresh_process_pid": os.getpid(),
        "python_executable": sys.executable,
        "checkpoint": str(checkpoint),
        "strict": True,
        "local_files_only": True,
        "force_download": False,
        "empty_hf_cache_before": True,
        "hub_or_model_cache_files_after": 0,
        "local_dataset_cache_files_after": local_dataset_cache_file_count,
        "network_attempt_count": 0,
        "hub_api_attempt_count": 0,
        "formal_rejection": formal_rejection,
        "training_step": step,
        "training_stage": policy.config.training_stage,
        "all_parameters_trainable": True,
        "optimizer_state_sha256": optimizer_state_sha256(optimizer),
        "optimizer_groups": optimizer_groups,
        "scheduler_state": scheduler.state_dict(),
        "scaler_enabled": scaler.is_enabled(),
        "sampler_cursor_after_continuation": sampler.cursor,
        "rng_continuation_exact": True,
        "sampler_continuation_exact": True,
        "fixed_validation_L_flow": validation_scalar,
        "parity_sha256": actual_parity["parity_sha256"],
        "parity_exact": True,
        "exact_resume_dry_run": True,
        "gate_contract_version": "v4.2-b4x1-single-pass-exact-resume",
        "peak_memory": {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "cpu_fallback_used": False,
        "robot_actions_sent": 0,
        "p9_started": False,
        "detached_signature": None,
        "approval": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
