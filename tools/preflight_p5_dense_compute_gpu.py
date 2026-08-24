#!/usr/bin/env python3
"""CUDA-only real-data P5 gate; never downgrades the frozen architecture."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(_sha256(path).encode() + b"\n")
    return digest.hexdigest()


def _require_offline() -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 required")
    socket.socket.connect = lambda self, address: (_ for _ in ()).throw(
        RuntimeError(f"NETWORK_ACCESS_FORBIDDEN: {address}")
    )


def _validate_static_spec(spec: dict) -> None:
    expected = {
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "variant": "force_token_dense_compute",
        "training_stage": "offline_full_finetune",
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise RuntimeError(f"P5_SPEC_DRIFT: {key}={spec.get(key)!r}")
    if spec["rng"] != {
        "seed": 42,
        "python": 42,
        "numpy": 42,
        "torch_cpu": 42,
        "torch_cuda": 42,
    }:
        raise RuntimeError("P5_RNG_SPEC_DRIFT")
    if spec.get("optimizer_no_weight_decay") != [
        "bias",
        "normalization",
        "embedding",
        "alpha",
        "learned_action_slot",
    ]:
        raise RuntimeError("P5_OPTIMIZER_NO_WEIGHT_DECAY_SPEC_DRIFT")
    if spec["fusion_layout"] != {
        "camera1": [0, 64],
        "camera2": [64, 128],
        "language": [128, 176],
        "fusion_selection_indices": [0, 176],
        "state_token_excluded": True,
        "force_slot_index": 176,
        "N_fused_physical": 177,
        "segment_ids": {"camera1": 0, "camera2": 1, "language": 2, "force": 3},
    }:
        raise RuntimeError("P5_FUSION_LAYOUT_SPEC_DRIFT")
    cross = spec["force_cross_attention"]
    if (
        cross["type"] != "single_head_scaled_dot_product"
        or cross["num_heads"] != 1
        or cross["head_dim"] != 720
        or cross["internal_output_projection"] is not False
    ):
        raise RuntimeError("P5_SINGLE_HEAD_SPEC_DRIFT")


def _validate_p4_prerequisite(root: Path, spec: dict, binding: dict | None = None) -> dict:
    from forcesmolvla.acceptance import load_development_parity_threshold

    prerequisite = spec.get("p4_prerequisite")
    if not isinstance(prerequisite, dict) or set(prerequisite) != {
        "acceptance_config",
        "source_binding_sha256",
        "artifacts",
        "required_gate",
        "required_gate_status",
        "required_acceptance_status",
        "required_formal_eligible",
    }:
        raise RuntimeError("P5_P4_PREREQUISITE_SPEC_MISSING_OR_DRIFTED")
    if (
        prerequisite["required_gate"] != "P4_bare_smolvla_parity"
        or prerequisite["required_gate_status"] != "pass"
        or prerequisite["required_acceptance_status"] != "development_only"
        or prerequisite["required_formal_eligible"] is not False
        or set(prerequisite["artifacts"]) != {"fp32", "bf16"}
    ):
        raise RuntimeError("P5_P4_PREREQUISITE_SEMANTICS_DRIFT")
    acceptance = prerequisite["acceptance_config"]
    if set(acceptance) != {"path", "sha256"}:
        raise RuntimeError("P5_P4_ACCEPTANCE_BINDING_DRIFT")
    acceptance_path = root / acceptance["path"]
    if _sha256(acceptance_path) != acceptance["sha256"]:
        raise RuntimeError("P5_P4_ACCEPTANCE_CONFIG_HASH_MISMATCH")
    observed = {}
    for precision in ("fp32", "bf16"):
        threshold = load_development_parity_threshold(
            acceptance_path, gate="P4", precision=precision
        )
        if threshold.config_sha256 != acceptance["sha256"]:
            raise RuntimeError("P5_P4_ACCEPTANCE_LOADER_HASH_MISMATCH")
        artifact = prerequisite["artifacts"][precision]
        if set(artifact) != {"path", "sha256"}:
            raise RuntimeError("P5_P4_ARTIFACT_BINDING_DRIFT")
        artifact_path = root / artifact["path"]
        if _sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"P5_P4_{precision.upper()}_ARTIFACT_HASH_MISMATCH")
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        exact_contracts = payload.get("structural_contracts_exact")
        if (
            payload.get("gate") != prerequisite["required_gate"]
            or payload.get("gate_status") != prerequisite["required_gate_status"]
            or payload.get("acceptance_status")
            != prerequisite["required_acceptance_status"]
            or payload.get("formal_eligible")
            is not prerequisite["required_formal_eligible"]
            or payload.get("precision") != precision
            or payload.get("source_binding_sha256")
            != prerequisite["source_binding_sha256"]
            or payload.get("tolerance", {}).get("acceptance_config_sha256")
            != acceptance["sha256"]
            or payload.get("missing_keys") != []
            or payload.get("unexpected_keys") != []
            or not isinstance(exact_contracts, dict)
            or not exact_contracts
            or not all(value is True for value in exact_contracts.values())
        ):
            raise RuntimeError(f"P5_P4_{precision.upper()}_GATE_CONTENT_INVALID")
        observed[precision] = {
            "artifact_sha256": artifact["sha256"],
            "source_binding_sha256": payload["source_binding_sha256"],
            "max_abs_error": payload["max_abs_error"],
            "structural_contracts_exact": exact_contracts,
        }
    if binding is not None and binding.get("p4_prerequisite") != prerequisite:
        raise RuntimeError("P5_SOURCE_BINDING_P4_PREREQUISITE_MISMATCH")
    return {
        "acceptance_config_sha256": acceptance["sha256"],
        "source_binding_sha256": prerequisite["source_binding_sha256"],
        "artifacts": observed,
    }


def _validate_action_target_population_prerequisite(
    root: Path,
    dataset_root: Path,
    binding: dict | None = None,
) -> dict:
    artifact_relative = "artifacts/development/action_target_population_parity_r1.json"
    artifact_path = root / artifact_relative
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    required_assertions = {
        "valid_pair_count_exact",
        "ordered_pair_identity_exact",
        "ordered_pair_identity_sha256_exact",
        "action_target7_tensor_exact",
        "action_target7_tensor_sha256_exact",
        "global_and_per_horizon_statistics_exact",
        "padding_value_invariant",
        "val_test_zero_influence",
        "absolute_gripper_to_from_delta",
        "anchor_state_not_future_same_frame_state_sentinel",
        "normalizer_manifest_population_binding_exact",
        "conversion_manifest_fit_contract_exact",
    }
    assertions = payload.get("assertions")
    if (
        payload.get("gate") != "ActionTargetPopulationParityGate"
        or payload.get("gate_status") != "pass"
        or payload.get("acceptance_status") != "development_only"
        or payload.get("formal_eligible") is not False
        or payload.get("robot_actions_sent") != 0
        or payload.get("valid_pair_count") != 1_544_650
        or payload.get("forcevla_numeric_comparison_role")
        != "auxiliary_only_not_acceptance_oracle"
        or not isinstance(assertions, dict)
        or set(assertions) != required_assertions
        or not all(value is True for value in assertions.values())
        or payload.get("normalizer_manifest_sha256")
        != _sha256(dataset_root / "normalizer_manifest.json")
        or payload.get("conversion_manifest_sha256")
        != _sha256(dataset_root / "conversion_manifest.json")
        or payload.get("target_builder_source_sha256")
        != _sha256(root / "src/forcesmolvla/normalizer.py")
        or payload.get("oracle_source_sha256")
        != _sha256(root / "tools/action_target_population_parity_gate.py")
    ):
        raise RuntimeError("P5_ACTION_TARGET_POPULATION_PREREQUISITE_INVALID")
    observed = {
        "path": artifact_relative,
        "sha256": _sha256(artifact_path),
        "valid_pair_count": payload["valid_pair_count"],
        "ordered_pair_identity_sha256": payload["ordered_pair_identity_sha256"],
        "action_target7_float64_tensor_sha256": payload[
            "action_target7_float64_tensor_sha256"
        ],
        "statistics_sha256": payload["statistics_sha256"],
        "required_assertions": sorted(required_assertions),
    }
    if binding is not None and binding.get(
        "action_target_population_prerequisite"
    ) != observed:
        raise RuntimeError("P5_SOURCE_BINDING_ACTION_TARGET_POPULATION_MISMATCH")
    return observed


def _validate_source_binding(
    root: Path,
    binding: dict,
    *,
    dataset_root: Path | None = None,
    repo_id: str | None = None,
) -> None:
    if (
        binding.get("schema_version") != "1.0"
        or binding.get("status") != "development_only"
        or binding.get("detached_signature") is not None
    ):
        raise RuntimeError("P5_SOURCE_BINDING_STATUS_INVALID")
    vendor_root = root / "vendor/lerobot"
    commit = subprocess.run(
        ["git", "-C", str(vendor_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(vendor_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if commit != binding["lerobot"]["commit"] or dirty:
        raise RuntimeError("P5_LEROBOT_COMMIT_OR_CLEANLINESS_MISMATCH")
    for relative, expected in binding["lerobot"]["files"].items():
        actual = _sha256(vendor_root / relative)
        if actual != expected:
            raise RuntimeError(f"P5_LEROBOT_SOURCE_HASH_MISMATCH: {relative}")
    for relative, expected in binding["forcesmolvla_files"].items():
        actual = _sha256(root / relative)
        if actual != expected:
            raise RuntimeError(f"P5_PROJECT_SOURCE_HASH_MISMATCH: {relative}")
    if "base_assets" in binding:
        base = binding["base_assets"]
        for relative, expected in base["base_checkpoint_files"].items():
            if _sha256(root / "assets/base_checkpoint" / relative) != expected:
                raise RuntimeError(f"P5_BASE_ASSET_HASH_MISMATCH: {relative}")
        if (
            _tree_sha256(root / "assets/smolvlm_constructor")
            != base["constructor_tree_sha256"]
        ):
            raise RuntimeError("P5_CONSTRUCTOR_ASSET_TREE_HASH_MISMATCH")
    if dataset_root is not None or repo_id is not None:
        if dataset_root is None or repo_id is None or "dataset" not in binding:
            raise RuntimeError("P5_DATASET_BINDING_MISSING")
        dataset = binding["dataset"]
        if dataset.get("repo_id") != repo_id:
            raise RuntimeError("P5_DATASET_REPO_ID_MISMATCH")
        for relative, expected in dataset["manifest_files"].items():
            if _sha256(dataset_root / relative) != expected:
                raise RuntimeError(f"P5_DATASET_MANIFEST_HASH_MISMATCH: {relative}")


def _gradient_summary(named_parameters: list[tuple[str, object]]) -> dict:
    import torch

    summary = {}
    for group_name, predicate in (
        ("all", lambda name: True),
        ("base", lambda name: not name.startswith(("model.force_branch.", "model.force_adapter."))),
        ("force", lambda name: name.startswith(("model.force_branch.", "model.force_adapter."))),
        ("vision", lambda name: name.startswith("model.vlm_with_expert.vlm.model.vision_model.")),
        ("vlm_text", lambda name: name.startswith("model.vlm_with_expert.vlm.model.text_model.")),
        ("action_expert", lambda name: name.startswith("model.vlm_with_expert.lm_expert.")),
        (
            "action_io",
            lambda name: name.startswith(
                (
                    "model.state_proj.",
                    "model.action_in_proj.",
                    "model.action_out_proj.",
                    "model.action_time_mlp_",
                )
            ),
        ),
    ):
        selected = [(name, parameter) for name, parameter in named_parameters if predicate(name)]
        with_grad = [(name, parameter) for name, parameter in selected if parameter.grad is not None]
        nonzero = [
            name
            for name, parameter in with_grad
            if bool(torch.count_nonzero(parameter.grad.detach()).item())
        ]
        summary[group_name] = {
            "parameter_tensors": len(selected),
            "parameter_elements": sum(parameter.numel() for _, parameter in selected),
            "with_gradient_tensors": len(with_grad),
            "nonzero_gradient_tensors": len(nonzero),
            "coverage_with_gradient": len(with_grad) / len(selected) if selected else 1.0,
            "coverage_nonzero_gradient": len(nonzero) / len(selected) if selected else 1.0,
            "missing_gradient_names": [name for name, parameter in selected if parameter.grad is None],
            "zero_gradient_names": [
                name
                for name, parameter in with_grad
                if not bool(torch.count_nonzero(parameter.grad.detach()).item())
            ],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-output", type=Path, required=True)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).parents[1] / "configs/p5_force_token_dense_compute.development.json",
    )
    parser.add_argument(
        "--source-binding",
        type=Path,
        default=Path(__file__).parents[1] / "artifacts/development/source_binding.json",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("P5 gate requires at least two optimizer steps for post-zero-init coverage")
    if args.output.exists() or args.resolved_output.exists():
        raise FileExistsError("refusing to overwrite a P5 gate artifact")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("PYTHONHASHSEED=42 required before interpreter startup")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _require_offline()

    import numpy as np
    import torch

    from forcesmolvla.checkpoint import load_offline_base_policy
    from forcesmolvla.configuration_forcesmolvla import (
        CAMERA1,
        CAMERA2,
        FORCE_TOKEN_DENSE_COMPUTE,
        OFFLINE_FULL_FINETUNE,
    )
    from forcesmolvla.dataset_v3 import load_dataset_split
    from forcesmolvla.training_data import load_runtime_artifacts, prepare_training_sample
    from forcesmolvla.router_training import _no_decay_parameter_names
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")
    gpu_name = torch.cuda.get_device_name(0)
    if "4090 D" not in gpu_name and "4090D" not in gpu_name:
        raise RuntimeError(f"P5_REQUIRES_RTX_4090D: got {gpu_name!r}")

    root = Path(__file__).parents[1].resolve()
    spec_path = args.spec.resolve()
    source_binding_path = args.source_binding.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source_binding = json.loads(source_binding_path.read_text(encoding="utf-8"))
    _validate_static_spec(spec)
    p4_prerequisite = _validate_p4_prerequisite(root, spec, source_binding)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset_root = args.dataset_root.resolve()
    conversion_manifest = json.loads(
        (dataset_root / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    repo_id = conversion_manifest.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("P5_CONVERSION_REPO_ID_MISSING")
    action_target_population_prerequisite = (
        _validate_action_target_population_prerequisite(
            root, dataset_root, source_binding
        )
    )
    _validate_source_binding(
        root, source_binding, dataset_root=dataset_root, repo_id=repo_id
    )
    dataset = load_dataset_split(
        dataset_root,
        repo_id=repo_id,
        split_name="train",
        artifact_use="development",
        delta_timestamps={"action": [index / 30 for index in range(50)]},
    )
    batch_size = 4
    sample_indices = list(range(args.sample_index, args.sample_index + batch_size))
    if args.sample_index < 0 or sample_indices[-1] >= len(dataset):
        raise IndexError("P5 requires four consecutive samples inside the train split")
    runtime_artifacts = load_runtime_artifacts(
        dataset_root,
        calibration_bundle_path=root / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=root / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=root / "artifacts/development/action_delta_spec.json",
        expected_repo_id=repo_id,
    )
    prepared = [
        prepare_training_sample(dataset[index], runtime_artifacts.normalizer)
        for index in sample_indices
    ]

    device = torch.device("cuda:0")
    with contextlib.redirect_stdout(sys.stderr):
        policy, base_report = load_offline_base_policy(
            root / "assets/base_checkpoint",
            root / "assets/smolvlm_constructor",
            device="cuda",
            training_stage=OFFLINE_FULL_FINETUNE,
            force_variant=FORCE_TOKEN_DENSE_COMPUTE,
            acceptance_status="development_only",
            force_init_seed=seed,
        )
    initial_tensor_hash = policy.force_initialization_tensor_hash()
    if not initial_tensor_hash:
        raise RuntimeError("P5_INITIALIZATION_HASH_MISSING")
    if tuple(base_report.missing_keys) != policy.force_initialization_state_keys():
        raise RuntimeError("P5_BASE_LOAD_MISSING_KEY_ALLOWLIST_DRIFT")

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    encoded = tokenizer(
        [sample["task"] + "\n" for sample in prepared],
        padding="max_length",
        truncation=True,
        max_length=48,
        return_tensors="pt",
    )
    batch = {
        CAMERA1: torch.stack([sample["camera1"] for sample in prepared]).to(device),
        CAMERA2: torch.stack([sample["camera2"] for sample in prepared]).to(device),
        "observation.state": torch.from_numpy(
            np.stack([sample["state7"] for sample in prepared])
        ).to(device),
        "observation.wrench": torch.from_numpy(
            np.stack([sample["wrench6"] for sample in prepared])
        ).to(device),
        ACTION: torch.from_numpy(
            np.stack([sample["delta_action7"] for sample in prepared])
        ).to(device),
        "action_valid_mask": torch.from_numpy(
            np.stack([sample["action_valid_mask"] for sample in prepared])
        ).to(device),
        OBS_LANGUAGE_TOKENS: encoded["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: encoded["attention_mask"].to(device=device, dtype=torch.bool),
    }
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    noise = torch.randn(
        batch_size, 50, 7, generator=generator, device=device, dtype=torch.float32
    )
    timestep = torch.tensor([0.2, 0.4, 0.6, 0.8], device=device, dtype=torch.float32)

    policy.train()
    named = list(policy.named_parameters())
    frozen = [name for name, parameter in named if not parameter.requires_grad]
    if frozen:
        raise RuntimeError(f"P5_OFFLINE_FULL_FINETUNE_FROZEN_PARAMETERS: {frozen}")
    if not policy.model.vlm_with_expert.vlm.training:
        raise RuntimeError("P5_VLM_NOT_IN_TRAIN_MODE")
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    named_trainable = dict(trainable)
    no_decay_names = _no_decay_parameter_names(policy)
    decay_names = set(named_trainable) - no_decay_names
    if decay_names & no_decay_names or decay_names | no_decay_names != set(named_trainable):
        raise RuntimeError("P5_OPTIMIZER_GROUP_PARTITION_INVALID")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [named_trainable[name] for name in sorted(decay_names)],
                "weight_decay": 1e-10,
            },
            {
                "params": [named_trainable[name] for name in sorted(no_decay_names)],
                "weight_decay": 0.0,
            },
        ],
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if (
        len(optimizer_parameters) != len(trainable)
        or len({id(parameter) for parameter in optimizer_parameters}) != len(trainable)
        or {id(parameter) for parameter in optimizer_parameters}
        != {id(parameter) for _, parameter in trainable}
    ):
        raise RuntimeError("P5_OPTIMIZER_PARAMETER_PARTITION_INVALID")
    optimizer_group_digest = hashlib.sha256()
    for group, names in (("decay", decay_names), ("no_decay", no_decay_names)):
        for name in sorted(names):
            optimizer_group_digest.update(f"{group}\0{name}\n".encode())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    step_reports = []
    total_start = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        after_forward = torch.cuda.Event(enable_timing=True)
        after_backward = torch.cuda.Event(enable_timing=True)
        after_step = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(batch, noise=noise, time=timestep)
        after_forward.record()
        if not torch.isfinite(loss):
            raise FloatingPointError("P5_NONFINITE_LOSS")
        loss.backward()
        after_backward.record()
        wout_grad = policy.model.force_adapter.w_out.weight.grad
        if wout_grad is None or not torch.count_nonzero(wout_grad):
            raise RuntimeError("P5_W_OUT_GRADIENT_MISSING_OR_ZERO")
        coverage = _gradient_summary(trainable)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], max_norm=10.0
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("P5_NONFINITE_GRADIENT_NORM")
        optimizer.step()
        after_step.record()
        torch.cuda.synchronize(device)
        step_reports.append(
            {
                "step": step + 1,
                "gradient_source": "L_flow_only",
                "loss": float(loss.detach().cpu()),
                "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                "forward_ms": start.elapsed_time(after_forward),
                "backward_ms": after_forward.elapsed_time(after_backward),
                "optimizer_step_ms": after_backward.elapsed_time(after_step),
                "gradient_coverage": coverage,
            }
        )
    wall_seconds = time.perf_counter() - total_start
    if step_reports[-1]["gradient_coverage"]["force"]["coverage_nonzero_gradient"] < 1.0:
        raise RuntimeError("P5_FORCE_GRADIENT_COVERAGE_INCOMPLETE_AFTER_ZERO_INIT_STEP")
    final_coverage = step_reports[-1]["gradient_coverage"]
    for group_name in ("vision", "vlm_text", "action_expert", "action_io"):
        if final_coverage[group_name]["coverage_nonzero_gradient"] != 1.0:
            raise RuntimeError(f"P5_BASE_GRADIENT_COVERAGE_INCOMPLETE:{group_name}")
    if (
        final_coverage["base"]["missing_gradient_names"]
        != ["model.vlm_with_expert.vlm.lm_head.weight"]
        or final_coverage["base"]["zero_gradient_names"]
    ):
        raise RuntimeError("P5_BASE_GRADIENT_EXCEPTION_SET_DRIFT")

    total_parameters = sum(parameter.numel() for _, parameter in named)
    force_parameters = sum(
        parameter.numel()
        for name, parameter in named
        if name.startswith(("model.force_branch.", "model.force_adapter."))
    )
    source_binding_sha256 = _sha256(source_binding_path)
    spec_sha256 = _sha256(spec_path)
    resolved = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "variant": "force_token_dense_compute",
        "training_stage": "offline_full_finetune",
        "source_binding_sha256": source_binding_sha256,
        "static_spec_sha256": spec_sha256,
        "initialization_tensor_sha256": initial_tensor_hash,
        "initialization_seed": seed,
        "p4_prerequisite": p4_prerequisite,
        "action_target_population_prerequisite": action_target_population_prerequisite,
        "dimensions": spec["dimensions"],
        "fusion_layout": spec["fusion_layout"],
        "force_cross_attention": spec["force_cross_attention"],
        "total_parameters": total_parameters,
        "trainable_parameters": total_parameters,
        "frozen_parameters": 0,
        "force_parameters": force_parameters,
        "detached_signature": None,
        "approval": None,
    }
    result = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "production_shadow_eligible": False,
        "gate": "P5",
        "gate_status": "pass",
        "cpu_fallback_used": False,
        "architecture_downgrade_used": False,
        "gpu": {
            "name": gpu_name,
            "index": 0,
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "real_batch": {
            "dataset": str(dataset_root),
            "repo_id": repo_id,
            "split": "train",
            "sample_indices": sample_indices,
            "batch_sha256": [sample["batch_sha256"] for sample in prepared],
            "batch_size": batch_size,
            "camera_count": 2,
            "horizon": 50,
            "action_physical_dim": 7,
            "wrench_dim": 6,
        },
        "rng": {
            "pythonhashseed": os.environ["PYTHONHASHSEED"],
            "python": seed,
            "numpy": seed,
            "torch_cpu": seed,
            "torch_cuda": seed,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        "base_load": base_report.to_dict(),
        "resolved_config_sha256": None,
        "source_binding_sha256": source_binding_sha256,
        "static_spec_sha256": spec_sha256,
        "initialization_tensor_sha256": initial_tensor_hash,
        "p4_prerequisite": p4_prerequisite,
        "action_target_population_prerequisite": action_target_population_prerequisite,
        "gradient_source_audit": {
            "step_1": "L_flow_only_to_W_out_nonzero",
            "step_2": "L_flow_only_to_force_upstream_and_full_base_nonzero",
            "router_auxiliary_losses_present": False,
        },
        "optimizer_parameter_partition": {
            "trainable_tensor_count": len(trainable),
            "optimizer_tensor_count": len(optimizer_parameters),
            "each_trainable_parameter_exactly_once": True,
            "decay_tensor_count": len(decay_names),
            "no_decay_tensor_count": len(no_decay_names),
            "group_name_sha256": optimizer_group_digest.hexdigest(),
            "learned_action_slot_weight_decay": 0.0,
        },
        "steps": step_reports,
        "peak_memory": {
            "allocated_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "wall_seconds": wall_seconds,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "remaining_formal_blockers": [
            "trusted detached signature algorithm/key/approver unresolved",
            "formal threshold approvals unresolved",
            "P6-P9 gates not executed"
        ],
    }
    if _validate_p4_prerequisite(root, spec, source_binding) != p4_prerequisite:
        raise RuntimeError("P5_P4_PREREQUISITE_CHANGED_DURING_PREFLIGHT")
    if (
        _validate_action_target_population_prerequisite(
            root, dataset_root, source_binding
        )
        != action_target_population_prerequisite
    ):
        raise RuntimeError("P5_ACTION_TARGET_POPULATION_CHANGED_DURING_PREFLIGHT")
    args.resolved_output.parent.mkdir(parents=True, exist_ok=True)
    args.resolved_output.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    result["resolved_config_sha256"] = _sha256(args.resolved_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
