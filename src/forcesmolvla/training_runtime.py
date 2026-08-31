"""Reusable ForceSmolVLA training and validation runtime primitives."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Any


def resolve_task_output_root(
    repository_root: Path,
    *,
    task_id: str,
    output_root: Path | None = None,
) -> Path:
    """Return the sole task-scoped training output root."""

    task_id = task_id.strip()
    if not task_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in task_id):
        raise ValueError("TASK_ID_INVALID")
    selected = repository_root / "outputs" / task_id if output_root is None else output_root
    return Path(selected).expanduser().resolve()


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


def validate_source_binding(
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
        if file_sha256(vendor_root / relative) != expected:
            raise RuntimeError(f"P5_LEROBOT_SOURCE_HASH_MISMATCH: {relative}")
    for relative, expected in binding["forcesmolvla_files"].items():
        if file_sha256(root / relative) != expected:
            raise RuntimeError(f"P5_PROJECT_SOURCE_HASH_MISMATCH: {relative}")
    if "base_assets" in binding:
        base = binding["base_assets"]
        for relative, expected in base["base_checkpoint_files"].items():
            if file_sha256(root / "assets/base_checkpoint" / relative) != expected:
                raise RuntimeError(f"P5_BASE_ASSET_HASH_MISMATCH: {relative}")
        if tree_sha256(root / "assets/smolvlm_constructor") != base["constructor_tree_sha256"]:
            raise RuntimeError("P5_CONSTRUCTOR_ASSET_TREE_HASH_MISMATCH")
    if dataset_root is not None or repo_id is not None:
        if dataset_root is None or repo_id is None or "dataset" not in binding:
            raise RuntimeError("P5_DATASET_BINDING_MISSING")
        dataset = binding["dataset"]
        if dataset.get("repo_id") != repo_id:
            raise RuntimeError("P5_DATASET_REPO_ID_MISMATCH")
        for relative, expected in dataset["manifest_files"].items():
            if file_sha256(dataset_root / relative) != expected:
                raise RuntimeError(f"P5_DATASET_MANIFEST_HASH_MISMATCH: {relative}")


def validate_dense_compute_spec(spec: dict) -> None:
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


def validate_action_target_prerequisite(
    root: Path, spec: dict, binding: dict | None = None
) -> dict:
    from forcesmolvla.acceptance import load_development_parity_threshold

    prerequisite = spec.get("p4_prerequisite")
    required = {
        "acceptance_config",
        "source_binding_sha256",
        "artifacts",
        "required_gate",
        "required_gate_status",
        "required_acceptance_status",
        "required_formal_eligible",
    }
    if not isinstance(prerequisite, dict) or set(prerequisite) != required:
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
    if file_sha256(acceptance_path) != acceptance["sha256"]:
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
        if file_sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"P5_P4_{precision.upper()}_ARTIFACT_HASH_MISMATCH")
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        exact = payload.get("structural_contracts_exact")
        if (
            payload.get("gate") != prerequisite["required_gate"]
            or payload.get("gate_status") != prerequisite["required_gate_status"]
            or payload.get("acceptance_status") != prerequisite["required_acceptance_status"]
            or payload.get("formal_eligible") is not prerequisite["required_formal_eligible"]
            or payload.get("precision") != precision
            or payload.get("source_binding_sha256") != prerequisite["source_binding_sha256"]
            or payload.get("tolerance", {}).get("acceptance_config_sha256")
            != acceptance["sha256"]
            or payload.get("missing_keys") != []
            or payload.get("unexpected_keys") != []
            or not isinstance(exact, dict)
            or not exact
            or not all(value is True for value in exact.values())
        ):
            raise RuntimeError(f"P5_P4_{precision.upper()}_GATE_CONTENT_INVALID")
        observed[precision] = {
            "artifact_sha256": artifact["sha256"],
            "source_binding_sha256": payload["source_binding_sha256"],
            "max_abs_error": payload["max_abs_error"],
            "structural_contracts_exact": exact,
        }
    if binding is not None and binding.get("p4_prerequisite") != prerequisite:
        raise RuntimeError("P5_SOURCE_BINDING_P4_PREREQUISITE_MISMATCH")
    return {
        "acceptance_config_sha256": acceptance["sha256"],
        "source_binding_sha256": prerequisite["source_binding_sha256"],
        "artifacts": observed,
    }


def validate_action_target_population_prerequisite(
    root: Path, dataset_root: Path, binding: dict | None = None
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
        != file_sha256(dataset_root / "normalizer_manifest.json")
        or payload.get("conversion_manifest_sha256")
        != file_sha256(dataset_root / "conversion_manifest.json")
        or payload.get("target_builder_source_sha256")
        != file_sha256(root / "src/forcesmolvla/normalizer.py")
        or payload.get("oracle_source_sha256")
        != file_sha256(root / "tools/action_target_population_parity_gate.py")
    ):
        raise RuntimeError("P5_ACTION_TARGET_POPULATION_PREREQUISITE_INVALID")
    observed = {
        "path": artifact_relative,
        "sha256": file_sha256(artifact_path),
        "valid_pair_count": payload["valid_pair_count"],
        "ordered_pair_identity_sha256": payload["ordered_pair_identity_sha256"],
        "action_target7_float64_tensor_sha256": payload[
            "action_target7_float64_tensor_sha256"
        ],
        "statistics_sha256": payload["statistics_sha256"],
        "required_assertions": sorted(required_assertions),
    }
    if binding is not None and binding.get("action_target_population_prerequisite") != observed:
        raise RuntimeError("P5_SOURCE_BINDING_ACTION_TARGET_POPULATION_MISMATCH")
    return observed


def gradient_summary(named_parameters: list[tuple[str, object]]) -> dict:
    import torch

    summary = {}
    groups = (
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
    )
    for group_name, predicate in groups:
        selected = [(name, value) for name, value in named_parameters if predicate(name)]
        with_grad = [(name, value) for name, value in selected if value.grad is not None]
        nonzero = [
            name
            for name, value in with_grad
            if bool(torch.count_nonzero(value.grad.detach()).item())
        ]
        summary[group_name] = {
            "parameter_tensors": len(selected),
            "parameter_elements": sum(value.numel() for _, value in selected),
            "with_gradient_tensors": len(with_grad),
            "nonzero_gradient_tensors": len(nonzero),
            "coverage_with_gradient": len(with_grad) / len(selected) if selected else 1.0,
            "coverage_nonzero_gradient": len(nonzero) / len(selected) if selected else 1.0,
            "missing_gradient_names": [name for name, value in selected if value.grad is None],
            "zero_gradient_names": [
                name
                for name, value in with_grad
                if not bool(torch.count_nonzero(value.grad.detach()).item())
            ],
        }
    return summary


def build_training_batch(policy: Any, prepared_samples: list[dict], device: Any) -> dict:
    import numpy as np
    import torch

    from forcesmolvla.configuration_forcesmolvla import CAMERA1, CAMERA2
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

    from forcesmolvla.context import ChunkContext

    normalizer_sha = file_sha256(dataset_root / "normalizer_manifest.json")
    calibration_sha = file_sha256(root / "configs/calibration_bundle.development.json")
    geometry_sha = file_sha256(root / "configs/wrench_geometry_spec.development.json")
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
        "chunk_id": [
            f"sft-validation-{index}" for index in range(len(raw_samples))
        ],
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


def validate_training_recipe(recipe: dict) -> None:
    if (
        recipe.get("acceptance_status") != "development_only"
        or recipe.get("formal_eligible") is not False
        or recipe.get("execution_role") != "single_pass_gate_with_exact_two_pass_oracle"
        or recipe.get("long_running_sft_allowed") is not False
        or recipe.get("training_stage") != "offline_full_finetune"
        or recipe.get("all_existing_parameters_require_grad") is not True
        or recipe.get("variants") != ["force_token_moe", "force_token_moe_additive"]
    ):
        raise RuntimeError("P7_RECIPE_STATUS_DRIFT")
    if recipe["optimizer"] != {
        "type": "AdamW",
        "lr": 0.0001,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 1e-10,
        "no_decay": [
            "bias",
            "normalization_scale",
            "embedding",
            "alpha",
            "learned_action_slot",
        ],
        "parameter_partition": "each_trainable_parameter_exactly_once",
        "grad_clip_norm": 10.0,
    }:
        raise RuntimeError("P7_OPTIMIZER_RECIPE_DRIFT")
    schedule = recipe["reference_single_pass_sft_schedule"]
    if (
        schedule["primary_budget_unit"] != "samples"
        or schedule["target_samples"] != 80_000
        or schedule["effective_samples_per_update"] != 4
        or schedule["derived_optimizer_updates"] != 20_000
        or schedule["warmup_samples"] != 4_000
        or schedule["derived_warmup_updates"] != 1_000
        or schedule["peak_lr"] != 1e-4
        or schedule["decay_lr"] != 2.5e-6
        or schedule["decay_end_samples"] != 80_000
        or schedule["derived_decay_end_update"] != 20_000
        or recipe["single_pass_gate_budget"]
        != {"primary_budget_unit": "samples", "target_samples": 4, "derived_optimizer_updates": 1}
        or recipe["exact_two_pass_oracle_budget"]
        != {"primary_budget_unit": "samples", "target_samples": 16, "derived_optimizer_updates": 1}
        or recipe["single_pass_batching"]
        != {
            "batch_per_gpu": 4,
            "gradient_accumulation_microbatches": 1,
            "effective_samples_per_gpu_update": 4,
        }
        or recipe["exact_two_pass_oracle_batching"]
        != {
            "batch_per_gpu": 2,
            "gradient_accumulation_microbatches": 8,
            "effective_samples_per_gpu_update": 16,
        }
        or recipe["loss"]["balance_weight"] != 0.01
        or recipe["loss"]["z_weight"] != 0.001
        or recipe["loss"]["num_experts"] != 4
        or recipe["loss"]["active_training_router_algorithm"] != "single_pass_batch_local"
        or recipe["loss"]["acceptance_oracle_router_algorithm"]
        != "exact_two_pass_all_microbatches_all_ranks"
        or recipe["determinism"]["seeds"] != [42, 43, 44]
        or recipe["determinism"]["cublas_workspace_config"] != ":4096:8"
    ):
        raise RuntimeError("P7_TRAIN_RECIPE_DRIFT")
    if (
        recipe["training_phase_boundary"]["checkpoint_strict_reload"]
        != "requires_exact_resume_validation"
        or recipe["training_phase_boundary"]["long_development_sft"]
        != "requires_exact_resume_and_force_parity"
    ):
        raise RuntimeError("P7_TO_P8_BOUNDARY_DRIFT")
