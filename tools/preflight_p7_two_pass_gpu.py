#!/usr/bin/env python3
"""CUDA-only P7 B4x1 single-pass gate with an isolated exact two-pass oracle."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from preflight_p5_dense_compute_gpu import (
    _gradient_summary,
    _require_offline,
    _sha256,
    _validate_source_binding,
)
from preflight_p6_variants_gpu import (
    _dataset_storage_binding,
    _pytest_evidence_summary,
    _validate_p6_source_binding,
    _validate_runtime_import_roots,
    _validate_spec as _validate_p6_spec,
)


def _validate_recipe(recipe: dict) -> None:
    if (
        recipe.get("acceptance_status") != "development_only"
        or recipe.get("formal_eligible") is not False
        or recipe.get("execution_role") != "single_pass_gate_with_exact_two_pass_oracle"
        or recipe.get("long_running_sft_allowed") is not False
        or recipe.get("training_stage") != "offline_full_finetune"
        or recipe.get("all_existing_parameters_require_grad") is not True
    ):
        raise RuntimeError("P7_RECIPE_STATUS_DRIFT")
    if recipe.get("variants") != ["force_token_moe", "force_token_moe_additive"]:
        raise RuntimeError("P7_VARIANT_DRIFT")
    optimizer = recipe["optimizer"]
    schedule = recipe["reference_single_pass_sft_schedule"]
    single_budget = recipe["single_pass_gate_budget"]
    oracle_budget = recipe["exact_two_pass_oracle_budget"]
    single_batching = recipe["single_pass_batching"]
    oracle_batching = recipe["exact_two_pass_oracle_batching"]
    loss = recipe["loss"]
    determinism = recipe["determinism"]
    if optimizer != {
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
        or single_budget != {
            "primary_budget_unit": "samples",
            "target_samples": 4,
            "derived_optimizer_updates": 1,
        }
        or oracle_budget != {
            "primary_budget_unit": "samples",
            "target_samples": 16,
            "derived_optimizer_updates": 1,
        }
        or single_batching != {
            "batch_per_gpu": 4,
            "gradient_accumulation_microbatches": 1,
            "effective_samples_per_gpu_update": 4,
        }
        or oracle_batching != {
            "batch_per_gpu": 2,
            "gradient_accumulation_microbatches": 8,
            "effective_samples_per_gpu_update": 16,
        }
        or loss["balance_weight"] != 0.01
        or loss["z_weight"] != 0.001
        or loss["num_experts"] != 4
        or loss["active_training_router_algorithm"] != "single_pass_batch_local"
        or loss["acceptance_oracle_router_algorithm"]
        != "exact_two_pass_all_microbatches_all_ranks"
        or determinism["seeds"] != [42, 43, 44]
        or determinism["cublas_workspace_config"] != ":4096:8"
    ):
        raise RuntimeError("P7_TRAIN_RECIPE_DRIFT")
    if (
        recipe["p7_phase_boundary"]["checkpoint_strict_reload"]
        != "not_accepted_until_P8"
        or recipe["p7_phase_boundary"]["long_development_sft"]
        != "blocked_until_P8_exact_resume_and_force_parity"
    ):
        raise RuntimeError("P7_TO_P8_BOUNDARY_DRIFT")


def _validate_p6_prerequisite(
    root: Path,
    recipe: dict,
    *,
    dataset_root: Path,
    repo_id: str,
    binding: dict | None = None,
) -> dict:
    prerequisite = recipe.get("p6_prerequisite")
    expected_keys = {
        "static_spec",
        "source_binding",
        "resolved_config",
        "gate_result",
        "required_gate_status",
        "required_acceptance_status",
        "required_formal_eligible",
    }
    if not isinstance(prerequisite, dict) or set(prerequisite) != expected_keys:
        raise RuntimeError("P7_P6_PREREQUISITE_SPEC_MISSING_OR_DRIFTED")
    if (
        prerequisite["required_gate_status"] != "pass"
        or prerequisite["required_acceptance_status"] != "development_only"
        or prerequisite["required_formal_eligible"] is not False
    ):
        raise RuntimeError("P7_P6_PREREQUISITE_SEMANTICS_DRIFT")
    payloads = {}
    for name in ("static_spec", "source_binding", "resolved_config", "gate_result"):
        artifact = prerequisite[name]
        if set(artifact) != {"path", "sha256"}:
            raise RuntimeError(f"P7_P6_{name.upper()}_BINDING_DRIFT")
        path = root / artifact["path"]
        if _sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"P7_P6_{name.upper()}_HASH_MISMATCH")
        payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    p6_spec = payloads["static_spec"]
    p6_binding = payloads["source_binding"]
    p6_resolved = payloads["resolved_config"]
    p6_result = payloads["gate_result"]
    _validate_p6_spec(p6_spec)
    _validate_p6_source_binding(
        root,
        p6_binding,
        dataset_root=dataset_root,
        repo_id=repo_id,
        spec=p6_spec,
    )
    if (
        p6_result.get("gate") != "P6"
        or p6_result.get("gate_status") != "pass"
        or p6_result.get("acceptance_status") != "development_only"
        or p6_result.get("formal_eligible") is not False
        or p6_result.get("source_binding_sha256")
        != prerequisite["source_binding"]["sha256"]
        or p6_result.get("resolved_config_sha256")
        != prerequisite["resolved_config"]["sha256"]
        or p6_result.get("static_spec_sha256")
        != prerequisite["static_spec"]["sha256"]
        or p6_resolved.get("source_binding_sha256")
        != prerequisite["source_binding"]["sha256"]
        or p6_resolved.get("static_spec_sha256")
        != prerequisite["static_spec"]["sha256"]
    ):
        raise RuntimeError("P7_PARENT_P6_GATE_NOT_ELIGIBLE")
    if binding is not None and binding.get("p6_prerequisite") != prerequisite:
        raise RuntimeError("P7_SOURCE_BINDING_P6_PREREQUISITE_MISMATCH")
    return {
        name: prerequisite[name]["sha256"]
        for name in ("static_spec", "source_binding", "resolved_config", "gate_result")
    }


def _validate_p7_source_binding(
    root: Path,
    binding: dict,
    *,
    dataset_root: Path,
    repo_id: str,
    recipe: dict,
) -> dict:
    if (
        binding.get("stage") != "P7"
        or binding.get("formal_eligible") is not False
        or binding.get("signature_status") != "development_only_untrusted"
    ):
        raise RuntimeError("P7_SOURCE_BINDING_STATUS_DRIFT")
    _validate_source_binding(root, binding, dataset_root=dataset_root, repo_id=repo_id)
    runtime_imports = _validate_runtime_import_roots(root)
    if binding.get("runtime_imports") != runtime_imports:
        raise RuntimeError("P7_SOURCE_BINDING_RUNTIME_IMPORT_MISMATCH")
    if binding["dataset"].get("storage_tree") != _dataset_storage_binding(dataset_root):
        raise RuntimeError("P7_DATASET_STORAGE_TREE_HASH_MISMATCH")
    parent = _validate_p6_prerequisite(
        root,
        recipe,
        dataset_root=dataset_root,
        repo_id=repo_id,
        binding=binding,
    )
    tests = _pytest_evidence_summary(root, binding.get("test_evidence", {}))
    return {
        "p6_prerequisite": parent,
        "dataset_storage_tree_sha256": binding["dataset"]["storage_tree"]["tree_sha256"],
        "dataset_storage_file_count": binding["dataset"]["storage_tree"]["file_count"],
        "pytest": tests,
        "runtime_imports": runtime_imports,
    }


def _nonzero_gradient_group(policy, *, name: str, prefixes: tuple[str, ...]) -> dict:
    import torch

    selected = [
        (parameter_name, parameter)
        for parameter_name, parameter in policy.named_parameters()
        if parameter_name.startswith(prefixes)
    ]
    if not selected:
        raise RuntimeError(f"P7_GRADIENT_GROUP_EMPTY:{name}")
    missing = [parameter_name for parameter_name, parameter in selected if parameter.grad is None]
    zero = [
        parameter_name
        for parameter_name, parameter in selected
        if parameter.grad is not None
        and not bool(torch.count_nonzero(parameter.grad.detach()).item())
    ]
    return {
        "parameter_tensors": len(selected),
        "nonzero_gradient_tensors": len(selected) - len(missing) - len(zero),
        "missing_gradient_names": missing,
        "zero_gradient_names": zero,
        "all_nonzero": not missing and not zero,
    }


def _single_pass_terms(policy, microbatch):
    import torch

    from forcesmolvla.force_token import RouterState
    from forcesmolvla.router_training import collect_pass_a_statistics, microbatch_two_pass_terms

    device_type = microbatch.noise7.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type in {"cpu", "cuda"},
    ):
        flow_losses, feature_mask, router_state = policy.forward_single_pass_training_terms(
            microbatch.batch, noise=microbatch.noise7, time=microbatch.time
        )
        detached = RouterState(
            logits_fp32=router_state.logits_fp32.detach(),
            probabilities_fp32=router_state.probabilities_fp32.detach(),
            route_ids=router_state.route_ids.detach(),
            valid_mask=router_state.valid_mask.detach(),
        )
        statistics = collect_pass_a_statistics([detached], [feature_mask])
        terms = microbatch_two_pass_terms(flow_losses, router_state, statistics)
    return terms, router_state, statistics


def _flow_gradient_audit(policy, microbatch, *, after_optimizer_step: bool) -> dict:
    import torch

    policy.zero_grad(set_to_none=True)
    terms, router_state, _statistics = _single_pass_terms(policy, microbatch)
    terms.flow.backward()
    w_out = _nonzero_gradient_group(
        policy, name="w_out", prefixes=("model.force_adapter.w_out.",)
    )
    if not w_out["all_nonzero"]:
        raise RuntimeError("P7_L_FLOW_TO_W_OUT_GRADIENT_MISSING")
    report = {
        "after_optimizer_step": after_optimizer_step,
        "L_flow": float(terms.flow.detach().cpu()),
        "w_out": w_out,
    }
    if after_optimizer_step:
        active_experts = sorted(
            set(router_state.route_ids[router_state.valid_mask].detach().cpu().tolist())
        )
        groups = {
            "force_mlp": ("model.force_branch.force_mlp.",),
            "fusion": (
                "model.force_branch.segment_embedding.",
                "model.force_branch.fusion_position_embedding.",
                "model.force_branch.fusion_blocks.",
                "model.force_branch.guidance_projection.",
            ),
            "cross_attention_qkv": (
                "model.force_adapter.cross_attention.q_proj.",
                "model.force_adapter.cross_attention.k_proj.",
                "model.force_adapter.cross_attention.v_proj.",
            ),
            "conditioner": (
                "model.force_adapter.learned_action_slot",
                "model.force_adapter.time_projection.",
                "model.force_adapter.noisy_action_projection.",
            ),
        }
        group_reports = {
            group_name: _nonzero_gradient_group(policy, name=group_name, prefixes=prefixes)
            for group_name, prefixes in groups.items()
        }
        expert_reports = {
            str(expert_id): _nonzero_gradient_group(
                policy,
                name=f"routed_expert_{expert_id}",
                prefixes=(f"model.force_branch.refiner.experts.{expert_id}.",),
            )
            for expert_id in active_experts
        }
        failed = [name for name, value in group_reports.items() if not value["all_nonzero"]]
        failed.extend(
            f"routed_expert_{name}"
            for name, value in expert_reports.items()
            if not value["all_nonzero"]
        )
        if failed:
            raise RuntimeError(f"P7_POST_STEP_L_FLOW_UPSTREAM_GRADIENT_MISSING:{failed}")
        report.update(
            {
                "active_expert_ids": active_experts,
                "upstream_groups": group_reports,
                "routed_experts": expert_reports,
            }
        )
    policy.zero_grad(set_to_none=True)
    return report


def _router_aux_gradient_audit(policy, microbatch, *, loss_name: str) -> dict:
    import torch

    if loss_name not in {"balance", "z"}:
        raise ValueError("P7 router auxiliary audit requires balance or z")
    policy.zero_grad(set_to_none=True)
    terms, _router_state, _statistics = _single_pass_terms(policy, microbatch)
    loss = terms.balance if loss_name == "balance" else terms.z
    loss.backward()
    router = _nonzero_gradient_group(
        policy, name=f"L_{loss_name}_router", prefixes=("model.force_branch.refiner.router.",)
    )
    if not router["all_nonzero"]:
        raise RuntimeError(f"P7_L_{loss_name.upper()}_TO_ROUTER_GRADIENT_MISSING")
    policy.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach().cpu()), "router": router}


def _tensor_sha256(tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=tensor.dtype).contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _make_batch(policy, prepared_samples: list[dict], device):
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
            __import__("numpy").stack([sample["state7"] for sample in prepared_samples])
        ).to(device),
        "observation.wrench": torch.from_numpy(
            __import__("numpy").stack([sample["wrench6"] for sample in prepared_samples])
        ).to(device),
        ACTION: torch.from_numpy(
            __import__("numpy").stack([sample["delta_action7"] for sample in prepared_samples])
        ).to(device),
        "action_valid_mask": torch.from_numpy(
            __import__("numpy").stack(
                [sample["action_valid_mask"] for sample in prepared_samples]
            )
        ).to(device),
        OBS_LANGUAGE_TOKENS: encoded["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: encoded["attention_mask"].to(
            device=device, dtype=torch.bool
        ),
    }


def _compare_complete_state(main, additive) -> dict:
    main_state = main.state_dict()
    additive_state = additive.state_dict()
    if list(main_state) != list(additive_state):
        raise RuntimeError("P7_ADDITIVE_STATE_NAME_MISMATCH")
    schema = hashlib.sha256()
    for name in main_state:
        left = main_state[name]
        right = additive_state[name]
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not __import__("torch").equal(left.detach().cpu(), right.detach().cpu())
        ):
            raise RuntimeError(f"P7_ADDITIVE_INITIAL_TENSOR_MISMATCH: {name}")
        schema.update(f"{name}\0{left.dtype}\0{tuple(left.shape)}\n".encode())
    return {
        "state_tensor_count": len(main_state),
        "state_schema_sha256": schema.hexdigest(),
        "complete_tensor_equality": True,
    }


def _validation_scalar(policy, batch, noise, timestep) -> float:
    import torch

    policy.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        losses, feature_mask, _router = policy.forward_single_pass_training_terms(
            batch, noise=noise, time=timestep
        )
        scalar = losses.sum() / feature_mask.sum()
    torch.cuda.synchronize()
    return float(scalar.cpu())


def _build_validation_fixture(
    *, root: Path, dataset_root: Path, raw_samples: list[dict], batch, noise, timestep
) -> dict:
    import torch

    from forcesmolvla.context import ChunkContext

    normalizer_sha = _sha256(dataset_root / "normalizer_manifest.json")
    calibration_sha = _sha256(root / "configs/calibration_bundle.development.json")
    geometry_sha = _sha256(root / "configs/wrench_geometry_spec.development.json")
    action_mask = batch["action_valid_mask"].detach().cpu()
    tuple_list = [
        {
            "split": "val",
            "fixture_position": index,
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
        }
        for index, sample in enumerate(raw_samples)
    ]
    chunk_payload = {
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
        "chunk_id": [f"p7-validation-{index}" for index in range(len(raw_samples))],
        "action_valid_mask": action_mask.tolist(),
        "suffix_valid_mask": action_mask.tolist(),
        "calibration_bundle_hash": [calibration_sha] * len(raw_samples),
        "wrench_geometry_spec_hash": [geometry_sha] * len(raw_samples),
        "normalizer_hash": [normalizer_sha] * len(raw_samples),
        "calibration_mapping_hash_or_none": [None] * len(raw_samples),
        "wrench_geometry_valid": [True] * len(raw_samples),
        "runtime_artifact_compatible": [True] * len(raw_samples),
        "selected_provenance": tuple_list,
    }
    context = ChunkContext(
        policy_generation=0,
        raw_state_snapshot=torch.tensor(chunk_payload["raw_state_snapshot"]),
        t_ref_ns=torch.tensor(chunk_payload["t_ref_ns"], dtype=torch.int64),
        tau0_ns=torch.tensor(chunk_payload["tau0_ns"], dtype=torch.int64),
        clock_domain_id=tuple(chunk_payload["clock_domain_id"]),
        episode_id=tuple(chunk_payload["episode_id"]),
        session_id=tuple(chunk_payload["session_id"]),
        sample_id=tuple(chunk_payload["sample_id"]),
        chunk_id=tuple(chunk_payload["chunk_id"]),
        action_valid_mask=action_mask,
        suffix_valid_mask=action_mask,
        calibration_bundle_hash=tuple(chunk_payload["calibration_bundle_hash"]),
        wrench_geometry_spec_hash=tuple(chunk_payload["wrench_geometry_spec_hash"]),
        normalizer_hash=tuple(chunk_payload["normalizer_hash"]),
        calibration_mapping_hash_or_none=tuple(chunk_payload["calibration_mapping_hash_or_none"]),
        wrench_geometry_valid=torch.ones(len(raw_samples), dtype=torch.bool),
        runtime_artifact_compatible=torch.ones(len(raw_samples), dtype=torch.bool),
        selected_provenance=tuple(tuple_list),
    )
    context.validate(batch_size=len(raw_samples), horizon=50, policy_generation=0)
    return {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "mode": "fixed_validation_development",
        "tuple_list": tuple_list,
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
        "chunk_context": chunk_payload,
        "chunk_context_sha256": _canonical_sha256(chunk_payload),
        "checkpoint_selection_use": (
            "single_pass_global_valid_feature_token_weighted_L_flow_only"
        ),
        "validation_algorithm": "single_pass_batch_local",
        "detached_signature": None,
        "approval": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-output", type=Path, required=True)
    parser.add_argument("--fixture-output", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path(__file__).parents[1] / "configs/p7_training_recipe.development.yaml",
    )
    parser.add_argument(
        "--source-binding",
        type=Path,
        default=Path(__file__).parents[1]
        / "artifacts/development/p7_v4_2_r2_source_binding.json",
    )
    args = parser.parse_args()
    for path in (args.output, args.resolved_output, args.fixture_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite P7 artifact: {path}")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("PYTHONHASHSEED=42 required before interpreter startup")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required before CUDA initialization")
    _require_offline()

    import numpy as np
    import torch

    from forcesmolvla.checkpoint import load_offline_base_policy
    from forcesmolvla.configuration_forcesmolvla import (
        FORCE_TOKEN_MOE,
        FORCE_TOKEN_MOE_ADDITIVE,
        OFFLINE_FULL_FINETUNE,
    )
    from forcesmolvla.dataset_v3 import load_dataset_split
    from forcesmolvla.router_training import (
        MoEMicrobatch,
        SerializableUniformSampler,
        build_p7_optimizer_and_scheduler,
        single_pass_optimizer_update,
        two_pass_optimizer_update,
    )
    from forcesmolvla.training_data import load_runtime_artifacts, prepare_training_sample

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")
    gpu_name = torch.cuda.get_device_name(0)
    if "4090 D" not in gpu_name and "4090D" not in gpu_name:
        raise RuntimeError(f"P7_REQUIRES_RTX_4090D: got {gpu_name!r}")
    root = Path(__file__).parents[1].resolve()
    recipe_path = args.recipe.resolve()
    binding_path = args.source_binding.resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    _validate_recipe(recipe)
    if binding.get("status") != "development_only" or binding.get("stage") != "P7":
        raise RuntimeError("P7_SOURCE_BINDING_STATUS_DRIFT")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    dataset_root = args.dataset_root.resolve()
    conversion = json.loads(
        (dataset_root / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    repo_id = conversion.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("P7_DATASET_REPO_ID_MISSING")
    binding_evidence = _validate_p7_source_binding(
        root,
        binding,
        dataset_root=dataset_root,
        repo_id=repo_id,
        recipe=recipe,
    )
    delta_timestamps = {"action": [index / 30 for index in range(50)]}
    train_dataset = load_dataset_split(
        dataset_root,
        repo_id=repo_id,
        split_name="train",
        artifact_use="development",
        delta_timestamps=delta_timestamps,
    )
    val_dataset = load_dataset_split(
        dataset_root,
        repo_id=repo_id,
        split_name="val",
        artifact_use="development",
        delta_timestamps=delta_timestamps,
    )
    if len(train_dataset) < 16 or len(val_dataset) < 2:
        raise RuntimeError("P7_DATASET_TOO_SMALL_FOR_FROZEN_FIXTURES")
    normalizer = load_runtime_artifacts(
        dataset_root,
        calibration_bundle_path=root / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=root / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=root / "artifacts/development/action_delta_spec.json",
        expected_repo_id=repo_id,
    ).normalizer
    sampler = SerializableUniformSampler(list(range(len(train_dataset))), seed=42)
    sampled_indices = sampler.draw(16)
    prepared_train = [
        prepare_training_sample(train_dataset[index], normalizer) for index in sampled_indices
    ]
    raw_val = [val_dataset[0], val_dataset[1]]
    prepared_val = [prepare_training_sample(sample, normalizer) for sample in raw_val]

    def load_variant(variant, device_name):
        with contextlib.redirect_stdout(sys.stderr):
            return load_offline_base_policy(
                root / "assets/base_checkpoint",
                root / "assets/smolvlm_constructor",
                device=device_name,
                training_stage=OFFLINE_FULL_FINETUNE,
                force_variant=variant,
                acceptance_status="development_only",
                force_init_seed=42,
            )

    main_policy, main_base_report = load_variant(FORCE_TOKEN_MOE, "cuda")
    additive_policy, additive_base_report = load_variant(FORCE_TOKEN_MOE_ADDITIVE, "cpu")
    if main_base_report.to_dict() != additive_base_report.to_dict():
        raise RuntimeError("P7_ADDITIVE_BASE_LOAD_REPORT_MISMATCH")
    complete_state = _compare_complete_state(main_policy, additive_policy)
    if main_policy.force_initialization_tensor_hash() != additive_policy.force_initialization_tensor_hash():
        raise RuntimeError("P7_ADDITIVE_FORCE_INIT_HASH_MISMATCH")
    main_parameters = sum(parameter.numel() for parameter in main_policy.parameters())
    additive_parameters = sum(parameter.numel() for parameter in additive_policy.parameters())
    if main_parameters != additive_parameters or main_parameters != 505_620_341:
        raise RuntimeError("P7_ADDITIVE_PARAMETER_COUNT_MISMATCH")
    if not all(parameter.requires_grad for parameter in main_policy.parameters()) or not all(
        parameter.requires_grad for parameter in additive_policy.parameters()
    ):
        raise RuntimeError("P7_FROZEN_PARAMETER_DETECTED")

    validation_batch = _make_batch(main_policy, prepared_val, device)
    generator = torch.Generator(device=device).manual_seed(43)
    validation_noise = torch.randn(
        2, 50, 7, generator=generator, device=device, dtype=torch.float32
    )
    validation_time = torch.tensor([0.25, 0.75], device=device, dtype=torch.float32)
    main_initial_scalar = _validation_scalar(
        main_policy, validation_batch, validation_noise, validation_time
    )
    additive_validation_batch = _make_batch(additive_policy, prepared_val, device)
    main_policy.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    additive_policy.config.device = "cuda"
    additive_policy.to("cuda")
    additive_initial_scalar = _validation_scalar(
        additive_policy, additive_validation_batch, validation_noise, validation_time
    )
    if main_initial_scalar != additive_initial_scalar:
        raise RuntimeError("P7_ADDITIVE_STEP_ZERO_NATIVE_OUTPUT_MISMATCH")
    _additive_optimizer, _additive_scheduler, additive_groups = build_p7_optimizer_and_scheduler(
        additive_policy
    )
    del _additive_optimizer, _additive_scheduler, additive_validation_batch, additive_policy
    gc.collect()
    torch.cuda.empty_cache()
    main_policy.to("cuda")

    main_policy.train()
    optimizer, scheduler, main_groups = build_p7_optimizer_and_scheduler(main_policy)
    if main_groups != additive_groups:
        raise RuntimeError("P7_ADDITIVE_OPTIMIZER_GROUP_MISMATCH")
    learning_rate_before = optimizer.param_groups[0]["lr"]
    train_generator = torch.Generator(device=device).manual_seed(44)
    single_batch = _make_batch(main_policy, prepared_train[:4], device)
    single_noise = torch.randn(
        4, 50, 7, generator=train_generator, device=device, dtype=torch.float32
    )
    single_time = torch.tensor([0.1, 0.3, 0.6, 0.8], device=device, dtype=torch.float32)
    single_microbatch = MoEMicrobatch(
        batch=single_batch,
        noise7=single_noise,
        time=single_time,
        identity="p7-b4x1-single-pass-update-0",
    )

    step1_flow_audit = _flow_gradient_audit(
        main_policy, single_microbatch, after_optimizer_step=False
    )
    balance_gradient_audit = _router_aux_gradient_audit(
        main_policy, single_microbatch, loss_name="balance"
    )
    z_gradient_audit = _router_aux_gradient_audit(
        main_policy, single_microbatch, loss_name="z"
    )

    vlm_forward_calls = 0

    def count_vlm_forward(_module, _inputs, _output):
        nonlocal vlm_forward_calls
        vlm_forward_calls += 1

    original_vlm_forward = main_policy.model.vlm_with_expert.forward

    def counted_vlm_forward(*args, **kwargs):
        count_vlm_forward(None, None, None)
        return original_vlm_forward(*args, **kwargs)

    main_policy.model.vlm_with_expert.forward = counted_vlm_forward
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    single_start = torch.cuda.Event(enable_timing=True)
    single_end = torch.cuda.Event(enable_timing=True)
    single_wall_start = time.perf_counter()
    single_start.record()
    try:
        single_update_report = single_pass_optimizer_update(
            main_policy,
            single_microbatch,
            optimizer,
            scheduler=scheduler,
            grad_clip_norm=10.0,
        )
    finally:
        main_policy.model.vlm_with_expert.forward = original_vlm_forward
    single_end.record()
    torch.cuda.synchronize(device)
    single_wall_seconds = time.perf_counter() - single_wall_start
    single_cuda_ms = single_start.elapsed_time(single_end)
    single_peak = {
        "allocated_bytes": torch.cuda.max_memory_allocated(device),
        "reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    if (
        single_update_report.get("training_update_algorithm")
        != "single_pass_batch_local"
        or single_update_report.get("microbatch_count") != 1
        or single_update_report.get("optimizer_steps") != 1
        or single_update_report.get("scheduler_steps") != 1
        or vlm_forward_calls != 1
    ):
        raise RuntimeError(
            "P7_B4X1_SINGLE_PASS_UPDATE_CONTRACT_DRIFT: "
            f"report={single_update_report}, vlm_forward_calls={vlm_forward_calls}"
        )
    single_gradient_coverage = _gradient_summary(list(main_policy.named_parameters()))
    for group_name in ("vision", "vlm_text", "action_expert", "action_io"):
        if single_gradient_coverage[group_name]["coverage_nonzero_gradient"] != 1.0:
            raise RuntimeError(f"P7_SINGLE_PASS_BASE_GRADIENT_COVERAGE_INCOMPLETE:{group_name}")
    if (
        single_gradient_coverage["base"]["missing_gradient_names"]
        != ["model.vlm_with_expert.vlm.lm_head.weight"]
        or single_gradient_coverage["base"]["zero_gradient_names"]
    ):
        raise RuntimeError("P7_SINGLE_PASS_BASE_GRADIENT_EXCEPTION_SET_DRIFT")
    step2_flow_audit = _flow_gradient_audit(
        main_policy, single_microbatch, after_optimizer_step=True
    )
    scheduler_state = {
        "learning_rate_before_update": learning_rate_before,
        "learning_rate_after_update": optimizer.param_groups[0]["lr"],
        "last_epoch": scheduler.last_epoch,
    }

    first_validation = _validation_scalar(
        main_policy, validation_batch, validation_noise, validation_time
    )
    second_validation = _validation_scalar(
        main_policy, validation_batch, validation_noise, validation_time
    )
    if first_validation != second_validation:
        raise RuntimeError("P7_FIXED_VALIDATION_NOT_BITWISE_REPRODUCIBLE")
    fixture = _build_validation_fixture(
        root=root,
        dataset_root=dataset_root,
        raw_samples=raw_val,
        batch=validation_batch,
        noise=validation_noise,
        timestep=validation_time,
    )
    fixture["evaluation"] = {
        "algorithm": "single_pass_batch_local",
        "after_development_update_L_flow_run_1": first_validation,
        "after_development_update_L_flow_run_2": second_validation,
        "bitwise_equal_python_float": True,
    }
    args.fixture_output.parent.mkdir(parents=True, exist_ok=True)
    args.fixture_output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    force_initialization_sha256 = main_policy.force_initialization_tensor_hash()
    del optimizer, scheduler, single_batch, single_microbatch, validation_batch, main_policy
    gc.collect()
    torch.cuda.empty_cache()

    oracle_policy, oracle_base_report = load_variant(FORCE_TOKEN_MOE, "cuda")
    if oracle_base_report.to_dict() != main_base_report.to_dict():
        raise RuntimeError("P7_ORACLE_BASE_LOAD_REPORT_MISMATCH")
    oracle_policy.train()
    oracle_optimizer, oracle_scheduler, oracle_groups = build_p7_optimizer_and_scheduler(
        oracle_policy
    )
    if oracle_groups != main_groups:
        raise RuntimeError("P7_ORACLE_OPTIMIZER_GROUP_MISMATCH")
    oracle_generator = torch.Generator(device=device).manual_seed(44)
    oracle_microbatches = []
    for index in range(8):
        oracle_batch = _make_batch(
            oracle_policy, prepared_train[index * 2 : index * 2 + 2], device
        )
        oracle_noise = torch.randn(
            2, 50, 7, generator=oracle_generator, device=device, dtype=torch.float32
        )
        oracle_time = torch.tensor(
            [0.1 + index * 0.01, 0.8 - index * 0.01],
            device=device,
            dtype=torch.float32,
        )
        oracle_microbatches.append(
            MoEMicrobatch(
                batch=oracle_batch,
                noise7=oracle_noise,
                time=oracle_time,
                identity=f"p7-exact-oracle-window-0-q-{index}",
            )
        )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    oracle_start = torch.cuda.Event(enable_timing=True)
    oracle_end = torch.cuda.Event(enable_timing=True)
    oracle_wall_start = time.perf_counter()
    oracle_start.record()
    exact_oracle_report = two_pass_optimizer_update(
        oracle_policy,
        oracle_microbatches,
        oracle_optimizer,
        oracle_mode=True,
        scheduler=oracle_scheduler,
        expected_microbatches=8,
        grad_clip_norm=10.0,
    )
    oracle_end.record()
    torch.cuda.synchronize(device)
    oracle_wall_seconds = time.perf_counter() - oracle_wall_start
    oracle_cuda_ms = oracle_start.elapsed_time(oracle_end)
    oracle_peak = {
        "allocated_bytes": torch.cuda.max_memory_allocated(device),
        "reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    if (
        exact_oracle_report["optimizer_steps"] != 1
        or exact_oracle_report["scheduler_steps"] != 1
        or exact_oracle_report["max_router_probability_replay_error"] != 0.0
    ):
        raise RuntimeError("P7_EXACT_TWO_PASS_ORACLE_CONTRACT_DRIFT")
    exact_oracle_gradient_coverage = _gradient_summary(
        list(oracle_policy.named_parameters())
    )
    del oracle_optimizer, oracle_scheduler, oracle_microbatches, oracle_policy
    gc.collect()
    torch.cuda.empty_cache()

    recipe_sha = _sha256(recipe_path)
    binding_sha = _sha256(binding_path)
    fixture_sha = _sha256(args.fixture_output)
    resolved = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "training_stage": "offline_full_finetune",
        "source_binding_sha256": binding_sha,
        "training_recipe_sha256": recipe_sha,
        "fixed_validation_fixture_sha256": fixture_sha,
        "fixed_validation_algorithm": "single_pass_batch_local",
        "seeds": [42, 43, 44],
        "force_initialization_tensor_sha256": force_initialization_sha256,
        "complete_main_additive_state": complete_state,
        "optimizer_groups": main_groups,
        "scheduler": scheduler_state,
        "training_update_algorithm": "single_pass_batch_local",
        "exact_two_pass_role": "acceptance_oracle_only",
        "p6_prerequisite": binding_evidence["p6_prerequisite"],
        "p7_phase_boundary": recipe["p7_phase_boundary"],
        "detached_signature": None,
        "approval": None,
    }
    args.resolved_output.parent.mkdir(parents=True, exist_ok=True)
    args.resolved_output.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "gate": "P7",
        "gate_status": "pass",
        "gpu": {
            "name": gpu_name,
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "real_data": {
            "dataset": str(dataset_root),
            "repo_id": repo_id,
            "split": "train",
            "sampled_indices": sampled_indices,
            "single_pass_batch_per_gpu": 4,
            "single_pass_gradient_accumulation_microbatches": 1,
            "exact_oracle_batch_per_gpu": 2,
            "exact_oracle_microbatches": 8,
            "horizon": 50,
            "camera_count": 2,
            "sampler_cursor_after_draw": sampler.cursor,
        },
        "all_parameters_trainable": True,
        "total_parameters": main_parameters,
        "frozen_parameters": 0,
        "force_initialization_tensor_sha256": force_initialization_sha256,
        "additive_parameter_match": {
            **complete_state,
            "total_parameters": additive_parameters,
            "force_initialization_equal": True,
            "optimizer_groups_equal": True,
            "initial_native_L_flow_main": main_initial_scalar,
            "initial_native_L_flow_additive": additive_initial_scalar,
        },
        "single_pass_update": {
            **single_update_report,
            "batch_size": 4,
            "gradient_accumulation_microbatches": 1,
            "vlm_with_expert_forward_calls": vlm_forward_calls,
        },
        "exact_two_pass_oracle": {
            **exact_oracle_report,
            "execution_role": "acceptance_oracle_only",
            "long_running_sft_allowed": False,
        },
        "gradient_source_audit": {
            "step_1_L_flow_only": step1_flow_audit,
            "step_2_L_flow_only_after_optimizer_step": step2_flow_audit,
            "L_balance_only": balance_gradient_audit,
            "L_z_only": z_gradient_audit,
        },
        "single_pass_gradient_coverage": single_gradient_coverage,
        "exact_oracle_gradient_coverage": exact_oracle_gradient_coverage,
        "scheduler": resolved["scheduler"],
        "fixed_validation": {
            "algorithm": "single_pass_batch_local",
            "fixture_sha256": fixture_sha,
            "L_flow_run_1": first_validation,
            "L_flow_run_2": second_validation,
            "exact_replay": True,
        },
        "latency": {
            "single_pass_optimizer_update_cuda_ms": single_cuda_ms,
            "single_pass_optimizer_update_wall_seconds": single_wall_seconds,
            "exact_two_pass_oracle_cuda_ms": oracle_cuda_ms,
            "exact_two_pass_oracle_wall_seconds": oracle_wall_seconds,
        },
        "peak_memory": {
            "single_pass": single_peak,
            "exact_two_pass_oracle": oracle_peak,
        },
        "cpu_fallback_used": False,
        "architecture_downgrade_used": False,
        "source_binding_sha256": binding_sha,
        "source_binding_evidence": binding_evidence,
        "training_recipe_sha256": recipe_sha,
        "resolved_config_sha256": _sha256(args.resolved_output),
        "p8_started": False,
        "remaining_blockers": [
            "P8 strict checkpoint save/reload and cold-start parity not executed",
            "P9 offline record/replay Shadow not implemented",
            "trusted detached signature and formal approvals unresolved",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
