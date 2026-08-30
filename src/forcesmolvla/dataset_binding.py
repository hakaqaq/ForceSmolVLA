"""Dataset storage and runtime binding validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forcesmolvla.training_runtime import (
    file_sha256,
    validate_dense_compute_spec,
    validate_action_target_prerequisite,
    validate_source_binding,
)


def dataset_storage_binding(dataset_root: Path) -> dict:
    files = sorted(
        path
        for directory in ("data", "videos", "meta")
        for path in (dataset_root / directory).rglob("*")
        if path.is_file()
    )
    if not files:
        raise RuntimeError("P6_DATASET_STORAGE_TREE_EMPTY")
    hashes = {
        path.relative_to(dataset_root).as_posix(): file_sha256(path) for path in files
    }
    digest = hashlib.sha256()
    for relative, value in hashes.items():
        digest.update(f"{relative}\0{value}\n".encode())
    return {
        "roots": ["data", "videos", "meta"],
        "file_count": len(hashes),
        "tree_sha256": digest.hexdigest(),
        "files": hashes,
    }


def validate_runtime_import_roots(root: Path) -> dict:
    import forcesmolvla
    import lerobot

    expected = {
        "forcesmolvla": root / "src/forcesmolvla",
        "lerobot": root / "vendor/lerobot/src/lerobot",
    }
    observed = {}
    for name, module in (("forcesmolvla", forcesmolvla), ("lerobot", lerobot)):
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError(f"P6_{name.upper()}_IMPORT_FILE_MISSING")
        resolved = Path(module_file).resolve()
        if not resolved.is_relative_to(expected[name].resolve()):
            raise RuntimeError(f"P6_{name.upper()}_IMPORT_ROOT_MISMATCH: {resolved}")
        observed[name] = str(resolved)
    return observed


def validate_variant_spec(spec: dict) -> None:
    if (
        spec.get("acceptance_status") != "development_only"
        or spec.get("formal_eligible") is not False
        or spec.get("training_stage") != "offline_full_finetune"
        or spec.get("seed") != 42
    ):
        raise RuntimeError("P6_STATIC_SPEC_STATUS_DRIFT")
    if spec.get("common_architecture_binding") != {
        "p5_static_spec_sha256": "7ab3307874c4b27e723b27972b4fee2d80629ecd60da60a9e463b13d380d9405",
        "D_vlm": 960,
        "D_expert": 720,
        "fusion_blocks": 2,
        "fusion_heads": 8,
        "force_cross_attention_heads": 1,
        "N_fused_physical": 177,
    }:
        raise RuntimeError("P6_COMMON_ARCHITECTURE_BINDING_DRIFT")
    dense, moe = spec["dense_param"], spec["moe"]
    if (
        dense["hidden_dim"] != 15364
        or dense["refiner_total_parameters_including_norm"] != 29_517_124
        or dense["total_parameters"] != 505_621_301
    ):
        raise RuntimeError("P6_DENSE_PARAM_BUDGET_DRIFT")
    if (
        moe["num_experts"] != 4
        or moe["top_k"] != 1
        or moe["capacity_free"] is not True
        or moe["token_drop"] is not False
        or moe["fallback_expert"] is not False
        or moe.get("router_weight_initialization")
        != "deterministic_normal_mean_0_std_0.02"
        or moe.get("router_bias_initialization") != "zeros"
        or moe.get("router_initialization_basis")
        != "ForceVLA_FlaxFormer_RouterWeights_default_kernel_init"
        or moe.get("router_jitter") != 0.0
        or moe["total_parameters"] != 505_620_341
    ):
        raise RuntimeError("P6_MOE_BUDGET_DRIFT")
    if spec.get("execution") != {
        "batch_size": 4,
        "gradient_accumulation_microbatches": 1,
        "camera_count": 2,
        "horizon": 50,
        "optimizer_steps": 2,
    }:
        raise RuntimeError("P6_EXECUTION_SPEC_DRIFT")


def validate_dense_compute_prerequisite(
    root: Path,
    spec: dict,
    *,
    dataset_root: Path,
    repo_id: str,
    binding: dict | None = None,
) -> dict:
    prerequisite = spec.get("p5_prerequisite")
    expected = {
        "static_spec",
        "source_binding",
        "resolved_config",
        "gate_result",
        "required_gate_status",
        "required_acceptance_status",
        "required_formal_eligible",
    }
    if not isinstance(prerequisite, dict) or set(prerequisite) != expected:
        raise RuntimeError("P6_P5_PREREQUISITE_SPEC_MISSING_OR_DRIFTED")
    if (
        prerequisite["required_gate_status"] != "pass"
        or prerequisite["required_acceptance_status"] != "development_only"
        or prerequisite["required_formal_eligible"] is not False
    ):
        raise RuntimeError("P6_P5_PREREQUISITE_SEMANTICS_DRIFT")
    payloads = {}
    for name in ("static_spec", "source_binding", "resolved_config", "gate_result"):
        artifact = prerequisite[name]
        if set(artifact) != {"path", "sha256"}:
            raise RuntimeError(f"P6_P5_{name.upper()}_BINDING_DRIFT")
        path = root / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"P6_P5_{name.upper()}_HASH_MISMATCH")
        payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    parent_spec = payloads["static_spec"]
    parent_binding = payloads["source_binding"]
    parent_resolved = payloads["resolved_config"]
    parent_result = payloads["gate_result"]
    validate_dense_compute_spec(parent_spec)
    validate_action_target_prerequisite(root, parent_spec, parent_binding)
    validate_source_binding(
        root, parent_binding, dataset_root=dataset_root, repo_id=repo_id
    )
    if (
        parent_result.get("gate") != "P5"
        or parent_result.get("gate_status") != prerequisite["required_gate_status"]
        or parent_result.get("acceptance_status") != prerequisite["required_acceptance_status"]
        or parent_result.get("formal_eligible") is not prerequisite["required_formal_eligible"]
        or parent_result.get("source_binding_sha256") != prerequisite["source_binding"]["sha256"]
        or parent_result.get("resolved_config_sha256") != prerequisite["resolved_config"]["sha256"]
        or parent_result.get("static_spec_sha256") != prerequisite["static_spec"]["sha256"]
        or parent_result.get("real_batch", {}).get("repo_id") != repo_id
        or parent_result.get("real_batch", {}).get("batch_size") != 4
        or parent_resolved.get("source_binding_sha256") != prerequisite["source_binding"]["sha256"]
        or parent_resolved.get("static_spec_sha256") != prerequisite["static_spec"]["sha256"]
    ):
        raise RuntimeError("P6_PARENT_P5_GATE_NOT_ELIGIBLE")
    if binding is not None and binding.get("p5_prerequisite") != prerequisite:
        raise RuntimeError("P6_SOURCE_BINDING_P5_PREREQUISITE_MISMATCH")
    return {
        name: prerequisite[name]["sha256"]
        for name in ("static_spec", "source_binding", "resolved_config", "gate_result")
    }


def validate_dataset_source_binding(
    root: Path,
    binding: dict,
    *,
    dataset_root: Path,
    repo_id: str,
    spec: dict,
) -> dict:
    if (
        binding.get("stage") != "P6"
        or binding.get("formal_eligible") is not False
        or binding.get("signature_status") != "development_only_untrusted"
    ):
        raise RuntimeError("P6_SOURCE_BINDING_STATUS_DRIFT")
    validate_source_binding(root, binding, dataset_root=dataset_root, repo_id=repo_id)
    runtime_imports = validate_runtime_import_roots(root)
    if binding.get("runtime_imports") != runtime_imports:
        raise RuntimeError("P6_SOURCE_BINDING_RUNTIME_IMPORT_MISMATCH")
    storage = dataset_storage_binding(dataset_root)
    if binding["dataset"].get("storage_tree") != storage:
        raise RuntimeError("P6_DATASET_STORAGE_TREE_HASH_MISMATCH")
    parent = validate_dense_compute_prerequisite(
        root,
        spec,
        dataset_root=dataset_root,
        repo_id=repo_id,
        binding=binding,
    )
    return {
        "p5_prerequisite": parent,
        "dataset_storage_tree_sha256": storage["tree_sha256"],
        "dataset_storage_file_count": storage["file_count"],
        "runtime_imports": runtime_imports,
    }


def validate_dataset_variant_prerequisite(
    root: Path,
    recipe: dict,
    *,
    dataset_root: Path,
    repo_id: str,
    binding: dict | None = None,
) -> dict:
    """Validate dataset-variant inputs consumed by the SFT recipe."""

    prerequisite = recipe.get("model_architecture_prerequisite")
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
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"P7_P6_{name.upper()}_HASH_MISMATCH")
        payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    architecture_spec = payloads["static_spec"]
    architecture_binding = payloads["source_binding"]
    resolved_architecture = payloads["resolved_config"]
    architecture_validation = payloads["gate_result"]
    validate_variant_spec(architecture_spec)
    validate_dataset_source_binding(
        root,
        architecture_binding,
        dataset_root=dataset_root,
        repo_id=repo_id,
        spec=architecture_spec,
    )
    if (
        architecture_validation.get("gate") != "P6"  # persisted artifact ABI
        or architecture_validation.get("gate_status") != "pass"
        or architecture_validation.get("acceptance_status") != "development_only"
        or architecture_validation.get("formal_eligible") is not False
        or architecture_validation.get("source_binding_sha256")
        != prerequisite["source_binding"]["sha256"]
        or architecture_validation.get("resolved_config_sha256")
        != prerequisite["resolved_config"]["sha256"]
        or architecture_validation.get("static_spec_sha256")
        != prerequisite["static_spec"]["sha256"]
        or resolved_architecture.get("source_binding_sha256")
        != prerequisite["source_binding"]["sha256"]
        or resolved_architecture.get("static_spec_sha256")
        != prerequisite["static_spec"]["sha256"]
    ):
        raise RuntimeError("P7_PARENT_P6_GATE_NOT_ELIGIBLE")
    if (
        binding is not None
        and binding.get("model_architecture_prerequisite") != prerequisite
    ):
        raise RuntimeError("P7_SOURCE_BINDING_P6_PREREQUISITE_MISMATCH")
    return {
        name: prerequisite[name]["sha256"]
        for name in ("static_spec", "source_binding", "resolved_config", "gate_result")
    }
