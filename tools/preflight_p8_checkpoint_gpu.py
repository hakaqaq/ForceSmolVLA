#!/usr/bin/env python3
"""CUDA-only P8 complete checkpoint save plus fresh-process strict reload gate."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

from forcesmolvla.training_runtime import (
    build_training_batch as _make_batch,
    file_sha256 as _sha256,
    require_offline_environment as _require_offline,
    validate_source_binding as _validate_source_binding,
    validate_training_recipe as _validate_recipe,
    validation_scalar as _validation_scalar,
)
from forcesmolvla.dataset_binding import (
    dataset_storage_binding as _dataset_storage_binding,
    validate_runtime_import_roots as _validate_runtime_import_roots,
)


def _copy_payloads(
    root: Path,
    dataset_root: Path,
    checkpoint: Path,
    *,
    contract: dict,
    p8_source_binding: Path,
    force_parity_fp32: Path,
    force_parity_bf16: Path,
) -> dict[str, str]:
    p7 = contract["p7_prerequisite"]
    sources = {
        "manifests/normalizer_manifest.json": dataset_root / "normalizer_manifest.json",
        "manifests/conversion_manifest.json": dataset_root / "conversion_manifest.json",
        "manifests/split_manifest.json": dataset_root / "split_manifest.json",
        "manifests/action_delta_spec.json": root / "artifacts/development/action_delta_spec.json",
        "manifests/feature_mask_spec.json": root / "artifacts/development/feature_mask_spec.json",
        "manifests/processor_graph_manifest.json": root / "artifacts/development/processor_graph_manifest.json",
        "manifests/visual_language_manifest.json": root / "artifacts/development/visual_language_manifest.json",
        "manifests/flow_matching_spec.json": root / "artifacts/development/flow_matching_spec.json",
        "manifests/prefix_layout_spec.json": root / "artifacts/development/prefix_layout_spec.json",
        "manifests/prefix_cache_contract.json": root / "artifacts/development/prefix_cache_contract.json",
        "manifests/resolved_force_config.json": root / "artifacts/development/resolved_force_config.json",
        "manifests/base_asset_manifest.json": root / "artifacts/development/base_asset_manifest.json",
        "manifests/p7_resolved_config.json": root / p7["resolved_config"]["path"],
        "manifests/p7_source_binding.json": root / p7["source_binding"]["path"],
        "manifests/p7_gate_result.json": root / p7["gate_result"]["path"],
        "manifests/p8_source_binding.json": p8_source_binding,
        "manifests/p7_validation_fixture.json": root / p7["validation_fixture"]["path"],
        "manifests/parity_acceptance.development.json": root
        / "configs/p8_parity_acceptance.development.json",
        "manifests/p8_force_parity_gpu_fp32.json": force_parity_fp32,
        "manifests/p8_force_parity_gpu_bf16.json": force_parity_bf16,
        "manifests/implementation_spec_v4_2.md": root
        / "ForceSmolVLA_Implementation_Spec_v4_2.md",
        "manifests/wrench_geometry_spec.development.json": root / "configs/wrench_geometry_spec.development.json",
        "manifests/force_quality_thresholds.development.yaml": root / "configs/force_quality_thresholds.development.yaml",
        "manifests/wrench_filter_resample_spec.development.json": root / "configs/wrench_filter_resample_spec.development.json",
        "manifests/calibration_bundle.development.json": root / "configs/calibration_bundle.development.json",
        "manifests/converter_runtime_spec.development.json": root / "configs/converter_runtime_spec.development.json",
        "manifests/training_stage.development.json": root / "configs/training_stage.development.json",
        "manifests/p7_training_recipe.development.yaml": root
        / p7["training_recipe"]["path"],
        "manifests/p8_checkpoint_contract.development.json": root / "configs/p8_checkpoint_contract.development.json",
        "manifests/approval_checklist.yaml": root / "configs/approval_checklist.yaml",
        "manifests/environment_manifest.json": root / "artifacts/development/environment_manifest.json",
        "environment/conda-explicit.txt": root / "environment-manifest/conda-explicit.txt",
        "environment/conda-from-history.yml": root / "environment-manifest/conda-from-history.yml",
        "environment/pip-freeze.txt": root / "environment-manifest/pip-freeze.txt",
        "environment/requirements.lock": root / "environment-manifest/requirements.lock",
    }
    provenance = {}
    for relative, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"P8 required source payload missing: {source}")
        target = checkpoint / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        provenance[relative] = _sha256(source)
    return provenance


def _validate_contract(contract: dict) -> None:
    if (
        contract.get("acceptance_status") != "development_only"
        or contract.get("formal_eligible") is not False
        or contract.get("training_stage") != "offline_full_finetune"
        or contract.get("model_variant") != "force_token_moe"
        or contract.get("gate_contract_version")
        != "v4.2-b4x1-single-pass-exact-resume"
        or contract.get("training_update_algorithm") != "single_pass_batch_local"
        or contract.get("p8_phase_boundary", {}).get("shadow") != "not_implemented_until_P9"
    ):
        raise RuntimeError("P8_CHECKPOINT_CONTRACT_DRIFT")


def _validate_p7_prerequisite(
    root: Path,
    contract: dict,
    *,
    dataset_root: Path,
    repo_id: str,
    binding: dict | None = None,
) -> dict[str, str]:
    prerequisite = contract.get("p7_prerequisite")
    expected_keys = {
        "training_recipe",
        "source_binding",
        "resolved_config",
        "gate_result",
        "validation_fixture",
        "required_gate_status",
        "required_acceptance_status",
        "required_formal_eligible",
    }
    if not isinstance(prerequisite, dict) or set(prerequisite) != expected_keys:
        raise RuntimeError("P8_P7_PREREQUISITE_SPEC_MISSING_OR_DRIFTED")
    if (
        prerequisite["required_gate_status"] != "pass"
        or prerequisite["required_acceptance_status"] != "development_only"
        or prerequisite["required_formal_eligible"] is not False
    ):
        raise RuntimeError("P8_P7_PREREQUISITE_SEMANTICS_DRIFT")
    payloads: dict[str, dict] = {}
    for name in (
        "training_recipe",
        "source_binding",
        "resolved_config",
        "gate_result",
        "validation_fixture",
    ):
        artifact = prerequisite[name]
        if set(artifact) != {"path", "sha256"}:
            raise RuntimeError(f"P8_P7_{name.upper()}_BINDING_DRIFT")
        path = root / artifact["path"]
        if _sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"P8_P7_{name.upper()}_HASH_MISMATCH")
        payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    recipe = payloads["training_recipe"]
    p7_binding = payloads["source_binding"]
    resolved = payloads["resolved_config"]
    result = payloads["gate_result"]
    fixture = payloads["validation_fixture"]
    _validate_recipe(recipe)
    _validate_p7_source_binding(
        root,
        p7_binding,
        dataset_root=dataset_root,
        repo_id=repo_id,
        recipe=recipe,
    )
    if (
        result.get("gate") != "P7"
        or result.get("gate_status") != prerequisite["required_gate_status"]
        or result.get("acceptance_status")
        != prerequisite["required_acceptance_status"]
        or result.get("formal_eligible") is not prerequisite["required_formal_eligible"]
        or result.get("source_binding_sha256")
        != prerequisite["source_binding"]["sha256"]
        or result.get("training_recipe_sha256")
        != prerequisite["training_recipe"]["sha256"]
        or result.get("resolved_config_sha256")
        != prerequisite["resolved_config"]["sha256"]
        or result.get("p8_started") is not False
        or resolved.get("source_binding_sha256")
        != prerequisite["source_binding"]["sha256"]
        or resolved.get("training_recipe_sha256")
        != prerequisite["training_recipe"]["sha256"]
        or resolved.get("fixed_validation_fixture_sha256")
        != prerequisite["validation_fixture"]["sha256"]
        or fixture.get("acceptance_status") != "development_only"
        or fixture.get("formal_eligible") is not False
    ):
        raise RuntimeError("P8_PARENT_P7_GATE_NOT_ELIGIBLE")
    if binding is not None and binding.get("p7_prerequisite") != prerequisite:
        raise RuntimeError("P8_SOURCE_BINDING_P7_PREREQUISITE_MISMATCH")
    return {
        name: prerequisite[name]["sha256"]
        for name in (
            "training_recipe",
            "source_binding",
            "resolved_config",
            "gate_result",
            "validation_fixture",
        )
    }


def _validate_p8_source_binding(
    root: Path,
    binding: dict,
    *,
    dataset_root: Path,
    repo_id: str,
    contract: dict,
) -> dict:
    if (
        binding.get("stage") != "P8"
        or binding.get("formal_eligible") is not False
        or binding.get("signature_status") != "development_only_untrusted"
    ):
        raise RuntimeError("P8_SOURCE_BINDING_STATUS_DRIFT")
    _validate_source_binding(root, binding, dataset_root=dataset_root, repo_id=repo_id)
    runtime_imports = _validate_runtime_import_roots(root)
    if binding.get("runtime_imports") != runtime_imports:
        raise RuntimeError("P8_SOURCE_BINDING_RUNTIME_IMPORT_MISMATCH")
    if binding["dataset"].get("storage_tree") != _dataset_storage_binding(dataset_root):
        raise RuntimeError("P8_DATASET_STORAGE_TREE_HASH_MISMATCH")
    parent = _validate_p7_prerequisite(
        root,
        contract,
        dataset_root=dataset_root,
        repo_id=repo_id,
        binding=binding,
    )
    tests = _pytest_evidence_summary(root, binding.get("test_evidence", {}))
    return {
        "p7_prerequisite": parent,
        "dataset_storage_tree_sha256": binding["dataset"]["storage_tree"]["tree_sha256"],
        "dataset_storage_file_count": binding["dataset"]["storage_tree"]["file_count"],
        "pytest": tests,
        "runtime_imports": runtime_imports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-output", type=Path, required=True)
    parser.add_argument("--cold-output", type=Path, required=True)
    parser.add_argument(
        "--source-binding",
        type=Path,
        default=Path(__file__).parents[1] / "artifacts/development/p8_source_binding.json",
    )
    parser.add_argument(
        "--force-parity-fp32",
        type=Path,
        default=Path(__file__).parents[1]
        / "artifacts/development/p8_force_parity_gpu_fp32.json",
    )
    parser.add_argument(
        "--force-parity-bf16",
        type=Path,
        default=Path(__file__).parents[1]
        / "artifacts/development/p8_force_parity_gpu_bf16.json",
    )
    args = parser.parse_args()
    for path in (args.checkpoint, args.output, args.resolved_output, args.cold_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite P8 artifact: {path}")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("PYTHONHASHSEED=42 required before interpreter startup")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required")
    _require_offline()

    import numpy as np
    import torch

    from forcesmolvla.checkpoint import (
        load_offline_base_policy,
        save_p8_training_state,
        sha256_file,
        validate_force_artifact_manifest,
        validate_p8_payload_contract,
        write_development_artifact_manifest,
        write_trainability_manifest,
    )
    from forcesmolvla.configuration_forcesmolvla import FORCE_TOKEN_MOE, OFFLINE_FULL_FINETUNE
    from forcesmolvla.dataset_v3 import load_dataset_split
    from forcesmolvla.router_training import (
        MoEMicrobatch,
        SerializableUniformSampler,
        build_p7_optimizer_and_scheduler,
        single_pass_optimizer_update,
    )
    from forcesmolvla.training_data import load_runtime_artifacts, prepare_training_sample
    from p8_checkpoint_common import compute_fixed_parity, load_fixed_validation_inputs

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")
    gpu_name = torch.cuda.get_device_name(0)
    if "4090 D" not in gpu_name and "4090D" not in gpu_name:
        raise RuntimeError(f"P8_REQUIRES_RTX_4090D: {gpu_name}")
    root = Path(__file__).parents[1].resolve()
    dataset_root = args.dataset_root.resolve()
    binding_path = args.source_binding.resolve()
    contract_path = root / "configs/p8_checkpoint_contract.development.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract)
    conversion = json.loads(
        (dataset_root / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    repo_id = conversion.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("P8_DATASET_REPO_ID_MISSING")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding_evidence = _validate_p8_source_binding(
        root,
        binding,
        dataset_root=dataset_root,
        repo_id=repo_id,
        contract=contract,
    )
    p7_prerequisite = contract["p7_prerequisite"]
    p7_fixture_path = root / p7_prerequisite["validation_fixture"]["path"]
    p7_resolved_path = root / p7_prerequisite["resolved_config"]["path"]
    force_parity = {}
    acceptance_config_sha = _sha256(
        root / "configs/p8_parity_acceptance.development.json"
    )
    for precision, path in (
        ("fp32", args.force_parity_fp32.resolve()),
        ("bf16", args.force_parity_bf16.resolve()),
    ):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("gate") != "P8_force_parity"
            or report.get("gate_status") != "pass"
            or report.get("acceptance_status") != "development_only"
            or report.get("formal_eligible") is not False
            or report.get("precision") != precision
            or report.get("source_binding_sha256") != _sha256(binding_path)
            or report.get("tolerance", {}).get("acceptance_config_sha256")
            != acceptance_config_sha
            or not report.get("structural_contracts_exact")
            or not all(report["structural_contracts_exact"].values())
        ):
            raise RuntimeError(f"P8_FORCE_PARITY_REPORT_INVALID:{precision}")
        force_parity[precision] = {
            "report_sha256": _sha256(path),
            "acceptance_config_sha256": acceptance_config_sha,
            "max_abs_error": report["max_abs_error"],
        }

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    train_dataset = load_dataset_split(
        dataset_root,
        repo_id=repo_id,
        split_name="train",
        artifact_use="development",
        delta_timestamps={"action": [index / 30 for index in range(50)]},
    )
    normalizer = load_runtime_artifacts(
        dataset_root,
        calibration_bundle_path=root / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=root / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=root / "artifacts/development/action_delta_spec.json",
        expected_repo_id=repo_id,
    ).normalizer
    sampler = SerializableUniformSampler(list(range(len(train_dataset))), seed=42)
    sampled_indices = sampler.draw(4)
    prepared_train = [
        prepare_training_sample(train_dataset[index], normalizer) for index in sampled_indices
    ]
    fixture = json.loads(
        p7_fixture_path.read_text(encoding="utf-8")
    )

    with contextlib.redirect_stdout(sys.stderr):
        policy, base_report = load_offline_base_policy(
            root / "assets/base_checkpoint",
            root / "assets/smolvlm_constructor",
            device="cuda",
            training_stage=OFFLINE_FULL_FINETUNE,
            force_variant=FORCE_TOKEN_MOE,
            acceptance_status="development_only",
            force_init_seed=42,
        )
    if not all(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("P8_OFFLINE_FULL_FINETUNE_FROZEN_PARAMETER_DETECTED")
    optimizer, scheduler, optimizer_groups = build_p7_optimizer_and_scheduler(policy)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator(device=device).manual_seed(44)
    batch = _make_batch(policy, prepared_train, device)
    noise = torch.randn(4, 50, 7, generator=generator, device=device, dtype=torch.float32)
    timestep = torch.tensor([0.1, 0.3, 0.6, 0.8], device=device, dtype=torch.float32)
    microbatch = MoEMicrobatch(
        batch=batch,
        noise7=noise,
        time=timestep,
        identity="p8-b4x1-single-pass-update-0",
    )

    policy.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    update_report = single_pass_optimizer_update(
        policy,
        microbatch,
        optimizer,
        scheduler=scheduler,
    )
    if (
        update_report.get("training_update_algorithm") != "single_pass_batch_local"
        or update_report.get("microbatch_count") != 1
        or update_report.get("optimizer_steps") != 1
    ):
        raise RuntimeError("P8_B4X1_SINGLE_PASS_UPDATE_CONTRACT_DRIFT")
    end_event.record()
    torch.cuda.synchronize()
    update_wall_seconds = time.perf_counter() - wall_start
    update_cuda_ms = start_event.elapsed_time(end_event)
    update_peak = {
        "allocated_bytes": torch.cuda.max_memory_allocated(device),
        "reserved_bytes": torch.cuda.max_memory_reserved(device),
    }

    validation_batch, _raw_val, validation_normalizer = load_fixed_validation_inputs(
        policy, dataset_root, fixture, device
    )
    validation_noise = torch.tensor(
        fixture["epsilon7"]["tensor"], dtype=torch.float32, device=device
    )
    validation_time = torch.tensor(
        fixture["time"]["tensor"], dtype=torch.float32, device=device
    )
    validation_scalar_1 = _validation_scalar(
        policy, validation_batch, validation_noise, validation_time
    )
    validation_scalar_2 = _validation_scalar(
        policy, validation_batch, validation_noise, validation_time
    )
    expected_validation = fixture["evaluation"]["after_development_update_L_flow_run_1"]
    if validation_scalar_1 != validation_scalar_2 or validation_scalar_1 != expected_validation:
        raise RuntimeError("P8_FIXED_VALIDATION_REPLAY_MISMATCH")
    parity = compute_fixed_parity(policy, validation_batch, validation_normalizer, fixture)

    checkpoint = args.checkpoint.resolve()
    checkpoint.mkdir(parents=True)
    policy.save_pretrained(checkpoint)
    config_path = checkpoint / "config.json"
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    if saved_config.get("type") != "force_smolvla":
        raise RuntimeError("P8_SAVED_CONFIG_TYPE_MISMATCH")
    saved_config["vlm_model_name"] = "base_assets/smolvlm_constructor"
    saved_config["load_vlm_weights"] = False
    config_path.write_text(json.dumps(saved_config, indent=4) + "\n", encoding="utf-8")
    shutil.copytree(
        root / "assets/smolvlm_constructor",
        checkpoint / "base_assets/smolvlm_constructor",
    )
    copied_payloads = _copy_payloads(
        root,
        dataset_root,
        checkpoint,
        contract=contract,
        p8_source_binding=binding_path,
        force_parity_fp32=args.force_parity_fp32.resolve(),
        force_parity_bf16=args.force_parity_bf16.resolve(),
    )
    trainability = write_trainability_manifest(policy, checkpoint)
    if trainability["frozen_parameters"] != 0:
        raise RuntimeError("P8_TRAINABILITY_MANIFEST_FROZEN_PARAMETER_DETECTED")

    save_p8_training_state(
        checkpoint,
        step=1,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        accumulation_phase=0,
        batch_size=4,
        gradient_accumulation_microbatches=1,
        resume_contract={
            "schema_version": "1.0",
            "acceptance_status": "development_only",
            "formal_eligible": False,
            "source_binding_sha256": _sha256(binding_path),
            "checkpoint_contract_sha256": _sha256(contract_path),
            "optimizer_groups": optimizer_groups,
            "gate_contract_version": contract["gate_contract_version"],
            "training_update_algorithm": "single_pass_batch_local",
        },
    )
    expected_rng = {
        "python": random.random(),
        "numpy": float(np.random.rand()),
        "torch_cpu": torch.rand(4).tolist(),
        "torch_cuda": torch.rand(4, device=device).cpu().tolist(),
    }
    expected_sampler_indices = sampler.draw(4)
    resume_path = checkpoint / "training_state/resume_contract.json"
    resume_contract = json.loads(resume_path.read_text(encoding="utf-8"))
    resume_contract["expected_next_rng"] = expected_rng
    resume_contract["expected_next_sampler_indices"] = expected_sampler_indices
    resume_path.write_text(
        json.dumps(resume_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    parity_reference = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "fixture_sha256": _sha256(p7_fixture_path),
        "parity": parity,
        "detached_signature": None,
        "approval": None,
    }
    (checkpoint / "parity_reference.json").write_text(
        json.dumps(parity_reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    resolved = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "gate": "P8",
        "training_stage": OFFLINE_FULL_FINETUNE,
        "force_variant": FORCE_TOKEN_MOE,
        "all_parameters_trainable": True,
        "total_parameters": trainability["total_parameters"],
        "trainable_name_sha256": trainability["trainable_name_sha256"],
        "force_initialization_tensor_sha256": policy.force_initialization_tensor_hash(),
        "source_binding_sha256": _sha256(binding_path),
        "checkpoint_contract_sha256": _sha256(contract_path),
        "p7_resolved_config_sha256": _sha256(p7_resolved_path),
        "p7_validation_fixture_sha256": _sha256(p7_fixture_path),
        "p7_prerequisite": binding_evidence["p7_prerequisite"],
        "config_sha256": sha256_file(config_path),
        "model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "trainability_manifest_sha256": sha256_file(checkpoint / "trainability_manifest.json"),
        "resume_contract_sha256": sha256_file(resume_path),
        "parity_reference_sha256": sha256_file(checkpoint / "parity_reference.json"),
        "copied_payload_source_sha256": copied_payloads,
        "strict_reload": {
            "strict": True,
            "local_files_only": True,
            "force_download": False,
            "embedded_constructor_assets": True,
        },
        "formal_signature_algorithm": None,
        "formal_key_id": None,
        "formal_approver": None,
        "formal_eligible_reason": "trusted signature fields and approval remain unresolved",
        "p9_started": False,
        "gate_contract_version": contract["gate_contract_version"],
        "training_update_algorithm": "single_pass_batch_local",
        "detached_signature": None,
        "approval": None,
    }
    args.resolved_output.parent.mkdir(parents=True, exist_ok=True)
    args.resolved_output.write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(args.resolved_output, checkpoint / "manifests/p8_resolved_config.json")

    manifest = write_development_artifact_manifest(
        checkpoint,
        metadata={
            "training_stage": OFFLINE_FULL_FINETUNE,
            "force_variant": FORCE_TOKEN_MOE,
            "optimizer_update": 1,
            "gate_contract_version": contract["gate_contract_version"],
            "source_binding_sha256": _sha256(binding_path),
            "resolved_config_sha256": _sha256(args.resolved_output),
            "parity_sha256": parity["parity_sha256"],
        },
    )
    validate_force_artifact_manifest(checkpoint, artifact_use="development")
    validate_p8_payload_contract(checkpoint)
    for required in contract["required_payloads"]:
        if not (checkpoint / required).exists():
            raise RuntimeError(f"P8_REQUIRED_PAYLOAD_MISSING_AFTER_SAVE: {required}")

    del microbatch, validation_batch, prepared_train, train_dataset
    del policy, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory(prefix="forcesmolvla_p8_hf_") as cache_dir:
        cold_env = os.environ.copy()
        cold_env.update(
            {
                "PYTHONHASHSEED": "42",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HOME": cache_dir,
                "HF_HUB_CACHE": str(Path(cache_dir) / "hub"),
                "TRANSFORMERS_CACHE": str(Path(cache_dir) / "transformers"),
                "PYTHONPATH": "",
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "tools/p8_cold_start_worker.py"),
                "--checkpoint",
                str(checkpoint),
                "--dataset-root",
                str(dataset_root),
                "--output",
                str(args.cold_output.resolve()),
            ],
            cwd=root,
            env=cold_env,
            text=True,
            capture_output=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "P8_COLD_START_SUBPROCESS_FAILED\n"
            + completed.stdout[-4000:]
            + "\n"
            + completed.stderr[-8000:]
        )
    cold_result = json.loads(args.cold_output.read_text(encoding="utf-8"))
    if (
        cold_result.get("gate_status") != "pass"
        or cold_result.get("parity_exact") is not True
        or cold_result.get("exact_resume_dry_run") is not True
        or cold_result.get("gate_contract_version") != contract["gate_contract_version"]
    ):
        raise RuntimeError("P8_COLD_START_RESULT_NOT_PASS")

    result = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "gate": "P8",
        "gate_status": "pass",
        "gpu": {"name": gpu_name, "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
        "real_data": {
            "dataset": str(dataset_root),
            "repo_id": repo_id,
            "split": "train",
            "sampled_indices": sampled_indices,
            "batch_per_gpu": 4,
            "microbatches": 1,
            "horizon": 50,
            "camera_count": 2,
        },
        "full_parameter_update": update_report,
        "exact_resume_dry_run": True,
        "force_full_parity": force_parity,
        "long_development_sft_unlocked": True,
        "gate_contract_version": contract["gate_contract_version"],
        "update_latency": {"cuda_ms": update_cuda_ms, "wall_seconds": update_wall_seconds},
        "update_peak_memory": update_peak,
        "fixed_validation": {
            "L_flow_run_1": validation_scalar_1,
            "L_flow_run_2": validation_scalar_2,
            "exact": True,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "artifact_manifest_sha256": sha256_file(checkpoint / "artifact_manifest.json"),
            "payload_count": len(manifest["payloads"]),
            "model_sha256": resolved["model_sha256"],
            "strict_local_only": True,
            "embedded_base_assets": True,
            "optimizer_update": 1,
        },
        "resume": {
            "optimizer_scheduler_scaler_rng_sampler_accumulation": "exact",
            "cross_stage_optimizer_restore": "rejected",
        },
        "cold_start": cold_result,
        "source_binding_evidence": binding_evidence,
        "source_binding_sha256": _sha256(binding_path),
        "resolved_config_sha256": _sha256(args.resolved_output),
        "cpu_fallback_used": False,
        "architecture_downgrade_used": False,
        "robot_actions_sent": 0,
        "p9_started": False,
        "remaining_blockers": [
            "trusted detached signature algorithm/key/approver unresolved",
            "formal checkpoint acceptance remains fail-closed",
            "P9 offline record/replay Shadow is not started and requires approval",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
