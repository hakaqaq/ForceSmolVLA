#!/usr/bin/env python3
"""GPU-only ForceSmolVLA single-pass full SFT."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from forcesmolvla.training_runtime import (
    build_training_batch as _make_batch,
    build_validation_fixture as _build_validation_fixture,
    canonical_sha256 as _canonical_sha256,
    file_sha256 as _sha256,
    require_offline_environment as _require_offline,
    tree_sha256 as _tree_sha256,
    validate_action_target_population_prerequisite,
    validation_scalar as _validation_scalar,
)


EXPECTED_PARAMETERS = 505_620_341
MICROBATCHES = 1
BATCH_SIZE = 4
HORIZON = 50
DATA_LOADING_THREADS = 8
SCHEDULER_PRESET_WARMUP_UPDATES = 1_000
SCHEDULER_PRESET_DECAY_UPDATES = 20_000
SCHEDULER_PEAK_LR = 1e-4
SCHEDULER_FINAL_LR = 2.5e-6


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "name",
        "output_dir",
        "data_scope",
        "dataset_validation",
        "converter_runtime_spec",
        "training_readiness",
        "logging",
    }
    training_readiness = config.get("training_readiness")
    logging = config.get("logging")
    if (
        config.get("schema_version") != "1.0"
        or config.get("acceptance_status") != "development_only"
        or config.get("formal_eligible") is not False
        or not required.issubset(config)
        or not isinstance(config["name"], str)
        or not config["name"]
        or any(
            not isinstance(config.get(field), str) or not config[field]
            for field in (
                "output_dir",
                "data_scope",
                "dataset_validation",
                "converter_runtime_spec",
            )
        )
        or not isinstance(training_readiness, dict)
        or not isinstance(training_readiness.get("gate_report"), str)
        or not isinstance(training_readiness.get("source_binding"), str)
        or not isinstance(logging, dict)
        or not isinstance(logging.get("interval_samples"), int)
        or logging["interval_samples"] <= 0
    ):
        raise RuntimeError("TRAIN_CONFIG_INVALID")
    return config


def _load_data_scope(
    root: Path,
    dataset_root: Path,
    repo_id: str,
    config: dict,
) -> tuple[dict, str]:
    path = _resolve_path(root, config["data_scope"])
    scope_sha256 = _sha256(path)
    scope = json.loads(path.read_text(encoding="utf-8"))
    dataset = scope.get("dataset", {})
    session = scope.get("session_provenance", {})
    budget = scope.get("training_budget", {})
    schedule = budget.get("effective_schedule", {})
    raw_session_path = _resolve_path(root, session.get("raw_session_manifest_path", ""))
    recipe_path = _resolve_path(root, budget.get("recipe_path", ""))
    if (
        scope.get("acceptance_status") != "development_only"
        or scope.get("formal_eligible") is not False
        or dataset_root != _resolve_path(root, dataset.get("path", ""))
        or repo_id != dataset.get("repo_id")
        or _sha256(dataset_root / "conversion_manifest.json")
        != dataset.get("conversion_manifest_sha256")
        or _sha256(raw_session_path) != session.get("raw_session_manifest_sha256")
        or _sha256(recipe_path) != budget.get("recipe_sha256")
        or not isinstance(budget.get("target_samples"), int)
        or budget.get("target_samples", 0) <= 0
        or budget.get("batch_per_gpu") != BATCH_SIZE
        or budget.get("gradient_accumulation_microbatches") != MICROBATCHES
        or budget.get("effective_samples_per_update") != BATCH_SIZE * MICROBATCHES
        or budget.get("target_samples") % (BATCH_SIZE * MICROBATCHES) != 0
        or budget.get("derived_optimizer_updates")
        != budget.get("target_samples") // (BATCH_SIZE * MICROBATCHES)
        or budget.get("checkpoint_policy") != "final_update_only"
        or budget.get("final_checkpoint_training_samples") != budget.get("target_samples")
        or schedule.get("peak_lr") != SCHEDULER_PEAK_LR
        or schedule.get("final_lr") != SCHEDULER_FINAL_LR
        or not isinstance(budget.get("validation_interval_samples"), int)
        or budget.get("validation_interval_samples", 0) <= 0
    ):
        raise RuntimeError("DATA_SCOPE_DRIFT")
    updates = budget["derived_optimizer_updates"]
    expected_warmup = (
        int(SCHEDULER_PRESET_WARMUP_UPDATES * updates / SCHEDULER_PRESET_DECAY_UPDATES)
        if updates < SCHEDULER_PRESET_DECAY_UPDATES
        else SCHEDULER_PRESET_WARMUP_UPDATES
    )
    expected_decay = min(updates, SCHEDULER_PRESET_DECAY_UPDATES)
    if (
        schedule.get("warmup_updates") != expected_warmup
        or schedule.get("decay_end_update") != expected_decay
        or schedule.get("lerobot_short_run_auto_scale")
        != (updates < SCHEDULER_PRESET_DECAY_UPDATES)
    ):
        raise RuntimeError("DATA_SCOPE_SCHEDULE_DRIFT")
    raw_session = json.loads(raw_session_path.read_text(encoding="utf-8"))
    if (
        raw_session.get("raw_format_version") != session.get("raw_format_version")
        or raw_session.get("created_at") != session.get("created_at")
    ):
        raise RuntimeError("RAW_SESSION_PROVENANCE_DRIFT")
    return scope, scope_sha256


def _bind_fixture_provenance(fixture: dict, data_scope: dict) -> None:
    session = data_scope["session_provenance"]
    legacy_session_id = session.get("legacy_fixture_session_id")
    collection_scope_id = session.get("collection_scope_id")
    if legacy_session_id is None or collection_scope_id is None:
        return
    chunk_context = fixture["chunk_context"]
    if set(chunk_context["session_id"]) != {legacy_session_id}:
        raise RuntimeError("VALIDATION_FIXTURE_SESSION_ID_DRIFT")
    chunk_context["session_id"] = [collection_scope_id] * len(
        chunk_context["session_id"]
    )
    for provenance in chunk_context["selected_provenance"]:
        provenance.update(
            {
                "collection_scope_id": collection_scope_id,
                "physical_session_id": None,
                "session_id_semantics": session["collection_scope_id_semantics"],
                "replaced_legacy_fixture_session_id": session[
                    "legacy_fixture_session_id"
                ],
            }
        )
    fixture["chunk_context_sha256"] = _canonical_sha256(chunk_context)


def _final_checkpoint_due(step: int, max_updates: int) -> bool:
    return step == max_updates


def _source_binding(
    root: Path,
    dataset_root: Path,
    repo_id: str,
    config_path: Path,
    config: dict,
    *,
    readiness_report: Path,
    readiness_source_binding: Path,
) -> dict:
    configured_files = [
        config_path,
        _resolve_path(root, config["data_scope"]),
        _resolve_path(root, config["dataset_validation"]),
        _resolve_path(root, config["converter_runtime_spec"]),
    ]
    data_scope = json.loads(configured_files[1].read_text(encoding="utf-8"))
    configured_files.append(
        _resolve_path(root, data_scope["training_budget"]["recipe_path"])
    )
    try:
        configured_relative = [
            path.relative_to(root).as_posix() for path in configured_files
        ]
    except ValueError as error:
        raise RuntimeError("TRAIN_CONFIG_ARTIFACT_OUTSIDE_PROJECT") from error
    files = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "src/forcesmolvla").glob("*.py")
    ) + [
        "src/forcesmolvla/training_runtime.py",
        "tools/action_target_population_parity_gate.py",
        "tools/train_forcesmolvla_sft.py",
        "artifacts/development/action_target_population_parity_r1.json",
        "configs/forcesmolvla_sft_recipe.development.yaml",
        "configs/training_checkpoint_contract.development.json",
        "ForceSmolVLA_Implementation_Spec_v4_2.md",
    ] + configured_relative
    files = sorted(set(files))
    project_hashes = {relative: _sha256(root / relative) for relative in files}
    dataset_hashes = {
        name: _sha256(dataset_root / name)
        for name in (
            "conversion_manifest.json",
            "normalizer_manifest.json",
            "split_manifest.json",
        )
    }
    lerobot_commit = subprocess.run(
        ["git", "-C", str(root / "vendor/lerobot"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lerobot_dirty = subprocess.run(
        ["git", "-C", str(root / "vendor/lerobot"), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if lerobot_dirty:
        raise RuntimeError("LEROBOT_VENDOR_DIRTY_WORKTREE_FORBIDDEN")
    vendor_files = (
        "src/lerobot/policies/smolvla/configuration_smolvla.py",
        "src/lerobot/policies/smolvla/modeling_smolvla.py",
        "src/lerobot/policies/smolvla/smolvlm_with_expert.py",
    )
    return {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "dataset_manifest_sha256": dataset_hashes,
        "experiment_name": config["name"],
        "train_config_sha256": _sha256(config_path),
        "data_scope_sha256": _sha256(_resolve_path(root, config["data_scope"])),
        "project_file_sha256": project_hashes,
        "base_checkpoint_model_sha256": _sha256(
            root / "assets/base_checkpoint/model.safetensors"
        ),
        "base_checkpoint_config_sha256": _sha256(
            root / "assets/base_checkpoint/config.json"
        ),
        "constructor_assets_tree_sha256": _tree_sha256(root / "assets/smolvlm_constructor"),
        "training_readiness_report_sha256": _sha256(readiness_report),
        "training_readiness_source_binding_sha256": _sha256(
            readiness_source_binding
        ),
        "lerobot_commit": lerobot_commit,
        "lerobot_dirty_worktree": False,
        "lerobot_file_sha256": {
            relative: _sha256(root / "vendor/lerobot" / relative)
            for relative in vendor_files
        },
        "detached_signature": None,
        "approval": None,
    }


def _resolved_config(
    root: Path,
    config: dict,
    dataset_root: Path,
    repo_id: str,
    run_root: Path,
    binding_sha256: str,
    budget: dict,
    data_scope: dict,
    data_scope_sha256: str,
) -> dict:
    return {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "run_name": config["name"],
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "output_dir": str(run_root),
        "training_stage": "offline_full_finetune",
        "force_variant": "force_token_moe",
        "all_parameters_trainable": True,
        "expected_parameter_count": EXPECTED_PARAMETERS,
        "training_budget": budget,
        "data_scope_sha256": data_scope_sha256,
        "session_provenance": data_scope["session_provenance"],
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 1e-10,
            "grad_clip_norm": 10.0,
        },
        "batching": {
            "batch_per_gpu": BATCH_SIZE,
            "gradient_accumulation_microbatches": MICROBATCHES,
            "effective_samples_per_update": BATCH_SIZE * MICROBATCHES,
            "horizon": HORIZON,
            "cameras": 2,
            "data_loading_threads": int(
                config.get("dataloader", {}).get("workers", DATA_LOADING_THREADS)
            ),
            "ordered_prefetch_windows": 1,
            "prefetched_indices_saved_in_resume_contract": True,
        },
        "loss": "L_flow + 0.01*L_balance + 0.001*L_z; single shared full forward",
        "training_update_algorithm": "single_pass_batch_local",
        "exact_two_pass_validation_role": "validation_only",
        "checkpoint_interval_samples": budget["checkpoint_interval_samples"],
        "derived_checkpoint_interval_updates": budget[
            "derived_checkpoint_interval_updates"
        ],
        "validation_interval_samples": budget["validation_interval_samples"],
        "derived_validation_interval_updates": budget[
            "derived_validation_interval_updates"
        ],
        "checkpoint_policy": "final_update_only",
        "intermediate_checkpoint_save": False,
        "checkpoint_interval_semantics": (
            "recorded for compatibility; final-only policy saves no intermediate checkpoint"
        ),
        "best_metric_tracking": "fixed single-pass validation L_flow; metrics only",
        "seeds": {"initialization": 42, "validation": 43, "training": 44},
        "source_binding_sha256": binding_sha256,
        "training_readiness_report_sha256": _sha256(
            _resolve_path(root, config["training_readiness"]["gate_report"])
        ),
        "training_readiness_source_binding_sha256": _sha256(
            _resolve_path(root, config["training_readiness"]["source_binding"])
        ),
        "training_readiness_contract_version": (
            "v4.2-b4x1-single-pass-exact-resume"
        ),
        "cpu_fallback": "forbidden",
        "robot_actions_sent": 0,
        "detached_signature": None,
        "approval": None,
    }


def _copy_checkpoint_payloads(
    root: Path,
    dataset_root: Path,
    run_root: Path,
    checkpoint: Path,
    experiment_config_path: Path,
    experiment_config: dict,
) -> None:
    data_scope_path = _resolve_path(root, experiment_config["data_scope"])
    data_scope = json.loads(data_scope_path.read_text(encoding="utf-8"))
    recipe_path = _resolve_path(
        root, data_scope["training_budget"]["recipe_path"]
    )
    sources = {
        "manifests/training_checkpoint_contract.development.json": root
        / "configs/training_checkpoint_contract.development.json",
        "manifests/resolved_training_config.json": run_root / "resolved_training_config.json",
        "manifests/source_binding.json": run_root / "source_binding.json",
        "manifests/implementation_spec_v4_2.md": root
        / "ForceSmolVLA_Implementation_Spec_v4_2.md",
        "manifests/train_config.json": experiment_config_path,
        "manifests/fixed_validation_fixture.json": run_root / "fixed_validation_fixture.json",
        "manifests/dataset_validation.json": _resolve_path(
            root, experiment_config["dataset_validation"]
        ),
        "manifests/normalizer_manifest.json": dataset_root / "normalizer_manifest.json",
        "manifests/conversion_manifest.json": dataset_root / "conversion_manifest.json",
        "manifests/split_manifest.json": dataset_root / "split_manifest.json",
        "manifests/converter_runtime_spec.json": _resolve_path(
            root, experiment_config["converter_runtime_spec"]
        ),
        "manifests/data_scope.json": data_scope_path,
        "manifests/action_delta_spec.json": root / "artifacts/development/action_delta_spec.json",
        "manifests/feature_mask_spec.json": root / "artifacts/development/feature_mask_spec.json",
        "manifests/processor_graph_manifest.json": root
        / "artifacts/development/processor_graph_manifest.json",
        "manifests/visual_language_manifest.json": root
        / "artifacts/development/visual_language_manifest.json",
        "manifests/wrench_geometry_spec.development.json": root
        / "configs/wrench_geometry_spec.development.json",
        "manifests/calibration_bundle.development.json": root
        / "configs/calibration_bundle.development.json",
        "manifests/training_stage.development.json": root
        / "configs/training_stage.development.json",
        "manifests/forcesmolvla_sft_recipe.development.yaml": root
        / "configs/forcesmolvla_sft_recipe.development.yaml",
        "manifests/offline_sft_training_recipe.development.yaml": recipe_path,
        "manifests/parity_acceptance.development.json": root
        / "configs/parity_acceptance.development.json",
        "manifests/environment_manifest.json": root
        / "artifacts/development/environment_manifest.json",
        "environment/conda-explicit.txt": root / "environment-manifest/conda-explicit.txt",
        "environment/conda-from-history.yml": root
        / "environment-manifest/conda-from-history.yml",
        "environment/pip-freeze.txt": root / "environment-manifest/pip-freeze.txt",
        "environment/requirements.lock": root / "environment-manifest/requirements.lock",
    }
    for relative, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint payload source missing: {source}")
        target = checkpoint / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _save_checkpoint(
    *,
    root: Path,
    dataset_root: Path,
    run_root: Path,
    policy,
    optimizer,
    scheduler,
    scaler,
    sampler,
    step: int,
    update_report: dict,
    validation_loss: float,
    optimizer_groups: dict,
    prefetched_sample_indices: list[int],
    experiment_config_path: Path,
    experiment_config: dict,
    repo_id: str,
) -> Path:
    from forcesmolvla.checkpoint import (
        save_sft_training_state,
        validate_force_artifact_manifest,
        validate_training_payload_contract,
        write_development_artifact_manifest,
        write_trainability_manifest,
    )

    source_binding = json.loads((run_root / "source_binding.json").read_text())

    checkpoints = run_root / "checkpoints"
    target = checkpoints / f"step_{step:06d}"
    temporary = checkpoints / f".step_{step:06d}.tmp"
    if target.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    temporary.mkdir(parents=True)
    try:
        policy.save_pretrained(temporary)
        model_config_path = temporary / "config.json"
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        model_config["vlm_model_name"] = "base_assets/smolvlm_constructor"
        model_config["load_vlm_weights"] = False
        model_config_path.write_text(
            json.dumps(model_config, indent=4) + "\n", encoding="utf-8"
        )
        shutil.copytree(
            root / "assets/smolvlm_constructor",
            temporary / "base_assets/smolvlm_constructor",
        )
        _copy_checkpoint_payloads(
            root,
            dataset_root,
            run_root,
            temporary,
            experiment_config_path,
            experiment_config,
        )
        trainability = write_trainability_manifest(policy, temporary)
        if (
            trainability["total_parameters"] != EXPECTED_PARAMETERS
            or trainability["frozen_parameters"] != 0
        ):
            raise RuntimeError("FULL_FINETUNE_TRAINABILITY_DRIFT")
        save_sft_training_state(
            temporary,
            step=step,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
            accumulation_phase=0,
            batch_size=BATCH_SIZE,
            gradient_accumulation_microbatches=MICROBATCHES,
            resume_contract={
                "schema_version": "1.0",
                "acceptance_status": "development_only",
                "formal_eligible": False,
                "artifact_type": "forcesmolvla_training_checkpoint",
                "source_binding_sha256": _sha256(run_root / "source_binding.json"),
                "dataset_manifest_sha256": source_binding["dataset_manifest_sha256"],
                "resolved_training_config_sha256": _sha256(
                    run_root / "resolved_training_config.json"
                ),
                "optimizer_groups": optimizer_groups,
                "prefetched_sample_indices": prefetched_sample_indices,
            },
        )
        write_development_artifact_manifest(
            temporary,
            artifact_type="forcesmolvla_training_checkpoint",
            metadata={
                "experiment": experiment_config["name"],
                "dataset": repo_id,
                "training_stage": "offline_full_finetune",
                "force_variant": "force_token_moe",
                "training_samples_seen": step * BATCH_SIZE * MICROBATCHES,
                "derived_optimizer_update": step,
                "validation_L_flow": validation_loss,
                "update_report": update_report,
            },
        )
        validate_force_artifact_manifest(temporary, artifact_use="development")
        validate_training_payload_contract(temporary)
        temporary.rename(target)
    except BaseException:
        raise
    _write_json(
        run_root / "latest_checkpoint.json",
        {
            "acceptance_status": "development_only",
            "training_samples_seen": step * BATCH_SIZE * MICROBATCHES,
            "derived_optimizer_update": step,
            "checkpoint": str(target),
            "artifact_manifest_sha256": _sha256(target / "artifact_manifest.json"),
        },
    )
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ForceSmolVLA from a LeRobot v3 dataset and experiment config."
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="LeRobot v3 dataset directory"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="training experiment JSON"
    )
    parser.add_argument("--resume", type=Path, help="checkpoint to resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("PYTHONHASHSEED=42 required before interpreter startup")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required")
    _require_offline()

    import numpy as np
    import torch

    from forcesmolvla.checkpoint import load_offline_base_policy, load_sft_training_state
    from forcesmolvla.configuration_forcesmolvla import FORCE_TOKEN_MOE, OFFLINE_FULL_FINETUNE
    from forcesmolvla.dataset_v3 import load_dataset_split
    from forcesmolvla.router_training import (
        MoEMicrobatch,
        SerializableUniformSampler,
        build_sft_optimizer_and_scheduler,
        derive_optimizer_updates,
        single_pass_optimizer_update,
    )
    from forcesmolvla.training_data import load_runtime_artifacts, prepare_training_sample
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")
    gpu_name = torch.cuda.get_device_name(0)
    root = Path(__file__).parents[1].resolve()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    dataset_root = args.dataset.resolve()
    run_root = _resolve_path(root, config["output_dir"])
    conversion = json.loads((dataset_root / "conversion_manifest.json").read_text())
    repo_id = conversion.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("DATASET_REPO_ID_MISSING")
    validate_action_target_population_prerequisite(root, dataset_root)
    data_scope, data_scope_sha256 = _load_data_scope(
        root, dataset_root, repo_id, config
    )
    if (
        conversion.get("artifact_status") != "development_only"
        or conversion.get("formal_ready") is not False
        or len(conversion.get("episodes", ())) < 3
    ):
        raise RuntimeError("CONVERSION_MANIFEST_GATE_FAILED")
    readiness_report_path = _resolve_path(
        root, config["training_readiness"]["gate_report"]
    )
    readiness_report = json.loads(
        readiness_report_path.read_text(encoding="utf-8")
    )
    readiness_binding_path = _resolve_path(
        root, config["training_readiness"]["source_binding"]
    )
    checkpoint_path = Path(
        readiness_report.get("checkpoint", {}).get("path", "")
    )
    if (
        readiness_report.get("gate") != "P8"  # persisted artifact ABI
        or readiness_report.get("gate_status") != "pass"
        or readiness_report.get("acceptance_status") != "development_only"
        or readiness_report.get("formal_eligible") is not False
        or readiness_report.get("gate_contract_version")
        != "v4.2-b4x1-single-pass-exact-resume"
        or readiness_report.get("exact_resume_dry_run") is not True
        or readiness_report.get("long_development_sft_unlocked") is not True
        or set(readiness_report.get("force_full_parity", {}))
        != {"fp32", "bf16"}
        or readiness_report.get("real_data", {}).get("repo_id") != repo_id
        or readiness_report.get("real_data", {}).get("batch_per_gpu")
        != BATCH_SIZE
        or readiness_report.get("real_data", {}).get("microbatches")
        != MICROBATCHES
        or readiness_report.get("source_binding_sha256")
        != _sha256(readiness_binding_path)
        or not checkpoint_path.is_dir()
        or not (checkpoint_path / "artifact_manifest.json").is_file()
    ):
        raise RuntimeError("LONG_SFT_REQUIRES_CURRENT_P8_B4X1_EXACT_RESUME_GATE")
    if run_root.exists() and args.resume is None:
        raise FileExistsError(f"refusing to overwrite training run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    binding = _source_binding(
        root,
        dataset_root,
        repo_id,
        config_path,
        config,
        readiness_report=readiness_report_path,
        readiness_source_binding=readiness_binding_path,
    )
    binding_path = run_root / "source_binding.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise RuntimeError("TRAINING_SOURCE_BINDING_DRIFT")
    else:
        _write_json(binding_path, binding)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    delta_timestamps = {"action": [index / 30 for index in range(HORIZON)]}
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
    if len(train_dataset) < BATCH_SIZE * MICROBATCHES or len(val_dataset) < 2:
        raise RuntimeError("DATASET_TOO_SMALL")
    runtime_artifacts = load_runtime_artifacts(
        dataset_root,
        calibration_bundle_path=root / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=root / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=root / "artifacts/development/action_delta_spec.json",
        expected_repo_id=repo_id,
    )
    normalizer = runtime_artifacts.normalizer
    sampler = SerializableUniformSampler(list(range(len(train_dataset))), seed=42)
    effective_samples_per_update = BATCH_SIZE * MICROBATCHES
    scope_budget = data_scope["training_budget"]
    target_samples = scope_budget["target_samples"]
    checkpoint_interval_samples = scope_budget.get(
        "checkpoint_interval_samples",
        scope_budget.get("legacy_recipe_checkpoint_interval_samples", target_samples),
    )
    validation_interval_samples = scope_budget["validation_interval_samples"]
    log_interval_samples = config["logging"].get("interval_samples")
    if (
        not isinstance(log_interval_samples, int)
        or log_interval_samples <= 0
        or checkpoint_interval_samples <= 0
    ):
        raise RuntimeError("TRAIN_SAMPLE_INTERVAL_INVALID")
    max_updates = derive_optimizer_updates(
        target_samples, effective_samples_per_update
    )
    if max_updates < SCHEDULER_PRESET_DECAY_UPDATES:
        effective_warmup_updates = int(
            SCHEDULER_PRESET_WARMUP_UPDATES
            * max_updates
            / SCHEDULER_PRESET_DECAY_UPDATES
        )
        effective_decay_updates = max_updates
    else:
        effective_warmup_updates = SCHEDULER_PRESET_WARMUP_UPDATES
        effective_decay_updates = SCHEDULER_PRESET_DECAY_UPDATES
    checkpoint_interval_updates = derive_optimizer_updates(
        checkpoint_interval_samples, effective_samples_per_update
    )
    validation_interval_updates = derive_optimizer_updates(
        validation_interval_samples, effective_samples_per_update
    )
    log_interval_updates = derive_optimizer_updates(
        log_interval_samples, effective_samples_per_update
    )
    budget = {
        "primary_unit": "samples",
        "target_samples": target_samples,
        "resolved_train_split_samples": len(train_dataset),
        "target_equivalent_epochs": target_samples / len(train_dataset),
        "effective_samples_per_update": effective_samples_per_update,
        "derived_optimizer_updates": max_updates,
        "scheduler_preset_warmup_updates": SCHEDULER_PRESET_WARMUP_UPDATES,
        "scheduler_preset_decay_updates": SCHEDULER_PRESET_DECAY_UPDATES,
        "lerobot_short_run_auto_scale": max_updates < SCHEDULER_PRESET_DECAY_UPDATES,
        "effective_warmup_updates": effective_warmup_updates,
        "effective_decay_end_update": effective_decay_updates,
        "scheduler_peak_lr": SCHEDULER_PEAK_LR,
        "scheduler_final_lr": SCHEDULER_FINAL_LR,
        "derived_checkpoint_interval_updates": checkpoint_interval_updates,
        "derived_validation_interval_updates": validation_interval_updates,
        "derived_log_interval_updates": log_interval_updates,
        "checkpoint_interval_samples": checkpoint_interval_samples,
        "validation_interval_samples": validation_interval_samples,
        "log_interval_samples": log_interval_samples,
    }
    offline_recipe_path = _resolve_path(
        root, data_scope["training_budget"]["recipe_path"]
    )
    offline_recipe = json.loads(offline_recipe_path.read_text())
    if (
        offline_recipe.get("training_stage") != "offline_full_finetune"
        or offline_recipe.get("all_existing_parameters_require_grad") is not True
        or offline_recipe["schedule"]["primary_budget_unit"] != "samples"
        or offline_recipe["schedule"]["target_samples"] != target_samples
        or offline_recipe["schedule"]["derived_optimizer_updates"] != max_updates
        or offline_recipe["schedule"]["derived_warmup_updates"]
        != effective_warmup_updates
        or offline_recipe["schedule"]["derived_decay_end_update"]
        != effective_decay_updates
        or offline_recipe["schedule"]["peak_lr"] != SCHEDULER_PEAK_LR
        or offline_recipe["schedule"]["decay_lr"] != SCHEDULER_FINAL_LR
        or offline_recipe["batching"]["effective_samples_per_gpu_update"]
        != effective_samples_per_update
        or offline_recipe["checkpoint_interval_samples"]
        != checkpoint_interval_samples
        or offline_recipe["validation_interval_samples"]
        != validation_interval_samples
        or offline_recipe["loss"]["router_algorithm"] != "single_pass_batch_local"
        or offline_recipe["optimizer"].get("parameter_partition")
        != "each_trainable_parameter_exactly_once"
        or "learned_action_slot" not in offline_recipe["optimizer"].get("no_decay", ())
        or offline_recipe["exact_two_pass_validation"]["active_sft_loop"] is not False
        or offline_recipe["exact_two_pass_validation"]["long_running_sft_allowed"]
        is not False
    ):
        raise RuntimeError("OFFLINE_SINGLE_PASS_SAMPLE_BUDGET_CONTRACT_DRIFT")
    resolved = _resolved_config(
        root,
        config,
        dataset_root,
        repo_id,
        run_root,
        _sha256(binding_path),
        budget,
        data_scope,
        data_scope_sha256,
    )
    resolved_path = run_root / "resolved_training_config.json"
    if resolved_path.exists():
        if json.loads(resolved_path.read_text()) != resolved:
            raise RuntimeError("RESOLVED_TRAINING_CONFIG_DRIFT")
    else:
        _write_json(resolved_path, resolved)

    with contextlib.redirect_stdout(sys.stderr):
        if args.resume is None:
            policy, base_report = load_offline_base_policy(
                root / "assets/base_checkpoint",
                root / "assets/smolvlm_constructor",
                device="cuda",
                training_stage=OFFLINE_FULL_FINETUNE,
                force_variant=FORCE_TOKEN_MOE,
                acceptance_status="development_only",
                force_init_seed=42,
            )
        else:
            from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy

            policy = ForceSmolVLAPolicy.from_pretrained(
                args.resume.resolve(),
                local_files_only=True,
                force_download=False,
                strict=True,
                artifact_use="development",
            )
            base_report = None
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    if parameter_count != EXPECTED_PARAMETERS or not all(
        parameter.requires_grad for parameter in policy.parameters()
    ):
        raise RuntimeError("OFFLINE_FULL_FINETUNE_PARAMETER_GATE_FAILED")
    optimizer, scheduler, optimizer_groups = build_sft_optimizer_and_scheduler(
        policy, derived_optimizer_updates=max_updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_step = 0
    resume_contract = {}
    if args.resume is not None:
        start_step, resume_contract = load_sft_training_state(
            args.resume.resolve(),
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
            batch_size=BATCH_SIZE,
            gradient_accumulation_microbatches=MICROBATCHES,
            expected_resume_contract={
                "source_binding_sha256": _sha256(binding_path),
                "dataset_manifest_sha256": binding["dataset_manifest_sha256"],
                "resolved_training_config_sha256": _sha256(resolved_path),
                "optimizer_groups": optimizer_groups,
            },
        )
    if start_step >= max_updates:
        raise RuntimeError("RESUME_CHECKPOINT_ALREADY_AT_OR_BEYOND_SAMPLE_BUDGET")

    raw_val = [val_dataset[0], val_dataset[1]]
    prepared_val = [prepare_training_sample(sample, normalizer) for sample in raw_val]
    validation_batch = _make_batch(policy, prepared_val, device)
    validation_generator = torch.Generator(device=device).manual_seed(43)
    validation_noise = torch.randn(
        2, HORIZON, 7, generator=validation_generator, device=device, dtype=torch.float32
    )
    validation_time = torch.tensor([0.25, 0.75], device=device, dtype=torch.float32)
    fixture = _build_validation_fixture(
        root=root,
        dataset_root=dataset_root,
        raw_samples=raw_val,
        batch=validation_batch,
        noise=validation_noise,
        timestep=validation_time,
    )
    _bind_fixture_provenance(fixture, data_scope)
    fixture_path = run_root / "fixed_validation_fixture.json"
    if fixture_path.exists():
        if json.loads(fixture_path.read_text()) != fixture:
            raise RuntimeError("FIXED_VALIDATION_FIXTURE_DRIFT")
    else:
        _write_json(fixture_path, fixture)
    initial_validation = _validation_scalar(
        policy, validation_batch, validation_noise, validation_time
    )
    if start_step == 0:
        random.seed(44)
        np.random.seed(44)
        torch.manual_seed(44)
        torch.cuda.manual_seed(44)
        torch.cuda.manual_seed_all(44)
    print(
        json.dumps(
            {
                "event": "training_start",
                "acceptance_status": "development_only",
                "gpu": gpu_name,
                "dataset": str(dataset_root),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "episodes": len(conversion["episodes"]),
                "excluded_episodes": len(conversion.get("excluded_episodes", ())),
                "parameters": parameter_count,
                "trainable_parameters": parameter_count,
                "training_samples_seen": start_step * effective_samples_per_update,
                "target_samples": target_samples,
                "equivalent_epochs_seen": (
                    start_step * effective_samples_per_update / len(train_dataset)
                ),
                "target_equivalent_epochs": budget["target_equivalent_epochs"],
                "derived_start_optimizer_update": start_step,
                "derived_optimizer_updates": max_updates,
                "initial_validation_L_flow": initial_validation,
                "force_initialization_tensor_sha256": policy.force_initialization_tensor_hash(),
                "base_load": None if base_report is None else base_report.to_dict(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    metrics_path = run_root / "metrics.jsonl"
    best_path = run_root / "best_checkpoint.json"
    if best_path.exists():
        previous_best = json.loads(best_path.read_text(encoding="utf-8"))
        best_loss = float(previous_best["validation_L_flow"])
        best_step = int(previous_best["derived_optimizer_update"])
    else:
        best_loss = float("inf")
        best_step = None
    wall_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    data_loading_threads = int(config.get("dataloader", {}).get("workers", DATA_LOADING_THREADS))
    if data_loading_threads <= 0:
        raise RuntimeError("DATALOADER_WORKERS_INVALID")
    data_pool = ThreadPoolExecutor(
        max_workers=data_loading_threads,
        thread_name_prefix=f"{config['name']}_decode",
    )

    def prepare_index(index: int) -> dict:
        return prepare_training_sample(train_dataset[index], normalizer)

    def submit_window(indices: list[int]) -> list[Future]:
        return [data_pool.submit(prepare_index, index) for index in indices]

    if args.resume is None:
        sampled_indices = sampler.draw(BATCH_SIZE * MICROBATCHES)
    else:
        sampled_indices = [
            int(value) for value in resume_contract.get("prefetched_sample_indices", ())
        ]
        if len(sampled_indices) != BATCH_SIZE * MICROBATCHES:
            raise RuntimeError("RESUME_PREFETCHED_SAMPLE_INDICES_INVALID")
    prepared_futures = submit_window(sampled_indices)
    for step in range(start_step + 1, max_updates + 1):
        update_start = time.perf_counter()
        data_wait_start = time.perf_counter()
        prepared = [future.result() for future in prepared_futures]
        data_wait_seconds = time.perf_counter() - data_wait_start
        if step < max_updates:
            next_sampled_indices = sampler.draw(BATCH_SIZE * MICROBATCHES)
            next_prepared_futures = submit_window(next_sampled_indices)
        else:
            next_sampled_indices = []
            next_prepared_futures = []
        batch_prepare_start = time.perf_counter()
        microbatches = []
        for microbatch_index in range(MICROBATCHES):
            batch = _make_batch(
                policy,
                prepared[
                    microbatch_index * BATCH_SIZE : (microbatch_index + 1) * BATCH_SIZE
                ],
                device,
            )
            noise = torch.randn(
                BATCH_SIZE, HORIZON, 7, device=device, dtype=torch.float32
            )
            timestep = policy.model.sample_time(BATCH_SIZE, device)
            microbatches.append(
                MoEMicrobatch(
                    batch=batch,
                    noise7=noise,
                    time=timestep,
                    identity=(
                        f"{config['name']}-step-{step}-microbatch-{microbatch_index}"
                    ),
                )
            )
        batch_prepare_seconds = time.perf_counter() - batch_prepare_start
        policy.train()
        optimizer_update_start = time.perf_counter()
        report = single_pass_optimizer_update(
            policy,
            microbatches[0],
            optimizer,
            scheduler=scheduler,
            grad_clip_norm=10.0,
        )
        torch.cuda.synchronize(device)
        optimizer_update_seconds = time.perf_counter() - optimizer_update_start
        update_seconds = time.perf_counter() - update_start
        validation_loss = None
        if (
            step % validation_interval_updates == 0
            or step == 1
            or step == max_updates
        ):
            validation_loss = _validation_scalar(
                policy, validation_batch, validation_noise, validation_time
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_step = step
        record = {
            "training_samples_seen": step * effective_samples_per_update,
            "equivalent_epochs_seen": (
                step * effective_samples_per_update / len(train_dataset)
            ),
            "derived_optimizer_update": step,
            "train_total": report["backward_total_sum"],
            "train_flow": report["backward_flow_sum"],
            "train_balance": report["backward_balance_sum"],
            "train_z": report["backward_z_sum"],
            "gradient_norm_before_clip": report["gradient_norm_before_clip"],
            "route_counts": report["route_counts"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_L_flow": validation_loss,
            "sample_indices_sha256": hashlib.sha256(
                json.dumps(sampled_indices, separators=(",", ":")).encode()
            ).hexdigest(),
            "data_wait_seconds": data_wait_seconds,
            "batch_prepare_seconds": batch_prepare_seconds,
            "optimizer_update_seconds": optimizer_update_seconds,
            "update_seconds": update_seconds,
            "elapsed_seconds": time.perf_counter() - wall_start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "acceptance_status": "development_only",
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if step % log_interval_updates == 0 or step == 1:
            remaining = max_updates - step
            eta_seconds = remaining * (record["elapsed_seconds"] / (step - start_step))
            print(
                json.dumps(
                    {
                        "event": "update",
                        **record,
                        "eta_seconds": eta_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        checkpoint_due = _final_checkpoint_due(step, max_updates)
        if checkpoint_due:
            if validation_loss is None:
                validation_loss = _validation_scalar(
                    policy, validation_batch, validation_noise, validation_time
                )
            checkpoint = _save_checkpoint(
                root=root,
                dataset_root=dataset_root,
                run_root=run_root,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                sampler=sampler,
                step=step,
                update_report=report,
                validation_loss=validation_loss,
                optimizer_groups=optimizer_groups,
                prefetched_sample_indices=next_sampled_indices,
                experiment_config_path=config_path,
                experiment_config=config,
                repo_id=repo_id,
            )
            if best_step == step:
                _write_json(
                    run_root / "best_checkpoint.json",
                    {
                        "acceptance_status": "development_only",
                        "selection_metric": "fixed_single_pass_validation_L_flow",
                        "validation_algorithm": "single_pass_batch_local",
                        "training_samples_seen": step * effective_samples_per_update,
                        "derived_optimizer_update": step,
                        "validation_L_flow": validation_loss,
                        "checkpoint": str(checkpoint),
                    },
                )
            print(
                json.dumps(
                    {
                        "event": "checkpoint_saved",
                        "training_samples_seen": step * effective_samples_per_update,
                        "derived_optimizer_update": step,
                        "checkpoint": str(checkpoint),
                        "validation_L_flow": validation_loss,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del microbatches, prepared
        gc.collect()
        sampled_indices = next_sampled_indices
        prepared_futures = next_prepared_futures

    data_pool.shutdown(wait=True)

    summary = {
        "acceptance_status": "development_only",
        "status": "complete",
        "primary_budget_unit": "samples",
        "training_samples_seen": target_samples,
        "equivalent_epochs_seen": target_samples / len(train_dataset),
        "derived_optimizer_updates": max_updates,
        "best_training_samples_seen": (
            None if best_step is None else best_step * effective_samples_per_update
        ),
        "derived_best_optimizer_update": best_step,
        "best_validation_L_flow": best_loss,
        "elapsed_seconds": time.perf_counter() - wall_start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "robot_actions_sent": 0,
    }
    _write_json(run_root / "training_summary.json", summary)
    print(json.dumps({"event": "training_complete", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
