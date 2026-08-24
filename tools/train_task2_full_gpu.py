#!/usr/bin/env python3
"""GPU-only ForceVLA x SmolVLA single-pass full SFT on task2_lerobotv3."""

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

from preflight_p5_dense_compute_gpu import _require_offline, _sha256
from preflight_p7_two_pass_gpu import (
    _build_validation_fixture,
    _canonical_sha256,
    _make_batch,
    _validation_scalar,
)


EXPECTED_GPU_NAMES = ("4090 D", "4090D")
EXPECTED_PARAMETERS = 505_620_341
MICROBATCHES = 1
BATCH_SIZE = 4
HORIZON = 50
DATA_LOADING_THREADS = 8
DEFAULT_TARGET_SAMPLES = 40_000
DEFAULT_CHECKPOINT_INTERVAL_SAMPLES = 2_000
DEFAULT_VALIDATION_INTERVAL_SAMPLES = 2_000
DEFAULT_LOG_INTERVAL_SAMPLES = 40
SCHEDULER_PRESET_WARMUP_UPDATES = 1_000
SCHEDULER_PRESET_DECAY_UPDATES = 20_000
SCHEDULER_PEAK_LR = 1e-4
SCHEDULER_FINAL_LR = 2.5e-6


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode() + b"\0")
        digest.update(_sha256(path).encode() + b"\n")
    return digest.hexdigest()


def _load_task2_data_scope(
    root: Path, dataset_root: Path, args: argparse.Namespace
) -> tuple[dict, str]:
    path = root / "configs/task2_development_data_scope.json"
    scope_sha256 = _sha256(path)
    scope = json.loads(path.read_text(encoding="utf-8"))
    dataset = scope.get("dataset", {})
    session = scope.get("session_provenance", {})
    budget = scope.get("training_budget", {})
    schedule = budget.get("effective_schedule", {})
    raw_session_path = Path(session.get("raw_session_manifest_path", ""))
    recipe_path = root / budget.get("recipe_path", "")
    if (
        scope.get("acceptance_status") != "development_only"
        or scope.get("formal_eligible") is not False
        or dataset_root != (root / dataset.get("path", "")).resolve()
        or args.repo_id != dataset.get("repo_id")
        or _sha256(dataset_root / "conversion_manifest.json")
        != dataset.get("conversion_manifest_sha256")
        or _sha256(raw_session_path) != session.get("raw_session_manifest_sha256")
        or session.get("explicit_physical_session_id") is not None
        or session.get("physical_session_id_status") != "not_recorded_in_source"
        or session.get("legacy_fixture_session_id_status")
        != "invalid_legacy_metadata_for_task2"
        or _sha256(recipe_path) != budget.get("recipe_sha256")
        or budget.get("target_samples") != DEFAULT_TARGET_SAMPLES
        or args.target_samples != budget.get("target_samples")
        or budget.get("batch_per_gpu") != BATCH_SIZE
        or budget.get("gradient_accumulation_microbatches") != MICROBATCHES
        or budget.get("effective_samples_per_update") != BATCH_SIZE * MICROBATCHES
        or budget.get("derived_optimizer_updates")
        != DEFAULT_TARGET_SAMPLES // (BATCH_SIZE * MICROBATCHES)
        or budget.get("checkpoint_policy") != "final_update_only"
        or budget.get("final_checkpoint_training_samples") != DEFAULT_TARGET_SAMPLES
        or schedule.get("warmup_updates") != 500
        or schedule.get("decay_end_update")
        != DEFAULT_TARGET_SAMPLES // (BATCH_SIZE * MICROBATCHES)
        or schedule.get("peak_lr") != SCHEDULER_PEAK_LR
        or schedule.get("final_lr") != SCHEDULER_FINAL_LR
        or schedule.get("lerobot_short_run_auto_scale") is not True
        or budget.get("legacy_recipe_checkpoint_interval_samples")
        != DEFAULT_CHECKPOINT_INTERVAL_SAMPLES
    ):
        raise RuntimeError("TASK2_DEVELOPMENT_DATA_SCOPE_DRIFT")
    raw_session = json.loads(raw_session_path.read_text(encoding="utf-8"))
    if (
        "session_id" in raw_session
        or raw_session.get("raw_format_version") != session.get("raw_format_version")
        or raw_session.get("created_at") != session.get("created_at")
    ):
        raise RuntimeError("TASK2_RAW_SESSION_PROVENANCE_DRIFT")
    return scope, scope_sha256


def _bind_task2_fixture_provenance(fixture: dict, data_scope: dict) -> None:
    session = data_scope["session_provenance"]
    chunk_context = fixture["chunk_context"]
    if set(chunk_context["session_id"]) != {session["legacy_fixture_session_id"]}:
        raise RuntimeError("TASK2_VALIDATION_FIXTURE_LEGACY_SESSION_ID_DRIFT")
    collection_scope_id = session["collection_scope_id"]
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
    *,
    p8_gate_report: Path,
    p8_source_binding: Path,
) -> dict:
    files = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "src/forcesmolvla").glob("*.py")
    ) + [
        "tools/preflight_p5_dense_compute_gpu.py",
        "tools/preflight_p7_two_pass_gpu.py",
        "tools/action_target_population_parity_gate.py",
        "tools/train_task2_full_gpu.py",
        "artifacts/development/action_target_population_parity_r1.json",
        "artifacts/development/task2_lerobotv3_validation.json",
        "configs/p7_training_recipe.development.yaml",
        "configs/offline_sft_training_recipe.development.yaml",
        "configs/training_checkpoint_contract.development.json",
        "configs/converter_runtime_spec.task2.development.json",
        "configs/task2_development_data_scope.json",
        "ForceSmolVLA_Implementation_Spec_v4_2.md",
    ]
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
        "task2_data_scope_sha256": _sha256(
            root / "configs/task2_development_data_scope.json"
        ),
        "project_file_sha256": project_hashes,
        "base_checkpoint_model_sha256": _sha256(
            root / "assets/base_checkpoint/model.safetensors"
        ),
        "base_checkpoint_config_sha256": _sha256(
            root / "assets/base_checkpoint/config.json"
        ),
        "constructor_assets_tree_sha256": _tree_sha256(root / "assets/smolvlm_constructor"),
        "p8_gate_report_sha256": _sha256(p8_gate_report),
        "p8_source_binding_sha256": _sha256(p8_source_binding),
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
    args: argparse.Namespace,
    binding_sha256: str,
    budget: dict,
    data_scope: dict,
    data_scope_sha256: str,
) -> dict:
    return {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "run_name": "task2_lerobotv3_full_finetune",
        "dataset_root": str(args.dataset_root.resolve()),
        "repo_id": args.repo_id,
        "training_stage": "offline_full_finetune",
        "force_variant": "force_token_moe",
        "all_parameters_trainable": True,
        "expected_parameter_count": EXPECTED_PARAMETERS,
        "training_budget": budget,
        "task2_data_scope_sha256": data_scope_sha256,
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
            "data_loading_threads": DATA_LOADING_THREADS,
            "ordered_prefetch_windows": 1,
            "prefetched_indices_saved_in_resume_contract": True,
        },
        "loss": "L_flow + 0.01*L_balance + 0.001*L_z; single shared full forward",
        "training_update_algorithm": "single_pass_batch_local",
        "p7_exact_two_pass_role": "gate_only",
        "checkpoint_interval_samples": args.checkpoint_interval_samples,
        "derived_checkpoint_interval_updates": budget[
            "derived_checkpoint_interval_updates"
        ],
        "validation_interval_samples": args.validation_interval_samples,
        "derived_validation_interval_updates": budget[
            "derived_validation_interval_updates"
        ],
        "checkpoint_policy": "final_update_only",
        "intermediate_checkpoint_save": False,
        "checkpoint_interval_semantics": (
            "legacy frozen recipe field; ignored for saving by final-only task2 override"
        ),
        "best_metric_tracking": "fixed single-pass validation L_flow; metrics only",
        "seeds": {"initialization": 42, "validation": 43, "training": 44},
        "source_binding_sha256": binding_sha256,
        "p8_gate_report_sha256": _sha256(args.p8_gate_report.resolve()),
        "p8_source_binding_sha256": _sha256(args.p8_source_binding.resolve()),
        "p8_gate_contract_version": "v4.2-b4x1-single-pass-exact-resume",
        "cpu_fallback": "forbidden",
        "robot_actions_sent": 0,
        "detached_signature": None,
        "approval": None,
    }


def _copy_checkpoint_payloads(
    root: Path, dataset_root: Path, run_root: Path, checkpoint: Path
) -> None:
    sources = {
        "manifests/training_checkpoint_contract.development.json": root
        / "configs/training_checkpoint_contract.development.json",
        "manifests/resolved_training_config.json": run_root / "resolved_training_config.json",
        "manifests/source_binding.json": run_root / "source_binding.json",
        "manifests/implementation_spec_v4_2.md": root
        / "ForceSmolVLA_Implementation_Spec_v4_2.md",
        "manifests/fixed_validation_fixture.json": run_root / "fixed_validation_fixture.json",
        "manifests/task2_lerobotv3_validation.json": root
        / "artifacts/development/task2_lerobotv3_validation.json",
        "manifests/normalizer_manifest.json": dataset_root / "normalizer_manifest.json",
        "manifests/conversion_manifest.json": dataset_root / "conversion_manifest.json",
        "manifests/split_manifest.json": dataset_root / "split_manifest.json",
        "manifests/converter_runtime_spec.task2.development.json": root
        / "configs/converter_runtime_spec.task2.development.json",
        "manifests/task2_development_data_scope.json": root
        / "configs/task2_development_data_scope.json",
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
        "manifests/p7_training_recipe.development.yaml": root
        / "configs/p7_training_recipe.development.yaml",
        "manifests/offline_sft_training_recipe.development.yaml": root
        / "configs/offline_sft_training_recipe.development.yaml",
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
) -> Path:
    from forcesmolvla.checkpoint import (
        save_p8_training_state,
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
        config_path = temporary / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["vlm_model_name"] = "base_assets/smolvlm_constructor"
        config["load_vlm_weights"] = False
        config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
        shutil.copytree(
            root / "assets/smolvlm_constructor",
            temporary / "base_assets/smolvlm_constructor",
        )
        _copy_checkpoint_payloads(root, dataset_root, run_root, temporary)
        trainability = write_trainability_manifest(policy, temporary)
        if (
            trainability["total_parameters"] != EXPECTED_PARAMETERS
            or trainability["frozen_parameters"] != 0
        ):
            raise RuntimeError("FULL_FINETUNE_TRAINABILITY_DRIFT")
        save_p8_training_state(
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
                "dataset": "task2_lerobotv3",
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
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=root / "datasets/task2_lerobotv3"
    )
    parser.add_argument("--repo-id", default="local/task2_lerobotv3")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "outputs/development/task2_lerobotv3_hybrid_full_finetune",
    )
    parser.add_argument("--target-samples", type=int, default=DEFAULT_TARGET_SAMPLES)
    parser.add_argument(
        "--checkpoint-interval-samples",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL_SAMPLES,
    )
    parser.add_argument(
        "--validation-interval-samples",
        type=int,
        default=DEFAULT_VALIDATION_INTERVAL_SAMPLES,
    )
    parser.add_argument(
        "--log-interval-samples", type=int, default=DEFAULT_LOG_INTERVAL_SAMPLES
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--p8-gate-report",
        type=Path,
        default=root / "artifacts/development/p8_v4_2_r7_gpu_preflight.json",
    )
    parser.add_argument(
        "--p8-source-binding",
        type=Path,
        default=root / "artifacts/development/p8_v4_2_r7_source_binding.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("PYTHONHASHSEED=42 required before interpreter startup")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required")
    if (
        args.target_samples <= 0
        or args.checkpoint_interval_samples <= 0
        or args.validation_interval_samples <= 0
        or args.log_interval_samples <= 0
    ):
        raise ValueError("training sample budget and sample intervals must be positive")
    _require_offline()

    import numpy as np
    import torch

    from forcesmolvla.checkpoint import load_offline_base_policy, load_p8_training_state
    from forcesmolvla.configuration_forcesmolvla import FORCE_TOKEN_MOE, OFFLINE_FULL_FINETUNE
    from forcesmolvla.dataset_v3 import load_dataset_split
    from forcesmolvla.router_training import (
        MoEMicrobatch,
        SerializableUniformSampler,
        build_p7_optimizer_and_scheduler,
        derive_optimizer_updates,
        single_pass_optimizer_update,
    )
    from forcesmolvla.training_data import load_runtime_artifacts, prepare_training_sample
    from preflight_p5_dense_compute_gpu import (
        _validate_action_target_population_prerequisite,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")
    gpu_name = torch.cuda.get_device_name(0)
    if not any(value in gpu_name for value in EXPECTED_GPU_NAMES):
        raise RuntimeError(f"TRAINING_REQUIRES_RTX_4090D: got {gpu_name!r}")
    root = Path(__file__).parents[1].resolve()
    dataset_root = args.dataset_root.resolve()
    run_root = args.run_root.resolve()
    conversion = json.loads((dataset_root / "conversion_manifest.json").read_text())
    _validate_action_target_population_prerequisite(root, dataset_root)
    data_scope, data_scope_sha256 = _load_task2_data_scope(
        root, dataset_root, args
    )
    if (
        conversion.get("repo_id") != args.repo_id
        or conversion.get("artifact_status") != "development_only"
        or conversion.get("formal_ready") is not False
        or len(conversion.get("episodes", ())) < 3
    ):
        raise RuntimeError("TASK2_CONVERSION_MANIFEST_GATE_FAILED")
    p8_gate_path = args.p8_gate_report.resolve()
    p8_gate = json.loads(p8_gate_path.read_text(encoding="utf-8"))
    p8_binding_path = args.p8_source_binding.resolve()
    checkpoint_path = Path(p8_gate.get("checkpoint", {}).get("path", ""))
    if (
        p8_gate.get("gate") != "P8"
        or p8_gate.get("gate_status") != "pass"
        or p8_gate.get("acceptance_status") != "development_only"
        or p8_gate.get("formal_eligible") is not False
        or p8_gate.get("gate_contract_version")
        != "v4.2-b4x1-single-pass-exact-resume"
        or p8_gate.get("exact_resume_dry_run") is not True
        or p8_gate.get("long_development_sft_unlocked") is not True
        or set(p8_gate.get("force_full_parity", {})) != {"fp32", "bf16"}
        or p8_gate.get("real_data", {}).get("repo_id") != args.repo_id
        or p8_gate.get("real_data", {}).get("batch_per_gpu") != BATCH_SIZE
        or p8_gate.get("real_data", {}).get("microbatches") != MICROBATCHES
        or p8_gate.get("source_binding_sha256") != _sha256(p8_binding_path)
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
        args.repo_id,
        p8_gate_report=p8_gate_path,
        p8_source_binding=p8_binding_path,
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
        repo_id=args.repo_id,
        split_name="train",
        artifact_use="development",
        delta_timestamps=delta_timestamps,
    )
    val_dataset = load_dataset_split(
        dataset_root,
        repo_id=args.repo_id,
        split_name="val",
        artifact_use="development",
        delta_timestamps=delta_timestamps,
    )
    if len(train_dataset) < BATCH_SIZE * MICROBATCHES or len(val_dataset) < 2:
        raise RuntimeError("TASK2_DATASET_TOO_SMALL")
    runtime_artifacts = load_runtime_artifacts(
        dataset_root,
        calibration_bundle_path=root / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=root / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=root / "artifacts/development/action_delta_spec.json",
        expected_repo_id=args.repo_id,
    )
    normalizer = runtime_artifacts.normalizer
    sampler = SerializableUniformSampler(list(range(len(train_dataset))), seed=42)
    effective_samples_per_update = BATCH_SIZE * MICROBATCHES
    max_updates = derive_optimizer_updates(
        args.target_samples, effective_samples_per_update
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
        args.checkpoint_interval_samples, effective_samples_per_update
    )
    validation_interval_updates = derive_optimizer_updates(
        args.validation_interval_samples, effective_samples_per_update
    )
    log_interval_updates = derive_optimizer_updates(
        args.log_interval_samples, effective_samples_per_update
    )
    budget = {
        "primary_unit": "samples",
        "target_samples": args.target_samples,
        "resolved_train_split_samples": len(train_dataset),
        "target_equivalent_epochs": args.target_samples / len(train_dataset),
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
    }
    offline_recipe = json.loads(
        (root / "configs/offline_sft_training_recipe.development.yaml").read_text()
    )
    if (
        offline_recipe.get("training_stage") != "offline_full_finetune"
        or offline_recipe.get("all_existing_parameters_require_grad") is not True
        or offline_recipe["schedule"]["primary_budget_unit"] != "samples"
        or offline_recipe["schedule"]["target_samples"] != args.target_samples
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
        != args.checkpoint_interval_samples
        or offline_recipe["validation_interval_samples"]
        != args.validation_interval_samples
        or offline_recipe["loss"]["router_algorithm"] != "single_pass_batch_local"
        or offline_recipe["optimizer"].get("parameter_partition")
        != "each_trainable_parameter_exactly_once"
        or "learned_action_slot" not in offline_recipe["optimizer"].get("no_decay", ())
        or offline_recipe["p7_exact_two_pass"]["active_sft_loop"] is not False
        or offline_recipe["p7_exact_two_pass"]["long_running_sft_allowed"] is not False
    ):
        raise RuntimeError("OFFLINE_SINGLE_PASS_SAMPLE_BUDGET_CONTRACT_DRIFT")
    resolved = _resolved_config(
        args, _sha256(binding_path), budget, data_scope, data_scope_sha256
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
    optimizer, scheduler, optimizer_groups = build_p7_optimizer_and_scheduler(
        policy, derived_optimizer_updates=max_updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_step = 0
    resume_contract = {}
    if args.resume is not None:
        start_step, resume_contract = load_p8_training_state(
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
    _bind_task2_fixture_provenance(fixture, data_scope)
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
                "target_samples": args.target_samples,
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
    data_pool = ThreadPoolExecutor(
        max_workers=DATA_LOADING_THREADS, thread_name_prefix="task2_decode"
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
                    identity=f"task2-step-{step}-microbatch-{microbatch_index}",
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
        "training_samples_seen": args.target_samples,
        "equivalent_epochs_seen": args.target_samples / len(train_dataset),
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
