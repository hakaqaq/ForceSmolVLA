#!/usr/bin/env python3
"""CUDA-only P9 task2 record/replay gate; no ROS, queues, or action transport."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time

import numpy as np

from forcesmolvla.training_runtime import (
    file_sha256 as _sha256,
    require_offline_environment as _require_offline,
    validate_source_binding as _validate_source_binding,
)
from forcesmolvla.dataset_binding import (
    dataset_storage_binding as _dataset_storage_binding,
    validate_runtime_import_roots as _validate_runtime_import_roots,
)


def _jsonable(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _context_payload(context) -> dict:
    return {field.name: _jsonable(getattr(context, field.name)) for field in fields(context)}


def _tensor_sha256(tensor) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _map(clock_map: dict, name: str, timestamp_ns: int) -> int:
    mapping = clock_map["mappings"][name]
    return (
        timestamp_ns * int(mapping["slope_numerator"])
        // int(mapping["slope_denominator"])
        + int(mapping["offset_ns"])
    )


def _inverse_map_exact(clock_map: dict, name: str, controller_ns: int) -> int:
    mapping = clock_map["mappings"][name]
    numerator = (controller_ns - int(mapping["offset_ns"])) * int(
        mapping["slope_denominator"]
    )
    denominator = int(mapping["slope_numerator"])
    if denominator <= 0 or numerator % denominator:
        raise RuntimeError(f"P9_TEST_CLOCK_MAP_NOT_EXACTLY_INVERTIBLE:{name}")
    source_ns = numerator // denominator
    if _map(clock_map, name, source_ns) != controller_ns:
        raise RuntimeError(f"P9_TEST_CLOCK_MAP_INVERSE_MISMATCH:{name}")
    return source_ns


def _episode_clock_diagnostics(conversion: dict, output_episode_index: int) -> dict:
    matches = [
        episode["diagnostics"]
        for episode in conversion.get("episodes", [])
        if int(episode.get("output_episode_index", -1)) == int(output_episode_index)
    ]
    if len(matches) != 1:
        raise RuntimeError("P9_EPISODE_CLOCK_MAP_NOT_UNIQUE")
    diagnostics = matches[0]
    clock_hash = diagnostics.get("clock_map_sha256")
    if (
        not isinstance(clock_hash, str)
        or len(clock_hash) != 64
        or diagnostics.get("clock_map_id") != f"sha256:{clock_hash}"
        or not isinstance(diagnostics.get("clock_offset_ns"), int)
    ):
        raise RuntimeError("P9_EPISODE_CLOCK_MAP_INVALID")
    return diagnostics


def _source_stamp_to_host_monotonic(source_stamp_ns: int, diagnostics: dict) -> int:
    return int(source_stamp_ns) + int(diagnostics["clock_offset_ns"])


def _validate_contract(config: dict) -> None:
    if (
        config.get("acceptance_status") != "development_only"
        or config.get("formal_eligible") is not False
        or config.get("gate") != "P9"
        or config.get("shadow_scope") != "pure_offline_record_replay"
        or config.get("input_profile_revision")
        != "v4.2-p9-task2-user-confirmed-2026-08-21"
        or config.get("allowed_inputs")
        != ["datasets/task2_lerobotv3", "golden_fixtures", "tests/fixtures"]
        or config.get("dataset") != "datasets/task2_lerobotv3"
        or config.get("checkpoint")
        != "outputs/development/p8_v4_2_r4_checkpoint_seed42_step000001"
        or set(config.get("test_only_expected_outcome", {}))
        != {
            "synthetic_gpu_ready_offset_ns",
            "candidate_valid",
            "candidate_reasons",
            "actual_dispatched_indices",
        }
    ):
        raise RuntimeError("P9_CONTRACT_STATUS_DRIFT")
    scope_amendment = config.get("scope_amendment")
    if not isinstance(scope_amendment, dict) or set(scope_amendment) != {
        "path",
        "sha256",
    }:
        raise RuntimeError("P9_SCOPE_AMENDMENT_BINDING_MISSING_OR_DRIFTED")
    required_forbidden = {
        "ROS",
        "live_robot_interfaces",
        "robot_action_send",
        "Franky_queue",
        "RTC_queue",
        "native_select_action",
    }
    if set(config.get("forbidden", [])) != required_forbidden:
        raise RuntimeError("P9_FORBIDDEN_BOUNDARY_DRIFT")
    prerequisite = config.get("p8_prerequisite")
    if not isinstance(prerequisite, dict) or set(prerequisite) != {
        "source_binding",
        "resolved_config",
        "cold_start",
        "gate_result",
        "checkpoint_manifest",
        "required_gate",
        "required_gate_status",
        "required_acceptance_status",
        "required_formal_eligible",
        "required_exact_resume",
        "required_p9_started",
        "required_robot_actions_sent",
    }:
        raise RuntimeError("P9_P8_PREREQUISITE_MISSING_OR_DRIFTED")
    for name in (
        "source_binding",
        "resolved_config",
        "cold_start",
        "gate_result",
        "checkpoint_manifest",
    ):
        if set(prerequisite[name]) != {"path", "sha256"}:
            raise RuntimeError(f"P9_P8_{name.upper()}_BINDING_DRIFT")
    if (
        prerequisite["required_gate"] != "P8"
        or prerequisite["required_gate_status"] != "pass"
        or prerequisite["required_acceptance_status"] != "development_only"
        or prerequisite["required_formal_eligible"] is not False
        or prerequisite["required_exact_resume"] is not True
        or prerequisite["required_p9_started"] is not False
        or prerequisite["required_robot_actions_sent"] != 0
    ):
        raise RuntimeError("P9_P8_PREREQUISITE_SEMANTICS_DRIFT")


def _load_scope_amendment(root: Path, config: dict) -> tuple[dict, dict]:
    binding = config["scope_amendment"]
    amendment_path = root / binding["path"]
    if _sha256(amendment_path) != binding["sha256"]:
        raise RuntimeError("P9_SCOPE_AMENDMENT_HASH_MISMATCH")
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (
        amendment.get("acceptance_status") != "development_only"
        or amendment.get("formal_eligible") is not False
        or amendment.get("scope") != "P9_only"
        or amendment.get("supersedes_visible_p9_dataset") != "task1_v4_1"
        or amendment.get("dataset") != config["dataset"]
        or amendment.get("checkpoint") != config["checkpoint"]
        or amendment.get("allowed_inputs") != config["allowed_inputs"]
        or amendment.get("p4_p8_artifacts_unchanged") is not True
        or amendment.get("p4_p8_rerun_required") is not False
        or amendment.get("production_shadow") is not False
    ):
        raise RuntimeError("P9_SCOPE_AMENDMENT_SEMANTICS_DRIFT")

    data_binding = amendment.get("task2_data_scope")
    if not isinstance(data_binding, dict) or set(data_binding) != {"path", "sha256"}:
        raise RuntimeError("P9_TASK2_DATA_SCOPE_BINDING_MISSING_OR_DRIFTED")
    data_scope_path = root / data_binding["path"]
    if _sha256(data_scope_path) != data_binding["sha256"]:
        raise RuntimeError("P9_TASK2_DATA_SCOPE_HASH_MISMATCH")
    data_scope = json.loads(data_scope_path.read_text(encoding="utf-8"))
    dataset_scope = data_scope.get("dataset", {})
    session_scope = data_scope.get("session_provenance", {})
    budget = data_scope.get("training_budget", {})
    if (
        data_scope.get("acceptance_status") != "development_only"
        or data_scope.get("formal_eligible") is not False
        or dataset_scope.get("path") != config["dataset"]
        or dataset_scope.get("repo_id") != amendment.get("repo_id")
        or session_scope.get("explicit_physical_session_id") is not None
        or session_scope.get("physical_session_id_status")
        != "not_recorded_in_source"
        or session_scope.get("legacy_fixture_session_id") != "task1_within_session"
        or session_scope.get("legacy_fixture_session_id_status")
        != "invalid_legacy_metadata_for_task2"
        or session_scope.get("replacement_chunk_context_session_id")
        != amendment.get("session_context", {}).get("replacement_session_id")
        or budget.get("primary_unit") != "samples"
        or budget.get("target_samples") != 80_000
        or budget.get("batch_per_gpu") != 4
        or budget.get("gradient_accumulation_microbatches") != 1
        or budget.get("effective_samples_per_update") != 4
        or budget.get("derived_optimizer_updates") != 20_000
        or budget.get("checkpoint_policy") != "final_update_only"
        or budget.get("final_checkpoint_training_samples") != 80_000
        or budget.get("legacy_recipe_checkpoint_interval_samples") != 2_000
        or budget.get("validation_interval_samples") != 2_000
    ):
        raise RuntimeError("P9_TASK2_DATA_SCOPE_SEMANTICS_DRIFT")

    raw_session_path = Path(session_scope["raw_session_manifest_path"])
    if _sha256(raw_session_path) != session_scope["raw_session_manifest_sha256"]:
        raise RuntimeError("P9_TASK2_RAW_SESSION_MANIFEST_HASH_MISMATCH")
    raw_session = json.loads(raw_session_path.read_text(encoding="utf-8"))
    if (
        "session_id" in raw_session
        or raw_session.get("raw_format_version") != session_scope["raw_format_version"]
        or raw_session.get("created_at") != session_scope["created_at"]
    ):
        raise RuntimeError("P9_TASK2_RAW_SESSION_PROVENANCE_DRIFT")

    recipe_path = root / budget["recipe_path"]
    if _sha256(recipe_path) != budget["recipe_sha256"]:
        raise RuntimeError("P9_TASK2_TRAINING_RECIPE_HASH_MISMATCH")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if (
        recipe.get("training_stage") != "offline_full_finetune"
        or recipe.get("schedule", {}).get("target_samples") != 80_000
        or recipe.get("schedule", {}).get("derived_optimizer_updates") != 20_000
        or recipe.get("batching", {}).get("batch_per_gpu") != 4
        or recipe.get("batching", {}).get("gradient_accumulation_microbatches") != 1
        or recipe.get("checkpoint_interval_samples") != 2_000
        or recipe.get("validation_interval_samples") != 2_000
    ):
        raise RuntimeError("P9_TASK2_TRAINING_BUDGET_DRIFT")

    scheduler = amendment.get("scheduler_index_semantics", {})
    if (
        scheduler.get("exact_integer_formula")
        != "j=ceil((t_candidate_controller_ns-tau0_controller_ns)*action_period_denominator/action_period_numerator_ns)"
        or scheduler.get("t_candidate_definition")
        != "t_ready_controller_ns+transport_ns"
        or scheduler.get("action_period_numerator_ns") != 100_000_000
        or scheduler.get("action_period_denominator") != 3
        or scheduler.get("observed_candidate_minus_tau0_ns") != 205_000_000
        or scheduler.get("observed_unrounded_index") != "6.15"
        or scheduler.get("expected_j") != 7
        or scheduler.get("t_apply_based_formula") is not False
    ):
        raise RuntimeError("P9_SCHEDULER_INDEX_SEMANTICS_DRIFT")
    return amendment, data_scope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-output", type=Path, required=True)
    parser.add_argument(
        "--source-binding",
        type=Path,
        default=Path(__file__).parents[1]
        / "artifacts/development/p9_v4_2_r5_source_binding.json",
    )
    args = parser.parse_args()
    for path in (args.records_output, args.replay_output, args.output, args.resolved_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite P9 artifact: {path}")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("PYTHONHASHSEED=42 required before interpreter startup")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required")
    _require_offline()

    import torch

    from forcesmolvla.action_delta import ActionDeltaProcessor
    from forcesmolvla.checkpoint import (
        sha256_file,
        validate_force_artifact_manifest,
        validate_p8_payload_contract,
    )
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.shadow import (
        ShadowProtocol,
        build_shadow_record_artifact,
        evaluate_shadow_candidate,
        replay_shadow_record_artifact,
        resolve_shadow_artifacts,
    )
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
    from p8_checkpoint_common import chunk_context_from_fixture, load_fixed_validation_inputs

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")
    gpu_name = torch.cuda.get_device_name(0)
    if "4090 D" not in gpu_name and "4090D" not in gpu_name:
        raise RuntimeError(f"P9_REQUIRES_RTX_4090D: {gpu_name}")
    root = Path(__file__).parents[1].resolve()
    config_path = root / "configs/p9_shadow_replay.development.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_contract(config)
    scope_amendment, data_scope = _load_scope_amendment(root, config)
    dataset_root = args.dataset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    expected_dataset = (root / config["dataset"]).resolve()
    expected_checkpoint = (root / config["checkpoint"]).resolve()
    if dataset_root != expected_dataset or checkpoint != expected_checkpoint:
        raise RuntimeError("P9_INPUT_PATH_OUTSIDE_FROZEN_SCOPE")
    protocol = ShadowProtocol.from_dict(config["protocol"])
    binding_path = args.source_binding.resolve()
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if (
        binding.get("status") != "development_only"
        or binding.get("acceptance_status") != "development_only"
        or binding.get("formal_eligible") is not False
        or binding.get("stage") != "P9"
    ):
        raise RuntimeError("P9_SOURCE_BINDING_STATUS_DRIFT")
    conversion = json.loads(
        (dataset_root / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    repo_id = conversion.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("P9_DATASET_REPO_ID_MISSING")
    dataset_scope = data_scope["dataset"]
    if (
        repo_id != dataset_scope["repo_id"]
        or _sha256(dataset_root / "conversion_manifest.json")
        != dataset_scope["conversion_manifest_sha256"]
        or conversion.get("raw_source_tree_sha256")
        != dataset_scope["raw_source_tree_sha256"]
        or conversion.get("evaluation_scope")
        != "within-session offline fine-tuning; not cross-session generalization"
    ):
        raise RuntimeError("P9_TASK2_DATASET_SCOPE_DRIFT")
    _validate_source_binding(
        root, binding, dataset_root=dataset_root, repo_id=repo_id
    )
    if _validate_runtime_import_roots(root) != binding.get("runtime_imports"):
        raise RuntimeError("P9_RUNTIME_IMPORT_ROOT_MISMATCH")
    if _dataset_storage_binding(dataset_root) != binding.get("dataset", {}).get(
        "storage_tree"
    ):
        raise RuntimeError("P9_DATASET_STORAGE_TREE_MISMATCH")
    _pytest_evidence_summary(root, binding["test_evidence"])
    for relative, expected in binding.get("bound_inputs", {}).items():
        if _sha256(root / relative) != expected:
            raise RuntimeError(f"P9_BOUND_INPUT_HASH_MISMATCH: {relative}")
    prerequisite = config["p8_prerequisite"]
    if binding.get("p8_prerequisite") != prerequisite:
        raise RuntimeError("P9_SOURCE_BINDING_P8_PREREQUISITE_MISMATCH")
    if (
        binding.get("scope_amendment") != config["scope_amendment"]
        or binding.get("task2_data_scope") != scope_amendment["task2_data_scope"]
        or binding.get("session_provenance") != data_scope["session_provenance"]
        or binding.get("training_budget") != data_scope["training_budget"]
    ):
        raise RuntimeError("P9_SOURCE_BINDING_TASK2_SCOPE_MISMATCH")
    for item in (
        "source_binding",
        "resolved_config",
        "cold_start",
        "gate_result",
        "checkpoint_manifest",
    ):
        artifact = prerequisite[item]
        if _sha256(root / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"P9_P8_{item.upper()}_HASH_MISMATCH")

    p8_gate = json.loads(
        (root / prerequisite["gate_result"]["path"]).read_text(encoding="utf-8")
    )
    p8_cold = json.loads(
        (root / prerequisite["cold_start"]["path"]).read_text(encoding="utf-8")
    )
    if (
        p8_gate.get("gate") != prerequisite["required_gate"]
        or p8_gate.get("gate_status") != prerequisite["required_gate_status"]
        or p8_gate.get("acceptance_status")
        != prerequisite["required_acceptance_status"]
        or p8_gate.get("formal_eligible")
        is not prerequisite["required_formal_eligible"]
        or p8_gate.get("exact_resume_dry_run")
        is not prerequisite["required_exact_resume"]
        or p8_gate.get("p9_started") is not prerequisite["required_p9_started"]
        or p8_gate.get("robot_actions_sent")
        != prerequisite["required_robot_actions_sent"]
        or p8_gate.get("source_binding_sha256")
        != prerequisite["source_binding"]["sha256"]
        or p8_gate.get("resolved_config_sha256")
        != prerequisite["resolved_config"]["sha256"]
        or p8_gate.get("checkpoint", {}).get("path") != str(checkpoint)
        or p8_gate.get("checkpoint", {}).get("artifact_manifest_sha256")
        != prerequisite["checkpoint_manifest"]["sha256"]
        or p8_cold.get("gate_status") != "pass"
        or p8_cold.get("exact_resume_dry_run") is not True
        or p8_cold.get("parity_exact") is not True
        or p8_cold.get("rng_continuation_exact") is not True
        or p8_cold.get("sampler_continuation_exact") is not True
        or p8_cold.get("formal_eligible") is not False
        or p8_cold.get("p9_started") is not False
        or p8_cold.get("robot_actions_sent") != 0
    ):
        raise RuntimeError("P8_GATE_NOT_PASS_P9_FORBIDDEN")
    checkpoint_manifest = validate_force_artifact_manifest(checkpoint, artifact_use="development")
    validate_p8_payload_contract(checkpoint)

    rules_path = (root / config["rulespec_test_only"]).resolve()
    clock_path = (root / config["clock_map_test_only"]).resolve()
    schema_path = root / "schemas/rulespec.schema.json"
    resolution = resolve_shadow_artifacts(
        mode="test_only",
        rules_path=rules_path,
        schema_path=schema_path,
        clock_map_path=clock_path,
        test_fixture_root=root / "tests/fixtures",
    )
    if not resolution.valid:
        raise RuntimeError(f"P9_TEST_ONLY_RESOLUTION_FAILED: {resolution.reasons}")
    production_missing = resolve_shadow_artifacts(
        mode="production",
        rules_path=rules_path,
        schema_path=schema_path,
        clock_map_path=None,
        test_fixture_root=root / "tests/fixtures",
    )
    production_test_assets = resolve_shadow_artifacts(
        mode="production",
        rules_path=rules_path,
        schema_path=schema_path,
        clock_map_path=clock_path,
        test_fixture_root=root / "tests/fixtures",
    )
    if production_missing.valid or production_test_assets.valid:
        raise RuntimeError("P9_PRODUCTION_FAIL_CLOSED_PROBE_FAILED")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    with contextlib.redirect_stdout(sys.stderr):
        policy = ForceSmolVLAPolicy.from_pretrained(
            checkpoint,
            local_files_only=True,
            force_download=False,
            strict=True,
            artifact_use="development",
        )
    if policy.config.rtc_config is not None or getattr(policy, "rtc_processor", None) is not None:
        raise RuntimeError("P9_RTC_MUST_BE_ABSENT")
    fixture = json.loads(
        (checkpoint / "manifests/p7_validation_fixture.json").read_text(encoding="utf-8")
    )
    batch, raw_samples, runtime_artifacts = load_fixed_validation_inputs(
        policy, dataset_root, fixture, device
    )
    base_context = chunk_context_from_fixture(
        fixture, policy_generation=policy._context_generation
    )
    clock_map = resolution.clock_map
    t_ref_sensor = [int(sample["provenance.tuple_host_monotonic_ns"]) for sample in raw_samples]
    t_ref_controller = [
        _map(clock_map, "sensor_to_controller", timestamp) for timestamp in t_ref_sensor
    ]
    context = replace(
        base_context,
        t_ref_ns=torch.tensor(t_ref_controller, dtype=torch.int64),
        tau0_ns=torch.tensor(t_ref_controller, dtype=torch.int64),
        clock_domain_id=(clock_map["controller_clock_domain"],) * 2,
        session_id=(
            data_scope["session_provenance"]["replacement_chunk_context_session_id"],
        )
        * 2,
        chunk_id=("p9-task2-record-0", "p9-task2-record-1"),
        selected_provenance=tuple(
            {
                "episode_index": int(sample["episode_index"]),
                "frame_index": int(sample["frame_index"]),
                "tuple_host_monotonic_ns": int(sample["provenance.tuple_host_monotonic_ns"]),
                "mode": "test_only_clock_algorithmic_development_replay",
                "collection_scope_id": data_scope["session_provenance"][
                    "collection_scope_id"
                ],
                "physical_session_id": None,
                "session_id_semantics": data_scope["session_provenance"][
                    "collection_scope_id_semantics"
                ],
                "replaced_legacy_fixture_session_id": base_context.session_id[index],
            }
            for index, sample in enumerate(raw_samples)
        ),
    )
    noise7 = torch.tensor(fixture["epsilon7"]["tensor"], dtype=torch.float32, device=device)
    policy.eval()
    policy.bind_runtime_artifacts(runtime_artifacts)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        normalized_delta7, absolute_tensor = policy._predict_action_chunks(
            batch, chunk_context=context, noise=noise7
        )
    end.record()
    torch.cuda.synchronize()
    inference_wall_seconds = time.perf_counter() - wall_start
    inference_cuda_ms = float(start.elapsed_time(end))
    peak_memory = {
        "allocated_bytes": torch.cuda.max_memory_allocated(device),
        "reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    normalized_numpy = normalized_delta7.detach().cpu().to(torch.float32).numpy().astype(np.float64)
    raw_state = batch["raw_state_snapshot"].detach().cpu().numpy().astype(np.float64)
    absolute7 = absolute_tensor.detach().cpu().numpy().astype(np.float64)
    unnormalized_delta7 = ActionDeltaProcessor.to_delta(absolute7, raw_state)

    resolved = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "gate": "P9",
        "shadow_scope": "pure_offline_record_replay",
        "shadow_status": "algorithmic_development_replay",
        "production_shadow": False,
        "source_binding_sha256": _sha256(binding_path),
        "p8_source_binding_sha256": prerequisite["source_binding"]["sha256"],
        "p8_resolved_config_sha256": prerequisite["resolved_config"]["sha256"],
        "p8_checkpoint_artifact_manifest_sha256": sha256_file(checkpoint / "artifact_manifest.json"),
        "p8_checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "p9_contract_sha256": _sha256(config_path),
        "p9_scope_amendment_sha256": config["scope_amendment"]["sha256"],
        "task2_data_scope_sha256": scope_amendment["task2_data_scope"]["sha256"],
        "task2_session_provenance": data_scope["session_provenance"],
        "training_budget": data_scope["training_budget"],
        "scheduler_index_semantics": scope_amendment["scheduler_index_semantics"],
        "test_only_rulespec_sha256": resolution.rules_sha256,
        "test_only_clock_map_sha256": resolution.clock_map_sha256,
        "test_only_threshold_values_embedded_in_resolved_config": False,
        "dataset_conversion_manifest_sha256": _sha256(dataset_root / "conversion_manifest.json"),
        "dataset_split_manifest_sha256": _sha256(dataset_root / "split_manifest.json"),
        "dataset_normalizer_manifest_sha256": _sha256(dataset_root / "normalizer_manifest.json"),
        "protocol": config["protocol"],
        "model_force_initialization_tensor_sha256": policy.force_initialization_tensor_hash(),
        "strict_checkpoint_reload": True,
        "local_files_only": True,
        "rtc_config": None,
        "native_queue_used": False,
        "ros_connected": False,
        "robot_actions_sent": 0,
        "formal_signature_algorithm": None,
        "formal_key_id": None,
        "formal_approver": None,
        "detached_signature": None,
        "approval": None,
    }
    resolved_text = json.dumps(resolved, indent=2, sort_keys=True) + "\n"
    resolved_sha = hashlib.sha256(resolved_text.encode()).hexdigest()

    sample = raw_samples[0]
    episode_clock = _episode_clock_diagnostics(
        conversion, int(sample["episode_index"])
    )
    t_ref = t_ref_sensor[0]
    t_ref_controller_0 = t_ref_controller[0]
    expected_outcome = config["test_only_expected_outcome"]
    t_ready_controller = t_ref_controller_0 + int(
        expected_outcome["synthetic_gpu_ready_offset_ns"]
    )
    t_ready_gpu = _inverse_map_exact(clock_map, "gpu_to_controller", t_ready_controller)
    source = {
        "generation": 0,
        "policy_tick_index": 0,
        "sensor_clock_domain": clock_map["mappings"]["sensor_to_controller"]["source_clock_domain"],
        "gpu_clock_domain": clock_map["mappings"]["gpu_to_controller"]["source_clock_domain"],
        "t_ref_sensor_ns": t_ref,
        "t_ready_gpu_ns": t_ready_gpu,
        "transport_ns": 5_000_000,
        "tau0_controller_ns": t_ref_controller_0,
        "observation_timestamps_sensor_ns": {
            "camera1": int(sample["provenance.camera1_receive_monotonic_ns"]),
            "camera2": int(sample["provenance.camera2_receive_monotonic_ns"]),
            "state": _source_stamp_to_host_monotonic(
                sample["provenance.state_pose_source_stamp_ns"], episode_clock
            ),
            "wrench": _source_stamp_to_host_monotonic(
                sample["provenance.wrench_raw_source_stamp_ns"], episode_clock
            ),
        },
        "raw_state7": raw_state[0].tolist(),
        "normalized_delta7_chunk": normalized_numpy[0].tolist(),
        "absolute_action7_chunk": absolute7[0].tolist(),
        "action_valid_mask": context.action_valid_mask[0].tolist(),
        "runtime_artifact_compatible": bool(context.runtime_artifact_compatible[0]),
        "wrench_geometry_valid": bool(context.wrench_geometry_valid[0]),
        "chunk_context": _context_payload(context),
        "calibration_id": json.loads(
            (dataset_root / "conversion_manifest.json").read_text(encoding="utf-8")
        )["calibration_id_by_index"][str(int(sample["provenance.calibration_index"]))],
        "calibration_bundle_hash": context.calibration_bundle_hash[0],
        "normalizer_hash": context.normalizer_hash[0],
        "wrench_geometry_spec_hash": context.wrench_geometry_spec_hash[0],
        "raw_and_filter_timestamps": {
            "raw_wrench_source_ns": int(sample["provenance.wrench_raw_source_stamp_ns"]),
            "filter_output_source_ns": int(sample["provenance.wrench_filter_output_stamp_ns"]),
            "pose_source_ns": int(sample["provenance.pose_source_stamp_ns"]),
            "raw_wrench_host_monotonic_ns": _source_stamp_to_host_monotonic(
                sample["provenance.wrench_raw_source_stamp_ns"], episode_clock
            ),
            "filter_output_host_monotonic_ns": _source_stamp_to_host_monotonic(
                sample["provenance.wrench_filter_output_stamp_ns"], episode_clock
            ),
            "pose_host_monotonic_ns": _source_stamp_to_host_monotonic(
                sample["provenance.pose_source_stamp_ns"], episode_clock
            ),
            "pose_age_ms": float(sample["provenance.pose_age_ms"]),
            "validity_bits": int(sample["provenance.validity_bits"]),
        },
        "conversion_clock_map": {
            "clock_map_id": episode_clock["clock_map_id"],
            "clock_map_sha256": episode_clock["clock_map_sha256"],
            "clock_offset_ns": episode_clock["clock_offset_ns"],
            "mapping": "host_monotonic_ns=source_stamp_ns+clock_offset_ns",
        },
        "camera": {
            "camera1": {
                "id": "D435-third-person",
                "timestamp_ns": int(sample["provenance.camera1_receive_monotonic_ns"]),
                "sha256": _tensor_sha256(sample["observation.images.camera1"]),
            },
            "camera2": {
                "id": "D405-wrist",
                "timestamp_ns": int(sample["provenance.camera2_receive_monotonic_ns"]),
                "sha256": _tensor_sha256(sample["observation.images.camera2"]),
            },
        },
        "prompt": {
            "text": str(sample["task"]),
            "text_sha256": _text_sha256(str(sample["task"])),
            "token_sha256": _tensor_sha256(batch[OBS_LANGUAGE_TOKENS][0]),
        },
        "noise": {
            "seed": None,
            "source": "p7_validation_fixture_explicit_tensor",
            "tensor_sha256": _tensor_sha256(noise7[0]),
        },
    }
    artifact_hashes = {
        "source_binding_sha256": _sha256(binding_path),
        "resolved_config_sha256": resolved_sha,
        "p8_checkpoint_artifact_manifest_sha256": sha256_file(checkpoint / "artifact_manifest.json"),
        "p8_checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "p8_checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "p8_parity_reference_sha256": sha256_file(checkpoint / "parity_reference.json"),
        "dataset_conversion_manifest_sha256": _sha256(dataset_root / "conversion_manifest.json"),
        "dataset_split_manifest_sha256": _sha256(dataset_root / "split_manifest.json"),
        "dataset_normalizer_manifest_sha256": _sha256(dataset_root / "normalizer_manifest.json"),
        "rulespec_sha256": resolution.rules_sha256,
        "clock_map_sha256": resolution.clock_map_sha256,
        "p9_scope_amendment_sha256": config["scope_amendment"]["sha256"],
        "task2_data_scope_sha256": scope_amendment["task2_data_scope"]["sha256"],
    }
    preview = evaluate_shadow_candidate(source, resolution=resolution, protocol=protocol)
    planned = preview["timing"]["planned_arrival_ns"]
    run_end = (planned[-1] if planned else t_ready_controller) + protocol.max_hold_extension_ns
    records = build_shadow_record_artifact(
        [source],
        resolution=resolution,
        protocol=protocol,
        run_end_controller_ns=run_end,
        artifact_hashes=artifact_hashes,
    )
    replay = replay_shadow_record_artifact(records)
    records_text = json.dumps(records, indent=2, sort_keys=True) + "\n"
    replay_text = json.dumps(replay, indent=2, sort_keys=True) + "\n"
    records_sha = hashlib.sha256(records_text.encode()).hexdigest()
    replay_sha = hashlib.sha256(replay_text.encode()).hexdigest()

    forbidden_loaded = sorted(
        name for name in sys.modules if name == "rclpy" or name.startswith("rclpy.") or name == "rospy"
    )
    if forbidden_loaded:
        raise RuntimeError(f"P9_ROS_MODULE_LOADED: {forbidden_loaded}")
    record_outcome = records["records"][0]["candidate_outcome"]
    actual_indices = records["run_outcome"]["actual_dispatched_indices"]
    scheduler_semantics = scope_amendment["scheduler_index_semantics"]
    candidate_delta_ns = (
        record_outcome["timing"]["t_candidate_controller_ns"]
        - source["tau0_controller_ns"]
    )
    gate_checks = {
        "replay_exact": replay.get("replay_exact") is True,
        "absolute_chunk_finite": bool(np.all(np.isfinite(absolute7))),
        "candidate_valid_consistent_with_reasons": record_outcome["candidate_valid"]
        is (not record_outcome["candidate_reasons"]),
        "candidate_valid_matches_expected": record_outcome["candidate_valid"]
        is bool(expected_outcome["candidate_valid"]),
        "candidate_reasons_match_expected": record_outcome["candidate_reasons"]
        == expected_outcome["candidate_reasons"],
        "actual_indices_match_expected": actual_indices
        == expected_outcome["actual_dispatched_indices"],
        "invalid_candidate_dispatched_nothing": record_outcome["candidate_valid"]
        or not actual_indices,
        "scheduler_candidate_delta_matches_amendment": candidate_delta_ns
        == scheduler_semantics["observed_candidate_minus_tau0_ns"],
        "scheduler_index_matches_candidate_formula": record_outcome["timing"]["j"]
        == scheduler_semantics["expected_j"]
        == protocol.chunk_index(
            record_outcome["timing"]["t_candidate_controller_ns"],
            source["tau0_controller_ns"],
        ),
        "task2_collection_scope_replaced_legacy_session_id": all(
            value == data_scope["session_provenance"]["collection_scope_id"]
            for value in source["chunk_context"]["session_id"]
        )
        and "task1_within_session" not in source["chunk_context"]["session_id"],
        "physical_session_id_not_invented": all(
            item["physical_session_id"] is None
            for item in source["chunk_context"]["selected_provenance"]
        ),
    }
    if not all(gate_checks.values()):
        raise RuntimeError(f"P9_ACCEPTANCE_ASSERTION_FAILED:{gate_checks}")
    result = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "gate": "P9",
        "gate_status": "pass",
        "shadow_status": "algorithmic_development_replay",
        "production_shadow": False,
        "gpu": {
            "name": gpu_name,
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "inference": {
            "batch_size": 2,
            "recorded_candidates": 1,
            "camera_count": 2,
            "horizon": 50,
            "execution_horizon": 3,
            "cuda_ms": inference_cuda_ms,
            "wall_seconds": inference_wall_seconds,
            "peak_memory": peak_memory,
        },
        "offline_record": {
            "dataset": str(dataset_root),
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "candidate_valid": record_outcome["candidate_valid"],
            "candidate_reasons": record_outcome["candidate_reasons"],
            "actual_dispatched_indices": actual_indices,
            "record_artifact_sha256": records["artifact_sha256"],
            "replay_exact": replay["replay_exact"],
            "candidate_minus_tau0_ns": candidate_delta_ns,
            "scheduler_index_j": record_outcome["timing"]["j"],
        },
        "scope_amendment": {
            "path": config["scope_amendment"]["path"],
            "sha256": config["scope_amendment"]["sha256"],
            "task2_data_scope_sha256": scope_amendment["task2_data_scope"]["sha256"],
            "collection_scope_id": data_scope["session_provenance"]["collection_scope_id"],
            "physical_session_id": None,
            "training_target_samples": data_scope["training_budget"]["target_samples"],
            "derived_optimizer_updates": data_scope["training_budget"][
                "derived_optimizer_updates"
            ],
            "scheduler_index_semantics": scheduler_semantics,
        },
        "acceptance_assertions": gate_checks,
        "production_fail_closed": {
            "missing_clock_map_candidate_valid": False,
            "missing_clock_map_reasons": list(production_missing.reasons),
            "test_only_assets_accepted": False,
            "test_only_asset_reasons": list(production_test_assets.reasons),
            "stale_or_mismatched_clock_behavior": "covered by tests; candidate_valid=false",
        },
        "source_binding_sha256": _sha256(binding_path),
        "resolved_config_sha256": resolved_sha,
        "records_file_sha256": records_sha,
        "replay_file_sha256": replay_sha,
        "p8_checkpoint_payload_count": len(checkpoint_manifest["payloads"]),
        "cpu_fallback_used": False,
        "rtc_configured": False,
        "native_queue_used": False,
        "ros_connected": False,
        "robot_actions_sent": 0,
        "formal_blockers": [
            "production sensor->controller and GPU->controller clock map absent",
            "shadow safety thresholds remain unapproved/null outside tests/fixtures",
            "trusted detached signature algorithm/key/approver unresolved",
            "task2 result is within-session algorithmic development replay only",
            "task2 source does not record an explicit physical session id",
        ],
        "detached_signature": None,
        "approval": None,
    }
    for path in (
        args.resolved_output,
        args.records_output,
        args.replay_output,
        args.output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.resolved_output.write_text(resolved_text, encoding="utf-8")
    args.records_output.write_text(records_text, encoding="utf-8")
    args.replay_output.write_text(replay_text, encoding="utf-8")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
