#!/usr/bin/env python3
"""Run the isolated Stage-3 G4P GPU numerical preflight.

This tool loads only the approved-hybrid parent, reuses the accepted Stage-2
dataset/model pipeline and Stage-3 loss APIs, and discards every optimizer/model
instance before exit.  It has no publication, network, ROS, or robot path.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping
from unittest.mock import patch

from jsonschema import Draft202012Validator
import numpy as np
import torch
from torch import Tensor, nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/stage3_gpu_preflight.v1.development.yaml"
DEFAULT_SCHEMA = ROOT / "schemas/stage3_gpu_preflight_report.v1.schema.json"

ACTOR_GROUPS = (
    "vision_smolvlm_language",
    "state_prefix_projection",
    "force_mlp",
    "fusion_moe_refiner_router",
    "force_cross_attention",
    "force_action_adapter",
    "action_expert",
    "action_io",
)


class G4PError(RuntimeError):
    """Fail-closed G4P contract violation."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise G4PError(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical_report_sha256(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("canonical_report_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def freeze_g4p_evidence(
    report: dict[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach evidence-only boundaries and per-cycle eta diagnostics."""
    require(report.get("tool_status") == "PASS", "G4P_FREEZE_REQUIRES_PASS")
    require(report.get("preflight_only") is True, "G4P_FREEZE_REQUIRES_PREFLIGHT")
    require(
        report["data"]["critic_batch"].get("R_source")
        == report["data"]["actor_batch"].get("R_source")
        == config["data"].get("online_R_source")
        == "synthetic_preflight_R_only",
        "G4P_FREEZE_R_SOURCE",
    )
    require(
        report.get("parent_checkpoint_mutated") is False
        and report.get("runtime_optimizer_state_persisted") is False
        and report.get("robot_execution_authorized") is False,
        "G4P_FREEZE_MUTATION_BOUNDARY",
    )
    require(
        report["safety"].get("CRITIC_READY") is False
        and report["safety"].get("ACTOR_Q_GUIDANCE_ENABLED") is False
        and report["safety"].get("G5_AND_LATER") == "NOT_RUN",
        "G4P_FREEZE_RUNTIME_SAFETY",
    )

    preflight_eta = float(config["loss"]["eta_actor_q"])
    require(preflight_eta > 0.0, "G4P_FREEZE_ETA")
    actor_updates = sorted(report["actor_updates"], key=lambda item: item["cycle"])
    require(
        [item["cycle"] for item in actor_updates] == [0, 1, 2, 3],
        "G4P_FREEZE_ACTOR_CYCLES",
    )
    per_cycle = []
    for update in actor_updates:
        geometry = update["gradient_geometry"]
        weighted_q = float(geometry["weighted_q_norm"])
        weighted_fm = float(geometry["weighted_fm_norm"])
        require(
            math.isfinite(weighted_q)
            and math.isfinite(weighted_fm)
            and weighted_q >= 0.0
            and weighted_fm > 0.0,
            "G4P_FREEZE_GRADIENT_DIAGNOSTIC",
        )
        ratio = weighted_q / weighted_fm
        per_cycle.append({
            "cycle": int(update["cycle"]),
            "weighted_q_norm": weighted_q,
            "weighted_fm_norm": weighted_fm,
            "weighted_q_over_weighted_fm": ratio,
            "eta_3_linear_rescale_q_over_fm": ratio * (3.0 / preflight_eta),
        })

    report["evidence_freeze"] = {
        "G4P_RESULT": "PASS",
        "R_SOURCE": "synthetic_preflight_R_only",
        "REAL_ONLINE_R_USED": False,
        "PREFLIGHT_ACTOR_STEPS_DISPOSABLE": True,
        "PRODUCTION_ACTOR_STATE_MUTATED": False,
        "RUNTIME_OPTIMIZER_STATE_PERSISTED": False,
        "CRITIC_WARMUP_STARTED": False,
        "CRITIC_READY": False,
        "ACTOR_Q_GUIDANCE_ENABLED": False,
        "ETA_3_APPROVED": False,
        "GPU_COEXISTENCE_VALIDATED": False,
        "G5_AND_LATER": "NOT_RUN",
        "ROBOT_EXECUTION_AUTHORIZED": False,
    }
    report["eta_gradient_diagnostic"] = {
        "source_fields": [
            "actor_updates[].gradient_geometry.weighted_q_norm",
            "actor_updates[].gradient_geometry.weighted_fm_norm",
        ],
        "preflight_eta": preflight_eta,
        "candidate_eta": 3.0,
        "candidate_approved": False,
        "per_cycle": per_cycle,
        "statements": [
            "eta=3 remains a provisional numerical-preflight candidate.",
            "No eta calibration or Actor Q-guidance approval is granted by G4P.",
        ],
    }
    return report


def _load_mapping(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    value = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    require(isinstance(value, dict), f"G4P_MAPPING_REQUIRED:{path}")
    return value


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_gpu_preflight_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(config)
    require(
        value.get("schema_version") == "forcesmolvla_stage3_gpu_preflight.v1.development"
        and value.get("authorization") == "isolated_gpu_numerical_preflight_only"
        and value.get("preflight_only") is True,
        "G4P_CONFIG_SCOPE",
    )
    parent = value["parent_binding"]
    require(
        parent.get("binding_type") == "new_hybrid_stage3_bootstrap"
        and parent.get("actor_source") == "cycle210_evaluation"
        and parent.get("critic_source") == parent.get("target_critic_source") == "G7A-r2"
        and parent.get("strict_load") is True
        and parent.get("random_critic_fallback") is False
        and parent.get("target_copy_fallback") is False,
        "G4P_CONFIG_PARENT",
    )
    batching = value["batching"]
    require(
        (
            batching.get("critic_batch_size"),
            batching.get("critic_R_count"),
            batching.get("critic_D_count"),
            batching.get("actor_batch_size"),
            batching.get("actor_R_count"),
            batching.get("actor_D_count"),
            batching.get("flow_inference_subbatch"),
            batching.get("flow_horizon"),
            batching.get("flow_steps"),
            batching.get("critic_slots"),
            batching.get("action_features"),
        )
        == (64, 32, 32, 24, 12, 12, 4, 50, 10, 3, 7),
        "G4P_CONFIG_BATCH_OR_TOPOLOGY",
    )
    cycles = value["cycles"]
    require(
        (
            cycles.get("warmup_joint_cycles"),
            cycles.get("measured_joint_cycles"),
            cycles.get("critic_updates_per_cycle"),
            cycles.get("actor_updates_per_cycle"),
            cycles.get("target_polyak_updates_per_cycle"),
        )
        == (1, 3, 2, 1, 2),
        "G4P_CONFIG_CYCLES",
    )
    optimizer = value["optimizer"]
    require(
        optimizer["actor"].get("lr") == 1e-5
        and optimizer["critic"].get("lr") == 3e-4
        and optimizer.get("polyak_tau") == 0.005
        and not any(
            optimizer.get(name, True)
            for name in (
                "inherit_optimizer_state",
                "inherit_scheduler",
                "inherit_rng",
                "inherit_sampler",
                "persist_runtime_state",
            )
        ),
        "G4P_CONFIG_OPTIMIZER",
    )
    loss = value["loss"]
    require(
        loss.get("online_critic") == "pure_td"
        and loss.get("calql_enabled") is False
        and loss.get("cql_enabled") is False
        and loss.get("random_candidates_enabled") is False
        and loss.get("mc_return_input_enabled") is False,
        "G4P_CONFIG_LOSS_SCOPE",
    )
    safety = value["safety"]
    require(
        not any(safety.values()),
        "G4P_CONFIG_SAFETY_MUST_REMAIN_DISABLED",
    )
    return value


def validate_report(report: Mapping[str, Any], schema_path: Path = DEFAULT_SCHEMA) -> dict[str, Any]:
    schema = _load_mapping(schema_path)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(dict(report))
    require(report.get("canonical_report_sha256") == canonical_report_sha256(report), "G4P_REPORT_DIGEST")
    return dict(report)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(value)
        stream.flush()
        temporary = Path(stream.name)
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def _current_cpu_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return 0.0


def _peak_cpu_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _cuda_memory(device: torch.device) -> dict[str, float]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_mib": _mib(torch.cuda.memory_allocated(device)),
        "reserved_mib": _mib(torch.cuda.memory_reserved(device)),
        "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated(device)),
        "device_free_mib": _mib(free),
        "device_total_mib": _mib(total),
        "cpu_rss_mib": _current_cpu_rss_mib(),
        "peak_cpu_rss_mib": _peak_cpu_rss_mib(),
    }


def _gpu_uuid(properties: Any) -> str:
    value = str(properties.uuid)
    return value if value.startswith("GPU-") else f"GPU-{value}"


def _freeze_environment(config: Mapping[str, Any]) -> tuple[torch.device, dict[str, Any]]:
    expected = config["environment"]
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == expected["expected_cuda_visible_devices"],
        "G4P_CUDA_VISIBLE_DEVICES_NOT_EXPLICITLY_APPROVED",
    )
    require(sys.executable == expected["python_executable"], "G4P_PYTHON_EXECUTABLE_DRIFT")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "G4P_SINGLE_VISIBLE_CUDA_REQUIRED")
    device = torch.device(f"cuda:{expected['visible_cuda_device_index']}")
    properties = torch.cuda.get_device_properties(device)
    require(properties.name == expected["expected_gpu_name"], f"G4P_GPU_NAME:{properties.name}")
    require(_gpu_uuid(properties) == expected["expected_gpu_uuid"], f"G4P_GPU_UUID:{_gpu_uuid(properties)}")
    random.seed(expected["seed"])
    np.random.seed(expected["seed"])
    torch.manual_seed(expected["seed"])
    torch.cuda.manual_seed_all(expected["seed"])
    torch.use_deterministic_algorithms(expected["deterministic_algorithms"])
    torch.backends.cudnn.benchmark = expected["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = expected["cudnn_deterministic"]
    free, total = torch.cuda.mem_get_info(device)
    return device, {
        "physical_cuda_device_index": expected["physical_cuda_device_index"],
        "cuda_device_index": expected["visible_cuda_device_index"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": properties.name,
        "gpu_uuid": _gpu_uuid(properties),
        "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        "gpu_total_vram_mib": _mib(total),
        "initial_free_vram_mib": _mib(free),
        "nvidia_smi_driver_version_at_authorization": expected["nvidia_smi_driver_version_at_authorization"],
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def _selected_parent_records(binding: Mapping[str, Any]) -> list[tuple[str, Path, str]]:
    records = [("actor", Path(binding["actor_parent"]["absolute_path"]), binding["actor_parent"]["sha256"])]
    for group in ("critic_parent", "target_critic_parent"):
        records.extend(
            (item["logical_role"], Path(item["absolute_path"]), item["sha256"])
            for item in binding[group]["artifacts"]
        )
    for name in (
        "normalizer_binding",
        "action_contract_binding",
        "task_feature_binding",
        "calibration_binding",
        "runtime_contract_binding",
    ):
        item = binding[name]
        records.append((name, Path(item["absolute_path"]), item["sha256"]))
    return records


def _hash_parent_records(records: Iterable[tuple[str, Path, str]]) -> dict[str, dict[str, Any]]:
    result = {}
    for role, path, expected in records:
        require(path.is_file(), f"G4P_PARENT_MISSING:{role}:{path}")
        actual = sha256_file(path)
        require(actual == expected, f"G4P_PARENT_SHA:{role}:{actual}")
        result[role] = {"path": str(path.resolve()), "sha256": actual, "size_bytes": path.stat().st_size}
    return result


def _strict_load_parents(
    binding: Mapping[str, Any], config: Mapping[str, Any], device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, dict[str, Any]]:
    from safetensors import safe_open
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.critic import build_twin_q
    from forcesmolvla.rft.stage3.parent import validate_critic_state_against_expected

    actor_path = Path(binding["actor_parent"]["absolute_path"])
    actor_container = Path(binding["actor_parent"]["architecture_binding"]["container_path"])
    with safe_open(actor_path, framework="pt", device="cpu") as tensors:
        actor_header = {
            name: {"shape": list(tensors.get_slice(name).get_shape()), "dtype": str(tensors.get_slice(name).get_dtype())}
            for name in tensors.keys()
        }
    started = time.perf_counter()
    with redirect_stdout(sys.stderr):
        actor = ForceSmolVLAPolicy.from_pretrained(
            actor_container,
            local_files_only=True,
            force_download=False,
            strict=True,
            artifact_use="development",
        )
    actor_state = actor.state_dict()
    missing = sorted(set(actor_header) - set(actor_state))
    unexpected = sorted(set(actor_state) - set(actor_header))
    shape_mismatch = sorted(
        name for name in actor_header.keys() & actor_state.keys()
        if list(actor_state[name].shape) != actor_header[name]["shape"]
    )
    require(not missing and not unexpected and not shape_mismatch, "G4P_ACTOR_STRICT_COVERAGE")
    actor.to(device)
    actor_load_seconds = time.perf_counter() - started

    data = config["data"]
    q1, q2, q1_target, q2_target, conversion = build_twin_q(
        _resolve(data["critic_backbone_npz"]),
        _resolve(data["critic_backbone_manifest"]),
        seed=0,
    )
    expected_state = q1.state_dict()
    role_to_module = {
        "online_q1": q1,
        "online_q2": q2,
        "target_q1": q1_target,
        "target_q2": q2_target,
    }
    strict_records = {}
    critic_started = time.perf_counter()
    for group in ("critic_parent", "target_critic_parent"):
        for record in binding[group]["artifacts"]:
            role = record["logical_role"]
            state = torch.load(record["absolute_path"], map_location="cpu", weights_only=True)
            validate_critic_state_against_expected(state, expected_state, role)
            incompatible = role_to_module[role].load_state_dict(state, strict=True)
            require(not incompatible.missing_keys and not incompatible.unexpected_keys, f"G4P_CRITIC_STRICT:{role}")
            strict_records[role] = {
                "key_count": len(state),
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
                "map_location": "cpu",
                "weights_only": True,
                "strict": True,
            }
            del state
    q1.train(True)
    q2.train(True)
    q1_target.make_permanent_eval_target()
    q2_target.make_permanent_eval_target()
    q1, q2, q1_target, q2_target = (
        module.to(device) for module in (q1, q2, q1_target, q2_target)
    )
    critic_load_seconds = time.perf_counter() - critic_started
    require(all(not target.training for target in (q1_target, q2_target)), "G4P_TARGET_NOT_EVAL")
    require(
        all(not parameter.requires_grad for target in (q1_target, q2_target) for parameter in target.parameters()),
        "G4P_TARGET_REQUIRES_GRAD",
    )
    return actor, q1, q2, q1_target, q2_target, {
        "actor_strict_load": True,
        "actor_missing_keys": missing,
        "actor_unexpected_keys": unexpected,
        "actor_shape_mismatches": shape_mismatch,
        "actor_tensor_count": len(actor_header),
        "actor_load_seconds": actor_load_seconds,
        "critic_strict_load": True,
        "target_critic_strict_load": True,
        "critic_records": strict_records,
        "critic_load_seconds": critic_load_seconds,
        "target_fallback_from_online": False,
        "random_critic_fallback": False,
        "critic_backbone_conversion": conversion,
    }


def _actor_group(name: str) -> str:
    if name.startswith("model.vlm_with_expert.vlm."):
        return "vision_smolvlm_language"
    if name.startswith("model.state_proj."):
        return "state_prefix_projection"
    if name.startswith("model.force_branch.force_mlp."):
        return "force_mlp"
    if name.startswith("model.force_branch."):
        return "fusion_moe_refiner_router"
    if name.startswith("model.force_adapter.cross_attention."):
        return "force_cross_attention"
    if name.startswith("model.force_adapter."):
        return "force_action_adapter"
    if name.startswith("model.vlm_with_expert.lm_expert."):
        return "action_expert"
    if name.startswith(("model.action_in_proj.", "model.action_out_proj.", "model.action_time_mlp_in.", "model.action_time_mlp_out.")):
        return "action_io"
    raise G4PError(f"G4P_UNCLASSIFIED_ACTOR_PARAMETER:{name}")


def _actor_group_inventory(actor: nn.Module) -> dict[str, dict[str, int]]:
    result = {name: {"tensor_count": 0, "parameter_count": 0, "trainable_tensor_count": 0, "trainable_parameter_count": 0} for name in ACTOR_GROUPS}
    for name, parameter in actor.named_parameters():
        group = _actor_group(name)
        result[group]["tensor_count"] += 1
        result[group]["parameter_count"] += parameter.numel()
        if parameter.requires_grad:
            result[group]["trainable_tensor_count"] += 1
            result[group]["trainable_parameter_count"] += parameter.numel()
    return result


def _optimizer_factory(
    actor: nn.Module, q1: nn.Module, q2: nn.Module, q1_target: nn.Module, q2_target: nn.Module,
    config: Mapping[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer, dict[str, Any]]:
    from forcesmolvla.rft.frozen_vlm_trainability import (
        apply_frozen_vlm_trainability,
        build_frozen_vlm_actor_optimizer,
    )

    manifest = apply_frozen_vlm_trainability(actor)
    actor_optimizer, _fresh_actor_scheduler, existing_actor_ownership = build_frozen_vlm_actor_optimizer(
        actor, lr=config["optimizer"]["actor"]["lr"],
    )
    critic_parameters = [
        parameter for critic in (q1, q2) for parameter in critic.parameters()
        if parameter.requires_grad
    ]
    require(critic_parameters, "G4P_EMPTY_CRITIC_PARAMETERS")
    critic_optimizer = torch.optim.Adam(
        critic_parameters,
        lr=config["optimizer"]["critic"]["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    actor_ids = [id(parameter) for group in actor_optimizer.param_groups for parameter in group["params"]]
    critic_ids = [id(parameter) for group in critic_optimizer.param_groups for parameter in group["params"]]
    all_optimizer_ids = set(actor_ids) | set(critic_ids)
    actor_trainable_ids = {id(parameter) for parameter in actor.parameters() if parameter.requires_grad}
    critic_trainable_ids = {
        id(parameter) for critic in (q1, q2) for parameter in critic.parameters() if parameter.requires_grad
    }
    frozen_ids = {
        id(parameter)
        for module in (actor, q1, q2)
        for parameter in module.parameters()
        if not parameter.requires_grad
    }
    target_ids = {id(parameter) for target in (q1_target, q2_target) for parameter in target.parameters()}
    fresh_entries = len(actor_optimizer.state) + len(critic_optimizer.state)
    exact = (
        len(actor_ids) == len(set(actor_ids))
        and len(critic_ids) == len(set(critic_ids))
        and set(actor_ids) == actor_trainable_ids
        and set(critic_ids) == critic_trainable_ids
        and not (set(actor_ids) & set(critic_ids))
        and not (all_optimizer_ids & frozen_ids)
        and not (all_optimizer_ids & target_ids)
    )
    inventory = _actor_group_inventory(actor)
    for name in ("vision_smolvlm_language", "state_prefix_projection"):
        require(inventory[name]["trainable_tensor_count"] == 0, f"G4P_FROZEN_ACTOR_GROUP:{name}")
    for name in ACTOR_GROUPS[2:]:
        require(inventory[name]["trainable_tensor_count"] > 0, f"G4P_TRAINABLE_ACTOR_GROUP:{name}")
    require(exact and fresh_entries == 0, "G4P_OPTIMIZER_OWNERSHIP")
    require(
        not actor.model.vlm_with_expert.vlm.training and not actor.model.state_proj.training,
        "G4P_FROZEN_MODULE_NOT_EVAL",
    )
    return actor_optimizer, critic_optimizer, {
        "factory_validated": True,
        "apply_frozen_vlm_trainability_called": True,
        "actor_optimizer_type": type(actor_optimizer).__name__,
        "critic_optimizer_type": type(critic_optimizer).__name__,
        "actor_lr": actor_optimizer.param_groups[0]["lr"],
        "critic_lr": critic_optimizer.param_groups[0]["lr"],
        "actor_trainable_tensor_count": manifest.trainable_actor_parameter_tensors,
        "actor_trainable_parameter_count": manifest.trainable_actor_parameter_count,
        "actor_frozen_tensor_count": manifest.frozen_parameter_tensors,
        "actor_frozen_parameter_count": manifest.frozen_parameter_count,
        "critic_trainable_tensor_count": len(critic_ids),
        "critic_trainable_parameter_count": sum(parameter.numel() for parameter in critic_parameters),
        "actor_critic_parameter_id_intersection": len(set(actor_ids) & set(critic_ids)),
        "frozen_parameters_in_optimizers": len(all_optimizer_ids & frozen_ids),
        "target_parameters_in_optimizers": len(all_optimizer_ids & target_ids),
        "fresh_initial_state_entries": fresh_entries,
        "each_trainable_parameter_exactly_one_owner": exact,
        "actor_groups": inventory,
        "target_trainable_parameter_count": sum(parameter.numel() for target in (q1_target, q2_target) for parameter in target.parameters() if parameter.requires_grad),
        "fresh_constant_actor_scheduler_created_by_existing_factory": True,
        "existing_actor_factory_ownership": existing_actor_ownership,
        "inherited_optimizer_state": False,
        "inherited_scheduler_state": False,
        "inherited_rng_state": False,
        "inherited_sampler_state": False,
    }


def select_fixed_indices(rows: list[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    """Choose non-overlapping real rows, with only the R pool label synthesized."""

    nonterminal = [
        index for index, row in enumerate(rows)
        if all(row["executed_action_mask"]) and not row["terminated"]
    ]
    terminal = [
        index for index, row in enumerate(rows)
        if all(row["executed_action_mask"]) and row["terminated"]
    ]
    require(len(nonterminal) >= 87 and terminal, "G4P_REAL_BATCH_ROWS_UNAVAILABLE")
    rng = random.Random(seed)
    rng.shuffle(nonterminal)
    rng.shuffle(terminal)
    critic = nonterminal[:63] + [terminal[0]]
    actor = nonterminal[63:87]
    require(len(set(critic + actor)) == 88, "G4P_ROW_SELECTION_OVERLAP")
    return {
        "critic_indices": critic,
        "critic_origin_pool": ["synthetic_preflight_R_only"] * 32 + ["offline_D"] * 32,
        "actor_indices": actor,
        "actor_origin_pool": ["synthetic_preflight_R_only"] * 12 + ["offline_D"] * 12,
        "terminal_critic_position": 63,
    }


def _row_records(train_data: Any, indices: list[int], pools: list[str]) -> list[dict[str, Any]]:
    records = train_data.identity_records(indices)
    require(len(records) == len(pools), "G4P_ROW_POOL_COUNT")
    for record, pool in zip(records, pools, strict=True):
        record["origin_pool"] = pool
        record["fixture_kind"] = "real_observation_row_with_preflight_pool_label"
    return records


def _load_real_batches(
    actor: nn.Module,
    binding: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    tools_path = str(ROOT / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from forcesmolvla.rft.training_cycle import FlowCounter, TrainData
    from forcesmolvla.rft.critic import frozen_task_feature, frozen_task_feature_sha256

    data_config = config["data"]
    for path_key, sha_key in (
        ("transition_root", "transition_manifest_sha256"),
        ("lerobot_v3_root", "conversion_manifest_sha256"),
        ("lerobot_v3_root", "normalizer_manifest_sha256"),
        ("critic_backbone_npz", "critic_backbone_npz_sha256"),
        ("critic_backbone_manifest", "critic_backbone_manifest_sha256"),
    ):
        path = _resolve(data_config[path_key])
        if sha_key == "transition_manifest_sha256":
            path /= "g1_manifest.json"
        elif sha_key == "conversion_manifest_sha256":
            path /= "conversion_manifest.json"
        elif sha_key == "normalizer_manifest_sha256":
            path /= "normalizer_manifest.json"
        require(sha256_file(path) == data_config[sha_key], f"G4P_DATA_SHA:{sha_key}")

    train_data = TrainData()
    selection = select_fixed_indices(
        train_data.rows, seed=data_config["row_selection_seed"],
    )
    task = torch.from_numpy(frozen_task_feature()).to(device=device, dtype=torch.float32)
    require(
        frozen_task_feature_sha256() == binding["task_feature_binding"]["logical_object_sha256"],
        "G4P_TASK_FEATURE_BINDING",
    )
    critic_batch = train_data.build_batch(
        selection["critic_indices"], actor, device,
        canonical_task_feature=task, include_flow_actions=False,
    )
    actor_batch = train_data.build_batch(
        selection["actor_indices"], actor, device,
        canonical_task_feature=task, include_flow_actions=True,
    )
    require(
        bool(critic_batch["behavior_mask"].all())
        and critic_batch["terminated"][-1]
        and not bool(critic_batch["terminated"][:-1].any()),
        "G4P_CRITIC_TERMINAL_LAYOUT",
    )
    require(
        torch.equal(critic_batch["discount"], (~critic_batch["terminated"]).float() * 0.99)
        and torch.equal(critic_batch["bootstrap_mask"].bool(), ~critic_batch["terminated"]),
        "G4P_REWARD_TERMINAL_DISCOUNT",
    )
    # Retain only tensors consumed by the Stage-3 pure-TD and Actor APIs.
    for key in ("current_actor_batch", "mc_return", "indices", "identities"):
        critic_batch.pop(key, None)
    for key in (
        "next_observation", "next_actor_batch", "behavior_action", "behavior_mask",
        "reward", "terminated", "bootstrap_mask", "discount", "mc_return", "indices",
        "identities",
    ):
        actor_batch.pop(key, None)
    gc.collect()
    evidence = {
        "source": "real Phase-2 automatic-detector G1 train rows through TrainData.build_batch",
        "lerobot_v3_root": str(_resolve(data_config["lerobot_v3_root"])),
        "transition_root": str(_resolve(data_config["transition_root"])),
        "critic_batch": {
            "size": 64,
            "R_count": 32,
            "D_count": 32,
            "R_source": "synthetic_preflight_R_only",
            "D_source": "offline_D",
            "rows": _row_records(
                train_data,
                selection["critic_indices"],
                selection["critic_origin_pool"],
            ),
        },
        "actor_batch": {
            "size": 24,
            "R_count": 12,
            "D_count": 12,
            "R_source": "synthetic_preflight_R_only",
            "D_source": "offline_D_expert_flow_matching",
            "rows": _row_records(
                train_data,
                selection["actor_indices"],
                selection["actor_origin_pool"],
            ),
        },
        "row_sets_non_overlapping": True,
        "images": {
            "actor": "real decoded RGB float32 [0,1] via actor_batch",
            "critic": "real decoded RGB uint8 [0,255] via CriticObservation",
            "camera_count": 2,
        },
        "state_wrench": "real normalized state7/wrench6 through frozen runtime normalizer",
        "normalizer_binding": {
            "path": binding["normalizer_binding"]["absolute_path"],
            "sha256": binding["normalizer_binding"]["sha256"],
            "exactly_once": True,
        },
        "task_feature_digest": frozen_task_feature_sha256(),
        "task_feature_dimension": 256,
        "reward_terminal_source": data_config["reward_terminal_source"],
        "synthetic_R_reward_terminal_note": "reused frozen real-row values for numerical shape only; not online evidence",
        "writes_real_replay": False,
        "validation_or_test_rows_read": 0,
        "manual_labels_read": 0,
        "mc_return_input_to_online_loss": False,
    }
    return critic_batch, actor_batch, evidence, FlowCounter(
        inference_batch_size=config["batching"]["flow_inference_subbatch"]
    )


def _stats(value: Tensor) -> dict[str, float]:
    data = value.detach().float()
    require(bool(torch.isfinite(data).all()), "G4P_NONFINITE_STAT_INPUT")
    return {
        "mean": float(data.mean().cpu()),
        "std": float(data.std(unbiased=False).cpu()),
        "min": float(data.min().cpu()),
        "max": float(data.max().cpu()),
    }


def _snapshot_trainable(*named_modules: tuple[str, nn.Module]) -> dict[str, Tensor]:
    return {
        f"{owner}.{name}": parameter.detach().cpu().clone()
        for owner, module in named_modules
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _delta_summary(
    snapshot: Mapping[str, Tensor], *named_modules: tuple[str, nn.Module], actor_groups: bool = False,
) -> dict[str, Any]:
    squares: dict[str, float] = {}
    maxima: dict[str, float] = {}
    changed: dict[str, int] = {}
    tensors: dict[str, int] = {}
    for owner, module in named_modules:
        for name, parameter in module.named_parameters():
            key = f"{owner}.{name}"
            if key not in snapshot:
                continue
            group = _actor_group(name) if actor_groups else owner
            delta = parameter.detach().cpu().float() - snapshot[key].float()
            squares[group] = squares.get(group, 0.0) + float(delta.square().sum())
            maxima[group] = max(maxima.get(group, 0.0), float(delta.abs().max()))
            changed[group] = changed.get(group, 0) + int(bool(torch.count_nonzero(delta)))
            tensors[group] = tensors.get(group, 0) + 1
    return {
        group: {
            "l2_norm": math.sqrt(squares[group]),
            "max_abs": maxima[group],
            "changed_tensor_count": changed[group],
            "tensor_count": tensors[group],
        }
        for group in sorted(squares)
    }


def _gradient_norms_by_actor_group(actor: nn.Module) -> dict[str, float]:
    squares = {name: 0.0 for name in ACTOR_GROUPS}
    for name, parameter in actor.named_parameters():
        if parameter.grad is not None:
            squares[_actor_group(name)] += float(parameter.grad.detach().float().square().sum())
    return {name: math.sqrt(value) for name, value in squares.items()}


def _global_gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    values = [float(parameter.grad.detach().float().square().sum()) for parameter in parameters if parameter.grad is not None]
    return math.sqrt(sum(values))


def _gradient_geometry(
    left: Iterable[Tensor | None], right: Iterable[Tensor | None], *, left_weight: float, right_weight: float,
) -> dict[str, float]:
    left_sq = right_sq = dot = 0.0
    for a, b in zip(left, right, strict=True):
        if a is None and b is None:
            continue
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        a32, b32 = a.detach().float(), b.detach().float()
        left_sq += float(a32.square().sum())
        right_sq += float(b32.square().sum())
        dot += float((a32 * b32).sum())
    left_norm, right_norm = math.sqrt(left_sq), math.sqrt(right_sq)
    return {
        "raw_fm_norm": left_norm,
        "raw_q_norm": right_norm,
        "weighted_fm_norm": abs(left_weight) * left_norm,
        "weighted_q_norm": abs(right_weight) * right_norm,
        "gradient_cosine": dot / max(left_norm * right_norm, torch.finfo(torch.float32).tiny),
    }


def _module_all_finite(*modules: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for module in modules
        for value in (*module.parameters(), *module.buffers())
        if value.is_floating_point()
    )


def _critic_step(
    *,
    cycle: int,
    substep: int,
    actor: nn.Module,
    q1: nn.Module,
    q2: nn.Module,
    q1_target: nn.Module,
    q2_target: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    flow_counter: Any,
    noise_generator: torch.Generator,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from forcesmolvla.rft.training_cycle import slice_actor_batch
    from forcesmolvla.rft.critic_action_adapter_v2 import critic_action_for_q_guidance_v2
    from forcesmolvla.rft.stage3.losses import compute_online_twin_q_td_loss
    from forcesmolvla.rft.training_cycle import module_state_sha256, polyak_update_verified

    device = batch["reward"].device
    nonterminal = ~batch["terminated"]
    nonterminal_count = int(nonterminal.sum())
    require(nonterminal_count == 63 and bool(batch["terminated"][-1]), "G4P_FIXED_TERMINAL_LAYOUT")
    next_actor_batch = slice_actor_batch(batch["next_actor_batch"], 0, nonterminal_count)
    noise = torch.randn(
        nonterminal_count, 50, 7,
        generator=noise_generator, device=device, dtype=torch.float32,
    )
    noise_digest = tensor_sha256(noise)
    actor_before = module_state_sha256(actor)
    critic_snapshot = _snapshot_trainable(("q1", q1), ("q2", q2))
    optimizer.zero_grad(set_to_none=True)
    actor.zero_grad(set_to_none=True)
    q1_target.zero_grad(set_to_none=True)
    q2_target.zero_grad(set_to_none=True)
    target_outputs: dict[str, list[Tensor]] = {"q1": [], "q2": []}
    hooks = [
        q1_target.register_forward_hook(lambda _m, _i, out: target_outputs["q1"].append(out.detach().clone())),
        q2_target.register_forward_hook(lambda _m, _i, out: target_outputs["q2"].append(out.detach().clone())),
    ]

    def next_action(_next_observation: Any) -> Tensor:
        training = actor.training
        actor.eval()
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunk = flow_counter.sample(
                    actor, next_actor_batch, noise,
                    call_id=f"g4p-cycle={cycle}-critic={substep}", purpose="td_next",
                )
            return critic_action_for_q_guidance_v2(
                chunk,
                delta_action_mean7=batch["delta_mean"],
                delta_action_std7=batch["delta_std"],
            ).detach().float()
        finally:
            actor.train(training)

    started = time.perf_counter()
    try:
        result = compute_online_twin_q_td_loss(
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            observation=batch["current_observation"],
            next_observation=batch["next_observation"],
            ack_behavior_action_k7=batch["behavior_action"],
            behavior_mask=batch["behavior_mask"],
            reward=batch["reward"],
            discount=batch["discount"],
            terminated=batch["terminated"],
            bootstrap_mask=batch["bootstrap_mask"].bool(),
            next_policy_action_fn=next_action,
        )
    finally:
        for hook in hooks:
            hook.remove()
    require(len(target_outputs["q1"]) == len(target_outputs["q2"]) == 1, "G4P_TARGET_FORWARD_COUNT")
    expected_target = batch["reward"].clone()
    expected_target[nonterminal] += batch["discount"][nonterminal] * torch.minimum(
        target_outputs["q1"][0], target_outputs["q2"][0]
    )
    require(torch.equal(result.target, expected_target.float()), "G4P_TARGET_NOT_MIN_TWIN_Q")
    require(
        result.calql_candidate_calls == result.random_candidate_calls == result.mc_return_reads == 0,
        "G4P_PURE_TD_COUNTER",
    )
    result.total.backward()
    trainable = [
        parameter for critic in (q1, q2) for parameter in critic.parameters()
        if parameter.requires_grad
    ]
    require(
        all(parameter.grad is None for parameter in actor.parameters())
        and all(parameter.grad is None for target in (q1_target, q2_target) for parameter in target.parameters()),
        "G4P_CRITIC_GRADIENT_OWNERSHIP",
    )
    require(
        all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in trainable),
        "G4P_CRITIC_GRADIENT_NONFINITE",
    )
    preclip = _global_gradient_norm(trainable)
    torch.nn.utils.clip_grad_norm_(trainable, config["optimizer"]["critic"]["grad_clip_norm"])
    postclip = _global_gradient_norm(trainable)
    optimizer.step()
    delta = _delta_summary(critic_snapshot, ("q1", q1), ("q2", q2))
    require(all(value["l2_norm"] > 0 for value in delta.values()), "G4P_CRITIC_PARAMETER_DELTA")
    polyak_records = []
    for online, target, name in ((q1, q1_target, "q1_target"), (q2, q2_target, "q2_target")):
        record = polyak_update_verified(
            online, target, tau=config["optimizer"]["polyak_tau"], target_name=name,
        )
        polyak_records.append({key: value for key, value in record.items() if key != "tensors"})
    optimizer.zero_grad(set_to_none=True)
    elapsed = time.perf_counter() - started
    require(module_state_sha256(actor) == actor_before, "G4P_CRITIC_STEP_CHANGED_ACTOR")
    require(_module_all_finite(q1, q2, q1_target, q2_target), "G4P_CRITIC_OR_TARGET_NONFINITE")
    residual = torch.cat((result.q1_value - result.target, result.q2_value - result.target))
    target_min = torch.minimum(target_outputs["q1"][0], target_outputs["q2"][0])
    return {
        "cycle": cycle,
        "critic_substep": substep,
        "loss": {
            "td_total": float(result.total.detach().cpu()),
            "td_q1": float(result.q1_loss.detach().cpu()),
            "td_q2": float(result.q2_loss.detach().cpu()),
        },
        "statistics": {
            "q1": _stats(result.q1_value),
            "q2": _stats(result.q2_value),
            "target_q_min": _stats(target_min),
            "td_target": _stats(result.target),
            "bellman_residual": _stats(residual),
        },
        "gradient": {"preclip_norm": preclip, "postclip_norm": postclip, "finite": True},
        "trainable_parameter_delta": delta,
        "actor_parameter_delta_exact_zero": True,
        "target_gradient_count": 0,
        "target_uses_min_twin_q": True,
        "next_actor_calls": result.next_actor_calls,
        "target_q1_calls": result.target_q1_calls,
        "target_q2_calls": result.target_q2_calls,
        "terminal_row_count": int(batch["terminated"].sum()),
        "terminal_rows_filtered_before_next_actor_and_target_q": True,
        "flow_noise_sha256": noise_digest,
        "polyak": polyak_records,
        "wall_seconds": elapsed,
    }


def _actor_step(
    *,
    cycle: int,
    actor: nn.Module,
    q1: nn.Module,
    q2: nn.Module,
    q1_target: nn.Module,
    q2_target: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    origin_pools: list[str],
    flow_counter: Any,
    fm_noise_generator: torch.Generator,
    fm_time_generator: torch.Generator,
    q_noise_generator: torch.Generator,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from forcesmolvla.rft.training_cycle import slice_actor_batch
    from forcesmolvla.force_token import RouterState
    from forcesmolvla.rft.critic_action_adapter_v2 import critic_action_for_q_guidance_v2
    from forcesmolvla.rft.frozen_vlm_trainability import frozen_prefix_flow_matching_terms
    from forcesmolvla.rft.stage3.losses import (
        compute_stage3_actor_objective,
        compute_stage3_min_twin_q_actor_loss,
    )
    from forcesmolvla.rft.training_cycle import module_state_sha256
    from forcesmolvla.router_training import collect_pass_a_statistics, microbatch_two_pass_terms

    device = batch["current_observation"].camera1.device
    batch_size = config["batching"]["actor_batch_size"]
    microbatch = config["batching"]["flow_inference_subbatch"]
    require(batch_size == 24 and microbatch == 4 and len(origin_pools) == batch_size, "G4P_ACTOR_BATCH")
    actor_parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    actor_snapshot = _snapshot_trainable(("actor", actor))
    critic_before = {
        name: module_state_sha256(module)
        for name, module in (("q1", q1), ("q2", q2), ("q1_target", q1_target), ("q2_target", q2_target))
    }
    optimizer.zero_grad(set_to_none=True)
    for critic in (q1, q2, q1_target, q2_target):
        critic.zero_grad(set_to_none=True)
    valid = batch["current_actor_batch"]["action_valid_mask"].bool()
    expert_rows = torch.tensor(
        [pool == "offline_D" for pool in origin_pools], dtype=torch.bool, device=device,
    )
    total_expert_features = int((valid & expert_rows[:, None]).sum()) * 7
    total_expert_tokens = int((valid & expert_rows[:, None]).sum())
    require(total_expert_features > 0, "G4P_NO_EXPERT_FM_FEATURES")
    records = []
    fm_total = q_total = balance_total = z_total = 0.0
    q1_values: list[Tensor] = []
    q2_values: list[Tensor] = []
    tcp_q_gradient_square = 0.0
    gripper_q_gradient_max = 0.0
    post_k_q_gradient_max = 0.0
    expert_gripper_fm_gradient_square = 0.0
    autonomous_fm_gradient_max = 0.0
    zero_expert_batches = 0
    zero_expert_graph_connected_finite = True
    prefix_contracts = []
    noise_records = []
    geometry: dict[str, float] | None = None
    drift_probe: dict[str, Any] | None = None
    actor.train(True)
    started = time.perf_counter()
    for start in range(0, batch_size, microbatch):
        stop = start + microbatch
        index = torch.arange(start, stop, device=device)
        actor_micro = slice_actor_batch(batch["current_actor_batch"], start, stop)
        observation = batch["current_observation"].index(index)
        local_valid = valid[start:stop]
        local_expert_rows = expert_rows[start:stop]
        expert_mask = local_expert_rows[:, None, None].expand(-1, 50, 7).clone()
        fm_noise = torch.randn(
            microbatch, 50, 7, generator=fm_noise_generator, device=device, dtype=torch.float32,
        )
        fm_time = torch.rand(
            microbatch, generator=fm_time_generator, device=device, dtype=torch.float32,
        )
        velocity_outputs: list[Tensor] = []

        def capture_velocity(_module: nn.Module, _inputs: Any, output: Tensor) -> None:
            output.retain_grad()
            velocity_outputs.append(output)

        hook = actor.model.action_out_proj.register_forward_hook(capture_velocity)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                flow_losses, feature_mask, router_state, prefix_contract = frozen_prefix_flow_matching_terms(
                    actor,
                    actor_micro,
                    noise=fm_noise,
                    time=fm_time,
                    call_id=f"g4p-cycle={cycle}-fm={start}:{stop}",
                )
        finally:
            hook.remove()
        require(len(velocity_outputs) == 1, "G4P_FM_VELOCITY_CAPTURE")
        prefix_contracts.append(prefix_contract)
        detached_router = RouterState(
            logits_fp32=router_state.logits_fp32.detach(),
            probabilities_fp32=router_state.probabilities_fp32.detach(),
            route_ids=router_state.route_ids.detach(),
            valid_mask=router_state.valid_mask.detach(),
        )
        statistics = collect_pass_a_statistics([detached_router], [feature_mask])
        auxiliary = microbatch_two_pass_terms(flow_losses, router_state, statistics)
        flow7 = flow_losses[..., :7]

        q_noise = torch.randn(
            microbatch, 50, 7, generator=q_noise_generator, device=device, dtype=torch.float32,
        )
        actor.eval()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action_chunk = flow_counter.sample(
                actor,
                actor_micro,
                q_noise,
                call_id=f"g4p-cycle={cycle}-actor-q={start}:{stop}",
                purpose="actor_guidance",
            )
            action_chunk.retain_grad()
            q_contract_loss, q1_value, q2_value, action_for_q = compute_stage3_min_twin_q_actor_loss(
                q1=q1,
                q2=q2,
                observation=observation,
                normalized_flow_action_chunk7=action_chunk,
                delta_action_mean7=batch["delta_mean"],
                delta_action_std7=batch["delta_std"],
            )
        if drift_probe is None:
            drift_probe = {
                "actor_batch": actor_micro,
                "noise": q_noise.detach().clone(),
                "action_before": action_for_q.detach().clone(),
            }
        actor.train(True)
        expert_count = int((local_valid & local_expert_rows[:, None]).sum()) * 7
        fm_weight = expert_count / total_expert_features
        q_weight = microbatch / batch_size
        actor_terms = compute_stage3_actor_objective(
            per_feature_flow_loss=flow7,
            action_valid_mask_h50=local_valid,
            expert_feature_mask_h50x7=expert_mask,
            q1_actor_value=q1_value,
            q2_actor_value=q2_value,
            actor_q_valid=torch.ones(microbatch, dtype=torch.bool, device=device),
            balance_loss=auxiliary.balance,
            z_loss=auxiliary.z,
            beta=config["loss"]["beta_expert_flow_matching"] * fm_weight,
            eta=config["loss"]["eta_actor_q"] * q_weight,
            balance_weight=config["loss"]["balance_weight"] / (batch_size / microbatch),
            z_weight=config["loss"]["z_weight"] / (batch_size / microbatch),
        )
        require(torch.equal(q_contract_loss, actor_terms.actor_q), "G4P_ACTOR_NOT_MIN_TWIN_Q")
        q_action_gradient = torch.autograd.grad(
            q_contract_loss, action_chunk, retain_graph=True,
        )[0]
        tcp_q_gradient_square += float(q_action_gradient[:, :3, :6].float().square().sum())
        gripper_q_gradient_max = max(
            gripper_q_gradient_max,
            float(q_action_gradient[:, :3, 6].float().abs().max()),
        )
        post_k_q_gradient_max = max(
            post_k_q_gradient_max,
            float(q_action_gradient[:, 3:].float().abs().max()),
        )
        fm_feature_gradient = torch.autograd.grad(
            actor_terms.expert_flow_matching, flow7, retain_graph=True,
        )[0]
        autonomous_fm_gradient_max = max(
            autonomous_fm_gradient_max,
            float(fm_feature_gradient[~local_expert_rows].abs().max())
            if bool((~local_expert_rows).any()) else 0.0,
        )
        if expert_count == 0:
            zero_expert_batches += 1
            zero_expert_graph_connected_finite &= (
                actor_terms.expert_feature_count == 0
                and actor_terms.expert_flow_matching.grad_fn is not None
                and bool(torch.isfinite(actor_terms.expert_flow_matching))
                and float(fm_feature_gradient.abs().max()) == 0.0
            )
        else:
            velocity_gradient = torch.autograd.grad(
                actor_terms.expert_flow_matching, velocity_outputs[0], retain_graph=True,
            )[0]
            expert_gripper_fm_gradient_square += float(
                velocity_gradient[local_expert_rows, :, 6].float().square().sum()
            )
            if geometry is None:
                fm_gradients = torch.autograd.grad(
                    actor_terms.expert_flow_matching,
                    actor_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                q_gradients = torch.autograd.grad(
                    q_contract_loss,
                    actor_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                geometry = _gradient_geometry(
                    fm_gradients,
                    q_gradients,
                    left_weight=config["loss"]["beta_expert_flow_matching"] * fm_weight,
                    right_weight=config["loss"]["eta_actor_q"] * q_weight,
                )
        actor_terms.total.backward()
        q1_values.append(q1_value.detach())
        q2_values.append(q2_value.detach())
        fm_total += fm_weight * float(actor_terms.expert_flow_matching.detach())
        q_total += q_weight * float(actor_terms.actor_q.detach())
        balance_total += float(auxiliary.balance.detach()) / (batch_size / microbatch)
        z_total += float(auxiliary.z.detach()) / (batch_size / microbatch)
        noise_records.append({
            "rows": [start, stop],
            "fm_noise_sha256": tensor_sha256(fm_noise),
            "fm_time_sha256": tensor_sha256(fm_time),
            "actor_q_noise_sha256": tensor_sha256(q_noise),
        })
        records.append({
            "rows": [start, stop],
            "origin_pools": origin_pools[start:stop],
            "expert_feature_count": actor_terms.expert_feature_count,
            "fm_loss": float(actor_terms.expert_flow_matching.detach()),
            "actor_q_loss": float(actor_terms.actor_q.detach()),
            "balance": float(auxiliary.balance.detach()),
            "z": float(auxiliary.z.detach()),
        })
        del flow_losses, flow7, action_chunk, action_for_q, actor_terms, velocity_outputs
        gc.collect()
        torch.cuda.empty_cache()

    require(geometry is not None and all(math.isfinite(value) for value in geometry.values()), "G4P_GRADIENT_GEOMETRY")
    require(
        all(parameter.grad is None for critic in (q1, q2, q1_target, q2_target) for parameter in critic.parameters()),
        "G4P_ACTOR_BACKWARD_TOUCHED_CRITIC",
    )
    require(
        all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in actor_parameters),
        "G4P_ACTOR_GRADIENT_NONFINITE",
    )
    module_norms = _gradient_norms_by_actor_group(actor)
    require(all(module_norms[name] > 0 for name in ACTOR_GROUPS[2:]), f"G4P_REQUIRED_ACTOR_GRADIENT:{module_norms}")
    preclip = _global_gradient_norm(actor_parameters)
    torch.nn.utils.clip_grad_norm_(actor_parameters, config["optimizer"]["actor"]["grad_clip_norm"])
    postclip = _global_gradient_norm(actor_parameters)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    require(drift_probe is not None, "G4P_ACTION_DRIFT_PROBE_MISSING")
    actor.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        drift_chunk_after = flow_counter.sample(
            actor,
            drift_probe["actor_batch"],
            drift_probe["noise"],
            call_id=f"g4p-cycle={cycle}-post-step-drift",
            purpose="actor_guidance",
        )
    drift_action_after = critic_action_for_q_guidance_v2(
        drift_chunk_after,
        delta_action_mean7=batch["delta_mean"],
        delta_action_std7=batch["delta_std"],
    ).detach().float()
    actor.train(True)
    drift_action_before = drift_probe["action_before"].float()
    tcp_drift = drift_action_after[..., :6] - drift_action_before[..., :6]
    gripper_changed = drift_action_after[..., 6] != drift_action_before[..., 6]
    action_drift = {
        "probe_rows": 4,
        "fixed_noise_sha256": tensor_sha256(drift_probe["noise"]),
        "tcp6_l2_norm": float(tcp_drift.square().sum().sqrt()),
        "tcp6_mean_abs": float(tcp_drift.abs().mean()),
        "tcp6_max_abs": float(tcp_drift.abs().max()),
        "gripper_changed_slot_count": int(gripper_changed.sum()),
        "gripper_changed_slot_rate": float(gripper_changed.float().mean()),
        "gripper_open_fraction_before": float(
            (drift_action_before[..., 6] == drift_action_before[..., 6].max()).float().mean()
        ),
        "gripper_open_fraction_after": float(
            (drift_action_after[..., 6] == drift_action_after[..., 6].max()).float().mean()
        ),
    }
    require(all(math.isfinite(value) for key, value in action_drift.items() if isinstance(value, float)), "G4P_ACTION_DRIFT_NONFINITE")
    actor_delta = _delta_summary(actor_snapshot, ("actor", actor), actor_groups=True)
    require(
        all(actor_delta[name]["l2_norm"] > 0 for name in ACTOR_GROUPS[2:]),
        f"G4P_ACTOR_PARAMETER_DELTA:{actor_delta}",
    )
    critic_after = {
        name: module_state_sha256(module)
        for name, module in (("q1", q1), ("q2", q2), ("q1_target", q1_target), ("q2_target", q2_target))
    }
    require(critic_before == critic_after, "G4P_ACTOR_STEP_CHANGED_CRITIC")
    require(
        tcp_q_gradient_square > 0.0
        and gripper_q_gradient_max == 0.0
        and post_k_q_gradient_max == 0.0
        and expert_gripper_fm_gradient_square > 0.0
        and autonomous_fm_gradient_max == 0.0
        and zero_expert_batches == 3
        and zero_expert_graph_connected_finite,
        "G4P_ACTOR_GRADIENT_SEMANTICS",
    )
    elapsed = time.perf_counter() - started
    return {
        "cycle": cycle,
        "loss": {
            "fm_expert": fm_total,
            "actor_q_min_twin": q_total,
            "balance": balance_total,
            "z": z_total,
            "weighted_total": (
                config["loss"]["beta_expert_flow_matching"] * fm_total
                + config["loss"]["eta_actor_q"] * q_total
                + config["loss"]["balance_weight"] * balance_total
                + config["loss"]["z_weight"] * z_total
            ),
        },
        "actor_action_q": {
            "q1": _stats(torch.cat(q1_values)),
            "q2": _stats(torch.cat(q2_values)),
        },
        "expert_feature_count": total_expert_features,
        "expert_token_count": total_expert_tokens,
        "zero_expert_batch_count": zero_expert_batches,
        "zero_expert_batch_rate": zero_expert_batches / (batch_size / microbatch),
        "zero_expert_graph_connected_finite": zero_expert_graph_connected_finite,
        "autonomous_fm_gradient_max_abs": autonomous_fm_gradient_max,
        "tcp6_q_gradient_norm": math.sqrt(tcp_q_gradient_square),
        "gripper_q_gradient_max_abs": gripper_q_gradient_max,
        "post_K_q_gradient_max_abs": post_k_q_gradient_max,
        "expert_gripper_fm_gradient_norm": math.sqrt(expert_gripper_fm_gradient_square),
        "gradient_geometry": geometry,
        "per_module_gradient_norm": module_norms,
        "gradient_preclip_norm": preclip,
        "gradient_postclip_norm": postclip,
        "trainable_parameter_delta": actor_delta,
        "action_drift": action_drift,
        "online_and_target_critic_parameter_delta_exact_zero": True,
        "prefix_contract": {
            "all_prefix_prefills_no_grad": all(not item["prefix_grad_enabled"] for item in prefix_contracts),
            "all_prefix_representations_detached": all(item["prefix_representation_detached"] for item in prefix_contracts),
            "all_prefix_caches_detached": all(item["prefix_cache_detached"] for item in prefix_contracts),
            "prefix_prefill_count": sum(item["prefix_prefill_count"] for item in prefix_contracts),
            "force_kv_projection_count": sum(item["force_kv_projection_count"] for item in prefix_contracts),
        },
        "noise": noise_records,
        "microbatches": records,
        "wall_seconds": elapsed,
    }


def _terminal_probe(
    *, q1: nn.Module, q2: nn.Module, q1_target: nn.Module, q2_target: nn.Module,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    from forcesmolvla.rft.stage3.losses import compute_online_twin_q_td_loss

    index = batch["terminated"]
    calls = {"actor": 0, "q1_target": 0, "q2_target": 0}

    def forbidden(_observation: Any) -> Tensor:
        calls["actor"] += 1
        raise G4PError("G4P_TERMINAL_NEXT_ACTOR_CALLED")

    hooks = [
        q1_target.register_forward_hook(lambda *_: calls.__setitem__("q1_target", calls["q1_target"] + 1)),
        q2_target.register_forward_hook(lambda *_: calls.__setitem__("q2_target", calls["q2_target"] + 1)),
    ]
    try:
        result = compute_online_twin_q_td_loss(
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            observation=batch["current_observation"].index(index),
            next_observation=batch["next_observation"].index(index),
            ack_behavior_action_k7=batch["behavior_action"][index],
            behavior_mask=batch["behavior_mask"][index],
            reward=batch["reward"][index],
            discount=batch["discount"][index],
            terminated=batch["terminated"][index],
            bootstrap_mask=batch["bootstrap_mask"][index].bool(),
            next_policy_action_fn=forbidden,
        )
    finally:
        for hook in hooks:
            hook.remove()
    require(calls == {"actor": 0, "q1_target": 0, "q2_target": 0}, f"G4P_TERMINAL_CALLS:{calls}")
    require(torch.equal(result.target, batch["reward"][index]), "G4P_TERMINAL_TARGET_NOT_REWARD")
    return {
        "terminal_rows": int(index.sum()),
        "next_actor_calls": calls["actor"],
        "target_q1_calls": calls["q1_target"],
        "target_q2_calls": calls["q2_target"],
        "target_equals_reward_exact": True,
    }


def _summarize_cycle_performance(
    cycle: int, kind: str, started: float, device: torch.device, sync_points: list[str],
) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    sync_points.append(f"cycle={cycle}:end")
    elapsed = time.perf_counter() - started
    memory = _cuda_memory(device)
    return {
        "cycle": cycle,
        "kind": kind,
        "wall_seconds": elapsed,
        "effective_row_uses": 2 * 64 + 24,
        "effective_row_uses_per_second": (2 * 64 + 24) / elapsed,
        "cycles_per_hour_estimate": 3600.0 / elapsed,
        **memory,
    }


def run_gpu_preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    from forcesmolvla.rft.frozen_vlm_trainability import frozen_state_digest
    from forcesmolvla.rft.stage3.parent import (
        load_parent_binding,
        preflight_parent_binding,
        validate_parent_binding_semantics,
    )

    config_path = Path(config_path).resolve()
    config = validate_gpu_preflight_config(_load_mapping(config_path))
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG", config["environment"]["cublas_workspace_config"]
    )
    require(
        os.environ["CUBLAS_WORKSPACE_CONFIG"] == config["environment"]["cublas_workspace_config"],
        "G4P_CUBLAS_WORKSPACE_CONFIG",
    )
    binding_path = _resolve(config["parent_binding"]["path"])
    # G0A is rerun before any CUDA initialization.
    require(not torch.cuda.is_initialized(), "G4P_CUDA_INITIALIZED_BEFORE_G0A")
    g0a = preflight_parent_binding(binding_path)
    require(
        g0a["tool_status"] == "PASS"
        and g0a["G0_FINAL_PARENT_BINDING"] == "BOUND_APPROVED_HYBRID",
        "G4P_G0A_PARENT_PREFLIGHT",
    )
    binding = validate_parent_binding_semantics(load_parent_binding(binding_path))
    require(
        binding["binding_id"] == config["parent_binding"]["binding_id"]
        and binding["binding_type"] == config["parent_binding"]["binding_type"],
        "G4P_BINDING_ID_OR_TYPE",
    )
    parent_records = _selected_parent_records(binding)
    parent_before = _hash_parent_records(parent_records)
    binding_sha_before = sha256_file(binding_path)
    device, environment = _freeze_environment(config)
    sync_points: list[str] = []
    torch.cuda.synchronize(device)
    sync_points.append("environment:after_freeze")
    torch.cuda.reset_peak_memory_stats(device)

    actor = q1 = q2 = q1_target = q2_target = None
    actor_optimizer = critic_optimizer = None
    critic_batch = actor_batch = None
    try:
        actor, q1, q2, q1_target, q2_target, parent_load = _strict_load_parents(
            binding, config, device,
        )
        torch.cuda.synchronize(device)
        sync_points.append("parent_load:end")
        load_only = _cuda_memory(device)
        actor_optimizer, critic_optimizer, ownership = _optimizer_factory(
            actor, q1, q2, q1_target, q2_target, config,
        )
        frozen_before = frozen_state_digest(actor)
        critic_batch, actor_batch, data_evidence, flow_counter = _load_real_batches(
            actor, binding, config, device,
        )
        torch.cuda.synchronize(device)
        sync_points.append("real_batch_load:end")
        terminal_probe = _terminal_probe(
            q1=q1, q2=q2, q1_target=q1_target, q2_target=q2_target,
            batch=critic_batch,
        )

        seed = config["environment"]["seed"]
        generators = {
            "td_noise": torch.Generator(device=device).manual_seed(seed + 101),
            "fm_noise": torch.Generator(device=device).manual_seed(seed + 102),
            "fm_time": torch.Generator(device=device).manual_seed(seed + 103),
            "actor_q_noise": torch.Generator(device=device).manual_seed(seed + 104),
        }
        generator_initial_digests = {
            name: tensor_sha256(generator.get_state()) for name, generator in generators.items()
        }
        critic_reports: list[dict[str, Any]] = []
        actor_reports: list[dict[str, Any]] = []
        cycle_performance: list[dict[str, Any]] = []
        counters = {"calql": 0, "cql": 0, "random_candidates": 0, "mc_return": 0}

        def forbidden(name: str):
            def call(*_args: Any, **_kwargs: Any) -> None:
                counters[name] += 1
                raise G4PError(f"G4P_FORBIDDEN_ONLINE_LOSS_PATH:{name}")
            return call

        from forcesmolvla.rft import losses as stage2_losses

        torch.cuda.reset_peak_memory_stats(device)
        with (
            patch.object(stage2_losses, "compute_calql_penalty", side_effect=forbidden("calql")),
            patch.object(stage2_losses, "compute_twin_q_critic_loss", side_effect=forbidden("cql")),
            patch.object(stage2_losses, "evaluate_calql_candidates", side_effect=forbidden("random_candidates")),
            patch.object(stage2_losses, "validate_mc_return_recurrence", side_effect=forbidden("mc_return")),
        ):
            for cycle in range(4):
                kind = "warmup" if cycle == 0 else "measured"
                torch.cuda.synchronize(device)
                sync_points.append(f"cycle={cycle}:start")
                cycle_started = time.perf_counter()
                for substep in (1, 2):
                    critic_reports.append(
                        _critic_step(
                            cycle=cycle,
                            substep=substep,
                            actor=actor,
                            q1=q1,
                            q2=q2,
                            q1_target=q1_target,
                            q2_target=q2_target,
                            optimizer=critic_optimizer,
                            batch=critic_batch,
                            flow_counter=flow_counter,
                            noise_generator=generators["td_noise"],
                            config=config,
                        )
                    )
                actor_reports.append(
                    _actor_step(
                        cycle=cycle,
                        actor=actor,
                        q1=q1,
                        q2=q2,
                        q1_target=q1_target,
                        q2_target=q2_target,
                        optimizer=actor_optimizer,
                        batch=actor_batch,
                        origin_pools=data_evidence["actor_batch"]["rows"] and [
                            row["origin_pool"] for row in data_evidence["actor_batch"]["rows"]
                        ],
                        flow_counter=flow_counter,
                        fm_noise_generator=generators["fm_noise"],
                        fm_time_generator=generators["fm_time"],
                        q_noise_generator=generators["actor_q_noise"],
                        config=config,
                    )
                )
                cycle_performance.append(
                    _summarize_cycle_performance(
                        cycle, kind, cycle_started, device, sync_points,
                    )
                )
                if cycle == 0:
                    warmup_peak = _cuda_memory(device)
                    torch.cuda.reset_peak_memory_stats(device)

        require(counters == {"calql": 0, "cql": 0, "random_candidates": 0, "mc_return": 0}, f"G4P_FORBIDDEN_COUNTERS:{counters}")
        require(len(critic_reports) == 8 and len(actor_reports) == 4, "G4P_UPDATE_COUNTS")
        measured_peak = _cuda_memory(device)
        frozen_after = frozen_state_digest(actor)
        frozen_unchanged = frozen_before == frozen_after
        require(frozen_unchanged, "G4P_FROZEN_HASH_CHANGED")
        require(
            all(not actor.model.vlm_with_expert.vlm.training for _ in (0,))
            and not actor.model.state_proj.training,
            "G4P_FROZEN_MODULE_MODE_DRIFT",
        )
        all_finite = _module_all_finite(actor, q1, q2, q1_target, q2_target)
        require(all_finite, "G4P_FINAL_MODEL_NONFINITE")
        require(
            all(parameter.grad is None for target in (q1_target, q2_target) for parameter in target.parameters()),
            "G4P_FINAL_TARGET_GRADIENT",
        )
        flow_report = flow_counter.report()
        require(
            set(flow_report["policy_action_chunks_by_purpose"]) == {"actor_guidance", "td_next"},
            f"G4P_FLOW_PURPOSE:{flow_report}",
        )
        ownership["actor_optimizer_final_state_entries"] = len(actor_optimizer.state)
        ownership["critic_optimizer_final_state_entries"] = len(critic_optimizer.state)
        ownership["runtime_optimizer_state_persisted"] = False
        optimizer_steps = {
            "warmup_joint_cycles": 1,
            "measured_joint_cycles": 3,
            "critic_optimizer_steps": len(critic_reports),
            "actor_optimizer_steps": len(actor_reports),
            "target_polyak_steps": len(critic_reports),
            "target_q1_polyak_applications": len(critic_reports),
            "target_q2_polyak_applications": len(critic_reports),
        }
        numerics = {
            "all_finite": all_finite,
            "frozen_hash_unchanged": frozen_unchanged,
            "frozen_hash_before": frozen_before,
            "frozen_hash_after": frozen_after,
            "gradient_ownership_passed": True,
            "calql_online_call_count": counters["calql"],
            "cql_online_call_count": counters["cql"],
            "random_candidate_online_call_count": counters["random_candidates"],
            "mc_return_online_call_count": counters["mc_return"],
            "terminal_probe": terminal_probe,
            "online_critic_is_pure_td": True,
            "actor_q_uses_min_twin_q": True,
            "autonomous_fm_contribution_exact_zero": all(
                report["autonomous_fm_gradient_max_abs"] == 0.0 for report in actor_reports
            ),
            "zero_expert_batches_graph_connected_finite": all(
                report["zero_expert_graph_connected_finite"] for report in actor_reports
            ),
            "tcp6_q_gradient_nonzero": all(report["tcp6_q_gradient_norm"] > 0 for report in actor_reports),
            "gripper_q_gradient_exact_zero": all(report["gripper_q_gradient_max_abs"] == 0 for report in actor_reports),
            "expert_gripper_fm_gradient_nonzero": all(report["expert_gripper_fm_gradient_norm"] > 0 for report in actor_reports),
            "critic_steps_actor_delta_exact_zero": all(report["actor_parameter_delta_exact_zero"] for report in critic_reports),
            "actor_steps_critic_delta_exact_zero": all(report["online_and_target_critic_parameter_delta_exact_zero"] for report in actor_reports),
            "target_gradient_count": 0,
            "nonfinite_count": 0,
        }
        report: dict[str, Any] = {
            "schema_version": "forcesmolvla_stage3_gpu_preflight_report.v1",
            "tool_status": "PASS",
            "preflight_only": True,
            "source_head": g0a["source_head"],
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "environment": environment,
            "parent_binding": {
                "path": str(binding_path),
                "sha256_before": binding_sha_before,
                "binding_id": binding["binding_id"],
                "binding_type": binding["binding_type"],
                "G0_FINAL_PARENT_BINDING": "BOUND_APPROVED_HYBRID",
            },
            "parent_load": parent_load,
            "data": data_evidence,
            "optimizer_ownership": ownership,
            "cycles": optimizer_steps,
            "critic_updates": critic_reports,
            "actor_updates": actor_reports,
            "flow": {
                **flow_report,
                "inference_subbatch": config["batching"]["flow_inference_subbatch"],
                "H": 50,
                "N": 10,
                "generator_initial_state_sha256": generator_initial_digests,
            },
            "numerics": numerics,
            "performance": {
                "load_only_vram_mib": load_only["allocated_mib"],
                "load_only": load_only,
                "warmup_peak_vram_mib": warmup_peak["peak_allocated_mib"],
                "warmup_peak": warmup_peak,
                "measured_peak_vram_mib": measured_peak["peak_allocated_mib"],
                "measured_peak": measured_peak,
                "final_vram_mib": -1.0,
                "peak_cpu_rss_mib": _peak_cpu_rss_mib(),
                "oom_count": 0,
                "nonfinite_count": 0,
                "cycles": cycle_performance,
                "cuda_synchronization_points": sync_points,
            },
            "safety": {
                "CRITIC_READY": False,
                "ACTOR_Q_GUIDANCE_ENABLED": False,
                "G0_FORMAL_GATE_PASSED": False,
                "G3_RECORDED_FIXTURE_LOOPBACK": "BLOCKED",
                "G5_AND_LATER": "NOT_RUN",
                "ROBOT_CONNECTION_COUNT": 0,
                "ROBOT_COMMAND_COUNT": 0,
            },
            "parent_checkpoint_mutated": False,
            "runtime_optimizer_state_persisted": False,
            "policy_revision_exported": False,
            "robot_execution_authorized": False,
            "production_capabilities_not_verified": [
                "Critic formal warmup or stability gate",
                "runtime optimizer persistence or exact resume",
                "recorded-live G3 fixture loopback",
                "shadow learner, online training, policy export or activation",
                "GPU coexistence, ROS, networking or robot execution",
            ],
        }
    finally:
        critic_batch = actor_batch = None
        actor_optimizer = critic_optimizer = None
        actor = q1 = q2 = q1_target = q2_target = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            sync_points.append("release:end")

    final_memory = _cuda_memory(device)
    report["performance"]["final_vram_mib"] = final_memory["allocated_mib"]
    report["performance"]["final"] = final_memory
    report["performance"]["peak_cpu_rss_mib"] = _peak_cpu_rss_mib()
    parent_after = _hash_parent_records(parent_records)
    binding_sha_after = sha256_file(binding_path)
    parent_equal = parent_before == parent_after and binding_sha_before == binding_sha_after
    require(parent_equal, "G4P_PARENT_MUTATED")
    report["parent_binding"]["sha256_after"] = binding_sha_after
    report["parent_binding"]["sha_before_after_equal"] = binding_sha_before == binding_sha_after
    report["parent_load"]["artifacts_before"] = parent_before
    report["parent_load"]["artifacts_after"] = parent_after
    report["parent_load"]["sha_before_after_equal"] = parent_equal
    report["parent_checkpoint_mutated"] = not parent_equal
    freeze_g4p_evidence(report, config)
    report["canonical_report_sha256"] = canonical_report_sha256(report)
    return validate_report(report, _resolve(config["output"]["schema"]))


def render_markdown(report: Mapping[str, Any]) -> str:
    performance = report["performance"]
    numerics = report["numerics"]
    ownership = report["optimizer_ownership"]
    evidence = report["evidence_freeze"]
    eta_diagnostic = report["eta_gradient_diagnostic"]
    lines = [
        "# Stage-3 G4P isolated GPU numerical preflight v1",
        "",
        "This is a disposable numerical preflight on the approved-hybrid parent. It is not Critic warmup, online training, a formal Stage-3 gate, policy publication, or robot authorization.",
        "",
        "## Result and immutable boundary",
        "",
        f"- `tool_status={report['tool_status']}`",
        "- `preflight_only=true`",
        "- `parent_checkpoint_mutated=false`",
        "- `runtime_optimizer_state_persisted=false`",
        "- `policy_revision_exported=false`",
        "- `robot_execution_authorized=false`",
        "- `CRITIC_READY=false`",
        "- `ACTOR_Q_GUIDANCE_ENABLED=false`",
        "- `CRITIC_WARMUP_STARTED=false`",
        "- `G5_AND_LATER=NOT_RUN`",
        "- `GPU_COEXISTENCE_VALIDATED=false`",
        "",
        "## Environment",
        "",
        f"- CUDA device `{report['environment']['physical_cuda_device_index']}` / visible `{report['environment']['cuda_device_index']}`: `{report['environment']['gpu_name']}` (`{report['environment']['gpu_uuid']}`).",
        f"- Python `{report['environment']['python_executable']}`; PyTorch `{report['environment']['torch_version']}`, CUDA `{report['environment']['torch_cuda_version']}`, cuDNN `{report['environment']['cudnn_version']}`.",
        f"- Initial free VRAM `{report['environment']['initial_free_vram_mib']:.1f}` MiB.",
        "",
        "## Parent load",
        "",
        f"- Actor strict load: `{report['parent_load']['actor_strict_load']}` with {report['parent_load']['actor_tensor_count']} tensors, 0 missing, 0 unexpected, 0 shape mismatches.",
        "- Online Q1/Q2 and stored target Q1/Q2 were loaded with CPU `weights_only=True` and strict keys/shapes/dtypes before moving to GPU.",
        "- G7A-r5, random Critic fallback, and target-from-online fallback were not used.",
        "- Binding and every selected parent artifact SHA are identical before and after.",
        "",
        "## Data and cycles",
        "",
        "- Critic C64 is exactly 32 `synthetic_preflight_R_only` + 32 real offline D rows. Actor B24 is exactly 12 + 12; all 88 underlying real observation rows are non-overlapping.",
        "- The R label is numerical-preflight-only. Images/state/wrench/action/reward/terminal values still come through the frozen real Phase-2 train pipeline and are not online evidence.",
        "- H=50, N=10, K=3; every Actor flow inference is subbatched at 4.",
        f"- Updates: {report['cycles']['critic_optimizer_steps']} Critic, {report['cycles']['actor_optimizer_steps']} Actor, {report['cycles']['target_polyak_steps']} paired target Polyak applications.",
        "",
        "## Optimizer and gradient ownership",
        "",
        f"- Fresh initial optimizer state entries: `{ownership['fresh_initial_state_entries']}`; Actor/Critic ID intersection: `{ownership['actor_critic_parameter_id_intersection']}`.",
        f"- Frozen parameters in optimizers: `{ownership['frozen_parameters_in_optimizers']}`; target parameters in optimizers: `{ownership['target_parameters_in_optimizers']}`.",
        "- `apply_frozen_vlm_trainability()` was called. Vision/SmolVLM/language embeddings and state-prefix projection stayed frozen/eval/detached; Force/Action modules stayed trainable.",
        "",
        "## Numerical evidence",
        "",
        f"- All finite: `{numerics['all_finite']}`; frozen hashes unchanged: `{numerics['frozen_hash_unchanged']}`.",
        f"- Cal-QL/CQL/random-candidate/MC-return online calls: `{numerics['calql_online_call_count']}/{numerics['cql_online_call_count']}/{numerics['random_candidate_online_call_count']}/{numerics['mc_return_online_call_count']}`.",
        f"- Terminal probe next Actor/target-Q calls: `{numerics['terminal_probe']['next_actor_calls']}/{numerics['terminal_probe']['target_q1_calls']}/{numerics['terminal_probe']['target_q2_calls']}`.",
        "- Critic is pure TD with stored target Twin-Q min. Actor uses expert-only FM plus current min Twin-Q; autonomous FM and gripper Q-gradient are exactly zero, TCP6 Q-gradient and expert gripper FM-gradient are nonzero.",
        "- Each Actor step also records a fixed 4-row/fixed-noise before/after TCP6 drift and binary gripper pattern-change probe.",
        "",
        "## Evidence-freeze and eta diagnostic",
        "",
        f"- `G4P_RESULT={evidence['G4P_RESULT']}`; `R_SOURCE={evidence['R_SOURCE']}`; `REAL_ONLINE_R_USED=false`.",
        "- The four Actor optimizer steps changed only the disposable preflight instance: `PREFLIGHT_ACTOR_STEPS_DISPOSABLE=true` and `PRODUCTION_ACTOR_STATE_MUTATED=false`.",
        "- `RUNTIME_OPTIMIZER_STATE_PERSISTED=false`; `ETA_3_APPROVED=false`; no Critic warmup was started.",
        "- Weighted Q/FM ratios below are recomputed independently from each cycle's stored weighted gradient norms. The eta=3 column is only a linear rescaling diagnostic.",
        "",
        "| Cycle | Weighted Q norm | Weighted FM norm | Q / FM | Linear eta=3 diagnostic |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {item['cycle']} | {item['weighted_q_norm']:.12g} | {item['weighted_fm_norm']:.12g} | {item['weighted_q_over_weighted_fm']:.12g} | {item['eta_3_linear_rescale_q_over_fm']:.12g} |"
        for item in eta_diagnostic["per_cycle"]
    )
    lines.extend([
        "",
        eta_diagnostic["statements"][0],
        eta_diagnostic["statements"][1],
        "",
        "| Cycle | Kind | Wall s | Peak allocated MiB | Cycles/hour |",
        "|---:|---|---:|---:|---:|",
    ])
    lines.extend(
        f"| {item['cycle']} | {item['kind']} | {item['wall_seconds']:.3f} | {item['peak_allocated_mib']:.1f} | {item['cycles_per_hour_estimate']:.2f} |"
        for item in performance["cycles"]
    )
    lines.extend([
        "",
        f"Load-only allocated VRAM: `{performance['load_only_vram_mib']:.1f}` MiB; warm-up peak: `{performance['warmup_peak_vram_mib']:.1f}` MiB; measured peak: `{performance['measured_peak_vram_mib']:.1f}` MiB; post-release allocated: `{performance['final_vram_mib']:.1f}` MiB; peak CPU RSS: `{performance['peak_cpu_rss_mib']:.1f}` MiB.",
        "",
        "## Deferred production capability",
        "",
    ])
    lines.extend(f"- {item}" for item in report["production_capabilities_not_verified"])
    lines.extend([
        "",
        "```text",
        f"canonical_report_sha256={report['canonical_report_sha256']}",
        "G0_FORMAL_GATE_PASSED=false",
        "G3_RECORDED_FIXTURE_LOOPBACK=BLOCKED",
        "G5_AND_LATER=NOT_RUN",
        "ROBOT_CONNECTION_COUNT=0",
        "ROBOT_COMMAND_COUNT=0",
        "ROBOT_EXECUTION_AUTHORIZED=false",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_gpu_preflight(args.config)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config = validate_gpu_preflight_config(_load_mapping(args.config))
    markdown = _resolve(config["output"]["markdown_report"])
    _atomic_json(output, report)
    _atomic_text(markdown, render_markdown(report))
    print(json.dumps({
        "tool_status": report["tool_status"],
        "canonical_report_sha256": report["canonical_report_sha256"],
        "output": str(output),
        "markdown": str(markdown),
        "critic_optimizer_steps": report["cycles"]["critic_optimizer_steps"],
        "actor_optimizer_steps": report["cycles"]["actor_optimizer_steps"],
        "target_polyak_steps": report["cycles"]["target_polyak_steps"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
