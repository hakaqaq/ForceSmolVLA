#!/usr/bin/env python3
"""Real RTX4090D P6 Dense-Param/MoE structure and budget gate."""

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
import xml.etree.ElementTree as ET

from preflight_p5_dense_compute_gpu import (
    _gradient_summary,
    _require_offline,
    _sha256,
    _validate_p4_prerequisite,
    _validate_static_spec,
    _validate_source_binding,
)


def _validate_spec(spec: dict) -> None:
    if (
        spec.get("acceptance_status") != "development_only"
        or spec.get("formal_eligible") is not False
        or spec.get("training_stage") != "offline_full_finetune"
        or spec.get("seed") != 42
    ):
        raise RuntimeError("P6_STATIC_SPEC_STATUS_DRIFT")
    expected_common = {
        "p5_static_spec_sha256": (
            "7ab3307874c4b27e723b27972b4fee2d80629ecd60da60a9e463b13d380d9405"
        ),
        "D_vlm": 960,
        "D_expert": 720,
        "fusion_blocks": 2,
        "fusion_heads": 8,
        "force_cross_attention_heads": 1,
        "N_fused_physical": 177,
    }
    if spec.get("common_architecture_binding") != expected_common:
        raise RuntimeError("P6_COMMON_ARCHITECTURE_BINDING_DRIFT")
    dense = spec["dense_param"]
    moe = spec["moe"]
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
    if spec.get("p6_phase_boundary") != {
        "router_auxiliary_loss": "implemented_not_accepted_until_P7_gate",
        "exact_two_pass_oracle": "implemented_not_accepted_until_P7_gate",
        "additive_adapter": "implemented_not_accepted_until_P7_gate",
        "checkpoint_strict_reload": "implemented_not_accepted_until_P8_gate",
        "shadow": "implemented_not_accepted_until_P9_gate",
    }:
        raise RuntimeError("P6_PHASE_BOUNDARY_DRIFT")


def _validate_runtime_import_roots(root: Path) -> dict:
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
            raise RuntimeError(
                f"P6_{name.upper()}_IMPORT_ROOT_MISMATCH: {resolved}"
            )
        observed[name] = str(resolved)
    return observed


def _dataset_storage_binding(dataset_root: Path) -> dict:
    files = sorted(
        path
        for directory in ("data", "videos", "meta")
        for path in (dataset_root / directory).rglob("*")
        if path.is_file()
    )
    if not files:
        raise RuntimeError("P6_DATASET_STORAGE_TREE_EMPTY")
    hashes = {
        path.relative_to(dataset_root).as_posix(): _sha256(path) for path in files
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


def _pytest_evidence_summary(root: Path, evidence: dict) -> dict:
    if set(evidence) != {"format", "report", "selection", "test_files"}:
        raise RuntimeError("P6_PYTEST_EVIDENCE_SCHEMA_DRIFT")
    if evidence["format"] != "junit_xml":
        raise RuntimeError("P6_PYTEST_EVIDENCE_FORMAT_DRIFT")
    report = evidence["report"]
    if set(report) != {"path", "sha256"}:
        raise RuntimeError("P6_PYTEST_REPORT_BINDING_DRIFT")
    report_path = root / report["path"]
    if _sha256(report_path) != report["sha256"]:
        raise RuntimeError("P6_PYTEST_REPORT_HASH_MISMATCH")
    selection = evidence["selection"]
    if set(selection) != {"stage", "included_files", "excluded_downstream_prefixes"}:
        raise RuntimeError("P6_PYTEST_SELECTION_SCHEMA_DRIFT")
    included_files = selection["included_files"]
    if not isinstance(included_files, list) or included_files != sorted(evidence["test_files"]):
        raise RuntimeError("P6_PYTEST_SELECTION_FILE_SET_DRIFT")
    for relative, expected in evidence["test_files"].items():
        if _sha256(root / relative) != expected:
            raise RuntimeError(f"P6_TEST_SOURCE_HASH_MISMATCH: {relative}")
    tree = ET.parse(report_path)
    suites = [tree.getroot()] if tree.getroot().tag == "testsuite" else list(tree.getroot())
    summary = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if summary["tests"] <= 0 or summary["failures"] or summary["errors"]:
        raise RuntimeError(f"P6_PYTEST_REPORT_NOT_PASSING: {summary}")
    report_files = {
        testcase.attrib["classname"].replace(".", "/") + ".py"
        for suite in suites
        for testcase in suite.iter("testcase")
        if testcase.attrib.get("classname", "").startswith("tests.")
    }
    if report_files != set(included_files):
        raise RuntimeError(
            f"P6_PYTEST_REPORT_SELECTION_MISMATCH: report={sorted(report_files)} "
            f"bound={included_files}"
        )
    summary["report_sha256"] = report["sha256"]
    summary["test_file_count"] = len(evidence["test_files"])
    summary["selection_stage"] = selection["stage"]
    return summary


def _validate_p5_prerequisite(
    root: Path,
    spec: dict,
    *,
    dataset_root: Path,
    repo_id: str,
    binding: dict | None = None,
) -> dict:
    prerequisite = spec.get("p5_prerequisite")
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
        if _sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"P6_P5_{name.upper()}_HASH_MISMATCH")
        payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    parent_spec = payloads["static_spec"]
    parent_binding = payloads["source_binding"]
    parent_resolved = payloads["resolved_config"]
    parent_result = payloads["gate_result"]
    _validate_static_spec(parent_spec)
    _validate_p4_prerequisite(root, parent_spec, parent_binding)
    _validate_source_binding(
        root, parent_binding, dataset_root=dataset_root, repo_id=repo_id
    )
    if (
        parent_result.get("gate") != "P5"
        or parent_result.get("gate_status") != prerequisite["required_gate_status"]
        or parent_result.get("acceptance_status")
        != prerequisite["required_acceptance_status"]
        or parent_result.get("formal_eligible")
        is not prerequisite["required_formal_eligible"]
        or parent_result.get("source_binding_sha256")
        != prerequisite["source_binding"]["sha256"]
        or parent_result.get("resolved_config_sha256")
        != prerequisite["resolved_config"]["sha256"]
        or parent_result.get("static_spec_sha256")
        != prerequisite["static_spec"]["sha256"]
        or parent_result.get("real_batch", {}).get("repo_id") != repo_id
        or parent_result.get("real_batch", {}).get("batch_size") != 4
        or parent_resolved.get("source_binding_sha256")
        != prerequisite["source_binding"]["sha256"]
        or parent_resolved.get("static_spec_sha256")
        != prerequisite["static_spec"]["sha256"]
    ):
        raise RuntimeError("P6_PARENT_P5_GATE_NOT_ELIGIBLE")
    if binding is not None and binding.get("p5_prerequisite") != prerequisite:
        raise RuntimeError("P6_SOURCE_BINDING_P5_PREREQUISITE_MISMATCH")
    return {
        name: prerequisite[name]["sha256"]
        for name in ("static_spec", "source_binding", "resolved_config", "gate_result")
    }


def _validate_p6_source_binding(
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
    _validate_source_binding(
        root, binding, dataset_root=dataset_root, repo_id=repo_id
    )
    runtime_imports = _validate_runtime_import_roots(root)
    if binding.get("runtime_imports") != runtime_imports:
        raise RuntimeError("P6_SOURCE_BINDING_RUNTIME_IMPORT_MISMATCH")
    if binding["dataset"].get("storage_tree") != _dataset_storage_binding(dataset_root):
        raise RuntimeError("P6_DATASET_STORAGE_TREE_HASH_MISMATCH")
    parent = _validate_p5_prerequisite(
        root,
        spec,
        dataset_root=dataset_root,
        repo_id=repo_id,
        binding=binding,
    )
    tests = _pytest_evidence_summary(root, binding.get("test_evidence", {}))
    return {
        "p5_prerequisite": parent,
        "dataset_storage_tree_sha256": binding["dataset"]["storage_tree"][
            "tree_sha256"
        ],
        "dataset_storage_file_count": binding["dataset"]["storage_tree"][
            "file_count"
        ],
        "pytest": tests,
        "runtime_imports": runtime_imports,
    }


def _make_batch(policy, prepared: list[dict], device):
    import torch
    import numpy as np

    from forcesmolvla.configuration_forcesmolvla import CAMERA1, CAMERA2
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

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
    return {
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


def _common_initialization_sha256(policy) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        if not (
            name.startswith("model.force_adapter.")
            or name.startswith("model.force_branch.")
            and not name.startswith("model.force_branch.refiner.")
        ):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}\0{value.dtype}\0{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def _run_variant(*, root: Path, spec: dict, variant: str, prepared: list[dict], device) -> dict:
    import numpy as np
    import torch

    from forcesmolvla.checkpoint import load_offline_base_policy
    from forcesmolvla.configuration_forcesmolvla import OFFLINE_FULL_FINETUNE
    from forcesmolvla.router_training import _no_decay_parameter_names

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with contextlib.redirect_stdout(sys.stderr):
        policy, base_report = load_offline_base_policy(
            root / "assets/base_checkpoint",
            root / "assets/smolvlm_constructor",
            device="cuda",
            training_stage=OFFLINE_FULL_FINETUNE,
            force_variant=variant,
            acceptance_status="development_only",
            force_init_seed=42,
        )
    section = spec["dense_param" if variant == "force_token_dense_param" else "moe"]
    named = list(policy.named_parameters())
    frozen = [name for name, parameter in named if not parameter.requires_grad]
    total_parameters = sum(parameter.numel() for _, parameter in named)
    force_parameters = sum(
        parameter.numel()
        for name, parameter in named
        if name.startswith(("model.force_branch.", "model.force_adapter."))
    )
    if frozen or total_parameters != section["total_parameters"] or force_parameters != section["force_parameters"]:
        raise RuntimeError(
            f"P6_PARAMETER_BUDGET_MISMATCH: variant={variant}, frozen={frozen}, "
            f"total={total_parameters}, force={force_parameters}"
        )
    if tuple(base_report.missing_keys) != policy.force_initialization_state_keys():
        raise RuntimeError("P6_BASE_LOAD_MISSING_KEY_ALLOWLIST_DRIFT")
    common_initialization_tensor_sha256 = _common_initialization_sha256(policy)

    batch = _make_batch(policy, prepared, device)
    generator = torch.Generator(device=device).manual_seed(43)
    batch_size = spec["execution"]["batch_size"]
    noise = torch.randn(
        batch_size, 50, 7, generator=generator, device=device, dtype=torch.float32
    )
    timestep = torch.tensor([0.2, 0.4, 0.6, 0.8], device=device, dtype=torch.float32)
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    named_trainable = dict(trainable)
    no_decay_names = _no_decay_parameter_names(policy)
    decay_names = set(named_trainable) - no_decay_names
    if decay_names & no_decay_names or decay_names | no_decay_names != set(named_trainable):
        raise RuntimeError("P6_OPTIMIZER_GROUP_PARTITION_INVALID")
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
        raise RuntimeError("P6_OPTIMIZER_PARAMETER_PARTITION_INVALID")

    observed_routes = []

    def capture_router(_module, _inputs, output):
        if output.router_state is None:
            return
        routes = output.router_state.route_ids.detach()
        valid = output.router_state.valid_mask.detach()
        valid_token_count = int(valid.sum().item())
        counts = torch.bincount(routes[valid], minlength=4).cpu().tolist()
        if sum(counts) != valid_token_count:
            raise RuntimeError("P6_MOE_TOKEN_DROP_DETECTED")
        observed_routes.append(
            {"counts": counts, "valid_token_count": valid_token_count, "dropped_tokens": 0}
        )

    hook = policy.model.force_branch.register_forward_hook(capture_router)
    policy.train()
    if not policy.model.vlm_with_expert.vlm.training:
        raise RuntimeError("P6_VLM_NOT_IN_TRAIN_MODE")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    steps = []
    wall_start = time.perf_counter()
    for step in range(2):
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
            raise FloatingPointError("P6_NONFINITE_LOSS")
        loss.backward()
        after_backward.record()
        wout_grad = policy.model.force_adapter.w_out.weight.grad
        if wout_grad is None or not torch.count_nonzero(wout_grad):
            raise RuntimeError("P6_W_OUT_GRADIENT_MISSING_OR_ZERO")
        coverage = _gradient_summary(trainable)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], max_norm=10.0
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("P6_NONFINITE_GRADIENT_NORM")
        optimizer.step()
        after_step.record()
        torch.cuda.synchronize(device)
        steps.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                "forward_ms": start.elapsed_time(after_forward),
                "backward_ms": after_forward.elapsed_time(after_backward),
                "optimizer_step_ms": after_backward.elapsed_time(after_step),
                "gradient_coverage": coverage,
            }
        )
    hook.remove()
    if variant == "force_token_dense_param":
        if steps[-1]["gradient_coverage"]["force"]["coverage_nonzero_gradient"] != 1.0:
            raise RuntimeError("P6_DENSE_PARAM_FORCE_GRADIENT_COVERAGE_INCOMPLETE")
        active_expert_ids = []
        inactive_expert_ids = []
    else:
        if not observed_routes or any(
            sum(item["counts"]) != item["valid_token_count"] for item in observed_routes
        ):
            raise RuntimeError("P6_MOE_ROUTE_ACCOUNTING_FAILED")
        router_grads = [
            parameter.grad
            for name, parameter in trainable
            if name.startswith("model.force_branch.refiner.router.")
        ]
        if len(router_grads) != 2 or any(grad is None or not torch.count_nonzero(grad) for grad in router_grads):
            raise RuntimeError("P6_MOE_ROUTER_GRADIENT_MISSING_OR_ZERO")
        active_expert_ids = sorted(
            {
                expert_id
                for item in observed_routes[-1:]
                for expert_id, count in enumerate(item["counts"])
                if count
            }
        )
        inactive_expert_ids = sorted(set(range(4)) - set(active_expert_ids))
        for name, parameter in trainable:
            if name.startswith(("model.force_branch.", "model.force_adapter.")) and not name.startswith(
                "model.force_branch.refiner.experts."
            ) and (parameter.grad is None or not torch.count_nonzero(parameter.grad)):
                raise RuntimeError(f"P6_MOE_COMMON_FORCE_GRADIENT_MISSING_OR_ZERO:{name}")
            if not name.startswith("model.force_branch.refiner.experts."):
                continue
            expert_id = int(name.split(".")[4])
            if expert_id in active_expert_ids and (
                parameter.grad is None or not torch.count_nonzero(parameter.grad)
            ):
                raise RuntimeError(f"P6_ACTIVE_EXPERT_GRADIENT_MISSING_OR_ZERO:{name}")
    final_coverage = steps[-1]["gradient_coverage"]
    for group_name in ("vision", "vlm_text", "action_expert", "action_io"):
        if final_coverage[group_name]["coverage_nonzero_gradient"] != 1.0:
            raise RuntimeError(f"P6_BASE_GRADIENT_COVERAGE_INCOMPLETE:{group_name}")
    if (
        final_coverage["base"]["missing_gradient_names"]
        != ["model.vlm_with_expert.vlm.lm_head.weight"]
        or final_coverage["base"]["zero_gradient_names"]
    ):
        raise RuntimeError("P6_BASE_GRADIENT_EXCEPTION_SET_DRIFT")

    result = {
        "variant": variant,
        "initialization_tensor_sha256": policy.force_initialization_tensor_hash(),
        "common_initialization_tensor_sha256": common_initialization_tensor_sha256,
        "total_parameters": total_parameters,
        "trainable_parameters": total_parameters,
        "frozen_parameters": 0,
        "precision": {
            "mode": "full_parameter_training_with_bf16_autocast_mixed_precision",
            "parameter_storage_dtypes": sorted(
                {str(parameter.dtype) for _, parameter in trainable}
            ),
            "optimizer_state_dtypes": sorted(
                {
                    str(value.dtype)
                    for state in optimizer.state.values()
                    for value in state.values()
                    if hasattr(value, "dtype")
                }
            ),
        },
        "optimizer_parameter_partition": {
            "trainable_tensor_count": len(trainable),
            "optimizer_tensor_count": len(optimizer_parameters),
            "each_trainable_parameter_exactly_once": True,
            "decay_tensor_count": len(decay_names),
            "no_decay_tensor_count": len(no_decay_names),
        },
        "force_parameters": force_parameters,
        "base_load_missing_tensor_count": len(base_report.missing_keys),
        "steps": steps,
        "route_counts_per_step": observed_routes,
        "active_expert_ids_step2": active_expert_ids,
        "inactive_expert_ids_step2": inactive_expert_ids,
        "peak_memory": {
            "allocated_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "wall_seconds": time.perf_counter() - wall_start,
        "memory_measurement_scope": "two_step_P6_variant_preflight_only_not_long_run",
    }
    del optimizer, policy, batch, noise, timestep
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-output", type=Path, required=True)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).parents[1] / "configs/p6_dense_param_moe.development.json",
    )
    parser.add_argument(
        "--source-binding",
        type=Path,
        default=Path(__file__).parents[1] / "artifacts/development/p6_source_binding.json",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.resolved_output.exists():
        raise FileExistsError("refusing to overwrite a P6 gate artifact")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("PYTHONHASHSEED=42 required before interpreter startup")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _require_offline()

    import numpy as np
    import torch

    from forcesmolvla.dataset_v3 import load_dataset_split
    from forcesmolvla.training_data import load_runtime_artifacts, prepare_training_sample

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")
    gpu_name = torch.cuda.get_device_name(0)
    if "4090 D" not in gpu_name and "4090D" not in gpu_name:
        raise RuntimeError(f"P6_REQUIRES_RTX_4090D: got {gpu_name!r}")
    root = Path(__file__).parents[1].resolve()
    runtime_imports = _validate_runtime_import_roots(root)
    spec_path = args.spec.resolve()
    binding_path = args.source_binding.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    _validate_spec(spec)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset_root = args.dataset_root.resolve()
    conversion_manifest = json.loads(
        (dataset_root / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    repo_id = conversion_manifest.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("P6_CONVERSION_REPO_ID_MISSING")
    source_evidence = _validate_p6_source_binding(
        root,
        binding,
        dataset_root=dataset_root,
        repo_id=repo_id,
        spec=spec,
    )
    dataset = load_dataset_split(
        dataset_root,
        repo_id=repo_id,
        split_name="train",
        artifact_use="development",
        delta_timestamps={"action": [index / 30 for index in range(50)]},
    )
    batch_size = spec["execution"]["batch_size"]
    sample_indices = list(range(args.sample_index, args.sample_index + batch_size))
    if args.sample_index < 0 or sample_indices[-1] >= len(dataset):
        raise IndexError("P6 requires four consecutive samples inside the train split")
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
    variants = [
        _run_variant(
            root=root,
            spec=spec,
            variant=variant,
            prepared=prepared,
            device=device,
        )
        for variant in ("force_token_dense_param", "force_token_moe")
    ]
    if len({item["common_initialization_tensor_sha256"] for item in variants}) != 1:
        raise RuntimeError("P6_DENSE_MOE_COMMON_INITIALIZATION_MISMATCH")
    zero_init_loss_abs_diff = abs(
        variants[0]["steps"][0]["loss"] - variants[1]["steps"][0]["loss"]
    )
    if zero_init_loss_abs_diff != 0.0:
        raise RuntimeError(
            f"P6_ZERO_INIT_DENSE_MOE_FLOW_LOSS_MISMATCH:{zero_init_loss_abs_diff}"
        )
    binding_sha256 = _sha256(binding_path)
    spec_sha256 = _sha256(spec_path)
    resolved = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "training_stage": "offline_full_finetune",
        "seed": 42,
        "source_binding_sha256": binding_sha256,
        "static_spec_sha256": spec_sha256,
        "variants": variants,
        "runtime_imports": runtime_imports,
        "source_evidence": source_evidence,
        "zero_init_dense_moe_flow_loss_abs_diff": zero_init_loss_abs_diff,
        "p6_phase_boundary": spec["p6_phase_boundary"],
        "detached_signature": None,
        "approval": None,
    }
    args.resolved_output.parent.mkdir(parents=True, exist_ok=True)
    args.resolved_output.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "gate": "P6",
        "gate_status": "pass",
        "gpu": {
            "name": gpu_name,
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
        },
        "cpu_fallback_used": False,
        "architecture_downgrade_used": False,
        "source_binding_sha256": binding_sha256,
        "static_spec_sha256": spec_sha256,
        "resolved_config_sha256": _sha256(args.resolved_output),
        "variants": variants,
        "runtime_imports": runtime_imports,
        "source_evidence": source_evidence,
        "zero_init_dense_moe_flow_loss_abs_diff": zero_init_loss_abs_diff,
        "p7_started": False,
        "remaining_blockers": [
            "P7 v4.2 router/additive/oracle gate not revalidated",
            "P8 v4.2 strict checkpoint reload not executed",
            "trusted detached signature/approvals unresolved"
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
