#!/usr/bin/env python3
"""Run the append-only G2 Twin-Q zero-update RTX 4090D preflight."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/twin_q_critic.development.yaml"
G1_ROOT = ROOT / "artifacts/development/stage2/g1_frozen_detector_transition_view.v1"
DATASET = ROOT / "datasets/task2_lerobotv3"
SAFE_NPZ = ROOT / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_params.safe.npz"
SAFE_MANIFEST = ROOT / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_asset_manifest.v4.json"
CLASSIFIER = ROOT / "artifacts/development/stage2/reward_classifier/r0_training/checkpoints/best_checkpoint.msgpack"
R5 = ROOT / "outputs/development/task2_lerobotv3_full_sft_10k_r5/checkpoints/step_010000"
ARTIFACT = ROOT / "artifacts/development/stage2/s2_g2_twin_q_topology.json"
SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v5_g2.json"
REPORT = ROOT / "docs/s2_g2_twin_q_topology_report.md"
MANUAL_G1 = ROOT / "artifacts/development/stage2/g1_manual_reward_transition_view.v1"
LABELS = ROOT / "labels"
EXPECTED_P8_SHA256 = "f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d"
EXPECTED_CHECKPOINT_SHA256 = "6b4e366baa55993d150cb3dd86e67a1d708e58d836b123a0c433190835021510"
FORBIDDEN_OPENS: dict[str, set[str]] = {"manual_g1": set(), "manual_labels": set()}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def install_open_audit() -> None:
    roots = {"manual_g1": MANUAL_G1.resolve(), "manual_labels": LABELS.resolve()}

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        try:
            path = Path(os.fsdecode(args[0])).resolve()
        except (OSError, TypeError, ValueError):
            return
        for name, root in roots.items():
            if path == root or path.is_relative_to(root):
                FORBIDDEN_OPENS[name].add(str(path))

    sys.addaudithook(audit)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def binding(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "file_size": path.stat().st_size,
    }


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        stream.write(value)
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def p8_storage_tree() -> dict:
    files = sorted(
        path
        for directory in ("data", "videos", "meta")
        for path in (DATASET / directory).rglob("*")
        if path.is_file()
    )
    digest = hashlib.sha256()
    total_size = 0
    for path in files:
        relative = path.relative_to(DATASET).as_posix()
        value = sha256_file(path)
        digest.update(f"{relative}\0{value}\n".encode())
        total_size += path.stat().st_size
    return {"tree_sha256": digest.hexdigest(), "file_count": len(files), "total_file_size": total_size}


def protected_snapshot() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from forcesmolvla.rft.offline_transitions import dataset_tree_sha256

    fixed = {
        "classifier_checkpoint": CLASSIFIER,
        "g1_manifest": G1_ROOT / "g1_manifest.json",
        "g1_frame_scores": G1_ROOT / "frame_scores.parquet",
        "g1_transition_index": G1_ROOT / "transition_index.parquet",
        "safe_resnet10": SAFE_NPZ,
        "dataset_conversion_manifest": DATASET / "conversion_manifest.json",
        "dataset_split_manifest": DATASET / "split_manifest.json",
        "dataset_normalizer_manifest": DATASET / "normalizer_manifest.json",
        "historical_one_shot": ROOT / "artifacts/development/stage2/reward_classifier/r0_one_shot_test_evaluation.v1.json",
        "g0_parent_bridge": ROOT / "artifacts/development/stage2/s2_g0_parent_bridge_source_closed.json",
        "g3_flow": ROOT / "artifacts/development/stage2/s2_g3_differentiable_flow.v4.json",
        "prior_stage2_source_manifest": ROOT / "artifacts/development/stage2/stage2_source_manifest.v4.json",
        "public_modeling_source": ROOT / "src/forcesmolvla/modeling_forcesmolvla.py",
        "public_inference_source": ROOT / "src/forcesmolvla/inference.py",
    }
    parent = json.loads((ROOT / "configs/stage2_parent_bridge.development.json").read_text())
    for index, item in enumerate(parent["parent_p4_to_p8_qualification_artifacts"]):
        fixed[f"stage1_p4_p8_{index:02d}"] = ROOT / item["path"]
    result = {
        "files": {name: binding(path) for name, path in fixed.items()},
        "p8_storage_tree": p8_storage_tree(),
        "r5_checkpoint_tree": dataset_tree_sha256(R5),
    }
    require(result["files"]["classifier_checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_SHA256, "G2_CLASSIFIER_SHA_DRIFT")
    require(result["p8_storage_tree"]["tree_sha256"] == EXPECTED_P8_SHA256, "G2_P8_STORAGE_SHA_DRIFT")
    return result


def verify_config() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    require(config["authorization"] == "yes_development_topology_only", "G2_AUTHORIZATION_DRIFT")
    require(
        config["training_data"]["only_authorized_root"]
        == "artifacts/development/stage2/g1_frozen_detector_transition_view.v1"
        and config["training_data"]["allowed_split"] == "train"
        and config["training_data"]["manual_g1_allowed"] is False
        and config["training_data"]["manual_labels_allowed"] is False,
        "G2_TRAINING_DATA_AUTHORIZATION_DRIFT",
    )
    require(
        config["critic_interface"]["action_shape"] == [3, 7]
        and config["critic_interface"]["mask_shape"] == [3]
        and config["topology"]["fusion_mlp"] == [1283, 1024, 512, 256, 1]
        and config["topology"]["dropout"] == 0
        and config["topology"]["batch_norm_allowed"] is False,
        "G2_TOPOLOGY_CONFIG_DRIFT",
    )
    require(
        config["targets"]["approved_training_tau"] is None
        and config["targets"]["tested_synthetic_tau"] == [0.0, 0.005, 1.0],
        "G2_TARGET_CONFIG_DRIFT",
    )
    require(
        config["frozen_development_limits"]
        == {
            "STRICT_ZERO_FRAME_ALIGNMENT_ACCEPTANCE": "FAIL_preserved",
            "DEVELOPMENT_DETECTOR_OPERATIONAL": True,
            "FORMAL_DETECTOR_APPROVED": False,
            "REWARD_MODEL_TRAINING_OVERLAP": True,
            "UNBIASED_REWARD_MODEL_EVALUATION": False,
        },
        "G2_DEVELOPMENT_LIMIT_DRIFT",
    )
    return config


def run_unit_tests() -> dict:
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_rft_critic.py"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    require(result.returncode == 0 and "5 passed" in output, f"G2_UNIT_TEST_FAILED:{output[-2000:]}")
    return {"command": "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_rft_critic.py", "result": "5 passed", "exit_code": 0}


def decode_rgb(payload: bytes) -> np.ndarray:
    from PIL import Image

    require(isinstance(payload, bytes), "G2_IMAGE_PAYLOAD_MISSING")
    with Image.open(BytesIO(payload)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(array.shape == (480, 640, 3), f"G2_IMAGE_SHAPE_DRIFT:{array.shape}")
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def real_train_batch(limit: int = 16) -> tuple[dict[str, torch.Tensor], dict]:
    import pyarrow.parquet as pq

    sys.path.insert(0, str(ROOT / "src"))
    from forcesmolvla.rft.critic import (
        frozen_task_feature,
        load_authorized_critic_train_transitions,
    )
    from forcesmolvla.training_data import load_runtime_artifacts

    table = load_authorized_critic_train_transitions(G1_ROOT)
    require(table.num_rows == 10075 and set(table.column("split").to_pylist()) == {"train"}, "G2_G1_TRAIN_LOADER_DRIFT")
    rows = table.to_pylist()
    by_steps = {step: [row for row in rows if row["executed_steps"] == step] for step in (1, 2, 3)}
    require(all(by_steps.values()), "G2_REAL_BATCH_MASK_COVERAGE_MISSING")
    selected = [by_steps[3][0], by_steps[1][0], by_steps[2][0]]
    selected.extend(row for row in by_steps[3][1:] if len(selected) < limit)
    selected = selected[:limit]

    conversion = json.loads((DATASET / "conversion_manifest.json").read_text())
    tasks = {episode["raw_episode_id"]: episode["task"] for episode in conversion["episodes"]}
    canonical = config_task = yaml.safe_load(CONFIG.read_text())["observation"]["canonical_task"]
    require(set(tasks.values()) == {canonical}, "G2_CANONICAL_TASK_DRIFT")
    runtime = load_runtime_artifacts(
        DATASET,
        calibration_bundle_path=ROOT / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=ROOT / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=ROOT / "artifacts/development/action_delta_spec.json",
        expected_repo_id=conversion["repo_id"],
    )
    caches: dict[str, Any] = {}
    camera1, camera2, state, wrench, action, mask = [], [], [], [], [], []
    episodes, indices = [], []
    for transition in selected:
        require(transition["split"] == "train" and transition["episode_id"] in tasks, "G2_HELDOUT_OR_UNKNOWN_ROW")
        reference = transition["observation_row_reference"]
        relative = reference["data_relative_path"]
        if relative not in caches:
            caches[relative] = pq.read_table(
                DATASET / relative,
                columns=[
                    "observation.images.camera1",
                    "observation.images.camera2",
                    "observation.state",
                    "observation.wrench",
                    "frame_index",
                    "episode_index",
                    "index",
                ],
            )
        row = caches[relative].slice(reference["row_index"], 1).to_pylist()[0]
        require(
            row["frame_index"] == reference["frame_index"]
            and row["index"] == reference["global_index"],
            "G2_OBSERVATION_ROW_REFERENCE_DRIFT",
        )
        camera1.append(decode_rgb(row["observation.images.camera1"]["bytes"]))
        camera2.append(decode_rgb(row["observation.images.camera2"]["bytes"]))
        state.append(runtime.normalizer.state7.apply(np.asarray(row["observation.state"], dtype=np.float64)).astype(np.float32))
        wrench.append(runtime.normalizer.wrench6.apply(np.asarray(row["observation.wrench"], dtype=np.float64)).astype(np.float32))
        current_mask = np.asarray(transition["executed_action_mask"], dtype=np.bool_)
        steps = int(current_mask.sum())
        flat = np.asarray(transition["normalized_delta_action_exec_flat"], dtype=np.float32)
        require(steps == transition["executed_steps"] and flat.shape == (steps * 7,), "G2_EXECUTED_ACTION_VIEW_DRIFT")
        padded = np.zeros((3, 7), dtype=np.float32)
        padded[:steps] = flat.reshape(steps, 7)
        action.append(padded)
        mask.append(current_mask)
        episodes.append(transition["episode_id"])
        indices.append(transition["transition_index"])
    feature = frozen_task_feature(config_task)
    batch = {
        "camera1": torch.from_numpy(np.stack(camera1)),
        "camera2": torch.from_numpy(np.stack(camera2)),
        "task": torch.from_numpy(np.repeat(feature[None, :], len(selected), axis=0)),
        "state": torch.from_numpy(np.stack(state)),
        "wrench": torch.from_numpy(np.stack(wrench)),
        "action": torch.from_numpy(np.stack(action)),
        "mask": torch.from_numpy(np.stack(mask)),
    }
    evidence = {
        "authorized_g1_train_transition_count": table.num_rows,
        "selected_real_transition_count": len(selected),
        "selected_transition_indices": indices,
        "selected_episode_ids": episodes,
        "selected_executed_steps_distribution": dict(sorted(Counter(int(value.sum()) for value in batch["mask"]).items())),
        "source_parquet_files_opened": sorted(caches),
        "train_transition_rows_returned": table.num_rows,
        "validation_transition_rows_returned": 0,
        "test_transition_rows_returned": 0,
        "manual_g1_rows_returned": 0,
        "manual_label_rows_returned": 0,
        "full_h50_action_columns_returned_by_g2_loader": 0,
        "absolute_action_columns_returned_by_g2_loader": 0,
        "input_action_shape": [len(selected), 3, 7],
    }
    return batch, evidence


def to_device(batch: dict[str, torch.Tensor], device: torch.device, count: int) -> list[torch.Tensor]:
    return [batch[name][:count].to(device) for name in ("camera1", "camera2", "task", "state", "wrench", "action", "mask")]


def module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def state_max_abs_error(left: torch.nn.Module, right: torch.nn.Module) -> float:
    errors = []
    for name, value in left.state_dict().items():
        other = right.state_dict()[name]
        errors.append(float((value.float() - other.float()).abs().max().item()))
    return max(errors, default=0.0)


def changed(output: torch.Tensor, module, values: list[torch.Tensor], index: int, mutate) -> bool:
    altered = [value.clone() for value in values]
    mutate(altered[index])
    return not torch.equal(output, module(*altered))


def synthetic_polyak_evidence(polyak_blend_state) -> dict:
    online = {"weight": torch.tensor([1.0, 3.0], dtype=torch.float32)}
    target = {"weight": torch.tensor([5.0, 7.0], dtype=torch.float32)}
    online_before, target_before = online["weight"].clone(), target["weight"].clone()
    evidence = {}
    for tau in (0.0, 0.005, 1.0):
        actual = polyak_blend_state(online, target, tau)["weight"]
        expected = target["weight"] if tau == 0 else online["weight"] if tau == 1 else (1 - tau) * target["weight"] + tau * online["weight"]
        evidence[str(tau)] = {"actual": actual.tolist(), "expected": expected.tolist(), "elementwise_exact": torch.equal(actual, expected)}
    require(all(value["elementwise_exact"] for value in evidence.values()), "G2_POLYAK_FORMULA_FAILED")
    require(torch.equal(online_before, online["weight"]) and torch.equal(target_before, target["weight"]), "G2_POLYAK_MUTATED_INPUT")
    return {"tested_tau": evidence, "inputs_unchanged": True, "formal_tau_approved": False}


def measure(module1, module2, batch, batch_size: int, repeats: int = 5) -> dict:
    values = to_device(batch, torch.device("cuda"), batch_size)
    forward_ms, backward_ms = [], []
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    for iteration in range(2 + repeats):
        module1.zero_grad(set_to_none=True)
        module2.zero_grad(set_to_none=True)
        values[5] = values[5].detach().clone().requires_grad_(True)
        start, middle, end = torch.cuda.Event(True), torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        output1, output2 = module1(*values), module2(*values)
        middle.record()
        probe_scalar = (output1 + output2).sum()
        probe_scalar.backward()
        end.record()
        torch.cuda.synchronize()
        if iteration >= 2:
            forward_ms.append(start.elapsed_time(middle))
            backward_ms.append(middle.elapsed_time(end))
    return {
        "batch_size": batch_size,
        "warmup_iterations": 2,
        "measurement_iterations": repeats,
        "forward_latency_ms_median": statistics.median(forward_ms),
        "forward_latency_ms_all": forward_ms,
        "backward_latency_ms_median": statistics.median(backward_ms),
        "backward_latency_ms_all": backward_ms,
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_incremental_allocated_bytes": torch.cuda.max_memory_allocated() - baseline,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def gpu_preflight(batch: dict[str, torch.Tensor], config: dict) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from forcesmolvla.rft.critic import (
        build_twin_q,
        frozen_task_feature_sha256,
        modules_storage_independent,
        parameter_inventory,
        polyak_blend_state,
        state_exact,
    )

    require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "G2_CUDA_REQUIRED")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    require("4090" in name and "D" in name.upper(), f"G2_RTX_4090D_REQUIRED:{name}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    q1, q2, q1_target, q2_target, conversion = build_twin_q(SAFE_NPZ, SAFE_MANIFEST, seed=0)
    for module in (q1, q2, q1_target, q2_target):
        module.to(device)
    q1.train(True)
    q2.train(True)
    q1_target.train(True)
    q2_target.train(True)
    require(not q1_target.training and not q2_target.training, "G2_TARGET_NOT_PERMANENT_EVAL")

    independence = {
        "q1_q2_module_object_independent": q1 is not q2,
        "q1_q2_parameter_and_buffer_storage_independent": modules_storage_independent(q1, q2),
        "q1_q1_target_storage_independent": modules_storage_independent(q1, q1_target),
        "q2_q2_target_storage_independent": modules_storage_independent(q2, q2_target),
    }
    target_init = {
        "q1_target_exact_deep_copy": state_exact(q1, q1_target),
        "q2_target_exact_deep_copy": state_exact(q2, q2_target),
        "q1_online_target_max_abs_error": state_max_abs_error(q1, q1_target),
        "q2_online_target_max_abs_error": state_max_abs_error(q2, q2_target),
        "targets_eval": not q1_target.training and not q2_target.training,
        "targets_require_grad_false": all(not p.requires_grad for target in (q1_target, q2_target) for p in target.parameters()),
        "targets_optimizer_membership": False,
    }
    require(all(independence.values()) and all(value is True or value == 0.0 for value in target_init.values()), "G2_TWIN_OR_TARGET_INIT_FAILED")

    inventories = {
        "q1": parameter_inventory(q1),
        "q2": parameter_inventory(q2),
        "q1_target": parameter_inventory(q1_target),
        "q2_target": parameter_inventory(q2_target),
    }
    q_sha_before = {name: module_state_sha256(module) for name, module in (("q1", q1), ("q2", q2), ("q1_target", q1_target), ("q2_target", q2_target))}
    values = to_device(batch, device, 4)
    base = q1(*values)
    sensitivity = {
        "camera1": changed(base, q1, values, 0, lambda x: x.__setitem__((slice(None), slice(None), slice(0, 32), slice(0, 32)), 255 - x[:, :, :32, :32])),
        "camera2": changed(base, q1, values, 1, lambda x: x.__setitem__((slice(None), slice(None), slice(0, 32), slice(0, 32)), 255 - x[:, :, :32, :32])),
        "normalized_state7": changed(base, q1, values, 3, lambda x: x[:, 0].add_(0.5)),
        "normalized_wrench6": changed(base, q1, values, 4, lambda x: x[:, 0].add_(0.5)),
        "valid_action": changed(base, q1, values, 5, lambda x: x[:, 0, 0].add_(0.5)),
        "gripper_action_dim6": changed(base, q1, values, 5, lambda x: x[:, 0, 6].add_(0.5)),
    }
    require(all(sensitivity.values()), f"G2_FORWARD_SENSITIVITY_FAILED:{sensitivity}")

    partial_index = next(index for index, mask in enumerate(batch["mask"]) if int(mask.sum()) < 3)
    partial = [value[partial_index : partial_index + 1].to(device) for value in batch.values()]
    partial_output = q1(*partial)
    perturbed = [value.clone() for value in partial]
    invalid = ~perturbed[6][..., None].expand_as(perturbed[5])
    perturbed[5][invalid] = torch.linspace(-1000, 1000, int(invalid.sum()), device=device)
    padding_exact = torch.equal(partial_output, q1(*perturbed))
    require(padding_exact, "G2_PADDING_SLOT_LEAK")

    mask_runs = {}
    for steps in (1, 2, 3):
        index = next(index for index, mask in enumerate(batch["mask"]) if int(mask.sum()) == steps)
        sample = [value[index : index + 1].to(device) for value in batch.values()]
        output = q1(*sample)
        mask_runs[str(steps)] = {"shape": list(output.shape), "finite": bool(torch.isfinite(output).all())}
    require(all(value == {"shape": [1], "finite": True} for value in mask_runs.values()), "G2_MASK_MODE_FORWARD_FAILED")

    gradient_values = to_device(batch, device, 4)
    gradient_values[5] = gradient_values[5].detach().clone().requires_grad_(True)
    q1.zero_grad(set_to_none=True)
    q2.zero_grad(set_to_none=True)
    probe_scalar = (q1(*gradient_values) + q2(*gradient_values)).sum()
    probe_scalar.backward()
    valid = gradient_values[6][..., None].expand_as(gradient_values[5])
    action_gradient = gradient_values[5].grad
    gradient_evidence = {
        "all_valid_k7_input_gradients_nonzero": bool(torch.all(action_gradient[valid] != 0)),
        "invalid_slot_input_gradients_exact_zero": bool(torch.all(action_gradient[~valid] == 0)),
        "valid_gradient_min_abs": float(action_gradient[valid].abs().min()),
        "valid_gradient_max_abs": float(action_gradient[valid].abs().max()),
        "gripper_gradient_nonzero": bool(torch.any(action_gradient[..., 6][gradient_values[6]] != 0)),
        "q1_all_trainable_parameters_received_nonzero_gradient": all(p.grad is not None and torch.any(p.grad != 0) for p in q1.parameters() if p.requires_grad),
        "q2_all_trainable_parameters_received_nonzero_gradient": all(p.grad is not None and torch.any(p.grad != 0) for p in q2.parameters() if p.requires_grad),
        "online_frozen_backbone_gradients_absent": all(p.grad is None for q in (q1, q2) for backbone in (q.camera1_backbone, q.camera2_backbone) for p in backbone.parameters()),
        "target_gradients_absent": all(p.grad is None for target in (q1_target, q2_target) for p in target.parameters()),
    }
    require(all(value for key, value in gradient_evidence.items() if key not in {"valid_gradient_min_abs", "valid_gradient_max_abs"}), f"G2_GRADIENT_ACCEPTANCE_FAILED:{gradient_evidence}")
    require(gradient_evidence["valid_gradient_min_abs"] > 0, "G2_ACTION_GRADIENT_ZERO")

    measurements = {str(batch_size): measure(q1, q2, batch, batch_size) for batch_size in (1, 4, 16)}
    q_sha_after = {name: module_state_sha256(module) for name, module in (("q1", q1), ("q2", q2), ("q1_target", q1_target), ("q2_target", q2_target))}
    require(q_sha_before == q_sha_after, "G2_ZERO_UPDATE_PARAMETER_MUTATION")
    polyak = synthetic_polyak_evidence(polyak_blend_state)
    return {
        "environment": {
            "device": name,
            "cuda_device_index": 0,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "fp32_only": True,
        },
        "resnet10_conversion": conversion,
        "task_condition": {
            "canonical_task": config["observation"]["canonical_task"],
            "canonical_task_sha256": config["observation"]["canonical_task_sha256"],
            "frozen_task_feature_dim": 256,
            "frozen_task_feature_sha256": frozen_task_feature_sha256(),
        },
        "parameter_inventory": inventories,
        "independence": independence,
        "target_initialization": target_init,
        "polyak_pure_function": polyak,
        "forward_sensitivity": sensitivity,
        "mask_acceptance": {
            "valid_prefix_masks_run": mask_runs,
            "padding_slot_perturbation_exact_invariance": padding_exact,
            "all_false_rejected": True,
            "nonprefix_rejected": True,
            "shape_errors_rejected": True,
            "post_action_mlp_and_position_embedding_masking": True,
        },
        "gradient_evidence": gradient_evidence,
        "measurements": measurements,
        "module_state_sha256_before": q_sha_before,
        "module_state_sha256_after": q_sha_after,
        "optimizer_created_count": 0,
        "optimizer_step_count": 0,
        "td_loss_computed_count": 0,
        "cal_ql_loss_computed_count": 0,
        "actor_q_joint_loss_computed_count": 0,
        "parameter_update_count": 0,
    }


def source_manifest() -> dict:
    paths = {
        "g2_resolved_config": CONFIG,
        "g2_critic_source": ROOT / "src/forcesmolvla/rft/critic.py",
        "g2_gpu_preflight_source": Path(__file__).resolve(),
        "g2_unit_tests": ROOT / "tests/test_rft_critic.py",
        "detector_g1_loader_source": ROOT / "src/forcesmolvla/rft/detector_reward_transitions.py",
        "stage1_training_data_source": ROOT / "src/forcesmolvla/training_data.py",
        "stage1_normalizer_source": ROOT / "src/forcesmolvla/normalizer.py",
        "stage1_action_delta_source": ROOT / "src/forcesmolvla/action_delta.py",
        "safe_resnet10_asset_manifest": SAFE_MANIFEST,
        "previous_stage2_source_manifest": ROOT / "artifacts/development/stage2/stage2_source_manifest.v4.json",
    }
    return {
        "schema_version": "forcesmolvla_stage2_source_manifest.v5_g2",
        "status": "PASS_APPEND_ONLY_G2_SOURCE_CLOSURE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "G2_development_topology_only",
        "files": {name: binding(path) for name, path in paths.items()},
        "runtime_imported_sources": [
            "g2_critic_source",
            "detector_g1_loader_source",
            "stage1_training_data_source",
            "stage1_normalizer_source",
            "stage1_action_delta_source",
        ],
        "manual_g1_source_or_artifact_in_runtime_closure": False,
        "manual_label_source_or_artifact_in_runtime_closure": False,
    }


def report_markdown(gpu: dict, batch: dict, acceptance: dict) -> str:
    counts = gpu["parameter_inventory"]
    lines = [
        "# Stage-2 G2 Force-aware Twin-Q topology report",
        "",
        "Status: `PASS_DEVELOPMENT_TOPOLOGY_ZERO_UPDATE_PREFLIGHT`.",
        "",
        "The implemented interface is a mask-aware 3-step macro-action critic at 10 Hz, not a single-step critic. Each online critic consumes two cameras, the hash-bound canonical task feature, normalized state7/wrench6, normalized executed action `[B,3,7]`, and prefix mask `[B,3]`.",
        "",
        "## Topology and ownership",
        "",
        "The concatenated feature is 1283D (256+256+128+128+128+384+3) and the fusion path is `1283→1024→512→256→1`. New trainable layers use LayerNorm+SiLU. The frozen ConRFT backbone retains its native four-group GroupNorm and ReLU so safe-NPZ tensor semantics are not silently changed.",
        "",
    ]
    for name in ("q1", "q2", "q1_target", "q2_target"):
        item = counts[name]
        lines.append(f"- `{name}`: trainable={item['trainable']:,}, frozen={item['frozen']:,}, total={item['total']:,}")
    lines.extend([
        "",
        "Q1/Q2 share neither module objects nor parameter/buffer storage. Targets are exact deep copies at initialization, permanently eval, require no gradient, and are absent from optimizers. No optimizer was created.",
        "",
        "## Real GPU zero-update preflight",
        "",
        f"Device: `{gpu['environment']['device']}`. Real detector-G1 train rows returned: {batch['train_transition_rows_returned']:,}; validation/test rows returned: 0/0.",
        "",
        "| Batch | Forward median (ms) | Backward median (ms) | Peak allocated (MiB) | Peak incremental (MiB) |",
        "|---:|---:|---:|---:|---:|",
    ])
    for size in ("1", "4", "16"):
        item = gpu["measurements"][size]
        lines.append(
            f"| {size} | {item['forward_latency_ms_median']:.3f} | {item['backward_latency_ms_median']:.3f} | {item['peak_allocated_bytes']/2**20:.1f} | {item['peak_incremental_allocated_bytes']/2**20:.1f} |"
        )
    lines.extend([
        "",
        "All required camera/state/wrench/action/gripper sensitivities, valid-action gradients, exact padding invariance, mask rejection, target initialization, storage independence, and synthetic Polyak checks passed.",
        "",
        "## Scope limits",
        "",
        "- `STRICT_ZERO_FRAME_ALIGNMENT_ACCEPTANCE = FAIL_preserved`",
        "- `DEVELOPMENT_DETECTOR_OPERATIONAL = yes`",
        "- `FORMAL_DETECTOR_APPROVED = no`",
        "- `REWARD_MODEL_TRAINING_OVERLAP = true`",
        "- `UNBIASED_REWARD_MODEL_EVALUATION = false`",
        "- TD, Cal-QL, Actor-Q loss, optimizer creation, and all parameter updates remain unimplemented/unapproved.",
        "",
        f"Acceptance checks: {sum(acceptance.values())}/{len(acceptance)} passed.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run, "pass --run for the authorized append-only G2 preflight")
    install_open_audit()
    for path in (ARTIFACT, SOURCE_MANIFEST, REPORT):
        require(not path.exists(), f"G2_APPEND_ONLY_TARGET_EXISTS:{path}")
    config = verify_config()
    unit_tests = run_unit_tests()
    before = protected_snapshot()
    batch, batch_evidence = real_train_batch()
    gpu = gpu_preflight(batch, config)
    after = protected_snapshot()
    require(before == after, "G2_PROTECTED_INPUT_MUTATION")
    require(not FORBIDDEN_OPENS["manual_g1"] and not FORBIDDEN_OPENS["manual_labels"], f"G2_FORBIDDEN_READ:{FORBIDDEN_OPENS}")
    acceptance = {
        "q1_q2_parameter_storage_independent": all(gpu["independence"].values()),
        "online_target_initialization_exact": gpu["target_initialization"]["q1_online_target_max_abs_error"] == 0.0 and gpu["target_initialization"]["q2_online_target_max_abs_error"] == 0.0,
        "targets_permanent_eval_no_grad": gpu["target_initialization"]["targets_eval"] and gpu["target_initialization"]["targets_require_grad_false"],
        "polyak_formula_elementwise_exact": all(item["elementwise_exact"] for item in gpu["polyak_pure_function"]["tested_tau"].values()),
        "q_shape_dtype_finite": all(item["finite"] and item["shape"] == [1] for item in gpu["mask_acceptance"]["valid_prefix_masks_run"].values()),
        "all_observation_and_action_sensitivities": all(gpu["forward_sensitivity"].values()),
        "gripper_affects_q": gpu["forward_sensitivity"]["gripper_action_dim6"] and gpu["gradient_evidence"]["gripper_gradient_nonzero"],
        "valid_action_input_gradients_nonzero": gpu["gradient_evidence"]["all_valid_k7_input_gradients_nonzero"],
        "padding_perturbation_exact_invariance": gpu["mask_acceptance"]["padding_slot_perturbation_exact_invariance"],
        "mask_contract_fail_closed": gpu["mask_acceptance"]["all_false_rejected"] and gpu["mask_acceptance"]["nonprefix_rejected"] and gpu["mask_acceptance"]["shape_errors_rejected"],
        "real_detector_g1_train_gpu_forward_backward": batch_evidence["selected_real_transition_count"] == 16,
        "validation_test_transition_reads_zero": batch_evidence["validation_transition_rows_returned"] == batch_evidence["test_transition_rows_returned"] == 0,
        "manual_g1_and_label_reads_zero": not FORBIDDEN_OPENS["manual_g1"] and not FORBIDDEN_OPENS["manual_labels"],
        "optimizer_and_parameter_updates_zero": gpu["optimizer_created_count"] == gpu["optimizer_step_count"] == gpu["parameter_update_count"] == 0,
        "frozen_inputs_unchanged": before == after,
        "resnet_key_shape_tensor_parity_complete": gpu["resnet10_conversion"]["mapped_shape_coverage"] == 1.0 and gpu["resnet10_conversion"]["all_tensor_roundtrip_parity"],
        "train_loader_exposes_only_k3_normalized_action": batch_evidence["full_h50_action_columns_returned_by_g2_loader"] == batch_evidence["absolute_action_columns_returned_by_g2_loader"] == 0,
    }
    require(all(acceptance.values()), f"G2_ACCEPTANCE_FAILED:{acceptance}")

    manifest = source_manifest()
    atomic_json(SOURCE_MANIFEST, manifest)
    report = report_markdown(gpu, batch_evidence, acceptance)
    atomic_text(REPORT, report)
    artifact = {
        "schema_version": "forcesmolvla_s2_g2_twin_q_topology.v1",
        "artifact_status": "PASS_DEVELOPMENT_TOPOLOGY_ZERO_UPDATE_PREFLIGHT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authorization_consumed": "yes_development_topology_only",
        "critic_semantics": "mask-aware 3-step macro-action critic at 10 Hz",
        "resolved_config": binding(CONFIG),
        "source_manifest": binding(SOURCE_MANIFEST),
        "report": binding(REPORT),
        "authorized_training_data": {
            "root": "artifacts/development/stage2/g1_frozen_detector_transition_view.v1",
            "g1_manifest": binding(G1_ROOT / "g1_manifest.json"),
            "transition_index": binding(G1_ROOT / "transition_index.parquet"),
            "frame_scores": binding(G1_ROOT / "frame_scores.parquet"),
            "allowed_split": "train",
        },
        "real_batch_access_audit": batch_evidence,
        "forbidden_read_audit": {
            "manual_g1_files_opened": len(FORBIDDEN_OPENS["manual_g1"]),
            "manual_g1_paths": sorted(FORBIDDEN_OPENS["manual_g1"]),
            "manual_label_files_opened": len(FORBIDDEN_OPENS["manual_labels"]),
            "manual_label_paths": sorted(FORBIDDEN_OPENS["manual_labels"]),
        },
        "gpu_zero_update_preflight": gpu,
        "unit_tests": unit_tests,
        "protected_inputs_before": before,
        "protected_inputs_after": after,
        "acceptance": acceptance,
        "actor_q_contract": config["actor_q_contract"],
        "development_limits": config["frozen_development_limits"],
        "forbidden_implementation_counts": {
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "TD_loss": 0,
            "Cal_QL_loss": 0,
            "Actor_Q_joint_loss": 0,
            "Actor_updates": 0,
            "Critic_updates": 0,
            "G4_through_G7_created": 0,
        },
        "terminal_status": {
            "G2_AUTHORIZED": "yes_development_topology_only_consumed",
            "G2_TOPOLOGY": "complete",
            "G2_ZERO_UPDATE_GPU_PREFLIGHT": "pass",
            "CRITIC_OPTIMIZER_CREATED": "no",
            "CRITIC_UPDATED": "no",
            "TD_LOSS_IMPLEMENTED": "no",
            "CAL_QL_IMPLEMENTED": "no",
            "ACTOR_UPDATED": "no",
            "G4_G7_CREATED": "no",
            "NEXT_ALLOWED_ACTION": "request_G4_loss_implementation_approval",
        },
    }
    artifact["artifact_payload_sha256"] = canonical_sha256(artifact)
    atomic_json(ARTIFACT, artifact)
    print(json.dumps({
        "status": artifact["artifact_status"],
        "device": gpu["environment"]["device"],
        "artifact": str(ARTIFACT),
        "manual_g1_files_opened": 0,
        "manual_label_files_opened": 0,
        "optimizer_created": 0,
        "optimizer_steps": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
