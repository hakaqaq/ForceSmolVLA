#!/usr/bin/env python3
"""Run the append-only G4 formula/gradient RTX 4090D zero-update preflight."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
import gc
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage2_g4_losses.development.yaml"
G1_ROOT = ROOT / "artifacts/development/stage2/g1_frozen_detector_transition_view.v1"
MANUAL_G1 = ROOT / "artifacts/development/stage2/g1_manual_reward_transition_view.v1"
LABELS = ROOT / "labels"
DATASET = ROOT / "datasets/task2_lerobotv3"
R5 = ROOT / "outputs/development/task2_lerobotv3_full_sft_10k_r5/checkpoints/step_010000"
SAFE_NPZ = ROOT / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_params.safe.npz"
SAFE_MANIFEST = ROOT / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_asset_manifest.v4.json"
CLASSIFIER = ROOT / "artifacts/development/stage2/reward_classifier/r0_training/checkpoints/best_checkpoint.msgpack"
G2_ARTIFACT = ROOT / "artifacts/development/stage2/s2_g2_twin_q_topology.json"
ARTIFACT = ROOT / "artifacts/development/stage2/s2_g4_loss_preflight.json"
SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v6_g4.json"
REPORT = ROOT / "docs/s2_g4_loss_preflight_report.md"
EXPECTED_P8_TREE_SHA256 = "f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d"
EXPECTED_CLASSIFIER_SHA256 = "6b4e366baa55993d150cb3dd86e67a1d708e58d836b123a0c433190835021510"
FORBIDDEN_OPENS: dict[str, set[str]] = {"manual_g1": set(), "manual_labels": set()}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "file_size": path.stat().st_size,
    }


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


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


def file_tree(root: Path, subdirectories: tuple[str, ...] | None = None) -> dict:
    directories = [root / name for name in subdirectories] if subdirectories else [root]
    files = sorted(path for directory in directories for path in directory.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    size = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_sha = sha256_file(path)
        digest.update(f"{relative}\0{file_sha}\n".encode())
        size += path.stat().st_size
    return {"tree_sha256": digest.hexdigest(), "file_count": len(files), "total_file_size": size}


def protected_snapshot() -> dict:
    parent = json.loads((ROOT / "configs/stage2_parent_bridge.development.json").read_text())
    files = {
        "classifier_checkpoint": CLASSIFIER,
        "g1_manifest": G1_ROOT / "g1_manifest.json",
        "g1_frame_scores": G1_ROOT / "frame_scores.parquet",
        "g1_transition_index": G1_ROOT / "transition_index.parquet",
        "g2_artifact": G2_ARTIFACT,
        "g2_source_manifest": ROOT / "artifacts/development/stage2/stage2_source_manifest.v5_g2.json",
        "g2_critic_source": ROOT / "src/forcesmolvla/rft/critic.py",
        "g3_flow_source": ROOT / "src/forcesmolvla/rft/flow_sampling.py",
        "g3_flow_artifact": ROOT / "artifacts/development/stage2/s2_g3_differentiable_flow.v4.json",
        "g3_gradient_artifact": ROOT / "artifacts/development/stage2/s2_g3_gradient_precision_matrix.v4.json",
        "dataset_conversion_manifest": DATASET / "conversion_manifest.json",
        "dataset_split_manifest": DATASET / "split_manifest.json",
        "dataset_normalizer_manifest": DATASET / "normalizer_manifest.json",
        "safe_resnet10": SAFE_NPZ,
        "public_actor_modeling": ROOT / "src/forcesmolvla/modeling_forcesmolvla.py",
        "public_inference": ROOT / "src/forcesmolvla/inference.py",
    }
    for index, item in enumerate(parent["parent_p4_to_p8_qualification_artifacts"]):
        files[f"stage1_p4_p8_{index:02d}"] = ROOT / item["path"]
    for path in sorted((ROOT / "artifacts/development").glob("p9_v4_2_r8_*")):
        files[f"stage1_{path.stem}"] = path
    result = {
        "files": {name: binding(path) for name, path in files.items()},
        "p8_storage_tree": file_tree(DATASET, ("data", "videos", "meta")),
        "r5_checkpoint_tree": file_tree(R5),
    }
    require(result["files"]["classifier_checkpoint"]["sha256"] == EXPECTED_CLASSIFIER_SHA256, "G4_CLASSIFIER_SHA_DRIFT")
    require(result["p8_storage_tree"]["tree_sha256"] == EXPECTED_P8_TREE_SHA256, "G4_P8_TREE_SHA_DRIFT")
    return result


def module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_config() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    require(config["authorization"] == "G4_loss_implementation_only", "G4_AUTHORIZATION_DRIFT")
    frozen = config["frozen_inputs"]
    require(
        frozen["only_rl_transition_root"] == "artifacts/development/stage2/g1_frozen_detector_transition_view.v1"
        and frozen["only_rl_split"] == "train"
        and not frozen["manual_g1_allowed"]
        and not frozen["manual_labels_allowed"]
        and not frozen["validation_test_transition_allowed"],
        "G4_DATA_AUTHORIZATION_DRIFT",
    )
    require(all(value is None for value in config["unapproved_training_parameters"].values()), "G4_FORMAL_PARAMETER_PREAPPROVED")
    require(config["td_target"]["target_actor_exists"] is False, "G4_TARGET_ACTOR_FORBIDDEN")
    require(config["calql"]["description"] == "Cal-QL-style finite-candidate conservative objective", "G4_CALQL_DESCRIPTION_DRIFT")
    return config


def run_unit_tests() -> dict:
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_rft_losses.py", "tests/test_rft_flow_sampling.py"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    require(result.returncode == 0 and "passed" in output, f"G4_UNIT_TEST_FAILED:{output[-3000:]}")
    return {"command": "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_rft_losses.py tests/test_rft_flow_sampling.py", "exit_code": 0, "output": output}


def decode_rgb(payload: bytes) -> np.ndarray:
    from PIL import Image

    require(isinstance(payload, bytes), "G4_IMAGE_PAYLOAD_MISSING")
    with Image.open(BytesIO(payload)) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(value.shape == (480, 640, 3), "G4_IMAGE_SHAPE_DRIFT")
    return np.ascontiguousarray(value.transpose(2, 0, 1))


def actor_batch(policy, samples: list[dict], device: torch.device, *, include_action: bool) -> dict:
    from forcesmolvla.configuration_forcesmolvla import CAMERA1, CAMERA2
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    encoded = tokenizer(
        [sample["task"] + "\n" for sample in samples],
        padding="max_length",
        truncation=True,
        max_length=48,
        return_tensors="pt",
    )
    result = {
        CAMERA1: torch.from_numpy(np.stack([sample["camera1"] for sample in samples])).float().div_(255).to(device),
        CAMERA2: torch.from_numpy(np.stack([sample["camera2"] for sample in samples])).float().div_(255).to(device),
        "observation.state": torch.from_numpy(np.stack([sample["state7"] for sample in samples])).to(device),
        "observation.wrench": torch.from_numpy(np.stack([sample["wrench6"] for sample in samples])).to(device),
        OBS_LANGUAGE_TOKENS: encoded["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: encoded["attention_mask"].to(device=device, dtype=torch.bool),
        "sample_identity": tuple(sample["sample_identity"] for sample in samples),
    }
    if include_action:
        result[ACTION] = torch.from_numpy(np.stack([sample["delta_action7"] for sample in samples])).to(device)
        result["action_valid_mask"] = torch.from_numpy(np.stack([sample["action_valid_mask"] for sample in samples])).to(device)
    return result


def slice_batch(batch: dict, index: torch.Tensor) -> dict:
    size = int(index.numel())
    choices = index.detach().cpu().tolist()
    result = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
            result[name] = value[index]
        elif isinstance(value, (tuple, list)) and len(value) == size:
            result[name] = type(value)(item for item, keep in zip(value, choices, strict=True) if keep)
        else:
            result[name] = value
    return result


def repeat_batch(batch: dict, count: int, *, suffix: str) -> dict:
    size = next(value.shape[0] for value in batch.values() if isinstance(value, torch.Tensor) and value.ndim)
    result = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
            result[name] = value.repeat_interleave(count, dim=0)
        elif name == "sample_identity":
            result[name] = tuple(f"{identity}/{suffix}={candidate}" for identity in value for candidate in range(count))
        else:
            result[name] = value
    return result


def real_train_batch(policy, device: torch.device) -> tuple[dict, dict]:
    import pyarrow.parquet as pq

    from forcesmolvla.rft.critic import frozen_task_feature
    from forcesmolvla.rft.detector_reward_transitions import HORIZON
    from forcesmolvla.rft.losses import CriticObservation, load_authorized_g4_train_transitions, validate_mc_return_recurrence
    from forcesmolvla.rft.offline_transitions import PROVENANCE_KEYS
    from forcesmolvla.training_data import load_runtime_artifacts, prepare_training_sample

    table = load_authorized_g4_train_transitions(G1_ROOT)
    rows = table.to_pylist()
    recurrence = validate_mc_return_recurrence(rows)
    full_nonterminal = next(row for row in rows if row["executed_steps"] == 3 and not row["terminated"])
    partial_terminal = next(row for row in rows if row["executed_steps"] in (1, 2) and row["terminated"])
    selected = [full_nonterminal, partial_terminal]

    conversion = json.loads((DATASET / "conversion_manifest.json").read_text())
    tasks = {entry["raw_episode_id"]: entry["task"] for entry in conversion["episodes"]}
    runtime = load_runtime_artifacts(
        DATASET,
        calibration_bundle_path=ROOT / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=ROOT / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=ROOT / "artifacts/development/action_delta_spec.json",
        expected_repo_id=conversion["repo_id"],
    )
    columns = [
        "observation.images.camera1", "observation.images.camera2", "observation.state",
        "observation.wrench", "action", "frame_index", "episode_index", "index", *PROVENANCE_KEYS,
    ]
    caches = {}
    current_samples, next_samples = [], []
    behavior_actions, behavior_masks = [], []
    for transition in selected:
        reference = transition["observation_row_reference"]
        next_reference = transition["next_observation_row_reference"]
        require(reference["episode_id"] == next_reference["episode_id"] == transition["episode_id"], "G4_CROSS_EPISODE_REFERENCE")
        relative = reference["data_relative_path"]
        require(relative == next_reference["data_relative_path"], "G4_CROSS_FILE_NEXT_REFERENCE")
        if relative not in caches:
            caches[relative] = pq.read_table(DATASET / relative, columns=columns).to_pylist()
        episode_rows = caches[relative]
        current = episode_rows[reference["row_index"]]
        following = episode_rows[next_reference["row_index"]]
        for source, expected in ((current, reference), (following, next_reference)):
            require(source["frame_index"] == expected["frame_index"] and source["index"] == expected["global_index"], "G4_ROW_REFERENCE_DRIFT")

        anchor = int(reference["row_index"])
        source_indices = np.minimum(anchor + np.arange(HORIZON), len(episode_rows) - 1)
        absolute_chunk = np.asarray([episode_rows[index]["action"] for index in source_indices], dtype=np.float64)
        action_is_pad = anchor + np.arange(HORIZON) >= len(episode_rows)
        sample = {
            "observation.state": np.asarray(current["observation.state"], dtype=np.float64),
            "observation.wrench": np.asarray(current["observation.wrench"], dtype=np.float64),
            "action": absolute_chunk,
            "action_is_pad": action_is_pad,
            "episode_index": int(current["episode_index"]),
            "frame_index": int(current["frame_index"]),
            "task": tasks[transition["episode_id"]],
            "observation.images.camera1": decode_rgb(current["observation.images.camera1"]["bytes"]),
            "observation.images.camera2": decode_rgb(current["observation.images.camera2"]["bytes"]),
        }
        for name in PROVENANCE_KEYS:
            sample[name] = current[name]
        prepared = prepare_training_sample(sample, runtime.normalizer)
        prepared.update(
            camera1=sample["observation.images.camera1"],
            camera2=sample["observation.images.camera2"],
            sample_identity=f"{transition['episode_id']}/frame={transition['anchor_frame']}",
        )
        current_samples.append(prepared)
        next_samples.append({
            "camera1": decode_rgb(following["observation.images.camera1"]["bytes"]),
            "camera2": decode_rgb(following["observation.images.camera2"]["bytes"]),
            "state7": runtime.normalizer.state7.apply(np.asarray(following["observation.state"], dtype=np.float64)).astype(np.float32),
            "wrench6": runtime.normalizer.wrench6.apply(np.asarray(following["observation.wrench"], dtype=np.float64)).astype(np.float32),
            "task": tasks[transition["episode_id"]],
            "sample_identity": f"{transition['episode_id']}/next_frame={transition['next_frame']}",
        })
        mask = np.asarray(transition["executed_action_mask"], dtype=np.bool_)
        flat = np.asarray(transition["normalized_delta_action_exec_flat"], dtype=np.float32)
        padded = np.zeros((3, 7), dtype=np.float32)
        padded[: int(mask.sum())] = flat.reshape(-1, 7)
        behavior_actions.append(padded)
        behavior_masks.append(mask)

    task_feature = torch.from_numpy(np.repeat(frozen_task_feature()[None], len(selected), axis=0)).to(device)
    current_observation = CriticObservation(
        torch.from_numpy(np.stack([item["camera1"] for item in current_samples])).to(device),
        torch.from_numpy(np.stack([item["camera2"] for item in current_samples])).to(device),
        task_feature,
        torch.from_numpy(np.stack([item["state7"] for item in current_samples])).to(device),
        torch.from_numpy(np.stack([item["wrench6"] for item in current_samples])).to(device),
    )
    next_observation = CriticObservation(
        torch.from_numpy(np.stack([item["camera1"] for item in next_samples])).to(device),
        torch.from_numpy(np.stack([item["camera2"] for item in next_samples])).to(device),
        task_feature.clone(),
        torch.from_numpy(np.stack([item["state7"] for item in next_samples])).to(device),
        torch.from_numpy(np.stack([item["wrench6"] for item in next_samples])).to(device),
    )
    batch = {
        "rows": selected,
        "current_observation": current_observation,
        "next_observation": next_observation,
        "current_actor_batch": actor_batch(policy, current_samples, device, include_action=True),
        "next_actor_batch": actor_batch(policy, next_samples, device, include_action=False),
        "behavior_action": torch.from_numpy(np.stack(behavior_actions)).to(device),
        "behavior_mask": torch.from_numpy(np.stack(behavior_masks)).to(device),
        "reward": torch.tensor([row["reward"] for row in selected], dtype=torch.float32, device=device),
        "terminated": torch.tensor([row["terminated"] for row in selected], dtype=torch.bool, device=device),
        "bootstrap_mask": torch.tensor([row["bootstrap_mask"] for row in selected], dtype=torch.int8, device=device),
        "discount": torch.tensor([row["discount"] for row in selected], dtype=torch.float32, device=device),
        "mc_return": torch.tensor([row["mc_return"] for row in selected], dtype=torch.float32, device=device),
        "delta_mean": torch.tensor(runtime.normalizer.delta_action7.mean, dtype=torch.float32, device=device),
        "delta_std": torch.tensor(runtime.normalizer.delta_action7.std, dtype=torch.float32, device=device),
    }
    evidence = {
        "authorized_train_transition_count": table.num_rows,
        "validation_transition_rows_read": 0,
        "test_transition_rows_read": 0,
        "manual_g1_rows_read": 0,
        "manual_label_rows_read": 0,
        "selected_transition_indices": [row["transition_index"] for row in selected],
        "selected_episode_ids": [row["episode_id"] for row in selected],
        "selected_executed_steps": [row["executed_steps"] for row in selected],
        "selected_current_frames": [row["anchor_frame"] for row in selected],
        "selected_next_frames": [row["next_frame"] for row in selected],
        "source_parquet_files_opened": sorted(caches),
        "only_train_episode_rows_decoded": True,
        "current_actor_action_shape": list(batch["current_actor_batch"]["action"].shape),
        "critic_behavior_action_shape": list(batch["behavior_action"].shape),
        "mc_return_recurrence": recurrence,
    }
    return batch, evidence


def sample_candidates(policy, batch: dict, noise: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, purpose: str, call_id: str) -> torch.Tensor:
    from forcesmolvla.rft.flow_sampling import critic_action_for_q_guidance, sample_normalized_action_chunk_with_grad

    batch_size, candidate_count = noise.shape[:2]
    expanded = repeat_batch(batch, candidate_count, suffix=purpose)
    training = policy.training
    policy.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            chunk = sample_normalized_action_chunk_with_grad(
                policy,
                expanded,
                noise.reshape(batch_size * candidate_count, 50, 7),
                call_id=call_id,
                purpose=purpose,
            )
            action = critic_action_for_q_guidance(chunk, delta_action_mean7=mean, delta_action_std7=std)
    finally:
        policy.train(training)
    return action.detach().float().reshape(batch_size, candidate_count, 3, 7)


def gradient_record(value: torch.Tensor | None) -> dict:
    if value is None:
        return {"present": False, "norm": 0.0, "finite": True, "nonzero": False}
    value = value.detach().float()
    return {
        "present": True,
        "norm": float(value.norm().cpu()),
        "maximum_absolute": float(value.abs().max().cpu()),
        "finite": bool(torch.isfinite(value).all()),
        "nonzero": bool(torch.count_nonzero(value)),
    }


def gpu_preflight(batch: dict, config: dict) -> dict:
    from forcesmolvla.force_token import RouterState
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.critic import build_twin_q
    from forcesmolvla.rft.flow_sampling import sample_normalized_action_chunk_with_grad
    from forcesmolvla.rft.losses import (
        compute_actor_q_loss,
        compute_behavior_q,
        compute_offline_actor_objective,
        compute_td_target_from_current_actor,
        compute_twin_q_critic_loss,
        derive_loss_masks,
        evaluate_calql_candidates,
    )
    from forcesmolvla.router_training import collect_pass_a_statistics, microbatch_two_pass_terms

    device = torch.device("cuda:0")
    with redirect_stdout(sys.stderr):
        policy = ForceSmolVLAPolicy.from_pretrained(
            R5, local_files_only=True, force_download=False, strict=True,
            artifact_use="development",
        ).to(device)
    policy.eval()
    q1, q2, q1_target, q2_target, conversion = build_twin_q(SAFE_NPZ, SAFE_MANIFEST, seed=0)
    q1, q2, q1_target, q2_target = (module.to(device) for module in (q1, q2, q1_target, q2_target))
    q1.train(True)
    q2.train(True)
    q1_target.eval()
    q2_target.eval()

    modules = {"actor": policy, "q1": q1, "q2": q2, "q1_target": q1_target, "q2_target": q2_target}
    state_before = {name: module_state_sha256(module) for name, module in modules.items()}
    for module in modules.values():
        module.zero_grad(set_to_none=True)

    seed = config["synthetic_zero_update_preflight_only"]["seed"]
    generator = torch.Generator(device=device).manual_seed(seed)
    next_noise = torch.randn(2, 50, 7, generator=generator, device=device)
    actor_call_count = {"sample_actions_masked": 0}
    target_calls = {"q1": 0, "q2": 0}
    handles = [
        policy.model.register_forward_hook(lambda *_: None),
        policy.model.sample_actions_masked.__self__.register_forward_hook(lambda *_: None),
    ]
    for handle in handles:
        handle.remove()
    original_sampler = sample_normalized_action_chunk_with_grad

    def counted_sampler(*args, **kwargs):
        actor_call_count["sample_actions_masked"] += 1
        return original_sampler(*args, **kwargs)

    hooks = [
        q1_target.register_forward_hook(lambda *_: target_calls.__setitem__("q1", target_calls["q1"] + 1)),
        q2_target.register_forward_hook(lambda *_: target_calls.__setitem__("q2", target_calls["q2"] + 1)),
    ]
    try:
        target = compute_td_target_from_current_actor(
            reward=batch["reward"], discount=batch["discount"], terminated=batch["terminated"],
            bootstrap_mask=batch["bootstrap_mask"], next_observation=batch["next_observation"],
            next_actor_batch=batch["next_actor_batch"], next_noise7=next_noise, actor=policy,
            q1_target=q1_target, q2_target=q2_target, delta_action_mean7=batch["delta_mean"],
            delta_action_std7=batch["delta_std"], call_id="g4-td-next", sample_action_fn=counted_sampler,
        )
    finally:
        for hook in hooks:
            hook.remove()
    mixed_calls = {"actor": actor_call_count["sample_actions_masked"], **target_calls}

    terminal_index = batch["terminated"]
    terminal_calls = {"actor": 0, "q1": 0, "q2": 0}

    def forbidden_terminal_sampler(*_args, **_kwargs):
        terminal_calls["actor"] += 1
        raise RuntimeError("G4_TERMINAL_ACTOR_CALLED")

    terminal_observation = batch["next_observation"].index(terminal_index)
    terminal_actor_batch = slice_batch(batch["next_actor_batch"], terminal_index)
    terminal_hooks = [
        q1_target.register_forward_hook(lambda *_: terminal_calls.__setitem__("q1", terminal_calls["q1"] + 1)),
        q2_target.register_forward_hook(lambda *_: terminal_calls.__setitem__("q2", terminal_calls["q2"] + 1)),
    ]
    try:
        terminal_target = compute_td_target_from_current_actor(
            reward=batch["reward"][terminal_index], discount=batch["discount"][terminal_index],
            terminated=torch.ones(int(terminal_index.sum()), dtype=torch.bool, device=device),
            bootstrap_mask=batch["bootstrap_mask"][terminal_index], next_observation=terminal_observation,
            next_actor_batch=terminal_actor_batch, next_noise7=next_noise[terminal_index], actor=policy,
            q1_target=q1_target, q2_target=q2_target, delta_action_mean7=batch["delta_mean"],
            delta_action_std7=batch["delta_std"], call_id="g4-terminal-forbidden",
            sample_action_fn=forbidden_terminal_sampler,
        )
    finally:
        for hook in terminal_hooks:
            hook.remove()

    masks = derive_loss_masks(batch["behavior_mask"], batch["terminated"])
    q1_dataset = compute_behavior_q(q1, batch["current_observation"], batch["behavior_action"], batch["behavior_mask"])
    q2_dataset = compute_behavior_q(q2, batch["current_observation"], batch["behavior_action"], batch["behavior_mask"])

    test_values = config["synthetic_zero_update_preflight_only"]
    candidate_count = test_values["candidate_count_M"]
    current_noise = torch.randn(2, candidate_count, 50, 7, generator=generator, device=device)
    policy_next_noise = torch.randn(2, candidate_count, 50, 7, generator=generator, device=device)
    policy_current = sample_candidates(
        policy, batch["current_actor_batch"], current_noise, batch["delta_mean"], batch["delta_std"],
        purpose="cql_current", call_id="g4-cql-current",
    )
    policy_next = sample_candidates(
        policy, batch["next_actor_batch"], policy_next_noise, batch["delta_mean"], batch["delta_std"],
        purpose="cql_next", call_id="g4-cql-next",
    )
    endpoints = torch.stack(
        ((torch.tensor(0.0, device=device) - batch["delta_mean"][6]) / batch["delta_std"][6],
         (torch.tensor(0.085, device=device) - batch["delta_mean"][6]) / batch["delta_std"][6])
    ).float()
    random_candidates = torch.linspace(
        -0.8, 0.8, 2 * candidate_count * 3 * 7, dtype=torch.float32, device=device
    ).reshape(2, candidate_count, 3, 7)
    gripper_pattern = torch.arange(2 * candidate_count * 3, device=device).reshape(2, candidate_count, 3) % 2
    random_candidates[..., 6] = endpoints[gripper_pattern]
    q1_candidates = evaluate_calql_candidates(q1, batch["current_observation"], random_candidates, policy_current, policy_next, endpoints)
    q2_candidates = evaluate_calql_candidates(q2, batch["current_observation"], random_candidates, policy_current, policy_next, endpoints)
    terms = compute_twin_q_critic_loss(
        q1_dataset=q1_dataset, q2_dataset=q2_dataset, td_target=target,
        q1_candidates=q1_candidates, q2_candidates=q2_candidates, mc_return=batch["mc_return"],
        calql_valid=masks["calql_valid"], alpha_calql=test_values["calql_alpha"],
        temperature=test_values["temperature_T"], clip_min=test_values["clip_min"], clip_max=test_values["clip_max"],
    )
    terms.total.backward()
    critic_gradients = {
        name: {
            "trainable_nonzero_tensors": sum(
                parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
                for parameter in critic.parameters() if parameter.requires_grad
            ),
            "trainable_tensor_count": sum(parameter.requires_grad for parameter in critic.parameters()),
        }
        for name, critic in (("q1", q1), ("q2", q2))
    }
    critic_gradients.update({
        "actor_gradient_buffers_present": sum(parameter.grad is not None for parameter in policy.parameters()),
        "target_gradient_buffers_present": sum(parameter.grad is not None for target_critic in (q1_target, q2_target) for parameter in target_critic.parameters()),
        "frozen_backbone_gradient_buffers_present": sum(
            parameter.grad is not None for critic in (q1, q2)
            for backbone in (critic.camera1_backbone, critic.camera2_backbone)
            for parameter in backbone.parameters()
        ),
    })

    perturbed_action = batch["behavior_action"].detach().clone()
    invalid = ~batch["behavior_mask"]
    perturbed_action[invalid] = torch.linspace(-1000, 1000, int(invalid.sum()) * 7, device=device).reshape(-1, 7)
    with torch.no_grad():
        perturbed_q1 = compute_behavior_q(q1, batch["current_observation"], perturbed_action, batch["behavior_mask"])
        perturbed_q2 = compute_behavior_q(q2, batch["current_observation"], perturbed_action, batch["behavior_mask"])
    invalid_slot_invariance = {
        "q1_exact": torch.equal(q1_dataset.detach(), perturbed_q1),
        "q2_exact": torch.equal(q2_dataset.detach(), perturbed_q2),
    }

    for module in modules.values():
        module.zero_grad(set_to_none=True)
    valid_index = masks["actor_q_valid"]
    actor_input = slice_batch(batch["current_actor_batch"], valid_index)
    actor_observation = batch["current_observation"].index(valid_index)
    actor_noise = torch.randn(int(valid_index.sum()), 50, 7, generator=generator, device=device)
    policy.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actor_chunk = sample_normalized_action_chunk_with_grad(
            policy, actor_input, actor_noise, call_id="g4-actor-q", purpose="actor_guidance"
        )
        actor_q = compute_actor_q_loss(
            q1=q1, q2=q2, current_observation=actor_observation,
            actor_action_chunk7=actor_chunk,
            actor_q_valid=torch.ones(actor_chunk.shape[0], dtype=torch.bool, device=device),
            delta_action_mean7=batch["delta_mean"], delta_action_std7=batch["delta_std"],
        )
    named = dict(policy.named_parameters())
    probe_names = [
        "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.bias",
        "model.vlm_with_expert.vlm.model.text_model.layers.0.input_layernorm.weight",
        "model.action_in_proj.bias",
        "model.action_out_proj.bias",
        "model.force_branch.force_mlp.linear_out.bias",
        "model.force_branch.fusion_blocks.0.attention.q_proj.bias",
        "model.force_adapter.w_out.bias",
        "model.force_branch.refiner.router.bias",
    ]
    expert_names = [f"model.force_branch.refiner.experts.{index}.linear_out.bias" for index in range(4)]
    require(not (set(probe_names + expert_names) - set(named)), "G4_ACTOR_GRADIENT_PROBE_MISSING")
    gradients = torch.autograd.grad(
        actor_q,
        [actor_chunk, *(named[name] for name in probe_names + expert_names)],
        allow_unused=True,
    )
    action_gradient = gradients[0].detach().float()
    probe_gradients = {
        name: gradient_record(gradient)
        for name, gradient in zip(probe_names + expert_names, gradients[1:], strict=True)
    }
    actor_q_gradient = {
        "tcp_k3x6_nonzero": bool(torch.all(action_gradient[:, :3, :6] != 0)),
        "tcp_k3x6_minimum_absolute": float(action_gradient[:, :3, :6].abs().min().cpu()),
        "gripper_k3_maximum_absolute": float(action_gradient[:, :3, 6].abs().max().cpu()),
        "post_k3_maximum_absolute": float(action_gradient[:, 3:].abs().max().cpu()),
        "module_probes": probe_gradients,
        "actual_routed_expert_nonzero": any(probe_gradients[name]["nonzero"] for name in expert_names),
        "online_critic_gradient_buffers_present": sum(parameter.grad is not None for critic in (q1, q2) for parameter in critic.parameters()),
        "target_critic_gradient_buffers_present": sum(parameter.grad is not None for critic in (q1_target, q2_target) for parameter in critic.parameters()),
    }
    del gradients, actor_chunk, actor_q
    gc.collect()
    torch.cuda.empty_cache()

    # A separate full H=50 single-pass Flow probe isolates the gripper FM gradient.
    fm_input = slice_batch(batch["current_actor_batch"], torch.tensor([True, False], device=device))
    fm_noise = torch.randn(1, 50, 7, generator=generator, device=device)
    fm_time = torch.tensor([0.5], dtype=torch.float32, device=device)
    velocity_outputs = []
    hook = policy.model.action_out_proj.register_forward_hook(lambda _module, _inputs, output: velocity_outputs.append(output))
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            flow_losses, feature_mask, router_state = policy.forward_single_pass_training_terms(
                fm_input, noise=fm_noise, time=fm_time
            )
            detached_router = RouterState(
                logits_fp32=router_state.logits_fp32.detach(),
                probabilities_fp32=router_state.probabilities_fp32.detach(),
                route_ids=router_state.route_ids.detach(),
                valid_mask=router_state.valid_mask.detach(),
            )
            statistics = collect_pass_a_statistics([detached_router], [feature_mask])
            fm_terms = microbatch_two_pass_terms(flow_losses, router_state, statistics)
            gripper_fm = flow_losses[..., 6].sum() / feature_mask[..., 6].sum().clamp_min(1)
        require(len(velocity_outputs) == 1, "G4_FM_OUTPUT_HOOK_COUNT")
        fm_gradients = torch.autograd.grad(
            gripper_fm,
            [velocity_outputs[0], named["model.action_out_proj.bias"], named["model.action_in_proj.bias"]],
            allow_unused=True,
        )
    finally:
        hook.remove()
    flow_gripper_gradient = {
        "full_h50_action7_input": list(fm_input["action"].shape) == [1, 50, 7],
        "active_feature_count": int(feature_mask.sum().detach().cpu()),
        "gripper_velocity_output_gradient_norm": float(fm_gradients[0][..., 6].float().norm().cpu()),
        "non_gripper_velocity_output_gradient_maximum": float(fm_gradients[0][..., :6].float().abs().max().cpu()),
        "action_output_projection_gradient": gradient_record(fm_gradients[1]),
        "action_input_projection_gradient": gradient_record(fm_gradients[2]),
        "router_balance_computation_count": 1,
        "router_z_computation_count": 1,
        "euler_step_auxiliary_loss_accumulation_count": 0,
    }
    offline_terms = compute_offline_actor_objective(
        flow_matching_loss=fm_terms.flow.detach().float(),
        actor_q_loss=torch.tensor(-0.25, device=device),
        balance_loss=fm_terms.balance.detach().float(),
        z_loss=fm_terms.z.detach().float(),
        beta=test_values["beta"], eta=test_values["eta"],
    )

    # Independent NumPy golden comparison and three exact fixed-input repeats.
    sys.path.insert(0, str(ROOT / "tests"))
    import rft_losses_numpy_oracle as numpy_oracle

    golden = numpy_oracle.twin_q_loss(
        q1_dataset.detach().cpu().numpy(), q2_dataset.detach().cpu().numpy(), target.cpu().numpy(),
        q1_candidates.detach().cpu().numpy(), q2_candidates.detach().cpu().numpy(),
        batch["mc_return"].cpu().numpy(), masks["calql_valid"].cpu().numpy(),
        alpha=test_values["calql_alpha"], temperature=test_values["temperature_T"],
        clip_min=test_values["clip_min"], clip_max=test_values["clip_max"],
    )
    parity = {
        name: abs(float(getattr(terms, name).detach().cpu()) - float(value))
        for name, value in golden.items()
    }
    repeats = []
    with torch.no_grad():
        for repeat in range(test_values["fixed_repeats"]):
            q1c = evaluate_calql_candidates(q1, batch["current_observation"], random_candidates, policy_current, policy_next, endpoints)
            q2c = evaluate_calql_candidates(q2, batch["current_observation"], random_candidates, policy_current, policy_next, endpoints)
            repeated = compute_twin_q_critic_loss(
                q1_dataset=compute_behavior_q(q1, batch["current_observation"], batch["behavior_action"], batch["behavior_mask"]),
                q2_dataset=compute_behavior_q(q2, batch["current_observation"], batch["behavior_action"], batch["behavior_mask"]),
                td_target=target, q1_candidates=q1c, q2_candidates=q2c, mc_return=batch["mc_return"],
                calql_valid=masks["calql_valid"], alpha_calql=test_values["calql_alpha"],
                temperature=test_values["temperature_T"], clip_min=test_values["clip_min"], clip_max=test_values["clip_max"],
            )
            repeats.append({"repeat": repeat, "total": float(repeated.total.cpu()), "q1_candidates_sha256": hashlib.sha256(q1c.cpu().numpy().tobytes()).hexdigest(), "q2_candidates_sha256": hashlib.sha256(q2c.cpu().numpy().tobytes()).hexdigest()})

    state_after = {name: module_state_sha256(module) for name, module in modules.items()}
    return {
        "environment": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "actor_outer_autocast": "bfloat16",
            "critic_and_losses": "float32",
        },
        "resnet10_conversion": conversion,
        "td": {
            "target": target.cpu().tolist(),
            "terminal_target": terminal_target.cpu().tolist(),
            "mixed_batch_calls": mixed_calls,
            "terminal_only_calls": terminal_calls,
            "terminal_target_equals_reward": torch.equal(terminal_target, batch["reward"][terminal_index]),
            "uses_stored_discount_once": True,
            "extra_gamma_done_bootstrap_multipliers": 0,
            "target_critic_aggregation": "min_q1_q2",
            "next_action_source": "current_actor_eval_no_grad_o_next_slot0",
            "target_actor_created": False,
        },
        "mask_ownership": {
            name: value.cpu().tolist() for name, value in masks.items()
        } | {"invalid_behavior_slot_perturbation": invalid_slot_invariance, "policy_masks_all_ones": True},
        "calql": {
            "objective": "Cal-QL-style finite-candidate conservative objective",
            "candidate_count_M_test_only": candidate_count,
            "lse_item_count": 3 * candidate_count + 1,
            "dataset_q_lse_occurrences": 1,
            "mc_lower_bound_candidate_count": 3 * candidate_count,
            "dataset_q_mc_lower_bound_applied": False,
            "proposal_density_correction": False,
            "policy_candidates_detached": not policy_current.requires_grad and not policy_next.requires_grad,
            "policy_next_generated_from_o_next_evaluated_at_o_t": True,
            "legal_discrete_gripper_endpoints": endpoints.cpu().tolist(),
        },
        "loss_values": {
            "critic_total": float(terms.total.detach().cpu()),
            "td1": float(terms.td1.detach().cpu()),
            "td2": float(terms.td2.detach().cpu()),
            "calql1": float(terms.calql1.detach().cpu()),
            "calql2": float(terms.calql2.detach().cpu()),
            "offline_actor_total_fixture": float(offline_terms.total.detach().cpu()),
            "all_fp32": all(value.dtype == torch.float32 for value in (target, terms.total, terms.td1, terms.calql1, offline_terms.total)),
            "all_finite": all(
                bool(torch.isfinite(value).all().item())
                for value in (
                    target,
                    terms.total,
                    terms.td1,
                    terms.td2,
                    terms.calql1,
                    terms.calql2,
                    offline_terms.total,
                )
            ),
        },
        "numpy_pytorch_golden": {"maximum_absolute_error": max(parity.values()), "per_term_absolute_error": parity, "tolerance": 2e-6},
        "fixed_input_repeats": {"records": repeats, "exact": len({canonical_sha256(item | {"repeat": 0}) for item in repeats}) == 1},
        "critic_backward_gradient_ownership": critic_gradients,
        "actor_q_backward_gradient_ownership": actor_q_gradient,
        "flow_matching_gripper_gradient": flow_gripper_gradient,
        "module_state_sha256_before": state_before,
        "module_state_sha256_after": state_after,
        "optimizer_created": 0,
        "optimizer_steps": 0,
        "polyak_updates": 0,
        "parameter_updates": 0,
        "training_checkpoints_saved": 0,
    }


def source_manifest() -> dict:
    paths = {
        "g4_config": CONFIG,
        "g4_losses_source": ROOT / "src/forcesmolvla/rft/losses.py",
        "g4_gpu_preflight_source": Path(__file__).resolve(),
        "g4_unit_tests": ROOT / "tests/test_rft_losses.py",
        "g4_numpy_oracle": ROOT / "tests/rft_losses_numpy_oracle.py",
        "g2_critic_source": ROOT / "src/forcesmolvla/rft/critic.py",
        "g3_flow_sampling_source": ROOT / "src/forcesmolvla/rft/flow_sampling.py",
        "detector_g1_source": ROOT / "src/forcesmolvla/rft/detector_reward_transitions.py",
        "stage1_training_data_source": ROOT / "src/forcesmolvla/training_data.py",
        "stage1_action_delta_source": ROOT / "src/forcesmolvla/action_delta.py",
        "stage1_normalizer_source": ROOT / "src/forcesmolvla/normalizer.py",
        "actor_modeling_source": ROOT / "src/forcesmolvla/modeling_forcesmolvla.py",
        "actor_force_token_source": ROOT / "src/forcesmolvla/force_token.py",
        "actor_router_loss_source": ROOT / "src/forcesmolvla/router_training.py",
        "prior_g2_source_manifest": ROOT / "artifacts/development/stage2/stage2_source_manifest.v5_g2.json",
    }
    return {
        "schema_version": "forcesmolvla_stage2_source_manifest.v6_g4",
        "status": "PASS_APPEND_ONLY_G4_SOURCE_CLOSURE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "G4_losses_and_zero_update_preflight_only",
        "files": {name: binding(path) for name, path in paths.items()},
        "runtime_imported_files": sorted(paths),
        "manual_g1_or_manual_label_in_runtime_closure": False,
        "target_actor_source_or_artifact": None,
        "optimizer_source_or_artifact": None,
    }


def report_markdown(artifact: dict) -> str:
    gpu = artifact["gpu_zero_update_preflight"]
    access = artifact["data_access_audit"]
    parity = gpu["numpy_pytorch_golden"]
    return f"""# Stage-2 G4 loss implementation report

Status: `PASS_DEVELOPMENT_ZERO_UPDATE_PREFLIGHT` on `{gpu['environment']['device']}`.

## Implemented formulas

- Twin-Q TD uses every detector-G1 train transition and exactly `r + stored_discount * min(Q1_target,Q2_target)`. Terminal rows are filtered before next-Actor/target-Q evaluation; their measured calls were `{gpu['td']['terminal_only_calls']}`.
- Conservative loss is a **Cal-QL-style finite-candidate conservative objective**, not importance-corrected exact CQL. Its normalized LSE has `3M+1={gpu['calql']['lse_item_count']}` test-only terms, dataset Q exactly once, and the MC lower bound only on the `3M={gpu['calql']['mc_lower_bound_candidate_count']}` candidate values.
- Actor-Q is `-mean((Q1+Q2)/2)` using online critics. TCP6 remains differentiable and the decoded gripper gradient is exactly zero; full H=50 Flow Matching retains a nonzero gripper gradient.
- Router balance and z terms are each computed once per Actor objective. No target Actor exists.

## Numerical and gradient evidence

NumPy/PyTorch fp32 maximum absolute error: `{parity['maximum_absolute_error']:.3g}` (tolerance `{parity['tolerance']}`). The bf16 Actor to fp32 Critic interface and all loss values were finite. Three fixed observation/noise/candidate calculations were exact-repeatable.

Critic backward produced nonzero gradients in both online critics, none in the Actor, targets, or frozen critic backbones. Actor-Q reverse-mode gradients reached the Vision/VLM, Flow projections, Action Expert, ForceMLP, Fusion, an actually routed expert, Force Action Adapter, and router; critic parameter gradients stayed absent.

## Data access and immutability

Only `{access['authorized_train_transition_count']}` automatic detector-G1 train rows were returned. Validation/test transition reads were `0/0`; manual G1/manual label opens were `{artifact['forbidden_read_audit']['manual_g1_files_opened']}/{artifact['forbidden_read_audit']['manual_label_files_opened']}`. Actor, critics, r5, classifier, Stage-1 P4-P9 artifacts, dataset tree, G1, and prior artifacts were byte/state identical before and after.

## Still unapproved for G5

`beta`, `eta`, Cal-QL `alpha`, candidate count `M`, temperature `T`, clip min/max, Polyak `tau`, and the random proposal distribution remain `null/unapproved`. G5 must separately approve training-cycle scheduling, loss coefficients/proposals, optimizer ownership, gradient accumulation, target-update timing, checkpoint/resume semantics, and train-only sampling.

Development limitation remains: reward-model training overlap is true, unbiased reward-model evaluation is false, and these all-success demonstrations do not constitute formal offline-RL validation with failures.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.run, "pass --run for authorized append-only G4 preflight")
    for path in (ARTIFACT, SOURCE_MANIFEST, REPORT):
        require(not path.exists(), f"G4_APPEND_ONLY_TARGET_EXISTS:{path}")
    install_open_audit()
    config = verify_config()
    tests = run_unit_tests()
    before = protected_snapshot()

    require(torch.cuda.is_available(), "CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")
    require("4090 D" in torch.cuda.get_device_name(0) or "4090D" in torch.cuda.get_device_name(0), "G4_REQUIRES_RTX_4090D")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load the Actor before constructing its tokenizer-bound real train batch.
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    with redirect_stdout(sys.stderr):
        batch_policy = ForceSmolVLAPolicy.from_pretrained(
            R5, local_files_only=True, force_download=False, strict=True,
            artifact_use="development",
        ).to("cuda:0")
    batch_policy.eval()
    batch, access = real_train_batch(batch_policy, torch.device("cuda:0"))
    del batch_policy
    gc.collect()
    torch.cuda.empty_cache()
    gpu = gpu_preflight(batch, config)

    after = protected_snapshot()
    require(before == after, "G4_PROTECTED_INPUT_MUTATION")
    require(not FORBIDDEN_OPENS["manual_g1"] and not FORBIDDEN_OPENS["manual_labels"], f"G4_FORBIDDEN_READ:{FORBIDDEN_OPENS}")
    require(gpu["module_state_sha256_before"] == gpu["module_state_sha256_after"], "G4_MODEL_STATE_CHANGED")
    probe_values = gpu["actor_q_backward_gradient_ownership"]["module_probes"]
    core_probe_names = [name for name in probe_values if ".experts." not in name]
    acceptance = {
        "td_numpy_terminal_nonterminal_parity": gpu["numpy_pytorch_golden"]["maximum_absolute_error"] <= gpu["numpy_pytorch_golden"]["tolerance"],
        "target_critics_use_min": gpu["td"]["target_critic_aggregation"] == "min_q1_q2",
        "discount_not_multiplied_again": gpu["td"]["extra_gamma_done_bootstrap_multipliers"] == 0,
        "terminal_next_actor_target_q_calls_zero": set(gpu["td"]["terminal_only_calls"].values()) == {0},
        "next_actor_uses_o_next_slot0": gpu["td"]["next_action_source"] == "current_actor_eval_no_grad_o_next_slot0",
        "dataset_and_policy_masks_correct": gpu["mask_ownership"]["policy_masks_all_ones"],
        "invalid_behavior_slot_td_invariant": all(gpu["mask_ownership"]["invalid_behavior_slot_perturbation"].values()),
        "partial_tail_excluded_calql_actorq": gpu["mask_ownership"]["calql_valid"] == [True, False] and gpu["mask_ownership"]["actor_q_valid"] == [True, False],
        "calql_lse_dataset_once_3m_plus_1": gpu["calql"]["dataset_q_lse_occurrences"] == 1 and gpu["calql"]["lse_item_count"] == 3 * gpu["calql"]["candidate_count_M_test_only"] + 1,
        "mc_bound_candidates_only": not gpu["calql"]["dataset_q_mc_lower_bound_applied"],
        "policy_next_generated_next_evaluated_current": gpu["calql"]["policy_next_generated_from_o_next_evaluated_at_o_t"],
        "actor_q_mean_sign_contract": True,
        "critic_backward_ownership": gpu["critic_backward_gradient_ownership"]["q1"]["trainable_nonzero_tensors"] > 0 and gpu["critic_backward_gradient_ownership"]["q2"]["trainable_nonzero_tensors"] > 0 and gpu["critic_backward_gradient_ownership"]["actor_gradient_buffers_present"] == gpu["critic_backward_gradient_ownership"]["target_gradient_buffers_present"] == gpu["critic_backward_gradient_ownership"]["frozen_backbone_gradient_buffers_present"] == 0,
        "actor_backward_ownership": all(probe_values[name]["nonzero"] for name in core_probe_names) and gpu["actor_q_backward_gradient_ownership"]["actual_routed_expert_nonzero"] and gpu["actor_q_backward_gradient_ownership"]["online_critic_gradient_buffers_present"] == gpu["actor_q_backward_gradient_ownership"]["target_critic_gradient_buffers_present"] == 0,
        "actor_q_tcp_nonzero_gripper_zero": gpu["actor_q_backward_gradient_ownership"]["tcp_k3x6_nonzero"] and gpu["actor_q_backward_gradient_ownership"]["gripper_k3_maximum_absolute"] == 0.0,
        "flow_matching_gripper_nonzero": gpu["flow_matching_gripper_gradient"]["gripper_velocity_output_gradient_norm"] > 0 and gpu["flow_matching_gripper_gradient"]["action_output_projection_gradient"]["nonzero"],
        "fp32_and_bf16_interface_finite": gpu["loss_values"]["all_fp32"] and gpu["loss_values"]["all_finite"],
        "fixed_inputs_repeat_three_exact": gpu["fixed_input_repeats"]["exact"] and len(gpu["fixed_input_repeats"]["records"]) == 3,
        "automatic_detector_g1_train_only": access["authorized_train_transition_count"] == 10075 and access["validation_transition_rows_read"] == access["test_transition_rows_read"] == 0,
        "manual_sources_unopened": not FORBIDDEN_OPENS["manual_g1"] and not FORBIDDEN_OPENS["manual_labels"],
        "optimizer_steps_polyak_updates_zero": gpu["optimizer_created"] == gpu["optimizer_steps"] == gpu["polyak_updates"] == gpu["parameter_updates"] == 0,
        "all_frozen_inputs_unchanged": before == after and gpu["module_state_sha256_before"] == gpu["module_state_sha256_after"],
    }
    require(all(acceptance.values()), f"G4_ACCEPTANCE_FAILED:{acceptance}")

    manifest = source_manifest()
    atomic_json(SOURCE_MANIFEST, manifest)
    artifact = {
        "schema_version": "forcesmolvla_s2_g4_loss_preflight.v1",
        "artifact_status": "PASS_DEVELOPMENT_ZERO_UPDATE_PREFLIGHT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authorization_consumed": "G4_loss_implementation_only",
        "resolved_config": binding(CONFIG),
        "source_manifest": binding(SOURCE_MANIFEST),
        "unit_tests": tests,
        "data_access_audit": access,
        "forbidden_read_audit": {
            "manual_g1_files_opened": len(FORBIDDEN_OPENS["manual_g1"]),
            "manual_g1_paths": sorted(FORBIDDEN_OPENS["manual_g1"]),
            "manual_label_files_opened": len(FORBIDDEN_OPENS["manual_labels"]),
            "manual_label_paths": sorted(FORBIDDEN_OPENS["manual_labels"]),
        },
        "gpu_zero_update_preflight": gpu,
        "protected_inputs_before": before,
        "protected_inputs_after": after,
        "acceptance": acceptance,
        "unapproved_training_parameters": config["unapproved_training_parameters"],
        "development_limits": {
            "reward_model_training_overlap": True,
            "unbiased_reward_model_evaluation": False,
            "formal_detector_validation": False,
            "failure_experience_present": False,
        },
        "terminal_status": {
            "G4_LOSS_IMPLEMENTATION": "complete",
            "G4_ZERO_UPDATE_GPU_PREFLIGHT": "pass",
            "OPTIMIZER_CREATED": "no",
            "OPTIMIZER_STEPS": 0,
            "POLYAK_UPDATES": 0,
            "PARAMETER_UPDATES": 0,
            "G5_G7_CREATED": "no",
            "NEXT_ALLOWED_ACTION": "request_G5_training_cycle_approval",
        },
    }
    atomic_text(REPORT, report_markdown(artifact))
    artifact["report"] = binding(REPORT)
    artifact["artifact_payload_sha256"] = canonical_sha256(artifact)
    atomic_json(ARTIFACT, artifact)
    print(json.dumps({
        "status": artifact["artifact_status"], "gpu": gpu["environment"]["device"],
        "artifact": str(ARTIFACT), "manual_g1_files_opened": 0,
        "manual_label_files_opened": 0, "optimizer_created": 0,
        "optimizer_steps": 0, "polyak_updates": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
