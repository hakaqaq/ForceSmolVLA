#!/usr/bin/env python3
"""Reusable Actor/Critic training-cycle runtime primitives."""

from __future__ import annotations

from collections import Counter, defaultdict
import gc
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/forcerft_training_cycle.development.yaml"
REWARD_TRANSITION_ROOT = ROOT / "artifacts/development/stage2/g1_frozen_detector_transition_view.v1"
MANUAL_REWARD_TRANSITION_ROOT = ROOT / "artifacts/development/stage2/g1_manual_reward_transition_view.v1"
LABELS = ROOT / "labels"
DATASET = ROOT / "datasets/task2_lerobotv3"
PARENT_ACTOR_CHECKPOINT = (
    ROOT / "outputs/task2/sft/checkpoints/forcesmolvla_sft_step_010000"
)
REWARD_BACKBONE_PARAMETERS = (
    ROOT / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_params.safe.npz"
)
REWARD_BACKBONE_MANIFEST = (
    ROOT
    / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_asset_manifest.v4.json"
)
EXPECTED_REWARD_MANIFEST_SHA256 = "96dcc37abc365c945a075086efd60198c3391ad2d5fb3f0b53ff869e565e7bd5"
EXPECTED_DATASET_TREE_SHA256 = "f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d"
FORBIDDEN_OPENS: dict[str, set[str]] = {
    "manual_reward_transitions": set(),
    "manual_labels": set(),
}


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


def install_open_audit() -> None:
    roots = {
        "manual_reward_transitions": MANUAL_REWARD_TRANSITION_ROOT.resolve(),
        "manual_labels": LABELS.resolve(),
    }

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
        value = sha256_file(path)
        digest.update(f"{relative}\0{value}\n".encode())
        size += path.stat().st_size
    return {"tree_sha256": digest.hexdigest(), "file_count": len(files), "total_file_size": size}


def protected_snapshot() -> dict:
    files = {
        "reward_manifest": REWARD_TRANSITION_ROOT / "g1_manifest.json",
        "reward_frame_scores": REWARD_TRANSITION_ROOT / "frame_scores.parquet",
        "reward_transition_index": REWARD_TRANSITION_ROOT / "transition_index.parquet",
        "critic_config": ROOT / "configs/twin_q_critic.development.yaml",
        "action_contract": ROOT / "configs/stage2_action_contract.v2.development.json",
        "training_cycle_config": CONFIG,
        "training_cycle_source": ROOT / "src/forcesmolvla/rft/training_cycle.py",
        "training_runtime_source": ROOT / "src/forcesmolvla/rft/training_cycle_runtime.py",
        "training_checkpoint_source": ROOT / "src/forcesmolvla/rft/training_checkpoint.py",
        "critic_source": ROOT / "src/forcesmolvla/rft/critic.py",
        "flow_sampling_source": ROOT / "src/forcesmolvla/rft/flow_sampling.py",
        "dataset_conversion": DATASET / "conversion_manifest.json",
        "dataset_split": DATASET / "split_manifest.json",
        "dataset_normalizer": DATASET / "normalizer_manifest.json",
        "actor_core": ROOT / "src/forcesmolvla/modeling_forcesmolvla.py",
    }
    result = {
        "files": {name: binding(path) for name, path in files.items()},
        "dataset_storage_tree": file_tree(DATASET, ("data", "videos", "meta")),
        "parent_actor_checkpoint_tree": file_tree(PARENT_ACTOR_CHECKPOINT),
    }
    require(
        result["files"]["reward_manifest"]["sha256"]
        == EXPECTED_REWARD_MANIFEST_SHA256,
        "TRAINING_CYCLE_REWARD_MANIFEST_DRIFT",
    )
    require(
        result["dataset_storage_tree"]["tree_sha256"]
        == EXPECTED_DATASET_TREE_SHA256,
        "TRAINING_CYCLE_DATASET_TREE_DRIFT",
    )
    return result


def verify_config() -> dict:
    config = yaml.safe_load(CONFIG.read_text())
    require(config["authorization"] == "G5_development_single_cycle_only", "G5_AUTHORIZATION_DRIFT")
    require(config["cycle"] == {
        "training_cycles": 1,
        "critic_updates_per_cycle": 2,
        "actor_updates_per_cycle": 1,
        "critic_warmup_bypassed_for_single_cycle_preflight": True,
        "eta_warmup_bypassed_for_single_cycle_preflight": True,
        "second_cycle_allowed": False,
    }, "G5_CYCLE_CONFIG_DRIFT")
    require(config["batching"] == {
        "critic_batch_size": 16, "calql_batch_size": 16,
        "actor_microbatch_size": 1, "actor_gradient_accumulation": 4,
        "actor_effective_batch_size": 4, "num_workers": 0,
        "data_augmentation": False,
    }, "G5_BATCH_CONFIG_DRIFT")
    require(
        config["loss"]["beta_flow"] == 1.0
        and config["loss"]["eta_actor_q"] == 0.01
        and config["loss"]["alpha_calql"] == 0.1
        and config["loss"]["cql_candidates_per_source_M"] == 2
        and config["loss"]["cql_temperature"] == 1.0
        and config["loss"]["cql_clipping_enabled"] is False
        and math.isinf(config["loss"]["cql_clip_min"])
        and math.isinf(config["loss"]["cql_clip_max"]),
        "G5_LOSS_CONFIG_DRIFT",
    )
    require(config["targets"]["polyak_tau"] == 0.005 and config["targets"]["actor_target_updates"] == 0, "G5_TARGET_CONFIG_DRIFT")
    require(not config["transition_contract"]["target_actor_exists"], "G5_TARGET_ACTOR_FORBIDDEN")
    return config


def decode_rgb(payload: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(value.shape == (480, 640, 3), "G5_IMAGE_SHAPE_DRIFT")
    return np.ascontiguousarray(value.transpose(2, 0, 1))


def stats(value: torch.Tensor) -> dict:
    value = value.detach().float()
    require(bool(torch.isfinite(value).all()), "G5_REPORTED_TENSOR_NONFINITE")
    return {"mean": float(value.mean().cpu()), "minimum": float(value.min().cpu()), "maximum": float(value.max().cpu())}


def row_identity(row: dict) -> str:
    return f"{row['transition_index']}|{row['episode_id']}|{row['anchor_frame']}|{row['next_frame']}"


class TrainData:
    """The sole automatic detector-G1 train view and its frozen populations."""

    def __init__(self) -> None:
        from forcesmolvla.rft.losses import (
            load_authorized_reward_train_transitions,
            validate_mc_return_recurrence,
        )
        from forcesmolvla.training_data import load_runtime_artifacts

        table = load_authorized_reward_train_transitions(REWARD_TRANSITION_ROOT)
        self.rows = table.to_pylist()
        require(len(self.rows) == 10075 and {row["split"] for row in self.rows} == {"train"}, "G5_TRAIN_ROWS_INVALID")
        self.mc_recurrence = validate_mc_return_recurrence(self.rows)
        self.td_population = tuple(range(len(self.rows)))
        self.calql_population = tuple(
            index for index, row in enumerate(self.rows)
            if all(row["executed_action_mask"]) and not row["terminated"] and math.isfinite(row["mc_return"])
        )
        self.actor_population = tuple(
            index for index, row in enumerate(self.rows) if all(row["executed_action_mask"])
        )
        self.proposal_population = self.actor_population
        conversion = json.loads((DATASET / "conversion_manifest.json").read_text())
        self.tasks = {item["raw_episode_id"]: item["task"] for item in conversion["episodes"]}
        reward_manifest = json.loads(
            (REWARD_TRANSITION_ROOT / "g1_manifest.json").read_text()
        )
        self.frame_counts = {
            item["episode_id"]: int(item["frame_count"])
            for item in reward_manifest["episode_detection_results"]
        }
        self.runtime = load_runtime_artifacts(
            DATASET,
            calibration_bundle_path=ROOT / "configs/calibration_bundle.development.json",
            wrench_geometry_spec_path=ROOT / "configs/wrench_geometry_spec.development.json",
            action_delta_spec_path=ROOT / "artifacts/development/action_delta_spec.json",
            expected_repo_id=conversion["repo_id"],
        )
        actions = []
        identities = []
        for index in self.proposal_population:
            row = self.rows[index]
            flat = np.asarray(row["normalized_delta_action_exec_flat"], dtype=np.float32)
            require(flat.shape == (21,), "G5_FULL_MACRO_POPULATION_SHAPE_INVALID")
            actions.append(flat.reshape(3, 7))
            identities.append(row_identity(row))
        self.proposal_actions = torch.from_numpy(np.stack(actions))
        physical_gripper = (
            self.proposal_actions[..., 6].numpy() * self.runtime.normalizer.delta_action7.std[6]
            + self.runtime.normalizer.delta_action7.mean[6]
        )
        legal_gripper = np.isclose(physical_gripper, 0.0, atol=1e-6) | np.isclose(
            physical_gripper, 0.085, atol=1e-6
        )
        require(bool(legal_gripper.all()), "G5_PROPOSAL_GRIPPER_ENDPOINT_INVALID")
        identity_bytes = "\n".join(identities).encode()
        self.population_manifest = {
            "proposal_source": "automatic_detector_g1_train_full_macro_valid_behavior_actions",
            "sampling": "whole_k3x7_macro_with_replacement",
            "population_count": len(actions),
            "row_identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
            "source_tensor_sha256": hashlib.sha256(self.proposal_actions.numpy().tobytes()).hexdigest(),
            "tensor_shape": list(self.proposal_actions.shape),
            "per_dimension_or_slot_mixing": False,
            "absolute_or_32d_action_created": False,
            "gripper_legal_discrete_endpoint_only": True,
            "status": "G5_test_only_not_long_train_or_paper_proposal",
        }

    def canonicalize_proposal_gripper_for_runtime(self, device: torch.device) -> None:
        """Resolve the frozen CPU/GPU normalizer's one-ULP endpoint difference."""

        source = self.proposal_actions.to(device)
        mean = torch.tensor(
            self.runtime.normalizer.delta_action7.mean, dtype=torch.float32, device=device
        )
        std = torch.tensor(
            self.runtime.normalizer.delta_action7.std, dtype=torch.float32, device=device
        )
        endpoints = torch.stack(
            ((source.new_tensor(0.0) - mean[6]) / std[6],
             (source.new_tensor(0.085) - mean[6]) / std[6])
        )
        values = source[..., 6]
        distances, endpoint_indices = (values[..., None] - endpoints).abs().min(dim=-1)
        tolerance = (
            torch.finfo(torch.float32).eps
            * torch.maximum(torch.ones_like(endpoints), endpoints.abs()).max()
        )
        require(
            bool((distances <= tolerance).all()),
            "G5_PROPOSAL_GRIPPER_NOT_ONE_ULP_EQUIVALENT_TO_RUNTIME_ENDPOINT",
        )
        canonical = source.clone()
        canonical[..., 6] = endpoints[endpoint_indices]
        correction = (canonical - source).abs()
        self.proposal_actions = canonical.cpu()
        self.population_manifest.update({
            "tensor_sha256": hashlib.sha256(
                self.proposal_actions.numpy().tobytes()
            ).hexdigest(),
            "runtime_endpoint_canonicalization": {
                "reason": "CPU/GPU_frozen_normalizer_open_endpoint_one_ULP_representation",
                "semantic_physical_endpoint_changed": False,
                "tcp_value_changed": False,
                "gripper_values_changed": int((correction[..., 6] != 0).sum().cpu()),
                "maximum_normalized_abs_correction": float(correction.max().cpu()),
                "maximum_allowed_abs_correction": float(tolerance.cpu()),
                "runtime_normalized_endpoints": endpoints.cpu().tolist(),
            },
        })

    def population_audit(self) -> dict:
        return {
            "train_transition_count": len(self.rows),
            "td_population_count": len(self.td_population),
            "calql_population_count": len(self.calql_population),
            "actor_population_count": len(self.actor_population),
            "proposal_population_count": len(self.proposal_population),
            "mc_return_recurrence": self.mc_recurrence,
            "validation_transition_reads": 0,
            "test_transition_reads": 0,
            "manual_reward_transition_reads": 0,
            "manual_label_reads": 0,
        }

    def identity_records(self, indices: list[int]) -> list[dict]:
        result = []
        for index in indices:
            row = self.rows[index]
            result.append({
                "transition_index": row["transition_index"],
                "episode_id": row["episode_id"],
                "anchor_frame": row["anchor_frame"],
                "next_frame": row["next_frame"],
                "row_identity": row_identity(row),
                "executed_steps": row["executed_steps"],
                "reward": row["reward"],
                "terminated": row["terminated"],
            })
        require(len({item["row_identity"] for item in result}) == len(result), "G5_BATCH_IDENTITY_DUPLICATE")
        return result

    def _raw_rows(self, requested: dict[str, set[int]], *, include_actions: bool) -> dict[tuple[str, int], dict]:
        import pyarrow.parquet as pq
        from forcesmolvla.rft.offline_transitions import PROVENANCE_KEYS

        base = [
            "observation.images.camera1", "observation.images.camera2",
            "observation.state", "observation.wrench", "frame_index",
            "episode_index", "index",
        ]
        columns = base + (["action", *PROVENANCE_KEYS] if include_actions else [])
        result = {}
        for relative, indices in requested.items():
            table = pq.read_table(DATASET / relative, columns=columns)
            for index in sorted(indices):
                result[(relative, index)] = table.slice(index, 1).to_pylist()[0]
            del table
        return result

    def build_batch(
        self,
        indices: list[int],
        policy,
        device: torch.device,
        *,
        canonical_task_feature: torch.Tensor,
        include_flow_actions: bool = False,
    ) -> dict:
        from forcesmolvla.rft.losses import CriticObservation
        from forcesmolvla.rft.offline_transitions import PROVENANCE_KEYS
        from forcesmolvla.training_data import prepare_training_sample
        from forcesmolvla.rft.batch import build_actor_batch

        require(len(indices) == len(set(indices)), "G5_BATCH_INDICES_NOT_UNIQUE")
        requested: dict[str, set[int]] = defaultdict(set)
        for index in indices:
            row = self.rows[index]
            for key in ("observation_row_reference", "next_observation_row_reference"):
                ref = row[key]
                requested[ref["data_relative_path"]].add(ref["row_index"])
            if include_flow_actions:
                ref = row["observation_row_reference"]
                stop = min(ref["row_index"] + 50, self.frame_counts[row["episode_id"]])
                requested[ref["data_relative_path"]].update(range(ref["row_index"], stop))
        raw = self._raw_rows(requested, include_actions=include_flow_actions)
        current_samples, next_samples = [], []
        behavior_actions, behavior_masks = [], []
        for index in indices:
            transition = self.rows[index]
            current_ref = transition["observation_row_reference"]
            next_ref = transition["next_observation_row_reference"]
            require(current_ref["episode_id"] == next_ref["episode_id"] == transition["episode_id"], "G5_CROSS_EPISODE")
            current = raw[(current_ref["data_relative_path"], current_ref["row_index"])]
            following = raw[(next_ref["data_relative_path"], next_ref["row_index"])]

            def observation_sample(source: dict, identity: str) -> dict:
                return {
                    "camera1": decode_rgb(source["observation.images.camera1"]["bytes"]),
                    "camera2": decode_rgb(source["observation.images.camera2"]["bytes"]),
                    "state7": self.runtime.normalizer.state7.apply(np.asarray(source["observation.state"], dtype=np.float64)).astype(np.float32),
                    "wrench6": self.runtime.normalizer.wrench6.apply(np.asarray(source["observation.wrench"], dtype=np.float64)).astype(np.float32),
                    "task": self.tasks[transition["episode_id"]],
                    "sample_identity": identity,
                }

            current_sample = observation_sample(current, f"{transition['episode_id']}/frame={transition['anchor_frame']}")
            next_sample = observation_sample(following, f"{transition['episode_id']}/next={transition['next_frame']}")
            if include_flow_actions:
                relative = current_ref["data_relative_path"]
                anchor = current_ref["row_index"]
                episode_last = self.frame_counts[transition["episode_id"]] - 1
                source_indices = np.minimum(anchor + np.arange(50), episode_last)
                absolute = np.asarray([raw[(relative, int(source))]["action"] for source in source_indices], dtype=np.float64)
                action_is_pad = anchor + np.arange(50) > episode_last
                sample = {
                    "observation.state": np.asarray(current["observation.state"], dtype=np.float64),
                    "observation.wrench": np.asarray(current["observation.wrench"], dtype=np.float64),
                    "action": absolute,
                    "action_is_pad": action_is_pad,
                    "episode_index": current["episode_index"],
                    "frame_index": current["frame_index"],
                    "task": current_sample["task"],
                    "observation.images.camera1": current_sample["camera1"],
                    "observation.images.camera2": current_sample["camera2"],
                }
                for name in PROVENANCE_KEYS:
                    sample[name] = current[name]
                prepared = prepare_training_sample(sample, self.runtime.normalizer)
                current_sample.update(
                    delta_action7=prepared["delta_action7"],
                    action_valid_mask=prepared["action_valid_mask"],
                )
            current_samples.append(current_sample)
            next_samples.append(next_sample)
            mask = np.asarray(transition["executed_action_mask"], dtype=np.bool_)
            flat = np.asarray(transition["normalized_delta_action_exec_flat"], dtype=np.float32)
            action = np.zeros((3, 7), dtype=np.float32)
            action[: int(mask.sum())] = flat.reshape(-1, 7)
            behavior_actions.append(action)
            behavior_masks.append(mask)

        require(
            canonical_task_feature.dtype == torch.float32
            and tuple(canonical_task_feature.shape) == (256,)
            and canonical_task_feature.device == device,
            "G5_CANONICAL_TASK_FEATURE_SOURCE_INVALID",
        )
        feature = canonical_task_feature.detach()[None, :].expand(len(indices), -1).clone()

        def critic_observation(samples: list[dict]) -> CriticObservation:
            return CriticObservation(
                torch.from_numpy(np.stack([item["camera1"] for item in samples])).to(device),
                torch.from_numpy(np.stack([item["camera2"] for item in samples])).to(device),
                feature.clone(),
                torch.from_numpy(np.stack([item["state7"] for item in samples])).to(device),
                torch.from_numpy(np.stack([item["wrench6"] for item in samples])).to(device),
            )

        rows = [self.rows[index] for index in indices]
        return {
            "indices": indices,
            "identities": self.identity_records(indices),
            "current_observation": critic_observation(current_samples),
            "next_observation": critic_observation(next_samples),
            "current_actor_batch": build_actor_batch(policy, current_samples, device, include_action=include_flow_actions),
            "next_actor_batch": build_actor_batch(policy, next_samples, device, include_action=False),
            "behavior_action": torch.from_numpy(np.stack(behavior_actions)).to(device),
            "behavior_mask": torch.from_numpy(np.stack(behavior_masks)).to(device),
            "reward": torch.tensor([row["reward"] for row in rows], dtype=torch.float32, device=device),
            "terminated": torch.tensor([row["terminated"] for row in rows], dtype=torch.bool, device=device),
            "bootstrap_mask": torch.tensor([row["bootstrap_mask"] for row in rows], dtype=torch.int8, device=device),
            "discount": torch.tensor([row["discount"] for row in rows], dtype=torch.float32, device=device),
            "mc_return": torch.tensor([row["mc_return"] for row in rows], dtype=torch.float32, device=device),
            "delta_mean": torch.tensor(self.runtime.normalizer.delta_action7.mean, dtype=torch.float32, device=device),
            "delta_std": torch.tensor(self.runtime.normalizer.delta_action7.std, dtype=torch.float32, device=device),
        }


def slice_actor_batch(batch: dict, start: int, stop: int) -> dict:
    size = next(value.shape[0] for value in batch.values() if isinstance(value, torch.Tensor) and value.ndim)
    result = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
            result[name] = value[start:stop]
        elif isinstance(value, (tuple, list)) and len(value) == size:
            result[name] = type(value)(value[start:stop])
        else:
            result[name] = value
    return result


def repeat_actor_batch(batch: dict, count: int, *, tag: str) -> dict:
    size = next(value.shape[0] for value in batch.values() if isinstance(value, torch.Tensor) and value.ndim)
    result = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
            result[name] = value.repeat_interleave(count, dim=0)
        elif name == "sample_identity":
            result[name] = tuple(
                f"{identity}/{tag}={candidate}"
                for identity in value
                for candidate in range(count)
            )
        else:
            result[name] = value
    return result


class FlowCounter:
    def __init__(self, *, inference_batch_size: int = 4) -> None:
        self.inference_batch_size = inference_batch_size
        self.flow_chunks_sampled = 0
        self.euler_velocity_evaluations = 0
        self.prefix_prefill_count = 0
        self.policy_action_chunks = 0
        self.by_purpose = Counter()

    def sample(self, policy, batch: dict, noise7: torch.Tensor, *, call_id: str, purpose: str) -> torch.Tensor:
        from forcesmolvla.rft.flow_sampling import sample_normalized_action_chunk_with_grad

        outputs = []
        for start in range(0, noise7.shape[0], self.inference_batch_size):
            stop = min(start + self.inference_batch_size, noise7.shape[0])
            outputs.append(
                sample_normalized_action_chunk_with_grad(
                    policy,
                    slice_actor_batch(batch, start, stop),
                    noise7[start:stop],
                    call_id=f"{call_id}/chunk={start}:{stop}",
                    purpose=purpose,
                )
            )
            self.flow_chunks_sampled += 1
            self.euler_velocity_evaluations += 10
            self.prefix_prefill_count += 1
            self.policy_action_chunks += stop - start
            self.by_purpose[purpose] += stop - start
        return torch.cat(outputs, dim=0)

    def report(self) -> dict:
        return {
            "flow_chunks_sampled": self.flow_chunks_sampled,
            "euler_velocity_evaluations": self.euler_velocity_evaluations,
            "prefix_prefill_count": self.prefix_prefill_count,
            "policy_action_chunks": self.policy_action_chunks,
            "policy_action_chunks_by_purpose": dict(sorted(self.by_purpose.items())),
        }


def sample_policy_candidates(
    policy,
    batch: dict,
    noise: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    counter: FlowCounter,
    *,
    purpose: str,
    call_id: str,
) -> torch.Tensor:
    from forcesmolvla.rft.flow_sampling import critic_action_for_q_guidance

    batch_size, candidates = noise.shape[:2]
    expanded = repeat_actor_batch(batch, candidates, tag=purpose)
    training = policy.training
    policy.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            chunk = counter.sample(
                policy,
                expanded,
                noise.reshape(batch_size * candidates, 50, 7),
                call_id=call_id,
                purpose=purpose,
            )
            action = critic_action_for_q_guidance(
                chunk, delta_action_mean7=mean, delta_action_std7=std
            )
    finally:
        policy.train(training)
    return action.detach().float().reshape(batch_size, candidates, 3, 7)


def named_generator(device: str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def parameter_group_gradient_norm(policy, prefixes: tuple[str, ...]) -> float:
    values = [
        parameter.grad.detach().float()
        for name, parameter in policy.named_parameters()
        if name.startswith(prefixes) and parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack([value.square().sum() for value in values]).sum()).cpu())


def actor_module_gradient_norms(policy) -> dict:
    groups = {
        "vision_vlm": ("model.vlm_with_expert.",),
        "action_io": ("model.action_in_proj.", "model.action_out_proj."),
        "action_expert": ("model.vlm_with_expert.lm_expert.",),
        "force_mlp": ("model.force_branch.force_mlp.",),
        "fusion": (
            "model.force_branch.segment_embedding.",
            "model.force_branch.fusion_position_embedding.",
            "model.force_branch.fusion_blocks.",
            "model.force_branch.guidance_projection.",
        ),
        "moe_experts": ("model.force_branch.refiner.experts.",),
        "force_action_adapter": ("model.force_adapter.",),
        "router": ("model.force_branch.refiner.router.",),
    }
    return {name: parameter_group_gradient_norm(policy, prefixes) for name, prefixes in groups.items()}


def critic_update(
    *,
    step: int,
    policy,
    q1,
    q2,
    q1_target,
    q2_target,
    optimizer,
    scheduler,
    td_batch: dict,
    calql_batch: dict,
    train_data: TrainData,
    proposal_sampler,
    generators: dict[str, torch.Generator],
    flow_counter: FlowCounter,
    config: dict,
) -> dict:
    from forcesmolvla.rft.losses import (
        compute_behavior_q,
        compute_calql_penalty,
        compute_td_target_from_current_actor,
        evaluate_calql_candidates,
    )
    from forcesmolvla.rft.training_cycle import (
        calql_unclipped_details,
        global_gradient_norm,
        gradients_finite,
        module_state_sha256,
        polyak_update_verified,
    )

    device = td_batch["reward"].device
    batch_size = config["batching"]["critic_batch_size"]
    candidates = config["loss"]["cql_candidates_per_source_M"]
    alpha = config["loss"]["alpha_calql"]
    temperature = config["loss"]["cql_temperature"]
    finite_limit = torch.finfo(torch.float32).max
    optimizer.zero_grad(set_to_none=True)
    actor_before = module_state_sha256(policy)
    target_before = {"q1": module_state_sha256(q1_target), "q2": module_state_sha256(q2_target)}
    online_before = {"q1": module_state_sha256(q1), "q2": module_state_sha256(q2)}
    torch.cuda.synchronize()
    candidate_started = time.perf_counter()

    td_noise = torch.randn(
        batch_size, 50, 7, generator=generators["td_next_action_flow_noise"],
        dtype=torch.float32, device=device,
    )

    def assert_target_task_binding(label: str) -> None:
        nonterminal = ~td_batch["terminated"]
        task = td_batch["next_observation"].task_feature[nonterminal]
        for target_name, target in (("q1_target", q1_target), ("q2_target", q2_target)):
            expected = target.canonical_task_feature[None, :].expand(task.shape[0], -1)
            maximum_error = float((task - expected).abs().max().cpu())
            require(
                torch.equal(task, expected),
                f"G5_TARGET_TASK_BINDING_DRIFT:{label}:{target_name}:shape={tuple(task.shape)}:max_abs={maximum_error}",
            )

    assert_target_task_binding("before_actor_flow")

    def counted_sample(actor, batch, noise, *, call_id, purpose):
        result = flow_counter.sample(actor, batch, noise, call_id=call_id, purpose=purpose)
        assert_target_task_binding("after_actor_flow")
        return result

    prehook_checks = []

    def target_pre_hook(module, args):
        task = args[2]
        expected = module.canonical_task_feature[None, :].expand(task.shape[0], -1)
        maximum_error = float((task - expected).abs().max().cpu())
        prehook_checks.append({
            "shape": list(task.shape),
            "dtype": str(task.dtype),
            "maximum_abs_error": maximum_error,
            "exact": bool(torch.equal(task, expected)),
        })
        require(prehook_checks[-1]["exact"], f"G5_TARGET_FORWARD_PREHOOK_TASK_DRIFT:{prehook_checks[-1]}")

    target_hook = q1_target.register_forward_pre_hook(target_pre_hook)

    try:
        td_target = compute_td_target_from_current_actor(
            reward=td_batch["reward"], discount=td_batch["discount"],
            terminated=td_batch["terminated"], bootstrap_mask=td_batch["bootstrap_mask"],
            next_observation=td_batch["next_observation"],
            next_actor_batch=td_batch["next_actor_batch"], next_noise7=td_noise,
            actor=policy, q1_target=q1_target, q2_target=q2_target,
            delta_action_mean7=td_batch["delta_mean"], delta_action_std7=td_batch["delta_std"],
            call_id=f"g5-critic-{step}-td", sample_action_fn=counted_sample,
        )
    finally:
        target_hook.remove()
    require(prehook_checks and prehook_checks[0]["exact"], "G5_TARGET_PREHOOK_NOT_EXECUTED")
    current_noise = torch.randn(
        batch_size, candidates, 50, 7,
        generator=generators["calql_current_policy_flow_noise"], device=device,
    )
    next_noise = torch.randn(
        batch_size, candidates, 50, 7,
        generator=generators["calql_next_policy_flow_noise"], device=device,
    )
    policy_current = sample_policy_candidates(
        policy, calql_batch["current_actor_batch"], current_noise,
        calql_batch["delta_mean"], calql_batch["delta_std"], flow_counter,
        purpose="cql_current", call_id=f"g5-critic-{step}-cql-current",
    )
    policy_next = sample_policy_candidates(
        policy, calql_batch["next_actor_batch"], next_noise,
        calql_batch["delta_mean"], calql_batch["delta_std"], flow_counter,
        purpose="cql_next", call_id=f"g5-critic-{step}-cql-next",
    )
    proposal_indices = proposal_sampler.draw(batch_size * candidates)
    random_candidates = train_data.proposal_actions[proposal_indices].to(device).reshape(
        batch_size, candidates, 3, 7
    )
    torch.cuda.synchronize()
    candidate_latency = time.perf_counter() - candidate_started

    endpoint = torch.stack(
        (
            (torch.tensor(0.0, device=device) - calql_batch["delta_mean"][6]) / calql_batch["delta_std"][6],
            (torch.tensor(0.085, device=device) - calql_batch["delta_mean"][6]) / calql_batch["delta_std"][6],
        )
    ).float()
    torch.cuda.synchronize()
    critic_started = time.perf_counter()
    q1_td = compute_behavior_q(q1, td_batch["current_observation"], td_batch["behavior_action"], td_batch["behavior_mask"])
    q2_td = compute_behavior_q(q2, td_batch["current_observation"], td_batch["behavior_action"], td_batch["behavior_mask"])
    q1_data = compute_behavior_q(q1, calql_batch["current_observation"], calql_batch["behavior_action"], calql_batch["behavior_mask"])
    q2_data = compute_behavior_q(q2, calql_batch["current_observation"], calql_batch["behavior_action"], calql_batch["behavior_mask"])
    q1_candidates = evaluate_calql_candidates(q1, calql_batch["current_observation"], random_candidates, policy_current, policy_next, endpoint)
    q2_candidates = evaluate_calql_candidates(q2, calql_batch["current_observation"], random_candidates, policy_current, policy_next, endpoint)
    td1 = torch.square(q1_td - td_target).mean()
    td2 = torch.square(q2_td - td_target).mean()
    valid = torch.ones(batch_size, dtype=torch.bool, device=device)
    calql1 = compute_calql_penalty(
        q1_data, q1_candidates, calql_batch["mc_return"], valid,
        temperature=temperature, clip_min=-finite_limit, clip_max=finite_limit,
    )
    calql2 = compute_calql_penalty(
        q2_data, q2_candidates, calql_batch["mc_return"], valid,
        temperature=temperature, clip_min=-finite_limit, clip_max=finite_limit,
    )
    detail1 = calql_unclipped_details(q1_data, q1_candidates, calql_batch["mc_return"], temperature=temperature)
    detail2 = calql_unclipped_details(q2_data, q2_candidates, calql_batch["mc_return"], temperature=temperature)
    require(torch.equal(calql1, detail1["difference"].mean()) and torch.equal(calql2, detail2["difference"].mean()), "G5_UNCLIPPED_CALQL_PARITY_FAILED")
    q1_loss = td1 + alpha * calql1
    q2_loss = td2 + alpha * calql2
    loss = (q1_loss + q2_loss) / 2.0
    tensors = [
        q1_td, q2_td, q1_data, q2_data, q1_candidates, q2_candidates,
        td_target, calql_batch["mc_return"], loss,
    ]
    require(all(bool(torch.isfinite(value).all()) and value.dtype == torch.float32 for value in tensors), "G5_CRITIC_FORWARD_NONFINITE_OR_NONFP32")
    loss.backward()
    trainable = [
        parameter for critic in (q1, q2) for parameter in critic.parameters()
        if parameter.requires_grad
    ]
    require(
        all(parameter.grad is None for parameter in policy.parameters())
        and all(parameter.grad is None for target in (q1_target, q2_target) for parameter in target.parameters())
        and all(
            parameter.grad is None
            for critic in (q1, q2)
            for backbone in (critic.camera1_backbone, critic.camera2_backbone)
            for parameter in backbone.parameters()
        ),
        "G5_CRITIC_GRADIENT_OWNERSHIP_FAILED",
    )
    require(gradients_finite(trainable), "G5_CRITIC_GRADIENT_NONFINITE")
    preclip = global_gradient_norm(trainable)
    torch.nn.utils.clip_grad_norm_(trainable, config["optimizers"]["critic"]["grad_clip_norm"])
    postclip = global_gradient_norm(trainable)
    require(gradients_finite(trainable), "G5_CRITIC_POSTCLIP_GRADIENT_NONFINITE")
    optimizer.step()
    online_after_step = {"q1": module_state_sha256(q1), "q2": module_state_sha256(q2)}
    require(online_after_step["q1"] != online_before["q1"] and online_after_step["q2"] != online_before["q2"], "G5_CRITIC_OPTIMIZER_DID_NOT_UPDATE_BOTH_Q")
    polyak = {
        "q1": polyak_update_verified(q1, q1_target, tau=0.005, target_name="q1_target"),
        "q2": polyak_update_verified(q2, q2_target, tau=0.005, target_name="q2_target"),
    }
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    critic_latency = time.perf_counter() - critic_started
    require(module_state_sha256(policy) == actor_before, "G5_ACTOR_CHANGED_DURING_CRITIC_STEP")
    require(all(parameter.grad is None for parameter in trainable), "G5_CRITIC_GRADIENT_NOT_CLEARED")
    nonterminal = ~td_batch["terminated"]
    target_min = (
        (td_target[nonterminal] - td_batch["reward"][nonterminal])
        / td_batch["discount"][nonterminal]
    )
    require(target_min.numel() > 0, "G5_TD_BATCH_HAS_NO_TARGET_Q_ROWS")
    differences = torch.cat((detail1["difference"], detail2["difference"]))
    activations = torch.cat((detail1["mc_lower_bound_activation"].reshape(-1), detail2["mc_lower_bound_activation"].reshape(-1)))
    return {
        "critic_substep": step,
        "td_batch": td_batch["identities"],
        "calql_batch": calql_batch["identities"],
        "proposal_population_indices": proposal_indices,
        "proposal_population_identity_sha256": hashlib.sha256(
            "\n".join(row_identity(train_data.rows[train_data.proposal_population[index]]) for index in proposal_indices).encode()
        ).hexdigest(),
        "loss": {
            "L_TD_Q1": float(td1.detach().cpu()), "L_TD_Q2": float(td2.detach().cpu()),
            "L_CalQL_Q1": float(calql1.detach().cpu()), "L_CalQL_Q2": float(calql2.detach().cpu()),
            "L_critic": float(loss.detach().cpu()),
        },
        "statistics": {
            "dataset_q": stats(torch.cat((q1_data, q2_data))),
            "candidate_q": stats(torch.cat((q1_candidates.flatten(), q2_candidates.flatten()))),
            "target_q_min": stats(target_min),
            "td_target": stats(td_target),
            "mc_return": stats(calql_batch["mc_return"]),
            "calql_unclipped_difference": stats(differences),
            "mc_lower_bound_activation_rate": float(activations.float().mean().cpu()),
        },
        "gradient": {
            "preclip_global_norm": float(preclip.cpu()),
            "postclip_global_norm": float(postclip.cpu()),
            "clip_threshold": 10.0,
            "finite_before_and_after": True,
        },
        "latency_seconds": {
            "candidate_sampling": candidate_latency,
            "critic_forward_backward_step_polyak_scheduler": critic_latency,
        },
        "terminal_rows": int(td_batch["terminated"].sum().cpu()),
        "terminal_next_actor_and_target_q_calls": 0,
        "state": {
            "actor_before_sha256": actor_before,
            "actor_after_sha256": module_state_sha256(policy),
            "online_before": online_before,
            "online_after_optimizer": online_after_step,
            "targets_before": target_before,
            "targets_after": {"q1": module_state_sha256(q1_target), "q2": module_state_sha256(q2_target)},
        },
        "polyak": polyak,
        "counters_increment": {
            "critic_optimizer_updates": 1,
            "q1_target_polyak_updates": 1,
            "q2_target_polyak_updates": 1,
            "critic_scheduler_steps": 1,
        },
    }


def flow_microbatch_terms(policy, batch: dict, noise: torch.Tensor, timestep: torch.Tensor):
    from forcesmolvla.force_token import RouterState
    from forcesmolvla.router_training import collect_pass_a_statistics, microbatch_two_pass_terms

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        losses, feature_mask, router_state = policy.forward_single_pass_training_terms(
            batch, noise=noise, time=timestep
        )
        detached = RouterState(
            logits_fp32=router_state.logits_fp32.detach(),
            probabilities_fp32=router_state.probabilities_fp32.detach(),
            route_ids=router_state.route_ids.detach(),
            valid_mask=router_state.valid_mask.detach(),
        )
        statistics = collect_pass_a_statistics([detached], [feature_mask])
        terms = microbatch_two_pass_terms(losses, router_state, statistics)
    return losses, feature_mask, terms, router_state


def actor_gradient_scale_probe(
    *,
    policy,
    q1,
    q2,
    microbatch: dict,
    generators: dict[str, torch.Generator],
    flow_counter: FlowCounter,
    eta: float,
) -> dict:
    from forcesmolvla.rft.losses import compute_actor_q_loss
    from forcesmolvla.rft.training_cycle import global_gradient_norm

    actor_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    for parameter in actor_parameters:
        parameter.grad = None
    policy.train(True)
    fm_noise = torch.randn(
        1, 50, 7, generator=generators["flow_matching_noise"],
        dtype=torch.float32, device=microbatch["reward"].device,
    )
    fm_time = torch.rand(
        1, generator=generators["flow_matching_timestep"],
        dtype=torch.float32, device=microbatch["reward"].device,
    )
    losses, feature_mask, _terms, _router = flow_microbatch_terms(
        policy, microbatch["current_actor_batch"], fm_noise, fm_time
    )
    fm_loss = losses.sum() / feature_mask.sum().clamp_min(1)
    fm_loss.backward()
    fm_global = float(global_gradient_norm(actor_parameters).cpu())
    fm_groups = actor_module_gradient_norms(policy)
    for parameter in actor_parameters:
        parameter.grad = None

    policy.eval()
    q_noise = torch.randn(
        1, 50, 7, generator=generators["actor_q_flow_noise"],
        dtype=torch.float32, device=microbatch["reward"].device,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        chunk = flow_counter.sample(
            policy, microbatch["current_actor_batch"], q_noise,
            call_id="g5-gradient-scale-probe", purpose="actor_guidance",
        )
        q_loss = compute_actor_q_loss(
            q1=q1, q2=q2, current_observation=microbatch["current_observation"],
            actor_action_chunk7=chunk,
            actor_q_valid=torch.ones(1, dtype=torch.bool, device=chunk.device),
            delta_action_mean7=microbatch["delta_mean"],
            delta_action_std7=microbatch["delta_std"],
        )
    q_loss.backward()
    q_global = float(global_gradient_norm(actor_parameters).cpu())
    q_groups = actor_module_gradient_norms(policy)
    for parameter in actor_parameters:
        parameter.grad = None
    policy.train(True)
    return {
        "scope": "fixed_single_actor_microbatch_diagnostic_only",
        "beta_flow": 1.0,
        "eta_actor_q": eta,
        "unweighted_global_fm_gradient_norm": fm_global,
        "unweighted_global_q_gradient_norm": q_global,
        "weighted_eta_grad_q_over_beta_grad_fm": eta * q_global / max(fm_global, torch.finfo(torch.float32).tiny),
        "module_gradient_ratios": {
            name: {
                "fm_gradient_norm": fm_groups[name],
                "q_gradient_norm": q_groups[name],
                "weighted_eta_q_over_beta_fm": eta * q_groups[name] / max(fm_groups[name], torch.finfo(torch.float32).tiny),
            }
            for name in fm_groups
        },
    }


def actor_update(
    *,
    policy,
    q1,
    q2,
    q1_target,
    q2_target,
    optimizer,
    scheduler,
    actor_batch: dict,
    generators: dict[str, torch.Generator],
    flow_counter: FlowCounter,
    config: dict,
) -> dict:
    from forcesmolvla.rft.losses import (
        build_actor_q_action,
        compute_actor_q_loss,
    )
    from forcesmolvla.rft.training_cycle import (
        global_gradient_norm,
        gradients_finite,
        module_state_sha256,
    )

    device = actor_batch["reward"].device
    accumulation = config["batching"]["actor_gradient_accumulation"]
    eta = config["loss"]["eta_actor_q"]
    require(accumulation == 4 and len(actor_batch["indices"]) == 4, "G5_ACTOR_WINDOW_INVALID")
    critic_before = {
        "q1": module_state_sha256(q1), "q2": module_state_sha256(q2),
        "q1_target": module_state_sha256(q1_target), "q2_target": module_state_sha256(q2_target),
    }
    actor_before = module_state_sha256(policy)
    actor_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    total_valid_features = int(actor_batch["current_actor_batch"]["action_valid_mask"].sum().cpu()) * 7
    require(total_valid_features > 0, "G5_ACTOR_WINDOW_NO_VALID_FLOW_FEATURES")
    records = []
    fm_local_sum = fm_window_sum = actor_q_sum = balance_sum = z_sum = 0.0
    q1_action_values, q2_action_values = [], []
    tcp_q_gradient_square = 0.0
    gripper_q_gradient_max = 0.0
    gripper_fm_gradient_square = 0.0
    fm_latency = q_latency = 0.0
    policy.train(True)
    for micro_index in range(accumulation):
        mask = torch.arange(accumulation, device=device) == micro_index
        micro = {
            "current_observation": actor_batch["current_observation"].index(mask),
            "current_actor_batch": {
                name: (
                    value[mask]
                    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == accumulation
                    else type(value)(item for item, keep in zip(value, mask.cpu().tolist(), strict=True) if keep)
                    if isinstance(value, (tuple, list)) and len(value) == accumulation
                    else value
                )
                for name, value in actor_batch["current_actor_batch"].items()
            },
        }
        fm_noise = torch.randn(
            1, 50, 7, generator=generators["flow_matching_noise"], device=device
        )
        fm_time_value = torch.rand(
            1, generator=generators["flow_matching_timestep"], device=device
        )
        velocity_outputs = []

        def capture(_module, _inputs, output):
            output.retain_grad()
            velocity_outputs.append(output)

        hook = policy.model.action_out_proj.register_forward_hook(capture)
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            policy.train(True)
            losses, feature_mask, terms, router_state = flow_microbatch_terms(
                policy, micro["current_actor_batch"], fm_noise, fm_time_value
            )
            fm_contribution = losses.sum() / total_valid_features
            auxiliary_contribution = (
                0.01 * terms.balance / accumulation
                + 0.001 * terms.z / accumulation
            )
            (fm_contribution + auxiliary_contribution).backward()
        finally:
            hook.remove()
        torch.cuda.synchronize()
        fm_latency += time.perf_counter() - started
        require(len(velocity_outputs) == 1 and velocity_outputs[0].grad is not None, "G5_FM_GRADIENT_HOOK_FAILED")
        gripper_fm_gradient_square += float(velocity_outputs[0].grad[..., 6].float().square().sum().cpu())
        active_experts = sorted(set(router_state.route_ids[router_state.valid_mask].detach().cpu().tolist()))

        q_noise = torch.randn(
            1, 50, 7, generator=generators["actor_q_flow_noise"], device=device
        )
        policy.eval()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action_chunk = flow_counter.sample(
                policy, micro["current_actor_batch"], q_noise,
                call_id=f"g5-actor-update-micro={micro_index}", purpose="actor_guidance",
            )
            action_chunk.retain_grad()
            actor_q_loss = compute_actor_q_loss(
                q1=q1, q2=q2, current_observation=micro["current_observation"],
                actor_action_chunk7=action_chunk,
                actor_q_valid=torch.ones(1, dtype=torch.bool, device=device),
                delta_action_mean7=actor_batch["delta_mean"],
                delta_action_std7=actor_batch["delta_std"],
            )
            (eta * actor_q_loss / accumulation).backward()
        torch.cuda.synchronize()
        q_latency += time.perf_counter() - started
        require(action_chunk.grad is not None, "G5_ACTOR_Q_ACTION_GRADIENT_MISSING")
        tcp_q_gradient_square += float(action_chunk.grad[:, :3, :6].float().square().sum().cpu())
        gripper_q_gradient_max = max(
            gripper_q_gradient_max,
            float(action_chunk.grad[:, :3, 6].float().abs().max().cpu()),
        )
        q_action = build_actor_q_action(
            action_chunk.detach(),
            delta_action_mean7=actor_batch["delta_mean"],
            delta_action_std7=actor_batch["delta_std"],
        )
        ones = torch.ones(1, 3, dtype=torch.bool, device=device)
        with torch.no_grad():
            q1_value = q1(*micro["current_observation"].as_tuple(), q_action, ones)
            q2_value = q2(*micro["current_observation"].as_tuple(), q_action, ones)
        q1_action_values.append(q1_value)
        q2_action_values.append(q2_value)
        fm_value = float((losses.sum() / feature_mask.sum().clamp_min(1)).detach().cpu())
        q_value = float(actor_q_loss.detach().cpu())
        fm_local_sum += fm_value
        fm_window_sum += float(fm_contribution.detach().cpu())
        actor_q_sum += q_value
        balance_sum += float(terms.balance.detach().cpu())
        z_sum += float(terms.z.detach().cpu())
        records.append({
            "microbatch_index": micro_index,
            "identity": actor_batch["identities"][micro_index],
            "flow_matching_local_mean": fm_value,
            "flow_valid_feature_count": int(feature_mask.sum().cpu()),
            "flow_matching_window_contribution": float(fm_contribution.detach().cpu()),
            "actor_q_loss": q_value,
            "actor_q_window_contribution": float((actor_q_loss / accumulation).detach().cpu()),
            "balance": float(terms.balance.detach().cpu()),
            "z": float(terms.z.detach().cpu()),
            "active_router_experts": active_experts,
        })
        del losses, feature_mask, terms, action_chunk, actor_q_loss, velocity_outputs
        gc.collect()
        torch.cuda.empty_cache()

    require(gradients_finite(actor_parameters), "G5_ACTOR_ACCUMULATED_GRADIENT_NONFINITE")
    module_norms = actor_module_gradient_norms(policy)
    required_groups = (
        "vision_vlm", "action_io", "action_expert", "force_mlp",
        "fusion", "moe_experts", "force_action_adapter", "router",
    )
    require(all(module_norms[name] > 0 for name in required_groups), f"G5_ACTOR_REQUIRED_MODULE_GRADIENT_MISSING:{module_norms}")
    preclip = global_gradient_norm(actor_parameters)
    torch.nn.utils.clip_grad_norm_(actor_parameters, config["optimizers"]["actor"]["grad_clip_norm"])
    postclip = global_gradient_norm(actor_parameters)
    require(gradients_finite(actor_parameters), "G5_ACTOR_POSTCLIP_GRADIENT_NONFINITE")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    policy.eval()
    actor_after = module_state_sha256(policy)
    critic_after = {
        "q1": module_state_sha256(q1), "q2": module_state_sha256(q2),
        "q1_target": module_state_sha256(q1_target), "q2_target": module_state_sha256(q2_target),
    }
    require(actor_after != actor_before, "G5_ACTOR_OPTIMIZER_DID_NOT_CHANGE_ACTOR")
    require(critic_before == critic_after, "G5_CRITIC_CHANGED_DURING_ACTOR_STEP")
    require(all(parameter.grad is None for parameter in actor_parameters), "G5_ACTOR_GRADIENT_NOT_CLEARED")
    weighted_total = (
        fm_window_sum
        + 0.01 * balance_sum / accumulation
        + 0.001 * z_sum / accumulation
        + eta * actor_q_sum / accumulation
    )
    return {
        "microbatches": records,
        "loss": {
            "L_FM_window": fm_window_sum,
            "L_FM_local_means_diagnostic": fm_local_sum / accumulation,
            "L_actor_Q_window": actor_q_sum / accumulation,
            "L_balance_equal_microbatch_mean": balance_sum / accumulation,
            "L_z_equal_microbatch_mean": z_sum / accumulation,
            "weighted_actor_total": weighted_total,
            "weighted_q_term": eta * actor_q_sum / accumulation,
            "weighted_fm_term": fm_window_sum,
            "weighted_q_term_over_weighted_fm_term": abs(eta * actor_q_sum / accumulation) / max(abs(fm_window_sum), torch.finfo(torch.float32).tiny),
        },
        "actor_action_q": {
            "q1_mean": float(torch.cat(q1_action_values).mean().cpu()),
            "q2_mean": float(torch.cat(q2_action_values).mean().cpu()),
        },
        "gradient": {
            "tcp6_actor_q_gradient_norm": math.sqrt(tcp_q_gradient_square),
            "gripper_actor_q_gradient_max_abs": gripper_q_gradient_max,
            "gripper_flow_matching_gradient_norm": math.sqrt(gripper_fm_gradient_square),
            "preclip_global_norm": float(preclip.cpu()),
            "postclip_global_norm": float(postclip.cpu()),
            "module_gradient_norms": module_norms,
            "finite_before_and_after": True,
        },
        "latency_seconds": {
            "flow_matching_forward_backward": fm_latency,
            "differentiable_n10_flow_and_actor_q_backward": q_latency,
        },
        "normalization": {
            "window_valid_feature_count": total_valid_features,
            "fm_normalized_over_entire_window": True,
            "actor_q_valid_transition_count": accumulation,
            "balance_z_equal_microbatch_average": True,
            "retain_graph_used": False,
            "clip_calls": 1,
            "optimizer_steps": 1,
        },
        "state": {
            "actor_before_sha256": actor_before,
            "actor_after_sha256": actor_after,
            "critics_before": critic_before,
            "critics_after": critic_after,
        },
        "counters_increment": {
            "actor_optimizer_updates": 1,
            "actor_scheduler_steps": 1,
            "actor_target_updates": 0,
        },
    }


def capture_rng_states(generators: dict[str, torch.Generator]) -> dict:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all(),
        "named_generator_states": {
            name: generator.get_state() for name, generator in generators.items()
        },
    }


def rng_state_summary(state: dict) -> dict:
    from forcesmolvla.rft.training_cycle import tensor_sha256

    numpy_state = state["numpy_random_state"]
    return {
        "python_random_state_sha256": hashlib.sha256(repr(state["python_random_state"]).encode()).hexdigest(),
        "numpy_rng_state_sha256": hashlib.sha256(
            repr((numpy_state[0], numpy_state[2:])).encode()
            + np.asarray(numpy_state[1], dtype=np.uint32).tobytes()
        ).hexdigest(),
        "torch_cpu_rng_state_sha256": tensor_sha256(state["torch_cpu_rng_state"]),
        "torch_cuda_rng_state_sha256": [tensor_sha256(value) for value in state["torch_cuda_rng_states"]],
        "named_generator_state_sha256": {
            name: tensor_sha256(value)
            for name, value in sorted(state["named_generator_states"].items())
        },
    }
