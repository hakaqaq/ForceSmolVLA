#!/usr/bin/env python3
"""Fresh offline Twin-Q warm-up worker and strict-load verifier."""

from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml

from forcesmolvla import action_delta, rules
from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
from forcesmolvla.rft import flow_sampling, losses
from forcesmolvla.rft.critic_action_adapter_v2 import (
    aligned_fresh_chunk_execution_index_map_v2,
    critic_action_for_q_guidance_v2,
    raw_gripper_out_of_public_tolerance_mask,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/twin_q_critic_warmup.development.yaml"
SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v10_g7a_r2.json"
TASK_ID = "task2"
REWARD_TRANSITION_ROOT = ROOT / "datasets/task2_forcerft_offline_reward_transitions"
ACTION_CONTRACT_DIAGNOSTIC = {
    "raw_gripper_values": 0,
    "raw_gripper_out_of_public_tolerance": 0,
    "projected_gripper_patterns": 0,
    "duplicate_projected_gripper_patterns": 0,
}


def audited_action_contract_v2_adapter(
    chunk, *, delta_action_mean7, delta_action_std7
):
    execution_index_map = aligned_fresh_chunk_execution_index_map_v2()
    raw = chunk[:, execution_index_map, 6]
    outside = raw_gripper_out_of_public_tolerance_mask(
        raw,
        gripper_mean=delta_action_mean7[6],
        gripper_std=delta_action_std7[6],
    )
    action = critic_action_for_q_guidance_v2(
        chunk,
        execution_index_map=execution_index_map,
        delta_action_mean7=delta_action_mean7,
        delta_action_std7=delta_action_std7,
    )
    patterns = action[..., 6].detach().float()
    ACTION_CONTRACT_DIAGNOSTIC["raw_gripper_values"] += raw.numel()
    ACTION_CONTRACT_DIAGNOSTIC["raw_gripper_out_of_public_tolerance"] += int(
        outside.sum().item()
    )
    ACTION_CONTRACT_DIAGNOSTIC["projected_gripper_patterns"] += patterns.shape[0]
    ACTION_CONTRACT_DIAGNOSTIC["duplicate_projected_gripper_patterns"] += int(
        patterns.shape[0] - torch.unique(patterns, dim=0).shape[0]
    )
    return action


def finalize_action_contract_v2_result(path: Path, calls: dict[str, int]) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    raw_count = ACTION_CONTRACT_DIAGNOSTIC["raw_gripper_values"]
    pattern_count = ACTION_CONTRACT_DIAGNOSTIC["projected_gripper_patterns"]
    result["action_contract_v2"] = {
        "status": "pass",
        "internal_gripper_projection": "total_binary",
        "public_execution_authorization_used": False,
        "public_call_counts": calls,
        "raw_gripper_out_of_public_tolerance_rate": (
            ACTION_CONTRACT_DIAGNOSTIC["raw_gripper_out_of_public_tolerance"]
            / raw_count
            if raw_count
            else 0.0
        ),
        "binary_gripper_pattern_duplicate_rate": (
            ACTION_CONTRACT_DIAGNOSTIC["duplicate_projected_gripper_patterns"]
            / pattern_count
            if pattern_count
            else 0.0
        ),
        "clipping_added": False,
        "resampling_added": False,
        "binary_ste_added": False,
    }
    atomic_json(path, result)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def environment_audit() -> dict:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    uuid, name, driver = [item.strip() for item in query.split(",", 2)]
    return {
        "pid": os.getpid(), "python": sys.version, "pytorch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu_uuid": uuid, "gpu_name": name, "driver": driver,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "actor_autocast_dtype": "torch.bfloat16", "critic_dtype": "torch.float32",
        "torch_compile": False, "data_augmentation": False, "num_workers": 0,
    }


def configure_runtime() -> torch.device:
    require(torch.cuda.is_available(), "OFFLINE_TWIN_Q_CUDA_REQUIRED_NO_CPU_FALLBACK")
    require("4090" in torch.cuda.get_device_name(0), "OFFLINE_TWIN_Q_RTX4090D_REQUIRED")
    require(os.environ.get("PYTHONHASHSEED") == "42", "OFFLINE_TWIN_Q_PYTHONHASHSEED_REQUIRED")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "OFFLINE_TWIN_Q_CUBLAS_REQUIRED")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda:0")


def verify_config() -> tuple[dict, dict]:
    from forcesmolvla.rft.training_cycle import (
        verify_config as verify_training_cycle_config,
    )

    training_cycle = verify_training_cycle_config()
    config = yaml.safe_load(CONFIG.read_text())
    warmup = config["warmup"]
    require(
        warmup == {
            "critic_updates": 256, "critic_batch_size": 16,
            "calql_batch_size": 16, "actor_optimizer_updates": 0,
            "actor_scheduler_steps": 0, "critic_optimizer": "Adam",
            "critic_lr": 3e-4, "critic_betas": [0.9, 0.999],
            "critic_eps": 1e-8, "critic_weight_decay": 0.0,
            "critic_grad_clip_norm": 10.0, "critic_scheduler": "constant",
            "polyak_tau": 0.005, "polyak_updates_per_target": 256,
            "calql_alpha": 0.1, "calql_candidates_per_source": 2,
            "calql_temperature": 1.0, "calql_clipping_enabled": False,
        },
        "OFFLINE_TWIN_Q_RESOLVED_RECIPE_DRIFT",
    )
    require(config["initialization"]["g5_or_g6_checkpoint_parent"] is False, "OFFLINE_TWIN_Q_SMOKE_PARENT_FORBIDDEN")
    require(
        training_cycle["loss"]["alpha_calql"] == warmup["calql_alpha"]
        and training_cycle["loss"]["cql_candidates_per_source_M"]
        == warmup["calql_candidates_per_source"]
        and training_cycle["loss"]["cql_temperature"]
        == warmup["calql_temperature"]
        and training_cycle["targets"]["polyak_tau"] == warmup["polyak_tau"],
        "OFFLINE_TWIN_Q_G4_G5_LOSS_SEMANTICS_DRIFT",
    )
    return config, training_cycle


def named_generator(device: str, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def load_split_rows(split: str) -> list[dict]:
    import pyarrow.parquet as pq
    from forcesmolvla.rft.losses import AUTHORIZED_G4_COLUMNS

    require(split in {"train", "val"}, "OFFLINE_TWIN_Q_TEST_SPLIT_FORBIDDEN")
    columns = list(AUTHORIZED_G4_COLUMNS) + ["detector_terminal_frame"]
    table = pq.read_table(
        REWARD_TRANSITION_ROOT / "forcerft_offline_td_transitions.parquet",
        columns=columns,
        filters=[("split", "=", split)],
    )
    require(set(table.column("split").to_pylist()) == {split}, "OFFLINE_TWIN_Q_SPLIT_FILTER_LEAK")
    return table.to_pylist()


def attach_distance(rows: list[dict]) -> None:
    for row in rows:
        remaining = int(row["detector_terminal_frame"]) - int(row["next_frame"])
        require(remaining >= 0, "OFFLINE_TWIN_Q_NEGATIVE_TERMINAL_DISTANCE")
        row["policy_decision_distance"] = int(math.ceil(remaining / 3))


def split_data(train_data, rows: list[dict]):
    result = copy.copy(train_data)
    result.rows = rows
    return result


def identity(row: dict) -> str:
    return f"{row['transition_index']}|{row['episode_id']}|{row['anchor_frame']}|{row['next_frame']}"


def fixed_tensor_manifest(value: Any) -> Any:
    from forcesmolvla.rft.canonical_state import canonicalize

    return canonicalize(value)


def create_fixed_diagnostics(
    config: dict, train_data, validation_rows: list[dict], device: torch.device,
) -> tuple[dict, dict]:
    from forcesmolvla.rft.critic_warmup_checkpoint import select_fixed_critic_probe
    from forcesmolvla.rft.training_cycle import SerializableUniqueSampler

    diag = config["diagnostics"]
    seeds = diag["fixed_seed"]
    train_indices = select_fixed_critic_probe(
        train_data.rows, int(diag["train_critic_probe_size"]),
        seed=int(seeds["train_probe_rows"]),
    )
    actor_generator = named_generator("cpu", int(seeds["actor_probe_rows"]))
    actor_sampler = SerializableUniqueSampler(
        "offline_twin_q_actor_gradient_probe", train_data.actor_population, actor_generator
    )
    actor_indices = actor_sampler.draw(int(diag["actor_gradient_probe_batches"]))
    m = 2

    def evaluation_tensors(count: int, prefix: str) -> dict:
        td = named_generator("cuda", int(seeds[f"{prefix}_td_noise"]))
        current = named_generator("cuda", int(seeds[f"{prefix}_current_policy_noise"]))
        following = named_generator("cuda", int(seeds[f"{prefix}_next_policy_noise"]))
        proposal = named_generator("cpu", int(seeds[f"{prefix}_random_proposals"]))
        return {
            "td_noise": torch.randn(
                count, 50, 7, generator=td, dtype=torch.float32, device=device
            ),
            "current_policy_noise": torch.randn(
                count, m, 50, 7, generator=current, dtype=torch.float32, device=device
            ),
            "next_policy_noise": torch.randn(
                count, m, 50, 7, generator=following, dtype=torch.float32, device=device
            ),
            "proposal_indices": torch.randint(
                len(train_data.proposal_population), (count, m), generator=proposal
            ),
        }

    q_generator = named_generator("cuda", int(seeds["actor_probe_q_noise"]))
    fm_generator = named_generator("cuda", int(seeds["actor_probe_fm_noise"]))
    time_generator = named_generator("cuda", int(seeds["actor_probe_fm_timestep"]))
    bundle = {
        "schema_version": "forcesmolvla_g7a_fixed_diagnostics.v1",
        "train_probe_indices": train_indices,
        "validation_indices": list(range(len(validation_rows))),
        "actor_probe_indices": actor_indices,
        "train_evaluation": evaluation_tensors(len(train_indices), "train"),
        "validation_evaluation": evaluation_tensors(len(validation_rows), "validation"),
        "actor_q_noise": torch.randn(
            len(actor_indices), 50, 7, generator=q_generator, device=device
        ),
        "actor_fm_noise": torch.randn(
            len(actor_indices), 50, 7, generator=fm_generator, device=device
        ),
        "actor_fm_timestep": torch.rand(
            len(actor_indices), generator=time_generator, device=device
        ),
        "seeds": dict(seeds),
        "actor_probe_sampler_state": actor_sampler.state_dict(),
    }
    manifest = {
        "schema_version": bundle["schema_version"],
        "train_probe_row_ids": [identity(train_data.rows[index]) for index in train_indices],
        "validation_row_ids": [identity(validation_rows[index]) for index in range(len(validation_rows))],
        "actor_probe_row_ids": [identity(train_data.rows[index]) for index in actor_indices],
        "train_probe_row_count": len(train_indices),
        "validation_row_count": len(validation_rows),
        "actor_probe_batch_count": len(actor_indices),
        "fixed_values": fixed_tensor_manifest({
            key: value for key, value in bundle.items()
            if key not in {"train_probe_indices", "validation_indices", "actor_probe_indices"}
        }),
        "seeds": dict(seeds),
        "created_before_update_0_evaluation": True,
        "used_at_updates": [0, 256],
        "validation_used_for_selection_or_early_stop": False,
    }
    return bundle, manifest


def normalized_gripper_endpoints(batch: dict) -> torch.Tensor:
    device = batch["reward"].device
    return torch.stack((
        (torch.tensor(0.0, device=device) - batch["delta_mean"][6]) / batch["delta_std"][6],
        (torch.tensor(0.085, device=device) - batch["delta_mean"][6]) / batch["delta_std"][6],
    )).float()


def evaluate_critic_split(
    *, label: str, rows: list[dict], indices: list[int], data, fixed: dict,
    policy, q1, q2, q1_target, q2_target, train_data, device, batch_size: int,
) -> dict:
    from forcesmolvla.rft.critic_warmup_checkpoint import describe, grouped_regression, regression_metrics
    from forcesmolvla.rft.losses import (
        compute_behavior_q, compute_td_target_from_current_actor,
        evaluate_calql_candidates,
    )
    from forcesmolvla.rft.training_cycle import calql_unclipped_details
    from forcesmolvla.rft.training_cycle import FlowCounter, sample_policy_candidates, slice_actor_batch

    modes = {module: module.training for module in (policy, q1, q2)}
    policy.eval(); q1.eval(); q2.eval(); q1_target.eval(); q2_target.eval()
    flow_counter = FlowCounter(inference_batch_size=4)
    per_row = []
    td_error1, td_error2, difference1, difference2 = [], [], [], []
    distribution = {name: [] for name in (
        "dataset_q", "current_policy_q", "next_policy_q", "random_proposal_q",
        "td_target", "q1_q2_disagreement",
    )}
    proposal_indices_all: list[int] = []
    clamp_count = clamp_total = 0
    policy_grippers = {"current": [], "next": []}
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                stop = min(start + batch_size, len(indices))
                local_indices = indices[start:stop]
                batch = data.build_batch(
                    local_indices, policy, device,
                    canonical_task_feature=q1.canonical_task_feature,
                )
                fixed_slice = slice(start, stop)
                td_noise = fixed["td_noise"][fixed_slice].to(device)

                def sampled(actor, actor_batch, noise, *, call_id, purpose):
                    return flow_counter.sample(
                        actor, actor_batch, noise, call_id=call_id, purpose=purpose
                    )

                td_target = compute_td_target_from_current_actor(
                    reward=batch["reward"], discount=batch["discount"],
                    terminated=batch["terminated"], bootstrap_mask=batch["bootstrap_mask"],
                    next_observation=batch["next_observation"],
                    next_actor_batch=batch["next_actor_batch"], next_noise7=td_noise,
                    actor=policy, q1_target=q1_target, q2_target=q2_target,
                    delta_action_mean7=batch["delta_mean"], delta_action_std7=batch["delta_std"],
                    call_id=f"offline-twin-q-{label}-{start}-td", sample_action_fn=sampled,
                )
                q1_data = compute_behavior_q(
                    q1, batch["current_observation"], batch["behavior_action"], batch["behavior_mask"]
                )
                q2_data = compute_behavior_q(
                    q2, batch["current_observation"], batch["behavior_action"], batch["behavior_mask"]
                )
                td_error1.extend(torch.square(q1_data - td_target).cpu().tolist())
                td_error2.extend(torch.square(q2_data - td_target).cpu().tolist())
                q_mean = (q1_data + q2_data) / 2.0
                disagreement = (q1_data - q2_data).abs()
                distribution["dataset_q"].extend(torch.cat((q1_data, q2_data)).cpu().tolist())
                distribution["td_target"].extend(td_target.cpu().tolist())
                distribution["q1_q2_disagreement"].extend(disagreement.cpu().tolist())

                valid = batch["behavior_mask"].all(dim=-1) & (~batch["terminated"])
                if bool(valid.any()):
                    valid_count = int(valid.sum())
                    current_batch = slice_actor_batch(batch["current_actor_batch"], 0, len(local_indices))
                    next_batch = slice_actor_batch(batch["next_actor_batch"], 0, len(local_indices))
                    current_batch = {
                        name: value[valid] if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(local_indices)
                        else type(value)(item for item, keep in zip(value, valid.cpu().tolist(), strict=True) if keep)
                        if isinstance(value, (tuple, list)) and len(value) == len(local_indices) else value
                        for name, value in current_batch.items()
                    }
                    next_batch = {
                        name: value[valid] if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(local_indices)
                        else type(value)(item for item, keep in zip(value, valid.cpu().tolist(), strict=True) if keep)
                        if isinstance(value, (tuple, list)) and len(value) == len(local_indices) else value
                        for name, value in next_batch.items()
                    }
                    current_noise = fixed["current_policy_noise"][fixed_slice].to(device)[valid]
                    next_noise = fixed["next_policy_noise"][fixed_slice].to(device)[valid]
                    policy_current = sample_policy_candidates(
                        policy, current_batch, current_noise, batch["delta_mean"], batch["delta_std"],
                        flow_counter, purpose="cql_current",
                        call_id=f"offline-twin-q-{label}-{start}-current",
                    )
                    policy_next = sample_policy_candidates(
                        policy, next_batch, next_noise, batch["delta_mean"], batch["delta_std"],
                        flow_counter, purpose="cql_next",
                        call_id=f"offline-twin-q-{label}-{start}-next",
                    )
                    proposal_indices = fixed["proposal_indices"][fixed_slice][valid.cpu()].reshape(-1).tolist()
                    random_candidates = train_data.proposal_actions[proposal_indices].to(device).reshape(
                        valid_count, 2, 3, 7
                    )
                    observation = batch["current_observation"].index(valid)
                    endpoint = normalized_gripper_endpoints(batch)
                    q1_candidates = evaluate_calql_candidates(
                        q1, observation, random_candidates, policy_current, policy_next, endpoint
                    )
                    q2_candidates = evaluate_calql_candidates(
                        q2, observation, random_candidates, policy_current, policy_next, endpoint
                    )
                    mc = batch["mc_return"][valid]
                    details1 = calql_unclipped_details(q1_data[valid], q1_candidates, mc, temperature=1.0)
                    details2 = calql_unclipped_details(q2_data[valid], q2_candidates, mc, temperature=1.0)
                    difference1.extend(details1["difference"].cpu().tolist())
                    difference2.extend(details2["difference"].cpu().tolist())
                    for values in (q1_candidates, q2_candidates):
                        distribution["random_proposal_q"].extend(values[:, 0:2].reshape(-1).cpu().tolist())
                        distribution["current_policy_q"].extend(values[:, 2:4].reshape(-1).cpu().tolist())
                        distribution["next_policy_q"].extend(values[:, 4:6].reshape(-1).cpu().tolist())
                    clamp_count += int((q1_candidates < mc[:, None]).sum().cpu())
                    clamp_count += int((q2_candidates < mc[:, None]).sum().cpu())
                    clamp_total += q1_candidates.numel() + q2_candidates.numel()
                    proposal_indices_all.extend(proposal_indices)
                    policy_grippers["current"].extend(
                        policy_current[..., 6].reshape(-1).cpu().tolist()
                    )
                    policy_grippers["next"].extend(
                        policy_next[..., 6].reshape(-1).cpu().tolist()
                    )

                for offset, row_index in enumerate(local_indices):
                    row = rows[row_index]
                    per_row.append({
                        "row_identity": identity(row), "terminated": bool(row["terminated"]),
                        "executed_steps": int(row["executed_steps"]),
                        "policy_decision_distance": int(row["policy_decision_distance"]),
                        "q1": float(q1_data[offset].cpu()), "q2": float(q2_data[offset].cpu()),
                        "q_mean": float(q_mean[offset].cpu()),
                        "mc_return": float(batch["mc_return"][offset].cpu()),
                        "td_target": float(td_target[offset].cpu()),
                    })
                del batch, td_target, q1_data, q2_data, q_mean, disagreement
                gc.collect(); torch.cuda.empty_cache()
    finally:
        policy.train(modes[policy]); q1.train(modes[q1]); q2.train(modes[q2])

    proposal_identities = [
        identity(train_data.rows[train_data.proposal_population[index]])
        for index in proposal_indices_all
    ]
    behavior_lengths = [int(rows[index]["executed_steps"]) for index in indices]
    q_values = [row["q_mean"] for row in per_row]
    mc_values = [row["mc_return"] for row in per_row]
    td1 = float(np.mean(td_error1)); td2 = float(np.mean(td_error2))
    calql1 = float(np.mean(difference1)); calql2 = float(np.mean(difference2))
    result = {
        "label": label, "row_count": len(per_row),
        "td_mse": {"q1": td1, "q2": td2, "twin_mean": (td1 + td2) / 2.0},
        "calql_conservative": {"q1": calql1, "q2": calql2, "twin_mean": (calql1 + calql2) / 2.0},
        "total_critic_loss": (td1 + 0.1 * calql1 + td2 + 0.1 * calql2) / 2.0,
        "q1_q2_disagreement": describe(distribution["q1_q2_disagreement"]),
        "q_vs_mc_return": regression_metrics(q_values, mc_values),
        "grouped_q_vs_mc_return": grouped_regression(per_row),
        "q_distributions": {name: describe(values) for name, values in distribution.items()},
        "candidate_mc_return_clamp_rate": clamp_count / clamp_total,
        "candidate_mc_return_clamp_count": clamp_count,
        "candidate_comparison_count": clamp_total,
        "proposal_support": {
            "draw_count": len(proposal_indices_all),
            "unique_population_indices": len(set(proposal_indices_all)),
            "duplicate_rate": 1.0 - len(set(proposal_indices_all)) / max(len(proposal_indices_all), 1),
            "row_identity_sha256": hashlib.sha256("\n".join(proposal_identities).encode()).hexdigest(),
            "whole_k3x7_macro": True, "candidate_mask_length": 3,
            "behavior_mask_length_distribution": {
                str(value): behavior_lengths.count(value) for value in sorted(set(behavior_lengths))
            },
            "random_gripper_values": sorted({
                float(value) for index in proposal_indices_all
                for value in train_data.proposal_actions[index, :, 6].tolist()
            }),
            "current_policy_gripper_values": sorted({
                float(value) for value in policy_grippers["current"]
            }),
            "next_policy_gripper_values": sorted({
                float(value) for value in policy_grippers["next"]
            }),
        },
        "finite": True,
        "terminal_next_actor_target_calls": 0,
        "policy_actions_detached": True,
        "validation_selection_or_early_stop": False,
        "flow_counts": flow_counter.report(),
        "latency_seconds": time.perf_counter() - started,
        "per_row_digest": hashlib.sha256(
            json.dumps(per_row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    require(all(math.isfinite(value) for value in (
        td1, td2, calql1, calql2, result["total_critic_loss"],
        result["candidate_mc_return_clamp_rate"],
    )), "OFFLINE_TWIN_Q_DIAGNOSTIC_NONFINITE")
    return result


def _metric_from_squares(values: dict[str, float]) -> dict:
    fm_norm = math.sqrt(values["fm_square"])
    q_norm = math.sqrt(values["q_square"])
    denominator = fm_norm * q_norm
    return {
        "fm_norm": fm_norm,
        "q_norm": q_norm,
        "raw_q_over_fm": q_norm / max(fm_norm, torch.finfo(torch.float32).tiny),
        "cosine_similarity": values["dot"] / denominator if denominator else 0.0,
        "fm_nonzero": fm_norm > 0.0,
        "q_nonzero": q_norm > 0.0,
    }


def measure_actor_gradient_scale(
    *, policy, q1, q2, train_data, actor_indices: list[int], fixed: dict,
    device, eta_candidates: list[float], band: list[float],
) -> dict:
    from forcesmolvla.rft.critic_warmup_checkpoint import (
        actor_gradient_group, aggregate_gradient_probes, module_component_digests,
    )
    from forcesmolvla.rft.losses import compute_actor_q_loss
    from forcesmolvla.rft.training_cycle import FlowCounter, flow_microbatch_terms

    actor_before = module_component_digests(policy)
    q_modes = (q1.training, q2.training)
    policy_mode = policy.training
    actor_parameters = [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad]
    flow_counter = FlowCounter(inference_batch_size=4)
    probes = []
    tcp_nonzero = True
    gripper_q_zero = True
    gripper_fm_nonzero = True
    started = time.perf_counter()
    try:
        for probe_index, row_index in enumerate(actor_indices):
            batch = train_data.build_batch(
                [row_index], policy, device,
                canonical_task_feature=q1.canonical_task_feature,
                include_flow_actions=True,
            )
            for _name, parameter in actor_parameters:
                parameter.grad = None
            policy.eval(); q1.eval(); q2.eval()
            q_noise = fixed["actor_q_noise"][probe_index:probe_index + 1].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                action_chunk = flow_counter.sample(
                    policy, batch["current_actor_batch"], q_noise,
                    call_id=f"offline-twin-q-gradient-probe={probe_index}", purpose="actor_guidance",
                )
                action_chunk.retain_grad()
                q_loss = compute_actor_q_loss(
                    q1=q1, q2=q2, current_observation=batch["current_observation"],
                    actor_action_chunk7=action_chunk,
                    actor_q_valid=torch.ones(1, dtype=torch.bool, device=device),
                    delta_action_mean7=batch["delta_mean"], delta_action_std7=batch["delta_std"],
                )
            q_loss.backward()
            require(action_chunk.grad is not None, "OFFLINE_TWIN_Q_ACTOR_Q_ACTION_GRADIENT_MISSING")
            tcp_norm = float(action_chunk.grad[:, :3, :6].float().norm().cpu())
            gripper_q = float(action_chunk.grad[:, :3, 6].float().abs().max().cpu())
            tcp_nonzero &= tcp_norm > 0.0
            gripper_q_zero &= gripper_q == 0.0
            require(
                all(parameter.grad is None for critic in (q1, q2) for parameter in critic.parameters()),
                "OFFLINE_TWIN_Q_Q_PROBE_CRITIC_PARAMETER_GRADIENT",
            )
            q_gradients = {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in actor_parameters if parameter.grad is not None
            }
            for _name, parameter in actor_parameters:
                parameter.grad = None

            policy.train(True)
            velocity_outputs = []

            def capture(_module, _inputs, output):
                output.retain_grad(); velocity_outputs.append(output)

            hook = policy.model.action_out_proj.register_forward_hook(capture)
            try:
                fm_noise = fixed["actor_fm_noise"][probe_index:probe_index + 1].to(device)
                fm_time = fixed["actor_fm_timestep"][probe_index:probe_index + 1].to(device)
                losses, feature_mask, _terms, _router = flow_microbatch_terms(
                    policy, batch["current_actor_batch"], fm_noise, fm_time
                )
                fm_loss = losses.sum() / feature_mask.sum().clamp_min(1)
                fm_loss.backward()
            finally:
                hook.remove()
            require(len(velocity_outputs) == 1 and velocity_outputs[0].grad is not None, "OFFLINE_TWIN_Q_FM_OUTPUT_GRADIENT_MISSING")
            gripper_fm = float(velocity_outputs[0].grad[..., 6].float().norm().cpu())
            gripper_fm_nonzero &= gripper_fm > 0.0

            accumulators = {
                "global": {"fm_square": 0.0, "q_square": 0.0, "dot": 0.0}
            }
            for name, parameter in actor_parameters:
                group = actor_gradient_group(name)
                values = accumulators.setdefault(
                    group, {"fm_square": 0.0, "q_square": 0.0, "dot": 0.0}
                )
                fm_gradient = parameter.grad
                q_gradient = q_gradients.get(name)
                fm_square = (
                    float(fm_gradient.detach().float().square().sum().cpu())
                    if fm_gradient is not None else 0.0
                )
                q_square = (
                    float(q_gradient.float().square().sum())
                    if q_gradient is not None else 0.0
                )
                dot = 0.0
                if fm_gradient is not None and q_gradient is not None:
                    dot = float(
                        (fm_gradient.detach().float() * q_gradient.to(device).float()).sum().cpu()
                    )
                for target in (values, accumulators["global"]):
                    target["fm_square"] += fm_square
                    target["q_square"] += q_square
                    target["dot"] += dot
            probe = {
                "probe_index": probe_index,
                "row_identity": identity(train_data.rows[row_index]),
                "L_FM": float(fm_loss.detach().cpu()),
                "L_actor_Q_unweighted": float(q_loss.detach().cpu()),
                "tcp6_actor_q_gradient_norm": tcp_norm,
                "gripper_actor_q_gradient_max_abs": gripper_q,
                "gripper_flow_matching_gradient_norm": gripper_fm,
                "global": _metric_from_squares(accumulators.pop("global")),
                "modules": {
                    name: _metric_from_squares(values)
                    for name, values in sorted(accumulators.items())
                },
            }
            require(all(math.isfinite(value) for value in (
                probe["L_FM"], probe["L_actor_Q_unweighted"], tcp_norm,
                gripper_q, gripper_fm, probe["global"]["raw_q_over_fm"],
                probe["global"]["cosine_similarity"],
            )), "OFFLINE_TWIN_Q_GRADIENT_PROBE_NONFINITE")
            probes.append(probe)
            for _name, parameter in actor_parameters:
                parameter.grad = None
            del batch, action_chunk, q_loss, q_gradients, losses, feature_mask, fm_loss, velocity_outputs
            gc.collect(); torch.cuda.empty_cache()
            print(f"OFFLINE_TWIN_Q_GRADIENT_PROBE {probe_index + 1}/{len(actor_indices)}", flush=True)
    finally:
        policy.train(policy_mode); q1.train(q_modes[0]); q2.train(q_modes[1])
        for _name, parameter in actor_parameters:
            parameter.grad = None
    actor_after = module_component_digests(policy)
    require(actor_before == actor_after, "OFFLINE_TWIN_Q_ACTOR_CHANGED_DURING_GRADIENT_MEASUREMENT")
    require(tcp_nonzero and gripper_q_zero and gripper_fm_nonzero, "OFFLINE_TWIN_Q_ACTION_GRADIENT_CONTRACT_FAILED")
    result = aggregate_gradient_probes(probes, eta_candidates, band)
    result.update({
        "per_probe": probes,
        "tcp6_q_gradient_nonzero_all_probes": tcp_nonzero,
        "gripper_q_gradient_exact_zero_all_probes": gripper_q_zero,
        "gripper_flow_matching_gradient_nonzero_all_probes": gripper_fm_nonzero,
        "critic_parameter_gradients": "none",
        "actor_state_before": actor_before,
        "actor_state_after": actor_after,
        "flow_counts": flow_counter.report(),
        "latency_seconds": time.perf_counter() - started,
    })
    return result


def compact_critic_report(report: dict) -> dict:
    polyak = {}
    for name, value in report["polyak"].items():
        polyak[name] = {key: item for key, item in value.items() if key != "tensors"}
        polyak[name]["tensor_records_digest"] = hashlib.sha256(
            json.dumps(value["tensors"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return {
        "critic_substep": report["critic_substep"],
        "td_batch": report["td_batch"], "calql_batch": report["calql_batch"],
        "proposal_population_indices": report["proposal_population_indices"],
        "proposal_population_identity_sha256": report["proposal_population_identity_sha256"],
        "loss": report["loss"], "statistics": report["statistics"],
        "gradient": report["gradient"], "terminal_rows": report["terminal_rows"],
        "terminal_next_actor_and_target_q_calls": report["terminal_next_actor_and_target_q_calls"],
        "state": report["state"], "polyak": polyak,
        "counters_increment": report["counters_increment"],
        "latency_seconds": report["latency_seconds"],
    }


def summarize_update_window(reports: list[dict]) -> dict:
    from forcesmolvla.rft.critic_warmup_checkpoint import describe

    return {
        "update_start": reports[0]["critic_substep"],
        "update_end": reports[-1]["critic_substep"],
        "L_TD_Q1": describe([item["loss"]["L_TD_Q1"] for item in reports]),
        "L_TD_Q2": describe([item["loss"]["L_TD_Q2"] for item in reports]),
        "L_CalQL_Q1": describe([item["loss"]["L_CalQL_Q1"] for item in reports]),
        "L_CalQL_Q2": describe([item["loss"]["L_CalQL_Q2"] for item in reports]),
        "L_critic": describe([item["loss"]["L_critic"] for item in reports]),
        "preclip_gradient_norm": describe([
            item["gradient"]["preclip_global_norm"] for item in reports
        ]),
        "postclip_gradient_norm": describe([
            item["gradient"]["postclip_global_norm"] for item in reports
        ]),
        "interpretation": "fixed_window_measurement_not_monotonicity_acceptance",
    }


def summarize_sampled_rows(reports: list[dict], rows: list[dict], train_data) -> dict:
    from forcesmolvla.rft.critic_warmup_checkpoint import describe

    by_transition = {int(row["transition_index"]): row for row in rows}
    result = {}
    for source in ("td_batch", "calql_batch"):
        samples = [item for report in reports for item in report[source]]
        resolved = [by_transition[int(item["transition_index"])] for item in samples]
        ids = [identity(row) for row in resolved]
        result[source] = {
            "draw_count": len(resolved), "unique_row_count": len(set(ids)),
            "row_identity_order_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "terminal_count": sum(bool(row["terminated"]) for row in resolved),
            "nonterminal_count": sum(not bool(row["terminated"]) for row in resolved),
            "executed_steps_distribution": {
                str(step): sum(int(row["executed_steps"]) == step for row in resolved)
                for step in (1, 2, 3)
            },
            "policy_decision_distance": describe([
                int(row["policy_decision_distance"]) for row in resolved
            ]),
        }
    proposal = [index for report in reports for index in report["proposal_population_indices"]]
    proposal_ids = [identity(train_data.rows[train_data.proposal_population[index]]) for index in proposal]
    result["empirical_random_proposal"] = {
        "draw_count": len(proposal), "unique_population_index_count": len(set(proposal)),
        "duplicate_rate": 1.0 - len(set(proposal)) / len(proposal),
        "row_identity_order_sha256": hashlib.sha256("\n".join(proposal_ids).encode()).hexdigest(),
        "whole_k3x7_macro_with_replacement": True,
    }
    return result


def initialize_fresh(*, device: torch.device, with_data: bool) -> dict:
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.critic import (
        build_twin_q, modules_storage_independent, state_exact,
    )
    from forcesmolvla.rft.training_cycle import module_state_sha256
    from forcesmolvla.rft.training_cycle import (
        DATASET,
        PARENT_ACTOR_CHECKPOINT,
        REWARD_BACKBONE_MANIFEST,
        REWARD_BACKBONE_PARAMETERS,
        TrainData,
    )

    data = None
    if with_data:
        data = TrainData()
        data.canonicalize_proposal_gripper_for_runtime(device)
        metadata = {
            int(row["transition_index"]): row
            for row in load_split_rows("train")
        }
        for row in data.rows:
            row["detector_terminal_frame"] = metadata[int(row["transition_index"])]["detector_terminal_frame"]
        attach_distance(data.rows)
    with redirect_stdout(sys.stderr):
        policy = ForceSmolVLAPolicy.from_pretrained(
            PARENT_ACTOR_CHECKPOINT,
            local_files_only=True,
            force_download=False,
            strict=True,
            artifact_use="development",
        ).to(device)
    policy.eval()
    dataset_conversion = json.loads(
        (DATASET / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    task_prompts = {str(item["task"]) for item in dataset_conversion["episodes"]}
    require(len(task_prompts) == 1, "OFFLINE_TWIN_Q_TASK_PROMPT_AMBIGUOUS")
    q1, q2, q1_target, q2_target, conversion = build_twin_q(
        REWARD_BACKBONE_PARAMETERS,
        REWARD_BACKBONE_MANIFEST,
        seed=0,
        task=task_prompts.pop(),
    )
    q1, q2, q1_target, q2_target = (
        module.to(device) for module in (q1, q2, q1_target, q2_target)
    )
    require(
        modules_storage_independent(q1, q2)
        and modules_storage_independent(q1, q1_target)
        and modules_storage_independent(q2, q2_target)
        and state_exact(q1, q1_target) and state_exact(q2, q2_target),
        "OFFLINE_TWIN_Q_FRESH_TWIN_Q_INITIALIZATION_INVALID",
    )
    q1.train(True); q2.train(True)
    q1_target.make_permanent_eval_target(); q2_target.make_permanent_eval_target()
    trainable = [
        parameter for critic in (q1, q2) for parameter in critic.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    actor_ids = {id(parameter) for parameter in policy.parameters()}
    critic_ids = [id(parameter) for parameter in trainable]
    target_ids = {
        id(parameter) for target in (q1_target, q2_target) for parameter in target.parameters()
    }
    require(
        not actor_ids.intersection(critic_ids)
        and not target_ids.intersection(critic_ids)
        and len(critic_ids) == len(set(critic_ids)),
        "OFFLINE_TWIN_Q_CRITIC_OPTIMIZER_OWNERSHIP_INVALID",
    )
    ownership = {
        "actor_optimizer_created": 0, "actor_scheduler_created": 0,
        "critic_optimizer_type": "Adam", "critic_trainable_tensor_count": len(trainable),
        "actor_critic_parameter_intersection": 0,
        "target_parameter_in_optimizer": 0,
        "frozen_backbone_in_optimizer": sum(
            id(parameter) in set(critic_ids)
            for critic in (q1, q2)
            for backbone in (critic.camera1_backbone, critic.camera2_backbone)
            for parameter in backbone.parameters()
        ),
        "q1_q2_storage_independent": modules_storage_independent(q1, q2),
    }
    require(ownership["frozen_backbone_in_optimizer"] == 0, "OFFLINE_TWIN_Q_FROZEN_BACKBONE_IN_OPTIMIZER")
    return {
        "actor": policy, "q1": q1, "q2": q2,
        "q1_target": q1_target, "q2_target": q2_target,
        "optimizer": optimizer, "scheduler": scheduler,
        "data": data, "ownership": ownership, "conversion": conversion,
        "actor_initial_sha256": module_state_sha256(policy),
    }


def warmup_generators(config: dict) -> dict[str, torch.Generator]:
    seeds = config["rng"]["named_stream_seeds"]
    return {
        "td_sampler": named_generator("cpu", int(seeds["td_sampler"])),
        "calql_sampler": named_generator("cpu", int(seeds["calql_sampler"])),
        "empirical_random_proposal": named_generator("cpu", int(seeds["empirical_random_proposal"])),
        "td_next_action_flow_noise": named_generator("cuda", int(seeds["td_next_action_flow_noise"])),
        "calql_current_policy_flow_noise": named_generator("cuda", int(seeds["calql_current_policy_flow_noise"])),
        "calql_next_policy_flow_noise": named_generator("cuda", int(seeds["calql_next_policy_flow_noise"])),
    }


def warmup_samplers(data, generators: dict):
    from forcesmolvla.rft.training_cycle import SerializableReplacementSampler, SerializableUniqueSampler

    return {
        "td": SerializableUniqueSampler("TD_sampler", data.td_population, generators["td_sampler"]),
        "calql": SerializableUniqueSampler("CalQL_sampler", data.calql_population, generators["calql_sampler"]),
        "empirical_random_proposal": SerializableReplacementSampler(
            "empirical_random_proposal", len(data.proposal_population),
            generators["empirical_random_proposal"],
        ),
    }


def run_warmup(args) -> None:
    from forcesmolvla.rft.canonical_state import canonical_digest, canonicalize
    from forcesmolvla.rft.critic import modules_storage_independent
    from forcesmolvla.rft.critic_warmup_checkpoint import (
        CRITIC_WARMUP_COUNTERS,
        module_component_digests,
        save_critic_warmup_checkpoint,
        sha256_file,
        validate_critic_warmup_checkpoint,
    )
    from forcesmolvla.rft.training_cycle import (
        ensure_all_gradients_none, module_state_sha256,
        optimizer_state_storage_independent,
    )
    from forcesmolvla.rft.training_cycle import (
        FORBIDDEN_OPENS,
        FlowCounter,
        PARENT_ACTOR_CHECKPOINT,
        capture_rng_states,
        critic_update,
        install_open_audit,
    )

    install_open_audit()
    device = configure_runtime()
    config, training_config = verify_config()
    context = initialize_fresh(device=device, with_data=True)
    policy, q1, q2 = context["actor"], context["q1"], context["q2"]
    q1_target, q2_target = context["q1_target"], context["q2_target"]
    data = context["data"]
    validation_rows = load_split_rows("val")
    attach_distance(validation_rows)
    validation_data = split_data(data, validation_rows)
    generators = warmup_generators(config)
    samplers = warmup_samplers(data, generators)
    actor_initial = module_component_digests(policy)
    q_initial = {name: module_state_sha256(module) for name, module in (
        ("q1", q1), ("q2", q2), ("q1_target", q1_target), ("q2_target", q2_target)
    )}
    backbone_initial = {
        f"{name}.{camera}": module_state_sha256(getattr(critic, camera))
        for name, critic in (("q1", q1), ("q2", q2))
        for camera in ("camera1_backbone", "camera2_backbone")
    }

    fixed, fixed_manifest = create_fixed_diagnostics(
        config, data, validation_rows, device
    )
    require(not args.fixed_diagnostics.exists(), "OFFLINE_TWIN_Q_FIXED_DIAGNOSTIC_TARGET_EXISTS")
    args.fixed_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixed, args.fixed_diagnostics)
    with args.fixed_diagnostics.open("rb") as stream:
        os.fsync(stream.fileno())
    fixed_manifest["bundle_path"] = args.fixed_diagnostics.name
    fixed_manifest["bundle_sha256"] = sha256_file(args.fixed_diagnostics)
    fixed_manifest["bundle_file_size"] = args.fixed_diagnostics.stat().st_size

    evaluation = {"update_0": {}, "update_256": {}}
    evaluation["update_0"]["train_probe"] = evaluate_critic_split(
        label="update0_train_probe", rows=data.rows,
        indices=fixed["train_probe_indices"], data=data,
        fixed=fixed["train_evaluation"], policy=policy, q1=q1, q2=q2,
        q1_target=q1_target, q2_target=q2_target, train_data=data,
        device=device, batch_size=16,
    )
    evaluation["update_0"]["validation"] = evaluate_critic_split(
        label="update0_validation", rows=validation_rows,
        indices=fixed["validation_indices"], data=validation_data,
        fixed=fixed["validation_evaluation"], policy=policy, q1=q1, q2=q2,
        q1_target=q1_target, q2_target=q2_target, train_data=data,
        device=device, batch_size=16,
    )
    print("OFFLINE_TWIN_Q_UPDATE0_DIAGNOSTICS_COMPLETE", flush=True)

    sampler_initial = {name: sampler.state_dict() for name, sampler in samplers.items()}
    rng_initial = capture_rng_states(generators)
    reports = []
    flow_counter = FlowCounter(inference_batch_size=4)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, 257):
        td_indices = samplers["td"].draw(16)
        calql_indices = samplers["calql"].draw(16)
        td_batch = data.build_batch(
            td_indices, policy, device, canonical_task_feature=q1.canonical_task_feature
        )
        calql_batch = data.build_batch(
            calql_indices, policy, device, canonical_task_feature=q1.canonical_task_feature
        )
        report = critic_update(
            step=step, policy=policy, q1=q1, q2=q2,
            q1_target=q1_target, q2_target=q2_target,
            optimizer=context["optimizer"], scheduler=context["scheduler"],
            td_batch=td_batch, calql_batch=calql_batch, train_data=data,
            proposal_sampler=samplers["empirical_random_proposal"],
            generators=generators, flow_counter=flow_counter, config=training_config,
        )
        reports.append(compact_critic_report(report))
        del td_batch, calql_batch, report
        gc.collect(); torch.cuda.empty_cache()
        if step % 16 == 0:
            print(f"OFFLINE_TWIN_Q_CRITIC_UPDATE {step}/256", flush=True)
    torch.cuda.synchronize()
    warmup_runtime = {
        "latency_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "flow_counts": flow_counter.report(),
    }

    require(context["scheduler"].last_epoch == 256, "OFFLINE_TWIN_Q_CRITIC_SCHEDULER_COUNTER_INVALID")
    require(module_component_digests(policy) == actor_initial, "OFFLINE_TWIN_Q_ACTOR_CHANGED_DURING_WARMUP")
    require(all(parameter.grad is None for parameter in policy.parameters()), "OFFLINE_TWIN_Q_ACTOR_GRADIENT_FROM_CRITIC")
    require(optimizer_state_storage_independent(context["optimizer"], q1, q2), "OFFLINE_TWIN_Q_Q_OPTIMIZER_STORAGE_SHARED")
    require(modules_storage_independent(q1, q2), "OFFLINE_TWIN_Q_Q_STORAGE_SHARED_AFTER_WARMUP")
    backbone_final = {
        f"{name}.{camera}": module_state_sha256(getattr(critic, camera))
        for name, critic in (("q1", q1), ("q2", q2))
        for camera in ("camera1_backbone", "camera2_backbone")
    }
    require(backbone_initial == backbone_final, "OFFLINE_TWIN_Q_FROZEN_BACKBONE_CHANGED")
    q_after_warmup = {name: module_state_sha256(module) for name, module in (
        ("q1", q1), ("q2", q2), ("q1_target", q1_target), ("q2_target", q2_target)
    )}
    require(
        q_after_warmup["q1"] != q_initial["q1"]
        and q_after_warmup["q2"] != q_initial["q2"]
        and q_after_warmup["q1_target"] != q_initial["q1_target"]
        and q_after_warmup["q2_target"] != q_initial["q2_target"],
        "OFFLINE_TWIN_Q_Q_OR_TARGET_DID_NOT_UPDATE",
    )

    evaluation["update_256"]["train_probe"] = evaluate_critic_split(
        label="update256_train_probe", rows=data.rows,
        indices=fixed["train_probe_indices"], data=data,
        fixed=fixed["train_evaluation"], policy=policy, q1=q1, q2=q2,
        q1_target=q1_target, q2_target=q2_target, train_data=data,
        device=device, batch_size=16,
    )
    evaluation["update_256"]["validation"] = evaluate_critic_split(
        label="update256_validation", rows=validation_rows,
        indices=fixed["validation_indices"], data=validation_data,
        fixed=fixed["validation_evaluation"], policy=policy, q1=q1, q2=q2,
        q1_target=q1_target, q2_target=q2_target, train_data=data,
        device=device, batch_size=16,
    )
    require(
        {name: module_state_sha256(module) for name, module in (
            ("q1", q1), ("q2", q2), ("q1_target", q1_target), ("q2_target", q2_target)
        )} == q_after_warmup,
        "OFFLINE_TWIN_Q_READONLY_DIAGNOSTIC_CHANGED_CRITICS",
    )
    print("OFFLINE_TWIN_Q_UPDATE256_DIAGNOSTICS_COMPLETE", flush=True)

    gradient_scale = measure_actor_gradient_scale(
        policy=policy, q1=q1, q2=q2, train_data=data,
        actor_indices=fixed["actor_probe_indices"], fixed=fixed,
        device=device, eta_candidates=config["diagnostics"]["eta_candidates"],
        band=config["diagnostics"]["reference_weighted_ratio_band"],
    )
    require(module_component_digests(policy) == actor_initial, "OFFLINE_TWIN_Q_ACTOR_CHANGED_AFTER_Q_SCALE_MEASUREMENT")
    require(
        {name: module_state_sha256(module) for name, module in (
            ("q1", q1), ("q2", q2), ("q1_target", q1_target), ("q2_target", q2_target)
        )} == q_after_warmup,
        "OFFLINE_TWIN_Q_Q_CHANGED_DURING_SCALE_MEASUREMENT",
    )
    ensure_all_gradients_none(policy, q1, q2, q1_target, q2_target)
    require(all(
        bool(torch.isfinite(parameter).all())
        for module in (policy, q1, q2, q1_target, q2_target)
        for parameter in module.parameters()
    ), "OFFLINE_TWIN_Q_NONFINITE_PARAMETER")

    counters = dict(CRITIC_WARMUP_COUNTERS)
    sampler_final = {name: sampler.state_dict() for name, sampler in samplers.items()}
    rng_final = capture_rng_states(generators)
    protected = json.loads(args.protected_snapshot.read_text())
    parent_actor_tree = protected["trees"].get(
        "parent_actor_checkpoint", protected["trees"].get("r5_checkpoint")
    )
    require(parent_actor_tree is not None, "OFFLINE_TWIN_Q_PARENT_ACTOR_TREE_MISSING")
    actor_binding = {
        "parent_actor_path": PARENT_ACTOR_CHECKPOINT.relative_to(ROOT).as_posix(),
        "r5_tree": parent_actor_tree,
        "state_initial": actor_initial, "state_final": module_component_digests(policy),
        "bitwise_unchanged": True, "optimizer_created": False,
        "scheduler_created": False, "optimizer_updates": 0,
        "scheduler_steps": 0, "target_actor": None,
    }
    startup = {
        "g7a/stage2_g7a_critic_warmup.development.yaml": CONFIG.read_bytes(),
        "g7a/critic_training.py": Path(__file__).read_bytes(),
        "g7a/protected_snapshot.json": args.protected_snapshot.read_bytes(),
        "reward_transitions/dataset_manifest.json": (
            REWARD_TRANSITION_ROOT / "dataset_manifest.json"
        ).read_bytes(),
    }
    rng_before_save = canonical_digest(rng_final)
    checkpoint_manifest = save_critic_warmup_checkpoint(
        args.checkpoint, critics={
            "q1": q1, "q2": q2, "q1_target": q1_target, "q2_target": q2_target,
        }, critic_optimizer=context["optimizer"], critic_scheduler=context["scheduler"],
        counters=counters, sampler_states=sampler_final, rng_states=rng_final,
        actor_binding=actor_binding, ownership_manifest=context["ownership"],
        fixed_diagnostics_manifest=fixed_manifest, protected_snapshot=protected,
        startup_snapshot_bytes=startup,
    )
    require(canonical_digest(capture_rng_states(generators)) == rng_before_save, "OFFLINE_TWIN_Q_CHECKPOINT_CONSUMED_RNG")
    validate_critic_warmup_checkpoint(args.checkpoint)

    result = {
        "worker_mode": "warmup", "environment": environment_audit(),
        "initialization": {
            "actor_source": "frozen_stage1_r5", "g5_g6_parent_loaded": False,
            "q_source": "g2_seed0_fresh", "targets_exact_initial": True,
            "target_actor": None, "q_initial": q_initial,
        },
        "resolved_recipe": config, "fixed_diagnostics": fixed_manifest,
        "evaluation": evaluation,
        "warmup_updates": reports,
        "fixed_windows": {
            "first_16": summarize_update_window(reports[:16]),
            "last_16": summarize_update_window(reports[-16:]),
        },
        "sampling_audit": summarize_sampled_rows(reports, data.rows, data),
        "sampler_initial_state_digest": canonical_digest(sampler_initial),
        "sampler_final_state_digest": canonical_digest(sampler_final),
        "sampler_initial_state_manifest": canonicalize(sampler_initial),
        "sampler_final_state_manifest": canonicalize(sampler_final),
        "rng_initial_state_digest": canonical_digest(rng_initial),
        "rng_final_state_digest": canonical_digest(rng_final),
        "rng_initial_state_manifest": canonicalize(rng_initial),
        "rng_final_state_manifest": canonicalize(rng_final),
        "gradient_scale": gradient_scale,
        "state": {
            "actor": actor_binding, "q_initial": q_initial,
            "q_after_warmup": q_after_warmup,
            "backbone_initial": backbone_initial, "backbone_final": backbone_final,
            "q1_q2_independent": True, "targets_changed_only_by_256_polyak_calls": True,
        },
        "counters": counters, "ownership": context["ownership"],
        "warmup_runtime": warmup_runtime,
        "checkpoint_manifest_payload_sha256": checkpoint_manifest["manifest_payload_sha256"],
        "checkpoint_save_rng_unchanged": True,
        "data_access_audit": {
            "train_transition_reads": len(data.rows),
            "validation_transition_reads": len(validation_rows),
            "test_transition_reads": 0, "test_image_reads": 0,
            "manual_reward_transition_files_opened": 0,
            "manual_label_files_opened": 0,
            "reward_classifier_inference_calls": 0,
            "reward_classifier_optimizer_updates": 0,
        },
        "research_limits": config["research_limits"],
    }
    require(
        not FORBIDDEN_OPENS["manual_reward_transitions"]
        and not FORBIDDEN_OPENS["manual_labels"],
        "CRITIC_WARMUP_FORBIDDEN_MANUAL_READ",
    )
    atomic_json(args.result, result)


def run_verify(args) -> None:
    from forcesmolvla.rft.exact_resume import restore_rng_states_last
    from forcesmolvla.rft.critic_warmup_checkpoint import (
        CRITIC_WARMUP_COUNTERS,
        module_component_digests,
        validate_critic_warmup_checkpoint,
    )
    from forcesmolvla.rft.critic import modules_storage_independent
    from forcesmolvla.rft.training_cycle import (
        SerializableReplacementSampler, SerializableUniqueSampler,
        ensure_all_gradients_none, optimizer_state_storage_independent,
    )

    device = configure_runtime()
    verify_config()
    manifest = validate_critic_warmup_checkpoint(args.checkpoint)
    require(
        (args.checkpoint / "startup_snapshot/g7a/stage2_g7a_critic_warmup.development.yaml").read_bytes()
        == CONFIG.read_bytes(),
        "OFFLINE_TWIN_Q_VERIFY_CONFIG_BINDING_MISMATCH",
    )
    require(
        (args.checkpoint / "startup_snapshot/g7a/critic_training.py").read_bytes()
        == Path(__file__).read_bytes(),
        "OFFLINE_TWIN_Q_VERIFY_SOURCE_BINDING_MISMATCH",
    )
    context = initialize_fresh(device=device, with_data=False)
    modules = {name: context[name] for name in ("q1", "q2", "q1_target", "q2_target")}
    for name, module in modules.items():
        state = torch.load(args.checkpoint / f"models/{name}_state.pt", map_location="cpu", weights_only=False)
        incompatible = module.load_state_dict(state, strict=True)
        require(not incompatible.missing_keys and not incompatible.unexpected_keys, f"OFFLINE_TWIN_Q_STRICT_MODEL_LOAD_FAILED:{name}")
    optimizer_state = torch.load(
        args.checkpoint / "optimizers/critic_optimizer_state.pt", map_location="cpu", weights_only=False
    )
    context["optimizer"].load_state_dict(optimizer_state)
    scheduler_state = torch.load(
        args.checkpoint / "schedulers/critic_scheduler_state.pt", map_location="cpu", weights_only=False
    )
    context["scheduler"].load_state_dict(scheduler_state)
    counters = json.loads((args.checkpoint / "state/counters.json").read_text())
    require(counters == CRITIC_WARMUP_COUNTERS, "OFFLINE_TWIN_Q_VERIFY_COUNTER_MISMATCH")
    sampler_states = torch.load(
        args.checkpoint / "state/sampler_states.pt", map_location="cpu", weights_only=False
    )
    rng_states = torch.load(
        args.checkpoint / "state/rng_states.pt", map_location="cpu", weights_only=False
    )
    generators = {}
    for name in rng_states["named_generator_states"]:
        generators[name] = named_generator(
            "cpu" if name in {"td_sampler", "calql_sampler", "empirical_random_proposal"} else "cuda",
            0,
        )
    samplers = {
        "td": SerializableUniqueSampler(
            sampler_states["td"]["name"], tuple(sampler_states["td"]["population"]),
            generators["td_sampler"], draws=int(sampler_states["td"]["draws"]),
        ),
        "calql": SerializableUniqueSampler(
            sampler_states["calql"]["name"], tuple(sampler_states["calql"]["population"]),
            generators["calql_sampler"], draws=int(sampler_states["calql"]["draws"]),
        ),
        "empirical_random_proposal": SerializableReplacementSampler(
            sampler_states["empirical_random_proposal"]["name"],
            int(sampler_states["empirical_random_proposal"]["population_size"]),
            generators["empirical_random_proposal"],
            draws=int(sampler_states["empirical_random_proposal"]["draws"]),
        ),
    }
    for name, sampler in samplers.items():
        require(sampler.draws == 256, f"OFFLINE_TWIN_Q_VERIFY_SAMPLER_DRAWS:{name}")
    sampler_to_generator = {
        "td": "td_sampler", "calql": "calql_sampler",
        "empirical_random_proposal": "empirical_random_proposal",
    }
    require(all(
        torch.equal(
            sampler_states[sampler_name]["generator_state"],
            rng_states["named_generator_states"][generator_name],
        )
        for sampler_name, generator_name in sampler_to_generator.items()
    ), "OFFLINE_TWIN_Q_VERIFY_SAMPLER_RNG_STATE_MISMATCH")
    context["actor"].eval(); context["q1"].train(True); context["q2"].train(True)
    context["q1_target"].make_permanent_eval_target()
    context["q2_target"].make_permanent_eval_target()
    ensure_all_gradients_none(context["actor"], *modules.values())
    actor_binding = json.loads((args.checkpoint / "manifests/actor_binding.json").read_text())
    require(module_component_digests(context["actor"]) == actor_binding["state_final"], "OFFLINE_TWIN_Q_VERIFY_R5_ACTOR_BINDING_MISMATCH")
    steps = {
        int(value["step"].item())
        for value in context["optimizer"].state_dict()["state"].values()
        if "step" in value
    }
    require(steps == {256}, f"OFFLINE_TWIN_Q_VERIFY_OPTIMIZER_STEP_MISMATCH:{steps}")
    require(
        context["scheduler"].last_epoch == 256
        and context["scheduler"].state_dict()["_step_count"] == 257,
        "OFFLINE_TWIN_Q_VERIFY_SCHEDULER_STEP_MISMATCH",
    )
    require(
        modules_storage_independent(context["q1"], context["q2"])
        and optimizer_state_storage_independent(context["optimizer"], context["q1"], context["q2"]),
        "OFFLINE_TWIN_Q_VERIFY_Q_STORAGE_NOT_INDEPENDENT",
    )
    require(
        not context["q1_target"].training and not context["q2_target"].training
        and not any(parameter.requires_grad for target in (context["q1_target"], context["q2_target"]) for parameter in target.parameters()),
        "OFFLINE_TWIN_Q_VERIFY_TARGET_OWNERSHIP_INVALID",
    )
    require(all(
        bool(torch.isfinite(parameter).all())
        for module in (*modules.values(), context["actor"])
        for parameter in module.parameters()
    ), "OFFLINE_TWIN_Q_VERIFY_NONFINITE_PARAMETER")
    # Restoring all global/named RNG is deliberately the final operation.
    restore_rng_states_last(rng_states, generators)
    result = {
        "worker_mode": "strict_fresh_process_load", "environment": environment_audit(),
        "checkpoint_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "strict_model_load": True, "strict_optimizer_load": True,
        "strict_scheduler_load": True, "sampler_state_loaded": True,
        "rng_restored_last": True, "random_draws_after_rng_restore": 0,
        "optimizer_steps_after_load": 256, "scheduler_steps_after_load": 256,
        "actor_loaded_from_r5_not_checkpoint": True, "actor_optimizer_created": 0,
        "actor_scheduler_created": 0, "parameter_updates": 0,
        "sampler_draws_after_load": 0, "data_transition_reads": 0,
        "test_transition_reads": 0,
        "manual_reward_transition_files_opened": 0,
        "manual_label_files_opened": 0, "reward_classifier_calls": 0,
    }
    atomic_json(args.result, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("warmup", "verify"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--fixed-diagnostics", type=Path)
    parser.add_argument("--protected-snapshot", type=Path)
    parser.add_argument("--task-id", default="task2")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--reward-transition-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    from forcesmolvla.rft import training_cycle_runtime
    from forcesmolvla.training_runtime import resolve_task_reward_transition_root

    global TASK_ID, REWARD_TRANSITION_ROOT
    TASK_ID = args.task_id
    REWARD_TRANSITION_ROOT = resolve_task_reward_transition_root(
        ROOT,
        task_id=args.task_id,
        reward_transition_root=args.reward_transition_root,
    )
    training_cycle_runtime.configure_task_paths(
        task_id=args.task_id,
        dataset_root=args.dataset_root,
        reward_transition_root=REWARD_TRANSITION_ROOT,
        output_root=args.output_root,
    )
    require(CONFIG.is_file(), "OFFLINE_TWIN_Q_STARTUP_CONFIG_MISSING")
    flow_sampling.critic_action_for_q_guidance = audited_action_contract_v2_adapter
    losses.critic_action_for_q_guidance = audited_action_contract_v2_adapter
    forbidden = RuntimeError("OFFLINE_TWIN_Q_PUBLIC_EXECUTION_PATH_CALLED")
    with (
        patch.object(
            action_delta.ActionDeltaProcessor,
            "from_delta",
            side_effect=forbidden,
        ) as inverse,
        patch.object(
            action_delta.ActionSafetyProfile,
            "validate_chunk",
            side_effect=forbidden,
        ) as validator,
        patch.object(
            action_delta,
            "decode_binary_gripper_width",
            side_effect=forbidden,
        ) as decoder,
        patch.object(
            rules,
            "load_and_validate_rulespec",
            side_effect=forbidden,
        ) as rulespec,
        patch.object(
            ForceSmolVLAPolicy,
            "predict_action_chunk",
            side_effect=forbidden,
        ) as predict,
    ):
        if args.mode == "warmup":
            require(
                args.fixed_diagnostics is not None
                and args.protected_snapshot is not None,
                "OFFLINE_TWIN_Q_WARMUP_INPUT_MISSING",
            )
            run_warmup(args)
        else:
            run_verify(args)
    calls = {
        "public_validator_calls": validator.call_count,
        "absolute_inverse_calls": inverse.call_count,
        "public_binary_decoder_calls": decoder.call_count,
        "RuleSpec_calls": rulespec.call_count,
        "predict_action_chunk_calls": predict.call_count,
    }
    require(not any(calls.values()), f"OFFLINE_TWIN_Q_PUBLIC_PATH_CALL:{calls}")
    finalize_action_contract_v2_result(args.result, calls)


if __name__ == "__main__":
    main()
