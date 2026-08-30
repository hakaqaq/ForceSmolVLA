#!/usr/bin/env python3
"""Fresh-process Frozen-VLM Stage-2 Actor/Critic/joint batch benchmark."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import types
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml


ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(ROOT / "tools"))
CONFIG = ROOT / "configs/stage2_batch_scaling.development.yaml"
CONTRACT = ROOT / "configs/stage2_trainability_contract.v2.development.json"
OUTPUT = ROOT / "artifacts/development/stage2/batch_scaling/stage2"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size and np.isfinite(array).all(), "STAGE2_BATCH_STAT_INVALID")
    return {
        "count": int(array.size), "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)), "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()), "maximum": float(array.max()),
        "range": float(array.max() - array.min()),
    }


class GpuTelemetry:
    def __init__(self) -> None:
        self.utilization: list[float] = []
        self.power: list[float] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self):
        def collect() -> None:
            while not self.stop.is_set():
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        util, power = result.stdout.splitlines()[0].split(",", 1)
                        self.utilization.append(float(util)); self.power.append(float(power))
                    except ValueError:
                        pass
                self.stop.wait(0.2)
        self.thread = threading.Thread(target=collect, daemon=True); self.thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


class TimedFlowCounter:
    def __init__(self, inference_batch_size: int) -> None:
        from forcesmolvla.rft.training_cycle import FlowCounter
        self.inner = FlowCounter(inference_batch_size=inference_batch_size)
        self.seconds = {"td_next": 0.0, "cql_current": 0.0, "cql_next": 0.0, "actor_guidance": 0.0}

    def sample(self, *args, purpose: str, **kwargs):
        torch.cuda.synchronize(); started = time.perf_counter()
        value = self.inner.sample(*args, purpose=purpose, **kwargs)
        torch.cuda.synchronize(); self.seconds[purpose] += time.perf_counter() - started
        return value

    def report(self):
        return self.inner.report()


def configure_runtime() -> torch.device:
    from forcesmolvla.rft import critic_training as g7a
    return g7a.configure_runtime()


def load_context(device: torch.device):
    from forcesmolvla.rft import critic_training as g7a
    import run_s2_g7b_worker as g7b
    from forcesmolvla.rft.frozen_vlm_trainability import (
        apply_frozen_vlm_trainability, build_frozen_vlm_actor_optimizer,
    )

    context = g7a.initialize_fresh(device=device, with_data=True)
    parent_sampler_states, parent_rng = g7b.load_parent(context)
    trainability = apply_frozen_vlm_trainability(context["actor"])
    actor_optimizer, actor_scheduler, actor_ownership = build_frozen_vlm_actor_optimizer(context["actor"])
    training = yaml.safe_load((ROOT / "configs/stage2_g5_single_cycle.v2.development.yaml").read_text())
    training = copy.deepcopy(training); training["loss"]["eta_actor_q"] = 10.0
    generators = g7b.build_generators(training)
    samplers = g7b.build_samplers(context["data"], generators, parent_sampler_states)
    g7b.restore_parent_rng(parent_rng, generators)
    return context, training, generators, samplers, actor_optimizer, actor_scheduler, actor_ownership, trainability


def gradient_groups(policy) -> dict[str, float]:
    from forcesmolvla.rft.frozen_vlm_trainability import gradient_norm_for_prefixes
    groups = {
        "frozen_vlm": ("model.vlm_with_expert.vlm.",),
        "frozen_state_prefix": ("model.state_proj.",),
        "force": ("model.force_branch.", "model.force_adapter."),
        "action_expert": ("model.vlm_with_expert.lm_expert.",),
        "action_io": (
            "model.action_in_proj.", "model.action_out_proj.",
            "model.action_time_mlp_in.", "model.action_time_mlp_out.",
        ),
        "router": ("model.force_branch.refiner.router.",),
    }
    return {name: gradient_norm_for_prefixes(policy, prefixes) for name, prefixes in groups.items()}


def actor_update(
    *, context: dict, batch: dict, optimizer, scheduler, actor_batch_size: int,
    flow_noise_generator: torch.Generator, flow_time_generator: torch.Generator,
    actor_q_generator: torch.Generator, update_id: str,
) -> dict:
    from forcesmolvla.force_token import RouterState
    from forcesmolvla.rft.critic_action_adapter_v2 import raw_gripper_out_of_public_tolerance_mask
    from forcesmolvla.rft.frozen_vlm_trainability import (
        compute_min_twin_q_actor_loss, frozen_prefix_flow_matching_terms,
    )
    from forcesmolvla.rft.training_cycle import global_gradient_norm, gradients_finite
    from forcesmolvla.router_training import collect_pass_a_statistics, microbatch_two_pass_terms

    policy, q1, q2 = context["actor"], context["q1"], context["q2"]
    device = batch["reward"].device
    trainable = [value for value in policy.parameters() if value.requires_grad]
    frozen = [value for value in policy.parameters() if not value.requires_grad]
    optimizer.zero_grad(set_to_none=True); policy.train(True)
    noise = torch.randn(actor_batch_size, 50, 7, generator=flow_noise_generator, device=device)
    timestep = torch.rand(actor_batch_size, generator=flow_time_generator, device=device)
    velocity_outputs = []
    hook = policy.model.action_out_proj.register_forward_hook(
        lambda _module, _inputs, output: (output.retain_grad(), velocity_outputs.append(output))[-1]
    )
    torch.cuda.synchronize(); fm_started = time.perf_counter()
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses, feature_mask, router_state, prefix_audit = frozen_prefix_flow_matching_terms(
                policy, batch["current_actor_batch"], noise=noise, time=timestep,
                call_id=f"{update_id}-fm",
            )
            detached = RouterState(
                logits_fp32=router_state.logits_fp32.detach(),
                probabilities_fp32=router_state.probabilities_fp32.detach(),
                route_ids=router_state.route_ids.detach(),
                valid_mask=router_state.valid_mask.detach(),
            )
            statistics = collect_pass_a_statistics([detached], [feature_mask])
            auxiliary = microbatch_two_pass_terms(losses, router_state, statistics)
            fm_loss = losses.sum() / feature_mask.sum().clamp_min(1)
            fm_objective = fm_loss + 0.01 * auxiliary.balance + 0.001 * auxiliary.z
        fm_objective.backward()
    finally:
        hook.remove()
    torch.cuda.synchronize(); fm_seconds = time.perf_counter() - fm_started
    require(len(velocity_outputs) == 1 and velocity_outputs[0].grad is not None, "STAGE2_BATCH_FM_GRADIENT")
    gripper_fm = float(velocity_outputs[0].grad[..., 6].float().norm().cpu())
    fm_groups = gradient_groups(policy)

    policy.eval()
    q_noise = torch.randn(actor_batch_size, 50, 7, generator=actor_q_generator, device=device)
    flow_counter = TimedFlowCounter(inference_batch_size=actor_batch_size)
    torch.cuda.synchronize(); q_started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        action_chunk = flow_counter.sample(
            policy, batch["current_actor_batch"], q_noise,
            call_id=f"{update_id}-q", purpose="actor_guidance",
        )
        action_chunk.retain_grad()
        q_loss, q1_value, q2_value, critic_action = compute_min_twin_q_actor_loss(
            q1=q1, q2=q2, observation=batch["current_observation"],
            normalized_flow_action_chunk7=action_chunk,
            delta_action_mean7=batch["delta_mean"], delta_action_std7=batch["delta_std"],
        )
        weighted_q = 10.0 * q_loss
    weighted_q.backward()
    torch.cuda.synchronize(); q_seconds = time.perf_counter() - q_started
    require(action_chunk.grad is not None, "STAGE2_BATCH_Q_ACTION_GRADIENT")
    tcp6_q = float(action_chunk.grad[:, :3, :6].float().norm().cpu())
    gripper_q = float(action_chunk.grad[:, :3, 6].float().abs().max().cpu())
    raw_out = float(raw_gripper_out_of_public_tolerance_mask(
        action_chunk[:, :3, 6].detach(),
        gripper_mean=batch["delta_mean"][6], gripper_std=batch["delta_std"][6],
    ).float().mean().cpu())
    combined_groups = gradient_groups(policy)
    require(
        fm_groups["frozen_vlm"] == fm_groups["frozen_state_prefix"] == 0.0
        and combined_groups["frozen_vlm"] == combined_groups["frozen_state_prefix"] == 0.0
        and all(combined_groups[name] > 0.0 for name in ("force", "action_expert", "action_io", "router"))
        and tcp6_q > 0.0 and gripper_q == 0.0 and gripper_fm > 0.0
        and all(value.grad is None for value in frozen)
        and gradients_finite(trainable),
        "STAGE2_BATCH_ACTOR_GRADIENT_CONTRACT",
    )
    preclip = float(global_gradient_norm(trainable).cpu())
    torch.cuda.synchronize(); optimizer_started = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(trainable, 10.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); optimizer_seconds = time.perf_counter() - optimizer_started
    policy.eval()
    total_value = float((fm_objective + weighted_q).detach().cpu())
    require(all(math.isfinite(value) for value in (
        float(fm_loss.detach()), float(q_loss.detach()), total_value, preclip,
        tcp6_q, gripper_fm,
        float(q1_value.mean().detach()), float(q2_value.mean().detach()),
    )), "STAGE2_BATCH_ACTOR_NONFINITE")
    return {
        "loss": {
            "flow_matching": float(fm_loss.detach().cpu()),
            "actor_q_min_twin": float(q_loss.detach().cpu()),
            "balance": float(auxiliary.balance.detach().cpu()),
            "z": float(auxiliary.z.detach().cpu()),
            "weighted_total": total_value,
        },
        "q": {"q1_mean": float(q1_value.mean().detach().cpu()), "q2_mean": float(q2_value.mean().detach().cpu())},
        "gradient": {
            "tcp6_q_norm": tcp6_q, "gripper_q_max_abs": gripper_q,
            "gripper_fm_norm": gripper_fm, "preclip_global_norm": preclip,
            "fm_groups": fm_groups, "combined_groups": combined_groups,
        },
        "timing": {
            "flow_matching_forward_backward": fm_seconds,
            "differentiable_n10_flow_twin_q_actor_q_backward": q_seconds,
            "actor_optimizer": optimizer_seconds,
        },
        "prefix_audit": prefix_audit,
        "flow_counts": flow_counter.report(),
        "critic_action_shape": list(critic_action.shape),
        "raw_gripper_out_of_public_tolerance_rate": raw_out,
    }


def critic_update(*, context, training, generators, samplers, batch_size: int, update_id: int) -> dict:
    from forcesmolvla.rft import training_cycle as g5
    from forcesmolvla.rft import training_cycle

    data, policy, q1 = context["data"], context["actor"], context["q1"]
    td_indices = samplers["td"].draw(batch_size)
    calql_indices = samplers["calql"].draw(batch_size)
    load_started = time.perf_counter()
    td_batch = data.build_batch(td_indices, policy, q1.canonical_task_feature.device, canonical_task_feature=q1.canonical_task_feature)
    calql_batch = data.build_batch(calql_indices, policy, q1.canonical_task_feature.device, canonical_task_feature=q1.canonical_task_feature)
    data_seconds = time.perf_counter() - load_started
    configured = copy.deepcopy(training); configured["batching"]["critic_batch_size"] = batch_size
    flow_counter = TimedFlowCounter(inference_batch_size=4)
    timings = {"optimizer": 0.0, "polyak": 0.0, "scheduler": 0.0}
    original_step = context["optimizer"].step
    original_scheduler = context["scheduler"].step
    original_polyak = training_cycle.polyak_update_verified

    def timed_optimizer(*args, **kwargs):
        torch.cuda.synchronize(); started = time.perf_counter(); value = original_step(*args, **kwargs)
        torch.cuda.synchronize(); timings["optimizer"] += time.perf_counter() - started; return value

    def timed_scheduler(*args, **kwargs):
        started = time.perf_counter(); value = original_scheduler(*args, **kwargs)
        timings["scheduler"] += time.perf_counter() - started; return value

    def timed_polyak(*args, **kwargs):
        torch.cuda.synchronize(); started = time.perf_counter(); value = original_polyak(*args, **kwargs)
        torch.cuda.synchronize(); timings["polyak"] += time.perf_counter() - started; return value

    with (
        patch.object(context["optimizer"], "step", side_effect=timed_optimizer),
        patch.object(context["scheduler"], "step", side_effect=timed_scheduler),
        patch.object(training_cycle, "polyak_update_verified", side_effect=timed_polyak),
    ):
        report = g5.critic_update(
            step=update_id, policy=policy, q1=context["q1"], q2=context["q2"],
            q1_target=context["q1_target"], q2_target=context["q2_target"],
            optimizer=context["optimizer"], scheduler=context["scheduler"],
            td_batch=td_batch, calql_batch=calql_batch, train_data=data,
            proposal_sampler=samplers["empirical_random_proposal"], generators=generators,
            flow_counter=flow_counter, config=configured,
        )
    candidate_total = float(report["latency_seconds"]["candidate_sampling"])
    policy_sampling = sum(flow_counter.seconds.values())
    critic_total = float(report["latency_seconds"]["critic_forward_backward_step_polyak_scheduler"])
    return {
        "loss": report["loss"], "statistics": report["statistics"],
        "gradient": report["gradient"], "terminal_rows": report["terminal_rows"],
        "timing": {
            "data_loading": data_seconds,
            "td_next_action_sampling": flow_counter.seconds["td_next"],
            "calql_current_policy_sampling": flow_counter.seconds["cql_current"],
            "calql_next_policy_sampling": flow_counter.seconds["cql_next"],
            "calql_empirical_proposal_and_overhead": max(0.0, candidate_total - policy_sampling),
            "q_forward_backward_excluding_optimizer_polyak": max(0.0, critic_total - timings["optimizer"] - timings["polyak"] - timings["scheduler"]),
            "optimizer": timings["optimizer"], "polyak": timings["polyak"],
            "scheduler": timings["scheduler"],
        },
        "flow_counts": flow_counter.report(),
        "row_identities": {"td": report["td_batch"], "calql": report["calql_batch"]},
    }


def public_audit(context: dict, batch: dict, noise: torch.Tensor, cycle: int) -> dict:
    import run_s2_g7b_worker as g7b
    started = time.perf_counter()
    try:
        result = g7b.public_diagnostic(context["actor"], context["data"], batch["current_actor_batch"], noise, cycle)
        result["semantic_success"] = True
    except Exception as error:
        result = {"semantic_success": False, "error_type": type(error).__name__, "error": str(error)}
    result["latency_seconds"] = time.perf_counter() - started
    return result


def worker(candidate: dict) -> dict:
    from forcesmolvla.rft.frozen_vlm_trainability import frozen_state_digest
    from forcesmolvla.rft.training_cycle import module_state_sha256

    device = configure_runtime()
    context, training, generators, samplers, actor_optimizer, actor_scheduler, ownership, manifest = load_context(device)
    mode = candidate["mode"]
    actor_batch_size = int(candidate["actor_batch_size"])
    critic_batch_size = int(candidate["critic_batch_size"])
    frozen_before = frozen_state_digest(context["actor"])
    actor_initial = module_state_sha256(context["actor"])
    critics_initial = {name: module_state_sha256(context[name]) for name in ("q1", "q2", "q1_target", "q2_target")}
    public_indices = list(context["data"].actor_population[:1])
    public_batch = context["data"].build_batch(public_indices, context["actor"], device, canonical_task_feature=context["q1"].canonical_task_feature, include_flow_actions=True)
    public_noise = torch.randn(1, 50, 7, generator=torch.Generator(device=device).manual_seed(7440), device=device)
    public_before = public_audit(context, public_batch, public_noise, 0)
    require(public_before["semantic_success"], f"STAGE2_BATCH_PUBLIC_BEFORE_FAILED:{public_before}")
    records = []
    telemetry = None
    prefix_prefill_times: list[float] = []
    original_encode_prefix = context["actor"].model.encode_prefix

    def timed_encode_prefix(_model, *args, **kwargs):
        torch.cuda.synchronize(); started = time.perf_counter()
        value = original_encode_prefix(*args, **kwargs)
        torch.cuda.synchronize(); prefix_prefill_times.append(time.perf_counter() - started)
        return value

    context["actor"].model.encode_prefix = types.MethodType(
        timed_encode_prefix, context["actor"].model
    )
    torch.cuda.reset_peak_memory_stats(device)
    for local in range(4):
        prefix_start = len(prefix_prefill_times)
        cycle_started = time.perf_counter(); parts = {"critic": [], "actor": None}
        if local == 1:
            telemetry = GpuTelemetry().__enter__()
        if mode in {"critic", "joint"}:
            critic_count = 2 if mode == "joint" else 1
            for substep in range(critic_count):
                parts["critic"].append(critic_update(
                    context=context, training=training, generators=generators,
                    samplers=samplers, batch_size=critic_batch_size,
                    update_id=257 + local * critic_count + substep,
                ))
        if mode in {"actor", "joint"}:
            draw = samplers["actor"].draw(actor_batch_size)
            load_started = time.perf_counter()
            batch = context["data"].build_batch(
                draw, context["actor"], device, canonical_task_feature=context["q1"].canonical_task_feature,
                include_flow_actions=True,
            )
            data_seconds = time.perf_counter() - load_started
            parts["actor"] = actor_update(
                context=context, batch=batch, optimizer=actor_optimizer, scheduler=actor_scheduler,
                actor_batch_size=actor_batch_size,
                flow_noise_generator=generators["flow_matching_noise"],
                flow_time_generator=generators["flow_matching_timestep"],
                actor_q_generator=generators["actor_q_flow_noise"],
                update_id=f"{candidate['candidate_id']}-cycle{local}",
            )
            parts["actor"]["timing"]["data_loading"] = data_seconds
        torch.cuda.synchronize(); cycle_seconds = time.perf_counter() - cycle_started
        records.append({
            "local_cycle": local, "warmup": local == 0,
            "cycle_seconds": cycle_seconds,
            "frozen_prefix_prefill_seconds_embedded": sum(prefix_prefill_times[prefix_start:]),
            "frozen_prefix_prefill_call_count": len(prefix_prefill_times) - prefix_start,
            **parts,
        })
        print(f"STAGE2_BATCH {candidate['candidate_id']} cycle={local + 1}/4", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    context["actor"].model.encode_prefix = original_encode_prefix
    require(telemetry is not None, "STAGE2_BATCH_TELEMETRY_NOT_STARTED"); telemetry.__exit__(None, None, None)
    public_after = public_audit(context, public_batch, public_noise, 1)
    require(public_after["semantic_success"], "STAGE2_BATCH_PUBLIC_AFTER_FAILED")
    frozen_after = frozen_state_digest(context["actor"])
    require(frozen_before == frozen_after, "STAGE2_BATCH_FROZEN_HASH_CHANGED")
    actor_final = module_state_sha256(context["actor"])
    critics_final = {name: module_state_sha256(context[name]) for name in ("q1", "q2", "q1_target", "q2_target")}
    expected_actor_updates = 4 if mode in {"actor", "joint"} else 0
    expected_critic_updates = (8 if mode == "joint" else 4 if mode == "critic" else 0)
    require((actor_initial != actor_final) == bool(expected_actor_updates), "STAGE2_BATCH_ACTOR_CHANGE_MATRIX")
    require(all((critics_initial[name] != critics_final[name]) == bool(expected_critic_updates) for name in critics_final), "STAGE2_BATCH_CRITIC_CHANGE_MATRIX")
    measured = records[1:]
    actor_samples = actor_batch_size * 3 if mode in {"actor", "joint"} else 0
    critic_samples = critic_batch_size * 3 * (2 if mode == "joint" else 1) if mode in {"critic", "joint"} else 0
    elapsed = sum(item["cycle_seconds"] for item in measured)
    all_actor = [item["actor"] for item in measured if item["actor"] is not None]
    all_critics = [sub for item in measured for sub in item["critic"]]
    contract_valid = (
        all(item["gradient"]["tcp6_q_norm"] > 0.0 and item["gradient"]["gripper_q_max_abs"] == 0.0 and item["gradient"]["gripper_fm_norm"] > 0.0 for item in all_actor)
        and frozen_before == frozen_after and public_before["semantic_success"] and public_after["semantic_success"]
    )
    return {
        "schema_version": "forcesmolvla_stage2_batch_candidate.v2",
        "status": "pass", "candidate_id": candidate["candidate_id"], "repeat": candidate["repeat"],
        "mode": mode, "actor_physical_batch_size": actor_batch_size,
        "critic_physical_batch_size": critic_batch_size,
        "warmup_cycles": 1, "measured_cycles": 3,
        "actor_transitions_per_second": actor_samples / elapsed if actor_samples else 0.0,
        "critic_transitions_per_second": critic_samples / elapsed if critic_samples else 0.0,
        "joint_cycles_per_hour": 3.0 / elapsed * 3600 if mode == "joint" else 0.0,
        "seconds_per_cycle": describe([item["cycle_seconds"] for item in measured]),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "total_gpu_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "gpu_utilization_percent": describe(telemetry.utilization or [0.0]),
        "gpu_power_watts": describe(telemetry.power or [0.0]),
        "all_finite": True, "contract_valid": contract_valid,
        "frozen_parameter_hash_unchanged": frozen_before == frozen_after,
        "public_inference": {"before": public_before, "after": public_after, "behavior_implementation_changed": False},
        "parameter_updates": {
            "actor": expected_actor_updates, "critic": expected_critic_updates,
            "polyak_per_target": expected_critic_updates,
        },
        "trainability_counts": {
            "frozen": manifest.frozen_parameter_count,
            "actor": manifest.trainable_actor_parameter_count,
            "critic": sum(value.numel() for name in ("q1", "q2") for value in context[name].parameters() if value.requires_grad),
        },
        "actor_optimizer_ownership": ownership,
        "measured_records": measured,
        "candidate_state_discarded": True, "checkpoint_created": False,
        "access_audit": {"validation_reads": 0, "test_reads": 0, "manual_g1_opens": 0, "manual_label_opens": 0, "reward_classifier_inference": 0},
    }


def run_worker(candidate: dict, result_path: Path) -> dict:
    config_path = result_path.with_suffix(".config.json"); atomic_json(config_path, candidate)
    environment = os.environ.copy(); environment.update({
        "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT / 'tools'}:{ROOT}",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    })
    completed = subprocess.run(
        [sys.executable, __file__, "--worker", "--candidate", str(config_path), "--result", str(result_path)],
        cwd=ROOT, env=environment, check=False,
    )
    require(completed.returncode == 0 or result_path.is_file(), f"STAGE2_BATCH_WORKER_FAILED:{candidate['candidate_id']}:{completed.returncode}")
    return json.loads(result_path.read_text())


def aggregate(mode: str, actor_batch: int, critic_batch: int, repeats: list[dict]) -> dict:
    passed = [item for item in repeats if item["status"] == "pass"]
    result = {
        "mode": mode, "actor_physical_batch_size": actor_batch,
        "critic_physical_batch_size": critic_batch, "repeat_count": len(repeats),
        "pass_count": len(passed), "oom_count": sum(item["status"] == "oom" for item in repeats),
        "all_pass": len(passed) == len(repeats),
        "all_finite": bool(passed) and all(item["all_finite"] for item in passed),
        "all_contract_valid": bool(passed) and all(item["contract_valid"] for item in passed),
        "peak_allocated_bytes": max((item.get("peak_allocated_bytes", 0) for item in passed), default=0),
        "peak_reserved_bytes": max((item.get("peak_reserved_bytes", 0) for item in passed), default=0),
        "result_paths": [item["result_path"] for item in repeats],
    }
    if passed:
        for key in ("actor_transitions_per_second", "critic_transitions_per_second", "joint_cycles_per_hour"):
            result[key] = describe([float(item[key]) for item in passed])
        result["seconds_per_cycle"] = describe([float(item["seconds_per_cycle"]["median"]) for item in passed])
        result["gpu_utilization_percent"] = describe([float(item["gpu_utilization_percent"]["mean"]) for item in passed])
        result["gpu_power_watts"] = describe([float(item["gpu_power_watts"]["mean"]) for item in passed])
    return result


def coordinator() -> None:
    from forcesmolvla.rft.batch_scaling import eligible

    require(not OUTPUT.exists(), "STAGE2_BATCH_OUTPUT_EXISTS"); OUTPUT.mkdir(parents=True)
    config = yaml.safe_load(CONFIG.read_text())
    total_memory = int(subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()[0]) * 1024 * 1024
    aggregates = {"actor": [], "critic": [], "joint": []}

    def run_candidate(mode: str, actor_batch: int, critic_batch: int) -> dict:
        repeats = []
        for repeat in range(1, 4):
            candidate_id = f"{mode}_a{actor_batch}_c{critic_batch}_repeat{repeat}"
            candidate = {"candidate_id": candidate_id, "mode": mode, "actor_batch_size": actor_batch, "critic_batch_size": critic_batch, "repeat": repeat}
            path = OUTPUT / "candidate_results" / f"{candidate_id}.json"
            result = run_worker(candidate, path); result["result_path"] = path.relative_to(ROOT).as_posix()
            repeats.append(result); print(f"STAGE2_BATCH_RESULT {candidate_id} {result['status']}", flush=True)
        value = aggregate(mode, actor_batch, critic_batch, repeats); aggregates[mode].append(value); return value

    actor_results = []
    for batch in config["benchmark"]["actor_mandatory_physical_batches"]:
        actor_results.append(run_candidate("actor", int(batch), 16))
    b8, b16 = actor_results[1], actor_results[2]
    expand_actor = eligible(b16, total_memory_bytes=total_memory) and b16["actor_transitions_per_second"]["median"] >= 1.05 * b8["actor_transitions_per_second"]["median"]
    if expand_actor:
        previous = b16
        for batch in config["benchmark"]["actor_conditional_physical_batches"]:
            current = run_candidate("actor", int(batch), 16); actor_results.append(current); previous = current
        b32 = actor_results[-1]
        if eligible(b32, total_memory_bytes=total_memory) and b32["actor_transitions_per_second"]["median"] >= 1.05 * actor_results[-2]["actor_transitions_per_second"]["median"]:
            for batch in config["benchmark"]["actor_extended_conditional_physical_batches"]:
                current = run_candidate("actor", int(batch), 16); actor_results.append(current)
                if not eligible(current, total_memory_bytes=total_memory) or current["actor_transitions_per_second"].get("median", 0) < 1.05 * previous["actor_transitions_per_second"]["median"]:
                    break
                previous = current

    critic_results = []
    for batch in config["benchmark"]["critic_mandatory_physical_batches"]:
        critic_results.append(run_candidate("critic", 4, int(batch)))
    b64, b128 = critic_results[1], critic_results[2]
    if eligible(b128, total_memory_bytes=total_memory) and b128["critic_transitions_per_second"]["median"] >= 1.05 * b64["critic_transitions_per_second"]["median"]:
        for batch in config["benchmark"]["critic_conditional_physical_batches"]:
            critic_results.append(run_candidate("critic", 4, int(batch)))

    valid_actor = [item for item in actor_results if eligible(item, total_memory_bytes=total_memory)]
    valid_critic = [item for item in critic_results if eligible(item, total_memory_bytes=total_memory)]
    require(valid_actor and valid_critic, "STAGE2_BATCH_NO_VALID_INDIVIDUAL_CANDIDATE")
    best_actor = max(valid_actor, key=lambda item: item["actor_transitions_per_second"]["median"])
    best_critic = max(valid_critic, key=lambda item: item["critic_transitions_per_second"]["median"])
    actor_ranked = sorted(valid_actor, key=lambda item: item["actor_transitions_per_second"]["median"], reverse=True)
    critic_ranked = sorted(valid_critic, key=lambda item: item["critic_transitions_per_second"]["median"], reverse=True)
    combinations = []
    for pair in (
        (best_actor, best_critic),
        (actor_ranked[min(1, len(actor_ranked) - 1)], best_critic),
        (best_actor, critic_ranked[min(1, len(critic_ranked) - 1)]),
    ):
        key = (pair[0]["actor_physical_batch_size"], pair[1]["critic_physical_batch_size"])
        if key not in combinations:
            combinations.append(key)
    joint_results = [run_candidate("joint", actor_batch, critic_batch) for actor_batch, critic_batch in combinations]
    valid_joint = [item for item in joint_results if eligible(item, total_memory_bytes=total_memory)]
    require(valid_joint, "STAGE2_BATCH_NO_VALID_JOINT_CANDIDATE")
    selected = max(valid_joint, key=lambda item: (
        item["actor_transitions_per_second"]["median"] + item["critic_transitions_per_second"]["median"]
    ))
    summary = {
        "schema_version": "forcesmolvla_stage2_batch_scaling_summary.v2",
        "status": "pass", "config": CONFIG.relative_to(ROOT).as_posix(),
        "trainability_contract": CONTRACT.relative_to(ROOT).as_posix(),
        "candidate_aggregates": aggregates,
        "recommended_actor_physical_batch": selected["actor_physical_batch_size"],
        "recommended_critic_physical_batch": selected["critic_physical_batch_size"],
        "recommended_actor_transitions_per_second": selected["actor_transitions_per_second"]["median"],
        "recommended_critic_transitions_per_second": selected["critic_transitions_per_second"]["median"],
        "recommended_joint_cycles_per_hour": selected["joint_cycles_per_hour"]["median"],
        "recommended_joint_result": selected,
        "selection_rule": "highest_stable_valid_sample_throughput_under_85pct_vram; prefer_smaller_when_increment_under_5pct",
        "total_gpu_memory_bytes": total_memory,
        "long_run_started": False, "checkpoint_created": False,
    }
    atomic_json(OUTPUT / "stage2_summary.json", summary)
    print("STAGE2_BATCH_SCALING complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true"); parser.add_argument("--worker", action="store_true")
    parser.add_argument("--candidate", type=Path); parser.add_argument("--result", type=Path); args = parser.parse_args()
    if args.worker:
        require(args.candidate and args.result and not args.result.exists(), "STAGE2_BATCH_WORKER_ARGUMENTS")
        candidate = json.loads(args.candidate.read_text())
        try:
            result = worker(candidate)
        except BaseException as error:
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
            if not is_oom:
                raise
            result = {
                "schema_version": "forcesmolvla_stage2_batch_candidate.v2", "status": "oom",
                "candidate_id": candidate["candidate_id"], "repeat": candidate["repeat"],
                "mode": candidate["mode"], "actor_physical_batch_size": candidate["actor_batch_size"],
                "critic_physical_batch_size": candidate["critic_batch_size"], "error": str(error),
                "all_finite": False, "contract_valid": False, "checkpoint_created": False,
                "candidate_state_discarded": True,
            }
        atomic_json(args.result, result)
    else:
        require(args.run, "pass --run"); coordinator()


if __name__ == "__main__":
    main()
