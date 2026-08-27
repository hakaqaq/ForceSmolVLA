#!/usr/bin/env python3
"""Run one disposable fresh-process Stage-2 batch benchmark candidate."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(ROOT / "tools"))


def require(value: bool, message: str) -> None:
    if not value:
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


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0 and np.isfinite(array).all(), "BENCHMARK_STATISTIC_INPUT_INVALID")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


class GpuUtilizationSampler:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.values: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        def collect() -> None:
            while not self._stop.is_set():
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        self.values.append(float(result.stdout.strip().splitlines()[0]))
                    except ValueError:
                        pass
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=collect, name="gpu-utilization-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class TimedFlowCounter:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.seconds = Counter()

    def sample(self, policy, batch, noise7, *, call_id: str, purpose: str):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = self.delegate.sample(policy, batch, noise7, call_id=call_id, purpose=purpose)
        torch.cuda.synchronize()
        self.seconds[purpose] += time.perf_counter() - started
        return result

    def report(self) -> dict:
        report = self.delegate.report()
        report["seconds_by_purpose"] = dict(sorted(self.seconds.items()))
        return report


def _slice_observation(observation, start: int, stop: int, size: int, device: torch.device):
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[start:stop] = True
    return observation.index(mask)


def actor_update_batched(
    *, policy, q1, q2, q1_target, q2_target, optimizer, scheduler,
    actor_batch: dict, generators: dict[str, torch.Generator], flow_counter,
    config: dict, candidate_id: str,
) -> dict:
    """Frozen G5 objective with a variable physical microbatch layout."""
    import preflight_s2_g5_single_cycle_gpu as g5
    from forcesmolvla.rft.losses import build_actor_q_action, compute_actor_q_loss
    from forcesmolvla.rft.training_cycle import global_gradient_norm, gradients_finite, module_state_sha256

    device = actor_batch["reward"].device
    microbatch = int(config["batching"]["actor_microbatch_size"])
    accumulation = int(config["batching"]["actor_gradient_accumulation"])
    effective = int(config["batching"]["actor_effective_batch_size"])
    eta = float(config["loss"]["eta_actor_q"])
    require(microbatch * accumulation == effective == len(actor_batch["indices"]), "BENCHMARK_ACTOR_LAYOUT_INVALID")
    critic_before = {
        "q1": module_state_sha256(q1), "q2": module_state_sha256(q2),
        "q1_target": module_state_sha256(q1_target), "q2_target": module_state_sha256(q2_target),
    }
    actor_before = module_state_sha256(policy)
    actor_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    total_valid_features = int(actor_batch["current_actor_batch"]["action_valid_mask"].sum().cpu()) * 7
    require(total_valid_features > 0, "BENCHMARK_ACTOR_WINDOW_NO_VALID_FEATURES")
    records = []
    fm_window_sum = actor_q_sum = balance_sum = z_sum = 0.0
    tcp_q_gradient_square = gripper_fm_gradient_square = 0.0
    gripper_q_gradient_max = 0.0
    fm_latency = q_latency = 0.0
    q1_action_values, q2_action_values = [], []

    for micro_index in range(accumulation):
        start, stop = micro_index * microbatch, (micro_index + 1) * microbatch
        micro_actor = g5.slice_actor_batch(actor_batch["current_actor_batch"], start, stop)
        micro_observation = _slice_observation(actor_batch["current_observation"], start, stop, effective, device)
        fm_noise = torch.randn(microbatch, 50, 7, generator=generators["flow_matching_noise"], device=device)
        fm_time = torch.rand(microbatch, generator=generators["flow_matching_timestep"], device=device)
        velocity_outputs = []

        def capture(_module, _inputs, output):
            output.retain_grad()
            velocity_outputs.append(output)

        hook = policy.model.action_out_proj.register_forward_hook(capture)
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            policy.train(True)
            losses, feature_mask, terms, router_state = g5.flow_microbatch_terms(policy, micro_actor, fm_noise, fm_time)
            fm_contribution = losses.sum() / total_valid_features
            auxiliary = 0.01 * terms.balance / accumulation + 0.001 * terms.z / accumulation
            (fm_contribution + auxiliary).backward()
        finally:
            hook.remove()
        torch.cuda.synchronize()
        fm_latency += time.perf_counter() - started
        require(len(velocity_outputs) == 1 and velocity_outputs[0].grad is not None, "BENCHMARK_FM_GRADIENT_HOOK_FAILED")
        gripper_fm_gradient_square += float(velocity_outputs[0].grad[..., 6].float().square().sum().cpu())

        q_noise = torch.randn(microbatch, 50, 7, generator=generators["actor_q_flow_noise"], device=device)
        policy.eval()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action_chunk = flow_counter.sample(
                policy, micro_actor, q_noise,
                call_id=f"benchmark-{candidate_id}-actor-micro={micro_index}", purpose="actor_guidance",
            )
            action_chunk.retain_grad()
            actor_q_loss = compute_actor_q_loss(
                q1=q1, q2=q2, current_observation=micro_observation,
                actor_action_chunk7=action_chunk,
                actor_q_valid=torch.ones(microbatch, dtype=torch.bool, device=device),
                delta_action_mean7=actor_batch["delta_mean"], delta_action_std7=actor_batch["delta_std"],
            )
            (eta * actor_q_loss / accumulation).backward()
        torch.cuda.synchronize()
        q_latency += time.perf_counter() - started
        require(action_chunk.grad is not None, "BENCHMARK_ACTOR_Q_ACTION_GRADIENT_MISSING")
        tcp_q_gradient_square += float(action_chunk.grad[:, :3, :6].float().square().sum().cpu())
        gripper_q_gradient_max = max(gripper_q_gradient_max, float(action_chunk.grad[:, :3, 6].float().abs().max().cpu()))
        q_action = build_actor_q_action(
            action_chunk.detach(), delta_action_mean7=actor_batch["delta_mean"], delta_action_std7=actor_batch["delta_std"]
        )
        ones = torch.ones(microbatch, 3, dtype=torch.bool, device=device)
        with torch.no_grad():
            q1_action_values.append(q1(*micro_observation.as_tuple(), q_action, ones))
            q2_action_values.append(q2(*micro_observation.as_tuple(), q_action, ones))
        fm_window_sum += float(fm_contribution.detach().cpu())
        actor_q_sum += float(actor_q_loss.detach().cpu())
        balance_sum += float(terms.balance.detach().cpu())
        z_sum += float(terms.z.detach().cpu())
        records.append({
            "microbatch_index": micro_index,
            "size": microbatch,
            "identities": actor_batch["identities"][start:stop],
            "flow_valid_feature_count": int(feature_mask.sum().cpu()),
            "flow_matching_window_contribution": float(fm_contribution.detach().cpu()),
            "actor_q_loss": float(actor_q_loss.detach().cpu()),
            "balance": float(terms.balance.detach().cpu()),
            "z": float(terms.z.detach().cpu()),
            "active_router_experts": sorted(set(router_state.route_ids[router_state.valid_mask].detach().cpu().tolist())),
        })
        del losses, feature_mask, terms, router_state, action_chunk, actor_q_loss, q_action

    require(gradients_finite(actor_parameters), "BENCHMARK_ACTOR_GRADIENT_NONFINITE")
    module_norms = g5.actor_module_gradient_norms(policy)
    required = ("vision_vlm", "action_io", "action_expert", "force_mlp", "fusion", "moe_experts", "force_action_adapter", "router")
    require(all(module_norms[name] > 0 for name in required), f"BENCHMARK_ACTOR_MODULE_GRADIENT_MISSING:{module_norms}")
    preclip = global_gradient_norm(actor_parameters)
    torch.nn.utils.clip_grad_norm_(actor_parameters, config["optimizers"]["actor"]["grad_clip_norm"])
    postclip = global_gradient_norm(actor_parameters)
    require(gradients_finite(actor_parameters), "BENCHMARK_ACTOR_POSTCLIP_GRADIENT_NONFINITE")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    policy.eval()
    actor_after = module_state_sha256(policy)
    critic_after = {
        "q1": module_state_sha256(q1), "q2": module_state_sha256(q2),
        "q1_target": module_state_sha256(q1_target), "q2_target": module_state_sha256(q2_target),
    }
    require(actor_after != actor_before, "BENCHMARK_ACTOR_NOT_UPDATED")
    require(critic_before == critic_after, "BENCHMARK_CRITIC_CHANGED_DURING_ACTOR")
    require(all(parameter.grad is None for parameter in actor_parameters), "BENCHMARK_ACTOR_GRADIENT_NOT_CLEARED")
    weighted_total = fm_window_sum + 0.01 * balance_sum / accumulation + 0.001 * z_sum / accumulation + eta * actor_q_sum / accumulation
    return {
        "microbatches": records,
        "loss": {
            "L_FM_window": fm_window_sum,
            "L_actor_Q_window": actor_q_sum / accumulation,
            "L_balance_equal_microbatch_mean": balance_sum / accumulation,
            "L_z_equal_microbatch_mean": z_sum / accumulation,
            "weighted_actor_total": weighted_total,
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
            "actor_q_valid_transition_count": effective,
            "balance_z_microbatch_local_equal_average": True,
            "exact_global_router_objective_claimed": False,
            "optimizer_steps": 1,
        },
        "state": {"actor_before": actor_before, "actor_after": actor_after, "critics_exact_unchanged": critic_before == critic_after},
    }


def partial_mask_audit(data, policy, q1, q2, device) -> dict:
    from forcesmolvla.rft.losses import compute_behavior_q

    index = next(index for index in data.td_population if not bool(np.asarray(data.rows[index]["executed_action_mask"], dtype=bool).all()))
    batch = data.build_batch([index], policy, device, canonical_task_feature=q1.canonical_task_feature)
    action = batch["behavior_action"]
    mask = batch["behavior_mask"]
    modified = action.clone()
    modified[~mask] += 1000.0
    with torch.no_grad():
        q1_a = compute_behavior_q(q1, batch["current_observation"], action, mask)
        q1_b = compute_behavior_q(q1, batch["current_observation"], modified, mask)
        q2_a = compute_behavior_q(q2, batch["current_observation"], action, mask)
        q2_b = compute_behavior_q(q2, batch["current_observation"], modified, mask)
    return {
        "identity": batch["identities"][0],
        "action_shape": list(action.shape),
        "mask_shape": list(mask.shape),
        "invalid_slot_perturbation_exact_invariant": bool(torch.equal(q1_a, q1_b) and torch.equal(q2_a, q2_b)),
    }


def run_candidate(candidate: dict) -> dict:
    import preflight_s2_g5_single_cycle_gpu as g5
    import run_s2_g7a_worker as g7a_worker
    import run_s2_g7b_worker as g7b_worker
    from forcesmolvla.rft.training_cycle import module_state_sha256

    g5.install_open_audit()
    device = g7a_worker.configure_runtime()
    _g7b_config, training = g7b_worker.load_config()
    training = copy.deepcopy(training)
    micro = int(candidate["actor_microbatch"])
    accumulation = int(candidate["actor_accumulation"])
    effective = int(candidate["effective_actor_batch"])
    critic_batch = int(candidate["critic_batch"])
    require(micro * accumulation == effective, "BENCHMARK_CANDIDATE_ACTOR_BATCH_MISMATCH")
    require(int(training["loss"]["cql_candidates_per_source_M"]) == 2, "BENCHMARK_CALQL_M_DRIFT")
    training["batching"].update({
        "actor_microbatch_size": micro,
        "actor_gradient_accumulation": accumulation,
        "actor_effective_batch_size": effective,
        "critic_batch_size": critic_batch,
        "calql_batch_size": critic_batch,
    })
    training["loss"]["eta_actor_q"] = 10.0
    context, parent_sampler_states, parent_rng, actor_optimizer, actor_scheduler, _ownership, r2 = g7b_worker.load_models_and_state(device)
    data = context["data"]
    generators = g7b_worker.build_generators(training)
    samplers = g7b_worker.build_samplers(data, generators, parent_sampler_states)
    g7b_worker.restore_parent_rng(parent_rng, generators)
    policy, q1, q2 = context["actor"], context["q1"], context["q2"]
    q1_target, q2_target = context["q1_target"], context["q2_target"]
    critic_optimizer, critic_scheduler = context["optimizer"], context["scheduler"]
    initial = {name: module_state_sha256(module) for name, module in {
        "actor": policy, "q1": q1, "q2": q2, "q1_target": q1_target, "q2_target": q2_target,
    }.items()}
    backbone_initial = {
        f"{name}.{camera}": module_state_sha256(getattr(module, camera))
        for name, module in (("q1", q1), ("q2", q2))
        for camera in ("camera1_backbone", "camera2_backbone")
    }
    mask_audit = partial_mask_audit(data, policy, q1, q2, device)
    require(mask_audit["invalid_slot_perturbation_exact_invariant"], "BENCHMARK_MASK_LEAK")
    flow_counter = TimedFlowCounter(g5.FlowCounter(inference_batch_size=4))

    def one_cycle(cycle: int) -> tuple[dict, float]:
        torch.cuda.synchronize()
        cycle_started = time.perf_counter()
        critic_reports, data_seconds = [], 0.0
        with g7b_worker.critic_internal_only():
            for local in range(2):
                started = time.perf_counter()
                td_indices = samplers["td"].draw(critic_batch)
                calql_indices = samplers["calql"].draw(critic_batch)
                td_batch = data.build_batch(td_indices, policy, device, canonical_task_feature=q1.canonical_task_feature)
                calql_batch = data.build_batch(calql_indices, policy, device, canonical_task_feature=q1.canonical_task_feature)
                torch.cuda.synchronize()
                data_seconds += time.perf_counter() - started
                report = g5.critic_update(
                    step=256 + (cycle - 1) * 2 + local + 1,
                    policy=policy, q1=q1, q2=q2, q1_target=q1_target, q2_target=q2_target,
                    optimizer=critic_optimizer, scheduler=critic_scheduler,
                    td_batch=td_batch, calql_batch=calql_batch, train_data=data,
                    proposal_sampler=samplers["empirical_random_proposal"], generators=generators,
                    flow_counter=flow_counter, config=training,
                )
                critic_reports.append(g7a_worker.compact_critic_report(report))
                del td_batch, calql_batch, report
            started = time.perf_counter()
            actor_indices = samplers["actor"].draw(effective)
            actor_batch = data.build_batch(
                actor_indices, policy, device, canonical_task_feature=q1.canonical_task_feature,
                include_flow_actions=True,
            )
            torch.cuda.synchronize()
            data_seconds += time.perf_counter() - started
            actor_report = actor_update_batched(
                policy=policy, q1=q1, q2=q2, q1_target=q1_target, q2_target=q2_target,
                optimizer=actor_optimizer, scheduler=actor_scheduler, actor_batch=actor_batch,
                generators=generators, flow_counter=flow_counter, config=training,
                candidate_id=candidate["candidate_id"],
            )
            del actor_batch
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - cycle_started
        require(all(math.isfinite(float(value)) for report in critic_reports for value in report["loss"].values()), "BENCHMARK_NONFINITE_CRITIC_LOSS")
        require(all(math.isfinite(float(value)) for value in actor_report["loss"].values()), "BENCHMARK_NONFINITE_ACTOR_LOSS")
        require(actor_report["gradient"]["tcp6_actor_q_gradient_norm"] > 0.0, "BENCHMARK_TCP_Q_GRADIENT_ZERO")
        require(actor_report["gradient"]["gripper_actor_q_gradient_max_abs"] == 0.0, "BENCHMARK_GRIPPER_Q_GRADIENT_NONZERO")
        require(actor_report["gradient"]["gripper_flow_matching_gradient_norm"] > 0.0, "BENCHMARK_GRIPPER_FM_GRADIENT_ZERO")
        return {
            "cycle": cycle,
            "critic_updates": critic_reports,
            "actor_update": actor_report,
            "data_loading_and_host_to_device_seconds": data_seconds,
            "cycle_seconds": elapsed,
        }, elapsed

    warmup, _ = one_cycle(1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    timed_before = dict(flow_counter.seconds)
    measured = []
    with GpuUtilizationSampler(float(candidate["gpu_utilization_poll_seconds"])) as utilization:
        for cycle in range(2, 5):
            report, _ = one_cycle(cycle)
            measured.append(report)
            print(f"BATCH_CANDIDATE {candidate['candidate_id']} measured={cycle - 1}/3", flush=True)
    flow_seconds = {
        name: float(seconds - timed_before.get(name, 0.0))
        for name, seconds in flow_counter.seconds.items()
    }
    final = {name: module_state_sha256(module) for name, module in {
        "actor": policy, "q1": q1, "q2": q2, "q1_target": q1_target, "q2_target": q2_target,
    }.items()}
    backbone_final = {
        f"{name}.{camera}": module_state_sha256(getattr(module, camera))
        for name, module in (("q1", q1), ("q2", q2))
        for camera in ("camera1_backbone", "camera2_backbone")
    }
    require(all(initial[name] != final[name] for name in initial), "BENCHMARK_EXPECTED_PARAMETER_CHANGE_MISSING")
    require(backbone_initial == backbone_final, "BENCHMARK_FROZEN_BACKBONE_CHANGED")
    require(all(bool(torch.isfinite(parameter).all()) for module in (policy, q1, q2, q1_target, q2_target) for parameter in module.parameters()), "BENCHMARK_NONFINITE_PARAMETER")
    cycle_seconds = [item["cycle_seconds"] for item in measured]
    total_measured = sum(cycle_seconds)
    actor_samples = 3 * effective
    critic_td_samples = 3 * 2 * critic_batch
    critic_calql_samples = 3 * 2 * critic_batch
    measured_actor = [item["actor_update"] for item in measured]
    measured_critics = [report for item in measured for report in item["critic_updates"]]
    raw_count = r2.DIAGNOSTIC["raw_gripper_values"]
    return {
        "schema_version": "forcesmolvla_g7_batch_candidate.v1",
        "candidate_id": candidate["candidate_id"],
        "stage": candidate["stage"],
        "status": "pass",
        "pid": os.getpid(),
        "environment": g7a_worker.environment_audit(),
        "resolved_candidate": candidate,
        "parent": {
            "path": "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint",
            "critic_optimizer_step_loaded": 256,
            "g7b_checkpoint_loaded": False,
        },
        "warmup_joint_cycles": 1,
        "measured_joint_cycles": 3,
        "warmup_metrics_excluded": True,
        "warmup_finite": True,
        "measured_cycles": measured,
        "throughput": {
            "actor_samples": actor_samples,
            "actor_samples_per_second": actor_samples / total_measured,
            "critic_td_sample_memberships": critic_td_samples,
            "critic_calql_sample_memberships": critic_calql_samples,
            "critic_sample_memberships": critic_td_samples + critic_calql_samples,
            "critic_sample_memberships_per_second": (critic_td_samples + critic_calql_samples) / total_measured,
            "joint_training_sample_memberships": actor_samples + critic_td_samples + critic_calql_samples,
            "joint_training_sample_memberships_per_second": (actor_samples + critic_td_samples + critic_calql_samples) / total_measured,
        },
        "timing_seconds": {
            "joint_cycle": describe(cycle_seconds),
            "flow_matching_forward_backward": describe([item["latency_seconds"]["flow_matching_forward_backward"] for item in measured_actor]),
            "differentiable_n10_flow_forward_backward": describe([item["latency_seconds"]["differentiable_n10_flow_and_actor_q_backward"] for item in measured_actor]),
            "td_next_action_sampling_total": flow_seconds.get("td_next", 0.0),
            "calql_candidate_policy_sampling_total": flow_seconds.get("cql_current", 0.0) + flow_seconds.get("cql_next", 0.0),
            "critic_forward_backward": describe([item["latency_seconds"]["critic_forward_backward_step_polyak_scheduler"] for item in measured_critics]),
            "candidate_sampling": describe([item["latency_seconds"]["candidate_sampling"] for item in measured_critics]),
            "data_loading_and_host_to_device": describe([item["data_loading_and_host_to_device_seconds"] for item in measured]),
        },
        "gpu_utilization_percent": describe(utilization.values or [0.0]),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "all_finite": True,
        "update_counts": {
            "warmup_plus_measured_joint_cycles": 4,
            "critic_optimizer_updates": 8,
            "actor_optimizer_updates": 4,
            "polyak_updates_per_target": 8,
            "critic_scheduler_steps": 8,
            "actor_scheduler_steps": 4,
        },
        "action_contract_v2": {
            "passed": True,
            "tcp6_q_gradient_nonzero_all_measured": all(item["gradient"]["tcp6_actor_q_gradient_norm"] > 0 for item in measured_actor),
            "gripper_q_gradient_exact_zero_all_measured": all(item["gradient"]["gripper_actor_q_gradient_max_abs"] == 0 for item in measured_actor),
            "gripper_fm_gradient_nonzero_all_measured": all(item["gradient"]["gripper_flow_matching_gradient_norm"] > 0 for item in measured_actor),
            "invalid_action_slot_mask_audit": mask_audit,
            "raw_gripper_out_of_public_tolerance_rate": r2.DIAGNOSTIC["raw_gripper_out_of_public_tolerance"] / raw_count if raw_count else 0.0,
            "clipping_added": False, "resampling_added": False, "binary_ste_added": False,
        },
        "router_auxiliary": {
            "definition": "microbatch_local_equal_average",
            "global_router_objective_equivalence_claimed": False,
            "balance": describe([item["loss"]["L_balance_equal_microbatch_mean"] for item in measured_actor]),
            "z": describe([item["loss"]["L_z_equal_microbatch_mean"] for item in measured_actor]),
        },
        "parameter_change_matrix": {name: {"before": initial[name], "after": final[name], "changed": initial[name] != final[name]} for name in initial},
        "frozen_backbones_exact": backbone_initial == backbone_final,
        "flow_counts_total_including_warmup": flow_counter.report(),
        "public_inference_immutability": {
            "behavior_changed": False,
            "safety_threshold_changed": False,
            "action_delta_sha256": file_sha(ROOT / "src/forcesmolvla/action_delta.py"),
            "modeling_policy_sha256": file_sha(ROOT / "src/forcesmolvla/modeling_forcesmolvla.py"),
            "rulespec_sha256": file_sha(ROOT / "configs/live_action_safety.task2.development.yaml"),
            "public_gripper_candidate_tolerance_m": [-0.01, 0.095],
        },
        "data_access": {
            "train_transitions_available": len(data.rows),
            "validation_transition_reads": 0, "test_transition_reads": 0,
            "manual_g1_opens": 0, "manual_label_opens": 0,
            "reward_classifier_inference": 0, "reward_classifier_updates": 0,
        },
        "checkpoint_created": False,
        "candidate_state_discarded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    require(not args.result.exists(), "BENCHMARK_CANDIDATE_RESULT_APPEND_ONLY")
    candidate = json.loads(args.candidate_config.read_text())
    try:
        result = run_candidate(candidate)
    except BaseException as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        if not is_oom:
            raise
        result = {
            "schema_version": "forcesmolvla_g7_batch_candidate.v1",
            "candidate_id": candidate["candidate_id"], "stage": candidate["stage"],
            "status": "oom", "pid": os.getpid(), "resolved_candidate": candidate,
            "error_type": type(error).__name__, "error": str(error),
            "runtime_batch_fallback_used": False, "checkpoint_created": False,
            "candidate_state_discarded": True,
            "data_access": {"validation_transition_reads": 0, "test_transition_reads": 0, "manual_g1_opens": 0, "manual_label_opens": 0, "reward_classifier_inference": 0, "reward_classifier_updates": 0},
        }
    atomic_json(args.result, result)


if __name__ == "__main__":
    main()

