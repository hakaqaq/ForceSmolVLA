#!/usr/bin/env python3
"""Append-only fresh-process Stage-2 throughput-v2 benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import ExitStack
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml


ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(ROOT / "tools"))
CONFIG = ROOT / "configs/stage2_throughput_v2.development.yaml"
OUTPUT = ROOT / "artifacts/development/stage2/throughput_v2"
NORMALIZER_MANIFEST = ROOT / "datasets/task2_lerobotv3/normalizer_manifest.json"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0 and np.isfinite(array).all(), "THROUGHPUT_V2_STAT_INVALID")
    return {
        "count": int(array.size), "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()), "maximum": float(array.max()),
        "range": float(array.max() - array.min()),
    }


def cycle_component_times(record: dict[str, Any]) -> dict[str, float]:
    data = float(record["actor_data_loading_seconds"])
    policy_flow = 0.0
    critic_compute = 0.0
    polyak = 0.0
    for critic in record["critic"]:
        timing = critic["timing"]
        data += float(timing.get("data_loading", 0.0))
        policy_flow += sum(
            float(value) for name, value in timing.items()
            if "sampling" in name and "empirical" not in name
        )
        critic_compute += sum(
            float(value) for name, value in timing.items()
            if name in {
                "calql_empirical_proposal_and_overhead",
                "q_forward_backward_excluding_optimizer_polyak",
                "q_forward_backward_optimizer_polyak",
                "optimizer", "scheduler",
            }
        )
        polyak += float(timing.get("polyak", 0.0))
    actor_timing = record["actor"]["timing"]
    actor_compute = sum(float(value) for value in actor_timing.values())
    training = data + policy_flow + critic_compute + polyak + actor_compute
    return {
        "data_pipeline": data,
        "policy_flow_sampling": policy_flow,
        "critic_compute_optimizer": critic_compute,
        "polyak": polyak,
        "actor_compute_optimizer": actor_compute,
        "steady_state_training_body": training,
        "development_diagnostics_and_python_overhead": max(
            0.0, float(record["cycle_seconds"]) - training
        ),
        "checkpoint_and_report": 0.0,
    }


class ReplaySampler:
    def __init__(self, name: str, draws: list[list[int]]) -> None:
        self.name = name
        self.queue = deque(copy.deepcopy(draws))
        self.draws = 0

    def draw(self, count: int) -> list[int]:
        require(bool(self.queue), f"THROUGHPUT_V2_REPLAY_EXHAUSTED:{self.name}")
        value = self.queue.popleft()
        require(len(value) == count, f"THROUGHPUT_V2_REPLAY_COUNT:{self.name}")
        self.draws += 1
        return value

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "remaining": list(self.queue), "draws": self.draws}


def draw_plan(
    samplers: dict[str, Any], total_cycles: int
) -> tuple[dict[str, list[list[int]]], dict[str, ReplaySampler]]:
    plan = {name: [] for name in ("td", "calql", "actor", "empirical_random_proposal")}
    for _cycle in range(total_cycles):
        for _critic in range(2):
            plan["td"].append(samplers["td"].draw(128))
            plan["calql"].append(samplers["calql"].draw(128))
            plan["empirical_random_proposal"].append(
                samplers["empirical_random_proposal"].draw(256)
            )
        plan["actor"].append(samplers["actor"].draw(24))
    return plan, {name: ReplaySampler(name, values) for name, values in plan.items()}


class PersistentCpuPipeline:
    """Immutable Parquet-row and decoded-image cache, prefetched by CPU workers."""

    def __init__(self, data) -> None:
        import preflight_s2_g5_single_cycle_gpu as g5

        self.data = data
        self.g5 = g5
        self.original_raw = data._raw_rows
        self.original_decode = g5.decode_rgb
        self.tables: dict[str, list[dict]] = {}
        self.images: dict[bytes, np.ndarray] = {}
        self.stats = {
            "parquet_read_calls": 0,
            "parquet_rows_materialized": 0,
            "image_decode_calls": 0,
            "image_cache_hits": 0,
            "prefetch_workers": 8,
            "pinned_memory": False,
            "non_blocking_h2d": False,
        }

    def _load_table(self, relative: str) -> list[dict]:
        import pyarrow.parquet as pq
        from forcesmolvla.rft.offline_transitions import PROVENANCE_KEYS
        from preflight_s2_g5_single_cycle_gpu import DATASET

        if relative not in self.tables:
            columns = [
                "observation.images.camera1", "observation.images.camera2",
                "observation.state", "observation.wrench", "frame_index",
                "episode_index", "index", "action", *PROVENANCE_KEYS,
            ]
            rows = pq.read_table(DATASET / relative, columns=columns).to_pylist()
            self.tables[relative] = rows
            self.stats["parquet_read_calls"] += 1
            self.stats["parquet_rows_materialized"] += len(rows)
        return self.tables[relative]

    def raw_rows(self, requested: dict[str, set[int]], *, include_actions: bool):
        del include_actions
        return {
            (relative, index): self._load_table(relative)[index]
            for relative, indices in requested.items()
            for index in sorted(indices)
        }

    def decode(self, payload: bytes) -> np.ndarray:
        cached = self.images.get(payload)
        if cached is not None:
            self.stats["image_cache_hits"] += 1
            return cached
        decoded = self.original_decode(payload)
        self.images[payload] = decoded
        self.stats["image_decode_calls"] += 1
        return decoded

    def _observation_payloads(self, indices: list[int]) -> list[bytes]:
        payloads = []
        for index in indices:
            transition = self.data.rows[index]
            for key in ("observation_row_reference", "next_observation_row_reference"):
                reference = transition[key]
                row = self._load_table(reference["data_relative_path"])[reference["row_index"]]
                payloads.extend((
                    row["observation.images.camera1"]["bytes"],
                    row["observation.images.camera2"]["bytes"],
                ))
        return payloads

    def install_and_prefetch(self, plan: dict[str, list[list[int]]]) -> dict[str, Any]:
        started = time.perf_counter()
        indices = [
            index
            for name in ("td", "calql", "actor")
            for draw in plan[name]
            for index in draw
        ]
        payloads = self._observation_payloads(indices)
        unique = list(dict.fromkeys(payloads))
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="s2-data-prefetch") as pool:
            list(pool.map(self.decode, unique))
        self.data._raw_rows = types.MethodType(
            lambda _data, requested, *, include_actions: self.raw_rows(
                requested, include_actions=include_actions
            ),
            self.data,
        )
        self.g5.decode_rgb = self.decode
        return {
            "initialization_seconds_excluded_from_steady_state": time.perf_counter() - started,
            "planned_transition_occurrences": len(indices),
            "unique_encoded_image_payloads": len(unique),
            "decoded_image_cache_bytes": sum(value.nbytes for value in self.images.values()),
            **self.stats,
        }

    def close(self) -> None:
        self.data._raw_rows = self.original_raw
        self.g5.decode_rgb = self.original_decode


class CapturingTimedFlowCounter:
    def __init__(
        self,
        inference_batch_size: int,
        *,
        capture: bool,
        counter_type,
    ) -> None:
        # ``benchmark.TimedFlowCounter`` is patched to the factory while a
        # cycle runs.  Keep and use the pre-patch type to avoid recursively
        # invoking that factory.
        self.inner = counter_type(inference_batch_size)
        self.capture = capture
        self.captured: dict[str, torch.Tensor] = {}

    @property
    def seconds(self):
        return self.inner.seconds

    def sample(self, *args, call_id: str, purpose: str, **kwargs):
        result = self.inner.sample(*args, call_id=call_id, purpose=purpose, **kwargs)
        if self.capture:
            noise = args[2] if len(args) > 2 else kwargs["noise7"]
            self.captured[f"action|{call_id}|{purpose}"] = result.detach().cpu()
            self.captured[f"noise|{call_id}|{purpose}"] = noise.detach().cpu()
        return result

    def report(self):
        return self.inner.report()


def grouped_critic_update(
    *, context, training, generators, samplers, update_id: int, capture: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Same G4 loss/update with TD and Cal-QL Flow requests co-scheduled."""

    import preflight_s2_g5_single_cycle_gpu as g5
    from forcesmolvla.rft.critic_action_adapter_v2 import critic_action_for_q_guidance_v2
    from forcesmolvla.rft.losses import (
        compute_behavior_q,
        compute_calql_penalty,
        compute_td_target,
        evaluate_calql_candidates,
    )
    from forcesmolvla.rft.throughput_v2 import (
        fast_polyak_update,
        index_actor_batch,
        sample_grouped_flow_requests,
    )
    from forcesmolvla.rft.training_cycle import (
        calql_unclipped_details,
        global_gradient_norm,
        gradients_finite,
    )

    data, policy = context["data"], context["actor"]
    q1, q2 = context["q1"], context["q2"]
    q1_target, q2_target = context["q1_target"], context["q2_target"]
    optimizer, scheduler = context["optimizer"], context["scheduler"]
    device = q1.canonical_task_feature.device
    td_indices = samplers["td"].draw(128)
    calql_indices = samplers["calql"].draw(128)
    load_started = time.perf_counter()
    td_batch = data.build_batch(
        td_indices, policy, device, canonical_task_feature=q1.canonical_task_feature
    )
    calql_batch = data.build_batch(
        calql_indices, policy, device, canonical_task_feature=q1.canonical_task_feature
    )
    data_seconds = time.perf_counter() - load_started
    optimizer.zero_grad(set_to_none=True)
    candidates = 2
    td_noise = torch.randn(
        128, 50, 7, generator=generators["td_next_action_flow_noise"],
        dtype=torch.float32, device=device,
    )
    current_noise = torch.randn(
        128, candidates, 50, 7,
        generator=generators["calql_current_policy_flow_noise"], device=device,
    )
    next_noise = torch.randn(
        128, candidates, 50, 7,
        generator=generators["calql_next_policy_flow_noise"], device=device,
    )
    nonterminal_positions = torch.nonzero(~td_batch["terminated"], as_tuple=False).flatten().tolist()
    td_actor = index_actor_batch(td_batch["next_actor_batch"], nonterminal_positions)
    current_actor = g5.repeat_actor_batch(
        calql_batch["current_actor_batch"], candidates, tag="cql_current"
    )
    next_actor = g5.repeat_actor_batch(
        calql_batch["next_actor_batch"], candidates, tag="cql_next"
    )
    policy.train(False)
    torch.cuda.synchronize()
    flow_started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        chunks, flow_report, captured = sample_grouped_flow_requests(
            policy,
            [
                ("td_next", td_actor, td_noise[nonterminal_positions]),
                ("cql_current", current_actor, current_noise.reshape(-1, 50, 7)),
                ("cql_next", next_actor, next_noise.reshape(-1, 50, 7)),
            ],
            unique_observation_subbatch=16,
            call_id=f"throughput-v2-grouped-{update_id}",
            capture=capture,
        )
    torch.cuda.synchronize()
    flow_seconds = time.perf_counter() - flow_started
    mean, std = td_batch["delta_mean"], td_batch["delta_std"]
    td_action = critic_action_for_q_guidance_v2(
        chunks["td_next"], delta_action_mean7=mean, delta_action_std7=std
    ).detach().float()
    policy_current = critic_action_for_q_guidance_v2(
        chunks["cql_current"], delta_action_mean7=mean, delta_action_std7=std
    ).detach().float().reshape(128, candidates, 3, 7)
    policy_next = critic_action_for_q_guidance_v2(
        chunks["cql_next"], delta_action_mean7=mean, delta_action_std7=std
    ).detach().float().reshape(128, candidates, 3, 7)
    policy_mask = torch.ones(len(nonterminal_positions), 3, dtype=torch.bool, device=device)
    next_observation = td_batch["next_observation"].index(~td_batch["terminated"])
    with torch.no_grad():
        target1 = q1_target(*next_observation.as_tuple(), td_action, policy_mask)
        target2 = q2_target(*next_observation.as_tuple(), td_action, policy_mask)
    td_target = compute_td_target(
        td_batch["reward"], td_batch["discount"], td_batch["terminated"],
        td_batch["bootstrap_mask"], target1, target2,
    )
    proposal_indices = samplers["empirical_random_proposal"].draw(256)
    random_candidates = data.proposal_actions[proposal_indices].to(device).reshape(128, 2, 3, 7)
    endpoint = torch.stack((
        (torch.tensor(0.0, device=device) - mean[6]) / std[6],
        (torch.tensor(0.085, device=device) - mean[6]) / std[6],
    )).float()
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
    finite_limit = torch.finfo(torch.float32).max
    valid = torch.ones(128, dtype=torch.bool, device=device)
    calql1 = compute_calql_penalty(q1_data, q1_candidates, calql_batch["mc_return"], valid, temperature=1.0, clip_min=-finite_limit, clip_max=finite_limit)
    calql2 = compute_calql_penalty(q2_data, q2_candidates, calql_batch["mc_return"], valid, temperature=1.0, clip_min=-finite_limit, clip_max=finite_limit)
    detail1 = calql_unclipped_details(q1_data, q1_candidates, calql_batch["mc_return"], temperature=1.0)
    detail2 = calql_unclipped_details(q2_data, q2_candidates, calql_batch["mc_return"], temperature=1.0)
    loss = ((td1 + 0.1 * calql1) + (td2 + 0.1 * calql2)) / 2.0
    require(all(torch.isfinite(value).all() for value in (
        q1_td, q2_td, q1_data, q2_data, q1_candidates, q2_candidates, td_target, loss,
    )), "THROUGHPUT_V2_GROUPED_NONFINITE")
    loss.backward()
    trainable = [
        value for critic in (q1, q2) for value in critic.parameters() if value.requires_grad
    ]
    require(
        all(value.grad is None for value in policy.parameters())
        and all(value.grad is None for target in (q1_target, q2_target) for value in target.parameters())
        and gradients_finite(trainable),
        "THROUGHPUT_V2_GROUPED_GRADIENT_OWNERSHIP",
    )
    preclip = global_gradient_norm(trainable)
    torch.nn.utils.clip_grad_norm_(trainable, 10.0)
    postclip = global_gradient_norm(trainable)
    optimizer.step()
    fast_polyak_update(q1, q1_target, tau=0.005, target_name="q1_target")
    fast_polyak_update(q2, q2_target, tau=0.005, target_name="q2_target")
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    critic_seconds = time.perf_counter() - critic_started
    target_min = (td_target[~td_batch["terminated"]] - td_batch["reward"][~td_batch["terminated"]]) / td_batch["discount"][~td_batch["terminated"]]
    differences = torch.cat((detail1["difference"], detail2["difference"]))
    activations = torch.cat((detail1["mc_lower_bound_activation"].reshape(-1), detail2["mc_lower_bound_activation"].reshape(-1)))
    return {
        "loss": {
            "L_TD_Q1": float(td1.detach()), "L_TD_Q2": float(td2.detach()),
            "L_CalQL_Q1": float(calql1.detach()), "L_CalQL_Q2": float(calql2.detach()),
            "L_critic": float(loss.detach()),
        },
        "statistics": {
            "dataset_q": g5.stats(torch.cat((q1_data, q2_data))),
            "candidate_q": g5.stats(torch.cat((q1_candidates.flatten(), q2_candidates.flatten()))),
            "target_q_min": g5.stats(target_min), "td_target": g5.stats(td_target),
            "mc_return": g5.stats(calql_batch["mc_return"]),
            "calql_unclipped_difference": g5.stats(differences),
            "mc_lower_bound_activation_rate": float(activations.float().mean()),
        },
        "gradient": {
            "preclip_global_norm": float(preclip), "postclip_global_norm": float(postclip),
            "finite_before_and_after": True,
        },
        "terminal_rows": int(td_batch["terminated"].sum()),
        "timing": {
            "data_loading": data_seconds,
            "grouped_td_calql_policy_flow_sampling": flow_seconds,
            "q_forward_backward_optimizer_polyak": critic_seconds,
        },
        "flow_counts": flow_report,
        "row_identities": {
            "td": data.identity_records(td_indices),
            "calql": data.identity_records(calql_indices),
        },
    }, captured


def worker(candidate: dict[str, Any], result_path: Path) -> dict[str, Any]:
    import benchmark_stage2_batch_scaling_gpu as benchmark
    import preflight_s2_g5_single_cycle_gpu as g5
    import run_s2_g7b_worker as g7b
    import run_stage2b_long_run_half_pass_worker as stage2b
    from forcesmolvla.rft.frozen_vlm_trainability import frozen_state_digest
    from forcesmolvla.rft.throughput_v2 import (
        FrozenPrefixFlowCounter, fast_polyak_update, lightweight_state_token,
    )
    from forcesmolvla.rft.training_cycle import module_state_sha256
    from forcesmolvla.rft.training_cycle import generator_state_sha256
    from forcesmolvla.rft import training_cycle

    g5.install_open_audit()
    benchmark_config = yaml.safe_load(CONFIG.read_text())["benchmark"]
    warmup_cycles = int(benchmark_config["warmup_joint_cycles"])
    measured_cycles = int(benchmark_config["measured_joint_cycles"])
    total_cycles = warmup_cycles + measured_cycles
    require(warmup_cycles == 1 and measured_cycles >= 1, "THROUGHPUT_V2_CYCLE_CONTRACT")
    device = benchmark.configure_runtime()
    context = benchmark.load_context(device)
    model, training, generators, original_samplers, actor_optimizer, actor_scheduler, ownership, trainability = context
    context = model
    plan, samplers = draw_plan(original_samplers, total_cycles)
    pipeline = None
    pipeline_report = {
        "mode": "current_synchronous_no_dataloader",
        "initialization_seconds_excluded_from_steady_state": 0.0,
    }
    if candidate["persistent_cpu_prefetch_cache"]:
        pipeline = PersistentCpuPipeline(context["data"])
        pipeline_report = {
            "mode": "persistent_parquet_row_and_threaded_image_prefetch_cache",
            **pipeline.install_and_prefetch(plan),
        }

    frozen_before = frozen_state_digest(context["actor"])
    actor_before = module_state_sha256(context["actor"])
    critic_before = {
        name: module_state_sha256(context[name])
        for name in ("q1", "q2", "q1_target", "q2_target")
    }
    public_indices = [context["data"].actor_population[0]]
    public_batch = context["data"].build_batch(
        public_indices, context["actor"], device,
        canonical_task_feature=context["q1"].canonical_task_feature,
        include_flow_actions=True,
    )
    public_noise = torch.randn(
        1, 50, 7,
        generator=torch.Generator(device=device).manual_seed(7440),
        device=device,
    )
    public_before = g7b.public_diagnostic(
        context["actor"], context["data"], public_batch["current_actor_batch"],
        public_noise, 0,
    )
    telemetry = None
    records: list[dict[str, Any]] = []
    captured: dict[str, torch.Tensor] = {}
    prefix_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    original_encode = context["actor"].model.encode_prefix
    original_timed_flow_counter = benchmark.TimedFlowCounter

    def timed_encode(_model, *args, **kwargs):
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        value = original_encode(*args, **kwargs)
        finished.record()
        prefix_events.append((started, finished))
        return value

    context["actor"].model.encode_prefix = types.MethodType(timed_encode, context["actor"].model)
    public_path_guard = g7b.critic_internal_only()
    public_path_guard.__enter__()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for local in range(total_cycles):
            if local == 1:
                telemetry = benchmark.GpuTelemetry().__enter__()
            cycle_counters = []
            prefix_start = len(prefix_events)
            torch.cuda.synchronize()
            cycle_started = time.perf_counter()

            def counter_factory(requested: int | None = None, **kwargs):
                if requested is None:
                    requested = int(kwargs["inference_batch_size"])
                capture = local == 0
                if candidate["frozen_prefix_candidate_reuse"] and requested == 4:
                    counter = FrozenPrefixFlowCounter(
                        int(candidate["critic_flow_subbatch"]), capture=capture
                    )
                else:
                    counter = CapturingTimedFlowCounter(
                        requested,
                        capture=capture,
                        counter_type=original_timed_flow_counter,
                    )
                cycle_counters.append(counter)
                return counter

            critic_reports = []
            grouped_flow_reports = []
            actor_before_critics = (
                module_state_sha256(context["actor"])
                if candidate["inner_full_sha_and_polyak_audit"]
                else lightweight_state_token(context["actor"])
            )
            for substep in range(2):
                if candidate["grouped_td_calql_flow"]:
                    report, grouped_capture = grouped_critic_update(
                        context=context, training=training, generators=generators,
                        samplers=samplers, update_id=257 + local * 2 + substep,
                        capture=local == 0,
                    )
                    critic_reports.append(report)
                    grouped_flow_reports.append(report["flow_counts"])
                    for name, tensor in grouped_capture.items():
                        captured[f"grouped-critic{substep}/{name}"] = tensor
                else:
                    with ExitStack() as stack:
                        stack.enter_context(patch.object(benchmark, "TimedFlowCounter", side_effect=counter_factory))
                        if not candidate["inner_full_sha_and_polyak_audit"]:
                            stack.enter_context(patch.object(training_cycle, "module_state_sha256", side_effect=lightweight_state_token))
                            stack.enter_context(patch.object(training_cycle, "polyak_update_verified", side_effect=fast_polyak_update))
                        critic_reports.append(benchmark.critic_update(
                            context=context, training=training, generators=generators,
                            samplers=samplers, batch_size=128,
                            update_id=257 + local * 2 + substep,
                        ))
                if candidate["per_cycle_gc_empty_cache"]:
                    gc.collect()
                    torch.cuda.empty_cache()
            actor_after_critics = (
                module_state_sha256(context["actor"])
                if candidate["inner_full_sha_and_polyak_audit"]
                else lightweight_state_token(context["actor"])
            )
            require(actor_before_critics == actor_after_critics, "THROUGHPUT_V2_ACTOR_CHANGED_DURING_CRITICS")
            actor_draw = samplers["actor"].draw(24)
            load_started = time.perf_counter()
            actor_batch = context["data"].build_batch(
                actor_draw, context["actor"], device,
                canonical_task_feature=context["q1"].canonical_task_feature,
                include_flow_actions=True,
            )
            actor_load = time.perf_counter() - load_started
            with patch.object(benchmark, "TimedFlowCounter", side_effect=counter_factory):
                actor_report = stage2b.actor_update_eta3(
                    cycle=local, context=context, batch=actor_batch,
                    optimizer=actor_optimizer, scheduler=actor_scheduler,
                    generators=generators,
                )
            del actor_batch
            torch.cuda.synchronize()
            cycle_seconds = time.perf_counter() - cycle_started
            for counter_index, counter in enumerate(cycle_counters):
                for name, tensor in getattr(counter, "captured", {}).items():
                    captured[f"counter{counter_index}/{name}"] = tensor
            record = {
                "local_cycle": local, "warmup": local == 0,
                "cycle_seconds": cycle_seconds,
                "critic": critic_reports, "actor": actor_report,
                "actor_data_loading_seconds": actor_load,
                "prefix_prefill_seconds": sum(
                    started.elapsed_time(finished) / 1000.0
                    for started, finished in prefix_events[prefix_start:]
                ),
                "prefix_prefill_calls": len(prefix_events) - prefix_start,
                "flow_reports": grouped_flow_reports + [counter.report() for counter in cycle_counters],
            }
            records.append(record)
            print(
                f"THROUGHPUT_V2 {candidate['id']} cycle={local + 1}/{total_cycles}",
                flush=True,
            )
            if candidate["per_cycle_gc_empty_cache"]:
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        public_path_guard.__exit__(None, None, None)
        context["actor"].model.encode_prefix = original_encode
        if telemetry is not None:
            telemetry.__exit__(None, None, None)
        if pipeline is not None:
            pipeline.close()

    frozen_after = frozen_state_digest(context["actor"])
    require(telemetry is not None, "THROUGHPUT_V2_TELEMETRY_NOT_STARTED")
    public_after = g7b.public_diagnostic(
        context["actor"], context["data"], public_batch["current_actor_batch"],
        public_noise, total_cycles,
    )
    actor_after = module_state_sha256(context["actor"])
    critic_after = {
        name: module_state_sha256(context[name])
        for name in ("q1", "q2", "q1_target", "q2_target")
    }
    require(frozen_before == frozen_after, "THROUGHPUT_V2_FROZEN_HASH_CHANGED")
    require(
        not g5.FORBIDDEN_OPENS["manual_g1"]
        and not g5.FORBIDDEN_OPENS["manual_labels"],
        "THROUGHPUT_V2_FORBIDDEN_MANUAL_OPEN",
    )
    require(actor_before != actor_after, "THROUGHPUT_V2_ACTOR_NOT_UPDATED")
    require(all(critic_before[name] != critic_after[name] for name in critic_after), "THROUGHPUT_V2_CRITIC_NOT_UPDATED")
    measured = records[1:]
    elapsed = sum(item["cycle_seconds"] for item in measured)
    component_records = [cycle_component_times(item) for item in measured]
    require(all(
        math.isfinite(value)
        for item in measured
        for critic in item["critic"]
        for value in critic["loss"].values()
    ), "THROUGHPUT_V2_NONFINITE_CRITIC")
    require(all(
        item["actor"]["gradient"]["tcp6_q_norm"] > 0.0
        and item["actor"]["gradient"]["gripper_q_max_abs"] == 0.0
        and item["actor"]["gradient"]["gripper_fm_norm"] > 0.0
        for item in measured
    ), "THROUGHPUT_V2_ACTION_GRADIENT_CONTRACT")
    trace_path = result_path.with_suffix(".warmup_trace.pt")
    torch.save(captured, trace_path)
    return {
        "schema_version": "forcesmolvla_stage2_throughput_v2_candidate.v1",
        "status": "pass", "candidate": candidate,
        "warmup_joint_cycles": warmup_cycles,
        "measured_joint_cycles": measured_cycles,
        "measurement_scope": benchmark_config.get("measurement_scope"),
        "seconds_per_cycle": describe([item["cycle_seconds"] for item in measured]),
        "actor_transitions_per_second": (24.0 * measured_cycles) / elapsed,
        "critic_transitions_per_second": (
            128.0 * 2.0 * measured_cycles
        ) / elapsed,
        "joint_cycles_per_hour": measured_cycles / elapsed * 3600.0,
        "gpu_utilization_percent": describe(telemetry.utilization or [0.0]),
        "gpu_power_watts": describe(telemetry.power or [0.0]),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "pipeline": pipeline_report,
        "mean_cycle_decomposition_seconds": {
            name: float(np.mean([item[name] for item in component_records]))
            for name in component_records[0]
        },
        "frozen_parameter_hash_unchanged": True,
        "all_losses_and_gradients_finite": True,
        "action_contract_v2": True,
        "internal_path_call_audit": {
            "public_predict_calls": 0,
            "absolute_inverse_calls": 0,
            "public_safety_check_calls": 0,
            "rulespec_calls": 0,
        },
        "public_inference": {
            "before": public_before,
            "after": public_after,
            "interface_decoder_tolerance_and_rulespec_implementation_changed": False,
            "both_calls_succeeded": bool(public_before["success"] and public_after["success"]),
        },
        "parameter_updates": {
            "actor": total_cycles,
            "critic": 2 * total_cycles,
            "polyak_per_target": 2 * total_cycles,
        },
        "warmup_trace_path": trace_path.relative_to(ROOT).as_posix(),
        "warmup_trace_sha256": sha256_file(trace_path),
        "warmup_record": records[0],
        "named_generator_final_sha256": {
            name: generator_state_sha256(generator)
            for name, generator in sorted(generators.items())
        },
        "records": measured,
        "candidate_state_discarded": True,
        "training_checkpoint_created": False,
        "access_audit": {
            "validation_reads": 0, "test_reads": 0, "manual_g1_opens": 0,
            "manual_label_opens": 0, "reward_classifier_inference": 0,
        },
        "trainability": {
            "frozen_parameter_count": trainability.frozen_parameter_count,
            "trainable_actor_parameter_count": trainability.trainable_actor_parameter_count,
            "actor_optimizer": ownership,
        },
    }


def run_worker(candidate: dict[str, Any], result_path: Path) -> dict[str, Any]:
    candidate_path = result_path.with_suffix(".config.json")
    atomic_json(candidate_path, candidate)
    environment = os.environ.copy()
    environment.update({
        "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT / 'tools'}:{ROOT}",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    })
    completed = subprocess.run(
        [sys.executable, __file__, "--worker", "--candidate", str(candidate_path), "--result", str(result_path)],
        cwd=ROOT, env=environment, check=False,
    )
    if completed.returncode != 0 and not result_path.exists():
        raise RuntimeError(f"THROUGHPUT_V2_WORKER_FAILED:{candidate['id']}:{completed.returncode}")
    return json.loads(result_path.read_text())


def _trace_groups(path: Path) -> dict[tuple[str, str], torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    grouped: dict[tuple[str, str], list[tuple[str, torch.Tensor]]] = {}
    for key, value in raw.items():
        marker = "action|" if "action|" in key else "noise|" if "noise|" in key else None
        if marker is None:
            continue
        kind = marker[:-1]
        purpose = key.rsplit("|", 1)[-1]
        grouped.setdefault((kind, purpose), []).append((key, value.float()))
    return {
        key: torch.cat([value for _name, value in sorted(values)], dim=0)
        for key, values in grouped.items()
    }


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for name, item in value.items():
            result.update(_numeric_leaves(item, f"{prefix}/{name}"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def _critic_action_trace_view(value: torch.Tensor) -> torch.Tensor:
    """Project captured Hx7 Flow output to the frozen Critic Kx7 contract."""

    from forcesmolvla.rft.critic_action_adapter_v2 import (
        critic_action_for_q_guidance_v2,
    )

    manifest = json.loads(NORMALIZER_MANIFEST.read_text())
    statistics = manifest["action_target_population"]["statistics"]["global"]
    mean = torch.tensor(statistics["mean"], dtype=torch.float32)
    std = torch.tensor(statistics["std"], dtype=torch.float32)
    return critic_action_for_q_guidance_v2(
        value.float(), delta_action_mean7=mean, delta_action_std7=std
    )


def compare_equivalence(baseline: dict[str, Any], candidate: dict[str, Any], config: dict) -> dict[str, Any]:
    baseline_trace = _trace_groups(ROOT / baseline["warmup_trace_path"])
    candidate_trace = _trace_groups(ROOT / candidate["warmup_trace_path"])
    shared = sorted(set(baseline_trace) & set(candidate_trace))
    action_atol = float(config["equivalence"]["bf16_action_atol"])
    action_rtol = float(config["equivalence"]["bf16_action_rtol"])
    traces = {}
    for key in shared:
        left, right = baseline_trace[key], candidate_trace[key]
        shape_equal = left.shape == right.shape
        raw_maximum = (
            float((left - right).abs().max())
            if shape_equal and left.numel()
            else 0.0
        )
        if key[0] == "action" and shape_equal:
            left_view = _critic_action_trace_view(left)
            right_view = _critic_action_trace_view(right)
            difference = (left_view - right_view).abs()
            maximum = float(difference.max()) if difference.numel() else 0.0
            within_tolerance = bool(
                torch.allclose(
                    left_view[..., :6], right_view[..., :6],
                    atol=action_atol, rtol=action_rtol,
                )
                and torch.equal(left_view[..., 6], right_view[..., 6])
            )
            bitwise_equal = torch.equal(left_view, right_view)
            gripper_exact = torch.equal(left_view[..., 6], right_view[..., 6])
        else:
            maximum = raw_maximum
            within_tolerance = bool(shape_equal and torch.equal(left, right))
            bitwise_equal = within_tolerance
            gripper_exact = None
        traces[f"{key[0]}/{key[1]}"] = {
            "shape_equal": shape_equal,
            "bitwise_equal": bool(bitwise_equal),
            "comparison_domain": (
                "ActionContract_v2_critic_Kx7"
                if key[0] == "action"
                else "raw_noise_Hx7"
            ),
            "raw_flow_maximum_abs_error_diagnostic_only": raw_maximum,
            "maximum_abs_error": maximum,
            "gripper_endpoint_exact": gripper_exact,
            "within_tolerance": within_tolerance,
        }
    baseline_rows = [
        critic["row_identities"]
        for critic in baseline["warmup_record"]["critic"]
    ]
    candidate_rows = [
        critic["row_identities"]
        for critic in candidate["warmup_record"]["critic"]
    ]
    baseline_core = {
        "critic": [
            {"loss": item["loss"], "statistics": item["statistics"], "gradient": item["gradient"]}
            for item in baseline["warmup_record"]["critic"]
        ],
        "actor": {
            "loss": baseline["warmup_record"]["actor"]["loss"],
            "q": baseline["warmup_record"]["actor"]["q"],
            "gradient": baseline["warmup_record"]["actor"]["gradient"],
        },
    }
    candidate_core = {
        "critic": [
            {"loss": item["loss"], "statistics": item["statistics"], "gradient": item["gradient"]}
            for item in candidate["warmup_record"]["critic"]
        ],
        "actor": {
            "loss": candidate["warmup_record"]["actor"]["loss"],
            "q": candidate["warmup_record"]["actor"]["q"],
            "gradient": candidate["warmup_record"]["actor"]["gradient"],
        },
    }
    left_values = _numeric_leaves(baseline_core)
    right_values = _numeric_leaves(candidate_core)
    shared_values = sorted(set(left_values) & set(right_values))
    errors = {
        name: abs(left_values[name] - right_values[name])
        for name in shared_values
    }
    numeric_max = max(errors.values(), default=0.0)
    numeric_pass = all(
        math.isclose(
            left_values[name], right_values[name],
            abs_tol=float(config["equivalence"]["fp32_loss_atol"]),
            rel_tol=float(config["equivalence"]["fp32_loss_rtol"]),
        )
        for name in shared_values
    )
    required_trace_keys = {
        (kind, purpose)
        for kind in ("action", "noise")
        for purpose in ("td_next", "cql_current", "cql_next", "actor_guidance")
    }
    trace_complete = required_trace_keys <= set(baseline_trace) and required_trace_keys <= set(candidate_trace)
    passed = bool(
        trace_complete
        and all(item["within_tolerance"] for item in traces.values())
        and baseline_rows == candidate_rows
        and numeric_pass
        and baseline["named_generator_final_sha256"] == candidate["named_generator_final_sha256"]
        and candidate["frozen_parameter_hash_unchanged"]
        and candidate["action_contract_v2"]
    )
    return {
        "pass": passed,
        "classification": (
            "bitwise_exact" if passed and all(item["bitwise_equal"] for item in traces.values()) and numeric_max == 0.0
            else "numerically_equivalent_with_declared_tolerance" if passed
            else "failed"
        ),
        "trace_complete": trace_complete,
        "trace_comparison": traces,
        "row_identity_and_order_exact": baseline_rows == candidate_rows,
        "named_generator_final_state_exact": baseline["named_generator_final_sha256"] == candidate["named_generator_final_sha256"],
        "numeric_loss_q_gradient_max_abs_error": numeric_max,
        "numeric_loss_q_gradient_within_tolerance": numeric_pass,
        "action_tolerance": {"atol": action_atol, "rtol": action_rtol},
        "fp32_summary_tolerance": {
            "atol": float(config["equivalence"]["fp32_loss_atol"]),
            "rtol": float(config["equivalence"]["fp32_loss_rtol"]),
        },
    }


def write_summary(
    *, config: dict[str, Any], results: list[dict[str, Any]], destination: Path,
) -> dict[str, Any]:
    baseline = results[0]
    equivalence = {
        item["candidate"]["id"]: compare_equivalence(baseline, item, config)
        for item in results
        if item["status"] == "pass"
    }
    eligible = [
        item for item in results
        if item["status"] == "pass"
        and equivalence[item["candidate"]["id"]]["pass"]
    ]
    require(eligible, "THROUGHPUT_V2_NO_EQUIVALENT_CANDIDATE")
    selected = min(eligible, key=lambda item: item["seconds_per_cycle"]["mean"])
    summary = {
        "schema_version": "forcesmolvla_stage2_throughput_v2_summary.v2",
        "status": "pass",
        "config": CONFIG.relative_to(ROOT).as_posix(),
        "candidates": results,
        "candidate_failures_preserved": [
            {"candidate": item["candidate"]["id"], "status": item["status"], "error": item.get("error")}
            for item in results if item["status"] != "pass"
        ],
        "equivalence": equivalence,
        "speedup_vs_baseline": {
            item["candidate"]["id"]: baseline["seconds_per_cycle"]["mean"] / item["seconds_per_cycle"]["mean"]
            for item in results
        },
        "recommended_candidate": selected["candidate"]["id"],
        "recommended_cycle_seconds_mean": selected["seconds_per_cycle"]["mean"],
        "recommended_actor_transitions_per_second": selected["actor_transitions_per_second"],
        "recommended_critic_transitions_per_second": selected["critic_transitions_per_second"],
        "recommended_joint_cycles_per_hour": selected["joint_cycles_per_hour"],
        "long_run_started": False,
        "training_checkpoint_created": False,
        "robot_execution_authorized": False,
    }
    atomic_json(destination, summary)
    return summary


def coordinator() -> None:
    require(not OUTPUT.exists(), "THROUGHPUT_V2_OUTPUT_EXISTS")
    OUTPUT.mkdir(parents=True)
    config = yaml.safe_load(CONFIG.read_text())
    results = []
    for candidate in config["benchmark"]["candidates"]:
        path = OUTPUT / f"{candidate['id']}.json"
        result = run_worker(candidate, path)
        results.append(result)
        print(f"THROUGHPUT_V2_RESULT {candidate['id']} {result['status']}", flush=True)
    write_summary(
        config=config, results=results, destination=OUTPUT / "summary.json"
    )


def recompute_existing_summary() -> None:
    destination = OUTPUT / "summary.v2.json"
    require(
        OUTPUT.exists() and not destination.exists(),
        "THROUGHPUT_V2_RECOMPUTE_TARGET",
    )
    config = yaml.safe_load(CONFIG.read_text())
    results = [
        json.loads((OUTPUT / f"{candidate['id']}.json").read_text())
        for candidate in config["benchmark"]["candidates"]
    ]
    write_summary(config=config, results=results, destination=destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--recompute-existing-summary", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        require(args.candidate and args.result and not args.result.exists(), "THROUGHPUT_V2_WORKER_ARGS")
        candidate = json.loads(args.candidate.read_text())
        try:
            result = worker(candidate, args.result)
        except BaseException as error:
            if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
                result = {
                    "schema_version": "forcesmolvla_stage2_throughput_v2_candidate.v1",
                    "status": "oom", "candidate": candidate, "error": str(error),
                    "candidate_state_discarded": True, "training_checkpoint_created": False,
                }
            else:
                raise
        atomic_json(args.result, result)
    elif args.recompute_existing_summary:
        recompute_existing_summary()
    else:
        require(args.run, "pass --run")
        coordinator()


if __name__ == "__main__":
    main()
