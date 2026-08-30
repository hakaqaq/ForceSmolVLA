#!/usr/bin/env python3
"""Append-only Candidate-B worker for bounded-cache benchmarks and resume tests."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import torch
import yaml


ROOT = Path(__file__).parents[1].resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/lerobot/src"), str(ROOT / "tools")]
CONFIG = ROOT / "configs/stage2b_long_run_half_pass_throughput_v2.development.yaml"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: Any) -> str:
    from forcesmolvla.rft.canonical_state import canonical_digest

    return canonical_digest(value)


class CycleTrace:
    """Exact-preflight trace; disabled during throughput measurements."""

    def __init__(self, context: dict, enabled: bool) -> None:
        self.context = context
        self.enabled = enabled
        self.gradient_digests: list[dict] = []
        self.q_output_digests: list[dict] = []
        self.critic_action_digests: list[dict] = []
        self.actor_random_digests: list[dict] = []
        self.hooks = []
        self.stack = ExitStack()

    def __enter__(self):
        if not self.enabled:
            return self
        original_clip = torch.nn.utils.clip_grad_norm_

        def clip(parameters, *args, **kwargs):
            values = list(parameters)
            before = {
                str(index): parameter.grad
                for index, parameter in enumerate(values)
                if parameter.grad is not None
            }
            result = original_clip(values, *args, **kwargs)
            after = {
                str(index): parameter.grad
                for index, parameter in enumerate(values)
                if parameter.grad is not None
            }
            self.gradient_digests.append({
                "preclip": tensor_digest(before),
                "postclip": tensor_digest(after),
                "parameter_count": len(values),
            })
            return result

        self.stack.enter_context(patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=clip))
        original_randn = torch.randn
        original_rand = torch.rand

        def capture_random(label, original):
            def wrapped(*args, **kwargs):
                value = original(*args, **kwargs)
                generator = kwargs.get("generator")
                if generator is not None and value.is_cuda:
                    self.actor_random_digests.append({
                        "kind": label,
                        "shape": list(value.shape),
                        "sha256": tensor_digest(value),
                    })
                return value
            return wrapped

        self.stack.enter_context(patch.object(torch, "randn", side_effect=capture_random("randn", original_randn)))
        self.stack.enter_context(patch.object(torch, "rand", side_effect=capture_random("rand", original_rand)))

        def q_hook(name):
            def hook(_module, inputs, output):
                self.q_output_digests.append({"module": name, "sha256": tensor_digest(output)})
                actions = [
                    value for value in inputs
                    if isinstance(value, torch.Tensor) and tuple(value.shape[-2:]) == (3, 7)
                ]
                masks = [
                    value for value in inputs
                    if isinstance(value, torch.Tensor) and value.dtype == torch.bool
                    and value.ndim == 2 and value.shape[-1] == 3
                ]
                if actions:
                    self.critic_action_digests.append({
                        "module": name,
                        "action_sha256": tensor_digest(actions[0]),
                        "mask_sha256": tensor_digest(masks[0]) if masks else None,
                    })
            return hook

        for name in ("q1", "q2", "q1_target", "q2_target"):
            self.hooks.append(self.context[name].register_forward_hook(q_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for hook in self.hooks:
            hook.remove()
        self.stack.__exit__(exc_type, exc, tb)

    def result(self) -> dict:
        result = {
            "gradient_tensors": self.gradient_digests,
            "q_tensors": self.q_output_digests,
            "critic_action_contract_tensors": self.critic_action_digests,
            "actor_flow_noise_and_timestep_tensors": self.actor_random_digests,
        }
        result["digest"] = tensor_digest(result)
        return result


def full_training_state(
    *, context: dict, actor_optimizer, actor_scheduler, samplers: dict,
    generators: dict, cycle: int, ownership: dict, g5,
) -> dict:
    from forcesmolvla.rft.canonical_state import (
        module_mode_and_grad_record,
        optimizer_parameter_name_groups,
        training_state_payload,
    )
    from forcesmolvla.rft.long_run_checkpoint import counters_for_cycle

    modules = {name: context[name] for name in ("actor", "q1", "q2", "q1_target", "q2_target")}
    actor_names = dict(context["actor"].named_parameters())
    critic_names = {
        **{f"q1.{name}": value for name, value in context["q1"].named_parameters()},
        **{f"q2.{name}": value for name, value in context["q2"].named_parameters()},
    }
    boundary = module_mode_and_grad_record(modules)
    boundary["cycle"] = cycle
    return training_state_payload(
        modules=modules,
        actor_optimizer=actor_optimizer,
        critic_optimizer=context["optimizer"],
        actor_parameter_name_groups=optimizer_parameter_name_groups(actor_optimizer, actor_names),
        critic_parameter_name_groups=optimizer_parameter_name_groups(context["optimizer"], critic_names),
        actor_scheduler_state=actor_scheduler.state_dict(),
        critic_scheduler_state=context["scheduler"].state_dict(),
        sampler_states={name: sampler.state_dict() for name, sampler in samplers.items()},
        rng_states=g5.capture_rng_states(generators),
        counters=counters_for_cycle(cycle),
        boundary_manifest=boundary,
        ownership_manifest=ownership,
    )


def save_recovery(
    path: Path, *, cycle: int, context: dict, actor_optimizer, actor_scheduler,
    samplers: dict, generators: dict, ownership: dict, g5,
) -> dict:
    from forcesmolvla.rft.canonical_state import canonical_digest
    from forcesmolvla.rft.long_run_checkpoint import save_cycle_checkpoint

    rng = g5.capture_rng_states(generators)
    before = canonical_digest(rng)
    manifest = save_cycle_checkpoint(
        path,
        cycle=cycle,
        modules={name: context[name] for name in ("actor", "q1", "q2", "q1_target", "q2_target")},
        actor_optimizer=actor_optimizer,
        critic_optimizer=context["optimizer"],
        actor_scheduler=actor_scheduler,
        critic_scheduler=context["scheduler"],
        sampler_states={name: sampler.state_dict() for name, sampler in samplers.items()},
        rng_states=rng,
        ownership_manifest=ownership,
        protected_snapshot={"scope": "throughput_v2_exact_resume_preflight_only"},
        startup_snapshot_bytes={
            "config/stage2b_long_run_half_pass_throughput_v2.development.yaml": CONFIG.read_bytes(),
        },
        replace_rolling=False,
    )
    after = canonical_digest(g5.capture_rng_states(generators))
    require(before == after, "THROUGHPUT_V2_CHECKPOINT_CONSUMED_RNG")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "save_side_effect_digest_before": before,
        "save_side_effect_digest_after": after,
        "save_side_effect_free": True,
    }


def run(args) -> dict:
    import benchmark_stage2_batch_scaling_gpu as benchmark
    from forcesmolvla.rft import training_cycle as g5
    from forcesmolvla.rft import critic_training as g7a
    import run_s2_g7b_worker as g7b
    import run_stage2b_long_run_half_pass_worker as legacy
    from forcesmolvla.rft.canonical_state import canonical_digest, module_record
    from forcesmolvla.rft.frozen_vlm_trainability import frozen_state_digest
    from forcesmolvla.rft.throughput_v2 import (
        FrozenPrefixFlowCounter,
        fast_polyak_update,
        lightweight_state_token,
    )
    from forcesmolvla.rft.throughput_v2_long_run import install_bounded_training_cache
    from forcesmolvla.rft import training_cycle

    config = yaml.safe_load(CONFIG.read_text())
    require(config["authorization"] == "integration_preflight_only_no_long_run", "THROUGHPUT_V2_WORKER_AUTH")
    require(args.critic_batch in {64, 96, 128}, "THROUGHPUT_V2_CRITIC_BATCH")
    require(args.cycles >= 1 and args.warmup_cycles >= 0, "THROUGHPUT_V2_CYCLE_COUNT")
    g5.install_open_audit()
    device = g7a.configure_runtime()
    _old_config, training = legacy.load_config()
    training["batching"].update({
        "critic_batch_size": args.critic_batch,
        "calql_batch_size": args.critic_batch,
        "actor_microbatch_size": 24,
        "actor_gradient_accumulation": 1,
        "actor_effective_batch_size": 24,
    })
    training["loss"]["eta_actor_q"] = 3.0
    (
        context, parent_sampler_states, parent_rng,
        actor_optimizer, actor_scheduler, actor_ownership, trainability, _r2,
    ) = legacy.build_context(device, with_data=True)
    if args.resume_checkpoint:
        samplers, generators, resume_audit = legacy.load_checkpoint(
            args.resume_checkpoint,
            expected_cycle=args.start_cycle,
            context=context,
            actor_optimizer=actor_optimizer,
            actor_scheduler=actor_scheduler,
            training=training,
        )
    else:
        require(args.start_cycle == 0, "THROUGHPUT_V2_FRESH_START_CYCLE")
        generators = g7b.build_generators(training)
        samplers = g7b.build_samplers(context["data"], generators, parent_sampler_states)
        g7b.restore_parent_rng(parent_rng, generators)
        resume_audit = None
    ownership = {
        "actor": actor_ownership,
        "critic": context["ownership"],
        "actor_critic_parameter_intersection": 0,
        "frozen_actor_parameter_in_actor_optimizer": 0,
        "target_in_optimizer": 0,
        "target_actor": None,
    }
    frozen_before = frozen_state_digest(context["actor"])
    initial = {
        name: module_record(context[name])["digest"]
        for name in ("actor", "q1", "q2", "q1_target", "q2_target")
    }
    telemetry = benchmark.GpuTelemetry().__enter__()
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    cache_report = None
    cold_started = time.perf_counter()
    with install_bounded_training_cache(
        context["data"],
        max_bytes=int(config["cache"]["decoded_cache_max_bytes"]),
        prefetch_workers=int(config["cache"]["decode_workers"]),
    ) as cache:
        cold_initialization_seconds = time.perf_counter() - cold_started
        try:
            for offset in range(args.cycles):
                cycle = args.start_cycle + offset + 1
                measured = offset >= args.warmup_cycles
                counters = []
                captured_flow: dict[str, str] = {}

                def counter_factory(*factory_args, **factory_kwargs):
                    requested = (
                        int(factory_args[0]) if factory_args
                        else int(factory_kwargs["inference_batch_size"])
                    )
                    require(requested == 4, "THROUGHPUT_V2_FLOW_SUBBATCH_DRIFT")
                    counter = FrozenPrefixFlowCounter(4, capture=args.trace)
                    counters.append(counter)
                    return counter

                torch.cuda.synchronize()
                cycle_started = time.perf_counter()
                critic_reports = []
                step_states = []
                actor_token = lightweight_state_token(context["actor"])
                with CycleTrace(context, args.trace) as trace, g7b.critic_internal_only():
                    for substep in range(2):
                        with (
                            patch.object(benchmark, "TimedFlowCounter", side_effect=counter_factory),
                            patch.object(training_cycle, "module_state_sha256", side_effect=lightweight_state_token),
                            patch.object(training_cycle, "polyak_update_verified", side_effect=fast_polyak_update),
                        ):
                            report = benchmark.critic_update(
                                context=context,
                                training=training,
                                generators=generators,
                                samplers=samplers,
                                batch_size=args.critic_batch,
                                update_id=256 + 2 * (cycle - 1) + substep + 1,
                            )
                        critic_reports.append(legacy.compact_critic(report))
                        if args.trace:
                            step_states.append({
                                "substep": substep + 1,
                                "q1": module_record(context["q1"])["digest"],
                                "q2": module_record(context["q2"])["digest"],
                                "q1_target": module_record(context["q1_target"])["digest"],
                                "q2_target": module_record(context["q2_target"])["digest"],
                                "rng": canonical_digest(g5.capture_rng_states(generators)),
                                "samplers": canonical_digest({name: sampler.state_dict() for name, sampler in samplers.items()}),
                            })
                    require(lightweight_state_token(context["actor"]) == actor_token, "THROUGHPUT_V2_ACTOR_CHANGED_DURING_CRITICS")
                    actor_indices = samplers["actor"].draw(24)
                    actor_load_started = time.perf_counter()
                    actor_batch = context["data"].build_batch(
                        actor_indices,
                        context["actor"],
                        device,
                        canonical_task_feature=context["q1"].canonical_task_feature,
                        include_flow_actions=True,
                    )
                    actor_load_seconds = time.perf_counter() - actor_load_started
                    actor_report = legacy.actor_update_eta3(
                        cycle=cycle,
                        context=context,
                        batch=actor_batch,
                        optimizer=actor_optimizer,
                        scheduler=actor_scheduler,
                        generators=generators,
                    )
                    del actor_batch
                torch.cuda.synchronize()
                cycle_seconds = time.perf_counter() - cycle_started
                for counter_index, counter in enumerate(counters):
                    for name, tensor in counter.captured.items():
                        captured_flow[f"counter{counter_index}/{name}"] = tensor_digest(tensor)
                require(
                    all(math.isfinite(float(value)) for report in critic_reports for value in report["loss"].values())
                    and all(math.isfinite(float(value)) for value in actor_report["loss"].values()),
                    "THROUGHPUT_V2_NONFINITE",
                )
                require(
                    actor_report["gradient"]["tcp6_q_norm"] > 0.0
                    and actor_report["gradient"]["gripper_q_max_abs"] == 0.0
                    and actor_report["gradient"]["gripper_fm_norm"] > 0.0
                    and actor_report["prefix_audit"]["force_kv_projection_count"] == 1,
                    "THROUGHPUT_V2_ACTION_CONTRACT",
                )
                record = {
                    "cycle": cycle,
                    "warmup": not measured,
                    "cycle_seconds": cycle_seconds,
                    "critic_updates": critic_reports,
                    "actor_update": actor_report,
                    "actor_batch_count": 24,
                    "actor_batch_identity_sha256": legacy.canonical(
                        context["data"].identity_records(actor_indices)
                    ),
                    "actor_data_loading_seconds": actor_load_seconds,
                    "flow_counters": [counter.report() for counter in counters],
                    "trace": trace.result() if args.trace else None,
                    "captured_flow_noise_and_actions": captured_flow,
                    "post_critic_step_states": step_states,
                }
                if args.trace:
                    record["post_actor_state"] = {
                        "actor": module_record(context["actor"])["digest"],
                        "rng": canonical_digest(g5.capture_rng_states(generators)),
                        "samplers": canonical_digest({name: sampler.state_dict() for name, sampler in samplers.items()}),
                    }
                    record["cycle_training_state"] = full_training_state(
                        context=context,
                        actor_optimizer=actor_optimizer,
                        actor_scheduler=actor_scheduler,
                        samplers=samplers,
                        generators=generators,
                        cycle=cycle,
                        ownership=ownership,
                        g5=g5,
                    )
                records.append(record)
                print(
                    f"THROUGHPUT_V2_LONG_RUN_WORKER critic={args.critic_batch} "
                    f"cycle={cycle} local={offset + 1}/{args.cycles}",
                    flush=True,
                )
        finally:
            cache_report = cache.report()
            telemetry.__exit__(None, None, None)
    frozen_after = frozen_state_digest(context["actor"])
    require(frozen_before == frozen_after, "THROUGHPUT_V2_FROZEN_HASH_CHANGED")
    measured_records = [record for record in records if not record["warmup"]]
    measured_seconds = sum(record["cycle_seconds"] for record in measured_records)
    final_cycle = args.start_cycle + args.cycles
    state = full_training_state(
        context=context,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        samplers=samplers,
        generators=generators,
        cycle=final_cycle,
        ownership=ownership,
        g5=g5,
    )
    checkpoint = None
    if args.checkpoint_out:
        checkpoint = save_recovery(
            args.checkpoint_out,
            cycle=final_cycle,
            context=context,
            actor_optimizer=actor_optimizer,
            actor_scheduler=actor_scheduler,
            samplers=samplers,
            generators=generators,
            ownership=ownership,
            g5=g5,
        )
    final = {
        name: module_record(context[name])["digest"]
        for name in ("actor", "q1", "q2", "q1_target", "q2_target")
    }
    return {
        "schema_version": "forcesmolvla_stage2b_throughput_v2_worker.v1",
        "status": "pass",
        "pid": os.getpid(),
        "environment": g7a.environment_audit(),
        "start_cycle": args.start_cycle,
        "end_cycle": final_cycle,
        "critic_batch": args.critic_batch,
        "actor_batch": 24,
        "warmup_cycles": args.warmup_cycles,
        "measured_cycles": len(measured_records),
        "records": records,
        "seconds_per_measured_cycle": benchmark.describe(
            [record["cycle_seconds"] for record in measured_records]
        ) if measured_records else None,
        "actor_transitions_per_second": (
            24.0 * len(measured_records) / measured_seconds if measured_seconds else None
        ),
        "critic_td_transitions_per_second": (
            2.0 * args.critic_batch * len(measured_records) / measured_seconds
            if measured_seconds else None
        ),
        "critic_calql_transitions_per_second": (
            2.0 * args.critic_batch * len(measured_records) / measured_seconds
            if measured_seconds else None
        ),
        "total_critic_row_memberships_per_second": (
            4.0 * args.critic_batch * len(measured_records) / measured_seconds
            if measured_seconds else None
        ),
        "joint_cycles_per_hour": (
            len(measured_records) * 3600.0 / measured_seconds if measured_seconds else None
        ),
        "gpu_utilization_percent": benchmark.describe(telemetry.utilization or [0.0]),
        "gpu_power_watts": benchmark.describe(telemetry.power or [0.0]),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "peak_cpu_rss_bytes": cache_report["process_rss_peak_bytes"],
        "cache": cache_report,
        "cold_start_initialization_seconds": cold_initialization_seconds,
        "prefix_prefill_count": sum(
            int(counter["prefix_prefill_count"])
            for record in measured_records for counter in record["flow_counters"]
        ),
        "flow_call_count": sum(
            int(counter["flow_chunks_sampled"])
            for record in measured_records for counter in record["flow_counters"]
        ),
        "euler_velocity_evaluation_count": sum(
            int(counter["euler_velocity_evaluations"])
            for record in measured_records for counter in record["flow_counters"]
        ),
        "training_state": state,
        "checkpoint": checkpoint,
        "resume_audit": resume_audit,
        "parameter_change_matrix": {
            name: {"before": initial[name], "after": final[name], "changed": initial[name] != final[name]}
            for name in initial
        },
        "frozen_parameter_hash_unchanged": True,
        "all_losses_and_gradients_finite": True,
        "action_contract_v2": True,
        "flow_inference_subbatch": 4,
        "long_run_started": False,
        "candidate_state_discarded": args.checkpoint_out is None,
        "access_audit": {
            "validation_reads": 0,
            "test_reads": 0,
            "manual_g1_opens": len(g5.FORBIDDEN_OPENS["manual_g1"]),
            "manual_label_opens": len(g5.FORBIDDEN_OPENS["manual_labels"]),
            "reward_classifier_inference": 0,
        },
        "trainability": {
            "frozen_parameter_count": trainability.frozen_parameter_count,
            "trainable_actor_parameter_count": trainability.trainable_actor_parameter_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--critic-batch", type=int, required=True)
    parser.add_argument("--start-cycle", type=int, default=0)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--warmup-cycles", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    result = run(args)
    atomic_json(args.result, result)
    print(json.dumps({
        "status": result["status"],
        "critic_batch": result["critic_batch"],
        "end_cycle": result["end_cycle"],
        "cycles_per_hour": result["joint_cycles_per_hour"],
        "training_state_digest": result["training_state"]["training_state_digest"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
