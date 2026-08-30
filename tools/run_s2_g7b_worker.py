#!/usr/bin/env python3
"""G7-B eight-cycle joint-smoke worker and strict-load verifier."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
CONFIG = ROOT / "configs/stage2_g7b_joint_smoke.development.yaml"
G5_CONFIG = ROOT / "configs/stage2_g5_single_cycle.v2.development.yaml"
PARENT = ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value) -> None:
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


def load_config() -> tuple[dict, dict]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    training = yaml.safe_load(G5_CONFIG.read_text(encoding="utf-8"))
    joint = config["joint_smoke"]
    require(joint == {
        "eta_actor_q": 10.0,
        "eta_status": "development_smoke_only",
        "beta_flow": 1.0,
        "joint_cycles": 8,
        "critic_updates_per_actor_update": 2,
        "actor_updates_per_cycle": 1,
        "expected_critic_updates": 16,
        "expected_actor_updates": 8,
        "expected_polyak_updates_per_target": 16,
        "target_actor": None,
    }, "G7B_RECIPE_DRIFT")
    require(file_sha(G5_CONFIG) == config["frozen_recipe_parent"]["sha256"], "G7B_G5_CONFIG_SHA_DRIFT")
    require(training["loss"]["beta_flow"] == 1.0, "G7B_BETA_DRIFT")
    require(training["loss"]["alpha_calql"] == 0.1, "G7B_ALPHA_DRIFT")
    require(training["loss"]["cql_candidates_per_source_M"] == 2, "G7B_CANDIDATE_COUNT_DRIFT")
    require(training["targets"]["polyak_tau"] == 0.005, "G7B_POLYAK_DRIFT")
    training = copy.deepcopy(training)
    training["loss"]["eta_actor_q"] = 10.0
    return config, training


def generator(device: str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def build_generators(training: dict) -> dict[str, torch.Generator]:
    seeds = training["rng"]["named_stream_seeds"]
    cpu = {"td_sampler", "calql_sampler", "actor_sampler", "empirical_random_proposal"}
    return {name: generator("cpu" if name in cpu else "cuda", int(seed)) for name, seed in seeds.items()}


def restore_parent_rng(parent_rng: dict, generators: dict[str, torch.Generator]) -> None:
    import random

    for name, state in parent_rng["named_generator_states"].items():
        generators[name].set_state(state)
    random.setstate(parent_rng["python_random_state"])
    np.random.set_state(parent_rng["numpy_random_state"])
    torch.set_rng_state(parent_rng["torch_cpu_rng_state"])
    torch.cuda.set_rng_state_all(parent_rng["torch_cuda_rng_states"])


def build_samplers(data, generators, parent_states: dict):
    from forcesmolvla.rft.training_cycle import SerializableReplacementSampler, SerializableUniqueSampler

    td = parent_states["td"]
    calql = parent_states["calql"]
    proposal = parent_states["empirical_random_proposal"]
    samplers = {
        "td": SerializableUniqueSampler(td["name"], tuple(td["population"]), generators["td_sampler"], int(td["draws"])),
        "calql": SerializableUniqueSampler(calql["name"], tuple(calql["population"]), generators["calql_sampler"], int(calql["draws"])),
        "empirical_random_proposal": SerializableReplacementSampler(
            proposal["name"], int(proposal["population_size"]),
            generators["empirical_random_proposal"], int(proposal["draws"]),
        ),
        "actor": SerializableUniqueSampler("Actor_sampler", data.actor_population, generators["actor_sampler"]),
    }
    require(all(samplers[name].draws == 256 for name in ("td", "calql", "empirical_random_proposal")), "G7B_PARENT_SAMPLER_POSITION_DRIFT")
    return samplers


def load_parent(context: dict) -> tuple[dict, dict]:
    from forcesmolvla.rft.critic_warmup_checkpoint import (
        CRITIC_WARMUP_COUNTERS,
        module_component_digests,
        validate_critic_warmup_checkpoint,
    )

    manifest = validate_critic_warmup_checkpoint(PARENT)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    require(manifest["counters"] == CRITIC_WARMUP_COUNTERS, "G7B_PARENT_COUNTER_DRIFT")
    for name in ("q1", "q2", "q1_target", "q2_target"):
        state = torch.load(PARENT / f"models/{name}_state.pt", map_location="cpu", weights_only=False)
        incompatible = context[name].load_state_dict(state, strict=True)
        require(not incompatible.missing_keys and not incompatible.unexpected_keys, f"G7B_PARENT_MODEL_LOAD:{name}")
    context["optimizer"].load_state_dict(torch.load(
        PARENT / "optimizers/critic_optimizer_state.pt", map_location="cpu", weights_only=False
    ))
    context["scheduler"].load_state_dict(torch.load(
        PARENT / "schedulers/critic_scheduler_state.pt", map_location="cpu", weights_only=False
    ))
    require(context["scheduler"].last_epoch == 256, "G7B_PARENT_SCHEDULER_POSITION_DRIFT")
    actor_binding = json.loads((PARENT / "manifests/actor_binding.json").read_text())
    require(module_component_digests(context["actor"]) == actor_binding["state_final"], "G7B_PARENT_R5_ACTOR_BINDING_DRIFT")
    require(config["parent"]["tree_sha256"] == "f8c08b9058d173211a7306d370a97a848bfc1f7569ac52e6cc88baacff0c0d40", "G7B_PARENT_TREE_BINDING_DRIFT")
    sampler_states = torch.load(PARENT / "state/sampler_states.pt", map_location="cpu", weights_only=False)
    rng_states = torch.load(PARENT / "state/rng_states.pt", map_location="cpu", weights_only=False)
    return sampler_states, rng_states


@contextmanager
def critic_internal_only():
    from forcesmolvla import action_delta, rules
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy

    error = RuntimeError("G7B_PUBLIC_EXECUTION_PATH_CALLED_FROM_TRAINING")
    with (
        patch.object(action_delta.ActionDeltaProcessor, "from_delta", side_effect=error) as inverse,
        patch.object(action_delta.ActionSafetyProfile, "validate_chunk", side_effect=error) as safety,
        patch.object(rules, "load_and_validate_rulespec", side_effect=error) as rulespec,
        patch.object(ForceSmolVLAPolicy, "predict_action_chunk", side_effect=error) as predict,
    ):
        yield
    require(inverse.call_count == safety.call_count == rulespec.call_count == predict.call_count == 0, "G7B_INTERNAL_PUBLIC_CALL")


def fixed_diagnostic_bundle(config: dict, data, policy, q1, device) -> dict:
    row_index = int(data.actor_population[int(config["diagnostics"]["fixed_train_actor_population_offset"])])
    batch = data.build_batch([row_index], policy, device, canonical_task_feature=q1.canonical_task_feature, include_flow_actions=True)
    seeds = config["diagnostics"]
    make = lambda seed: torch.randn(1, 50, 7, generator=generator("cuda", int(seed)), device=device)
    fm_time_gen = generator("cuda", int(seeds["fixed_gradient_fm_timestep_seed"]))
    bundle = {
        "row_index": row_index,
        "batch": batch,
        "action_noise": make(seeds["fixed_action_noise_seed"]),
        "actor_q_noise": torch.cat([make(seeds["fixed_gradient_q_noise_seed"]) for _ in range(8)], dim=0),
        "actor_fm_noise": torch.cat([make(seeds["fixed_gradient_fm_noise_seed"]) for _ in range(8)], dim=0),
        "actor_fm_timestep": torch.rand(8, generator=fm_time_gen, device=device),
    }
    return bundle


def internal_action_diagnostic(policy, batch: dict, noise: torch.Tensor, delta_mean, delta_std) -> dict:
    from forcesmolvla.rft.critic_action_adapter_v2 import critic_action_for_q_guidance_v2
    from forcesmolvla.rft.training_cycle import tensor_sha256

    mode = policy.training
    policy.eval()
    # Use the same frozen N=10 cached Flow wrapper as Actor-Q.
    from forcesmolvla.rft.flow_sampling import sample_normalized_action_chunk_with_grad
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        chunk = sample_normalized_action_chunk_with_grad(
            policy, batch["current_actor_batch"], noise,
            call_id="g7b-fixed-action-diagnostic", purpose="td_next",
        )
    q_action = critic_action_for_q_guidance_v2(
        chunk, delta_action_mean7=delta_mean, delta_action_std7=delta_std
    )
    policy.train(mode)
    return {
        "normalized_tcp6": q_action[0, :, :6].detach().float().cpu().tolist(),
        "normalized_gripper": q_action[0, :, 6].detach().float().cpu().tolist(),
        "q_action_sha256": tensor_sha256(q_action),
        "finite": bool(torch.isfinite(q_action).all()),
    }


def public_diagnostic(policy, data, actor_batch: dict, noise: torch.Tensor, cycle: int) -> dict:
    from forcesmolvla.context import ChunkContext
    from forcesmolvla.rules import load_and_validate_rulespec
    from tools.serve_policy import bind_policy_action_safety

    policy.bind_runtime_artifacts(data.runtime)
    rules_path = ROOT / "configs/live_action_safety.task2.development.yaml"
    schema = ROOT / "schemas/rulespec.schema.json"
    rules = load_and_validate_rulespec(rules_path, schema, formal=False)
    bind_policy_action_safety(policy, rules, rules_sha256=file_sha(rules_path), approved_development_execution=True)
    batch = {}
    for name, value in actor_batch.items():
        if isinstance(value, torch.Tensor):
            batch[name] = value[:1]
        elif isinstance(value, (tuple, list)):
            batch[name] = type(value)(value[:1])
        else:
            batch[name] = value
    normalized_state = batch["observation.state"][0, :7].detach().cpu().numpy().astype(np.float64)
    raw_state = torch.tensor(data.runtime.normalizer.state7.inverse(normalized_state), dtype=torch.float32).view(1, 7)
    batch["raw_state_snapshot"] = raw_state.to(batch["observation.state"].device)
    valid = torch.ones(1, 50, dtype=torch.bool)
    runtime = data.runtime
    context = ChunkContext(
        policy_generation=policy._context_generation,
        raw_state_snapshot=raw_state,
        t_ref_ns=torch.tensor([cycle + 1], dtype=torch.int64),
        tau0_ns=torch.tensor([cycle + 1], dtype=torch.int64),
        clock_domain_id=("g7b_fixed_diagnostic",), episode_id=("g7b-train-probe",),
        session_id=("g7b",), sample_id=(f"g7b-{cycle}",), chunk_id=(f"g7b-public-{cycle}",),
        action_valid_mask=valid, suffix_valid_mask=valid.clone(),
        calibration_bundle_hash=(runtime.calibration_bundle_sha256,),
        wrench_geometry_spec_hash=(runtime.wrench_geometry_spec_sha256,),
        normalizer_hash=(runtime.normalizer_manifest_sha256,),
        calibration_mapping_hash_or_none=(None,), wrench_geometry_valid=torch.ones(1, dtype=torch.bool),
        runtime_artifact_compatible=torch.ones(1, dtype=torch.bool), selected_provenance=({"scope": "fixed_train_diagnostic"},),
    )
    mode = policy.training
    policy.eval()
    # Match the frozen public serving call-site's bf16 outer autocast.  The
    # model retains its previously approved fp32 islands internally.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = policy.predict_action_chunk(batch, chunk_context=context, noise=noise)
    policy.train(mode)
    require(tuple(action.shape) == (1, 50, 7) and bool(torch.isfinite(action).all()), "G7B_PUBLIC_PREDICT_INVALID")
    return {
        "cycle": cycle,
        "success": True,
        "absolute_action_sha256": hashlib.sha256(action.detach().float().cpu().numpy().tobytes()).hexdigest(),
        "binary_gripper_values_m": sorted(set(float(x) for x in action[0, :, 6].cpu().tolist())),
    }


def load_models_and_state(device, with_data: bool = True):
    from forcesmolvla.rft import critic_training as g7a_worker
    from forcesmolvla.rft.joint_training_checkpoint import build_actor_optimizer_scheduler

    context = g7a_worker.initialize_fresh(device=device, with_data=with_data)
    sampler_states, parent_rng = load_parent(context)
    actor_optimizer, actor_scheduler, actor_ownership = build_actor_optimizer_scheduler(context["actor"])
    return context, sampler_states, parent_rng, actor_optimizer, actor_scheduler, actor_ownership, g7a_worker


def train(args) -> None:
    from forcesmolvla.rft import critic_training as g7a_worker
    from forcesmolvla.rft import training_cycle as g5
    from forcesmolvla.rft.canonical_state import canonical_digest
    from forcesmolvla.rft.critic import modules_storage_independent
    from forcesmolvla.rft.critic_warmup_checkpoint import module_component_digests
    from forcesmolvla.rft.joint_training_checkpoint import (
        JOINT_TRAINING_COUNTERS,
        describe_p95,
        save_joint_training_checkpoint,
        validate_joint_training_checkpoint,
        validate_optimizer_step_sets,
    )
    from forcesmolvla.rft.training_cycle import ensure_all_gradients_none, module_state_sha256, optimizer_state_storage_independent

    g5.install_open_audit()
    device = g7a_worker.configure_runtime()
    config, training = load_config()
    context, parent_sampler_states, parent_rng, actor_optimizer, actor_scheduler, actor_ownership, r2 = load_models_and_state(device)
    data = context["data"]
    generators = build_generators(training)
    samplers = build_samplers(data, generators, parent_sampler_states)
    restore_parent_rng(parent_rng, generators)
    policy, q1, q2 = context["actor"], context["q1"], context["q2"]
    q1_target, q2_target = context["q1_target"], context["q2_target"]
    critic_optimizer, critic_scheduler = context["optimizer"], context["scheduler"]
    fixed = fixed_diagnostic_bundle(config, data, policy, q1, device)
    initial = {name: module_state_sha256(module) for name, module in {
        "actor": policy, "q1": q1, "q2": q2, "q1_target": q1_target, "q2_target": q2_target
    }.items()}
    backbones = {f"{name}.{camera}": module_state_sha256(getattr(module, camera)) for name, module in (("q1", q1), ("q2", q2)) for camera in ("camera1_backbone", "camera2_backbone")}
    diagnostic_initial = internal_action_diagnostic(policy, fixed["batch"], fixed["action_noise"], fixed["batch"]["delta_mean"], fixed["batch"]["delta_std"])
    rng_before_public = canonical_digest(g5.capture_rng_states(generators))
    public = [public_diagnostic(policy, data, fixed["batch"]["current_actor_batch"], fixed["action_noise"], 0)]
    require(canonical_digest(g5.capture_rng_states(generators)) == rng_before_public, "G7B_PUBLIC_DIAGNOSTIC_CONSUMED_TRAINING_RNG")

    flow_counter = g5.FlowCounter(inference_batch_size=4)
    reports, gradient_reports, action_reports = [], [], [diagnostic_initial]
    torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()
    for cycle in range(1, 9):
        cycle_started = time.perf_counter()
        critic_reports = []
        actor_before_critics = module_state_sha256(policy)
        with critic_internal_only():
            for local in range(2):
                step = 256 + (cycle - 1) * 2 + local + 1
                td_indices = samplers["td"].draw(16)
                calql_indices = samplers["calql"].draw(16)
                td_batch = data.build_batch(td_indices, policy, device, canonical_task_feature=q1.canonical_task_feature)
                calql_batch = data.build_batch(calql_indices, policy, device, canonical_task_feature=q1.canonical_task_feature)
                report = g5.critic_update(
                    step=step, policy=policy, q1=q1, q2=q2, q1_target=q1_target, q2_target=q2_target,
                    optimizer=critic_optimizer, scheduler=critic_scheduler, td_batch=td_batch,
                    calql_batch=calql_batch, train_data=data, proposal_sampler=samplers["empirical_random_proposal"],
                    generators=generators, flow_counter=flow_counter, config=training,
                )
                critic_reports.append(g7a_worker.compact_critic_report(report))
                del td_batch, calql_batch, report
                gc.collect(); torch.cuda.empty_cache()
            require(module_state_sha256(policy) == actor_before_critics, "G7B_ACTOR_CHANGED_DURING_CRITIC_UPDATES")
            fixed_scale = {
                "actor_q_noise": fixed["actor_q_noise"][cycle - 1:cycle],
                "actor_fm_noise": fixed["actor_fm_noise"][cycle - 1:cycle],
                "actor_fm_timestep": fixed["actor_fm_timestep"][cycle - 1:cycle],
            }
            gradient = g7a_worker.measure_actor_gradient_scale(
                policy=policy, q1=q1, q2=q2, train_data=data, actor_indices=[fixed["row_index"]],
                fixed=fixed_scale, device=device, eta_candidates=[10.0], band=[0.01, 0.10],
            )
            actor_indices = samplers["actor"].draw(4)
            actor_batch = data.build_batch(actor_indices, policy, device, canonical_task_feature=q1.canonical_task_feature, include_flow_actions=True)
            actor_report = g5.actor_update(
                policy=policy, q1=q1, q2=q2, q1_target=q1_target, q2_target=q2_target,
                optimizer=actor_optimizer, scheduler=actor_scheduler, actor_batch=actor_batch,
                generators=generators, flow_counter=flow_counter, config=training,
            )
            del actor_batch
        q_metric = gradient["per_probe"][0]["global"]
        gradient_reports.append({
            "cycle": cycle, "raw_q_over_fm": q_metric["raw_q_over_fm"],
            "weighted_eta_q_over_beta_fm": 10.0 * q_metric["raw_q_over_fm"],
            "cosine_similarity": q_metric["cosine_similarity"],
            "tcp6_q_gradient_norm": gradient["per_probe"][0]["tcp6_actor_q_gradient_norm"],
            "gripper_q_gradient_max_abs": gradient["per_probe"][0]["gripper_actor_q_gradient_max_abs"],
            "gripper_fm_gradient_norm": gradient["per_probe"][0]["gripper_flow_matching_gradient_norm"],
            "modules": gradient["per_probe"][0]["modules"],
        })
        action_now = internal_action_diagnostic(policy, fixed["batch"], fixed["action_noise"], fixed["batch"]["delta_mean"], fixed["batch"]["delta_std"])
        baseline_tcp = torch.tensor(diagnostic_initial["normalized_tcp6"])
        current_tcp = torch.tensor(action_now["normalized_tcp6"])
        action_now["cycle"] = cycle
        action_now["normalized_tcp_drift_l2"] = float((current_tcp - baseline_tcp).norm())
        action_now["binary_gripper_change_rate"] = float(np.mean(np.asarray(action_now["normalized_gripper"]) != np.asarray(diagnostic_initial["normalized_gripper"])))
        action_reports.append(action_now)
        rng_before_public = canonical_digest(g5.capture_rng_states(generators))
        public.append(public_diagnostic(policy, data, fixed["batch"]["current_actor_batch"], fixed["action_noise"], cycle))
        require(canonical_digest(g5.capture_rng_states(generators)) == rng_before_public, "G7B_PUBLIC_DIAGNOSTIC_CONSUMED_TRAINING_RNG")
        require(all(math.isfinite(float(value)) for item in critic_reports for value in item["loss"].values()), "G7B_NONFINITE_CRITIC_LOSS")
        require(all(math.isfinite(float(value)) for value in actor_report["loss"].values()), "G7B_NONFINITE_ACTOR_LOSS")
        reports.append({
            "cycle": cycle, "critic_updates": critic_reports, "actor_update": actor_report,
            "gradient_scale": gradient_reports[-1], "action_diagnostic": action_now,
            "public_predict": public[-1], "cycle_latency_seconds": time.perf_counter() - cycle_started,
        })
        print(f"G7B_JOINT_CYCLE {cycle}/8", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    ensure_all_gradients_none(policy, q1, q2, q1_target, q2_target)
    final = {name: module_state_sha256(module) for name, module in {
        "actor": policy, "q1": q1, "q2": q2, "q1_target": q1_target, "q2_target": q2_target
    }.items()}
    require(all(final[name] != initial[name] for name in final), "G7B_EXPECTED_PARAMETER_CHANGE_MISSING")
    require(critic_scheduler.last_epoch == 272 and actor_scheduler.last_epoch == 8, "G7B_SCHEDULER_COUNTER_DRIFT")
    critic_steps = {int(value["step"].item()) for value in critic_optimizer.state.values() if "step" in value}
    actor_steps = {int(value["step"].item()) for value in actor_optimizer.state.values() if "step" in value}
    validate_optimizer_step_sets(critic_steps, actor_steps)
    require(modules_storage_independent(q1, q2) and optimizer_state_storage_independent(critic_optimizer, q1, q2), "G7B_CRITIC_STORAGE_NOT_INDEPENDENT")
    require(backbones == {f"{name}.{camera}": module_state_sha256(getattr(module, camera)) for name, module in (("q1", q1), ("q2", q2)) for camera in ("camera1_backbone", "camera2_backbone")}, "G7B_FROZEN_BACKBONE_CHANGED")
    require(all(bool(torch.isfinite(parameter).all()) for module in (policy, q1, q2, q1_target, q2_target) for parameter in module.parameters()), "G7B_NONFINITE_PARAMETER")
    require(all(item["gripper_q_gradient_max_abs"] == 0.0 and item["tcp6_q_gradient_norm"] > 0.0 and item["gripper_fm_gradient_norm"] > 0.0 for item in gradient_reports), "G7B_ACTION_GRADIENT_CONTRACT")

    counters = dict(JOINT_TRAINING_COUNTERS)
    ownership = {
        "actor": actor_ownership, "critic": context["ownership"],
        "actor_critic_parameter_intersection": len({id(p) for p in policy.parameters()} & {id(p) for m in (q1, q2) for p in m.parameters()}),
        "target_in_optimizer": 0, "target_actor": None,
    }
    rng_states = g5.capture_rng_states(generators)
    sampler_states = {name: sampler.state_dict() for name, sampler in samplers.items()}
    startup_paths = {
        "config/stage2_g7b_joint_smoke.development.yaml": CONFIG,
        "config/stage2_g5_single_cycle.v2.development.yaml": G5_CONFIG,
        "parent/checkpoint_manifest.json": PARENT / "checkpoint_manifest.json",
    }
    startup = {name: path.read_bytes() for name, path in startup_paths.items()}
    modules = {"actor": policy, "q1": q1, "q2": q2, "q1_target": q1_target, "q2_target": q2_target}
    manifest = save_joint_training_checkpoint(
        args.checkpoint, modules=modules, actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer, actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler, counters=counters,
        parent_counters=json.loads((PARENT / "state/counters.json").read_text()),
        sampler_states=sampler_states, rng_states=rng_states,
        ownership_manifest=ownership, protected_snapshot=json.loads(args.protected.read_text()),
        startup_snapshot_bytes=startup,
    )
    validate_joint_training_checkpoint(args.checkpoint)
    raw = [item["raw_q_over_fm"] for item in gradient_reports]
    weighted = [item["weighted_eta_q_over_beta_fm"] for item in gradient_reports]
    cosine = [item["cosine_similarity"] for item in gradient_reports]
    raw_count = r2.DIAGNOSTIC["raw_gripper_values"]
    result = {
        "worker_mode": "train", "environment": g7a_worker.environment_audit(),
        "parent_loaded": True, "parent_critic_step": 256, "parent_actor_from_r5": True,
        "cycles": reports, "counters": counters,
        "gradient_scale_summary": {"raw": describe_p95(raw), "weighted_eta10": describe_p95(weighted), "cosine": describe_p95(cosine)},
        "parameter_change_matrix": {name: {"before": initial[name], "after": final[name], "changed": initial[name] != final[name]} for name in final},
        "action_diagnostics": action_reports, "public_predict_diagnostics": public,
        "action_contract_v2": {
            "internal_gripper_projection": "total_binary", "public_execution_authorization_separate": True,
            "raw_gripper_out_of_public_tolerance_rate": r2.DIAGNOSTIC["raw_gripper_out_of_public_tolerance"] / raw_count if raw_count else 0.0,
            "clipping_added": False, "resampling_added": False, "binary_ste_added": False,
        },
        "runtime": {"total_seconds": time.perf_counter() - run_started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "flow_counts": flow_counter.report()},
        "data_access": {"train_transitions_available": len(data.rows), "validation_transition_reads": 0, "test_transition_reads": 0, "manual_g1_opens": 0, "manual_label_opens": 0, "reward_classifier_inference": 0, "reward_classifier_updates": 0},
        "checkpoint_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "optimizer_parameter_step_values": {
            "critic": sorted(critic_steps), "actor_sparse_routing_aware": sorted(actor_steps),
            "global_critic_optimizer_updates": 16, "global_actor_optimizer_updates": 8,
        },
    }
    atomic_json(args.result, result)


def verify(args) -> None:
    from forcesmolvla.rft import critic_training as g7a_worker
    from forcesmolvla.rft.joint_training_checkpoint import (
        build_actor_optimizer_scheduler,
        validate_joint_training_checkpoint,
        validate_optimizer_step_sets,
    )
    from forcesmolvla.rft.training_cycle import SerializableReplacementSampler, SerializableUniqueSampler, ensure_all_gradients_none

    device = g7a_worker.configure_runtime()
    _config, training = load_config()
    context = g7a_worker.initialize_fresh(device=device, with_data=False)
    actor_optimizer, actor_scheduler, _ = build_actor_optimizer_scheduler(context["actor"])
    manifest = validate_joint_training_checkpoint(args.checkpoint)
    modules = {name: context[name] for name in ("actor", "q1", "q2", "q1_target", "q2_target")}
    for name, module in modules.items():
        incompatible = module.load_state_dict(torch.load(args.checkpoint / f"models/{name}_state.pt", map_location="cpu", weights_only=False), strict=True)
        require(not incompatible.missing_keys and not incompatible.unexpected_keys, f"G7B_VERIFY_MODEL_LOAD:{name}")
    actor_optimizer.load_state_dict(torch.load(args.checkpoint / "optimizers/actor_optimizer_state.pt", map_location="cpu", weights_only=False))
    context["optimizer"].load_state_dict(torch.load(args.checkpoint / "optimizers/critic_optimizer_state.pt", map_location="cpu", weights_only=False))
    actor_scheduler.load_state_dict(torch.load(args.checkpoint / "schedulers/actor_scheduler_state.pt", map_location="cpu", weights_only=False))
    context["scheduler"].load_state_dict(torch.load(args.checkpoint / "schedulers/critic_scheduler_state.pt", map_location="cpu", weights_only=False))
    states = torch.load(args.checkpoint / "state/sampler_states.pt", map_location="cpu", weights_only=False)
    rng = torch.load(args.checkpoint / "state/rng_states.pt", map_location="cpu", weights_only=False)
    generators = build_generators(training)
    samplers = {
        "td": SerializableUniqueSampler(states["td"]["name"], tuple(states["td"]["population"]), generators["td_sampler"], states["td"]["draws"]),
        "calql": SerializableUniqueSampler(states["calql"]["name"], tuple(states["calql"]["population"]), generators["calql_sampler"], states["calql"]["draws"]),
        "actor": SerializableUniqueSampler(states["actor"]["name"], tuple(states["actor"]["population"]), generators["actor_sampler"], states["actor"]["draws"]),
        "empirical_random_proposal": SerializableReplacementSampler(states["empirical_random_proposal"]["name"], states["empirical_random_proposal"]["population_size"], generators["empirical_random_proposal"], states["empirical_random_proposal"]["draws"]),
    }
    restore_parent_rng(rng, generators)
    context["q1_target"].make_permanent_eval_target(); context["q2_target"].make_permanent_eval_target()
    ensure_all_gradients_none(*modules.values())
    critic_steps = {int(v["step"].item()) for v in context["optimizer"].state.values() if "step" in v}
    actor_steps = {int(v["step"].item()) for v in actor_optimizer.state.values() if "step" in v}
    validate_optimizer_step_sets(critic_steps, actor_steps)
    require(context["scheduler"].last_epoch == 272 and actor_scheduler.last_epoch == 8, "G7B_VERIFY_SCHEDULER_DRIFT")
    require(all(bool(torch.isfinite(p).all()) for module in modules.values() for p in module.parameters()), "G7B_VERIFY_NONFINITE")
    atomic_json(args.result, {
        "worker_mode": "fresh_process_strict_load", "environment": g7a_worker.environment_audit(),
        "checkpoint_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "strict_model_load": True, "strict_optimizer_load": True, "strict_scheduler_load": True,
        "samplers_loaded": sorted(samplers), "rng_restored_last": True,
        "critic_optimizer_step": 272, "actor_optimizer_step": 8,
        "actor_sparse_parameter_step_values": sorted(actor_steps),
        "critic_scheduler_step": 272, "actor_scheduler_step": 8,
        "parameter_updates": 0, "sampler_draws_after_load": 0,
        "validation_transition_reads": 0, "test_transition_reads": 0,
        "manual_g1_opens": 0, "manual_label_opens": 0, "reward_classifier_calls": 0,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "verify"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--protected", type=Path)
    args = parser.parse_args()
    require(not args.result.exists(), "G7B_RESULT_APPEND_ONLY")
    if args.mode == "train":
        require(args.protected is not None and args.protected.is_file(), "G7B_PROTECTED_SNAPSHOT_REQUIRED")
        train(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
