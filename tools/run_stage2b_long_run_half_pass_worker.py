#!/usr/bin/env python3
"""Fresh-process segments and strict-load audit for the Stage-2B half pass."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(ROOT / "tools"))
CONFIG = ROOT / "configs/stage2b_long_run_half_pass.development.yaml"
SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v21_stage2b_long_run_half_pass.json"
PARENT = ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint"
FIXED = ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup/fixed_diagnostics.pt"
OUTPUT = ROOT / "artifacts/development/stage2/stage2b_long_run_half_pass"
CHECKPOINT_ROOT = ROOT / "artifacts/development/stage2/stage2b_long_run_half_pass_checkpoints"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_progress(value: dict) -> None:
    path = OUTPUT / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def load_config() -> tuple[dict, dict]:
    from forcesmolvla.rft.source_manifest import validate_stage2_source_manifest

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source = validate_stage2_source_manifest(ROOT, SOURCE)
    require(source["scope"] == "Stage2B_frozen_VLM_half_actor_pass", "STAGE2B_SOURCE_SCOPE")
    for item in config["contracts"].values():
        require(sha(ROOT / item["path"]) == item["sha256"], f"STAGE2B_CONTRACT_SHA:{item['path']}")
    require(sha(PARENT / "checkpoint_manifest.json") == config["parent"]["checkpoint_manifest_sha256"], "STAGE2B_PARENT_MANIFEST_SHA")
    recipe = config["recipe"]
    require(
        recipe["joint_cycles"] == 210
        and recipe["expected_critic_updates"] == 420
        and recipe["expected_actor_updates"] == 210
        and config["batching"]["actor_physical_batch_size"] == 24
        and config["batching"]["critic_physical_batch_size"] == 128
        and config["loss"]["eta_actor_q"] == 3.0
        and config["loss"]["beta_flow"] == 1.0
        and config["targets"]["polyak_tau"] == 0.005,
        "STAGE2B_RECIPE_DRIFT",
    )
    training = yaml.safe_load((ROOT / "configs/stage2_g5_single_cycle.v2.development.yaml").read_text())
    training = copy.deepcopy(training)
    training["batching"].update({
        "critic_batch_size": 128,
        "calql_batch_size": 128,
        "actor_microbatch_size": 24,
        "actor_gradient_accumulation": 1,
        "actor_effective_batch_size": 24,
    })
    training["loss"].update({
        "beta_flow": 1.0,
        "eta_actor_q": 3.0,
        "alpha_calql": 0.1,
    })
    training["targets"]["polyak_tau"] = 0.005
    return config, training


def build_context(device: torch.device, *, with_data: bool):
    import run_s2_g7a_r2_worker  # install the frozen ActionContract-v2 adapter
    import run_s2_g7a_worker as g7a
    import run_s2_g7b_worker as g7b
    from forcesmolvla.rft.frozen_vlm_trainability import (
        apply_frozen_vlm_trainability,
        build_frozen_vlm_actor_optimizer,
    )

    context = g7a.initialize_fresh(device=device, with_data=with_data)
    parent_sampler_states, parent_rng = g7b.load_parent(context)
    trainability = apply_frozen_vlm_trainability(context["actor"])
    actor_optimizer, actor_scheduler, actor_ownership = build_frozen_vlm_actor_optimizer(
        context["actor"], lr=1e-5
    )
    return (
        context, parent_sampler_states, parent_rng,
        actor_optimizer, actor_scheduler, actor_ownership, trainability,
        run_s2_g7a_r2_worker,
    )


def sampler_objects(states: dict, generators: dict[str, torch.Generator]):
    from forcesmolvla.rft.training_cycle import SerializableReplacementSampler, SerializableUniqueSampler

    return {
        "td": SerializableUniqueSampler(
            states["td"]["name"], tuple(states["td"]["population"]),
            generators["td_sampler"], int(states["td"]["draws"]),
        ),
        "calql": SerializableUniqueSampler(
            states["calql"]["name"], tuple(states["calql"]["population"]),
            generators["calql_sampler"], int(states["calql"]["draws"]),
        ),
        "actor": SerializableUniqueSampler(
            states["actor"]["name"], tuple(states["actor"]["population"]),
            generators["actor_sampler"], int(states["actor"]["draws"]),
        ),
        "empirical_random_proposal": SerializableReplacementSampler(
            states["empirical_random_proposal"]["name"],
            int(states["empirical_random_proposal"]["population_size"]),
            generators["empirical_random_proposal"],
            int(states["empirical_random_proposal"]["draws"]),
        ),
    }


def load_checkpoint(
    checkpoint: Path, *, expected_cycle: int, context: dict,
    actor_optimizer, actor_scheduler, training: dict,
) -> tuple[dict, dict, dict]:
    import run_s2_g7b_worker as g7b
    from forcesmolvla.rft.canonical_state import canonical_digest
    from forcesmolvla.rft.g7_long_run import validate_cycle_checkpoint
    from forcesmolvla.rft.training_cycle import ensure_all_gradients_none

    manifest = validate_cycle_checkpoint(checkpoint, expected_cycle=expected_cycle)
    modules = {name: context[name] for name in ("actor", "q1", "q2", "q1_target", "q2_target")}
    for name, module in modules.items():
        incompatible = module.load_state_dict(torch.load(
            checkpoint / f"models/{name}_state.pt", map_location="cpu", weights_only=False
        ), strict=True)
        require(not incompatible.missing_keys and not incompatible.unexpected_keys, f"STAGE2B_RESUME_MODEL:{name}")
    actor_optimizer.load_state_dict(torch.load(
        checkpoint / "optimizers/actor_optimizer_state.pt", map_location="cpu", weights_only=False
    ))
    context["optimizer"].load_state_dict(torch.load(
        checkpoint / "optimizers/critic_optimizer_state.pt", map_location="cpu", weights_only=False
    ))
    actor_scheduler.load_state_dict(torch.load(
        checkpoint / "schedulers/actor_scheduler_state.pt", map_location="cpu", weights_only=False
    ))
    context["scheduler"].load_state_dict(torch.load(
        checkpoint / "schedulers/critic_scheduler_state.pt", map_location="cpu", weights_only=False
    ))
    states = torch.load(checkpoint / "state/sampler_states.pt", map_location="cpu", weights_only=False)
    rng = torch.load(checkpoint / "state/rng_states.pt", map_location="cpu", weights_only=False)
    generators = g7b.build_generators(training)
    samplers = sampler_objects(states, generators)
    g7b.restore_parent_rng(rng, generators)  # RNG restoration must be last.
    for target in (context["q1_target"], context["q2_target"]):
        target.make_permanent_eval_target()
    ensure_all_gradients_none(*modules.values())
    require(context["scheduler"].last_epoch == 256 + 2 * expected_cycle, "STAGE2B_RESUME_CRITIC_SCHEDULER")
    require(actor_scheduler.last_epoch == expected_cycle, "STAGE2B_RESUME_ACTOR_SCHEDULER")
    return samplers, generators, {
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "rng_digest_after_restore": canonical_digest(rng),
        "sampler_draws": {name: sampler.draws for name, sampler in samplers.items()},
        "parameter_updates_before_resume": 0,
        "sampler_draws_after_restore": 0,
        "training_rng_consumption_after_restore": 0,
    }


def checkpoint_boundary(
    *, cycle: int, context: dict, actor_optimizer, actor_scheduler,
    samplers: dict, generators: dict, ownership: dict,
    protected: dict, startup: dict[str, bytes], g5,
) -> dict:
    from forcesmolvla.rft.canonical_state import canonical_digest
    from forcesmolvla.rft.g7_long_run import hardlink_milestone, save_cycle_checkpoint

    modules = {name: context[name] for name in ("actor", "q1", "q2", "q1_target", "q2_target")}
    rng = g5.capture_rng_states(generators)
    rng_digest = canonical_digest(rng)
    rolling = CHECKPOINT_ROOT / "recovery_latest"
    started = time.perf_counter()
    manifest = save_cycle_checkpoint(
        rolling, cycle=cycle, modules=modules,
        actor_optimizer=actor_optimizer, critic_optimizer=context["optimizer"],
        actor_scheduler=actor_scheduler, critic_scheduler=context["scheduler"],
        sampler_states={name: sampler.state_dict() for name, sampler in samplers.items()},
        rng_states=rng, ownership_manifest=ownership,
        protected_snapshot=protected, startup_snapshot_bytes=startup,
        replace_rolling=True,
    )
    require(canonical_digest(g5.capture_rng_states(generators)) == rng_digest, "STAGE2B_CHECKPOINT_CONSUMED_RNG")
    milestone = CHECKPOINT_ROOT / f"milestone_cycle_{cycle:06d}"
    hardlink_milestone(rolling, milestone, expected_cycle=cycle)
    return {
        "cycle": cycle,
        "recovery_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "milestone_path": milestone.relative_to(ROOT).as_posix(),
        "latency_seconds": time.perf_counter() - started,
    }


def _gradient_group(name: str) -> str:
    if name.startswith("model.force_branch.force_mlp."):
        return "ForceMLP"
    if name.startswith("model.force_branch.refiner.router."):
        return "router"
    if name.startswith("model.force_adapter."):
        return "ForceActionAdapter"
    if name.startswith("model.force_branch."):
        return "Fusion_MoE"
    if name.startswith("model.vlm_with_expert.lm_expert."):
        return "Action_Expert"
    if name.startswith((
        "model.action_in_proj.", "model.action_out_proj.",
        "model.action_time_mlp_in.", "model.action_time_mlp_out.",
    )):
        return "Action_IO"
    return "other_approved_action_path"


def _gradient_metrics(fm: dict[str, torch.Tensor], policy) -> dict:
    totals = {"global": {"fm2": 0.0, "q2": 0.0, "dot": 0.0}}
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        destinations = (totals["global"], totals.setdefault(
            _gradient_group(name), {"fm2": 0.0, "q2": 0.0, "dot": 0.0}
        ))
        fm_value = fm.get(name)
        q_value = parameter.grad.detach().float().cpu() if parameter.grad is not None else None
        fm2 = float(fm_value.square().sum()) if fm_value is not None else 0.0
        q2 = float(q_value.square().sum()) if q_value is not None else 0.0
        dot = float((fm_value * q_value).sum()) if fm_value is not None and q_value is not None else 0.0
        for destination in destinations:
            destination["fm2"] += fm2; destination["q2"] += q2; destination["dot"] += dot
    result = {}
    for name, values in totals.items():
        fm_norm, q_norm = math.sqrt(values["fm2"]), math.sqrt(values["q2"])
        result[name] = {
            "fm_norm": fm_norm,
            "q_norm": q_norm,
            "raw_q_over_fm": q_norm / max(fm_norm, torch.finfo(torch.float32).tiny),
            "weighted_eta3_q_over_beta1_fm": 3.0 * q_norm / max(fm_norm, torch.finfo(torch.float32).tiny),
            "cosine_similarity": values["dot"] / (fm_norm * q_norm) if fm_norm and q_norm else 0.0,
        }
    return result


def gradient_scale_diagnostic(
    *, cycle: int, context: dict, indices: list[int], device,
    generators: dict, g5,
) -> dict:
    import benchmark_stage2_batch_scaling_gpu as benchmark
    from forcesmolvla.rft.canonical_state import canonical_digest
    from forcesmolvla.rft.frozen_vlm_trainability import (
        compute_min_twin_q_actor_loss,
        frozen_prefix_flow_matching_terms,
    )
    from forcesmolvla.rft.training_cycle import module_state_sha256

    policy, q1, q2 = context["actor"], context["q1"], context["q2"]
    state_before = module_state_sha256(policy)
    rng_before = canonical_digest(g5.capture_rng_states(generators))
    batch = context["data"].build_batch(
        indices, policy, device, canonical_task_feature=q1.canonical_task_feature,
        include_flow_actions=True,
    )
    trainable = [(name, value) for name, value in policy.named_parameters() if value.requires_grad]
    frozen = [value for value in policy.parameters() if not value.requires_grad]
    for _name, value in trainable:
        value.grad = None
    fm_noise = torch.randn(24, 50, 7, generator=torch.Generator(device=device).manual_seed(19221), device=device)
    fm_time = torch.rand(24, generator=torch.Generator(device=device).manual_seed(19222), device=device)
    outputs = []
    hook = policy.model.action_out_proj.register_forward_hook(
        lambda _m, _i, output: (output.retain_grad(), outputs.append(output))[-1]
    )
    try:
        policy.train(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses, feature_mask, _router, prefix = frozen_prefix_flow_matching_terms(
                policy, batch["current_actor_batch"], noise=fm_noise, time=fm_time,
                call_id=f"stage2b-gradient-cycle-{cycle}-fm",
            )
            fm_loss = losses.sum() / feature_mask.sum().clamp_min(1)
        fm_loss.backward()
    finally:
        hook.remove()
    require(len(outputs) == 1 and outputs[0].grad is not None, "STAGE2B_GRADIENT_FM_OUTPUT")
    gripper_fm = float(outputs[0].grad[..., 6].float().norm().cpu())
    fm_gradients = {
        name: value.grad.detach().float().cpu().clone()
        for name, value in trainable if value.grad is not None
    }
    for _name, value in trainable:
        value.grad = None
    q_noise = torch.randn(24, 50, 7, generator=torch.Generator(device=device).manual_seed(19223), device=device)
    flow = benchmark.TimedFlowCounter(inference_batch_size=24)
    policy.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        chunk = flow.sample(
            policy, batch["current_actor_batch"], q_noise,
            call_id=f"stage2b-gradient-cycle-{cycle}-q", purpose="actor_guidance",
        )
        chunk.retain_grad()
        q_loss, _q1, _q2, _critic_action = compute_min_twin_q_actor_loss(
            q1=q1, q2=q2, observation=batch["current_observation"],
            normalized_flow_action_chunk7=chunk,
            delta_action_mean7=batch["delta_mean"], delta_action_std7=batch["delta_std"],
        )
    q_loss.backward()
    require(chunk.grad is not None, "STAGE2B_GRADIENT_Q_ACTION")
    tcp6 = float(chunk.grad[:, :3, :6].float().norm().cpu())
    gripper_q = float(chunk.grad[:, :3, 6].float().abs().max().cpu())
    metrics = _gradient_metrics(fm_gradients, policy)
    require(
        metrics["global"]["fm_norm"] > 0.0
        and metrics["global"]["q_norm"] > 0.0
        and tcp6 > 0.0 and gripper_q == 0.0 and gripper_fm > 0.0
        and all(value.grad is None for value in frozen),
        "STAGE2B_GRADIENT_CONTRACT",
    )
    for _name, value in trainable:
        value.grad = None
    require(module_state_sha256(policy) == state_before, "STAGE2B_GRADIENT_DIAGNOSTIC_CHANGED_ACTOR")
    require(canonical_digest(g5.capture_rng_states(generators)) == rng_before, "STAGE2B_GRADIENT_DIAGNOSTIC_CONSUMED_TRAINING_RNG")
    del fm_gradients, batch, losses, fm_loss, q_loss, chunk
    gc.collect(); torch.cuda.empty_cache()
    return {
        "cycle": cycle,
        "actor_physical_batch_size": 24,
        "fixed_row_indices": indices,
        "metrics": metrics,
        "tcp6_q_gradient_norm": tcp6,
        "gripper_q_gradient_max_abs": gripper_q,
        "gripper_fm_gradient_norm": gripper_fm,
        "prefix_audit": prefix,
        "eta3_is_measured_at_this_cycle": True,
    }


def actor_update_eta3(
    *, cycle: int, context: dict, batch: dict, optimizer, scheduler,
    generators: dict[str, torch.Generator],
) -> dict:
    import benchmark_stage2_batch_scaling_gpu as benchmark
    from forcesmolvla.force_token import RouterState
    from forcesmolvla.rft.critic_action_adapter_v2 import raw_gripper_out_of_public_tolerance_mask
    from forcesmolvla.rft.frozen_vlm_trainability import (
        compute_min_twin_q_actor_loss,
        frozen_prefix_flow_matching_terms,
    )
    from forcesmolvla.rft.training_cycle import global_gradient_norm, gradients_finite
    from forcesmolvla.router_training import collect_pass_a_statistics, microbatch_two_pass_terms

    policy, q1, q2 = context["actor"], context["q1"], context["q2"]
    device = batch["reward"].device
    trainable = [value for value in policy.parameters() if value.requires_grad]
    frozen = [value for value in policy.parameters() if not value.requires_grad]
    optimizer.zero_grad(set_to_none=True); policy.train(True)
    noise = torch.randn(24, 50, 7, generator=generators["flow_matching_noise"], device=device)
    timestep = torch.rand(24, generator=generators["flow_matching_timestep"], device=device)
    outputs = []
    hook = policy.model.action_out_proj.register_forward_hook(
        lambda _m, _i, output: (output.retain_grad(), outputs.append(output))[-1]
    )
    torch.cuda.synchronize(); fm_started = time.perf_counter()
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses, feature_mask, router_state, prefix = frozen_prefix_flow_matching_terms(
                policy, batch["current_actor_batch"], noise=noise, time=timestep,
                call_id=f"stage2b-cycle-{cycle}-fm",
            )
            detached = RouterState(
                logits_fp32=router_state.logits_fp32.detach(),
                probabilities_fp32=router_state.probabilities_fp32.detach(),
                route_ids=router_state.route_ids.detach(), valid_mask=router_state.valid_mask.detach(),
            )
            statistics = collect_pass_a_statistics([detached], [feature_mask])
            auxiliary = microbatch_two_pass_terms(losses, router_state, statistics)
            fm_loss = losses.sum() / feature_mask.sum().clamp_min(1)
            fm_objective = fm_loss + 0.01 * auxiliary.balance + 0.001 * auxiliary.z
        fm_objective.backward()
    finally:
        hook.remove()
    torch.cuda.synchronize(); fm_seconds = time.perf_counter() - fm_started
    require(len(outputs) == 1 and outputs[0].grad is not None, "STAGE2B_ACTOR_FM_OUTPUT")
    gripper_fm = float(outputs[0].grad[..., 6].float().norm().cpu())
    fm_groups = benchmark.gradient_groups(policy)

    q_noise = torch.randn(24, 50, 7, generator=generators["actor_q_flow_noise"], device=device)
    flow = benchmark.TimedFlowCounter(inference_batch_size=24)
    policy.eval(); torch.cuda.synchronize(); q_started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        chunk = flow.sample(
            policy, batch["current_actor_batch"], q_noise,
            call_id=f"stage2b-cycle-{cycle}-q", purpose="actor_guidance",
        )
        chunk.retain_grad()
        q_loss, q1_value, q2_value, critic_action = compute_min_twin_q_actor_loss(
            q1=q1, q2=q2, observation=batch["current_observation"],
            normalized_flow_action_chunk7=chunk,
            delta_action_mean7=batch["delta_mean"], delta_action_std7=batch["delta_std"],
        )
        weighted_q = 3.0 * q_loss
    weighted_q.backward()
    torch.cuda.synchronize(); q_seconds = time.perf_counter() - q_started
    require(chunk.grad is not None, "STAGE2B_ACTOR_Q_ACTION")
    tcp6 = float(chunk.grad[:, :3, :6].float().norm().cpu())
    gripper_q = float(chunk.grad[:, :3, 6].float().abs().max().cpu())
    raw_rate = float(raw_gripper_out_of_public_tolerance_mask(
        chunk[:, :3, 6].detach(),
        gripper_mean=batch["delta_mean"][6], gripper_std=batch["delta_std"][6],
    ).float().mean().cpu())
    combined = benchmark.gradient_groups(policy)
    require(
        fm_groups["frozen_vlm"] == fm_groups["frozen_state_prefix"] == 0.0
        and combined["frozen_vlm"] == combined["frozen_state_prefix"] == 0.0
        and all(combined[name] > 0.0 for name in ("force", "action_expert", "action_io", "router"))
        and tcp6 > 0.0 and gripper_q == 0.0 and gripper_fm > 0.0
        and all(value.grad is None for value in frozen)
        and gradients_finite(trainable),
        "STAGE2B_ACTOR_GRADIENT_CONTRACT",
    )
    preclip = float(global_gradient_norm(trainable).cpu())
    torch.cuda.synchronize(); optimizer_started = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(trainable, 10.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); optimizer_seconds = time.perf_counter() - optimizer_started
    policy.eval()
    values = (
        float(fm_loss.detach()), float(q_loss.detach()),
        float((fm_objective + weighted_q).detach()), preclip, tcp6, gripper_fm,
        float(q1_value.mean().detach()), float(q2_value.mean().detach()),
    )
    require(all(math.isfinite(value) for value in values), "STAGE2B_ACTOR_NONFINITE")
    return {
        "loss": {
            "flow_matching": values[0], "actor_q_min_twin": values[1],
            "balance": float(auxiliary.balance.detach()), "z": float(auxiliary.z.detach()),
            "weighted_total": values[2], "beta": 1.0, "eta": 3.0,
        },
        "q": {"q1_mean": values[6], "q2_mean": values[7]},
        "gradient": {
            "tcp6_q_norm": tcp6, "gripper_q_max_abs": gripper_q,
            "gripper_fm_norm": gripper_fm, "preclip_global_norm": preclip,
            "fm_groups": fm_groups, "combined_groups": combined,
        },
        "timing": {
            "flow_matching_forward_backward": fm_seconds,
            "differentiable_n10_flow_twin_q_actor_q_backward": q_seconds,
            "actor_optimizer": optimizer_seconds,
        },
        "prefix_audit": prefix,
        "flow_counts": flow.report(),
        "critic_action_shape": list(critic_action.shape),
        "raw_gripper_out_of_public_tolerance_rate": raw_rate,
    }


def validation(*, cycle: int, context: dict, fixed: dict, device, generators: dict, g5) -> dict:
    import run_s2_g7a_worker as g7a
    from forcesmolvla.rft.canonical_state import canonical_digest
    from forcesmolvla.rft.training_cycle import module_state_sha256

    modules = {name: context[name] for name in ("actor", "q1", "q2", "q1_target", "q2_target")}
    before = {name: module_state_sha256(module) for name, module in modules.items()}
    rng_before = canonical_digest(g5.capture_rng_states(generators))
    rows = g7a.load_split_rows("val")
    g7a.attach_distance(rows)
    result = g7a.evaluate_critic_split(
        label=f"stage2b-cycle-{cycle}-validation", rows=rows,
        indices=fixed["validation_indices"], data=g7a.split_data(context["data"], rows),
        fixed=fixed["validation_evaluation"], policy=context["actor"],
        q1=context["q1"], q2=context["q2"],
        q1_target=context["q1_target"], q2_target=context["q2_target"],
        train_data=context["data"], device=device, batch_size=16,
    )
    after = {name: module_state_sha256(module) for name, module in modules.items()}
    require(before == after, "STAGE2B_VALIDATION_CHANGED_MODEL")
    require(canonical_digest(g5.capture_rng_states(generators)) == rng_before, "STAGE2B_VALIDATION_CONSUMED_TRAINING_RNG")
    result.update({"cycle": cycle, "read_only": True, "selection_or_early_stop": False})
    return result


def boundary_audit(
    *, cycle: int, context: dict, frozen_reference: dict, fixed_indices: list[int],
    fixed_noise: torch.Tensor, device, generators: dict, g5,
) -> dict:
    import benchmark_stage2_batch_scaling_gpu as benchmark
    import run_s2_g7b_worker as g7b
    from forcesmolvla.rft.canonical_state import canonical_digest
    from forcesmolvla.rft.frozen_vlm_trainability import frozen_state_digest

    frozen_now = frozen_state_digest(context["actor"])
    require(frozen_now == frozen_reference, "STAGE2B_FROZEN_HASH_CHANGED")
    mask = benchmark.partial_mask_audit(
        context["data"], context["actor"], context["q1"], context["q2"], device
    )
    require(mask["invalid_slot_perturbation_exact_invariant"], "STAGE2B_INVALID_SLOT_MASK")
    batch = context["data"].build_batch(
        fixed_indices[:1], context["actor"], device,
        canonical_task_feature=context["q1"].canonical_task_feature,
        include_flow_actions=True,
    )
    action = g7b.internal_action_diagnostic(
        context["actor"], batch["current_actor_batch"], fixed_noise,
        batch["delta_mean"], batch["delta_std"],
    )
    rng_before = canonical_digest(g5.capture_rng_states(generators))
    public = benchmark.public_audit(context, batch, fixed_noise, cycle)
    require(public["semantic_success"], f"STAGE2B_PUBLIC_PREDICT:{public}")
    require(canonical_digest(g5.capture_rng_states(generators)) == rng_before, "STAGE2B_PUBLIC_CONSUMED_TRAINING_RNG")
    return {
        "cycle": cycle, "frozen_state": frozen_now,
        "frozen_parameter_buffer_hash_unchanged": True,
        "mask": mask, "internal_action": action, "public_predict": public,
        "frozen_modules_always_eval": True,
    }


def compact_critic(report: dict) -> dict:
    identities = report.pop("row_identities")
    return {
        **report,
        "row_identity_audit": {
            "td_count": len(identities["td"]),
            "calql_count": len(identities["calql"]),
            "td_sha256": canonical(identities["td"]),
            "calql_sha256": canonical(identities["calql"]),
        },
    }


def train_segment(args) -> None:
    import benchmark_stage2_batch_scaling_gpu as benchmark
    import preflight_s2_g5_single_cycle_gpu as g5
    import run_s2_g7a_worker as g7a
    import run_s2_g7b_worker as g7b
    from forcesmolvla.rft.critic import modules_storage_independent
    from forcesmolvla.rft.frozen_vlm_trainability import frozen_state_digest
    from forcesmolvla.rft.training_cycle import (
        ensure_all_gradients_none, module_state_sha256, optimizer_state_storage_independent,
    )

    require((args.start_cycle, args.end_cycle) in {(0, 105), (105, 210)}, "STAGE2B_SEGMENT_RANGE")
    g5.install_open_audit()
    device = g7a.configure_runtime()
    config, training = load_config()
    (
        context, parent_sampler_states, parent_rng,
        actor_optimizer, actor_scheduler, actor_ownership, trainability, r2,
    ) = build_context(device, with_data=True)
    data = context["data"]
    resume = None
    if args.start_cycle == 0:
        generators = g7b.build_generators(training)
        samplers = g7b.build_samplers(data, generators, parent_sampler_states)
        g7b.restore_parent_rng(parent_rng, generators)
    else:
        samplers, generators, resume = load_checkpoint(
            CHECKPOINT_ROOT / "recovery_latest", expected_cycle=105,
            context=context, actor_optimizer=actor_optimizer,
            actor_scheduler=actor_scheduler, training=training,
        )
    policy, q1, q2 = context["actor"], context["q1"], context["q2"]
    modules = {name: context[name] for name in ("actor", "q1", "q2", "q1_target", "q2_target")}
    frozen_reference = frozen_state_digest(policy)
    backbones = {
        f"{name}.{camera}": module_state_sha256(getattr(module, camera))
        for name, module in (("q1", q1), ("q2", q2))
        for camera in ("camera1_backbone", "camera2_backbone")
    }
    ownership = {
        "actor": actor_ownership, "critic": context["ownership"],
        "actor_critic_parameter_intersection": len(
            {id(value) for value in policy.parameters() if value.requires_grad}
            & {id(value) for module in (q1, q2) for value in module.parameters() if value.requires_grad}
        ),
        "frozen_actor_parameter_in_actor_optimizer": 0,
        "target_in_optimizer": 0,
        "target_actor": None,
    }
    require(ownership["actor_critic_parameter_intersection"] == 0, "STAGE2B_OPTIMIZER_OWNERSHIP")
    fixed = torch.load(FIXED, map_location=device, weights_only=False)
    require(sha(FIXED) == "002235cfc18cf939652c7a1bbe27ca0e752cf2e25e89fa415465ccfb3e8777e2", "STAGE2B_FIXED_SHA")
    fixed_indices = list(data.actor_population[:24])
    fixed_noise = torch.randn(1, 50, 7, generator=torch.Generator(device=device).manual_seed(19224), device=device)
    protected = json.loads(args.protected.read_text())
    startup = {
        "config/stage2b_long_run_half_pass.development.yaml": CONFIG.read_bytes(),
        "source/stage2_source_manifest.v21_stage2b_long_run_half_pass.json": SOURCE.read_bytes(),
        "parent/checkpoint_manifest.json": (PARENT / "checkpoint_manifest.json").read_bytes(),
        "contracts/stage2_trainability_contract.v2.development.json": (ROOT / config["contracts"]["trainability"]["path"]).read_bytes(),
        "contracts/stage2_action_contract.v2.development.json": (ROOT / config["contracts"]["action"]["path"]).read_bytes(),
        "bindings/protected_before.json": args.protected.read_bytes(),
    }
    initial = {name: module_state_sha256(module) for name, module in modules.items()}
    boundary_records = []
    validation_records = []
    gradient_records = []
    checkpoint_records = []
    if args.start_cycle == 0:
        boundary_records.append(boundary_audit(
            cycle=0, context=context, frozen_reference=frozen_reference,
            fixed_indices=fixed_indices, fixed_noise=fixed_noise,
            device=device, generators=generators, g5=g5,
        ))
        gradient_records.append(gradient_scale_diagnostic(
            cycle=0, context=context, indices=fixed_indices, device=device,
            generators=generators, g5=g5,
        ))
        validation_records.append(validation(
            cycle=0, context=context, fixed=fixed, device=device,
            generators=generators, g5=g5,
        ))
        checkpoint_records.append(checkpoint_boundary(
            cycle=0, context=context, actor_optimizer=actor_optimizer,
            actor_scheduler=actor_scheduler, samplers=samplers, generators=generators,
            ownership=ownership, protected=protected, startup=startup, g5=g5,
        ))
        append_progress({"cycle": 0, "status": "complete_boundary", "validation": True, "checkpoint": True})

    telemetry = benchmark.GpuTelemetry().__enter__()
    torch.cuda.reset_peak_memory_stats(device)
    cycles = []
    training_started = time.perf_counter()
    try:
        for cycle in range(args.start_cycle + 1, args.end_cycle + 1):
            torch.cuda.synchronize(); cycle_started = time.perf_counter()
            actor_before_critics = module_state_sha256(policy)
            critics = []
            with g7b.critic_internal_only():
                for local in range(2):
                    report = benchmark.critic_update(
                        context=context, training=training, generators=generators,
                        samplers=samplers, batch_size=128,
                        update_id=256 + 2 * (cycle - 1) + local + 1,
                    )
                    critics.append(compact_critic(report))
                    gc.collect(); torch.cuda.empty_cache()
                require(module_state_sha256(policy) == actor_before_critics, "STAGE2B_ACTOR_CHANGED_DURING_CRITIC")
                actor_indices = samplers["actor"].draw(24)
                load_started = time.perf_counter()
                actor_batch = data.build_batch(
                    actor_indices, policy, device,
                    canonical_task_feature=q1.canonical_task_feature,
                    include_flow_actions=True,
                )
                data_seconds = time.perf_counter() - load_started
                actor = actor_update_eta3(
                    cycle=cycle, context=context, batch=actor_batch,
                    optimizer=actor_optimizer, scheduler=actor_scheduler,
                    generators=generators,
                )
                del actor_batch
            torch.cuda.synchronize(); elapsed = time.perf_counter() - cycle_started
            require(all(math.isfinite(float(value)) for item in critics for value in item["loss"].values()), "STAGE2B_CRITIC_NONFINITE")
            require(all(math.isfinite(float(value)) for value in actor["loss"].values()), "STAGE2B_ACTOR_NONFINITE")
            require(
                actor["gradient"]["tcp6_q_norm"] > 0.0
                and actor["gradient"]["gripper_q_max_abs"] == 0.0
                and actor["gradient"]["gripper_fm_norm"] > 0.0
                and actor["prefix_audit"]["force_kv_projection_count"] == 1,
                "STAGE2B_RUNTIME_ACTION_CONTRACT",
            )
            cycle_record = {
                "cycle": cycle, "critic_updates": critics, "actor_update": actor,
                "actor_batch_identity_sha256": canonical(data.identity_records(actor_indices)),
                "actor_batch_count": len(actor_indices),
                "actor_data_loading_seconds": data_seconds,
                "cycle_seconds": elapsed,
                "cumulative_exposure": {
                    "actor_transitions": 24 * cycle,
                    "critic_td_transitions": 256 * cycle,
                    "critic_calql_transitions": 256 * cycle,
                },
            }
            cycles.append(cycle_record)
            if cycle != args.end_cycle:
                append_progress({
                    "cycle": cycle, "status": "complete_boundary",
                    "critic_loss": [item["loss"]["L_critic"] for item in critics],
                    "fm_loss": actor["loss"]["flow_matching"],
                    "actor_q_loss": actor["loss"]["actor_q_min_twin"],
                    "actor_total_loss": actor["loss"]["weighted_total"],
                    "cycle_seconds": elapsed,
                    "validation": False, "checkpoint": False,
                })
            print(f"STAGE2B_HALF_PASS_CYCLE {cycle}/210", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    finally:
        telemetry.__exit__(None, None, None)
    training_seconds = time.perf_counter() - training_started

    boundary_records.append(boundary_audit(
        cycle=args.end_cycle, context=context, frozen_reference=frozen_reference,
        fixed_indices=fixed_indices, fixed_noise=fixed_noise,
        device=device, generators=generators, g5=g5,
    ))
    gradient_records.append(gradient_scale_diagnostic(
        cycle=args.end_cycle, context=context, indices=fixed_indices, device=device,
        generators=generators, g5=g5,
    ))
    validation_records.append(validation(
        cycle=args.end_cycle, context=context, fixed=fixed, device=device,
        generators=generators, g5=g5,
    ))
    checkpoint_records.append(checkpoint_boundary(
        cycle=args.end_cycle, context=context, actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler, samplers=samplers, generators=generators,
        ownership=ownership, protected=protected, startup=startup, g5=g5,
    ))
    final_cycle = cycles[-1]
    append_progress({
        "cycle": args.end_cycle, "status": "complete_boundary",
        "critic_loss": [item["loss"]["L_critic"] for item in final_cycle["critic_updates"]],
        "fm_loss": final_cycle["actor_update"]["loss"]["flow_matching"],
        "actor_q_loss": final_cycle["actor_update"]["loss"]["actor_q_min_twin"],
        "actor_total_loss": final_cycle["actor_update"]["loss"]["weighted_total"],
        "cycle_seconds": final_cycle["cycle_seconds"],
        "validation": True, "checkpoint": True,
    })
    ensure_all_gradients_none(*modules.values())
    final = {name: module_state_sha256(module) for name, module in modules.items()}
    require(all(final[name] != initial[name] for name in modules), "STAGE2B_EXPECTED_UPDATE_MISSING")
    require(frozen_state_digest(policy) == frozen_reference, "STAGE2B_FINAL_FROZEN_HASH")
    require(backbones == {
        f"{name}.{camera}": module_state_sha256(getattr(module, camera))
        for name, module in (("q1", q1), ("q2", q2))
        for camera in ("camera1_backbone", "camera2_backbone")
    }, "STAGE2B_CRITIC_BACKBONE_CHANGED")
    require(modules_storage_independent(q1, q2) and optimizer_state_storage_independent(context["optimizer"], q1, q2), "STAGE2B_CRITIC_STORAGE")
    require(all(bool(torch.isfinite(parameter).all()) for module in modules.values() for parameter in module.parameters()), "STAGE2B_PARAMETER_NONFINITE")
    result = {
        "mode": "train_segment", "pid": os.getpid(),
        "start_cycle": args.start_cycle, "end_cycle": args.end_cycle,
        "environment": g7a.environment_audit(), "resume_audit": resume,
        "cycles": cycles, "boundary_audits": boundary_records,
        "gradient_scale_diagnostics": gradient_records,
        "validation_diagnostics": validation_records,
        "checkpoint_events": checkpoint_records,
        "parameter_change_matrix": {
            name: {"before": initial[name], "after": final[name], "changed": initial[name] != final[name]}
            for name in modules
        },
        "runtime": {
            "training_body_seconds": training_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "gpu_utilization_percent": benchmark.describe(telemetry.utilization or [0.0]),
            "gpu_power_watts": benchmark.describe(telemetry.power or [0.0]),
        },
        "sampler_draws": {name: sampler.draws for name, sampler in samplers.items()},
        "data_access": {
            **data.population_audit(),
            "validation_transition_reads": 1205 * (2 if args.start_cycle == 0 else 1),
            "test_transition_reads": 0,
            "manual_g1_opens": len(g5.FORBIDDEN_OPENS["manual_g1"]),
            "manual_label_opens": len(g5.FORBIDDEN_OPENS["manual_labels"]),
            "reward_classifier_inference": 0, "reward_classifier_updates": 0,
        },
        "trainability": {
            "frozen_parameter_count": trainability.frozen_parameter_count,
            "trainable_actor_parameter_count": trainability.trainable_parameter_count,
            "frozen_hash_unchanged": True,
        },
    }
    atomic_json(args.result, result)


def verify(args) -> None:
    import run_s2_g7a_worker as g7a
    from forcesmolvla.rft.training_cycle import ensure_all_gradients_none

    device = g7a.configure_runtime()
    _config, training = load_config()
    context, _ps, _pr, actor_optimizer, actor_scheduler, _ownership, _trainability, _r2 = build_context(
        device, with_data=False
    )
    samplers, generators, audit = load_checkpoint(
        CHECKPOINT_ROOT / "milestone_cycle_000210", expected_cycle=210,
        context=context, actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler, training=training,
    )
    modules = {name: context[name] for name in ("actor", "q1", "q2", "q1_target", "q2_target")}
    ensure_all_gradients_none(*modules.values())
    critic_steps = {int(value["step"].item()) for value in context["optimizer"].state.values() if "step" in value}
    actor_steps = {int(value["step"].item()) for value in actor_optimizer.state.values() if "step" in value}
    require(critic_steps == {676}, "STAGE2B_VERIFY_CRITIC_STEP")
    require(actor_steps and max(actor_steps) == 210 and min(actor_steps) >= 1, "STAGE2B_VERIFY_ACTOR_STEP")
    require(context["scheduler"].last_epoch == 676 and actor_scheduler.last_epoch == 210, "STAGE2B_VERIFY_SCHEDULER")
    require(all(bool(torch.isfinite(value).all()) for module in modules.values() for value in module.parameters()), "STAGE2B_VERIFY_NONFINITE")
    atomic_json(args.result, {
        "mode": "fresh_process_strict_load", "pid": os.getpid(),
        "environment": g7a.environment_audit(), "strict_model_load": True,
        "strict_optimizer_load": True, "strict_scheduler_load": True,
        "strict_sampler_load": True, "rng_restored_last": True,
        "checkpoint_cycle": 210, "stage_critic_updates": 420,
        "total_critic_optimizer_step": 676, "actor_optimizer_step": 210,
        "samplers": sorted(samplers), "named_generators": sorted(generators),
        "resume_audit": audit, "parameter_updates": 0,
        "sampler_draws_after_load": 0, "validation_reads": 0,
        "test_reads": 0, "manual_g1_opens": 0, "manual_label_opens": 0,
        "reward_classifier_inference": 0,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("segment", "verify"), required=True)
    parser.add_argument("--start-cycle", type=int, default=0)
    parser.add_argument("--end-cycle", type=int, default=0)
    parser.add_argument("--protected", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    require(not args.result.exists(), "STAGE2B_RESULT_APPEND_ONLY")
    if args.mode == "segment":
        require(args.protected and args.protected.is_file(), "STAGE2B_PROTECTED_REQUIRED")
        train_segment(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
