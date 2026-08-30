#!/usr/bin/env python3
"""G7 development long-run stage-1 trainer and strict-load verifier."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
CONFIG = ROOT / "configs/stage2_g7_long_run_stage1.development.yaml"
SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v18_g7_long_run.json"
CHECKPOINT_ROOT = ROOT / "artifacts/development/stage2/g7_long_run_stage1_checkpoints"
OUTPUT = ROOT / "artifacts/development/stage2/g7_long_run_stage1"
FIXED = ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup/fixed_diagnostics.pt"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_progress(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> tuple[dict, dict]:
    import run_s2_g7b_worker as g7b
    from forcesmolvla.rft.source_manifest import validate_stage2_source_manifest

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source = validate_stage2_source_manifest(ROOT, SOURCE)
    require(source["scope"] == "G7_development_long_run_stage1_ActionContract_v2", "G7_LONG_SOURCE_SCOPE")
    training = g7b.load_config()[1]
    recipe = config["recipe"]
    require(recipe == {
        "joint_cycles": 256, "critic_updates_per_cycle": 2, "actor_updates_per_cycle": 1,
        "expected_critic_updates": 512, "expected_actor_updates": 256,
        "expected_polyak_updates_per_target": 512, "eta_q": 10.0,
        "eta_status": "development_only", "beta_flow": 1.0,
        "frozen_cycle_config": "configs/stage2_g5_single_cycle.v2.development.yaml",
        "frozen_cycle_config_sha256": "a728c4544c11f3ff15ba2b3b7ceca9cea7a068169ddc3913fa5707127f0f0fd0",
    }, "G7_LONG_RECIPE_DRIFT")
    require(training["loss"]["eta_actor_q"] == 10.0 and training["loss"]["beta_flow"] == 1.0, "G7_LONG_LOSS_DRIFT")
    require(config["parent"]["g7b_smoke_checkpoint_used_as_parent"] is False, "G7_LONG_G7B_PARENT_FORBIDDEN")
    return config, training


def tensor_stats(value: torch.Tensor) -> dict:
    value = value.detach().float()
    require(bool(torch.isfinite(value).all()), "G7_LONG_Q_STAT_NONFINITE")
    return {
        "mean": float(value.mean().cpu()),
        "variance": float(value.var(unbiased=False).cpu()),
        "minimum": float(value.min().cpu()),
        "maximum": float(value.max().cpu()),
    }


def q_statistics(batch: dict, q1, q2, q1_target, q2_target) -> dict:
    from forcesmolvla.rft.losses import compute_behavior_q

    modes = {module: module.training for module in (q1, q2)}
    q1.eval(); q2.eval(); q1_target.eval(); q2_target.eval()
    with torch.no_grad():
        args = (batch["current_observation"], batch["behavior_action"], batch["behavior_mask"])
        values = {
            "q1_online": compute_behavior_q(q1, *args),
            "q2_online": compute_behavior_q(q2, *args),
            "q1_target_on_behavior": compute_behavior_q(q1_target, *args),
            "q2_target_on_behavior": compute_behavior_q(q2_target, *args),
        }
    q1.train(modes[q1]); q2.train(modes[q2])
    return {name: tensor_stats(value) for name, value in values.items()}


def startup_snapshot(protected_path: Path) -> dict[str, bytes]:
    paths = {
        "config/stage2_g7_long_run_stage1.development.yaml": CONFIG,
        "source/stage2_source_manifest.v18_g7_long_run.json": SOURCE,
        "parent/checkpoint_manifest.json": ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint/checkpoint_manifest.json",
        "action_contract/stage2_action_contract.v2.development.json": ROOT / "configs/stage2_action_contract.v2.development.json",
    }
    result = {name: path.read_bytes() for name, path in paths.items()}
    result["bindings/protected_before.json"] = protected_path.read_bytes()
    return result


def save_boundary(
    *, cycle: int, modules: dict, actor_optimizer, critic_optimizer,
    actor_scheduler, critic_scheduler, samplers, generators, ownership,
    protected: dict, startup: dict, g5,
) -> dict:
    from forcesmolvla.rft.canonical_state import canonical_digest
    from forcesmolvla.rft.long_run_checkpoint import hardlink_milestone, save_cycle_checkpoint, validate_cycle_checkpoint

    rng = g5.capture_rng_states(generators)
    digest = canonical_digest(rng)
    rolling = CHECKPOINT_ROOT / "recovery_latest"
    started = time.perf_counter()
    manifest = save_cycle_checkpoint(
        rolling, cycle=cycle, modules=modules, actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer, actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        sampler_states={name: sampler.state_dict() for name, sampler in samplers.items()},
        rng_states=rng, ownership_manifest=ownership, protected_snapshot=protected,
        startup_snapshot_bytes=startup, replace_rolling=True,
    )
    require(canonical_digest(g5.capture_rng_states(generators)) == digest, "G7_LONG_CHECKPOINT_CONSUMED_RNG")
    validate_cycle_checkpoint(rolling, expected_cycle=cycle)
    milestone = None
    if cycle in {0, 64, 128, 256}:
        milestone = CHECKPOINT_ROOT / f"milestone_cycle_{cycle:06d}"
        hardlink_milestone(rolling, milestone, expected_cycle=cycle)
    return {
        "cycle": cycle, "rolling_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "milestone_path": None if milestone is None else milestone.relative_to(ROOT).as_posix(),
        "latency_seconds": time.perf_counter() - started,
    }


def validation_diagnostic(*, cycle: int, context: dict, validation_rows: list[dict], validation_data, fixed: dict, device, generators, g5, g7a_worker) -> dict:
    from forcesmolvla.rft.canonical_state import canonical_digest
    from forcesmolvla.rft.training_cycle import module_state_sha256

    modules = {name: context[name] for name in ("actor", "q1", "q2", "q1_target", "q2_target")}
    before = {name: module_state_sha256(module) for name, module in modules.items()}
    rng_before = canonical_digest(g5.capture_rng_states(generators))
    result = g7a_worker.evaluate_critic_split(
        label=f"g7-long-cycle-{cycle}-validation", rows=validation_rows,
        indices=fixed["validation_indices"], data=validation_data,
        fixed=fixed["validation_evaluation"], policy=context["actor"],
        q1=context["q1"], q2=context["q2"], q1_target=context["q1_target"],
        q2_target=context["q2_target"], train_data=context["data"],
        device=device, batch_size=16,
    )
    after = {name: module_state_sha256(module) for name, module in modules.items()}
    require(before == after, "G7_LONG_VALIDATION_CHANGED_PARAMETERS")
    require(canonical_digest(g5.capture_rng_states(generators)) == rng_before, "G7_LONG_VALIDATION_CONSUMED_TRAINING_RNG")
    result["cycle"] = cycle
    result["gradient_enabled"] = False
    result["selection_or_early_stop"] = False
    return result


def train(args) -> None:
    from forcesmolvla.rft import training_cycle as g5
    from forcesmolvla.rft import critic_training as g7a_worker
    import run_s2_g7b_worker as g7b
    from forcesmolvla.rft.canonical_state import canonical_digest
    from forcesmolvla.rft.critic import modules_storage_independent
    from forcesmolvla.rft.long_run_checkpoint import counters_for_cycle
    from forcesmolvla.rft.joint_training_checkpoint import describe_p95
    from forcesmolvla.rft.training_cycle import ensure_all_gradients_none, module_state_sha256, optimizer_state_storage_independent

    g5.install_open_audit()
    device = g7a_worker.configure_runtime()
    config, training = load_config()
    context, parent_sampler_states, parent_rng, actor_optimizer, actor_scheduler, actor_ownership, r2 = g7b.load_models_and_state(device)
    data = context["data"]
    generators = g7b.build_generators(training)
    samplers = g7b.build_samplers(data, generators, parent_sampler_states)
    g7b.restore_parent_rng(parent_rng, generators)
    policy, q1, q2 = context["actor"], context["q1"], context["q2"]
    q1_target, q2_target = context["q1_target"], context["q2_target"]
    critic_optimizer, critic_scheduler = context["optimizer"], context["scheduler"]
    modules = {"actor": policy, "q1": q1, "q2": q2, "q1_target": q1_target, "q2_target": q2_target}
    initial = {name: module_state_sha256(module) for name, module in modules.items()}
    backbones = {f"{name}.{camera}": module_state_sha256(getattr(module, camera)) for name, module in (("q1", q1), ("q2", q2)) for camera in ("camera1_backbone", "camera2_backbone")}
    ownership = {
        "actor": actor_ownership, "critic": context["ownership"],
        "actor_critic_parameter_intersection": len({id(p) for p in policy.parameters()} & {id(p) for module in (q1, q2) for p in module.parameters()}),
        "target_in_optimizer": 0, "target_actor": None,
    }
    require(ownership["actor_critic_parameter_intersection"] == 0, "G7_LONG_OPTIMIZER_OWNERSHIP")
    smoke_config = g7b.load_config()[0]
    fixed_action = g7b.fixed_diagnostic_bundle(smoke_config, data, policy, q1, device)
    fixed_gradient = {
        "actor_q_noise": fixed_action["actor_q_noise"][0:1],
        "actor_fm_noise": fixed_action["actor_fm_noise"][0:1],
        "actor_fm_timestep": fixed_action["actor_fm_timestep"][0:1],
    }
    fixed_validation = torch.load(FIXED, map_location=device, weights_only=False)
    require(sha(FIXED) == "002235cfc18cf939652c7a1bbe27ca0e752cf2e25e89fa415465ccfb3e8777e2", "G7_LONG_FIXED_DIAGNOSTIC_SHA")
    validation_rows = g7a_worker.load_split_rows("val")
    g7a_worker.attach_distance(validation_rows)
    validation_data = g7a_worker.split_data(data, validation_rows)
    protected = json.loads(args.protected.read_text())
    startup = startup_snapshot(args.protected)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    progress_path = OUTPUT / "progress.jsonl"
    flow_counter = g5.FlowCounter(inference_batch_size=4)
    reports, gradients, actions, public, validations, checkpoints = [], [], [], [], [], []
    torch.cuda.reset_peak_memory_stats(device)

    action0 = g7b.internal_action_diagnostic(policy, fixed_action["batch"], fixed_action["action_noise"], fixed_action["batch"]["delta_mean"], fixed_action["batch"]["delta_std"])
    actions.append(action0)
    rng_before_public = canonical_digest(g5.capture_rng_states(generators))
    public.append(g7b.public_diagnostic(policy, data, fixed_action["batch"]["current_actor_batch"], fixed_action["action_noise"], 0))
    require(canonical_digest(g5.capture_rng_states(generators)) == rng_before_public, "G7_LONG_PUBLIC_CONSUMED_RNG")
    validations.append(validation_diagnostic(cycle=0, context=context, validation_rows=validation_rows, validation_data=validation_data, fixed=fixed_validation, device=device, generators=generators, g5=g5, g7a_worker=g7a_worker))
    checkpoints.append(save_boundary(cycle=0, modules=modules, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, actor_scheduler=actor_scheduler, critic_scheduler=critic_scheduler, samplers=samplers, generators=generators, ownership=ownership, protected=protected, startup=startup, g5=g5))
    append_progress(progress_path, {"cycle": 0, "status": "complete_boundary", "validation": True, "checkpoint": checkpoints[-1]})

    started = time.perf_counter()
    for cycle in range(1, 257):
        cycle_started = time.perf_counter()
        critic_reports, q_reports = [], []
        raw_before = dict(r2.DIAGNOSTIC)
        actor_before = module_state_sha256(policy)
        with g7b.critic_internal_only():
            for local in range(2):
                step = 256 + 2 * (cycle - 1) + local + 1
                td_indices = samplers["td"].draw(16)
                calql_indices = samplers["calql"].draw(16)
                td_batch = data.build_batch(td_indices, policy, device, canonical_task_feature=q1.canonical_task_feature)
                calql_batch = data.build_batch(calql_indices, policy, device, canonical_task_feature=q1.canonical_task_feature)
                report = g5.critic_update(
                    step=step, policy=policy, q1=q1, q2=q2,
                    q1_target=q1_target, q2_target=q2_target,
                    optimizer=critic_optimizer, scheduler=critic_scheduler,
                    td_batch=td_batch, calql_batch=calql_batch, train_data=data,
                    proposal_sampler=samplers["empirical_random_proposal"],
                    generators=generators, flow_counter=flow_counter, config=training,
                )
                critic_reports.append(g7a_worker.compact_critic_report(report))
                q_reports.append(q_statistics(td_batch, q1, q2, q1_target, q2_target))
                del td_batch, calql_batch, report
                gc.collect(); torch.cuda.empty_cache()
            require(module_state_sha256(policy) == actor_before, "G7_LONG_ACTOR_CHANGED_DURING_CRITIC")
            gradient = g7a_worker.measure_actor_gradient_scale(
                policy=policy, q1=q1, q2=q2, train_data=data,
                actor_indices=[fixed_action["row_index"]], fixed=fixed_gradient,
                device=device, eta_candidates=[10.0], band=[0.01, 0.10],
            )
            actor_indices = samplers["actor"].draw(4)
            actor_batch = data.build_batch(actor_indices, policy, device, canonical_task_feature=q1.canonical_task_feature, include_flow_actions=True)
            actor_report = g5.actor_update(
                policy=policy, q1=q1, q2=q2, q1_target=q1_target, q2_target=q2_target,
                optimizer=actor_optimizer, scheduler=actor_scheduler,
                actor_batch=actor_batch, generators=generators,
                flow_counter=flow_counter, config=training,
            )
            del actor_batch
        probe = gradient["per_probe"][0]
        metric = probe["global"]
        gradient_report = {
            "cycle": cycle, "raw_q_over_fm": metric["raw_q_over_fm"],
            "weighted_eta_q_over_beta_fm": 10.0 * metric["raw_q_over_fm"],
            "cosine_similarity": metric["cosine_similarity"],
            "tcp6_q_gradient_norm": probe["tcp6_actor_q_gradient_norm"],
            "gripper_q_gradient_max_abs": probe["gripper_actor_q_gradient_max_abs"],
            "gripper_fm_gradient_norm": probe["gripper_flow_matching_gradient_norm"],
            "modules": probe["modules"],
        }
        gradients.append(gradient_report)
        require(
            gradient_report["tcp6_q_gradient_norm"] > 0.0
            and gradient_report["gripper_q_gradient_max_abs"] == 0.0
            and gradient_report["gripper_fm_gradient_norm"] > 0.0,
            "G7_LONG_ACTION_GRADIENT_CONTRACT",
        )
        if cycle >= 32:
            rolling_median = statistics.median(
                item["weighted_eta_q_over_beta_fm"] for item in gradients[-32:]
            )
            require(rolling_median <= 1.0, f"G7_LONG_Q_GRADIENT_DOMINATES_FM:{cycle}:{rolling_median}")
        action = g7b.internal_action_diagnostic(policy, fixed_action["batch"], fixed_action["action_noise"], fixed_action["batch"]["delta_mean"], fixed_action["batch"]["delta_std"])
        action["cycle"] = cycle
        action["normalized_tcp_drift_l2"] = float((torch.tensor(action["normalized_tcp6"]) - torch.tensor(action0["normalized_tcp6"])).norm())
        action["binary_gripper_change_rate"] = float(np.mean(np.asarray(action["normalized_gripper"]) != np.asarray(action0["normalized_gripper"])))
        actions.append(action)
        rng_before_public = canonical_digest(g5.capture_rng_states(generators))
        public_result = g7b.public_diagnostic(policy, data, fixed_action["batch"]["current_actor_batch"], fixed_action["action_noise"], cycle)
        require(canonical_digest(g5.capture_rng_states(generators)) == rng_before_public, "G7_LONG_PUBLIC_CONSUMED_RNG")
        public.append(public_result)
        raw_delta = r2.DIAGNOSTIC["raw_gripper_values"] - raw_before["raw_gripper_values"]
        outside_delta = r2.DIAGNOSTIC["raw_gripper_out_of_public_tolerance"] - raw_before["raw_gripper_out_of_public_tolerance"]
        report = {
            "cycle": cycle, "critic_updates": critic_reports,
            "q_statistics": q_reports, "actor_update": actor_report,
            "gradient_scale": gradient_report, "action_diagnostic": action,
            "public_predict": public_result,
            "raw_gripper_out_of_public_tolerance_rate": outside_delta / raw_delta if raw_delta else 0.0,
            "cumulative_samples": {
                "td_rows": 32 * cycle, "calql_rows": 32 * cycle,
                "actor_rows": 4 * cycle, "empirical_proposal_macro_actions": 64 * cycle,
            },
            "cycle_latency_seconds_before_checkpoint_and_validation": time.perf_counter() - cycle_started,
        }
        require(all(math.isfinite(float(value)) for item in critic_reports for value in item["loss"].values()), "G7_LONG_CRITIC_LOSS_NONFINITE")
        require(all(math.isfinite(float(value)) for value in actor_report["loss"].values()), "G7_LONG_ACTOR_LOSS_NONFINITE")
        reports.append(report)
        if cycle in {64, 128, 256}:
            validations.append(validation_diagnostic(cycle=cycle, context=context, validation_rows=validation_rows, validation_data=validation_data, fixed=fixed_validation, device=device, generators=generators, g5=g5, g7a_worker=g7a_worker))
        if cycle % 32 == 0:
            checkpoints.append(save_boundary(cycle=cycle, modules=modules, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, actor_scheduler=actor_scheduler, critic_scheduler=critic_scheduler, samplers=samplers, generators=generators, ownership=ownership, protected=protected, startup=startup, g5=g5))
        append_progress(progress_path, {
            "cycle": cycle, "status": "complete_boundary",
            "L_critic": [item["loss"]["L_critic"] for item in critic_reports],
            "L_FM": actor_report["loss"]["L_FM_window"],
            "L_actor_Q": actor_report["loss"]["L_actor_Q_window"],
            "L_actor_total": actor_report["loss"]["weighted_actor_total"],
            "weighted_gradient_ratio": gradient_report["weighted_eta_q_over_beta_fm"],
            "gradient_cosine": gradient_report["cosine_similarity"],
            "validation": cycle in {64, 128, 256},
            "checkpoint": cycle % 32 == 0,
        })
        print(f"G7_LONG_RUN_CYCLE {cycle}/256", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    ensure_all_gradients_none(*modules.values())
    final = {name: module_state_sha256(module) for name, module in modules.items()}
    require(all(final[name] != initial[name] for name in final), "G7_LONG_EXPECTED_PARAMETER_CHANGE_MISSING")
    require(critic_scheduler.last_epoch == 768 and actor_scheduler.last_epoch == 256, "G7_LONG_SCHEDULER_COUNTER_DRIFT")
    critic_steps = {int(value["step"].item()) for value in critic_optimizer.state.values() if "step" in value}
    actor_steps = {int(value["step"].item()) for value in actor_optimizer.state.values() if "step" in value}
    require(critic_steps == {768} and actor_steps and max(actor_steps) == 256 and min(actor_steps) >= 1, f"G7_LONG_OPTIMIZER_COUNTER_DRIFT:{critic_steps}:{actor_steps}")
    require(modules_storage_independent(q1, q2) and optimizer_state_storage_independent(critic_optimizer, q1, q2), "G7_LONG_CRITIC_STORAGE")
    require(backbones == {f"{name}.{camera}": module_state_sha256(getattr(module, camera)) for name, module in (("q1", q1), ("q2", q2)) for camera in ("camera1_backbone", "camera2_backbone")}, "G7_LONG_BACKBONE_CHANGED")
    require(all(bool(torch.isfinite(parameter).all()) for module in modules.values() for parameter in module.parameters()), "G7_LONG_PARAMETER_NONFINITE")
    weighted = [item["weighted_eta_q_over_beta_fm"] for item in gradients]
    raw = [item["raw_q_over_fm"] for item in gradients]
    cosine = [item["cosine_similarity"] for item in gradients]
    raw_count = r2.DIAGNOSTIC["raw_gripper_values"]
    result = {
        "worker_mode": "train", "environment": g7a_worker.environment_audit(),
        "parent": {"g7a_r2_loaded": True, "critic_optimizer_step": 256, "actor_optimizer_step": 0, "g7b_smoke_parent_used": False},
        "counters": counters_for_cycle(256), "cycles": reports,
        "validation_diagnostics": validations, "checkpoint_events": checkpoints,
        "gradient_scale_summary": {"raw": describe_p95(raw), "weighted_eta10": describe_p95(weighted), "cosine": describe_p95(cosine)},
        "parameter_change_matrix": {name: {"before": initial[name], "after": final[name], "changed": initial[name] != final[name]} for name in modules},
        "action_diagnostics": actions, "public_predict_diagnostics": public,
        "action_contract_v2": {
            "tcp6_q_gradient_nonzero_all_cycles": all(item["tcp6_q_gradient_norm"] > 0 for item in gradients),
            "gripper_q_gradient_exact_zero_all_cycles": all(item["gripper_q_gradient_max_abs"] == 0 for item in gradients),
            "gripper_fm_gradient_nonzero_all_cycles": all(item["gripper_fm_gradient_norm"] > 0 for item in gradients),
            "internal_raw_gripper_out_of_public_tolerance_rate": r2.DIAGNOSTIC["raw_gripper_out_of_public_tolerance"] / raw_count if raw_count else 0.0,
            "clipping_added": False, "resampling_added": False, "binary_ste_added": False,
        },
        "runtime": {"training_body_seconds": time.perf_counter() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "flow_counts": flow_counter.report()},
        "data_access": {"train_update_batch_memberships": 68 * 256, "train_gradient_probe_memberships": 256, "fixed_train_diagnostic_memberships": 1, "validation_transition_reads": 1205 * 4, "test_transition_reads": 0, "manual_g1_opens": 0, "manual_label_opens": 0, "reward_classifier_inference": 0, "reward_classifier_updates": 0},
        "optimizer_parameter_step_values": {"critic": sorted(critic_steps), "actor_sparse_routing_aware": sorted(actor_steps)},
    }
    atomic_json(args.result, result)


def verify(args) -> None:
    from forcesmolvla.rft import critic_training as g7a_worker
    import run_s2_g7b_worker as g7b
    from forcesmolvla.rft.long_run_checkpoint import counters_for_cycle, validate_cycle_checkpoint
    from forcesmolvla.rft.training_cycle import SerializableReplacementSampler, SerializableUniqueSampler, ensure_all_gradients_none

    device = g7a_worker.configure_runtime()
    _config, training = load_config()
    context, _parent_samplers, _parent_rng, actor_optimizer, actor_scheduler, _ownership, _r2 = g7b.load_models_and_state(device, with_data=False)
    checkpoint = CHECKPOINT_ROOT / "milestone_cycle_000256"
    manifest = validate_cycle_checkpoint(checkpoint, expected_cycle=256)
    modules = {name: context[name] for name in ("actor", "q1", "q2", "q1_target", "q2_target")}
    for name, module in modules.items():
        incompatible = module.load_state_dict(torch.load(checkpoint / f"models/{name}_state.pt", map_location="cpu", weights_only=False), strict=True)
        require(not incompatible.missing_keys and not incompatible.unexpected_keys, f"G7_LONG_VERIFY_MODEL:{name}")
    actor_optimizer.load_state_dict(torch.load(checkpoint / "optimizers/actor_optimizer_state.pt", map_location="cpu", weights_only=False))
    context["optimizer"].load_state_dict(torch.load(checkpoint / "optimizers/critic_optimizer_state.pt", map_location="cpu", weights_only=False))
    actor_scheduler.load_state_dict(torch.load(checkpoint / "schedulers/actor_scheduler_state.pt", map_location="cpu", weights_only=False))
    context["scheduler"].load_state_dict(torch.load(checkpoint / "schedulers/critic_scheduler_state.pt", map_location="cpu", weights_only=False))
    states = torch.load(checkpoint / "state/sampler_states.pt", map_location="cpu", weights_only=False)
    rng = torch.load(checkpoint / "state/rng_states.pt", map_location="cpu", weights_only=False)
    generators = g7b.build_generators(training)
    samplers = {
        "td": SerializableUniqueSampler(states["td"]["name"], tuple(states["td"]["population"]), generators["td_sampler"], states["td"]["draws"]),
        "calql": SerializableUniqueSampler(states["calql"]["name"], tuple(states["calql"]["population"]), generators["calql_sampler"], states["calql"]["draws"]),
        "actor": SerializableUniqueSampler(states["actor"]["name"], tuple(states["actor"]["population"]), generators["actor_sampler"], states["actor"]["draws"]),
        "empirical_random_proposal": SerializableReplacementSampler(states["empirical_random_proposal"]["name"], states["empirical_random_proposal"]["population_size"], generators["empirical_random_proposal"], states["empirical_random_proposal"]["draws"]),
    }
    g7b.restore_parent_rng(rng, generators)
    context["q1_target"].make_permanent_eval_target(); context["q2_target"].make_permanent_eval_target()
    ensure_all_gradients_none(*modules.values())
    critic_steps = {int(value["step"].item()) for value in context["optimizer"].state.values() if "step" in value}
    actor_steps = {int(value["step"].item()) for value in actor_optimizer.state.values() if "step" in value}
    require(critic_steps == {768} and actor_steps and max(actor_steps) == 256, "G7_LONG_VERIFY_OPTIMIZER_STEP")
    require(context["scheduler"].last_epoch == 768 and actor_scheduler.last_epoch == 256, "G7_LONG_VERIFY_SCHEDULER")
    require(all(bool(torch.isfinite(parameter).all()) for module in modules.values() for parameter in module.parameters()), "G7_LONG_VERIFY_NONFINITE")
    atomic_json(args.result, {
        "worker_mode": "fresh_process_strict_load", "environment": g7a_worker.environment_audit(),
        "checkpoint_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "counters": counters_for_cycle(256), "strict_model_load": True,
        "strict_optimizer_load": True, "strict_scheduler_load": True,
        "samplers_loaded": sorted(samplers), "rng_restored_last": True,
        "critic_optimizer_step": 768, "actor_optimizer_step": 256,
        "actor_sparse_parameter_step_values": sorted(actor_steps),
        "critic_scheduler_step": 768, "actor_scheduler_step": 256,
        "parameter_updates": 0, "sampler_draws_after_load": 0,
        "validation_transition_reads": 0, "test_transition_reads": 0,
        "manual_g1_opens": 0, "manual_label_opens": 0, "reward_classifier_calls": 0,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "verify"), required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--protected", type=Path)
    args = parser.parse_args()
    require(not args.result.exists(), "G7_LONG_RESULT_APPEND_ONLY")
    if args.mode == "train":
        require(args.protected and args.protected.is_file(), "G7_LONG_PROTECTED_REQUIRED")
        train(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
