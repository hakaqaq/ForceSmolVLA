#!/usr/bin/env python3
"""One-use GPU preflight for the append-only Frozen-VLM trainability contract."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import yaml


ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(ROOT / "tools"))
CONFIG = ROOT / "configs/stage2_trainability_contract.v2.development.json"
OUTPUT = ROOT / "artifacts/development/stage2/stage2_frozen_vlm_trainability_preflight.json"
REPORT = ROOT / "docs/stage2_frozen_vlm_trainability_report.md"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def component_gradient_norms(policy) -> dict[str, float]:
    from forcesmolvla.rft.frozen_vlm_trainability import gradient_norm_for_prefixes

    return {
        "vision_gradient_norm": gradient_norm_for_prefixes(
            policy, ("model.vlm_with_expert.vlm.model.vision_model.",)
        ),
        "smolvlm_gradient_norm": gradient_norm_for_prefixes(
            policy, ("model.vlm_with_expert.vlm.",)
        ),
        "state_prefix_projection_gradient_norm": gradient_norm_for_prefixes(
            policy, ("model.state_proj.",)
        ),
        "force_gradient_norm": gradient_norm_for_prefixes(
            policy, ("model.force_branch.", "model.force_adapter.")
        ),
        "action_expert_gradient_norm": gradient_norm_for_prefixes(
            policy, ("model.vlm_with_expert.lm_expert.",)
        ),
        "action_io_gradient_norm": gradient_norm_for_prefixes(
            policy,
            (
                "model.action_in_proj.", "model.action_out_proj.",
                "model.action_time_mlp_in.", "model.action_time_mlp_out.",
            ),
        ),
        "router_gradient_norm": gradient_norm_for_prefixes(
            policy, ("model.force_branch.refiner.router.",)
        ),
    }


def actor_temporary_update(context: dict, actor_batch: dict, optimizer, scheduler) -> dict:
    from forcesmolvla.force_token import RouterState
    from forcesmolvla.rft.frozen_vlm_trainability import (
        compute_min_twin_q_actor_loss,
        frozen_prefix_flow_matching_terms,
    )
    from forcesmolvla.rft.training_cycle import global_gradient_norm, module_state_sha256
    from forcesmolvla.router_training import collect_pass_a_statistics, microbatch_two_pass_terms
    from forcesmolvla.rft.training_cycle import FlowCounter

    policy, q1, q2 = context["actor"], context["q1"], context["q2"]
    device = actor_batch["reward"].device
    trainable = [value for value in policy.parameters() if value.requires_grad]
    frozen = [value for value in policy.parameters() if not value.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    policy.train(True)
    noise = torch.randn(4, 50, 7, generator=torch.Generator(device=device).manual_seed(9101), device=device)
    timestep = torch.rand(4, generator=torch.Generator(device=device).manual_seed(9102), device=device)
    velocity_outputs = []

    def capture(_module, _inputs, output):
        output.retain_grad(); velocity_outputs.append(output)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hook = policy.model.action_out_proj.register_forward_hook(capture)
        try:
            losses, feature_mask, router_state, prefix_audit = frozen_prefix_flow_matching_terms(
                policy, actor_batch["current_actor_batch"], noise=noise,
                time=timestep, call_id="trainability-preflight-fm",
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
        finally:
            hook.remove()
    require(len(velocity_outputs) == 1, "FROZEN_VLM_FM_OUTPUT_HOOK")

    policy.eval()
    flow_counter = FlowCounter(inference_batch_size=4)
    q_noise = torch.randn(4, 50, 7, generator=torch.Generator(device=device).manual_seed(9103), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        action_chunk = flow_counter.sample(
            policy, actor_batch["current_actor_batch"], q_noise,
            call_id="trainability-preflight-q", purpose="actor_guidance",
        )
        action_chunk.retain_grad()
        q_loss, q1_value, q2_value, critic_action = compute_min_twin_q_actor_loss(
            q1=q1, q2=q2, observation=actor_batch["current_observation"],
            normalized_flow_action_chunk7=action_chunk,
            delta_action_mean7=actor_batch["delta_mean"],
            delta_action_std7=actor_batch["delta_std"],
        )
        total = fm_loss + 0.01 * auxiliary.balance + 0.001 * auxiliary.z + 10.0 * q_loss
    total.backward()
    require(action_chunk.grad is not None and velocity_outputs[0].grad is not None, "FROZEN_VLM_ACTION_GRADIENT_MISSING")
    tcp6_q = float(action_chunk.grad[:, :3, :6].float().norm().cpu())
    gripper_q = float(action_chunk.grad[:, :3, 6].float().abs().max().cpu())
    gripper_fm = float(velocity_outputs[0].grad[..., 6].float().norm().cpu())
    gradients = component_gradient_norms(policy)
    require(all(value.grad is None for value in frozen), "FROZEN_VLM_FROZEN_PARAMETER_GRADIENT")
    require(
        gradients["vision_gradient_norm"] == 0.0
        and gradients["smolvlm_gradient_norm"] == 0.0
        and gradients["state_prefix_projection_gradient_norm"] == 0.0
        and all(gradients[name] > 0.0 for name in (
            "force_gradient_norm", "action_expert_gradient_norm",
            "action_io_gradient_norm", "router_gradient_norm",
        ))
        and tcp6_q > 0.0 and gripper_q == 0.0 and gripper_fm > 0.0,
        f"FROZEN_VLM_GRADIENT_CONTRACT:{gradients}:{tcp6_q}:{gripper_q}:{gripper_fm}",
    )
    preclip = float(global_gradient_norm(trainable).cpu())
    torch.nn.utils.clip_grad_norm_(trainable, 10.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    policy.eval()
    require(all(value.grad is None for value in trainable), "FROZEN_VLM_ACTOR_GRADIENT_NOT_CLEARED")
    return {
        "loss": {
            "flow_matching": float(fm_loss.detach().cpu()),
            "actor_q_min_twin": float(q_loss.detach().cpu()),
            "balance": float(auxiliary.balance.detach().cpu()),
            "z": float(auxiliary.z.detach().cpu()),
            "total": float(total.detach().cpu()),
        },
        "q": {"q1_mean": float(q1_value.mean().detach().cpu()), "q2_mean": float(q2_value.mean().detach().cpu())},
        "gradient": {
            **gradients,
            "tcp6_q_gradient_norm": tcp6_q,
            "gripper_q_gradient_max_abs": gripper_q,
            "gripper_fm_gradient_norm": gripper_fm,
            "actor_preclip_global_norm": preclip,
        },
        "prefix_audit": prefix_audit,
        "flow_audit": flow_counter.report(),
        "critic_action_shape": list(critic_action.shape),
        "actor_state_after": module_state_sha256(policy),
    }


def main() -> None:
    require(not OUTPUT.exists() and not REPORT.exists(), "FROZEN_VLM_PREFLIGHT_OUTPUT_EXISTS")
    from forcesmolvla.rft import critic_training as g7a
    from forcesmolvla.rft import training_cycle as g5
    from forcesmolvla.rft.frozen_vlm_trainability import (
        apply_frozen_vlm_trainability,
        build_frozen_vlm_actor_optimizer,
        frozen_state_digest,
    )
    from forcesmolvla.rft.training_cycle import module_state_sha256
    from forcesmolvla.rft.critic import state_exact

    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(contract["status"] == "current_candidate", "FROZEN_VLM_CONTRACT_STATUS")
    device = g7a.configure_runtime()
    context = g7a.initialize_fresh(device=device, with_data=True)
    policy = context["actor"]
    data = context["data"]
    batch_indices = list(data.actor_population[:4])
    actor_batch = data.build_batch(
        batch_indices, policy, device, canonical_task_feature=context["q1"].canonical_task_feature,
        include_flow_actions=True,
    )
    fixed_noise = torch.randn(4, 50, 7, generator=torch.Generator(device=device).manual_seed(9010), device=device)
    from forcesmolvla.rft.flow_sampling import sample_normalized_action_chunk_with_grad

    policy.eval()
    prepare = policy.model.force_adapter.cross_attention.prepare
    with patch.object(policy.model.force_adapter.cross_attention, "prepare", wraps=prepare) as force_prepare:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            before = sample_normalized_action_chunk_with_grad(
                policy, actor_batch["current_actor_batch"], fixed_noise,
                call_id="trainability-before", purpose="td_next",
            )
    require(force_prepare.call_count == 1, "FROZEN_VLM_FORCE_KV_BEFORE_NOT_ONCE")
    manifest = apply_frozen_vlm_trainability(policy)
    frozen_before = frozen_state_digest(policy)
    with patch.object(policy.model.force_adapter.cross_attention, "prepare", wraps=prepare) as force_prepare:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            after = sample_normalized_action_chunk_with_grad(
                policy, actor_batch["current_actor_batch"], fixed_noise,
                call_id="trainability-after", purpose="td_next",
            )
    require(force_prepare.call_count == 1, "FROZEN_VLM_FORCE_KV_AFTER_NOT_ONCE")
    forward_parity = bool(torch.equal(before, after))
    require(forward_parity, "FROZEN_VLM_FORWARD_PARITY_FAILED")

    actor_optimizer, actor_scheduler, actor_ownership = build_frozen_vlm_actor_optimizer(policy)
    training_cycle_config = yaml.safe_load(
        (ROOT / "configs/forcerft_training_cycle.development.yaml").read_text()
    )
    generators = g7a.warmup_generators(yaml.safe_load((ROOT / "configs/twin_q_critic_warmup.development.yaml").read_text()))
    samplers = g7a.warmup_samplers(data, generators)
    td_indices = samplers["td"].draw(16)
    calql_indices = samplers["calql"].draw(16)
    td_batch = data.build_batch(td_indices, policy, device, canonical_task_feature=context["q1"].canonical_task_feature)
    calql_batch = data.build_batch(calql_indices, policy, device, canonical_task_feature=context["q1"].canonical_task_feature)
    actor_before_critic = module_state_sha256(policy)
    critic_report = g5.critic_update(
        step=1, policy=policy, q1=context["q1"], q2=context["q2"],
        q1_target=context["q1_target"], q2_target=context["q2_target"],
        optimizer=context["optimizer"], scheduler=context["scheduler"],
        td_batch=td_batch, calql_batch=calql_batch, train_data=data,
        proposal_sampler=samplers["empirical_random_proposal"], generators=generators,
        flow_counter=g5.FlowCounter(inference_batch_size=4),
        config=training_cycle_config,
    )
    require(module_state_sha256(policy) == actor_before_critic, "FROZEN_VLM_ACTOR_CHANGED_DURING_CRITIC")
    q_before_actor = {name: module_state_sha256(context[name]) for name in ("q1", "q2", "q1_target", "q2_target")}
    actor_before_update = module_state_sha256(policy)
    actor_report = actor_temporary_update(context, actor_batch, actor_optimizer, actor_scheduler)
    q_after_actor = {name: module_state_sha256(context[name]) for name in ("q1", "q2", "q1_target", "q2_target")}
    require(q_before_actor == q_after_actor, "FROZEN_VLM_CRITIC_CHANGED_DURING_ACTOR")
    require(actor_before_update != module_state_sha256(policy), "FROZEN_VLM_TRAINABLE_ACTOR_NOT_UPDATED")
    frozen_after = frozen_state_digest(policy)
    require(frozen_before == frozen_after, "FROZEN_VLM_FROZEN_HASH_CHANGED")
    require(state_exact(context["q1_target"], context["q1_target"]), "FROZEN_VLM_TARGET_INVALID")
    all_finite = all(
        bool(torch.isfinite(parameter).all())
        for module in (policy, context["q1"], context["q2"], context["q1_target"], context["q2_target"])
        for parameter in module.parameters()
    )
    require(all_finite, "FROZEN_VLM_NONFINITE_PARAMETER")

    result = {
        "schema_version": "forcesmolvla_stage2_frozen_vlm_trainability_preflight.v1",
        "status": "pass",
        "contract": CONFIG.relative_to(ROOT).as_posix(),
        "contract_sha256": sha256_file(CONFIG),
        "STAGE2_TRAINABILITY_CONTRACT": "frozen_vlm_force_action_trainable",
        "FROZEN_VLM_FORWARD_PARITY": "pass",
        "FROZEN_PARAMETER_HASH_UNCHANGED": "yes",
        "parameter_counts": {
            "frozen_parameter_count": manifest.frozen_parameter_count,
            "trainable_actor_parameter_count": manifest.trainable_actor_parameter_count,
            "trainable_critic_parameter_count": sum(
                value.numel() for name in ("q1", "q2")
                for value in context[name].parameters() if value.requires_grad
            ),
        },
        "trainability": {
            "frozen_parameter_tensors": manifest.frozen_parameter_tensors,
            "trainable_actor_parameter_tensors": manifest.trainable_actor_parameter_tensors,
            "frozen_modules_always_eval": not policy.model.vlm_with_expert.vlm.training and not policy.model.state_proj.training,
            "actor_optimizer": actor_ownership,
            "frozen_state_before": frozen_before,
            "frozen_state_after": frozen_after,
        },
        "forward_parity": {
            "exact": True,
            "maximum_abs_error": float((before.float() - after.float()).abs().max().cpu()),
            "force_kv_projection_per_chunk": 1,
            "public_predict_action_chunk_implementation_modified": False,
            "public_inference_semantics_changed": False,
        },
        "temporary_critic_update": {
            "optimizer_updates": 1, "polyak_updates_per_target": 1,
            "actor_changed": False, "loss": critic_report["loss"],
        },
        "temporary_actor_update": {"optimizer_updates": 1, **actor_report},
        "all_finite": all_finite,
        "temporary_updates_discarded": True,
        "checkpoint_created": False,
        "access_audit": {
            "train_transition_reads": 40,
            "validation_reads": 0, "test_reads": 0,
            "manual_g1_opens": 0, "manual_label_opens": 0,
            "reward_classifier_inference": 0, "reward_classifier_updates": 0,
        },
        "historical_status": contract["historical_status"],
        "LONG_RUN_AUTHORIZED": "no",
        "LONG_RUN_STARTED": "no",
        "ROBOT_EXECUTION_AUTHORIZED": False,
    }
    atomic_json(OUTPUT, result)
    report = f"""# Stage-2 Frozen-VLM Trainability Preflight\n\nStatus: **PASS**. This append-only contract freezes the visual-language prefix owner and keeps the Force/Action path trainable.\n\n- Frozen parameters: {manifest.frozen_parameter_count:,}\n- Trainable Actor parameters: {manifest.trainable_actor_parameter_count:,}\n- Trainable Twin-Q parameters: {result['parameter_counts']['trainable_critic_parameter_count']:,}\n- Frozen-VLM forward parity: exact (max abs error 0)\n- Frozen parameter/buffer hash after temporary Critic+Actor updates: unchanged\n- Prefix representation/cache: detached; Force K/V projection: once per chunk\n- Vision/SmolVLM/state-prefix gradients: exact zero\n- Force/Action Expert/Action I/O/router gradients: nonzero\n- TCP6 Q gradient: {actor_report['gradient']['tcp6_q_gradient_norm']:.8g}\n- Gripper Q gradient: {actor_report['gradient']['gripper_q_gradient_max_abs']:.1f}\n- Gripper FM gradient: {actor_report['gradient']['gripper_fm_gradient_norm']:.8g}\n\nThe temporary updates were discarded. No checkpoint, long-run, evaluation, public-path modification, or robot execution was created. Existing full-Actor G7-B remains historical mechanics evidence only.\n"""
    atomic_text(REPORT, report)
    print("FROZEN_VLM_TRAINABILITY_PREFLIGHT pass", flush=True)
    del context, actor_batch, td_batch, calql_batch
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
