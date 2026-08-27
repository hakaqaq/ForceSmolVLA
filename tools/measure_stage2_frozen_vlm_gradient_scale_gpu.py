#!/usr/bin/env python3
"""Discarded Frozen-VLM gradient-scale and one-step action-drift probe."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(ROOT / "tools"))
OUTPUT = ROOT / "artifacts/development/stage2/batch_scaling/stage2/frozen_vlm_gradient_scale.json"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def group(name: str) -> str:
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


def metrics(fm: dict[str, torch.Tensor], policy) -> dict:
    totals: dict[str, dict[str, float]] = {"global": {"fm2": 0.0, "q2": 0.0, "dot": 0.0}}
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        target = totals.setdefault(group(name), {"fm2": 0.0, "q2": 0.0, "dot": 0.0})
        fm_value = fm.get(name)
        q_value = parameter.grad.detach().float().cpu() if parameter.grad is not None else None
        fm2 = float(fm_value.square().sum()) if fm_value is not None else 0.0
        q2 = float(q_value.square().sum()) if q_value is not None else 0.0
        dot = float((fm_value * q_value).sum()) if fm_value is not None and q_value is not None else 0.0
        for destination in (target, totals["global"]):
            destination["fm2"] += fm2; destination["q2"] += q2; destination["dot"] += dot
    result = {}
    for name, values in totals.items():
        fm_norm, q_norm = math.sqrt(values["fm2"]), math.sqrt(values["q2"])
        result[name] = {
            "fm_norm": fm_norm, "q_norm": q_norm,
            "raw_q_over_fm": q_norm / max(fm_norm, torch.finfo(torch.float32).tiny),
            "weighted_eta10_q_over_beta1_fm": 10.0 * q_norm / max(fm_norm, torch.finfo(torch.float32).tiny),
            "cosine_similarity": values["dot"] / (fm_norm * q_norm) if fm_norm and q_norm else 0.0,
        }
    return result


def fixed_action(policy, batch, noise, mean, std, call_id: str):
    from forcesmolvla.rft.critic_action_adapter_v2 import (
        critic_action_for_q_guidance_v2, raw_gripper_out_of_public_tolerance_mask,
    )
    from forcesmolvla.rft.flow_sampling import sample_normalized_action_chunk_with_grad

    policy.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        chunk = sample_normalized_action_chunk_with_grad(
            policy, batch, noise, call_id=call_id, purpose="td_next"
        )
    action = critic_action_for_q_guidance_v2(
        chunk, delta_action_mean7=mean, delta_action_std7=std
    )
    raw_rate = float(raw_gripper_out_of_public_tolerance_mask(
        chunk[:, :3, 6], gripper_mean=mean[6], gripper_std=std[6]
    ).float().mean().cpu())
    return chunk.detach(), action.detach(), raw_rate


def main() -> None:
    require(not OUTPUT.exists(), "FROZEN_VLM_GRADIENT_SCALE_OUTPUT_EXISTS")
    import benchmark_stage2_batch_scaling_gpu as benchmark
    from forcesmolvla.rft.frozen_vlm_trainability import (
        compute_min_twin_q_actor_loss, frozen_prefix_flow_matching_terms, frozen_state_digest,
    )

    device = benchmark.configure_runtime()
    context, _training, generators, samplers, actor_optimizer, actor_scheduler, _ownership, _manifest = benchmark.load_context(device)
    policy, q1, q2 = context["actor"], context["q1"], context["q2"]
    indices = samplers["actor"].draw(24)
    batch = context["data"].build_batch(
        indices, policy, device, canonical_task_feature=q1.canonical_task_feature,
        include_flow_actions=True,
    )
    trainable = [(name, value) for name, value in policy.named_parameters() if value.requires_grad]
    frozen_before = frozen_state_digest(policy)
    for _name, value in trainable:
        value.grad = None

    fm_noise = torch.randn(24, 50, 7, generator=torch.Generator(device=device).manual_seed(9921), device=device)
    fm_time = torch.rand(24, generator=torch.Generator(device=device).manual_seed(9922), device=device)
    velocity_outputs = []
    hook = policy.model.action_out_proj.register_forward_hook(
        lambda _m, _i, output: (output.retain_grad(), velocity_outputs.append(output))[-1]
    )
    try:
        policy.train(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses, feature_mask, _router, prefix_audit = frozen_prefix_flow_matching_terms(
                policy, batch["current_actor_batch"], noise=fm_noise, time=fm_time,
                call_id="gradient-scale-fm",
            )
            fm_loss = losses.sum() / feature_mask.sum().clamp_min(1)
        fm_loss.backward()
    finally:
        hook.remove()
    require(len(velocity_outputs) == 1 and velocity_outputs[0].grad is not None, "FROZEN_VLM_GRADIENT_FM_OUTPUT")
    gripper_fm = float(velocity_outputs[0].grad[..., 6].float().norm().cpu())
    fm_gradients = {
        name: value.grad.detach().float().cpu().clone()
        for name, value in trainable if value.grad is not None
    }
    for _name, value in trainable:
        value.grad = None

    q_noise = torch.randn(24, 50, 7, generator=torch.Generator(device=device).manual_seed(9923), device=device)
    flow = benchmark.TimedFlowCounter(inference_batch_size=24)
    policy.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        chunk = flow.sample(
            policy, batch["current_actor_batch"], q_noise,
            call_id="gradient-scale-q", purpose="actor_guidance",
        )
        chunk.retain_grad()
        q_loss, _q1, _q2, _critic_action = compute_min_twin_q_actor_loss(
            q1=q1, q2=q2, observation=batch["current_observation"],
            normalized_flow_action_chunk7=chunk,
            delta_action_mean7=batch["delta_mean"], delta_action_std7=batch["delta_std"],
        )
    q_loss.backward()
    require(chunk.grad is not None, "FROZEN_VLM_GRADIENT_Q_ACTION")
    tcp6_q = float(chunk.grad[:, :3, :6].float().norm().cpu())
    gripper_q = float(chunk.grad[:, :3, 6].float().abs().max().cpu())
    scale = metrics(fm_gradients, policy)
    require(
        scale["global"]["fm_norm"] > 0 and scale["global"]["q_norm"] > 0
        and tcp6_q > 0 and gripper_q == 0 and gripper_fm > 0,
        "FROZEN_VLM_GRADIENT_SCALE_CONTRACT",
    )
    for _name, value in trainable:
        value.grad = None
    del fm_gradients, losses, fm_loss, q_loss, chunk
    torch.cuda.empty_cache()

    fixed_batch = {
        name: value[:1] if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 24
        else type(value)(value[:1]) if isinstance(value, (tuple, list)) and len(value) == 24
        else value
        for name, value in batch["current_actor_batch"].items()
    }
    fixed_noise = torch.randn(1, 50, 7, generator=torch.Generator(device=device).manual_seed(9930), device=device)
    raw_before, action_before, raw_rate_before = fixed_action(
        policy, fixed_batch, fixed_noise, batch["delta_mean"], batch["delta_std"], "drift-before"
    )
    public_before = benchmark.public_audit(context, batch, fixed_noise, 0)
    update = benchmark.actor_update(
        context=context, batch=batch, optimizer=actor_optimizer, scheduler=actor_scheduler,
        actor_batch_size=24, flow_noise_generator=generators["flow_matching_noise"],
        flow_time_generator=generators["flow_matching_timestep"],
        actor_q_generator=generators["actor_q_flow_noise"], update_id="gradient-scale-discarded-step",
    )
    raw_after, action_after, raw_rate_after = fixed_action(
        policy, fixed_batch, fixed_noise, batch["delta_mean"], batch["delta_std"], "drift-after"
    )
    public_after = benchmark.public_audit(context, batch, fixed_noise, 1)
    frozen_after = frozen_state_digest(policy)
    require(frozen_before == frozen_after and public_before["semantic_success"] and public_after["semantic_success"], "FROZEN_VLM_GRADIENT_PROBE_FROZEN_OR_PUBLIC")
    tcp_drift = float((action_after[..., :6] - action_before[..., :6]).float().norm(dim=-1).mean().cpu())
    gripper_change = float((action_after[..., 6] != action_before[..., 6]).float().mean().cpu())
    result = {
        "schema_version": "forcesmolvla_frozen_vlm_gradient_scale.v1",
        "status": "pass", "actor_physical_batch_size": 24,
        "critic_parent": "g7a_r2_critic_warmup_checkpoint",
        "actor_q_reduction": "min_online_twin_q",
        "eta": 10.0, "beta": 1.0,
        "L_FM": float(update["loss"]["flow_matching"]),
        "L_actor_Q": float(update["loss"]["actor_q_min_twin"]),
        "gradient_scale": scale,
        "tcp6_q_gradient_norm": tcp6_q,
        "gripper_q_gradient_max_abs": gripper_q,
        "gripper_fm_gradient_norm": gripper_fm,
        "prefix_audit": prefix_audit,
        "fixed_action_diagnostic": {
            "normalized_tcp_action_drift_mean_l2": tcp_drift,
            "binary_gripper_change_rate": gripper_change,
            "raw_gripper_out_of_public_tolerance_rate_before": raw_rate_before,
            "raw_gripper_out_of_public_tolerance_rate_after": raw_rate_after,
            "raw_flow_chunk_change_l2": float((raw_after - raw_before).float().norm().cpu()),
            "public_predict_before": public_before,
            "public_predict_after": public_after,
        },
        "frozen_parameter_hash_unchanged": True,
        "temporary_actor_optimizer_updates": 1,
        "temporary_state_discarded": True, "checkpoint_created": False,
        "validation_reads": 0, "test_reads": 0, "manual_g1_opens": 0,
        "manual_label_opens": 0, "reward_classifier_inference": 0,
    }
    require(all(math.isfinite(float(value)) for value in (
        scale["global"]["raw_q_over_fm"], scale["global"]["weighted_eta10_q_over_beta1_fm"],
        scale["global"]["cosine_similarity"], tcp_drift, gripper_change,
    )), "FROZEN_VLM_GRADIENT_RESULT_NONFINITE")
    atomic_json(OUTPUT, result)
    print("FROZEN_VLM_GRADIENT_SCALE pass", flush=True)


if __name__ == "__main__":
    main()
