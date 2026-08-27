#!/usr/bin/env python3
"""S2-G3 supplement: fp32/bf16 repeated cached-vs-uncached gradient measurements."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile


ROOT = Path(__file__).parents[1].resolve()


def _precision_context(torch, mode: str):
    if mode == "bf16_outer_with_v4_2_fp32_islands":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mode == "fp32":
        return contextlib.nullcontext()
    raise ValueError(f"unsupported precision mode: {mode}")


def _summary(values: list[float]) -> dict:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "repeat_variance": float(array.var(ddof=0)),
    }


def _repeat_variance(torch, values: list) -> dict:
    stacked = torch.stack([value.detach().float().cpu() for value in values], dim=0)
    variance = stacked.var(dim=0, unbiased=False)
    difference = (stacked - stacked[0:1]).abs()
    return {
        "elementwise_variance_max": float(variance.max()),
        "elementwise_variance_mean": float(variance.mean()),
        "max_abs_from_repeat_0": float(difference.max()),
    }


def _run_repeat(
    *,
    torch,
    policy,
    batch,
    noise7,
    probe_parameters,
    probe_names,
    probe_weights,
    normalizer_mean,
    normalizer_std,
    precision_mode,
    repeat_index,
):
    from forcesmolvla.rft.flow_sampling import (
        critic_action_for_q_guidance,
        sample_normalized_action_chunk_with_grad,
    )
    from forcesmolvla.action_delta import decode_binary_gripper_width
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    from preflight_s2_differentiable_flow_gpu import (
        _binding,
        _gradient_parity,
        _prepare_core_inputs,
        _tensor_sha256,
        _uncached_chunk_with_checkpoint,
    )

    (
        images,
        image_masks,
        state,
        wrench,
        suffix_valid,
        feature_mask,
        noise32,
    ) = _prepare_core_inputs(policy, batch, noise7)
    sample_identity = tuple(batch["sample_identity"])
    state_tail_probe = state.clone()
    state_tail_probe[..., 7:] = torch.linspace(-1000, 1000, 25, device=noise7.device)
    state_tail_probe = state_tail_probe * (
        torch.arange(32, device=noise7.device) < 7
    ).view(1, 32)
    state_tail_sanitization_error = float((state_tail_probe - state).abs().max().cpu())
    perturbed_noise32 = noise32.clone()
    perturbed_noise32[..., 7:] = torch.linspace(-1000, 1000, 25, device=noise7.device)
    direct_kv = {"k": 0, "v": 0}
    direct_handles = [
        policy.model.force_adapter.cross_attention.k_proj.register_forward_hook(
            lambda *_: direct_kv.__setitem__("k", direct_kv["k"] + 1)
        ),
        policy.model.force_adapter.cross_attention.v_proj.register_forward_hook(
            lambda *_: direct_kv.__setitem__("v", direct_kv["v"] + 1)
        ),
    ]
    try:
        with torch.no_grad(), _precision_context(torch, precision_mode):
            direct32 = policy.model.sample_actions_masked(
                images,
                image_masks,
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK],
                state,
                perturbed_noise32,
                action_feature_mask=feature_mask,
                suffix_valid_mask=suffix_valid,
                wrench=wrench,
                force_context_binding=_binding(
                    policy,
                    name=f"matrix-{precision_mode}-{repeat_index}",
                    sample_identity=sample_identity,
                    device=noise7.device,
                ),
                audit_cache=True,
            )
    finally:
        for handle in direct_handles:
            handle.remove()
    direct7 = direct32[..., :7].float()
    direct_padding_max = float(direct32.masked_select(~feature_mask).abs().max().cpu())
    action_tail_probe = noise32.detach().clone()
    action_tail_probe[..., 7:] = torch.linspace(-2000, 2000, 25, device=noise7.device)
    fixed_timestep = torch.ones(
        noise7.shape[0], dtype=torch.float32, device=noise7.device
    )
    with torch.no_grad(), _precision_context(torch, precision_mode):
        action_velocity_baseline = policy.model.velocity_full(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            noise32,
            fixed_timestep,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            wrench=wrench,
        )
        action_velocity_perturbed = policy.model.velocity_full(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state_tail_probe,
            action_tail_probe,
            fixed_timestep,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            wrench=wrench,
        )
    action_tail_velocity_error = float(
        (action_velocity_baseline - action_velocity_perturbed).abs().max().cpu()
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(noise7.device)
    cached_kv = {"k": 0, "v": 0}
    step_outputs = []

    def capture_step(_module, _inputs, output):
        step_outputs.append(output)

    cached_handles = [
        policy.model.force_adapter.cross_attention.k_proj.register_forward_hook(
            lambda *_: cached_kv.__setitem__("k", cached_kv["k"] + 1)
        ),
        policy.model.force_adapter.cross_attention.v_proj.register_forward_hook(
            lambda *_: cached_kv.__setitem__("v", cached_kv["v"] + 1)
        ),
        policy.model.action_out_proj.register_forward_hook(capture_step),
    ]
    try:
        with _precision_context(torch, precision_mode):
            cached_chunk = sample_normalized_action_chunk_with_grad(
                policy,
                batch,
                noise7,
                call_id=f"matrix-{precision_mode}-{repeat_index}",
                purpose="actor_guidance",
            )
        if len(step_outputs) != 10:
            raise RuntimeError(f"S2_G3_EULER_VELOCITY_COUNT_MISMATCH:{len(step_outputs)}")
        q_input = critic_action_for_q_guidance(
            cached_chunk,
            delta_action_mean7=normalizer_mean,
            delta_action_std7=normalizer_std,
        )
        physical_candidate = (
            cached_chunk[:, :3].detach().float().cpu().numpy()
            * normalizer_std.detach().cpu().numpy()
            + normalizer_mean.detach().cpu().numpy()
        )
        public_decoded = decode_binary_gripper_width(physical_candidate)
        public_normalized_gripper = torch.as_tensor(
            public_decoded[..., 6], dtype=torch.float32, device=noise7.device
        )
        public_normalized_gripper = (
            public_normalized_gripper - normalizer_mean[6]
        ) / normalizer_std[6]
        sidecar_public_gripper_exact = bool(
            torch.equal(q_input[..., 6].detach(), public_normalized_gripper)
        )
        q_cached = (q_input * probe_weights).sum()
        gradients = torch.autograd.grad(
            q_cached,
            [*probe_parameters, *step_outputs, cached_chunk],
            allow_unused=True,
        )
    finally:
        for handle in cached_handles:
            handle.remove()
    parameter_gradients_cached = gradients[: len(probe_parameters)]
    step_gradients = gradients[
        len(probe_parameters) : len(probe_parameters) + len(step_outputs)
    ]
    action_gradient = gradients[-1]
    cached_output = cached_chunk.detach()
    cached_peak = {
        "allocated_bytes": torch.cuda.max_memory_allocated(noise7.device),
        "reserved_bytes": torch.cuda.max_memory_reserved(noise7.device),
    }
    if any(value is None for value in parameter_gradients_cached):
        raise RuntimeError("S2_G3_CACHED_PARAMETER_GRADIENT_MISSING")
    if any(value is None for value in step_gradients) or action_gradient is None:
        raise RuntimeError("S2_G3_CACHED_FLOW_GRAPH_GRADIENT_MISSING")

    expected_action_gradient = torch.zeros_like(action_gradient)
    expected_action_gradient[:, :3, :6] = probe_weights[:, :6]
    action_gradient_error = float(
        (action_gradient - expected_action_gradient).abs().max().cpu()
    )
    tcp_gradient_min_abs = float(action_gradient[:, :3, :6].abs().min().cpu())
    gripper_gradient_max_abs = float(action_gradient[:, :3, 6].abs().max().cpu())

    closed_normalized = (0.0 - normalizer_mean[6]) / normalizer_std[6]
    open_normalized = (0.085 - normalizer_mean[6]) / normalizer_std[6]
    closed_chunk = cached_output.clone()
    open_chunk = cached_output.clone()
    closed_chunk[:, :3, 6] = closed_normalized
    open_chunk[:, :3, 6] = open_normalized
    closed_q = (
        critic_action_for_q_guidance(
            closed_chunk,
            delta_action_mean7=normalizer_mean,
            delta_action_std7=normalizer_std,
        )
        * probe_weights
    ).sum()
    open_q = (
        critic_action_for_q_guidance(
            open_chunk,
            delta_action_mean7=normalizer_mean,
            delta_action_std7=normalizer_std,
        )
        * probe_weights
    ).sum()
    gripper_mode_q_delta = float((open_q - closed_q).cpu())
    wrapper_direct_error = float((cached_output - direct7).abs().max().cpu())

    del cached_chunk, q_input, q_cached, gradients, step_outputs
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(noise7.device)
    with _precision_context(torch, precision_mode):
        uncached_chunk = _uncached_chunk_with_checkpoint(
            policy,
            images=images,
            image_masks=image_masks,
            tokens=batch[OBS_LANGUAGE_TOKENS],
            language_mask=batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=state,
            wrench=wrench,
            suffix_valid=suffix_valid,
            feature_mask=feature_mask,
            noise32=noise32,
        )
    q_uncached = (
        critic_action_for_q_guidance(
            uncached_chunk,
            delta_action_mean7=normalizer_mean,
            delta_action_std7=normalizer_std,
        )
        * probe_weights
    ).sum()
    parameter_gradients_uncached = torch.autograd.grad(
        q_uncached, probe_parameters, allow_unused=True
    )
    if any(value is None for value in parameter_gradients_uncached):
        raise RuntimeError("S2_G3_UNCACHED_PARAMETER_GRADIENT_MISSING")
    uncached_output = uncached_chunk.detach()
    uncached_peak = {
        "allocated_bytes": torch.cuda.max_memory_allocated(noise7.device),
        "reserved_bytes": torch.cuda.max_memory_reserved(noise7.device),
    }

    parity = {
        name: _gradient_parity(cached, uncached)
        for name, cached, uncached in zip(
            probe_names,
            parameter_gradients_cached,
            parameter_gradients_uncached,
            strict=True,
        )
    }
    step_norms = [float(value.detach().float().norm().cpu()) for value in step_gradients]
    measurement = {
        "repeat_index": repeat_index,
        "output": {
            "wrapper_vs_direct_core_max_abs": wrapper_direct_error,
            "cached_vs_uncached_max_abs": float(
                (cached_output - uncached_output).abs().max().cpu()
            ),
            "cached_sha256": _tensor_sha256(cached_output),
            "uncached_sha256": _tensor_sha256(uncached_output),
        },
        "per_module_gradient_parity": parity,
        "action_contract": {
            "tcp_k3x6_gradient_min_abs": tcp_gradient_min_abs,
            "gripper_k3_gradient_max_abs": gripper_gradient_max_abs,
            "full_chunk_expected_gradient_max_abs_error": action_gradient_error,
            "closed_to_open_q_probe_delta": gripper_mode_q_delta,
            "sidecar_public_gripper_decode_exact": sidecar_public_gripper_exact,
        },
        "ten_euler_step_gradient_norms": step_norms,
        "force_kv_projection_calls": {"direct": direct_kv, "cached": cached_kv},
        "prefix_append_crop_cache_audit_passed": True,
        "independent_padding_perturbations": {
            "state_7_to_32_pre_core_sanitization_max_abs_delta": state_tail_sanitization_error,
            "noise_7_to_32_flow_output_max_abs_delta": wrapper_direct_error,
            "action_7_to_32_velocity_max_abs_delta": action_tail_velocity_error,
            "action_output_padding_max_abs": direct_padding_max,
        },
        "peak_memory": {"cached": cached_peak, "uncached_checkpointed": uncached_peak},
    }
    internal = {
        "cached_gradients": [value.detach().float().cpu() for value in parameter_gradients_cached],
        "uncached_gradients": [
            value.detach().float().cpu() for value in parameter_gradients_uncached
        ],
        "cached_output": cached_output.detach().float().cpu(),
        "uncached_output": uncached_output.detach().float().cpu(),
    }
    del uncached_chunk, q_uncached
    gc.collect()
    torch.cuda.empty_cache()
    return measurement, internal


def _flow_matching_gripper_probe(
    *, torch, policy, batch, noise7, precision_mode, minimum_norm
) -> dict:
    from lerobot.policies.smolvla.modeling_smolvla import pad_vector
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    from preflight_s2_differentiable_flow_gpu import _prepare_core_inputs

    (
        images,
        image_masks,
        state,
        wrench,
        suffix_valid,
        feature_mask,
        noise32,
    ) = _prepare_core_inputs(policy, batch, noise7)
    actions32 = pad_vector(batch["action"], 32) * feature_mask.to(batch["action"].dtype)
    time = torch.full((noise7.shape[0],), 0.5, dtype=torch.float32, device=noise7.device)
    velocity_outputs = []
    handle = policy.model.action_out_proj.register_forward_hook(
        lambda _module, _inputs, output: velocity_outputs.append(output)
    )
    try:
        with _precision_context(torch, precision_mode):
            losses = policy.model.forward(
                images,
                image_masks,
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK],
                state,
                actions32,
                noise32,
                time,
                action_feature_mask=feature_mask,
                suffix_valid_mask=suffix_valid,
                wrench=wrench,
            )
        if len(velocity_outputs) != 1:
            raise RuntimeError("S2_G3_FM_ACTION_OUTPUT_HOOK_COUNT_INVALID")
        gripper_loss = losses[..., 6].sum() / suffix_valid.sum().clamp_min(1)
        related_names = ["model.action_out_proj.bias", "model.action_in_proj.bias"]
        named = dict(policy.named_parameters())
        gradients = torch.autograd.grad(
            gripper_loss,
            [velocity_outputs[0], *(named[name] for name in related_names)],
            allow_unused=True,
        )
    finally:
        handle.remove()
    if any(value is None for value in gradients):
        raise RuntimeError("S2_G3_FM_GRIPPER_GRADIENT_MISSING")
    output_gradient = gradients[0].detach().float()
    related = {
        name: float(gradient.detach().float().norm().cpu())
        for name, gradient in zip(related_names, gradients[1:], strict=True)
    }
    result = {
        "fixed_timestep": 0.5,
        "gripper_loss": float(gripper_loss.detach().float().cpu()),
        "gripper_velocity_output_gradient_norm": float(
            output_gradient[..., 6].norm().cpu()
        ),
        "non_gripper_velocity_output_gradient_max_abs": float(
            output_gradient[..., :6].abs().max().cpu()
        ),
        "related_parameter_gradient_norms": related,
    }
    result["pass"] = (
        result["gripper_velocity_output_gradient_norm"] > minimum_norm
        and result["non_gripper_velocity_output_gradient_max_abs"] == 0.0
        and all(value > minimum_norm for value in related.values())
    )
    return result


def _run_precision_mode(torch, config, args, mode: str, device) -> dict:
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from p8_checkpoint_common import load_fixed_validation_inputs
    from preflight_s2_common import module_state_dict_sha256
    from preflight_s2_differentiable_flow_gpu import _slice_first_row

    with contextlib.redirect_stdout(sys.stderr):
        policy = ForceSmolVLAPolicy.from_pretrained(
            args.checkpoint.resolve(),
            local_files_only=True,
            force_download=False,
            strict=True,
            artifact_use="development",
        ).to(device)
    if mode == "fp32":
        policy = policy.float()
    policy.eval()
    if not all(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("S2_G3_PARENT_ACTOR_NOT_FULLY_TRAINABLE")
    state_before = module_state_dict_sha256(policy)
    generation_before = policy.model.parameter_generation()
    fixture = json.loads(
        (args.checkpoint.resolve() / "manifests/fixed_validation_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    full_batch, _raw, _runtime = load_fixed_validation_inputs(
        policy, args.dataset_root.resolve(), fixture, device
    )
    batch = _slice_first_row(full_batch)
    row = fixture["tuple_list"][0]
    batch["sample_identity"] = (
        f"episode={row['episode_index']}/frame={row['frame_index']}",
    )
    noise7 = torch.tensor(
        fixture["epsilon7"]["tensor"][:1], dtype=torch.float32, device=device
    )
    normalizer = json.loads(
        (args.checkpoint.resolve() / "manifests/normalizer_manifest.json").read_text(
            encoding="utf-8"
        )
    )["features"]["delta_action7"]
    mean = torch.tensor(normalizer["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(normalizer["std"], dtype=torch.float32, device=device)
    probe_names = config["gradient_probe_parameters"]
    named = dict(policy.named_parameters())
    missing = [name for name in probe_names if name not in named]
    if missing:
        raise RuntimeError(f"S2_G3_GRADIENT_PROBE_PARAMETER_MISSING:{missing}")
    probe_parameters = [named[name] for name in probe_names]
    weights = torch.tensor(
        config["synthetic_k3_action_probe_weights"], dtype=torch.float32, device=device
    )
    if weights.shape != (3, 7) or torch.any(weights == 0):
        raise RuntimeError("S2_G3_SYNTHETIC_K3_PROBE_WEIGHTS_INVALID")

    repeats = []
    internal = []
    for repeat_index in range(config["fixed_input_repeats"]):
        measurement, private = _run_repeat(
            torch=torch,
            policy=policy,
            batch=batch,
            noise7=noise7,
            probe_parameters=probe_parameters,
            probe_names=probe_names,
            probe_weights=weights,
            normalizer_mean=mean,
            normalizer_std=std,
            precision_mode=mode,
            repeat_index=repeat_index,
        )
        repeats.append(measurement)
        internal.append(private)
        print(f"S2_G3_MATRIX:{mode}:repeat={repeat_index + 1}/3", flush=True)

    per_module = {}
    for parameter_index, name in enumerate(probe_names):
        records = [item["per_module_gradient_parity"][name] for item in repeats]
        per_module[name] = {
            "repeats": records,
            "summary": {
                "relative_l2_error": _summary(
                    [item["relative_l2_error"] for item in records]
                ),
                "cosine_similarity": _summary(
                    [item["cosine_similarity"] for item in records]
                ),
                "maximum_absolute_error": _summary(
                    [item["max_abs_error"] for item in records]
                ),
                "cached_gradient_repeat": _repeat_variance(
                    torch,
                    [item["cached_gradients"][parameter_index] for item in internal],
                ),
                "uncached_gradient_repeat": _repeat_variance(
                    torch,
                    [item["uncached_gradients"][parameter_index] for item in internal],
                ),
            },
        }
    fm_probe = _flow_matching_gripper_probe(
        torch=torch,
        policy=policy,
        batch=batch,
        noise7=noise7,
        precision_mode=mode,
        minimum_norm=float(config["minimum_nonzero_gradient_norm"]),
    )
    state_after = module_state_dict_sha256(policy)
    generation_after = policy.model.parameter_generation()
    all_values = [
        value
        for item in repeats
        for record in item["per_module_gradient_parity"].values()
        for value in (
            record["relative_l2_error"],
            record["cosine_similarity"],
            record["max_abs_error"],
        )
    ]
    contracts = {
        "three_fixed_input_repeats_complete": len(repeats) == 3,
        "all_measurements_finite": all(math.isfinite(value) for value in all_values),
        "tcp_k3x6_q_gradients_nonzero": all(
            item["action_contract"]["tcp_k3x6_gradient_min_abs"] > 0
            for item in repeats
        ),
        "gripper_q_gradient_exact_zero": all(
            item["action_contract"]["gripper_k3_gradient_max_abs"] == 0
            for item in repeats
        ),
        "gripper_mode_visible_to_q_forward": all(
            item["action_contract"]["closed_to_open_q_probe_delta"] != 0
            for item in repeats
        ),
        "sidecar_public_gripper_decode_exact": all(
            item["action_contract"]["sidecar_public_gripper_decode_exact"]
            for item in repeats
        ),
        "all_ten_euler_steps_receive_q_gradient": all(
            len(item["ten_euler_step_gradient_norms"]) == 10
            and min(item["ten_euler_step_gradient_norms"]) > 0
            for item in repeats
        ),
        "representative_q_parameter_gradients_nonzero": all(
            record["cached"]["norm"] > float(config["minimum_nonzero_gradient_norm"])
            and record["uncached"]["norm"]
            > float(config["minimum_nonzero_gradient_norm"])
            for item in repeats
            for record in item["per_module_gradient_parity"].values()
        ),
        "force_kv_once_per_cached_flow": all(
            item["force_kv_projection_calls"]["cached"] == {"k": 1, "v": 1}
            and item["force_kv_projection_calls"]["direct"] == {"k": 1, "v": 1}
            for item in repeats
        ),
        "prefix_append_crop_cache_audit": all(
            item["prefix_append_crop_cache_audit_passed"] for item in repeats
        ),
        "state_action_noise_padding_isolated": all(
            all(value == 0.0 for value in item["independent_padding_perturbations"].values())
            for item in repeats
        ),
        "flow_matching_gripper_gradient_nonzero": fm_probe["pass"],
        "state_dict_and_generation_unchanged": state_before == state_after
        and generation_before == generation_after,
        "no_parameter_grad_buffers_materialized": all(
            parameter.grad is None for parameter in policy.parameters()
        ),
    }
    result = {
        "precision_mode": mode,
        "repeats": repeats,
        "per_module_gradient_parity": per_module,
        "repeat_output_variance": {
            "cached": _repeat_variance(
                torch, [item["cached_output"] for item in internal]
            ),
            "uncached": _repeat_variance(
                torch, [item["uncached_output"] for item in internal]
            ),
        },
        "flow_matching_gripper_probe": fm_probe,
        "state_dict": {"before_sha256": state_before, "after_sha256": state_after},
        "parameter_generation": {"before": generation_before, "after": generation_after},
        "contracts": contracts,
    }
    del policy, batch, full_batch, internal
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/stage2_g3_gradient_matrix.development.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "outputs/development/task2_lerobotv3_full_sft_10k_r5/checkpoints/step_010000",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "datasets/task2_lerobotv3"
    )
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite G3 supplement artifact: {output}")
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 required")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("PYTHONHASHSEED=42 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required")

    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "tools"))
    from forcesmolvla.rft.source_manifest import sha256_file, stage2_source_manifest_binding

    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if (
        config.get("gate") != "S2-G3-supplement"
        or config.get("precision_modes")
        != ["fp32", "bf16_outer_with_v4_2_fp32_islands"]
        or config.get("fixed_input_repeats") != 3
        or config.get("gradient_parity_formal_thresholds") != "unapproved"
        or config.get("critic_action_slots") != 3
    ):
        raise RuntimeError("S2_G3_MATRIX_CONFIG_SEMANTICS_DRIFT")
    action_contract = json.loads(
        (ROOT / config["action_contract"]).read_text(encoding="utf-8")
    )
    if (
        action_contract.get("critic_action_input_dim") != 7
        or action_contract.get("critic_action_shape") != [3, 7]
        or action_contract.get("critic_duration_mode") != "fixed_k"
        or action_contract.get("partial_action_interface") is not False
        or action_contract.get("critic_receives_terminal_derived_duration_or_mask")
        is not False
        or action_contract.get("actor_q_guided_action_dims") != list(range(6))
        or action_contract.get("gripper_q_gradient") is not False
    ):
        raise RuntimeError("S2_G3_MATRIX_ACTION_CONTRACT_DRIFT")
    source_manifest_path = args.source_manifest or ROOT / config["stage2_source_manifest"]
    source_manifest = stage2_source_manifest_binding(ROOT, source_manifest_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")
    gpu_name = torch.cuda.get_device_name(0)
    if "4090 D" not in gpu_name and "4090D" not in gpu_name:
        raise RuntimeError(f"S2_G3_REQUIRES_RTX_4090D:{gpu_name}")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    measurements = {
        mode: _run_precision_mode(torch, config, args, mode, device)
        for mode in config["precision_modes"]
    }
    contracts = {
        "fp32_measurement_complete": all(
            measurements["fp32"]["contracts"].values()
        ),
        "bf16_measurement_complete": all(
            measurements["bf16_outer_with_v4_2_fp32_islands"]["contracts"].values()
        ),
        "formal_thresholds_unapproved": config["gradient_parity_formal_thresholds"]
        == "unapproved",
        "optimizer_not_created": True,
        "training_not_started": True,
    }
    result = {
        "schema_version": "1.0",
        "gate": "S2-G3-supplement",
        "gate_status": "pass" if all(contracts.values()) else "fail",
        "artifact_status": "development_only",
        "formal_eligible": False,
        "gpu": gpu_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_artifact_manifest_sha256": sha256_file(
            args.checkpoint.resolve() / "artifact_manifest.json"
        ),
        "config": {
            "relative_path": args.config.resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256_file(args.config.resolve()),
        },
        "action_contract": {
            "relative_path": config["action_contract"],
            "sha256": sha256_file(ROOT / config["action_contract"]),
            "paper_description": config["paper_description"],
        },
        "stage2_source_manifest": source_manifest,
        "gradient_parity_formal_thresholds": "unapproved",
        "router_q_origin_gradient": {
            "status": "measurement_only_deferred_to_G4",
            "parameter": "model.force_branch.refiner.router.bias",
            "semantics_modified_in_this_gate": False,
        },
        "measurements": measurements,
        "contracts": contracts,
        "critic_created": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "training_loop_created": False,
        "real_rft_training_started": False,
        "robot_actions_sent": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    print(json.dumps({"gate_status": result["gate_status"], "path": str(output)}))
    if result["gate_status"] != "pass":
        raise RuntimeError("S2_G3_GRADIENT_MATRIX_FAILED")


if __name__ == "__main__":
    main()
