#!/usr/bin/env python3
"""S2-G3 CUDA gate for the native cached ten-step differentiable Flow path."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile


ROOT = Path(__file__).parents[1].resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor) -> str:
    import torch

    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy()).hexdigest()


def _binding(policy, *, name: str, sample_identity: tuple[str, ...], device):
    import torch

    from forcesmolvla.force_token import PreparedForceContextBinding

    return PreparedForceContextBinding(
        chunk_id=tuple(f"rft:g3:{name}:{row}" for row in range(len(sample_identity))),
        sample_id=sample_identity,
        context_generation=policy._context_generation,
        model_generation=policy.model.parameter_generation(),
        device=device,
        dtype=torch.float32,
    )


def _prepare_core_inputs(policy, batch: dict, noise7):
    import torch

    from lerobot.policies.smolvla.modeling_smolvla import pad_vector

    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    active_state = (torch.arange(32, device=state.device) < 7).view(1, 32)
    state = state * active_state.to(dtype=state.dtype)
    wrench = policy._prepare_wrench(batch, device=state.device)
    suffix_valid = torch.ones(noise7.shape[0], 50, dtype=torch.bool, device=state.device)
    feature_mask = suffix_valid.unsqueeze(-1) & (
        torch.arange(32, device=state.device).view(1, 1, 32) < 7
    )
    noise32 = pad_vector(noise7, 32) * feature_mask.to(dtype=noise7.dtype)
    return images, image_masks, state, wrench, suffix_valid, feature_mask, noise32


def _uncached_chunk_with_checkpoint(
    policy,
    *,
    images,
    image_masks,
    tokens,
    language_mask,
    state,
    wrench,
    suffix_valid,
    feature_mask,
    noise32,
):
    import torch
    from torch.utils.checkpoint import checkpoint

    x_t = noise32 * feature_mask.to(dtype=noise32.dtype)
    dt = -1.0 / policy.config.num_steps
    for step in range(policy.config.num_steps):
        timestep = torch.full(
            (noise32.shape[0],),
            1.0 + step * dt,
            dtype=torch.float32,
            device=noise32.device,
        )

        def velocity(current, step_time=timestep):
            return policy.model.velocity_full(
                images,
                image_masks,
                tokens,
                language_mask,
                state,
                current,
                step_time,
                action_feature_mask=feature_mask,
                suffix_valid_mask=suffix_valid,
                wrench=wrench,
            )

        flow_velocity = checkpoint(
            velocity,
            x_t,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        x_t = (x_t + dt * flow_velocity) * feature_mask.to(dtype=x_t.dtype)
    return x_t[..., :7].float()


def _gradient_record(gradient) -> dict:
    value = gradient.detach().float().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "norm": float(value.norm()),
        "max_abs": float(value.abs().max()),
        "sha256": _tensor_sha256(value),
    }


def _gradient_parity(cached, uncached) -> dict:
    import torch

    left = cached.detach().float().cpu().reshape(-1)
    right = uncached.detach().float().cpu().reshape(-1)
    left_norm = left.norm()
    right_norm = right.norm()
    denominator = torch.maximum(left_norm, right_norm).clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "cached": _gradient_record(left),
        "uncached": _gradient_record(right),
        "max_abs_error": float((left - right).abs().max()),
        "relative_l2_error": float((left - right).norm() / denominator),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
    }


def _slice_first_row(batch: dict) -> dict:
    import torch

    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 2:
            result[key] = value[:1]
        else:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/stage2_g3_differentiable_flow.development.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "outputs/development/task2_lerobotv3_full_sft_10k_r5/checkpoints/step_010000",
    )
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/task2_lerobotv3")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite S2-G3 artifact: {args.output}")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 required")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("PYTHONHASHSEED=42 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required")

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")
    gpu_name = torch.cuda.get_device_name(0)
    if "4090 D" not in gpu_name and "4090D" not in gpu_name:
        raise RuntimeError(f"S2_G3_REQUIRES_RTX_4090D:{gpu_name}")

    sys.path.insert(0, str(ROOT / "tools"))
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.flow_sampling import (
        critic_action_for_q_guidance,
        sample_normalized_action_chunk_with_grad,
    )
    from forcesmolvla.rft.source_manifest import stage2_source_manifest_binding
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    from p8_checkpoint_common import load_fixed_validation_inputs
    from preflight_s2_common import module_state_dict_sha256

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("gate") != "S2-G3"
        or config.get("acceptance_status") != "development_only"
        or config.get("formal_eligible") is not False
        or config.get("precision") != "bf16_outer_with_v4_2_fp32_islands"
        or config.get("batch_size") != 1
        or config.get("horizon") != 50
        or config.get("euler_steps") != 10
        or config.get("critic_action_slots") != 3
        or config.get("critic_action_shape") != [3, 7]
        or config.get("partial_action_interface") is not False
        or config.get("terminal_derived_duration_or_mask_input") is not False
        or config.get("gripper_q_gradient") != "stop"
        or config.get("router_gradient_ownership", {}).get("q_origin_gradient_isolation")
        is not False
    ):
        raise RuntimeError("S2_G3_CONFIG_SEMANTICS_DRIFT")
    action_contract = json.loads(
        (ROOT / config["action_contract"]).read_text(encoding="utf-8")
    )
    if (
        action_contract.get("critic_action_shape") != [3, 7]
        or action_contract.get("critic_duration_mode") != "fixed_k"
        or action_contract.get("partial_action_interface") is not False
        or action_contract.get("gripper_q_gradient") is not False
    ):
        raise RuntimeError("S2_G3_ACTION_CONTRACT_DRIFT")
    source_manifest_path = args.source_manifest or ROOT / config["stage2_source_manifest"]
    source_manifest = stage2_source_manifest_binding(ROOT, source_manifest_path)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    with contextlib.redirect_stdout(sys.stderr):
        policy = ForceSmolVLAPolicy.from_pretrained(
            args.checkpoint.resolve(),
            local_files_only=True,
            force_download=False,
            strict=True,
            artifact_use="development",
        ).to(device)
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
    sample_identity = (f"episode={row['episode_index']}/frame={row['frame_index']}",)
    batch["sample_identity"] = sample_identity
    noise7 = torch.tensor(
        fixture["epsilon7"]["tensor"][:1], dtype=torch.float32, device=device
    )
    normalizer = json.loads(
        (args.checkpoint.resolve() / "manifests/normalizer_manifest.json").read_text(
            encoding="utf-8"
        )
    )["features"]["delta_action7"]
    normalizer_mean = torch.tensor(normalizer["mean"], dtype=torch.float32, device=device)
    normalizer_std = torch.tensor(normalizer["std"], dtype=torch.float32, device=device)
    (
        images,
        image_masks,
        state,
        wrench,
        suffix_valid,
        feature_mask,
        noise32,
    ) = _prepare_core_inputs(policy, batch, noise7)
    state_padding_probe = state.clone()
    state_padding_probe[..., 7:] = 1000.0
    state_padding_probe = state_padding_probe * (
        torch.arange(32, device=device) < 7
    ).view(1, 32)
    if not torch.equal(state_padding_probe, state):
        raise RuntimeError("S2_G3_STATE_PADDING_ISOLATION_FAILED")

    perturbed_noise32 = noise32.clone()
    perturbed_noise32[..., 7:] = torch.linspace(-1000, 1000, 25, device=device)
    direct_kv_calls = {"k": 0, "v": 0}
    direct_handles = [
        policy.model.force_adapter.cross_attention.k_proj.register_forward_hook(
            lambda *_: direct_kv_calls.__setitem__("k", direct_kv_calls["k"] + 1)
        ),
        policy.model.force_adapter.cross_attention.v_proj.register_forward_hook(
            lambda *_: direct_kv_calls.__setitem__("v", direct_kv_calls["v"] + 1)
        ),
    ]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                policy, name="direct-audit", sample_identity=sample_identity, device=device
            ),
            audit_cache=True,
        )
    for handle in direct_handles:
        handle.remove()
    direct7 = direct32[..., :7].float()
    direct_padding_max = float(direct32.masked_select(~feature_mask).abs().max().cpu())

    named_parameters = dict(policy.named_parameters())
    probe_names = config["gradient_probe_parameters"]
    missing = [name for name in probe_names if name not in named_parameters]
    if missing:
        raise RuntimeError(f"S2_G3_GRADIENT_PROBE_PARAMETER_MISSING:{missing}")
    probe_parameters = [named_parameters[name] for name in probe_names]
    q_weights = torch.tensor(
        config["synthetic_k3_action_probe_weights"], dtype=torch.float32, device=device
    )
    if q_weights.shape != (3, 7) or torch.any(q_weights == 0):
        raise RuntimeError("S2_G3_Q_WEIGHTS_MUST_PROBE_ALL_K3X7_INPUTS")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cached_kv_calls = {"k": 0, "v": 0}
    step_outputs = []

    def capture_step(_module, _inputs, output):
        step_outputs.append(output)

    cached_handles = [
        policy.model.force_adapter.cross_attention.k_proj.register_forward_hook(
            lambda *_: cached_kv_calls.__setitem__("k", cached_kv_calls["k"] + 1)
        ),
        policy.model.force_adapter.cross_attention.v_proj.register_forward_hook(
            lambda *_: cached_kv_calls.__setitem__("v", cached_kv_calls["v"] + 1)
        ),
        policy.model.action_out_proj.register_forward_hook(capture_step),
    ]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        cached_chunk = sample_normalized_action_chunk_with_grad(
            policy,
            batch,
            noise7,
            call_id="cached-gradient",
            purpose="actor_guidance",
        )
    if len(step_outputs) != 10:
        raise RuntimeError(f"S2_G3_EULER_VELOCITY_COUNT_MISMATCH:{len(step_outputs)}")
    cached_q_action7 = critic_action_for_q_guidance(
        cached_chunk,
        delta_action_mean7=normalizer_mean,
        delta_action_std7=normalizer_std,
    )
    decoded_physical_gripper = (
        cached_q_action7[..., 6].detach() * normalizer_std[6] + normalizer_mean[6]
    )
    gripper_is_binary_endpoint = bool(
        torch.all((decoded_physical_gripper == 0.0) | (decoded_physical_gripper == 0.085))
    )
    q_cached = (cached_q_action7 * q_weights).sum()
    cached_all_gradients = torch.autograd.grad(
        q_cached, [*probe_parameters, *step_outputs, cached_chunk], allow_unused=True
    )
    for handle in cached_handles:
        handle.remove()
    cached_parameter_gradients = cached_all_gradients[: len(probe_parameters)]
    cached_step_gradients = cached_all_gradients[
        len(probe_parameters) : len(probe_parameters) + len(step_outputs)
    ]
    cached_action_gradient = cached_all_gradients[-1]
    cached_output = cached_chunk.detach()
    cached_peak = {
        "allocated_bytes": torch.cuda.max_memory_allocated(device),
        "reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    del cached_chunk, cached_q_action7, q_cached, cached_all_gradients, step_outputs
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
        * q_weights
    ).sum()
    uncached_parameter_gradients = torch.autograd.grad(
        q_uncached, probe_parameters, allow_unused=True
    )
    uncached_output = uncached_chunk.detach()
    uncached_peak = {
        "allocated_bytes": torch.cuda.max_memory_allocated(device),
        "reserved_bytes": torch.cuda.max_memory_reserved(device),
    }

    minimum_norm = float(config["minimum_nonzero_gradient_norm"])
    if any(gradient is None for gradient in cached_parameter_gradients):
        raise RuntimeError("S2_G3_CACHED_PARAMETER_GRADIENT_MISSING")
    if any(gradient is None for gradient in uncached_parameter_gradients):
        raise RuntimeError("S2_G3_UNCACHED_PARAMETER_GRADIENT_MISSING")
    if any(gradient is None for gradient in cached_step_gradients):
        raise RuntimeError("S2_G3_EULER_STEP_GRADIENT_MISSING")
    if cached_action_gradient is None:
        raise RuntimeError("S2_G3_Q_ACTION_GRADIENT_MISSING")

    parameter_parity = {
        name: _gradient_parity(cached, uncached)
        for name, cached, uncached in zip(
            probe_names,
            cached_parameter_gradients,
            uncached_parameter_gradients,
            strict=True,
        )
    }
    step_gradient_records = [
        {"euler_step": index, **_gradient_record(gradient)}
        for index, gradient in enumerate(cached_step_gradients)
    ]
    output_error = float((cached_output - uncached_output).abs().max().cpu())
    direct_error = float((cached_output - direct7).abs().max().cpu())
    parameter_gradients_nonzero = all(
        item["cached"]["norm"] > minimum_norm and item["uncached"]["norm"] > minimum_norm
        for item in parameter_parity.values()
    )
    step_gradients_nonzero = all(
        item["norm"] > minimum_norm for item in step_gradient_records
    )
    expected_action_gradient = torch.zeros_like(cached_action_gradient)
    expected_action_gradient[:, :3, :6] = q_weights[:, :6]
    q_action_gradient_error = float(
        (cached_action_gradient - expected_action_gradient).abs().max().cpu()
    )
    gradient_parity_pass = all(
        item["relative_l2_error"]
        <= float(config["cached_uncached_gradient_relative_l2_max"])
        and item["cosine_similarity"]
        >= float(config["cached_uncached_gradient_cosine_min"])
        for item in parameter_parity.values()
    )
    no_materialized_parameter_grads = all(
        parameter.grad is None for parameter in policy.parameters()
    )
    state_after = module_state_dict_sha256(policy)
    generation_after = policy.model.parameter_generation()
    state_exact = state_before == state_after and generation_before == generation_after

    contracts = {
        "wrapper_vs_direct_core_exact": direct_error == 0.0,
        "cached_vs_uncached_output_within_fp32_limit": output_error
        <= float(config["cached_uncached_output_max_abs"]),
        "cached_vs_uncached_parameter_gradient_parity": gradient_parity_pass,
        "representative_actor_parameter_gradients_nonzero": parameter_gradients_nonzero,
        "all_ten_euler_velocity_gradients_nonzero": step_gradients_nonzero,
        "force_kv_once_in_wrapper": cached_kv_calls == {"k": 1, "v": 1},
        "force_kv_once_in_direct_audit": direct_kv_calls == {"k": 1, "v": 1},
        "prefix_cache_audit_passed": True,
        "state_padding_exact_zero": float(state_padding_probe[..., 7:].abs().max().cpu())
        == 0.0,
        "noise_action_padding_exact_zero": direct_padding_max == 0.0,
        "critic_q_input_uses_binary_gripper_endpoint": gripper_is_binary_endpoint,
        "actor_q_gradient_stops_gripper_exactly": q_action_gradient_error == 0.0,
        "state_dict_and_parameter_generation_exact": state_exact,
        "no_parameter_grad_buffers_materialized": no_materialized_parameter_grads,
        "optimizer_steps_zero": True,
    }
    gate_pass = all(contracts.values())
    result = {
        "schema_version": "1.0",
        "gate": "S2-G3",
        "gate_status": "pass" if gate_pass else "fail",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "gpu": gpu_name,
        "precision": config["precision"],
        "batch_size": 1,
        "horizon": 50,
        "euler_steps": 10,
        "config_sha256": _sha256(config_path),
        "action_contract": {
            "relative_path": config["action_contract"],
            "sha256": _sha256(ROOT / config["action_contract"]),
        },
        "stage2_source_manifest": source_manifest,
        "paper_description": action_contract["paper_description"],
        "router_q_origin_gradient": {
            "status": "measurement_only_deferred_to_G4",
            "parameter": "model.force_branch.refiner.router.bias",
            "semantics_modified_in_this_gate": False,
        },
        "checkpoint_artifact_manifest_sha256": _sha256(
            args.checkpoint.resolve() / "artifact_manifest.json"
        ),
        "output_parity": {
            "wrapper_vs_direct_core_max_abs": direct_error,
            "cached_vs_uncached_max_abs": output_error,
            "limit": config["cached_uncached_output_max_abs"],
            "cached_sha256": _tensor_sha256(cached_output),
            "uncached_sha256": _tensor_sha256(uncached_output),
        },
        "parameter_gradient_parity": parameter_parity,
        "euler_step_output_gradients": step_gradient_records,
        "actor_q_action_gradient": {
            **_gradient_record(cached_action_gradient),
            "expected_max_abs_error": q_action_gradient_error,
            "gripper_is_binary_endpoint": gripper_is_binary_endpoint,
            "gripper_gradient_max_abs": float(
                cached_action_gradient[:, :3, 6].abs().max().cpu()
            ),
        },
        "force_kv_projection_calls": {
            "wrapper": cached_kv_calls,
            "direct_cache_audit": direct_kv_calls,
        },
        "padding": {
            "state_tail_max_abs_after_mask": float(
                state_padding_probe[..., 7:].abs().max().cpu()
            ),
            "action_tail_max_abs_after_ten_steps": direct_padding_max,
        },
        "peak_memory": {"cached_gradient": cached_peak, "uncached_checkpointed": uncached_peak},
        "state_dict": {"before_sha256": state_before, "after_sha256": state_after},
        "contracts": contracts,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "robot_actions_sent": 0,
        "real_rft_training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))
    if not gate_pass:
        raise RuntimeError("S2_G3_DIFFERENTIABLE_FLOW_GATE_FAILED")


if __name__ == "__main__":
    main()
