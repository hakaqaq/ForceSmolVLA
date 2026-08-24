#!/usr/bin/env python3
"""Development P8 real-Force full/prefill and 10-step parity gate."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import sys
from pathlib import Path


P8_EXACT_CONTRACTS = [
    "prefix_layout",
    "prefix_mask",
    "prefix_physical_length",
    "invalid_suffix_velocity_zero",
    "zero_init_force_shared_fp32_output_parity",
    "state_padding_7d_25d_isolation",
    "action_noise_padding_7d_25d_isolation",
    "invalid_horizon_tail_isolation",
    "cache_append_crop_restoration",
    "cache_snapshot_unchanged",
    "force_kv_once_per_10_step_chunk",
]


def load_p8_threshold(path: Path, precision: str) -> dict:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "schema_version": "1.0",
        "mode": "development_only",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "production_allowed": False,
        "operator_overrides_allowed": False,
        "approval_status": "P8_development_only_approved",
        "detached_signature": None,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise RuntimeError("P8_PARITY_ACCEPTANCE_CONFIG_CONTRACT_DRIFT")
    formal = payload.get("formal_thresholds", {}).get("P8")
    if formal != {
        "fp32": {"atol": None, "rtol": None, "approval_status": "unapproved"},
        "bf16": {
            "prefix_hidden_atol": None,
            "velocity_cache_atol": None,
            "rtol": None,
            "approval_status": "unapproved",
        },
    }:
        raise RuntimeError("P8_FORMAL_THRESHOLDS_MUST_REMAIN_NULL_UNAPPROVED")
    p8 = payload.get("thresholds", {}).get("P8", {})
    if p8.get("structural_contracts_exact") != P8_EXACT_CONTRACTS:
        raise RuntimeError("P8_EXACT_CONTRACT_SCOPE_DRIFT")
    comparisons = p8.get("comparisons")
    if (
        comparisons is None
        or comparisons.get("prefix_hidden", {}).get("observed_bf16_max_abs") != 0.25
        or comparisons.get("velocity_cache", {}).get("observed_bf16_max_abs")
        != 0.06651902198791504
    ):
        raise RuntimeError("P8_COMPARISON_PROVENANCE_DRIFT")
    values = p8.get(precision)
    if precision == "fp32" and values == {
        "atol": 1e-5,
        "rtol": 0.0,
        "approval_scope": "P8_development_only",
    }:
        prefix_atol = velocity_atol = values["atol"]
        atol = values["atol"]
    elif precision == "bf16" and values == {
        "prefix_hidden_atol": 0.3,
        "velocity_cache_atol": 0.1,
        "rtol": 0.0,
        "approval_scope": "P8_development_only",
    }:
        prefix_atol = values["prefix_hidden_atol"]
        velocity_atol = values["velocity_cache_atol"]
        atol = None
    else:
        raise RuntimeError("P8_APPROVED_THRESHOLD_DRIFT")
    return {
        "atol": atol,
        "prefix_hidden_atol": prefix_atol,
        "velocity_cache_atol": velocity_atol,
        "rtol": values["rtol"],
        "config_id": payload["config_id"],
        "config_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def deny_network() -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required")

    def denied(self, address):  # noqa: ANN001
        raise RuntimeError(f"NETWORK_ACCESS_FORBIDDEN: {address}")

    socket.socket.connect = denied


def maximum_error(first, second) -> float:
    return float((first.float() - second.float()).abs().max().detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--acceptance-config",
        type=Path,
        default=Path(__file__).parents[1]
        / "configs/p8_parity_acceptance.development.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--measurement-only",
        action="store_true",
        help="record P8 numerics without accepting an unapproved threshold",
    )
    parser.add_argument(
        "--source-binding",
        type=Path,
        help="required hash-bound P8 source binding for an accepting gate run",
    )
    args = parser.parse_args()
    deny_network()

    import torch

    from forcesmolvla.checkpoint import load_offline_base_policy
    from forcesmolvla.configuration_forcesmolvla import FORCE_TOKEN_MOE
    from forcesmolvla.force_token import (
        PreparedForceContextBinding,
        fp32_action_projection,
    )
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from preflight_p5_dense_compute_gpu import _sha256, _validate_source_binding

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE")
    root = args.project_root.resolve()
    if not args.measurement_only and args.source_binding is None:
        raise RuntimeError("P8_ACCEPTING_GATE_REQUIRES_SOURCE_BINDING")
    source_binding_sha256 = None
    if args.source_binding is not None:
        source_binding_path = args.source_binding.resolve()
        source_binding = json.loads(source_binding_path.read_text(encoding="utf-8"))
        if source_binding.get("stage") != "P8":
            raise RuntimeError("P8_PARITY_SOURCE_BINDING_STAGE_DRIFT")
        _validate_source_binding(root, source_binding)
        source_binding_sha256 = _sha256(source_binding_path)
    threshold = load_p8_threshold(args.acceptance_config, args.precision)
    torch.manual_seed(4107)
    with contextlib.redirect_stdout(sys.stderr):
        policy, base_report = load_offline_base_policy(
            root / "assets" / "base_checkpoint",
            root / "assets" / "smolvlm_constructor",
            device=args.device,
            force_variant=FORCE_TOKEN_MOE,
        )
    model = policy.model
    device = torch.device(args.device)
    if args.precision == "fp32":
        policy.float()
    images = [
        torch.linspace(0, 1, 2 * 3 * 512 * 512, device=device).reshape(2, 3, 512, 512),
        torch.linspace(1, 0, 2 * 3 * 512 * 512, device=device).reshape(2, 3, 512, 512),
    ]
    image_masks = [torch.ones(2, dtype=torch.bool, device=device) for _ in range(2)]
    tokens = torch.arange(48, device=device).view(1, 48).expand(2, -1)
    language_mask = torch.tensor(
        [[True] * 48, [True] * 17 + [False] * 31], dtype=torch.bool, device=device
    )
    state = torch.zeros(2, 32, device=device)
    state[:, :7] = torch.tensor(
        [[0.5, -0.1, 0.2, 0.1, -0.2, 0.3, 0.05],
         [0.4, 0.2, 0.1, -0.2, 0.1, -0.3, 0.04]],
        device=device,
    )
    x_t = torch.zeros(2, 50, 32, device=device)
    x_t[:, :, :7] = torch.randn(2, 50, 7, device=device)
    suffix_valid = torch.tensor(
        [[True] * 50, [True] * 47 + [False] * 3], dtype=torch.bool, device=device
    )
    feature_mask = suffix_valid.unsqueeze(-1) & (
        torch.arange(32, device=device).view(1, 1, 32) < 7
    )
    timestep = torch.tensor([0.25, 0.75], dtype=torch.float32, device=device)
    wrench = torch.tensor(
        [[0.1, -0.2, 0.3, 0.01, -0.02, 0.03], [-0.3, 0.2, -0.1, -0.03, 0.02, -0.01]],
        dtype=torch.float32,
        device=device,
    )

    def binding(label: str, batch: int, tensor: torch.Tensor) -> PreparedForceContextBinding:
        return PreparedForceContextBinding(
            chunk_id=tuple(f"p8-{label}-chunk-{index}" for index in range(batch)),
            sample_id=tuple(f"p8-{label}-sample-{index}" for index in range(batch)),
            context_generation=0,
            model_generation=model.parameter_generation(),
            device=tensor.device,
            dtype=torch.float32,
        )

    autocast = (
        torch.autocast(device_type=args.device, dtype=torch.bfloat16)
        if args.precision == "bf16"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
        context = model.encode_prefix(images, image_masks, tokens, language_mask, state)
        expected_prefix_valid = torch.cat(
            [
                torch.ones(2, 128, dtype=torch.bool, device=device),
                language_mask,
                torch.ones(2, 1, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        if not torch.equal(context.prefix_valid_mask, expected_prefix_valid):
            raise RuntimeError("REAL_EMBED_PREFIX_LAYOUT_MISMATCH")
        force_context = model.build_force_context(
            context.prefix_out, context.prefix_valid_mask, wrench
        )
        batch_binding = binding("batch", 2, wrench)
        prepared_force_context = model.force_adapter.cross_attention.prepare(
            force_context, binding=batch_binding
        )
        cached_batch = model.velocity_cached(
            context, x_t, timestep,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            force_context=prepared_force_context,
            force_context_binding=batch_binding,
            audit_cache=True,
        )
        full_batch = model.velocity_full(
            images, image_masks, tokens, language_mask, state, x_t, timestep,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            wrench=wrench,
        )

        prefix_embs, prefix_valid, prefix_attention = model.embed_prefix(
            images, image_masks, tokens, language_mask, state=state
        )
        suffix_embs, suffix_pad, suffix_attention = model.embed_suffix(
            x_t, timestep, suffix_valid_mask=suffix_valid
        )
        full_valid = torch.cat([prefix_valid, suffix_pad], dim=1)
        full_attention = torch.cat([prefix_attention, suffix_attention], dim=1)
        full_position_ids = torch.cumsum(full_valid, dim=1) - 1
        (full_prefix_out, _full_suffix_out), _ = model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(full_valid, full_attention),
            position_ids=full_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        zero_init_force_context = model.build_force_context(
            full_prefix_out, prefix_valid, wrench
        )
        zero_init_force_projection = model.project_velocity(
            _full_suffix_out,
            x_t,
            timestep,
            zero_init_force_context,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
        )
        zero_init_bare_projection = fp32_action_projection(
            model.action_out_proj,
            _full_suffix_out[:, -model.config.chunk_size :].float(),
            feature_mask,
        )

        padded_action_noise = x_t.clone()
        padded_action_noise[..., 7:] = torch.linspace(
            -1000,
            1000,
            25,
            device=device,
            dtype=padded_action_noise.dtype,
        )
        padded_action_velocity = model.velocity_full(
            images,
            image_masks,
            tokens,
            language_mask,
            state,
            padded_action_noise,
            timestep,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            wrench=wrench,
        )
        invalid_tail_noise = x_t.clone()
        invalid_tail_noise[1, 47:, :7] = 1000
        invalid_tail_velocity = model.velocity_full(
            images,
            image_masks,
            tokens,
            language_mask,
            state,
            invalid_tail_noise,
            timestep,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            wrench=wrench,
        )
        state_padding_perturbed = state.clone()
        state_padding_perturbed[:, 7:] = 1000
        state_feature_mask = (
            torch.arange(32, device=device).view(1, 32) < 7
        )
        state_after_policy_mask = state_padding_perturbed * state_feature_mask.to(
            dtype=state_padding_perturbed.dtype
        )
        state_padding_velocity = model.velocity_full(
            images,
            image_masks,
            tokens,
            language_mask,
            state_after_policy_mask,
            x_t,
            timestep,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            wrench=wrench,
        )
        full_single = []
        cached_single = []
        for index in range(2):
            one_images = [image[index : index + 1] for image in images]
            one_image_masks = [mask[index : index + 1] for mask in image_masks]
            one_context = model.encode_prefix(
                one_images,
                one_image_masks,
                tokens[index : index + 1],
                language_mask[index : index + 1],
                state[index : index + 1],
            )
            one_force_context = model.build_force_context(
                one_context.prefix_out,
                one_context.prefix_valid_mask,
                wrench[index : index + 1],
            )
            one_binding = binding(f"single-{index}", 1, wrench[index : index + 1])
            one_prepared = model.force_adapter.cross_attention.prepare(
                one_force_context, binding=one_binding
            )
            cached_single.append(model.velocity_cached(
                one_context,
                x_t[index : index + 1],
                timestep[index : index + 1],
                action_feature_mask=feature_mask[index : index + 1],
                suffix_valid_mask=suffix_valid[index : index + 1],
                force_context=one_prepared,
                force_context_binding=one_binding,
                audit_cache=True,
            ))
            full_single.append(model.velocity_full(
                one_images,
                one_image_masks,
                tokens[index : index + 1],
                language_mask[index : index + 1],
                state[index : index + 1],
                x_t[index : index + 1],
                timestep[index : index + 1],
                action_feature_mask=feature_mask[index : index + 1],
                suffix_valid_mask=suffix_valid[index : index + 1],
                wrench=wrench[index : index + 1],
            ))

        kv_calls = {"k": 0, "v": 0}
        handles = [
            model.force_adapter.cross_attention.k_proj.register_forward_hook(
                lambda *_args: kv_calls.__setitem__("k", kv_calls["k"] + 1)
            ),
            model.force_adapter.cross_attention.v_proj.register_forward_hook(
                lambda *_args: kv_calls.__setitem__("v", kv_calls["v"] + 1)
            ),
        ]
        cached_chunk = model.sample_actions_masked(
            images,
            image_masks,
            tokens,
            language_mask,
            state,
            x_t,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            wrench=wrench,
            force_context_binding=binding("ten-step", 2, wrench),
            audit_cache=True,
        )
        for handle in handles:
            handle.remove()
        uncached_chunk = x_t * feature_mask
        dt = -1.0 / model.config.num_steps
        for step in range(model.config.num_steps):
            step_time = torch.full(
                (2,), 1.0 + step * dt, dtype=torch.float32, device=device
            )
            uncached_velocity = model.velocity_full(
                images,
                image_masks,
                tokens,
                language_mask,
                state,
                uncached_chunk,
                step_time,
                action_feature_mask=feature_mask,
                suffix_valid_mask=suffix_valid,
                wrench=wrench,
            )
            uncached_chunk = (uncached_chunk + dt * uncached_velocity) * feature_mask
    cached_single = torch.cat(cached_single)
    full_single = torch.cat(full_single)
    invalid_max = float(
        cached_batch.masked_select(~feature_mask).abs().max().detach().cpu()
    )
    velocity_scale = float(full_single.abs().max().detach().cpu())
    prefix_error = maximum_error(
        context.prefix_out[context.prefix_valid_mask],
        full_prefix_out[context.prefix_valid_mask],
    )
    errors = {
        "prefill_prefix_vs_full_prefix_valid_tokens": prefix_error,
        "cached_batch_vs_full_batch": maximum_error(cached_batch, full_batch),
        "cached_batch_vs_cached_single": maximum_error(cached_batch, cached_single),
        "full_batch_vs_full_single": maximum_error(full_batch, full_single),
        "cached_single_vs_full_single": maximum_error(cached_single, full_single),
        "cached_10_step_vs_uncached_10_step": maximum_error(cached_chunk, uncached_chunk),
        "invalid_velocity": invalid_max,
    }
    exact_errors = {
        "zero_init_force_vs_shared_fp32_action_projection": maximum_error(
            zero_init_force_projection, zero_init_bare_projection
        ),
        "action_noise_padding_7d_25d_isolation": maximum_error(
            padded_action_velocity, full_batch
        ),
        "invalid_horizon_tail_isolation": maximum_error(
            invalid_tail_velocity, full_batch
        ),
        "state_padding_7d_25d_isolation_after_policy_mask": maximum_error(
            state_padding_velocity, full_batch
        ),
    }
    reference_scale = max(
        velocity_scale,
        float(full_prefix_out.abs().max().detach().cpu()),
        float(uncached_chunk.abs().max().detach().cpu()),
    )
    prefix_numerical_limit = (
        threshold["prefix_hidden_atol"] + threshold["rtol"] * reference_scale
    )
    velocity_numerical_limit = (
        threshold["velocity_cache_atol"] + threshold["rtol"] * reference_scale
    )
    prefix_numerical_pass = (
        errors["prefill_prefix_vs_full_prefix_valid_tokens"]
        <= prefix_numerical_limit
    )
    velocity_numerical_pass = all(
        value <= velocity_numerical_limit
        for name, value in errors.items()
        if name != "prefill_prefix_vs_full_prefix_valid_tokens"
    )
    numerical_pass = prefix_numerical_pass and velocity_numerical_pass
    structural_exact = {
        "prefix_layout": True,
        "prefix_mask": True,
        "prefix_physical_length": context.layout.physical_length == 177,
        "invalid_suffix_velocity_zero": invalid_max == 0.0,
        "zero_init_force_shared_fp32_output_parity": exact_errors[
            "zero_init_force_vs_shared_fp32_action_projection"
        ]
        == 0.0,
        "state_padding_7d_25d_isolation": exact_errors[
            "state_padding_7d_25d_isolation_after_policy_mask"
        ]
        == 0.0,
        "action_noise_padding_7d_25d_isolation": exact_errors[
            "action_noise_padding_7d_25d_isolation"
        ]
        == 0.0,
        "invalid_horizon_tail_isolation": exact_errors[
            "invalid_horizon_tail_isolation"
        ]
        == 0.0,
        "cache_append_crop_restoration": True,
        "cache_snapshot_unchanged": True,
        "force_kv_once_per_10_step_chunk": kv_calls == {"k": 1, "v": 1},
    }
    if list(structural_exact) != P8_EXACT_CONTRACTS:
        raise RuntimeError("P8_RUNTIME_EXACT_CONTRACT_SCOPE_DRIFT")
    gate_pass = numerical_pass and all(structural_exact.values())
    status = "measurement_only" if args.measurement_only else ("pass" if gate_pass else "fail")
    result = {
        "status": status,
        "gate_status": "not_evaluated" if args.measurement_only else status,
        "gate": "P8_force_parity",
        "artifact_status": "development_only",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "source_binding_sha256": source_binding_sha256,
        "force_variant": FORCE_TOKEN_MOE,
        "device": args.device,
        "precision": args.precision,
        "dtype": (
            "bfloat16 autocast with fp32 velocity output"
            if args.precision == "bf16"
            else "float32 model and inputs"
        ),
        "seed": 4107,
        "batch_size": 2,
        "language_valid_lengths": language_mask.sum(dim=1).tolist(),
        "prefix_physical_lengths": [context.layout.physical_length] * 2,
        "suffix_valid_lengths": suffix_valid.sum(dim=1).tolist(),
        "max_abs_error": errors,
        "exact_contract_max_abs_error": exact_errors,
        "reference_max_abs_velocity": velocity_scale,
        "reference_scale_for_gate": reference_scale,
        "tolerance": {
            "atol": threshold["atol"],
            "prefix_hidden_atol": threshold["prefix_hidden_atol"],
            "velocity_cache_atol": threshold["velocity_cache_atol"],
            "rtol": threshold["rtol"],
            "source": "versioned_hash_bound_acceptance_config",
            "config_id": threshold["config_id"],
            "acceptance_config_sha256": threshold["config_sha256"],
            "approval_status": "unapproved_measurement_only" if args.measurement_only else "approved",
        },
        "numerical_limit": {
            "prefix_hidden": prefix_numerical_limit,
            "velocity_cache": velocity_numerical_limit,
        },
        "force_kv_projection_calls_for_10_step_chunk": kv_calls,
        "cache_unchanged_after_each_call": True,
        "structural_contracts_exact": structural_exact,
        "base_loaded_tensors": base_report.loaded_tensor_count,
        "missing_keys": list(base_report.missing_keys),
        "unexpected_keys": list(base_report.unexpected_keys),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite P8 parity report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    if not args.measurement_only and not gate_pass:
        raise RuntimeError("PREFIX_PARITY_GATE_FAILED")


if __name__ == "__main__":
    main()
