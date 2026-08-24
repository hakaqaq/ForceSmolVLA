#!/usr/bin/env python3
"""Development P4 bare SmolVLA topology, prefix and cache parity gate."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def _deny_network() -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required")

    def denied(self, address):  # noqa: ANN001
        raise RuntimeError(f"NETWORK_ACCESS_FORBIDDEN: {address}")

    socket.socket.connect = denied


def _maximum_error(first, second) -> float:
    return float((first.float() - second.float()).abs().max().detach().cpu())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(_sha256(path).encode() + b"\n")
    return digest.hexdigest()


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_binding(root: Path, acceptance_config: Path) -> dict:
    project_files = (
        "ForceSmolVLA_Implementation_Spec_v4_2.md",
        "configs/parity_acceptance.development.json",
        "src/forcesmolvla/acceptance.py",
        "src/forcesmolvla/checkpoint.py",
        "src/forcesmolvla/configuration_forcesmolvla.py",
        "src/forcesmolvla/force_token.py",
        "src/forcesmolvla/modeling_forcesmolvla.py",
        "src/forcesmolvla/prefix.py",
        "tools/preflight_p4_bare_parity.py",
    )
    vendor_root = root / "vendor/lerobot"
    vendor_files = (
        "src/lerobot/policies/smolvla/configuration_smolvla.py",
        "src/lerobot/policies/smolvla/modeling_smolvla.py",
        "src/lerobot/policies/smolvla/smolvlm_with_expert.py",
    )
    commit = subprocess.run(
        ["git", "-C", str(vendor_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(vendor_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("P4_LEROBOT_VENDOR_DIRTY_WORKTREE")
    if acceptance_config.resolve() != (root / project_files[1]).resolve():
        raise RuntimeError("P4_NONCANONICAL_ACCEPTANCE_CONFIG_FORBIDDEN")
    return {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "project_file_sha256": {
            relative: _sha256(root / relative) for relative in project_files
        },
        "lerobot_commit": commit,
        "lerobot_dirty_worktree": False,
        "lerobot_file_sha256": {
            relative: _sha256(vendor_root / relative) for relative in vendor_files
        },
        "base_checkpoint_config_sha256": _sha256(
            root / "assets/base_checkpoint/config.json"
        ),
        "base_checkpoint_model_sha256": _sha256(
            root / "assets/base_checkpoint/model.safetensors"
        ),
        "constructor_assets_tree_sha256": _tree_sha256(
            root / "assets/smolvlm_constructor"
        ),
        "detached_signature": None,
        "approval": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--acceptance-config",
        type=Path,
        default=Path(__file__).parents[1] / "configs/parity_acceptance.development.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _deny_network()

    import torch

    from forcesmolvla.acceptance import load_development_parity_threshold
    from forcesmolvla.checkpoint import load_offline_base_policy
    from forcesmolvla.configuration_forcesmolvla import SMOLVLA_CARTESIAN7D
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE")
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    root = args.project_root.resolve()
    source_binding = _source_binding(root, args.acceptance_config)
    threshold = load_development_parity_threshold(
        args.acceptance_config, gate="P4", precision=args.precision
    )
    torch.manual_seed(4107)
    with contextlib.redirect_stdout(sys.stderr):
        policy, base_report = load_offline_base_policy(
            root / "assets/base_checkpoint",
            root / "assets/smolvlm_constructor",
            device=args.device,
            force_variant=SMOLVLA_CARTESIAN7D,
        )
    policy.eval()
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
        [
            [0.5, -0.1, 0.2, 0.1, -0.2, 0.3, 0.05],
            [0.4, 0.2, 0.1, -0.2, 0.1, -0.3, 0.04],
        ],
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
            raise RuntimeError("P4_REAL_EMBED_PREFIX_LAYOUT_MISMATCH")

        cached_batch = model.velocity_cached(
            context,
            x_t,
            timestep,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            audit_cache=True,
        )
        full_batch = model.velocity_full(
            images,
            image_masks,
            tokens,
            language_mask,
            state,
            x_t,
            timestep,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
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
        (full_prefix_out, _), _ = model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(full_valid, full_attention),
            position_ids=full_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        cached_single = []
        full_single = []
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
            cached_single.append(
                model.velocity_cached(
                    one_context,
                    x_t[index : index + 1],
                    timestep[index : index + 1],
                    action_feature_mask=feature_mask[index : index + 1],
                    suffix_valid_mask=suffix_valid[index : index + 1],
                    audit_cache=True,
                )
            )
            full_single.append(
                model.velocity_full(
                    one_images,
                    one_image_masks,
                    tokens[index : index + 1],
                    language_mask[index : index + 1],
                    state[index : index + 1],
                    x_t[index : index + 1],
                    timestep[index : index + 1],
                    action_feature_mask=feature_mask[index : index + 1],
                    suffix_valid_mask=suffix_valid[index : index + 1],
                )
            )

        cached_chunk = model.sample_actions_masked(
            images,
            image_masks,
            tokens,
            language_mask,
            state,
            x_t,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            audit_cache=True,
        )
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
            )
            uncached_chunk = (uncached_chunk + dt * uncached_velocity) * feature_mask

    cached_single = torch.cat(cached_single)
    full_single = torch.cat(full_single)
    prefix_error = _maximum_error(
        context.prefix_out[context.prefix_valid_mask],
        full_prefix_out[context.prefix_valid_mask],
    )
    invalid_velocity = float(cached_batch.masked_select(~feature_mask).abs().max().cpu())
    errors = {
        "prefill_prefix_vs_full_prefix_valid_tokens": prefix_error,
        "cached_batch_vs_full_batch": _maximum_error(cached_batch, full_batch),
        "cached_batch_vs_cached_single": _maximum_error(cached_batch, cached_single),
        "full_batch_vs_full_single": _maximum_error(full_batch, full_single),
        "cached_single_vs_full_single": _maximum_error(cached_single, full_single),
        "cached_10_step_vs_uncached_10_step": _maximum_error(cached_chunk, uncached_chunk),
        "invalid_velocity": invalid_velocity,
    }
    prefix_reference_scale = float(
        full_prefix_out[context.prefix_valid_mask].abs().max().cpu()
    )
    velocity_reference_scale = max(
        float(full_single.abs().max().cpu()),
        float(uncached_chunk.abs().max().cpu()),
    )
    prefix_hidden_limit = (
        threshold.prefix_hidden_atol + threshold.rtol * prefix_reference_scale
    )
    velocity_cache_limit = (
        threshold.velocity_cache_atol + threshold.rtol * velocity_reference_scale
    )
    velocity_error_names = (
        "cached_batch_vs_full_batch",
        "cached_batch_vs_cached_single",
        "full_batch_vs_full_single",
        "cached_single_vs_full_single",
        "cached_10_step_vs_uncached_10_step",
    )
    prefix_hidden_pass = prefix_error <= prefix_hidden_limit
    velocity_cache_pass = all(
        errors[name] <= velocity_cache_limit for name in velocity_error_names
    )
    structural_exact = {
        "prefix_layout": True,
        "prefix_mask": True,
        "prefix_physical_length": context.layout.physical_length == 177,
        "invalid_suffix_velocity_zero": invalid_velocity == 0.0,
        "cache_append_crop_restoration": True,
        "cache_snapshot_unchanged": True,
    }
    gate_pass = (
        prefix_hidden_pass
        and velocity_cache_pass
        and all(structural_exact.values())
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
        peak_memory = {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    else:
        peak_memory = {"allocated_bytes": None, "reserved_bytes": None}
    result = {
        "schema_version": "1.0",
        "gate": "P4_bare_smolvla_parity",
        "gate_status": "pass" if gate_pass else "fail",
        "status": "pass" if gate_pass else "fail",
        "acceptance_status": "development_only",
        "artifact_status": "development_only",
        "formal_eligible": False,
        "force_variant": SMOLVLA_CARTESIAN7D,
        "device": args.device,
        "precision": args.precision,
        "seed": 4107,
        "batch_size": 2,
        "topology": {
            "attention_mode": model.config.attention_mode,
            "self_attn_every_n_layers": model.config.self_attn_every_n_layers,
            "num_vlm_layers": model.config.num_vlm_layers,
            "num_expert_layers": model.config.num_expert_layers,
            "expert_width_multiplier": model.config.expert_width_multiplier,
            "use_cache": model.config.use_cache,
            "add_image_special_tokens": model.config.add_image_special_tokens,
            "resize_imgs_with_padding": list(model.config.resize_imgs_with_padding),
        },
        "prefix_physical_lengths": [context.layout.physical_length] * 2,
        "prefix_valid_lengths": context.prefix_valid_mask.sum(dim=1).tolist(),
        "language_valid_lengths": language_mask.sum(dim=1).tolist(),
        "suffix_valid_lengths": suffix_valid.sum(dim=1).tolist(),
        "max_abs_error": errors,
        "comparison_groups": {
            "prefix_hidden": {
                "applies_to": ["prefill_prefix_vs_full_prefix_valid_tokens"],
                "dtype": args.precision,
                "direction": "prefill_valid_prefix_hidden_vs_full_valid_prefix_hidden",
                "reference_scale": prefix_reference_scale,
                "limit": prefix_hidden_limit,
                "pass": prefix_hidden_pass,
            },
            "velocity_cache": {
                "applies_to": list(velocity_error_names),
                "dtype": args.precision,
                "direction": "named_lhs_vs_named_rhs",
                "reference_scale": velocity_reference_scale,
                "limit": velocity_cache_limit,
                "pass": velocity_cache_pass,
            },
        },
        "structural_contracts_exact": structural_exact,
        "tolerance": {
            "prefix_hidden_atol": threshold.prefix_hidden_atol,
            "velocity_cache_atol": threshold.velocity_cache_atol,
            "rtol": threshold.rtol,
            "source": "versioned_hash_bound_acceptance_config",
            "config_id": threshold.config_id,
            "acceptance_config_sha256": threshold.config_sha256,
        },
        "source_binding": source_binding,
        "source_binding_sha256": _canonical_sha256(source_binding),
        "cache_unchanged_after_each_call": True,
        "debug_cache_comparison_hot_path": False,
        "peak_cuda_memory": peak_memory,
        "base_loaded_tensors": base_report.loaded_tensor_count,
        "missing_keys": list(base_report.missing_keys),
        "unexpected_keys": list(base_report.unexpected_keys),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite P4 report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    if not gate_pass:
        raise RuntimeError("P4_BARE_PREFIX_PARITY_GATE_FAILED")


if __name__ == "__main__":
    main()
