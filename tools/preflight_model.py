#!/usr/bin/env python3
"""Strict offline P0 constructor, base-load, prefix-layout, and RTC preflight."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
from pathlib import Path


def require_offline_environment() -> None:
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    bad = {name: os.environ.get(name) for name in required if os.environ.get(name) != "1"}
    if bad:
        raise RuntimeError(f"OFFLINE_ENV_REQUIRED: {bad}")

    original_connect = socket.socket.connect

    def deny_connect(self, address):  # noqa: ANN001
        raise RuntimeError(f"NETWORK_ACCESS_FORBIDDEN: {address}")

    socket.socket.connect = deny_connect
    globals()["_ORIGINAL_SOCKET_CONNECT"] = original_connect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()

    require_offline_environment()

    import torch

    from forcesmolvla.checkpoint import load_offline_base_policy

    with contextlib.redirect_stdout(sys.stderr):
        policy, load_report = load_offline_base_policy(
            root / "assets" / "base_checkpoint",
            root / "assets" / "smolvlm_constructor",
            device="cpu",
        )
    policy.train()
    frozen_parameter_names = [
        name for name, parameter in policy.named_parameters() if not parameter.requires_grad
    ]
    if frozen_parameter_names:
        raise RuntimeError(f"OFFLINE_FULL_FINETUNE_HAS_FROZEN_PARAMETERS: {frozen_parameter_names}")
    if not policy.model.vlm_with_expert.vlm.training:
        raise RuntimeError("OFFLINE_FULL_FINETUNE_VLM_NOT_IN_TRAIN_MODE")
    trainable_parameters = sum(parameter.numel() for parameter in policy.parameters())
    policy.eval()
    model = policy.model
    images = [torch.zeros(2, 3, 512, 512), torch.zeros(2, 3, 512, 512)]
    image_masks = [torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)]
    language_tokens = torch.zeros(2, 48, dtype=torch.long)
    language_mask = torch.tensor(
        [[True] * 48, [True] * 17 + [False] * 31], dtype=torch.bool
    )
    state32 = torch.zeros(2, 32)
    with torch.inference_mode():
        prefix, prefix_valid, prefix_attention = model.embed_prefix(
            images, image_masks, language_tokens, language_mask, state32
        )
        image_token_count = model.vlm_with_expert.embed_image(images[0]).shape[1]

    tokenizer = model.vlm_with_expert.processor.tokenizer
    result = {
        "status": "pass",
        "offline_socket_guard": True,
        "base_load": load_report.to_dict(),
        "resolved": {
            "policy_model_class": type(model).__name__,
            "d_vlm": model.vlm_with_expert.config.text_config.hidden_size,
            "d_expert": model.vlm_with_expert.expert_hidden_size,
            "d_action": policy.config.max_action_dim,
            "image_tokens_per_camera": image_token_count,
            "prefix_shape": list(prefix.shape),
            "prefix_valid_shape": list(prefix_valid.shape),
            "prefix_attention_shape": list(prefix_attention.shape),
            "prefix_valid_counts": prefix_valid.sum(dim=1).tolist(),
            "n_prefix_physical": prefix.shape[1],
            "tokenizer_max_length": policy.config.tokenizer_max_length,
            "tokenizer_padding_side": tokenizer.padding_side,
            "tokenizer_truncation_side": tokenizer.truncation_side,
            "chunk_size": policy.config.chunk_size,
            "n_action_steps": policy.config.n_action_steps,
            "num_steps": policy.config.num_steps,
            "training_stage": policy.config.training_stage,
            "total_parameters": trainable_parameters,
            "trainable_parameters": trainable_parameters,
            "frozen_parameter_count": 0,
            "vlm_train_mode_after_policy_train": True,
        },
        "rtc": {
            "supports_rtc": policy.supports_rtc(),
            "policy_enabled": policy._rtc_enabled(),
            "model_enabled": model._rtc_enabled(),
            "processor_is_none": policy.rtc_processor is None,
        },
    }
    if result["resolved"]["n_prefix_physical"] != policy.config.prefix_length:
        raise RuntimeError("PREFIX_LENGTH_MISMATCH")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
