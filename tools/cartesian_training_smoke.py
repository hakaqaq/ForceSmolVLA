#!/usr/bin/env python3
"""One offline synthetic Cartesian7D GPU forward/backward; no optimizer step."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import socket
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 required")
    socket.socket.connect = lambda self, address: (_ for _ in ()).throw(
        RuntimeError(f"NETWORK_ACCESS_FORBIDDEN: {address}")
    )

    import torch

    from forcesmolvla.checkpoint import load_offline_base_policy
    from forcesmolvla.configuration_forcesmolvla import CAMERA1, CAMERA2
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE")
    device_name = "cuda"
    root = args.project_root.resolve()
    with contextlib.redirect_stdout(sys.stderr):
        policy, report = load_offline_base_policy(
            root / "assets" / "base_checkpoint",
            root / "assets" / "smolvlm_constructor",
            device=device_name,
        )
    policy.train()
    device = torch.device(device_name)
    torch.manual_seed(4107)
    batch = {
        CAMERA1: torch.rand(1, 3, 480, 640, device=device),
        CAMERA2: torch.rand(1, 3, 480, 640, device=device),
        "observation.state": torch.randn(1, 7, device=device),
        ACTION: torch.randn(1, 50, 7, device=device),
        "action_valid_mask": torch.tensor(
            [[True] * 47 + [False] * 3], dtype=torch.bool, device=device
        ),
        OBS_LANGUAGE_TOKENS: torch.arange(48, device=device).view(1, 48),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor(
            [[True] * 19 + [False] * 29], dtype=torch.bool, device=device
        ),
    }
    noise7 = torch.randn(1, 50, 7, device=device)
    time = torch.tensor([0.4], dtype=torch.float32, device=device)
    policy.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
        loss, loss_report = policy.forward(batch, noise=noise7, time=time)
    if not torch.isfinite(loss):
        raise AssertionError("nonfinite loss")
    loss.backward()
    gradient_parameters = 0
    squared_norm = 0.0
    for parameter in policy.parameters():
        if parameter.grad is not None:
            gradient_parameters += 1
            squared_norm += float(parameter.grad.float().pow(2).sum().detach().cpu())
    if gradient_parameters == 0 or not math.isfinite(squared_norm):
        raise AssertionError("missing or nonfinite gradients")
    print(json.dumps({
        "status": "pass",
        "artifact_status": "development_only",
        "device": device_name,
        "dtype": "bf16 autocast",
        "batch_size": 1,
        "valid_horizon": 47,
        "valid_feature_tokens": loss_report["valid_feature_tokens"],
        "loss": float(loss.detach().cpu()),
        "gradient_parameter_tensors": gradient_parameters,
        "gradient_global_norm": math.sqrt(squared_norm),
        "optimizer_step_performed": False,
        "base_loaded_tensors": report.loaded_tensor_count,
        "missing_keys": list(report.missing_keys),
        "unexpected_keys": list(report.unexpected_keys),
        "robot_actions_sent": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
