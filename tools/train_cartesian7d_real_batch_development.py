#!/usr/bin/env python3
"""CUDA-only real-data optimizer smoke; never a formal ForceToken checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {args.output}")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 required")
    socket.socket.connect = lambda self, address: (_ for _ in ()).throw(
        RuntimeError(f"NETWORK_ACCESS_FORBIDDEN: {address}")
    )

    import torch
    from safetensors.torch import save_file

    from forcesmolvla.checkpoint import load_offline_base_policy
    from forcesmolvla.configuration_forcesmolvla import CAMERA1, CAMERA2, OFFLINE_FULL_FINETUNE
    from forcesmolvla.dataset_v3 import load_dataset_split
    from forcesmolvla.training_data import load_runtime_artifacts, prepare_training_sample
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE")
    device_name = "cuda"
    project_root = Path(__file__).parents[1].resolve()
    dataset_root = args.dataset_root.resolve()
    delta_timestamps = {"action": [index / 30 for index in range(50)]}
    dataset = load_dataset_split(
        dataset_root,
        repo_id="local/task1_forcesmolvla_v4_1",
        split_name="train",
        artifact_use="development",
        delta_timestamps=delta_timestamps,
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError("sample index outside train split")
    runtime_artifacts = load_runtime_artifacts(
        dataset_root,
        calibration_bundle_path=root / "configs/calibration_bundle.development.json",
        wrench_geometry_spec_path=root / "configs/wrench_geometry_spec.development.json",
        action_delta_spec_path=root / "artifacts/development/action_delta_spec.json",
        expected_repo_id="local/task1_forcesmolvla_v4_1",
    )
    prepared = prepare_training_sample(dataset[args.sample_index], runtime_artifacts.normalizer)

    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    with contextlib.redirect_stdout(sys.stderr):
        policy, base_report = load_offline_base_policy(
            project_root / "assets/base_checkpoint",
            project_root / "assets/smolvlm_constructor",
            device=device_name,
            training_stage=OFFLINE_FULL_FINETUNE,
        )
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    encoded = tokenizer(
        prepared["task"] + "\n",
        padding="max_length",
        truncation=True,
        max_length=48,
        return_tensors="pt",
    )
    batch = {
        CAMERA1: prepared["camera1"].unsqueeze(0).to(device),
        CAMERA2: prepared["camera2"].unsqueeze(0).to(device),
        "observation.state": torch.from_numpy(prepared["state7"]).unsqueeze(0).to(device),
        "observation.wrench": torch.from_numpy(prepared["wrench6"]).unsqueeze(0).to(device),
        ACTION: torch.from_numpy(prepared["delta_action7"]).unsqueeze(0).to(device),
        "action_valid_mask": torch.from_numpy(prepared["action_valid_mask"])
        .unsqueeze(0)
        .to(device),
        OBS_LANGUAGE_TOKENS: encoded["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: encoded["attention_mask"].to(device=device, dtype=torch.bool),
    }
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    noise = torch.randn(1, 50, 7, generator=generator, device=device, dtype=torch.float32)
    time = torch.tensor([0.4], dtype=torch.float32, device=device)

    trainable = [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in policy.named_parameters() if not parameter.requires_grad]
    if not trainable or frozen:
        raise RuntimeError(f"OFFLINE_FULL_FINETUNE_PARAMETER_MISMATCH: frozen={frozen}")
    decay, no_decay = [], []
    for name, parameter in trainable:
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") or "embed" in name else decay).append(
            parameter
        )
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-10},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    first_values = {name: float(parameter.detach().reshape(-1)[0].cpu()) for name, parameter in trainable}

    policy.eval()
    with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
        initial_loss, _ = policy.forward(batch, noise=noise, time=time)
    losses = []
    gradient_norms = []
    policy.train()
    if not policy.model.vlm_with_expert.vlm.training:
        raise AssertionError("offline full finetuning requires the VLM in train mode")
    for _ in range(args.samples):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            loss, _ = policy.forward(batch, noise=noise, time=time)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], max_norm=10.0
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("nonfinite gradient norm")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(gradient_norm.detach().cpu()))

    policy.eval()
    with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
        final_loss, _ = policy.forward(batch, noise=noise, time=time)
    changed_tensors = sum(
        float(parameter.detach().reshape(-1)[0].cpu()) != first_values[name]
        for name, parameter in trainable
    )
    if changed_tensors == 0:
        raise AssertionError("optimizer did not change any sampled trainable parameter")
    gradient_names = [name for name, parameter in trainable if parameter.grad is not None]
    nonzero_gradient_names = [
        name
        for name, parameter in trainable
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).detach().cpu())
    ]
    no_gradient_names = sorted(set(name for name, _ in trainable) - set(gradient_names))

    args.output.mkdir(parents=True)
    trainable_state = {
        name: parameter.detach().cpu().contiguous() for name, parameter in trainable
    }
    checkpoint_path = args.output / "model.safetensors"
    save_file(trainable_state, str(checkpoint_path))
    checkpoint_digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b""):
            checkpoint_digest.update(chunk)
    checkpoint_sha256 = checkpoint_digest.hexdigest()
    report = {
        "status": "pass",
        "artifact_status": "development_only",
        "formal_ready": False,
        "model_variant": "SmolVLA-Cartesian7D",
        "training_stage": OFFLINE_FULL_FINETUNE,
        "all_parameters_trainable": True,
        "wrench_injected_into_model": False,
        "wrench_normalized_and_audited": True,
        "purpose": "real-data single-batch optimizer overfit smoke",
        "dataset_root": str(dataset_root),
        "dataset_split": "train",
        "train_split_frames": len(dataset),
        "sample_index": args.sample_index,
        "sample_id": prepared["batch_id"],
        "sample_sha256": prepared["batch_sha256"],
        "seed": args.seed,
        "device": device_name,
        "dtype": "bf16 autocast",
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 1e-10,
            "grad_clip": 10.0,
            "scheduler": "none-development-overfit",
        },
        "primary_budget_unit": "samples",
        "training_samples_seen": args.samples,
        "derived_optimizer_updates": args.samples,
        "initial_fixed_loss": float(initial_loss.detach().cpu()),
        "training_losses": losses,
        "final_fixed_loss": float(final_loss.detach().cpu()),
        "gradient_norms_before_clip": gradient_norms,
        "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "trainable_parameter_tensors": len(trainable),
        "gradient_parameter_tensors": len(gradient_names),
        "nonzero_gradient_parameter_tensors": len(nonzero_gradient_names),
        "no_gradient_parameter_names": no_gradient_names,
        "sampled_changed_parameter_tensors": changed_tensors,
        "vlm_was_train_mode_during_updates": True,
        "base_load": base_report.to_dict(),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_contents": "complete model state; every existing parameter is trainable in this stage",
        "checkpoint_selection_performed": False,
        "formal_metrics_reported": False,
        "robot_actions_sent": 0,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    if not all(math.isfinite(value) for value in losses + gradient_norms):
        raise AssertionError("nonfinite training report")
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
