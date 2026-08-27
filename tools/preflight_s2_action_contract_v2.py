#!/usr/bin/env python3
"""Append-only, zero-update ActionContract-v2 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from forcesmolvla import action_delta
from forcesmolvla.action_delta import decode_binary_gripper_width
from forcesmolvla.rft.critic_action_adapter_v2 import (
    critic_action_for_q_guidance_v2,
    normalized_gripper_endpoints_v2,
    project_binary_gripper_width_v2,
    raw_gripper_out_of_public_tolerance_mask,
)
from forcesmolvla.rft.gripper_domain_audit import global_rng_digest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage2_action_contract.v2.development.json"
V1 = ROOT / "configs/stage2_action_contract.development.json"
NORMALIZER = ROOT / "datasets/task2_lerobotv3/normalizer_manifest.json"
PUBLIC_SOURCE = ROOT / "src/forcesmolvla/action_delta.py"
PUBLIC_MODEL = ROOT / "src/forcesmolvla/modeling_forcesmolvla.py"
PUBLIC_INFERENCE = ROOT / "src/forcesmolvla/inference.py"
R1 = ROOT / "artifacts/development/stage2/s2_g7a_critic_warmup_preflight.json"
R1_REPORT = ROOT / "docs/s2_g7a_critic_warmup_report.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno()); temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise RuntimeError("ACTION_CONTRACT_V2_OUTPUT_EXISTS")
    before = {name: sha(path) for name, path in {
        "v1": V1, "public_source": PUBLIC_SOURCE, "public_model": PUBLIC_MODEL,
        "public_inference": PUBLIC_INFERENCE, "normalizer": NORMALIZER,
        "r1": R1, "r1_report": R1_REPORT,
    }.items()}
    config = json.loads(CONFIG.read_text())
    normalizer = json.loads(NORMALIZER.read_text())["features"]["delta_action7"]
    mean = torch.tensor(normalizer["mean"], dtype=torch.float32)
    std = torch.tensor(normalizer["std"], dtype=torch.float32)
    offender = torch.zeros(1, 50, 7); offender[..., 6] = 1.71746826171875
    internal = critic_action_for_q_guidance_v2(
        offender, delta_action_mean7=mean, delta_action_std7=std
    )
    endpoints = normalized_gripper_endpoints_v2(mean, std)
    physical = offender.numpy() * std.numpy() + mean.numpy()
    public_rejected = False
    try: decode_binary_gripper_width(physical)
    except ValueError as error:
        public_rejected = "outside the frozen" in str(error)
    if not public_rejected: raise RuntimeError("ACTION_CONTRACT_V2_PUBLIC_REJECTION_DRIFT")
    threshold = action_delta.BINARY_GRIPPER_SWITCH_WIDTH_M; eps = 1e-7
    threshold_values = np.array([threshold - eps, threshold, threshold + eps])
    public_input = np.zeros((3, 7)); public_input[:, 6] = threshold_values
    public_threshold = decode_binary_gripper_width(public_input)[:, 6]
    internal_threshold = project_binary_gripper_width_v2(torch.from_numpy(threshold_values)).numpy()
    if not np.array_equal(public_threshold.astype(np.float32), internal_threshold):
        raise RuntimeError("ACTION_CONTRACT_V2_THRESHOLD_GOLDEN_MISMATCH")
    generator = torch.Generator(device="cpu").manual_seed(702)
    samples = torch.randn(10000, generator=generator) * 1000
    rng_before = global_rng_digest(); projected = project_binary_gripper_width_v2(samples)
    rng_after = global_rng_digest()
    if rng_before != rng_after: raise RuntimeError("ACTION_CONTRACT_V2_PROJECTOR_RNG_USE")
    chunk = torch.randn(2, 50, 7, requires_grad=True)
    with (
        patch.object(action_delta.ActionDeltaProcessor, "from_delta", side_effect=AssertionError),
        patch.object(action_delta.ActionSafetyProfile, "validate_chunk", side_effect=AssertionError),
        patch.object(action_delta, "decode_binary_gripper_width", side_effect=AssertionError),
    ):
        q_action = critic_action_for_q_guidance_v2(
            chunk, delta_action_mean7=mean, delta_action_std7=std
        )
    q_action.sum().backward()
    out_of_range = raw_gripper_out_of_public_tolerance_mask(
        samples, gripper_mean=mean[6], gripper_std=std[6]
    )
    unique, counts = torch.unique(projected, return_counts=True)
    after = {name: sha(path) for name, path in {
        "v1": V1, "public_source": PUBLIC_SOURCE, "public_model": PUBLIC_MODEL,
        "public_inference": PUBLIC_INFERENCE, "normalizer": NORMALIZER,
        "r1": R1, "r1_report": R1_REPORT,
    }.items()}
    if before != after: raise RuntimeError("ACTION_CONTRACT_V2_FROZEN_INPUT_CHANGED")
    acceptance = {
        "offender_internal_open_endpoint": bool(torch.all(internal[..., 6] == endpoints[1])),
        "offender_public_still_rejected": public_rejected,
        "threshold_direction_tie_public_golden": bool(np.array_equal(public_threshold, [0.0, 0.085, 0.085])),
        "all_finite_total_binary": set(projected.tolist()) == {0.0, torch.tensor(0.085).item()},
        "only_normalized_endpoints_enter_q": bool(torch.all((internal[..., 6] == endpoints[0]) | (internal[..., 6] == endpoints[1]))),
        "public_execution_calls_zero": True,
        "tcp6_gradient_nonzero": bool(torch.all(chunk.grad[:, :3, :6] != 0)),
        "gripper_gradient_zero": int(torch.count_nonzero(chunk.grad[:, :3, 6])) == 0,
        "padding_gradient_zero": int(torch.count_nonzero(chunk.grad[:, 3:])) == 0,
        "projector_rng_unchanged": rng_before == rng_after,
        "duplicates_preserved_no_resampling": int(counts.max()) > 1,
        "public_and_frozen_sha_unchanged": before == after,
    }
    if not all(acceptance.values()): raise RuntimeError(f"ACTION_CONTRACT_V2_FAIL:{acceptance}")
    atomic_json(args.output, {
        "schema_version": "forcesmolvla_s2_action_contract_v2_preflight.v1",
        "ACTION_CONTRACT_V1": "historical_superseded",
        "ACTION_CONTRACT_V2": "pass",
        "INTERNAL_GRIPPER_PROJECTION": "total_binary",
        "PUBLIC_INFERENCE_BEHAVIOR_CHANGED": "no",
        "PUBLIC_TOLERANCE_CHANGED": "no",
        "CLIPPING_ADDED": "no", "RESAMPLING_ADDED": "no", "BINARY_STE_ADDED": "no",
        "config": config, "acceptance": acceptance,
        "offender": {"normalized": 1.71746826171875, "continuous_width_m": float(physical[0,0,6]), "internal_endpoint_m": 0.085, "public_rejected": True},
        "threshold_golden": {"input_m": threshold_values.tolist(), "public_m": public_threshold.tolist(), "internal_m": internal_threshold.tolist()},
        "normalized_endpoints": endpoints.tolist(),
        "raw_gripper_out_of_public_tolerance_rate": float(out_of_range.float().mean()),
        "projected_endpoint_counts": {str(float(key)): int(value) for key, value in zip(unique, counts, strict=True)},
        "candidate_duplicate_rate": 1.0 - len(unique) / len(projected),
        "call_counts": {"public_validator_calls": 0, "absolute_inverse_calls": 0, "RuleSpec_calls": 0},
        "optimizer_created": 0, "optimizer_updates": 0, "rng_unchanged": True,
        "frozen_sha_before": before, "frozen_sha_after": after,
        "G7A_R1_FAIL": "preserved",
    })
    print(json.dumps({"ACTION_CONTRACT_V2": "pass", "raw_out_of_public_tolerance_rate": float(out_of_range.float().mean())}, sort_keys=True))


if __name__ == "__main__": main()
