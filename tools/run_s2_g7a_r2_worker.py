#!/usr/bin/env python3
"""Fresh G7-A-r2 worker using total-binary internal gripper projection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch

from forcesmolvla import action_delta, rules
from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
from forcesmolvla.rft import flow_sampling, g7a, losses
from forcesmolvla.rft.critic_action_adapter_v2 import (
    critic_action_for_q_guidance_v2,
    raw_gripper_out_of_public_tolerance_mask,
)
from forcesmolvla.rft.g7a_r2 import verify_g7a_r2_source_manifest

import preflight_s2_g5_single_cycle_gpu as g5
import run_s2_g7a_worker as worker


ROOT = Path(__file__).parents[1].resolve()
DIAGNOSTIC = {
    "raw_gripper_values": 0,
    "raw_gripper_out_of_public_tolerance": 0,
    "projected_gripper_patterns": 0,
    "duplicate_projected_gripper_patterns": 0,
}


def _audited_adapter(chunk, *, delta_action_mean7, delta_action_std7):
    raw = chunk[:, :3, 6]
    outside = raw_gripper_out_of_public_tolerance_mask(
        raw,
        gripper_mean=delta_action_mean7[6],
        gripper_std=delta_action_std7[6],
    )
    action = critic_action_for_q_guidance_v2(
        chunk,
        delta_action_mean7=delta_action_mean7,
        delta_action_std7=delta_action_std7,
    )
    patterns = action[..., 6].detach().float()
    DIAGNOSTIC["raw_gripper_values"] += raw.numel()
    DIAGNOSTIC["raw_gripper_out_of_public_tolerance"] += int(outside.sum().item())
    DIAGNOSTIC["projected_gripper_patterns"] += patterns.shape[0]
    DIAGNOSTIC["duplicate_projected_gripper_patterns"] += int(
        patterns.shape[0] - torch.unique(patterns, dim=0).shape[0]
    )
    return action


flow_sampling.critic_action_for_q_guidance = _audited_adapter
losses.critic_action_for_q_guidance = _audited_adapter
g7a.verify_source_manifest = verify_g7a_r2_source_manifest
g5.CONFIG = ROOT / "configs/stage2_g5_single_cycle.v2.development.yaml"
g5.SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v13_g5_v2.json"
worker.CONFIG = ROOT / "configs/stage2_g7a_r2_critic_warmup.development.yaml"
worker.SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v10_g7a_r2.json"


def _finalize_result(path: Path, calls: dict[str, int]) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    raw_count = DIAGNOSTIC["raw_gripper_values"]
    pattern_count = DIAGNOSTIC["projected_gripper_patterns"]
    result["action_contract_v2"] = {
        "status": "pass",
        "internal_gripper_projection": "total_binary",
        "public_execution_authorization_used": False,
        "public_call_counts": calls,
        "raw_gripper_out_of_public_tolerance_rate": (
            DIAGNOSTIC["raw_gripper_out_of_public_tolerance"] / raw_count if raw_count else 0.0
        ),
        "binary_gripper_pattern_duplicate_rate": (
            DIAGNOSTIC["duplicate_projected_gripper_patterns"] / pattern_count
            if pattern_count
            else 0.0
        ),
        "clipping_added": False,
        "resampling_added": False,
        "binary_ste_added": False,
    }
    worker.atomic_json(path, result)


if __name__ == "__main__":
    forbidden = RuntimeError("G7A_R2_PUBLIC_EXECUTION_PATH_CALLED")
    with (
        patch.object(action_delta.ActionDeltaProcessor, "from_delta", side_effect=forbidden) as inverse,
        patch.object(action_delta.ActionSafetyProfile, "validate_chunk", side_effect=forbidden) as validator,
        patch.object(action_delta, "decode_binary_gripper_width", side_effect=forbidden) as decoder,
        patch.object(rules, "load_and_validate_rulespec", side_effect=forbidden) as rulespec,
        patch.object(ForceSmolVLAPolicy, "predict_action_chunk", side_effect=forbidden) as predict,
    ):
        worker.main()
    calls = {
        "public_validator_calls": validator.call_count,
        "absolute_inverse_calls": inverse.call_count,
        "public_binary_decoder_calls": decoder.call_count,
        "RuleSpec_calls": rulespec.call_count,
        "predict_action_chunk_calls": predict.call_count,
    }
    worker.require(not any(calls.values()), f"G7A_R2_PUBLIC_PATH_CALL:{calls}")
    # The result path is parsed by the delegated worker; recover it without consuming RNG.
    import sys

    result_path = Path(sys.argv[sys.argv.index("--result") + 1])
    _finalize_result(result_path, calls)
