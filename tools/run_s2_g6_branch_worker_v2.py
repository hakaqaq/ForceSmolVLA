#!/usr/bin/env python3
"""Fresh-process G6-v2 worker; delegates frozen cycle math to the G6 worker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from forcesmolvla import action_delta, rules
from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
from forcesmolvla.rft import flow_sampling, losses
from forcesmolvla.rft.critic_action_adapter_v2 import critic_action_for_q_guidance_v2
from forcesmolvla.rft.exact_resume_v2 import install_exact_resume_v2

import preflight_s2_g5_single_cycle_gpu as g5
import run_s2_g6_branch_worker as worker


ROOT = Path(__file__).parents[1].resolve()
G5_CHECKPOINT = ROOT / "artifacts/development/stage2/g5_single_cycle_checkpoint.v2.development"
G6_CONFIG = ROOT / "configs/stage2_g6_exact_resume.v2.development.yaml"
G6_SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v14_g6_v2.json"

install_exact_resume_v2()
flow_sampling.critic_action_for_q_guidance = critic_action_for_q_guidance_v2
losses.critic_action_for_q_guidance = critic_action_for_q_guidance_v2
g5.CONFIG = ROOT / "configs/stage2_g5_single_cycle.v2.development.yaml"
g5.SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v13_g5_v2.json"
g5.CHECKPOINT = G5_CHECKPOINT
worker.G5_CHECKPOINT = G5_CHECKPOINT
worker.G6_CONFIG = G6_CONFIG
worker.G6_SOURCE_MANIFEST = G6_SOURCE


def _startup_snapshot_bytes_v2() -> dict[str, bytes]:
    paths = {
        "g6_v2/stage2_g6_exact_resume.v2.development.yaml": G6_CONFIG,
        "g6_v2/stage2_source_manifest.v14_g6_v2.json": G6_SOURCE,
        "g5_v2/stage2_g5_single_cycle.v2.development.yaml": g5.CONFIG,
        "g5_v2/stage2_source_manifest.v13_g5_v2.json": g5.SOURCE_MANIFEST,
        "g5_v2/checkpoint_manifest.json": G5_CHECKPOINT / "checkpoint_manifest.json",
        "action_contract_v2/stage2_action_contract.v2.development.json": ROOT / "configs/stage2_action_contract.v2.development.json",
        "automatic_g1/g1_manifest.json": ROOT / "artifacts/development/stage2/g1_frozen_detector_transition_view.v1/g1_manifest.json",
    }
    return {relative: path.read_bytes() for relative, path in paths.items()}


worker.startup_snapshot_bytes = _startup_snapshot_bytes_v2


if __name__ == "__main__":
    forbidden = RuntimeError("G6_V2_PUBLIC_EXECUTION_PATH_CALLED")
    with (
        patch.object(action_delta.ActionDeltaProcessor, "from_delta", side_effect=forbidden),
        patch.object(action_delta.ActionSafetyProfile, "validate_chunk", side_effect=forbidden),
        patch.object(action_delta, "decode_binary_gripper_width", side_effect=forbidden),
        patch.object(rules, "load_and_validate_rulespec", side_effect=forbidden),
        patch.object(ForceSmolVLAPolicy, "predict_action_chunk", side_effect=forbidden),
    ):
        worker.main()
