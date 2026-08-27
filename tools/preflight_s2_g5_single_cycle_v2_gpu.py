#!/usr/bin/env python3
"""Run one fresh G5 cycle under append-only ActionContract v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch

from forcesmolvla import action_delta, rules
from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
from forcesmolvla.rft import flow_sampling, losses
from forcesmolvla.rft.critic_action_adapter_v2 import (
    critic_action_for_q_guidance_v2,
    raw_gripper_out_of_public_tolerance_mask,
)

import preflight_s2_g5_single_cycle_gpu as legacy


ROOT = Path(__file__).parents[1].resolve()
WRAPPER = Path(__file__).resolve()
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

legacy.CONFIG = ROOT / "configs/stage2_g5_single_cycle.v2.development.yaml"
legacy.SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v13_g5_v2.json"
legacy.ARTIFACT = ROOT / "artifacts/development/stage2/s2_g5_single_cycle_preflight.v2.json"
legacy.REPORT = ROOT / "docs/s2_g5_single_cycle_preflight_report.v2.md"
legacy.CHECKPOINT = ROOT / "artifacts/development/stage2/g5_single_cycle_checkpoint.v2.development"


_legacy_tests = legacy.run_unit_tests


def _run_unit_tests_v2() -> dict:
    base = _legacy_tests()
    environment = legacy.os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_rft_critic_action_contract_v2.py"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    legacy.require(result.returncode == 0 and "passed" in output, f"G5_V2_TEST_FAILED:{output}")
    return {"legacy_g5": base, "action_contract_v2": {"exit_code": 0, "output": output}}


legacy.run_unit_tests = _run_unit_tests_v2

_legacy_protected_snapshot = legacy.protected_snapshot


def _protected_snapshot_v2() -> dict:
    result = _legacy_protected_snapshot()
    additions = {
        "action_contract_v2": ROOT / "configs/stage2_action_contract.v2.development.json",
        "action_adapter_v2": ROOT / "src/forcesmolvla/rft/critic_action_adapter_v2.py",
        "g3_v2_artifact": ROOT / "artifacts/development/stage2/s2_g3_differentiable_flow.v2.json",
        "g4_v2_artifact": ROOT / "artifacts/development/stage2/s2_g4_loss_preflight.v2.json",
        "g4_v2_source_manifest": ROOT / "artifacts/development/stage2/stage2_source_manifest.v12_g4_v2.json",
        "g5_v2_wrapper": WRAPPER,
    }
    result["files"].update({name: legacy.binding(path) for name, path in additions.items()})
    return result


legacy.protected_snapshot = _protected_snapshot_v2

_legacy_startup_snapshot = legacy.startup_snapshot


def _startup_snapshot_v2(protected: dict):
    values, manifest = _legacy_startup_snapshot(protected)
    additions = {
        "action_contract_v2/stage2_action_contract.v2.development.json": ROOT / "configs/stage2_action_contract.v2.development.json",
        "action_contract_v2/critic_action_adapter_v2.py": ROOT / "src/forcesmolvla/rft/critic_action_adapter_v2.py",
        "g3_v2/s2_g3_differentiable_flow.v2.json": ROOT / "artifacts/development/stage2/s2_g3_differentiable_flow.v2.json",
        "g4_v2/s2_g4_loss_preflight.v2.json": ROOT / "artifacts/development/stage2/s2_g4_loss_preflight.v2.json",
    }
    for relative, path in additions.items():
        value = path.read_bytes()
        values[relative] = value
        manifest["files"][relative] = {
            "source_path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(value).hexdigest(),
            "file_size": len(value),
        }
    return values, manifest


legacy.startup_snapshot = _startup_snapshot_v2

_legacy_report = legacy.report_markdown


def _report_v2(artifact: dict) -> str:
    return _legacy_report(artifact) + """

## ActionContract v2

This append-only rerun uses total binary internal gripper projection. Internal critic
canonicalization is not public execution authorization. Public tolerance, exception,
RuleSpec, and controller behavior are unchanged. No clipping, resampling, or binary
STE is used. Random empirical candidates are already frozen normalized endpoints;
their v2 projection is therefore an exact identity after endpoint validation.
"""


legacy.report_markdown = _report_v2


def _finalize_v2_artifact(call_counts: dict[str, int]) -> None:
    artifact = json.loads(legacy.ARTIFACT.read_text(encoding="utf-8"))
    raw_count = DIAGNOSTIC["raw_gripper_values"]
    pattern_count = DIAGNOSTIC["projected_gripper_patterns"]
    artifact["action_contract_v2"] = {
        "status": "pass",
        "contract": legacy.binding(ROOT / "configs/stage2_action_contract.v2.development.json"),
        "adapter": legacy.binding(ROOT / "src/forcesmolvla/rft/critic_action_adapter_v2.py"),
        "internal_critic_canonicalization": "total_binary",
        "public_execution_authorization_used": False,
        "public_call_counts": call_counts,
        "raw_gripper_out_of_public_tolerance_rate": (
            DIAGNOSTIC["raw_gripper_out_of_public_tolerance"] / raw_count if raw_count else 0.0
        ),
        "binary_gripper_pattern_duplicate_rate": (
            DIAGNOSTIC["duplicate_projected_gripper_patterns"] / pattern_count
            if pattern_count
            else 0.0
        ),
        "duplicates_allowed": True,
        "resampling_added": False,
        "clipping_added": False,
        "binary_ste_added": False,
        "random_candidate_projection": "identity_on_already_canonical_binary_endpoints",
    }
    artifact["schema_version"] = "forcesmolvla_s2_g5_single_cycle_preflight.v2"
    artifact["artifact_payload_sha256"] = legacy.canonical_sha256(
        {key: value for key, value in artifact.items() if key != "artifact_payload_sha256"}
    )
    legacy.atomic_json(legacy.ARTIFACT, artifact)


if __name__ == "__main__":
    forbidden = RuntimeError("G5_V2_PUBLIC_EXECUTION_PATH_CALLED")
    with (
        patch.object(action_delta.ActionDeltaProcessor, "from_delta", side_effect=forbidden) as inverse,
        patch.object(action_delta.ActionSafetyProfile, "validate_chunk", side_effect=forbidden) as validator,
        patch.object(action_delta, "decode_binary_gripper_width", side_effect=forbidden) as decoder,
        patch.object(rules, "load_and_validate_rulespec", side_effect=forbidden) as rulespec,
        patch.object(ForceSmolVLAPolicy, "predict_action_chunk", side_effect=forbidden) as predict,
    ):
        legacy.main()
    calls = {
        "public_validator_calls": validator.call_count,
        "absolute_inverse_calls": inverse.call_count,
        "public_binary_decoder_calls": decoder.call_count,
        "RuleSpec_calls": rulespec.call_count,
        "predict_action_chunk_calls": predict.call_count,
    }
    legacy.require(not any(calls.values()), f"G5_V2_PUBLIC_PATH_CALL:{calls}")
    _finalize_v2_artifact(calls)
