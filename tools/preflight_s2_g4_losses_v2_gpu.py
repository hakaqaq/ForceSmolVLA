#!/usr/bin/env python3
"""Run the frozen G4 gate with ActionContract-v2 internal canonicalization."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from forcesmolvla import action_delta, rules
from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
from forcesmolvla.rft import flow_sampling, losses
from forcesmolvla.rft.critic_action_adapter_v2 import critic_action_for_q_guidance_v2

import preflight_s2_g4_losses_gpu as legacy


ROOT = Path(__file__).parents[1].resolve()
WRAPPER = Path(__file__).resolve()

flow_sampling.critic_action_for_q_guidance = critic_action_for_q_guidance_v2
losses.critic_action_for_q_guidance = critic_action_for_q_guidance_v2

legacy.CONFIG = ROOT / "configs/stage2_g4_losses.v2.development.yaml"
legacy.ARTIFACT = ROOT / "artifacts/development/stage2/s2_g4_loss_preflight.v2.json"
legacy.SOURCE_MANIFEST = (
    ROOT / "artifacts/development/stage2/stage2_source_manifest.v12_g4_v2.json"
)
legacy.REPORT = ROOT / "docs/s2_g4_loss_preflight_report.v2.md"


_legacy_tests = legacy.run_unit_tests


def _run_unit_tests_v2() -> dict:
    base = _legacy_tests()
    environment = legacy.os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_rft_critic_action_contract_v2.py",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    legacy.require(result.returncode == 0 and "passed" in output, f"G4_V2_TEST_FAILED:{output}")
    return {"legacy_g4": base, "action_contract_v2": {"exit_code": 0, "output": output}}


legacy.run_unit_tests = _run_unit_tests_v2

_legacy_gpu_preflight = legacy.gpu_preflight


def _gpu_preflight_v2(batch: dict, config: dict) -> dict:
    forbidden = RuntimeError("G4_V2_PUBLIC_EXECUTION_PATH_CALLED")
    with (
        patch.object(action_delta.ActionDeltaProcessor, "from_delta", side_effect=forbidden) as inverse,
        patch.object(action_delta.ActionSafetyProfile, "validate_chunk", side_effect=forbidden) as validator,
        patch.object(action_delta, "decode_binary_gripper_width", side_effect=forbidden) as public_decoder,
        patch.object(rules, "load_and_validate_rulespec", side_effect=forbidden) as rulespec,
        patch.object(ForceSmolVLAPolicy, "predict_action_chunk", side_effect=forbidden) as predict,
    ):
        result = _legacy_gpu_preflight(batch, config)
    call_counts = {
        "public_validator_calls": validator.call_count,
        "absolute_inverse_calls": inverse.call_count,
        "public_binary_decoder_calls": public_decoder.call_count,
        "RuleSpec_calls": rulespec.call_count,
        "predict_action_chunk_calls": predict.call_count,
    }
    legacy.require(not any(call_counts.values()), f"G4_V2_PUBLIC_PATH_CALL:{call_counts}")
    result["action_contract_v2"] = {
        "adapter": legacy.binding(ROOT / "src/forcesmolvla/rft/critic_action_adapter_v2.py"),
        "contract": legacy.binding(ROOT / "configs/stage2_action_contract.v2.development.json"),
        "internal_critic_canonicalization": "total_binary",
        "public_execution_authorization_used": False,
        **call_counts,
    }
    result["candidate_duplicate_contract"] = {
        "duplicates_allowed": True,
        "resampling_to_remove_duplicates": False,
        "candidate_count_M_test_only": result["calql"]["candidate_count_M_test_only"],
        "projection_diagnostic_artifact": legacy.binding(
            ROOT / "artifacts/development/stage2/s2_action_contract_v2_preflight.v2.json"
        ),
    }
    return result


legacy.gpu_preflight = _gpu_preflight_v2

_legacy_source_manifest = legacy.source_manifest


def _source_manifest_v2() -> dict:
    result = _legacy_source_manifest()
    additions = {
        "action_contract_v2": ROOT / "configs/stage2_action_contract.v2.development.json",
        "action_adapter_v2": ROOT / "src/forcesmolvla/rft/critic_action_adapter_v2.py",
        "action_contract_v2_tests": ROOT / "tests/test_rft_critic_action_contract_v2.py",
        "g3_v2_artifact": ROOT / "artifacts/development/stage2/s2_g3_differentiable_flow.v2.json",
        "g3_v2_source_manifest": ROOT / "artifacts/development/stage2/stage2_source_manifest.v11_action_contract_v2_g3.json",
        "g4_v2_wrapper": WRAPPER,
    }
    result["schema_version"] = "forcesmolvla_stage2_source_manifest.v12_g4_v2"
    result["status"] = "PASS_APPEND_ONLY_G4_V2_SOURCE_CLOSURE"
    result["scope"] = "G4_v2_losses_and_zero_update_preflight_only"
    result["files"].update({name: legacy.binding(path) for name, path in additions.items()})
    result["runtime_imported_files"] = sorted(result["files"])
    return result


legacy.source_manifest = _source_manifest_v2


if __name__ == "__main__":
    legacy.main()
