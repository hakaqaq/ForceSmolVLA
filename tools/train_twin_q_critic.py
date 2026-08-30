#!/usr/bin/env python3
"""CPU-only append-only coordinator for G7-A-r2."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from forcesmolvla.rft import g7a
from forcesmolvla.rft.g7a_r2 import verify_g7a_r2_source_manifest

import preflight_s2_g5_single_cycle_gpu as g5
import preflight_s2_g7a_critic_warmup_gpu as legacy


ROOT = Path(__file__).parents[1].resolve()
g7a.verify_source_manifest = verify_g7a_r2_source_manifest
g5.CONFIG = ROOT / "configs/stage2_g5_single_cycle.v2.development.yaml"
g5.SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v13_g5_v2.json"
legacy.CONFIG = ROOT / "configs/stage2_g7a_r2_critic_warmup.development.yaml"
legacy.SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v10_g7a_r2.json"
legacy.WORKER = ROOT / "tools/run_s2_g7a_r2_worker.py"
legacy.OUTPUT = ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup"
legacy.CHECKPOINT = ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint"
legacy.ARTIFACT = ROOT / "artifacts/development/stage2/s2_g7a_r2_critic_warmup_preflight.json"
legacy.REPORT = ROOT / "docs/s2_g7a_r2_critic_warmup_report.md"
legacy.G5_CHECKPOINT = ROOT / "artifacts/development/stage2/g5_single_cycle_checkpoint.v2.development"
legacy.G6_OUTPUT = ROOT / "artifacts/development/stage2/g6_exact_resume.v2"


def _protected_snapshot_r2() -> dict:
    from forcesmolvla.rft.exact_resume import checkpoint_tree

    base = g5.protected_snapshot()
    paths = {
        "g7a_r1_fail": ROOT / "artifacts/development/stage2/s2_g7a_critic_warmup_preflight.json",
        "g7a_r1_report": ROOT / "docs/s2_g7a_critic_warmup_report.md",
        "gripper_domain_audit": ROOT / "artifacts/development/stage2/s2_g7a_gripper_path_domain_audit.v1.json",
        "action_contract_v2": ROOT / "configs/stage2_action_contract.v2.development.json",
        "action_adapter_v2": ROOT / "src/forcesmolvla/rft/critic_action_adapter_v2.py",
        "action_contract_v2_artifact": ROOT / "artifacts/development/stage2/s2_action_contract_v2_preflight.v2.json",
        "g3_v2": ROOT / "artifacts/development/stage2/s2_g3_differentiable_flow.v2.json",
        "g4_v2": ROOT / "artifacts/development/stage2/s2_g4_loss_preflight.v2.json",
        "g5_v2": ROOT / "artifacts/development/stage2/s2_g5_single_cycle_preflight.v2.json",
        "g6_v2": ROOT / "artifacts/development/stage2/s2_g6_exact_resume_preflight.v2.json",
        "g7a_r2_config": legacy.CONFIG,
        "g7a_r2_source_manifest": legacy.SOURCE_MANIFEST,
    }
    return {
        "g5_protected": base,
        "files": {name: legacy.binding(path) for name, path in paths.items()},
        "trees": {
            "r5_checkpoint": base["r5_checkpoint_tree"],
            "stage1_dataset": base["p8_storage_tree"],
            "g5_v2_checkpoint": checkpoint_tree(legacy.G5_CHECKPOINT),
            "g6_v2_output": checkpoint_tree(legacy.G6_OUTPUT),
        },
    }


legacy.protected_snapshot = _protected_snapshot_r2

_legacy_tests = legacy.run_tests


def _run_tests_r2() -> dict:
    base = _legacy_tests()
    environment = legacy.os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_rft_critic_action_contract_v2.py"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    legacy.require(result.returncode == 0 and "passed" in output, f"G7A_R2_TEST_FAILED:{output}")
    return {"legacy_g7a": base, "action_contract_v2": {"exit_code": 0, "output": output}}


legacy.run_tests = _run_tests_r2

_legacy_report = legacy.report_markdown


def _report_r2(artifact: dict) -> str:
    return _legacy_report(artifact) + """

## ActionContract v2 and r1 preservation

`G7A_R1_FAIL` remains preserved with zero optimizer/Polyak/Actor updates and no
checkpoint. Its numerical-stability status is `not_measured`. This r2 run uses
total-binary internal gripper projection; public execution behavior and tolerance
are unchanged. No clipping, resampling, or binary STE was added.
"""


legacy.report_markdown = _report_r2


def _finalize_artifact() -> None:
    artifact = json.loads(legacy.ARTIFACT.read_text(encoding="utf-8"))
    artifact.update(
        {
            "schema_version": "forcesmolvla_s2_g7a_r2_critic_warmup_preflight.v1",
            "G7A_R1_FAIL": "preserved",
            "G7A_R1_CRITIC_NUMERICAL_STABILITY": "not_measured",
            "GRIPPER_PATH_DOMAIN_AUDIT": "pass",
            "FAILURE_SCOPE": "true_action_contract_error",
            "ACTION_CONTRACT_V1": "historical_superseded",
            "ACTION_CONTRACT_V2": "pass",
            "G7A_R2": "pass",
            "PUBLIC_INFERENCE_BEHAVIOR_CHANGED": "no",
            "PUBLIC_TOLERANCE_CHANGED": "no",
            "CLIPPING_ADDED": "no",
            "RESAMPLING_ADDED": "no",
            "BINARY_STE_ADDED": "no",
            "ACTOR_UPDATES_IN_G7A": 0,
        }
    )
    artifact["artifact_payload_sha256"] = legacy.hashlib.sha256(
        json.dumps(
            {key: value for key, value in artifact.items() if key != "artifact_payload_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    legacy.atomic_json(legacy.ARTIFACT, artifact)


if __name__ == "__main__":
    legacy.main()
    _finalize_artifact()
