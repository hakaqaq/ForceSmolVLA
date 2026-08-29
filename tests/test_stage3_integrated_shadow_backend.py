from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from forcesmolvla.rft.stage3.integrated_capture import (
    IntegratedCaptureError,
    IntegratedCaptureLedger,
    RECORDER_CONTROL_CHAIN,
    build_capture_contract,
)
from forcesmolvla.rft.stage3.integrated_shadow_backend import (
    ForbiddenPolicyPublisher,
    IntegratedShadowBackend,
    ShadowArtifactStore,
    build_native_recorder_command,
)


ROOT = Path(__file__).parents[1]


def _contract():
    return build_capture_contract(
        mode="shadow",
        session_id="shadow-session-1",
        episode_id="episode_000000",
        policy_revision="4" * 64,
        policy_epoch=2,
        reset_generation=3,
        takeover_generation=4,
    )


def _arguments(tmp_path: Path) -> dict:
    return {
        "root": str(tmp_path / "native"),
        "task": "Pick up the purple ring and place it onto the red peg.",
        "episodes": 1,
        "episode_time": 60.0,
        "tool_profile": "onrobot_robotiq",
    }


def test_backend_owns_exactly_one_native_recorder_control_chain(tmp_path: Path) -> None:
    backend = IntegratedShadowBackend()
    assert backend.capabilities.controller_owner == "recorder"
    assert backend.capabilities.controller_process_count == 1
    assert backend.capabilities.starts_recorder_controller is True
    assert backend.capabilities.starts_deploy_controller is False
    assert backend.capabilities.control_chain_id == RECORDER_CONTROL_CHAIN

    command = build_native_recorder_command(_arguments(tmp_path))
    assert command[0] == sys.executable
    assert command[1].endswith("/record_franka_hilserl_impedance.py")
    assert "deploy_forcesmolvla.py" not in " ".join(command)
    assert "--execute" not in command
    assert command[command.index("--episodes") + 1] == "1"


def test_policy_publisher_is_a_fail_closed_non_dds_sentinel() -> None:
    publisher = ForbiddenPolicyPublisher("/fr3/hilserl/policy_action_control")
    with pytest.raises(IntegratedCaptureError, match="POLICY_PROPOSAL_PUBLISH_FORBIDDEN"):
        publisher.publish({"source": "policy"})


def test_proposal_and_human_ack_are_separate_and_cannot_claim_execution(
    tmp_path: Path,
) -> None:
    store = ShadowArtifactStore(tmp_path / "sidecar")
    proposal = {
        "actual_action_source": "human",
        "policy_execution": False,
        "executed": False,
        "real_online_r": False,
        "proposal_id": "proposal-1",
    }
    ack = {
        "actual_action_source": "human",
        "policy_result_id": None,
        "proposal_id": None,
        "policy_executed_transition": False,
        "real_online_r": False,
        "ack_id": "human-ack-1",
    }
    proposal_path = store.append("policy_shadow_proposal.jsonl", proposal)
    ack_path = store.append("policy_shadow_human_ack.jsonl", ack)
    assert proposal_path != ack_path
    assert json.loads(proposal_path.read_text())["executed"] is False
    recorded_ack = json.loads(ack_path.read_text())
    assert recorded_ack["policy_result_id"] is None
    assert recorded_ack["proposal_id"] is None
    with pytest.raises(IntegratedCaptureError, match="SHADOW_PROPOSAL_SEMANTICS_INVALID"):
        store.append(
            "policy_shadow_proposal.jsonl", {**proposal, "executed": True}
        )
    with pytest.raises(IntegratedCaptureError, match="SHADOW_HUMAN_ACK_SEMANTICS_INVALID"):
        store.append(
            "policy_shadow_human_ack.jsonl",
            {**ack, "policy_result_id": "policy-result-1"},
        )


def test_backend_rejects_forged_policy_execution_before_loading_runtime(
    tmp_path: Path,
) -> None:
    backend = IntegratedShadowBackend()
    contract = replace(
        _contract(),
        actual_action_source="policy",
        policy_execution=True,
        deploy_controller=True,
    )
    with pytest.raises(IntegratedCaptureError, match="CONTRACT_NOT_AUTHORIZED"):
        backend.capture(
            contract=contract,
            ledger=IntegratedCaptureLedger(_contract()),
            recorder_arguments=_arguments(tmp_path),
        )


def test_integrated_cli_passes_shadow_runtime_binding_without_launch(
    tmp_path: Path,
) -> None:
    profile = ROOT / "configs/deployment.active.development.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_stage3_integrated_capture.py"),
            "--mode",
            "shadow",
            "--root",
            str(tmp_path / "native"),
            "--task",
            "task",
            "--session-id",
            "session-1",
            "--episode-id",
            "episode_000000",
            "--policy-revision",
            "4" * 64,
            "--policy-port",
            "8123",
            "--deployment-profile",
            str(profile),
            "--shadow-inference-period",
            "0.2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "VALIDATED_NOT_LAUNCHED"
    assert payload["robot_or_ros_started"] is False
    arguments = payload["recorder_arguments"]
    assert arguments["policy_port"] == 8123
    assert arguments["shadow_inference_period"] == 0.2
    assert arguments["deployment_profile"] == str(profile.resolve())
