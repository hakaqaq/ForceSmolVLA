from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from forcesmolvla.rft.stage3.integrated_capture import (
    CaptureBackendCapabilities,
    IntegratedCaptureError,
    IntegratedCaptureLedger,
    RECORDER_CONTROL_CHAIN,
    build_capture_contract,
    capture_mode_semantics,
    run_integrated_capture,
)


ROOT = Path(__file__).parents[1]


def _contract():
    return build_capture_contract(
        mode="shadow",
        session_id="session-real-1",
        episode_id="episode_000001",
        policy_revision="revision-sha-1",
        policy_epoch=2,
        reset_generation=3,
        takeover_generation=4,
    )


def _observation(ledger: IntegratedCaptureLedger) -> None:
    names = (
        "measured_tcp_pose", "wrench_notch_sensor", "gripper_state",
        "external_camera", "wrist_camera",
    )
    ledger.record_observation(
        observation_id="observation-1",
        t_ref_ns=1_000_000_000,
        stream_timestamps_ns={name: 999_000_000 for name in names},
        stream_ids={name: f"{name}:record-1" for name in names},
    )


def _request() -> dict:
    return {
        "request_id": "request-1",
        "chunk_id": "chunk-1",
        "clock_domain_id": "upper_host_monotonic_ns",
        "provenance": {"t_ref_ns": 1_000_000_000},
    }


def _result() -> dict:
    return {
        "request_id": "request-1",
        "chunk_id": "chunk-1",
        "t_ref_ns": 1_000_000_000,
    }


def test_shadow_contract_is_human_inference_only_and_policy_execute_is_hard_disabled(
    tmp_path: Path,
) -> None:
    contract = _contract()
    assert contract.actual_action_source == "human"
    assert contract.policy_inference is True
    assert contract.policy_execution is False
    assert contract.formal_replay is False
    assert contract.real_online_r is False
    assert contract.controller_process_count == 1
    assert contract.recorder_controller is True
    assert contract.deploy_controller is False
    assert capture_mode_semantics("policy-execute") == {
        "actual_action_source": "policy",
        "policy_inference": True,
        "policy_execution": True,
        "formal_replay": False,
        "real_online_r": False,
        "activation_authorized": False,
        "unlock_requires": [
            "future_explicit_authorization", "verified_deployment_binding",
        ],
    }
    with pytest.raises(IntegratedCaptureError, match="POLICY_EXECUTE_HARD_DISABLED"):
        build_capture_contract(
            mode="policy-execute",
            session_id="session-real-1",
            episode_id="episode_000001",
            policy_revision="revision-sha-1",
            policy_epoch=2,
            reset_generation=3,
            takeover_generation=4,
            deployment_binding=tmp_path / "even-if-present.json",
        )


def test_shadow_proposal_cannot_be_bound_to_human_ack_or_real_online_r() -> None:
    ledger = IntegratedCaptureLedger(_contract())
    _observation(ledger)
    request = _request()
    ledger.record_policy_request(
        request, observation_id="observation-1", recorded_monotonic_ns=1_000_000_010,
    )
    proposal = ledger.record_policy_result(
        request, _result(), recorded_monotonic_ns=1_010_000_000,
    )
    assert proposal["shadow_proposal"] is True
    assert proposal["executed"] is False
    with pytest.raises(IntegratedCaptureError, match="SHADOW_PROPOSAL_CANNOT_BIND"):
        ledger.record_actual_action_ack(
            ack_id="human-ack-1",
            observation_id="observation-1",
            receive_monotonic_ns=1_020_000_000,
            actual_action_source="human",
            policy_result_id=proposal["result_id"],
            proposal_id=proposal["proposal_id"],
        )
    ack = ledger.record_actual_action_ack(
        ack_id="human-ack-1",
        observation_id="observation-1",
        receive_monotonic_ns=1_020_000_000,
        actual_action_source="human",
    )
    assert ack["policy_executed_transition"] is False
    assert ack["real_online_r"] is False
    seal = ledger.seal_episode(
        seal_id="seal-1",
        sealed_monotonic_ns=1_030_000_000,
        terminal_observation_id="observation-1",
    )
    assert seal["shadow_proposals_executed"] is False
    assert seal["formal_replay"] is False
    assert seal["real_online_r"] is False


def test_integrated_backend_is_called_once_and_must_own_only_recorder_controller() -> None:
    class Backend:
        capabilities = CaptureBackendCapabilities(
            controller_owner="recorder",
            controller_process_count=1,
            starts_recorder_controller=True,
            starts_deploy_controller=False,
            control_chain_id=RECORDER_CONTROL_CHAIN,
            shares_observation_store=True,
            emits_episode_seal=True,
        )

        def __init__(self) -> None:
            self.calls = 0

        def capture(self, *, contract, ledger, recorder_arguments):
            del contract, recorder_arguments
            self.calls += 1
            _observation(ledger)
            request = _request()
            ledger.record_policy_request(
                request, observation_id="observation-1", recorded_monotonic_ns=1_000_000_010,
            )
            ledger.record_policy_result(
                request, _result(), recorded_monotonic_ns=1_010_000_000,
            )
            ledger.record_actual_action_ack(
                ack_id="human-ack-1",
                observation_id="observation-1",
                receive_monotonic_ns=1_020_000_000,
                actual_action_source="human",
            )
            return ledger.seal_episode(
                seal_id="seal-1",
                sealed_monotonic_ns=1_030_000_000,
                terminal_observation_id="observation-1",
            )

    backend = Backend()
    seal = run_integrated_capture(
        contract=_contract(), backend=backend,
        recorder_arguments={"root": "/tmp/shadow", "episodes": 1},
    )
    assert backend.calls == 1
    assert seal["seal_id"] == "seal-1"

    backend.capabilities = CaptureBackendCapabilities(
        **{**backend.capabilities.__dict__, "starts_deploy_controller": True}
    )
    with pytest.raises(IntegratedCaptureError, match="BACKEND_CAPABILITIES_INVALID"):
        run_integrated_capture(
            contract=_contract(), backend=backend,
            recorder_arguments={"root": "/tmp/shadow", "episodes": 1},
        )
    assert backend.calls == 1


def test_integrated_capture_cli_modes_are_explicit_and_validate_without_ros(tmp_path: Path) -> None:
    command = [
        sys.executable, str(ROOT / "tools/run_stage3_integrated_capture.py"),
        "--mode", "shadow", "--root", str(tmp_path), "--task", "task",
        "--session-id", "session-1", "--episode-id", "episode_000001",
        "--policy-revision", "revision-1",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "VALIDATED_NOT_LAUNCHED"
    assert payload["robot_or_ros_started"] is False
    blocked = subprocess.run(
        [*command[:3], "policy-execute", *command[4:]],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert blocked.returncode == 2
    assert "POLICY_EXECUTE_HARD_DISABLED" in blocked.stdout
    assert json.loads(blocked.stdout)["requested_mode_semantics"]["policy_execution"] is True
