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
    validate_development_policy_package,
)


ROOT = Path(__file__).parents[1]
BASELINE_POLICY_REVISION = (
    "e24c1d6bb0a778921659514ac47c692b952178aa39af2601ccf0fc32bf94774d"
)
BASELINE_DEPLOYMENT_BINDING = ROOT / (
    "artifacts/development/live/"
    "task2_cycle210_policy_execution_smoke_binding.v1.json"
)


def _development_package(
    tmp_path: Path,
    revision: str,
    *,
    candidate: bool,
    published: bool = True,
    activated: bool = False,
) -> Path:
    package = tmp_path / (
        f"candidate-{revision[0]}" if candidate else f"baseline-{revision[0]}"
    )
    package.mkdir()
    metadata = {
        "artifact_purpose": (
            "stage3_development_candidate_actor"
            if candidate
            else "evaluation_smoke_only"
        )
    }
    if candidate:
        metadata.update(
            {
                "published": published,
                "activated": activated,
                "model_revision": revision,
            }
        )
        (package / "candidate.json").write_text(
            json.dumps(
                {
                    "state": "published" if published else "candidate",
                    "published": published,
                    "activated": activated,
                    "model_revision": revision,
                }
            ),
            encoding="utf-8",
        )
    (package / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "acceptance_status": "development_only",
                "formal_eligible": False,
                "metadata": metadata,
                "payloads": {"model.safetensors": {"sha256": revision}},
            }
        ),
        encoding="utf-8",
    )
    return package


def _development_binding(tmp_path: Path, revision: str, name: str = "binding") -> Path:
    binding = tmp_path / f"{name}.json"
    binding.write_text(
        json.dumps(
            {
                "schema_version": "forcesmolvla-live-deployment-binding-v1",
                "artifact_status": "approved",
                "model_sha256": revision,
                "approval": {"status": "approved"},
            }
        ),
        encoding="utf-8",
    )
    return binding


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


def test_policy_execute_requires_explicit_flag_and_approved_revision_binding(
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
            "explicit_development_policy_execution_smoke_flag",
            "approved_development_deployment_binding",
        ],
    }
    with pytest.raises(IntegratedCaptureError, match="EXPLICIT_FLAG_REQUIRED"):
        build_capture_contract(
            mode="policy-execute",
            session_id="session-real-1",
            episode_id="episode_000001",
            policy_revision=BASELINE_POLICY_REVISION,
            policy_epoch=2,
            reset_generation=3,
            takeover_generation=4,
            deployment_binding=BASELINE_DEPLOYMENT_BINDING,
        )
    with pytest.raises(IntegratedCaptureError, match="DEVELOPMENT_REVISION_MISMATCH"):
        build_capture_contract(
            mode="policy-execute",
            session_id="session-real-1",
            episode_id="episode_000000",
            policy_revision="wrong-revision",
            policy_epoch=0,
            reset_generation=0,
            takeover_generation=0,
            deployment_binding=BASELINE_DEPLOYMENT_BINDING,
            allow_development_policy_execution_smoke=True,
        )
    policy = build_capture_contract(
        mode="policy-execute",
        session_id="session-real-1",
        episode_id="episode_000000",
        policy_revision=BASELINE_POLICY_REVISION,
        policy_epoch=0,
        reset_generation=0,
        takeover_generation=0,
        deployment_binding=BASELINE_DEPLOYMENT_BINDING,
        allow_development_policy_execution_smoke=True,
    )
    assert policy.actual_action_source == "policy"
    assert policy.policy_execution is True
    assert policy.formal_replay is policy.real_online_r is False
    assert policy.controller_process_count == 1
    assert policy.deploy_controller is False


def test_development_package_accepts_baseline_and_published_inactive_candidate(
    tmp_path: Path,
) -> None:
    baseline_revision = "a" * 64
    baseline = _development_package(tmp_path, baseline_revision, candidate=False)
    assert validate_development_policy_package(baseline, baseline_revision) == {
        "kind": "baseline_evaluation_smoke",
        "activated": False,
    }

    candidate_revision = "b" * 64
    candidate = _development_package(
        tmp_path, candidate_revision, candidate=True, activated=False
    )
    assert validate_development_policy_package(candidate, candidate_revision) == {
        "kind": "published_development_candidate",
        "activated": False,
    }

    unpublished = _development_package(
        tmp_path, "c" * 64, candidate=True, published=False
    )
    with pytest.raises(
        IntegratedCaptureError, match="PUBLISHED_DEVELOPMENT_CANDIDATE_REQUIRED"
    ):
        validate_development_policy_package(unpublished, "c" * 64)

    r5 = tmp_path / "r5"
    r5.mkdir()
    (r5 / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "acceptance_status": "development_only",
                "formal_eligible": False,
                "metadata": {},
                "payloads": {"model.safetensors": {"sha256": "d" * 64}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        IntegratedCaptureError, match="APPROVED_DEVELOPMENT_PACKAGE_REQUIRED"
    ):
        validate_development_policy_package(r5, "d" * 64)


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


def test_policy_execute_ack_binds_lineage_current_next_and_takeover() -> None:
    contract = build_capture_contract(
        mode="policy-execute",
        session_id="session-policy-1",
        episode_id="episode_000000",
        policy_revision=BASELINE_POLICY_REVISION,
        policy_epoch=0,
        reset_generation=0,
        takeover_generation=0,
        deployment_binding=BASELINE_DEPLOYMENT_BINDING,
        allow_development_policy_execution_smoke=True,
    )
    ledger = IntegratedCaptureLedger(contract)
    _observation(ledger)
    request = _request()
    request_record = ledger.record_policy_request(
        request,
        observation_id="observation-1",
        recorded_monotonic_ns=1_000_000_010,
    )
    result = ledger.record_policy_result(
        request, _result(), recorded_monotonic_ns=1_010_000_000,
    )
    dispatch = ledger.bind_policy_dispatch(result["result_id"])
    assert dispatch["chunk_id"] == request_record["chunk_id"]
    names = (
        "measured_tcp_pose", "wrench_notch_sensor", "gripper_state",
        "external_camera", "wrist_camera",
    )
    ledger.record_observation(
        observation_id="observation-2",
        t_ref_ns=1_030_000_000,
        stream_timestamps_ns={name: 1_029_000_000 for name in names},
        stream_ids={name: f"{name}:record-2" for name in names},
    )
    ack = ledger.record_actual_action_ack(
        ack_id="policy-ack-1",
        observation_id="observation-1",
        next_observation_id="observation-2",
        receive_monotonic_ns=1_020_000_000,
        actual_action_source="policy",
        policy_result_id=result["result_id"],
        proposal_id=result["proposal_id"],
        accepted_absolute7=[0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.085],
    )
    assert ack["current_observation_id"] == "observation-1"
    assert ack["next_observation_id"] == "observation-2"
    assert ack["policy_executed_transition"] is True
    intervention = ledger.record_intervention(
        event="intervention_start",
        policy_epoch=1,
        receive_monotonic_ns=1_040_000_000,
        safe_action={"arbitration": {"event": "intervention_start"}},
    )
    assert intervention["old_policy_chunk_invalidated"] is True
    with pytest.raises(IntegratedCaptureError, match="POLICY_LINEAGE_STALE_GENERATION"):
        ledger.bind_policy_dispatch(result["result_id"])
    ledger.record_intervention(
        event="intervention_end",
        policy_epoch=1,
        receive_monotonic_ns=1_050_000_000,
        safe_action={"arbitration": {"event": "intervention_end"}},
    )
    seal = ledger.seal_episode(
        seal_id="policy-seal-1",
        sealed_monotonic_ns=1_060_000_000,
        terminal_observation_id="observation-2",
    )
    assert seal["executed_action_source"] == "policy"
    assert seal["policy_execution"] is True
    assert seal["formal_replay"] is seal["real_online_r"] is False
    assert seal["learner_started"] is False


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

    revision = "f" * 64
    package = _development_package(tmp_path, revision, candidate=True)
    binding = _development_binding(tmp_path, revision)
    profile = tmp_path / "candidate-profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "forcesmolvla-deployment-profile-v1",
                "artifact_status": "development_only",
                "checkpoint": str(package),
                "deployment_binding": str(binding),
            }
        ),
        encoding="utf-8",
    )
    policy_command = [
        sys.executable,
        str(ROOT / "tools/run_stage3_integrated_capture.py"),
        "--mode",
        "policy-execute",
        "--root",
        str(tmp_path / "capture"),
        "--task",
        "task",
        "--session-id",
        "session-1",
        "--episode-id",
        "episode_000000",
        "--policy-revision",
        revision,
        "--deployment-profile",
        str(profile),
    ]
    blocked = subprocess.run(
        policy_command,
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert blocked.returncode == 2
    assert "POLICY_EXECUTE_EXPLICIT_FLAG_REQUIRED" in blocked.stdout
    assert json.loads(blocked.stdout)["requested_mode_semantics"]["policy_execution"] is True
    enabled = subprocess.run(
        [*policy_command, "--allow-development-policy-execution-smoke"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert enabled.returncode == 0, enabled.stdout + enabled.stderr
    payload = json.loads(enabled.stdout)
    assert payload["status"] == "VALIDATED_NOT_LAUNCHED"
    assert payload["robot_or_ros_started"] is False
    assert payload["contract"]["policy_execution"] is True
    assert payload["contract"]["deploy_controller"] is False
    assert payload["contract"]["deployment_binding"] == str(binding.resolve())

    other_binding = _development_binding(tmp_path, revision, "other-binding")
    mismatch = subprocess.run(
        [
            *policy_command,
            "--deployment-binding",
            str(other_binding),
            "--allow-development-policy-execution-smoke",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert mismatch.returncode == 2
    assert "APPROVED_DEVELOPMENT_BINDING_MISMATCH" in mismatch.stdout
