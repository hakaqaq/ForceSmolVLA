from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import socket
import sys

import numpy as np
import pytest
import yaml

from forcesmolvla.inference import (
    CLOCK_DOMAIN,
    IMAGE_SHAPE,
    PROTOCOL_VERSION,
    load_checkpoint_inference_contract,
    validate_inference_request,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from serve_policy import (  # noqa: E402
    _approved_rulespec_for_execution,
    bind_policy_action_safety,
    development_live_contract,
    development_execution_metadata,
    load_deployment_binding,
    parse_args,
)


CHECKPOINT = Path(
    "/home/rlc123/ForceSmolVLA/outputs/task2/sft/checkpoints/"
    "forcesmolvla_sft_step_010000"
)
TEST_BINDING = {
    "state_pose_max_age_ms": 250.0,
    "camera_max_age_ms": 250.0,
    "max_intercamera_skew_ms": 250.0,
    "gripper_max_age_ms": 300.0,
    "controller_ack_timeout_ms": 20.0,
    "client_source_sha256": "a" * 64,
}


def encoded_black_image() -> dict:
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    return {
        "encoding": "raw-uint8-base64",
        "shape": list(IMAGE_SHAPE),
        "data": base64.b64encode(image.tobytes()).decode("ascii"),
    }


def valid_request(contract) -> dict:
    t_ref_ns = 10_000_000_000
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "request-1",
        "chunk_id": "chunk-1",
        "client_hostname": socket.gethostname(),
        "clock_domain_id": CLOCK_DOMAIN,
        "dataset_repo_id": contract.repo_id,
        "tool_profile_sha256": contract.tool_profile_sha256,
        "calibration_id": contract.calibration_id,
        "task": "Pick up the purple ring and place it onto the red peg.",
        "state7": [0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.085],
        "wrench6": [0.0] * 6,
        "camera1": encoded_black_image(),
        "camera2": encoded_black_image(),
        "provenance": {
            "t_ref_ns": t_ref_ns,
            "tau0_ns": t_ref_ns,
            "pose_receive_monotonic_ns": t_ref_ns - 5_000_000,
            "state_pose_age_ms": 5.0,
            "camera1_receive_monotonic_ns": t_ref_ns - 8_000_000,
            "camera1_age_ms": 8.0,
            "camera2_receive_monotonic_ns": t_ref_ns - 10_000_000,
            "camera2_age_ms": 10.0,
            "intercamera_skew_ms": 2.0,
            "gripper_receive_monotonic_ns": t_ref_ns - 3_000_000,
            "wrench_receive_monotonic_ns": t_ref_ns - 1_000_000,
            "geometry_pose_source_stamp_ns": 2_000_000_000,
            "wrench_raw_source_stamp_ns": 2_005_000_000,
            "wrench_filter_output_stamp_ns": 2_005_000_000,
            "geometry_pose_age_ms": 5.0,
            "filter_warmup_complete": True,
            "wrench_geometry_valid": True,
            "session_id": "test-session",
        },
    }


def test_task2_checkpoint_contract_and_request() -> None:
    contract = load_checkpoint_inference_contract(CHECKPOINT)
    assert contract.repo_id == "local/task2_lerobotv3"
    assert contract.filter_warmup_samples == 250
    state, wrench, camera1, camera2 = validate_inference_request(
        valid_request(contract), contract
    )
    assert state.shape == (7,)
    assert wrench.shape == (6,)
    assert camera1.shape == camera2.shape == IMAGE_SHAPE


def test_future_pose_is_rejected() -> None:
    contract = load_checkpoint_inference_contract(CHECKPOINT)
    request = valid_request(contract)
    request["provenance"]["geometry_pose_source_stamp_ns"] = 2_006_000_000
    with pytest.raises(RuntimeError, match="CAUSAL_GEOMETRY"):
        validate_inference_request(request, contract)


def test_camera_order_is_exactly_bound() -> None:
    contract = load_checkpoint_inference_contract(CHECKPOINT)
    request = valid_request(contract)
    request["dataset_repo_id"] = "local/another-dataset"
    with pytest.raises(RuntimeError, match="DATASET_REPO_ID"):
        validate_inference_request(request, contract)


def test_declared_camera_age_cannot_disagree_with_timestamp() -> None:
    contract = load_checkpoint_inference_contract(CHECKPOINT)
    request = valid_request(contract)
    request["provenance"]["camera1_age_ms"] = 7.0
    with pytest.raises(RuntimeError, match="CAMERA1_AGE_MS_ARITHMETIC"):
        validate_inference_request(request, contract)


def test_development_execution_metadata_requires_explicit_server_opt_in() -> None:
    assert development_execution_metadata(None, None) == {
        "robot_execution_allowed": False,
        "robot_execution_mode": "disabled",
        "development_execution_override": False,
        "deployment_binding_sha256": None,
        "required_client_source_sha256": None,
        "controller_ack_timeout_ms": None,
    }


def test_test_only_rulespec_cannot_authorize_execution() -> None:
    with pytest.raises(PermissionError, match="TEST_ONLY"):
        _approved_rulespec_for_execution({"mode": "test_only"})


def test_server_defaults_paths_but_keeps_execution_opt_in(monkeypatch) -> None:
    root = Path(__file__).parents[1]
    monkeypatch.setattr("serve_policy.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["serve_policy.py", "--allow-development-robot-execution"],
    )
    live = parse_args()
    assert live.rulespec == root / "configs/live_action_safety.task2.development.yaml"
    assert live.deployment_binding == (
        root / "artifacts/development/live/task2_r5_live_deployment_binding.json"
    )
    assert live.trusted_deployment_binding_sha256 == hashlib.sha256(
        live.deployment_binding.read_bytes()
    ).hexdigest()

    monkeypatch.setattr(sys, "argv", ["serve_policy.py"])
    disabled = parse_args()
    assert disabled.allow_development_robot_execution is False
    assert disabled.rulespec == (
        root / "tests/fixtures/shadow_safety_thresholds.test_only.yaml"
    )
    assert disabled.deployment_binding is None


def test_task2_unsigned_development_rulespec_authorizes_only_development() -> None:
    root = Path(__file__).parents[1]
    rules = yaml.safe_load(
        (root / "configs/live_action_safety.task2.development.yaml").read_text()
    )
    _approved_rulespec_for_execution(rules)
    assert rules["mode"] == "development_only"
    assert rules["signature"]["status"] == "configuration_pending"


def test_deployment_binding_requires_independent_exact_hash(tmp_path: Path) -> None:
    binding = {
        "schema_version": "forcesmolvla-live-deployment-binding-v1",
        "artifact_status": "approved",
        "model_sha256": "1" * 64,
        "rulespec_sha256": "2" * 64,
        "server_source_sha256": "3" * 64,
        "client_source_sha256": "4" * 64,
        "state_pose_max_age_ms": 250.0,
        "camera_max_age_ms": 250.0,
        "max_intercamera_skew_ms": 250.0,
        "gripper_max_age_ms": 300.0,
        "controller_ack_timeout_ms": 20.0,
        "approval": {
            "status": "approved",
            "approval_id": "test-only-approval",
            "approver_identity": "test-only",
            "approver_role": "test-only",
            "approved_at": "2026-01-01T00:00:00Z",
        },
    }
    path = tmp_path / "binding.json"
    encoded = json.dumps(binding, sort_keys=True).encode()
    path.write_bytes(encoded)
    trusted = hashlib.sha256(encoded).hexdigest()
    loaded, actual = load_deployment_binding(
        path,
        trusted,
        model_sha256="1" * 64,
        rulespec_sha256="2" * 64,
        server_source_sha256="3" * 64,
    )
    assert loaded == binding
    assert actual == trusted
    with pytest.raises(PermissionError, match="TRUST_ANCHOR"):
        load_deployment_binding(
            path,
            "f" * 64,
            model_sha256="1" * 64,
            rulespec_sha256="2" * 64,
            server_source_sha256="3" * 64,
        )


def test_approved_development_rules_use_frozen_intrinsic_parser() -> None:
    root = Path(__file__).parents[1]
    rules = yaml.safe_load(
        (root / "tests/fixtures/shadow_safety_thresholds.test_only.yaml").read_text()
    )
    rules["mode"] = "development_only"
    rules["artifact_status"] = "development_only"
    rules["acceptance_status"] = "development_only"
    rules["approval"].update(
        {
            "status": "approved",
            "approval_id": "test-only",
            "approver_identity": "test-only",
            "approver_role": "test-only",
            "approved_at": "2026-01-01T00:00:00Z",
        }
    )
    rules["signature"]["status"] = "configuration_pending"
    for rule in rules["rules"]:
        rule["threshold"]["approval_status"] = "approved"

    class PolicyProbe:
        _action_safety_profile = None

    policy = PolicyProbe()
    bind_policy_action_safety(
        policy,
        rules,
        rules_sha256="a" * 64,
        approved_development_execution=True,
    )
    assert policy._action_safety_profile.mode == "development_only"
    assert policy._action_safety_profile.rules_sha256 == "a" * 64
    assert development_execution_metadata(TEST_BINDING, "b" * 64) == {
        "robot_execution_allowed": True,
        "robot_execution_mode": "approved_binding_supervised_development",
        "development_execution_override": True,
        "deployment_binding_sha256": "b" * 64,
        "required_client_source_sha256": "a" * 64,
        "controller_ack_timeout_ms": 20.0,
    }


def test_development_robot_execution_uses_latest_frame_camera_semantics() -> None:
    checkpoint_contract = load_checkpoint_inference_contract(CHECKPOINT)
    assert development_live_contract(checkpoint_contract, None) is checkpoint_contract
    live_contract = development_live_contract(checkpoint_contract, TEST_BINDING)
    assert checkpoint_contract.camera_max_age_ms == 34.0
    assert checkpoint_contract.max_intercamera_skew_ms == 33.0
    assert live_contract.camera_max_age_ms == TEST_BINDING["camera_max_age_ms"]
    assert live_contract.max_intercamera_skew_ms == TEST_BINDING["max_intercamera_skew_ms"]

    request = valid_request(live_contract)
    t_ref_ns = request["provenance"]["t_ref_ns"]
    request["provenance"].update(
        {
            "camera1_receive_monotonic_ns": t_ref_ns - 200_000_000,
            "camera1_age_ms": 200.0,
            "camera2_receive_monotonic_ns": t_ref_ns - 100_000_000,
            "camera2_age_ms": 100.0,
            "intercamera_skew_ms": 100.0,
        }
    )
    validate_inference_request(request, live_contract)
    with pytest.raises(RuntimeError, match="CAMERA1_AGE_MS_EXCEEDED"):
        validate_inference_request(request, checkpoint_contract)


def test_development_live_state_pose_age_is_separate_from_wrench_geometry() -> None:
    checkpoint_contract = load_checkpoint_inference_contract(CHECKPOINT)
    live_contract = development_live_contract(checkpoint_contract, TEST_BINDING)
    assert checkpoint_contract.state_pose_max_age_ms == checkpoint_contract.max_pose_age_ms
    assert live_contract.state_pose_max_age_ms == TEST_BINDING["state_pose_max_age_ms"]
    assert live_contract.max_pose_age_ms == checkpoint_contract.max_pose_age_ms

    request = valid_request(live_contract)
    t_ref_ns = request["provenance"]["t_ref_ns"]
    request["provenance"].update(
        {
            "pose_receive_monotonic_ns": t_ref_ns - 200_000_000,
            "state_pose_age_ms": 200.0,
        }
    )
    validate_inference_request(request, live_contract)
    with pytest.raises(RuntimeError, match="STATE_POSE_AGE_MS_EXCEEDED"):
        validate_inference_request(request, checkpoint_contract)

    request["provenance"]["geometry_pose_age_ms"] = 20.0
    with pytest.raises(RuntimeError, match="GEOMETRY_POSE_AGE_MS_EXCEEDED"):
        validate_inference_request(request, live_contract)
