from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy
from types import SimpleNamespace
import threading
import time

import pytest

from forcesmolvla.rft.stage3.gripper_provenance import (
    GripperGeneration,
    GripperProvenanceError,
)
from forcesmolvla.rft.stage3.policy_lineage import (
    InitialGripperAuthority,
    PolicyLineageAudit,
    PolicyLineageError,
)


ROOT = Path(__file__).parents[1]
PRODUCTION_DEPLOY = Path(
    "/home/rlc123/fr3_client_ws/scripts/deploy_forcesmolvla.py"
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


def _audit() -> PolicyLineageAudit:
    return PolicyLineageAudit(
        episode_id="dataset/episode_000001",
        policy_revision="model-sha-1",
        reset_generation=1,
    )


def test_real_request_result_identity_closes_without_fake_ack() -> None:
    audit = _audit()
    audit.record_request(
        _request(),
        policy_epoch=2,
        takeover_generation=2,
        recorded_monotonic_ns=1_000_000_010,
    )
    result = audit.record_result(
        _request(), _result(), recorded_monotonic_ns=1_100_000_000
    )
    fields = audit.bind_dispatch(
        result, policy_epoch=2, takeover_generation=2
    )
    assert fields["request_id"] == "request-1"
    assert fields["result_id"] == "policy-result:request-1"
    assert fields["proposal_id"] == "policy-proposal:request-1"
    assert fields["chunk_id"] == "chunk-1"
    assert fields["policy_revision"] == "model-sha-1"
    assert fields["reset_generation"] == 1
    assert "pose_ack_id" not in fields
    assert "gripper_ack_id" not in fields


def test_result_before_request_and_mismatched_result_fail_closed() -> None:
    audit = _audit()
    with pytest.raises(PolicyLineageError, match="RESULT_BEFORE_REQUEST"):
        audit.record_result(
            _request(), _result(), recorded_monotonic_ns=1_100_000_000
        )
    audit.record_request(
        _request(),
        policy_epoch=2,
        takeover_generation=2,
        recorded_monotonic_ns=1_000_000_010,
    )
    with pytest.raises(PolicyLineageError, match="RESULT_BINDING_MISMATCH"):
        audit.record_result(
            _request(),
            {**_result(), "chunk_id": "wrong"},
            recorded_monotonic_ns=1_100_000_000,
        )


def test_takeover_generation_change_rejects_stale_result_dispatch() -> None:
    audit = _audit()
    audit.record_request(
        _request(),
        policy_epoch=2,
        takeover_generation=2,
        recorded_monotonic_ns=1_000_000_010,
    )
    result = audit.record_result(
        _request(), _result(), recorded_monotonic_ns=1_100_000_000
    )
    with pytest.raises(PolicyLineageError, match="STALE_GENERATION"):
        audit.bind_dispatch(result, policy_epoch=3, takeover_generation=3)


def _initial() -> InitialGripperAuthority:
    generation = GripperGeneration(
        episode_id="dataset/episode_000001",
        reset_generation=1,
        takeover_generation=2,
        policy_revision="model-sha-1",
        policy_epoch=2,
    )
    return InitialGripperAuthority(
        episode_id=generation.episode_id,
        origin_local_goal_sequence=1,
        origin_action_goal_id="real-ros-goal-id",
        origin_accepted_monotonic_ns=800_000_000,
        requested_state="OPEN",
        requested_width_m=0.085,
        terminal_outcome="reached",
        terminal_finished_monotonic_ns=900_000_000,
        feedback_width_m=0.084,
        feedback_state="OPEN",
        feedback_monotonic_ns=990_000_000,
        captured_monotonic_ns=1_000_000_000,
        feedback_age_ns=10_000_000,
        clock_domain_id="upper_host_monotonic",
        generation=generation,
    )


def test_initial_gripper_authority_round_trip_is_real_origin_bound() -> None:
    authority = _initial().validate(max_feedback_age_ns=100_000_000)
    restored = InitialGripperAuthority.from_mapping(authority.to_dict()).validate(
        max_feedback_age_ns=100_000_000
    )
    assert restored == authority
    assert restored.origin_action_goal_id == "real-ros-goal-id"


def test_initial_gripper_authority_rejects_missing_origin_and_stale_feedback() -> None:
    with pytest.raises(GripperProvenanceError, match="INITIAL_GRIPPER"):
        replace(_initial(), origin_action_goal_id="").validate(
            max_feedback_age_ns=100_000_000
        )
    with pytest.raises(GripperProvenanceError, match="INITIAL_GRIPPER"):
        replace(
            _initial(),
            feedback_monotonic_ns=800_000_000,
            feedback_age_ns=200_000_000,
        ).validate(max_feedback_age_ns=100_000_000)


def test_production_hook_is_explicit_opt_in_and_wrapper_adds_no_robot_behavior() -> None:
    source = PRODUCTION_DEPLOY.read_text(encoding="utf-8")
    assert "--stage3-lineage-audit" not in source
    wrapper = (ROOT / "tools/run_stage3_policy_lineage_deploy.py").read_text(
        encoding="utf-8"
    )
    assert '"--stage3-lineage-episode-id"' in wrapper
    assert "STAGE3_LINEAGE_WRAPPER_NOT_IN_APPROVED_DEPLOYMENT_BINDING" in wrapper
    assert "stage3_initial_gripper_authority" in wrapper
    assert "record_policy_selection" in wrapper
    assert "rclpy" not in wrapper
    for forbidden in (
        "move_to_recorded_home(",
        "_send_gripper_goal(",
        "publish_reference(",
        "recover_errors(",
    ):
        assert forbidden not in wrapper


def test_wrapper_binds_fake_production_request_result_selection_and_initial_lease() -> None:
    namespace = runpy.run_path(str(ROOT / "tools/run_stage3_policy_lineage_deploy.py"))

    class Observation:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.gripper_width_m = 0.085
            self.gripper_receive_ns = time.monotonic_ns()

        def request(self, metadata: dict) -> dict:
            del metadata
            now = time.monotonic_ns()
            return {
                "request_id": "request-wrapper",
                "chunk_id": "chunk-wrapper",
                "clock_domain_id": "upper_host_monotonic_ns",
                "provenance": {"t_ref_ns": now},
            }

    class Output:
        def __init__(self) -> None:
            self.selections: list[dict] = []
            self.safe: list[dict] = []

        def record_policy_selection(self, sequence: int, payload: dict) -> None:
            del sequence
            self.selections.append(dict(payload))

        def enqueue_safe_action(self, payload: dict) -> None:
            self.safe.append(payload)

    class Bridge:
        def __init__(self, output: Output) -> None:
            self.output = output
            self.arbiter = SimpleNamespace(policy_epoch=0)
            self._forcesmol_sequence = 0

        def submit_absolute_chunk(self, actions, **kwargs) -> int:
            del actions, kwargs
            sequence = self._forcesmol_sequence
            self._forcesmol_sequence += 1
            self.output.record_policy_selection(
                sequence,
                {
                    "action_index": 3,
                    "selected_absolute_action7": [0.0] * 7,
                },
            )
            return sequence

    class Publisher:
        def __init__(self) -> None:
            self.messages: list[object] = []

        def publish(self, message: object) -> None:
            self.messages.append(message)

    class Controller:
        def __init__(self) -> None:
            self.args = SimpleNamespace(
                gripper_open_width_m=0.085,
                gripper_closed_width_m=0.0,
            )

        def create_publisher(self, *args) -> Publisher:
            del args
            return Publisher()

    class String:
        data = ""

    deploy = SimpleNamespace(
        validate_metadata=lambda metadata: None,
        validate_response=lambda result, request, workspace: [[0.0] * 7] * 50,
        LiveForceSmolObservation=Observation,
        ForceSmolControlOutput=Output,
        ForceSmolActionBridge=Bridge,
        ForceSmolHeadlessRobotController=Controller,
        String=String,
        forcevla=SimpleNamespace(
            collector=SimpleNamespace(
                GRIPPER_TARGET_TOPIC="target",
                GRIPPER_STATUS_TOPIC="status",
            )
        ),
    )
    namespace["_install_audit"](
        deploy, episode_id="dataset/episode_000001"
    )
    deploy.validate_metadata({"model_sha256": "model-sha", "gripper_max_age_ms": 100.0})
    observation = deploy.LiveForceSmolObservation()
    output = deploy.ForceSmolControlOutput()
    bridge = deploy.ForceSmolActionBridge(output)
    controller = deploy.ForceSmolHeadlessRobotController()
    request = observation.request({})
    result = {
        "request_id": request["request_id"],
        "chunk_id": request["chunk_id"],
        "t_ref_ns": request["provenance"]["t_ref_ns"],
    }
    deploy.validate_response(result, request, {})
    assert bridge.submit_absolute_chunk(
        [[0.0] * 7] * 50,
        t_ref_ns=result["t_ref_ns"],
        fps=30,
        policy_epoch=0,
        chunk_id=result["chunk_id"],
    ) == 0
    selection = output.selections[0]
    assert selection["request_id"] == "request-wrapper"
    assert selection["result_id"] == "policy-result:request-wrapper"
    assert selection["proposal_id"] == "policy-proposal:request-wrapper"
    assert selection["dispatch_sequence"] == 0
    assert selection["selected_index"] == 3

    accepted_ns = time.monotonic_ns() - 10_000_000
    metadata = {
        "local_goal_sequence": 1,
        "action_goal_id": "real-origin-goal",
        "accepted_monotonic_ns": accepted_ns,
        "requested_state": "OPEN",
    }
    controller._on_gripper_goal_accepted(metadata)
    controller._on_gripper_goal_terminal(metadata, "reached")
    output.enqueue_safe_action(
        {
            "arbitration": {
                "raw_action": {"phase": "episode_start", "source": "human"}
            }
        }
    )
    initial = output.safe[0]["stage3_initial_gripper_authority"]
    assert initial["origin_action_goal_id"] == "real-origin-goal"


def test_wrapper_refuses_execute_until_included_in_approved_binding() -> None:
    namespace = runpy.run_path(str(ROOT / "tools/run_stage3_policy_lineage_deploy.py"))
    with pytest.raises(PermissionError, match="NOT_IN_APPROVED_DEPLOYMENT_BINDING"):
        namespace["_wrapper_args"](
            ["--stage3-lineage-episode-id", "dataset/episode_000001", "--execute"]
        )
