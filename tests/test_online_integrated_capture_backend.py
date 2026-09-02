from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from forcesmolvla.rft.online import integrated_capture_backend as capture_backend
from forcesmolvla.rft.online.integrated_capture import (
    IntegratedCaptureError,
    IntegratedCaptureLedger,
    RECORDER_CONTROL_CHAIN,
    build_capture_contract,
)
from forcesmolvla.rft.online.integrated_capture_backend import (
    ForbiddenPolicyPublisher,
    CaptureArtifactStore,
    IntegratedCaptureBackend,
    build_native_recorder_command,
)


ROOT = Path(__file__).parents[1]
CLIENT_SCRIPTS = Path("/home/rlc123/fr3_client_ws/scripts")
if str(CLIENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CLIENT_SCRIPTS))

from hilserl_impedance_protocol import GripperToggleAuthority  # noqa: E402

BASELINE_POLICY_REVISION = (
    "e24c1d6bb0a778921659514ac47c692b952178aa39af2601ccf0fc32bf94774d"
)
BASELINE_DEPLOYMENT_BINDING = ROOT / (
    "artifacts/development/live/"
    "task2_cycle210_policy_execution_smoke_binding.v1.json"
)


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
        "initial_policy_epoch": 2,
    }


def _policy_contract():
    return build_capture_contract(
        mode="policy-execute",
        session_id="policy-session-1",
        episode_id="episode_000000",
        policy_revision=BASELINE_POLICY_REVISION,
        policy_epoch=0,
        reset_generation=0,
        takeover_generation=0,
        deployment_binding=BASELINE_DEPLOYMENT_BINDING,
        allow_development_policy_execution_smoke=True,
    )


def test_backend_owns_exactly_one_native_recorder_control_chain(tmp_path: Path) -> None:
    backend = IntegratedCaptureBackend()
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
    assert command[command.index("--initial-policy-epoch") + 1] == "2"


def test_policy_publisher_is_a_fail_closed_non_dds_sentinel() -> None:
    publisher = ForbiddenPolicyPublisher("/fr3/hilserl/policy_action_control")
    with pytest.raises(IntegratedCaptureError, match="POLICY_PROPOSAL_PUBLISH_FORBIDDEN"):
        publisher.publish({"source": "policy"})


def test_session_manifest_waits_for_hilserl_enrichment(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"task": "task"}), encoding="utf-8")

    def finish_manifest() -> None:
        threading.Event().wait(0.03)
        path.write_text(json.dumps({
            "task": "task",
            "controller": {"name": RECORDER_CONTROL_CHAIN},
            "workspace": {"min_xyz_m": [0.0] * 3, "max_xyz_m": [1.0] * 3},
        }), encoding="utf-8")

    writer = threading.Thread(target=finish_manifest)
    writer.start()
    session = capture_backend._wait_for_session_manifest(
        path,
        SimpleNamespace(poll=lambda: None),
        capture_backend.time.monotonic() + 1.0,
    )
    writer.join()

    assert session["controller"]["name"] == RECORDER_CONTROL_CHAIN


def test_proposal_and_human_ack_are_separate_and_cannot_claim_execution(
    tmp_path: Path,
) -> None:
    store = CaptureArtifactStore(tmp_path / "sidecar")
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
    backend = IntegratedCaptureBackend()
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


def test_policy_chunk_selection_uses_apply_time_rational_grid() -> None:
    actions = np.arange(350, dtype=np.float64).reshape(50, 7)
    index, selected = capture_backend._selected_chunk_action(
        actions,
        t_ref_ns=1_000_000_000,
        fps=30,
        selection_ns=1_205_000_000,
    )
    assert index == 7
    assert np.array_equal(selected, actions[7])
    with pytest.raises(IntegratedCaptureError, match="CHUNK_EXPIRED"):
        capture_backend._selected_chunk_action(
            actions,
            t_ref_ns=1_000_000_000,
            fps=30,
            selection_ns=3_000_000_000,
        )


def test_policy_execution_artifacts_require_policy_ack_and_next_observation(
    tmp_path: Path,
) -> None:
    store = CaptureArtifactStore(tmp_path / "execution-sidecar")
    proposal = {
        "actual_action_source": "policy",
        "policy_execution": True,
        "formal_replay": False,
        "real_online_r": False,
    }
    store.append("policy_execute_proposal.jsonl", proposal)
    store.append(
        "policy_execute_chunk.jsonl",
        {
            "executed_action_source": "policy",
            "action_semantics": "absolute7",
            "actions_absolute7": [[0.0] * 7 for _ in range(50)],
            "formal_replay": False,
            "real_online_r": False,
        },
    )
    transition = {
        "executed_action_source": "policy",
        "policy_executed_transition": True,
        "current_observation_id": "observation-1",
        "next_observation_id": "observation-2",
        "formal_replay": False,
        "real_online_r": False,
    }
    store.append("policy_execute_transition.jsonl", transition)
    with pytest.raises(IntegratedCaptureError, match="TRANSITION_SEMANTICS_INVALID"):
        store.append(
            "policy_execute_transition.jsonl",
            {**transition, "next_observation_id": None},
        )


def test_initial_gripper_authority_accepts_inference_only_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_ns = 1_000_000_000
    observation = SimpleNamespace(
        _lock=threading.Lock(),
        gripper_width_m=0.084,
        gripper_receive_ns=captured_ns - 50_000_000,
        shadow_initial_gripper_origin=lambda _started_ns: (
            {
                "local_goal_sequence": 1,
                "action_goal_id": "initial-open-goal",
                "accepted_monotonic_ns": 800_000_000,
                "requested_state": "OPEN",
            },
            {"outcome": "reached", "finished_monotonic_ns": 900_000_000},
        ),
    )
    recorder_args = SimpleNamespace(
        gripper_open_width_m=0.085,
        gripper_closed_width_m=0.0,
    )
    monkeypatch.setattr(capture_backend.time, "monotonic_ns", lambda: captured_ns)

    authority = capture_backend._initial_gripper_authority(
        observation,
        recorder_args,
        {"gripper_max_age_ms": None},
        _contract(),
        episode_started_ns=700_000_000,
    )

    assert authority is not None
    assert authority["feedback_age_ns"] == 50_000_000


def test_session_binding_hydrates_frames_from_active_tool_profile() -> None:
    frames = {
        "base": "fr3_link0",
        "tcp": "franka_desk_ee_tcp",
        "sensor_body": "onrobot_hexe_body_link",
        "wrench_measurement": "onrobot_fts_measurement_link",
    }
    transform = {
        "xyz_m": [0.0, 0.0, -0.195],
        "rpy_rad": [0.0, 0.0, np.pi / 2.0],
    }
    session = {
        "task": "task",
        "tool_config_hash": "tool-hash",
        "controller": {"name": RECORDER_CONTROL_CHAIN},
        "primary_alignment_clock": "upper_host_receive_monotonic_ns",
        "workspace": {"min_xyz_m": [0.0] * 3, "max_xyz_m": [1.0] * 3},
        "frames": frames,
        "tool_profile": {
            "profile": {
                "frames": frames,
                "transforms": {"tcp_to_wrench_measurement": transform},
            }
        },
    }
    metadata = {
        "tool_profile_sha256": "tool-hash",
        "model_sha256": _contract().identity.policy_revision,
        "calibration_bundle": {
            "static_transform_tcp_sensor": {
                "translation_m": transform["xyz_m"],
                "quaternion_xyzw": Rotation.from_euler(
                    "xyz", transform["rpy_rad"]
                ).as_quat().tolist(),
            }
        },
    }
    recorder_args = SimpleNamespace(
        task="task",
        workspace_min=(0.0,) * 3,
        workspace_max=(1.0,) * 3,
        base_frame=None,
        tcp_frame=None,
        sensor_body_frame=None,
        wrench_measurement_frame=None,
    )
    deploy = SimpleNamespace(
        np=np,
        Rotation=Rotation,
        quaternion_xyzw_to_matrix=lambda value: Rotation.from_quat(
            value
        ).as_matrix(),
    )

    capture_backend._validate_session_binding(
        deploy, session, metadata, _contract(), recorder_args
    )

    assert recorder_args.base_frame == frames["base"]
    assert recorder_args.tcp_frame == frames["tcp"]
    assert recorder_args.sensor_body_frame == frames["sensor_body"]
    assert recorder_args.wrench_measurement_frame == frames["wrench_measurement"]
    recorder_args.tcp_frame = "conflicting_tcp"
    with pytest.raises(IntegratedCaptureError, match="SHADOW_SESSION_FRAME_MISMATCH"):
        capture_backend._validate_session_binding(
            deploy, session, metadata, _contract(), recorder_args
        )


def test_native_camera_tail_waits_for_complete_jpeg(tmp_path: Path) -> None:
    class FakeCv2:
        IMREAD_COLOR = 1
        COLOR_BGR2RGB = 2

        def __init__(self) -> None:
            self.reads = 0

        def imread(self, _path: str, _mode: int) -> np.ndarray:
            self.reads += 1
            return np.zeros((480, 640, 3), dtype=np.uint8)

        @staticmethod
        def cvtColor(image: np.ndarray, _conversion: int) -> np.ndarray:
            return image

    episode = tmp_path / "episode"
    image = episode / "images/external/frame_000000.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\xff\xd8partial")
    cv2 = FakeCv2()
    cameras = capture_backend._NativeCameraPair(episode, cv2)

    cameras._update_role("external")
    assert cv2.reads == 0
    assert cameras._next["external"] == 0

    image.write_bytes(b"\xff\xd8complete\xff\xd9")
    cameras._update_role("external")
    assert cv2.reads == 1
    assert cameras._next["external"] == 1
    assert cameras.external.frame is not None


def test_only_transient_camera_tuple_misses_are_retryable() -> None:
    assert capture_backend._retryable_camera_error(
        RuntimeError("CAMERA_AGE_EXCEEDED: camera1_age_ms=34.118")
    )
    assert capture_backend._retryable_camera_error(
        RuntimeError("INTERCAMERA_SKEW_EXCEEDED: intercamera_skew_ms=34.0")
    )
    assert not capture_backend._retryable_camera_error(
        RuntimeError("CAMERA_TIMESTAMP_IN_FUTURE")
    )
    assert not capture_backend._retryable_camera_error(
        RuntimeError("STATE_POSE_AGE_EXCEEDED")
    )


def test_policy_execution_waits_for_native_camera_first_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"camera_ready": False, "sleeps": 0}
    cameras = SimpleNamespace(ready=lambda: state["camera_ready"])
    observation = SimpleNamespace(
        cameras=cameras,
        ready=lambda: True,
        shadow_error=None,
    )
    process = SimpleNamespace(poll=lambda: None)

    def make_camera_ready(_seconds: float) -> None:
        state["sleeps"] += 1
        state["camera_ready"] = True

    monkeypatch.setattr(capture_backend.time, "sleep", make_camera_ready)
    capture_backend._wait_for_policy_observation_ready(
        observation,
        process,
        deadline=capture_backend.time.monotonic() + 1.0,
    )

    assert state == {"camera_ready": True, "sleeps": 1}
    state["camera_ready"] = False
    with pytest.raises(IntegratedCaptureError, match="RECORDER_EXITED"):
        capture_backend._wait_for_policy_observation_ready(
            observation,
            SimpleNamespace(poll=lambda: 1),
            deadline=capture_backend.time.monotonic() + 1.0,
        )


def test_policy_observation_capture_retries_filter_rewarm_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"samples": 250, "attempts": 0}
    observation = SimpleNamespace(
        cameras=SimpleNamespace(ready=lambda: True),
        ready=lambda: state["samples"] >= 250,
        shadow_error=None,
    )

    def capture():
        state["attempts"] += 1
        if state["attempts"] == 1:
            state["samples"] = 0
            raise RuntimeError("observation is incomplete")
        return "fresh"

    monkeypatch.setattr(
        capture_backend.time,
        "sleep",
        lambda _seconds: state.__setitem__("samples", state["samples"] + 1),
    )

    assert capture_backend._capture_policy_observation_when_ready(
        capture,
        observation,
        SimpleNamespace(poll=lambda: None),
        capture_backend.time.monotonic() + 1.0,
    ) == "fresh"
    assert state == {"samples": 250, "attempts": 2}


def test_filter_generation_change_discards_stale_result_then_resumes_fresh_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Observation:
        def __init__(self) -> None:
            self.generation = 0
            self.samples = 250
            self.bindings: dict[str, int] = {}
            self.cameras = SimpleNamespace(ready=lambda: True)
            self.shadow_error = None

        def request(self, request_id: str) -> dict:
            self.bindings[request_id] = self.generation
            t_ref_ns = (self.generation + 1) * 1_000_000_000
            return {
                "request_id": request_id,
                "chunk_id": f"chunk-{request_id}",
                "clock_domain_id": "upper_host_monotonic_ns",
                "provenance": {"t_ref_ns": t_ref_ns},
            }

        def assert_request_generation_current(self, request: dict) -> None:
            if self.bindings.pop(request["request_id"]) != self.generation:
                raise RuntimeError(
                    "WRENCH_FILTER_GENERATION_CHANGED_DURING_INFERENCE"
                )

        def ready(self) -> bool:
            return self.samples >= 250

    ledger = IntegratedCaptureLedger(_policy_contract())
    streams = (
        "measured_tcp_pose",
        "wrench_notch_sensor",
        "gripper_state",
        "external_camera",
        "wrist_camera",
    )

    def record_observation(observation_id: str, t_ref_ns: int) -> dict:
        return ledger.record_observation(
            observation_id=observation_id,
            t_ref_ns=t_ref_ns,
            stream_timestamps_ns={name: t_ref_ns - 1 for name in streams},
            stream_ids={name: f"{name}:{observation_id}" for name in streams},
        )

    observation = Observation()
    stale_observation = record_observation("observation-stale", 1_000_000_000)
    stale_request = observation.request("stale")
    ledger.record_policy_request(
        stale_request,
        observation_id=stale_observation["observation_id"],
        recorded_monotonic_ns=1_000_000_010,
    )
    observation.generation = 1
    observation.samples = 0
    stale_chunk = {"request_id": "stale"}
    transitions: list[dict] = []

    assert not capture_backend._inference_filter_generation_is_current(
        observation, stale_request
    )
    canceled = ledger.cancel_policy_request(
        "stale",
        reason="wrench_filter_generation_changed_during_inference",
        recorded_monotonic_ns=1_100_000_000,
    )
    stale_chunk = None

    def add_causal_sample(_seconds: float) -> None:
        observation.samples += 1

    monkeypatch.setattr(capture_backend.time, "sleep", add_causal_sample)
    capture_backend._wait_for_policy_observation_ready(
        observation,
        SimpleNamespace(poll=lambda: None),
        deadline=capture_backend.time.monotonic() + 1.0,
    )
    fresh_observation = record_observation("observation-fresh", 2_000_000_000)
    fresh_request = observation.request("fresh")
    ledger.record_policy_request(
        fresh_request,
        observation_id=fresh_observation["observation_id"],
        recorded_monotonic_ns=2_000_000_010,
    )

    assert observation.samples == 250
    assert transitions == []
    assert canceled["executed"] is False
    assert stale_chunk is None
    assert capture_backend._inference_filter_generation_is_current(
        observation, fresh_request
    )
    fresh_result = ledger.record_policy_result(
        fresh_request,
        {
            "request_id": "fresh",
            "chunk_id": "chunk-fresh",
            "t_ref_ns": 2_000_000_000,
        },
        recorded_monotonic_ns=2_100_000_000,
    )
    fresh_result_adopt_count = int(
        capture_backend._policy_context_is_current(
            ledger,
            fresh_observation,
            policy_epoch=fresh_result["policy_epoch"],
            takeover_generation=fresh_result["takeover_generation"],
            human_takeover_active=False,
            observation_id=fresh_observation["observation_id"],
        )
    )
    stale_chunk_dispatch_count = int(stale_chunk is not None)
    assert fresh_result_adopt_count == 1
    assert stale_chunk_dispatch_count == 0

    def wrench_unavailable(_request: dict) -> None:
        raise RuntimeError("WRENCH_AGE_EXCEEDED")

    with pytest.raises(RuntimeError, match="WRENCH_AGE_EXCEEDED"):
        capture_backend._inference_filter_generation_is_current(
            SimpleNamespace(assert_request_generation_current=wrench_unavailable),
            fresh_request,
        )


def test_takeover_and_episode_end_supersede_pending_policy_decision() -> None:
    class FakeObservation:
        def _safe_action_callback(self, _message: object) -> None:
            pass

    observation_type = capture_backend._shadow_observation_type(
        SimpleNamespace(LiveForceSmolObservation=FakeObservation),
        policy_execution=True,
    )
    observation = object.__new__(observation_type)
    observation._shadow_lock = threading.Lock()
    observation._shadow_safe_by_stamp = {}
    observation._policy_decisions = {}
    observation._observed_policy_epoch = 0
    observation._episode_ending = False
    observation._policy_decision_condition = threading.Condition(
        observation._shadow_lock
    )
    observation.shadow_error = None
    message = SimpleNamespace(
        data=json.dumps(
            {
                "arbitration": {
                    "policy_epoch": 1,
                    "event": "intervention_start",
                    "raw_action": {"source": "human", "sequence": 7},
                },
                "equilibrium_source_stamp_ns": 123,
            }
        )
    )

    observation._safe_action_callback(message)

    assert observation.wait_policy_decision(139, 0, 0.01) is None
    assert observation.shadow_error is None
    audit = observation.policy_audit_snapshot()
    assert audit[0]["payload"]["arbitration"]["event"] == "intervention_start"

    message.data = json.dumps(
        {
            "arbitration": {
                "policy_epoch": 1,
                "event": "episode_end",
                "raw_action": {
                    "source": "human",
                    "sequence": 8,
                    "phase": "episode_end",
                },
            },
            "equilibrium_source_stamp_ns": 124,
        }
    )
    observation._safe_action_callback(message)

    assert observation.episode_ending() is True
    assert observation.wait_policy_decision(497, 1, 0.01) is None


def test_policy_gripper_reuses_native_integer_episode_token() -> None:
    targets = [
        {
            "token": 1,
            "local_goal_sequence": 1,
            "action_goal_id": "initial-open-goal",
        }
    ]

    assert capture_backend._native_gripper_episode_token(targets) == 1
    with pytest.raises(IntegratedCaptureError, match="EPISODE_TOKEN_INVALID"):
        capture_backend._native_gripper_episode_token(
            [{**targets[0], "token": "stage3-policy-execute"}]
        )


def test_human_gripper_goal_temporarily_owns_authority() -> None:
    human = {"local_goal_sequence": 2, "action_goal_id": "human-close"}
    policy = {
        "local_goal_sequence": 1_000_000,
        "action_goal_id": "policy-open",
        "authority": "policy_execution_backend",
    }

    assert capture_backend._human_gripper_goal_active([human, policy], []) is True
    assert (
        capture_backend._human_gripper_goal_active(
            [human, policy],
            [{**human, "outcome": "stalled"}],
        )
        is False
    )


def test_stalled_close_is_an_accepted_closed_gripper_state() -> None:
    close = {
        "local_goal_sequence": 2,
        "action_goal_id": "human-close",
        "requested_closed": True,
    }
    stalled = {
        **close,
        "outcome": "stalled",
        "finished_monotonic_ns": 20,
    }
    opened = {
        "local_goal_sequence": 3,
        "action_goal_id": "human-open",
        "requested_closed": False,
    }

    assert capture_backend._completed_gripper_closed_state([close], [stalled]) is True
    assert (
        capture_backend._completed_gripper_closed_state(
            [close, opened],
            [stalled, {**opened, "outcome": "reached", "finished_monotonic_ns": 30}],
        )
        is False
    )


def _toggle_authority() -> GripperToggleAuthority:
    return GripperToggleAuthority(
        minimum_direction_delta_m=0.001,
        maximum_feedback_age_ns=100,
    )


def _complete_toggle_command(
    authority: GripperToggleAuthority,
    *,
    command_id: str,
    requested_closed: bool,
    start_width_m: float,
    end_width_m: float,
    outcome: str,
) -> bool:
    authority.observe_feedback(start_width_m, 100)
    authority.record_accepted(
        command_id,
        requested_closed=requested_closed,
        generation=1,
        accepted_monotonic_ns=110,
    )
    authority.observe_feedback(end_width_m, 120)
    return authority.record_terminal(
        command_id,
        outcome=outcome,
        current_generation=1,
        finished_monotonic_ns=130,
    )


def test_policy_closed_authority_makes_takeover_button_open() -> None:
    authority = _toggle_authority()
    assert _complete_toggle_command(
        authority,
        command_id="policy-close",
        requested_closed=True,
        start_width_m=0.085,
        end_width_m=0.04456,
        outcome="stalled",
    )
    assert authority.next_target_closed(
        current_generation=1, now_monotonic_ns=140
    ) is False


def test_policy_open_authority_makes_takeover_button_close() -> None:
    authority = _toggle_authority()
    assert _complete_toggle_command(
        authority,
        command_id="policy-open",
        requested_closed=False,
        start_width_m=0.04456,
        end_width_m=0.085,
        outcome="reached",
    )
    assert authority.next_target_closed(
        current_generation=1, now_monotonic_ns=140
    ) is True


def test_rejected_and_zero_motion_commands_do_not_update_toggle() -> None:
    authority = _toggle_authority()
    assert not _complete_toggle_command(
        authority,
        command_id="zero-close",
        requested_closed=True,
        start_width_m=0.085,
        end_width_m=0.085,
        outcome="stalled",
    )
    authority.record_accepted(
        "rejected-close",
        requested_closed=True,
        generation=1,
        accepted_monotonic_ns=140,
    )
    assert not authority.record_terminal(
        "rejected-close",
        outcome="rejected",
        current_generation=1,
        finished_monotonic_ns=150,
    )
    assert authority.next_target_closed(
        current_generation=1, now_monotonic_ns=160
    ) is True
    stale = _toggle_authority()
    stale.observe_feedback(0.085, 100)
    stale.record_accepted(
        "old-generation-close",
        requested_closed=True,
        generation=1,
        accepted_monotonic_ns=110,
    )
    stale.observe_feedback(0.04456, 120)
    assert not stale.record_terminal(
        "old-generation-close",
        outcome="stalled",
        current_generation=2,
        finished_monotonic_ns=130,
    )
    assert stale.authority_generation is None


def test_valid_stalled_close_updates_toggle_authority() -> None:
    authority = _toggle_authority()
    assert _complete_toggle_command(
        authority,
        command_id="contact-close",
        requested_closed=True,
        start_width_m=0.085,
        end_width_m=0.047,
        outcome="stalled",
    )
    assert authority.authority_closed is True


def test_stale_or_unknown_feedback_suppresses_takeover_gripper_goal() -> None:
    authority = _toggle_authority()
    authority.observe_feedback(0.085, 100)
    assert authority.next_target_closed(
        current_generation=1, now_monotonic_ns=201
    ) is None
    authority.observe_feedback(0.04456, 210)
    assert authority.next_target_closed(
        current_generation=1, now_monotonic_ns=220
    ) is None
    source = (
        CLIENT_SCRIPTS / "record_franka_hilserl_impedance.py"
    ).read_text(encoding="utf-8")
    controller = source[
        source.index("class HilserlIsolatedSpaceMouseController"):
        source.index("def _control_worker")
    ]
    assert "def _sync_gripper_toggle_for_takeover" in controller
    assert "Robotiq toggle inhibited" in controller
    assert "_send_gripper_goal(\n                    not logical_target" in controller


def test_integrated_cli_passes_shadow_runtime_binding_without_launch(
    tmp_path: Path,
) -> None:
    profile = ROOT / "configs/deployment.active.development.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_forcerft_integrated_capture.py"),
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


def test_async_runtime_binding_requires_exact_capture_identity() -> None:
    contract = _policy_contract()
    metadata = {
        "online_actor_learner": True,
        "runtime_session_id": contract.identity.session_id,
        "runtime_episode_id": contract.identity.episode_id,
        "active_actor_revision": "stage3-cycle10",
        "active_actor_model_revision": contract.identity.policy_revision,
        "learner_resume_checkpoint": "/tmp/cycle20",
        "checkpoint": "/tmp/cycle20/actor",
        "learner_started": False,
        "pending_candidate_id": "stage3-cycle21",
        "pending_candidate_published": False,
        "pending_candidate_activated": False,
        "server_persistent": True,
        "current_episode_sampling": False,
    }
    assert capture_backend._async_runtime_identity(metadata, contract) == {
        "session_id": contract.identity.session_id,
        "episode_id": contract.identity.episode_id,
        "policy_revision": contract.identity.policy_revision,
    }
    with pytest.raises(IntegratedCaptureError, match="RUNTIME_MISMATCH"):
        capture_backend._async_runtime_identity(
            {**metadata, "runtime_episode_id": "wrong"}, contract
        )
    with pytest.raises(IntegratedCaptureError, match="RUNTIME_MISMATCH"):
        capture_backend._async_runtime_identity(
            {**metadata, "checkpoint": "/tmp/other/actor"}, contract
        )


def test_async_runtime_completion_records_only_pending_candidate() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = []

        def _request(self, method, path, payload=None):
            self.calls.append((method, path, payload))
            if path == "/runtime/status":
                return {
                    "learner_state": "complete",
                    "learner_started": True,
                    "learner_critic_steps": 2,
                    "learner_actor_steps": 1,
                    "learner_polyak_steps": 2,
                    "current_episode_sampled": False,
                    "pending_candidate_published": False,
                    "pending_candidate_activated": False,
                    "actor_and_learner_concurrently_alive": True,
                    "nonfinite_count": 0,
                    "oom_count": 0,
                }
            return {}

    client = Client()
    status = capture_backend._complete_async_runtime(
        client,
        {
            "session_id": "session-1",
            "episode_id": "episode_000000",
            "policy_revision": "model-1",
        },
        deadline=1e100,
    )
    assert status["learner_critic_steps"] == 2
    assert client.calls[0][1] == "/runtime/episode-end"


def test_takeover_between_request_and_decision_invalidates_old_context() -> None:
    ledger = IntegratedCaptureLedger(_policy_contract())
    streams = (
        "measured_tcp_pose",
        "wrench_notch_sensor",
        "gripper_state",
        "external_camera",
        "wrist_camera",
    )

    def observation(observation_id: str, t_ref_ns: int) -> dict:
        return ledger.record_observation(
            observation_id=observation_id,
            t_ref_ns=t_ref_ns,
            stream_timestamps_ns={
                name: t_ref_ns - 1 for name in streams
            },
            stream_ids={
                name: f"{name}:{observation_id}" for name in streams
            },
        )

    def request(request_id: str, t_ref_ns: int) -> dict:
        return {
            "request_id": request_id,
            "chunk_id": f"chunk-{request_id}",
            "clock_domain_id": "upper_host_monotonic_ns",
            "provenance": {"t_ref_ns": t_ref_ns},
        }

    old_observation = observation("observation-old", 1_000_000_000)
    old_request = request("old-decision", 1_000_000_000)
    ledger.record_policy_request(
        old_request,
        observation_id=old_observation["observation_id"],
        recorded_monotonic_ns=1_000_000_010,
    )
    old_result = ledger.record_policy_result(
        old_request,
        {
            "request_id": "old-decision",
            "chunk_id": "chunk-old-decision",
            "t_ref_ns": 1_000_000_000,
        },
        recorded_monotonic_ns=1_100_000_000,
    )
    inflight_observation = observation("observation-inflight", 1_200_000_000)
    inflight_request = request("old-inflight", 1_200_000_000)
    ledger.record_policy_request(
        inflight_request,
        observation_id=inflight_observation["observation_id"],
        recorded_monotonic_ns=1_200_000_010,
    )

    ledger.record_intervention(
        event="intervention_start",
        policy_epoch=1,
        receive_monotonic_ns=1_300_000_000,
        safe_action={},
    )
    invalidated_requests = {"old-decision", "old-inflight"}
    current_observation = None
    old_decision_dispatch_count = int(
        capture_backend._policy_context_is_current(
            ledger,
            current_observation,
            policy_epoch=0,
            takeover_generation=0,
            human_takeover_active=True,
            observation_id=old_observation["observation_id"],
        )
    )
    inflight_result = ledger.record_policy_result(
        inflight_request,
        {
            "request_id": "old-inflight",
            "chunk_id": "chunk-old-inflight",
            "t_ref_ns": 1_200_000_000,
        },
        recorded_monotonic_ns=1_400_000_000,
    )
    old_result_adopt_count = int(
        inflight_result["request_id"] not in invalidated_requests
        and capture_backend._policy_context_is_current(
            ledger,
            current_observation,
            policy_epoch=inflight_result["policy_epoch"],
            takeover_generation=inflight_result["takeover_generation"],
            human_takeover_active=True,
        )
    )

    assert old_decision_dispatch_count == 0
    assert old_result_adopt_count == 0
    old_proposal = {
        **inflight_result,
        "invalidated_by_takeover": not bool(old_result_adopt_count),
    }
    assert old_proposal["invalidated_by_takeover"] is True
    with pytest.raises(IntegratedCaptureError, match="STALE_GENERATION"):
        ledger.bind_policy_dispatch(old_result["result_id"])

    ledger.record_intervention(
        event="intervention_end",
        policy_epoch=1,
        receive_monotonic_ns=1_500_000_000,
        safe_action={},
    )
    fresh_observation = observation("observation-fresh", 1_600_000_000)
    fresh_request = request("fresh-request", 1_600_000_000)
    fresh_request_record = ledger.record_policy_request(
        fresh_request,
        observation_id=fresh_observation["observation_id"],
        recorded_monotonic_ns=1_600_000_010,
    )
    assert (
        fresh_request_record["policy_epoch"],
        fresh_request_record["takeover_generation"],
    ) == (1, 1)
    assert capture_backend._policy_context_is_current(
        ledger,
        fresh_observation,
        policy_epoch=1,
        takeover_generation=1,
        human_takeover_active=False,
        observation_id=fresh_observation["observation_id"],
    )
    fresh_result = ledger.record_policy_result(
        fresh_request,
        {
            "request_id": "fresh-request",
            "chunk_id": "chunk-fresh-request",
            "t_ref_ns": 1_600_000_000,
        },
        recorded_monotonic_ns=1_700_000_000,
    )
    assert fresh_result["request_id"] not in invalidated_requests
    assert capture_backend._policy_context_is_current(
        ledger,
        fresh_observation,
        policy_epoch=fresh_result["policy_epoch"],
        takeover_generation=fresh_result["takeover_generation"],
        human_takeover_active=False,
    )

    toggle = _toggle_authority()
    assert _complete_toggle_command(
        toggle,
        command_id="policy-close",
        requested_closed=True,
        start_width_m=0.085,
        end_width_m=0.04456,
        outcome="stalled",
    )
    assert toggle.next_target_closed(current_generation=1, now_monotonic_ns=140) is False
