"""Recorder-owned integrated shadow and development policy-execution capture."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import importlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping

from .gripper_provenance import GripperGeneration
from .integrated_capture import (
    CaptureBackendCapabilities,
    IntegratedCaptureContract,
    IntegratedCaptureError,
    IntegratedCaptureLedger,
    RECORDER_CONTROL_CHAIN,
    RECORDER_ENTRY,
    validate_development_policy_package,
)
from .policy_lineage import InitialGripperAuthority, UPPER_CLOCK_DOMAIN


SHADOW_BACKEND_SCHEMA = "forcesmolvla-stage3-integrated-shadow-backend-v1"
POLICY_EXECUTION_BACKEND_SCHEMA = (
    "forcesmolvla-stage3-integrated-policy-execution-backend-v1"
)
DETECTOR_CONTRACT = Path(
    "/home/rlc123/ForceSmolVLA/"
    "configs/stage3_reward_terminal_contract.v1.development.json"
)
RETRYABLE_CAMERA_ERRORS = (
    "CAMERA_AGE_EXCEEDED:",
    "INTERCAMERA_SKEW_EXCEEDED:",
)
EXTERNAL_SCRIPTS = Path("/home/rlc123/fr3_client_ws/scripts")
DEFAULT_DEPLOYMENT_PROFILE = Path(
    "/home/rlc123/ForceSmolVLA/configs/deployment.active.development.json"
)


def _async_runtime_identity(
    metadata: Mapping[str, Any], contract: IntegratedCaptureContract
) -> dict[str, Any]:
    if (
        metadata.get("stage3_async_actor_learner") is not True
        or metadata.get("runtime_session_id") != contract.identity.session_id
        or metadata.get("runtime_episode_id") != contract.identity.episode_id
        or metadata.get("active_actor_model_revision")
        != contract.identity.policy_revision
        or not isinstance(metadata.get("active_actor_revision"), str)
        or not metadata["active_actor_revision"]
        or not isinstance(metadata.get("learner_resume_checkpoint"), str)
        or not metadata["learner_resume_checkpoint"]
        or not isinstance(metadata.get("pending_candidate_id"), str)
        or not metadata["pending_candidate_id"]
        or metadata.get("pending_candidate_published") is not False
        or metadata.get("pending_candidate_activated") is not False
        or metadata.get("learner_started") is not False
    ):
        raise IntegratedCaptureError("ASYNC_POLICY_LEARNER_RUNTIME_MISMATCH")
    return {
        "session_id": contract.identity.session_id,
        "episode_id": contract.identity.episode_id,
        "policy_revision": contract.identity.policy_revision,
    }


def _complete_async_runtime(
    client: Any,
    identity: Mapping[str, Any],
    *,
    deadline: float,
) -> dict[str, Any]:
    client._request("POST", "/runtime/episode-end", dict(identity))
    while True:
        status = client._request("GET", "/runtime/status")
        state = status.get("learner_state")
        if state == "failed":
            raise IntegratedCaptureError(
                f"ASYNC_POLICY_LEARNER_FAILED:{status.get('learner_error')}"
            )
        if state == "complete":
            break
        if time.monotonic() >= deadline:
            raise IntegratedCaptureError("ASYNC_POLICY_LEARNER_COMPLETION_TIMEOUT")
        time.sleep(0.05)
    if (
        status.get("learner_started") is not True
        or int(status.get("learner_critic_steps", -1)) != 2
        or int(status.get("learner_actor_steps", -1)) != 1
        or int(status.get("learner_polyak_steps", -1)) != 2
        or status.get("current_episode_sampled") is not False
        or status.get("pending_candidate_published") is not False
        or status.get("pending_candidate_activated") is not False
        or int(status.get("nonfinite_count", -1)) != 0
        or int(status.get("oom_count", -1)) != 0
        or status.get("actor_and_learner_concurrently_alive") is not True
    ):
        raise IntegratedCaptureError("ASYNC_POLICY_LEARNER_COMPLETION_INVALID")
    return status


class ForbiddenPolicyPublisher:
    """Sentinel used instead of creating a DDS policy-action publisher."""

    def __init__(self, topic: str) -> None:
        self.topic = topic

    def publish(self, _message: Any) -> None:
        raise IntegratedCaptureError("POLICY_PROPOSAL_PUBLISH_FORBIDDEN")


def build_native_recorder_command(arguments: Mapping[str, Any]) -> list[str]:
    """Build the sole robot-owning process command without a deploy controller."""

    try:
        root = Path(str(arguments["root"])).expanduser().resolve()
        task = str(arguments["task"]).strip()
        episodes = int(arguments["episodes"])
        episode_time = float(arguments["episode_time"])
        tool_profile = str(arguments["tool_profile"]).strip()
        initial_policy_epoch = int(arguments["initial_policy_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise IntegratedCaptureError("SHADOW_RECORDER_ARGUMENTS_INVALID") from error
    if (
        not task
        or not tool_profile
        or episodes != 1
        or episode_time <= 0.0
        or initial_policy_epoch < 0
    ):
        raise IntegratedCaptureError("SHADOW_RECORDER_ARGUMENTS_INVALID")
    return [
        sys.executable,
        str(RECORDER_ENTRY),
        "--root",
        str(root),
        "--task",
        task,
        "--episodes",
        "1",
        "--episode-time",
        str(episode_time),
        "--tool-profile",
        tool_profile,
        "--initial-policy-epoch",
        str(initial_policy_epoch),
    ]


class ShadowArtifactStore:
    """Append-only integrated artifacts with mode-specific fail-closed rows."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.streams = directory / "streams"
        self.streams.mkdir(parents=True, exist_ok=False)

    def write(self, name: str, value: Mapping[str, Any]) -> Path:
        path = self.streams / name
        path.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def append(self, name: str, value: Mapping[str, Any]) -> Path:
        row = dict(value)
        if name == "policy_shadow_proposal.jsonl":
            if (
                row.get("actual_action_source") != "human"
                or row.get("policy_execution") is not False
                or row.get("executed") is not False
                or row.get("real_online_r") is not False
            ):
                raise IntegratedCaptureError("SHADOW_PROPOSAL_SEMANTICS_INVALID")
        if name == "policy_shadow_human_ack.jsonl":
            if (
                row.get("actual_action_source") != "human"
                or row.get("policy_result_id") is not None
                or row.get("proposal_id") is not None
                or row.get("policy_executed_transition") is not False
                or row.get("real_online_r") is not False
            ):
                raise IntegratedCaptureError("SHADOW_HUMAN_ACK_SEMANTICS_INVALID")
        if name == "policy_execute_proposal.jsonl":
            if (
                row.get("actual_action_source") != "policy"
                or row.get("policy_execution") is not True
                or row.get("formal_replay") is not False
                or row.get("real_online_r") is not False
            ):
                raise IntegratedCaptureError("POLICY_EXECUTE_PROPOSAL_SEMANTICS_INVALID")
        if name == "policy_execute_chunk.jsonl":
            actions = row.get("actions_absolute7")
            if (
                row.get("executed_action_source") != "policy"
                or row.get("action_semantics") != "absolute7"
                or not isinstance(actions, list)
                or len(actions) != 50
                or any(not isinstance(action, list) or len(action) != 7 for action in actions)
                or row.get("formal_replay") is not False
                or row.get("real_online_r") is not False
            ):
                raise IntegratedCaptureError("POLICY_EXECUTE_CHUNK_SEMANTICS_INVALID")
        if name == "policy_execute_transition.jsonl":
            if (
                row.get("executed_action_source") != "policy"
                or row.get("policy_executed_transition") is not True
                or not row.get("current_observation_id")
                or not row.get("next_observation_id")
                or row.get("formal_replay") is not False
                or row.get("real_online_r") is not False
            ):
                raise IntegratedCaptureError("POLICY_EXECUTE_TRANSITION_SEMANTICS_INVALID")
        path = self.streams / name
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return path


@dataclass
class _CameraFrame:
    frame: Any = None
    receive_ns: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._paths_by_receive: dict[int, str] = {}

    def update(self, frame: Any, receive_ns: int, relative_path: str) -> None:
        with self._lock:
            self.frame = frame
            self.receive_ns = int(receive_ns)
            self._paths_by_receive[self.receive_ns] = relative_path
            for old in list(self._paths_by_receive)[:-256]:
                self._paths_by_receive.pop(old, None)

    def path_for(self, receive_ns: int) -> str:
        with self._lock:
            try:
                return self._paths_by_receive[int(receive_ns)]
            except KeyError as error:
                raise IntegratedCaptureError(
                    "SHADOW_CAMERA_FRAME_IDENTITY_MISSING"
                ) from error


class _NativeCameraPair:
    """Read the recorder's completed JPEGs; never opens either camera device."""

    def __init__(self, episode_dir: Path, cv2: Any) -> None:
        self.episode_dir = episode_dir
        self.cv2 = cv2
        self.external = _CameraFrame()
        self.wrist = _CameraFrame()
        self._next = {"external": 0, "wrist": 0}
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="stage3-native-camera-tail", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def ready(self) -> bool:
        return self.external.frame is not None and self.wrist.frame is not None

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @staticmethod
    def _jpeg_complete(path: Path) -> bool:
        try:
            if path.stat().st_size < 2:
                return False
            with path.open("rb") as handle:
                handle.seek(-2, 2)
                return handle.read(2) == b"\xff\xd9"
        except OSError:
            return False

    def _update_role(self, role: str) -> None:
        latest: tuple[Path, Any] | None = None
        while True:
            index = self._next[role]
            path = self.episode_dir / "images" / role / f"frame_{index:06d}.jpg"
            if not path.is_file() or not self._jpeg_complete(path):
                break
            bgr = self.cv2.imread(str(path), self.cv2.IMREAD_COLOR)
            if bgr is None:
                break
            latest = (path, bgr)
            self._next[role] += 1
        if latest is None:
            return
        path, bgr = latest
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        relative = str(path.relative_to(self.episode_dir))
        getattr(self, role).update(rgb, time.monotonic_ns(), relative)

    def _run(self) -> None:
        try:
            while not self._stop.wait(0.005):
                self._update_role("external")
                self._update_role("wrist")
        except Exception as error:  # surfaced by LiveForceSmolObservation
            message = f"SHADOW_CAMERA_TAIL_FAILED:{type(error).__name__}:{error}"
            self.external.error = message
            self.wrist.error = message


def _load_external_modules() -> tuple[Any, Any]:
    scripts = str(EXTERNAL_SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        return (
            importlib.import_module("deploy_forcesmolvla"),
            importlib.import_module("record_franka_hilserl_impedance"),
        )
    except ImportError as error:
        raise IntegratedCaptureError("SHADOW_EXTERNAL_RUNTIME_IMPORT_FAILED") from error


def _parse_recorder_arguments(recorder: Any, command: list[str]) -> Any:
    original = sys.argv
    try:
        sys.argv = [command[1], *command[2:]]
        return recorder.parse_args()
    except SystemExit as error:
        raise IntegratedCaptureError("SHADOW_RECORDER_ARGUMENT_PARSE_FAILED") from error
    finally:
        sys.argv = original


def _native_gripper_episode_token(targets: list[Mapping[str, Any]]) -> int:
    tokens = {row.get("token") for row in targets}
    if len(tokens) != 1:
        raise IntegratedCaptureError("POLICY_EXECUTE_GRIPPER_EPISODE_TOKEN_MISSING")
    token = next(iter(tokens))
    if isinstance(token, bool) or not isinstance(token, int) or token <= 0:
        raise IntegratedCaptureError("POLICY_EXECUTE_GRIPPER_EPISODE_TOKEN_INVALID")
    return token


def _human_gripper_goal_active(
    targets: list[Mapping[str, Any]], statuses: list[Mapping[str, Any]]
) -> bool:
    completed = {
        (int(row["local_goal_sequence"]), str(row["action_goal_id"]))
        for row in statuses
    }
    return any(
        row.get("authority") != "policy_execution_backend"
        and (int(row["local_goal_sequence"]), str(row["action_goal_id"]))
        not in completed
        for row in targets
    )


def _completed_gripper_closed_state(
    targets: list[Mapping[str, Any]], statuses: list[Mapping[str, Any]]
) -> bool | None:
    target_by_key = {
        (int(row["local_goal_sequence"]), str(row["action_goal_id"])): row
        for row in targets
    }
    completed = [
        (int(status["finished_monotonic_ns"]), target)
        for status in statuses
        if status.get("outcome") in {"reached", "stalled"}
        and (
            target := target_by_key.get(
                (
                    int(status["local_goal_sequence"]),
                    str(status["action_goal_id"]),
                )
            )
        )
        is not None
    ]
    if not completed:
        return None
    return bool(max(completed, key=lambda item: item[0])[1]["requested_closed"])


def _shadow_observation_type(deploy: Any, *, policy_execution: bool = False) -> type:
    class ShadowObservation(deploy.LiveForceSmolObservation):
        def __init__(self, args: Any, cameras: Any, metadata: dict, task: str) -> None:
            self._shadow_policy_topic = str(args.policy_action_topic)
            self._shadow_lock = threading.Lock()
            self._shadow_safe_by_stamp: dict[int, dict[str, Any]] = {}
            self._shadow_acks: list[dict[str, Any]] = []
            self._shadow_targets: list[dict[str, Any]] = []
            self._shadow_statuses: list[dict[str, Any]] = []
            self._shadow_gripper_sources: deque[tuple[int, int]] = deque(maxlen=512)
            self._policy_sequence = 0
            self._policy_decisions: dict[int, dict[str, Any]] = {}
            self._observed_policy_epoch = 0
            self._episode_ending = False
            self._policy_decision_condition = threading.Condition(self._shadow_lock)
            self._reference_ack_condition = threading.Condition(self._shadow_lock)
            self._policy_gripper_condition = threading.Condition(self._shadow_lock)
            self._policy_gripper_sequence = 1_000_000
            self._policy_gripper_active = False
            self._policy_gripper_results: dict[str, dict[str, Any]] = {}
            self.shadow_error: str | None = None
            super().__init__(args, cameras, metadata, task)
            if policy_execution:
                spacemouse = deploy.forcevla.spacemouse_base
                self._policy_gripper_api = spacemouse
                self._policy_gripper_client = spacemouse.ActionClient(
                    self, spacemouse.GripperCommand, args.gripper_action
                )
                self._policy_gripper_target_publisher = self.create_publisher(
                    deploy.String,
                    deploy.forcevla.collector.GRIPPER_TARGET_TOPIC,
                    10,
                )
                self._policy_gripper_status_publisher = self.create_publisher(
                    deploy.String,
                    deploy.forcevla.collector.GRIPPER_STATUS_TOPIC,
                    20,
                )
            self.create_subscription(
                deploy.String,
                args.reference_ack_topic,
                self._shadow_reference_ack_callback,
                256,
            )
            self.create_subscription(
                deploy.String,
                deploy.forcevla.collector.GRIPPER_TARGET_TOPIC,
                self._shadow_gripper_target_callback,
                10,
            )
            self.create_subscription(
                deploy.String,
                deploy.forcevla.collector.GRIPPER_STATUS_TOPIC,
                self._shadow_gripper_status_callback,
                20,
            )

        def create_publisher(self, message_type: Any, topic: str, *args: Any, **kwargs: Any) -> Any:
            if not policy_execution and str(topic) == self._shadow_policy_topic:
                return ForbiddenPolicyPublisher(str(topic))
            return super().create_publisher(message_type, topic, *args, **kwargs)

        def _fail_shadow(self, reason: str, error: Exception) -> None:
            self.shadow_error = f"{reason}:{type(error).__name__}:{error}"

        def _safe_action_callback(self, message: Any) -> None:
            super()._safe_action_callback(message)
            try:
                payload = json.loads(message.data)
                arbitration = payload["arbitration"]
                raw = arbitration["raw_action"]
                policy_epoch = int(arbitration["policy_epoch"])
                source_stamp = payload.get("equilibrium_source_stamp_ns")
                if not isinstance(raw, dict) or policy_epoch < 0:
                    raise ValueError("invalid safe-action identity")
                record = {
                    "shadow_receive_monotonic_ns": time.monotonic_ns(),
                    "source": str(raw["source"]),
                    "payload": payload,
                }
                with self._shadow_lock:
                    self._observed_policy_epoch = max(
                        self._observed_policy_epoch, policy_epoch
                    )
                    if (
                        raw.get("source") == "human"
                        and raw.get("phase") == "episode_end"
                    ):
                        self._episode_ending = True
                    if raw.get("source") == "policy":
                        self._policy_decisions[int(raw["sequence"])] = record
                    if source_stamp is not None:
                        stamp = int(source_stamp)
                        if stamp <= 0:
                            raise ValueError("invalid safe-action source stamp")
                        self._shadow_safe_by_stamp[stamp] = record
                        for old in list(self._shadow_safe_by_stamp)[:-512]:
                            self._shadow_safe_by_stamp.pop(old, None)
                    self._policy_decision_condition.notify_all()
            except Exception as error:
                self._fail_shadow("SHADOW_SAFE_ACTION_INVALID", error)

        def _shadow_reference_ack_callback(self, message: Any) -> None:
            try:
                payload = json.loads(message.data)
                if (
                    payload.get("schema") != deploy.REFERENCE_ACK_SCHEMA
                    or payload.get("accepted") is not True
                    or int(payload.get("request_stamp_ns", 0)) <= 0
                    or not isinstance(payload.get("accepted_pose"), dict)
                ):
                    raise ValueError("invalid accepted reference ACK")
                record = {
                    "upper_receive_monotonic_ns": time.monotonic_ns(),
                    "payload": payload,
                }
                with self._shadow_lock:
                    self._shadow_acks.append(record)
                    self._reference_ack_condition.notify_all()
            except Exception as error:
                self._fail_shadow("SHADOW_REFERENCE_ACK_INVALID", error)

        def _shadow_gripper_target_callback(self, message: Any) -> None:
            try:
                payload = json.loads(message.data)
                if payload.get("event_type") != "accepted_goal":
                    return
                with self._shadow_lock:
                    self._shadow_targets.append(dict(payload))
            except Exception as error:
                self._fail_shadow("SHADOW_GRIPPER_TARGET_INVALID", error)

        def _shadow_gripper_status_callback(self, message: Any) -> None:
            try:
                payload = json.loads(message.data)
                if payload.get("event_type") != "terminal_goal":
                    return
                with self._shadow_lock:
                    self._shadow_statuses.append(dict(payload))
            except Exception as error:
                self._fail_shadow("SHADOW_GRIPPER_STATUS_INVALID", error)

        def _gripper_callback(self, message: Any) -> None:
            super()._gripper_callback(message)
            if not message.position:
                return
            source_ns = int(deploy.forcevla.raw._stamp_ns(message))
            with self._lock:
                receive_ns = int(self.gripper_receive_ns)
            with self._shadow_lock:
                self._shadow_gripper_sources.append((receive_ns, source_ns))

        def shadow_ack_snapshot(self) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
            with self._shadow_lock:
                return list(self._shadow_acks), dict(self._shadow_safe_by_stamp)

        def policy_audit_snapshot(self) -> list[dict[str, Any]]:
            with self._shadow_lock:
                return list(self._shadow_safe_by_stamp.values())

        def publish_policy_action(
            self,
            normalized_action: Any,
            *,
            policy_epoch: int,
            observation_id: str,
        ) -> int:
            if not policy_execution:
                raise IntegratedCaptureError("POLICY_EXECUTE_PUBLISH_NOT_AUTHORIZED")
            if self.episode_ending():
                raise IntegratedCaptureError("POLICY_EXECUTE_EPISODE_END")
            values = tuple(float(value) for value in normalized_action)
            if len(values) != 7 or not all(math.isfinite(value) for value in values):
                raise IntegratedCaptureError("POLICY_EXECUTE_NORMALIZED_ACTION_INVALID")
            sequence = self._policy_sequence
            self._policy_sequence += 1
            raw_action = deploy.RawAction(
                source="policy",
                sequence=sequence,
                source_monotonic_ns=time.monotonic_ns(),
                action=values,
                intervention=False,
                phase="control",
                policy_epoch=int(policy_epoch),
                observation_id=str(observation_id),
            )
            message = deploy.String()
            message.data = raw_action.to_json()
            self.action_publisher.publish(message)
            return sequence

        def wait_policy_decision(
            self, sequence: int, policy_epoch: int, timeout_s: float
        ) -> dict[str, Any] | None:
            deadline = time.monotonic() + timeout_s
            with self._policy_decision_condition:
                if self._episode_ending:
                    return None
                while sequence not in self._policy_decisions:
                    if self.shadow_error:
                        raise IntegratedCaptureError(self.shadow_error)
                    if self._episode_ending:
                        return None
                    if self._observed_policy_epoch > policy_epoch:
                        return None
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise IntegratedCaptureError(
                            f"POLICY_EXECUTE_DECISION_TIMEOUT:{sequence}"
                        )
                    self._policy_decision_condition.wait(min(remaining, 0.02))
                return self._policy_decisions.pop(sequence)["payload"]

        def wait_reference_ack(
            self, request_stamp_ns: int, timeout_s: float
        ) -> dict[str, Any] | None:
            deadline = time.monotonic() + timeout_s
            with self._reference_ack_condition:
                while True:
                    if self._episode_ending:
                        return None
                    for row in self._shadow_acks:
                        if int(row["payload"]["request_stamp_ns"]) == request_stamp_ns:
                            payload = dict(row["payload"])
                            payload["upper_receive_monotonic_ns"] = int(
                                row["upper_receive_monotonic_ns"]
                            )
                            return payload
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise IntegratedCaptureError(
                            f"POLICY_EXECUTE_POSE_ACK_TIMEOUT:{request_stamp_ns}"
                        )
                    self._reference_ack_condition.wait(min(remaining, 0.02))

        def _publish_policy_gripper_event(
            self, metadata: Mapping[str, Any], *, outcome: str | None = None
        ) -> None:
            payload = dict(metadata)
            message = deploy.String()
            if outcome is None:
                payload["event_type"] = "accepted_goal"
                publisher = self._policy_gripper_target_publisher
            else:
                payload.update(
                    event_type="terminal_goal",
                    outcome=outcome,
                )
                payload.setdefault("finished_monotonic_ns", time.monotonic_ns())
                publisher = self._policy_gripper_status_publisher
            message.data = json.dumps(payload, separators=(",", ":"))
            publisher.publish(message)

        def _policy_gripper_goal_response(
            self, future: Any, command_id: str, metadata: dict[str, Any]
        ) -> None:
            try:
                goal_handle = future.result()
                if not goal_handle.accepted:
                    raise RuntimeError("POLICY_GRIPPER_GOAL_REJECTED")
                metadata["accepted_monotonic_ns"] = time.monotonic_ns()
                goal_uuid = getattr(getattr(goal_handle, "goal_id", None), "uuid", [])
                metadata["action_goal_id"] = bytes(
                    int(value) for value in goal_uuid
                ).hex()
                if not metadata["action_goal_id"]:
                    raise RuntimeError("POLICY_GRIPPER_GOAL_ID_MISSING")
                self._publish_policy_gripper_event(metadata)
                goal_handle.get_result_async().add_done_callback(
                    lambda completed: self._policy_gripper_terminal(
                        completed, command_id, metadata
                    )
                )
            except Exception as error:
                with self._policy_gripper_condition:
                    self._policy_gripper_active = False
                    self._policy_gripper_results[command_id] = {
                        "error": f"{type(error).__name__}:{error}"
                    }
                    self._policy_gripper_condition.notify_all()

        def _policy_gripper_terminal(
            self, future: Any, command_id: str, metadata: dict[str, Any]
        ) -> None:
            outcome = "result_error"
            try:
                result = future.result().result
                outcome = (
                    "reached" if result.reached_goal
                    else "stalled" if result.stalled
                    else "not_reached"
                )
            except Exception:
                pass
            terminal = {
                **metadata,
                "outcome": outcome,
                "finished_monotonic_ns": time.monotonic_ns(),
            }
            self._publish_policy_gripper_event(terminal, outcome=outcome)
            with self._policy_gripper_condition:
                self._policy_gripper_active = False
                self._policy_gripper_results[command_id] = terminal
                self._policy_gripper_condition.notify_all()

        def ensure_policy_gripper_authority(
            self,
            target_width_m: float,
            *,
            command_context: Mapping[str, Any],
            timeout_s: float,
        ) -> dict[str, Any]:
            if not policy_execution:
                raise IntegratedCaptureError("POLICY_EXECUTE_GRIPPER_NOT_AUTHORIZED")
            if self.episode_ending():
                raise IntegratedCaptureError("POLICY_EXECUTE_EPISODE_END")
            width = float(target_width_m)
            desired_closed = True if width <= 0.030 else False if width >= 0.055 else None
            if desired_closed is None:
                raise IntegratedCaptureError("POLICY_EXECUTE_GRIPPER_TARGET_AMBIGUOUS")
            with self._lock:
                feedback_width = self.gripper_width_m
                feedback_ns = int(self.gripper_receive_ns)
            if feedback_width is None or feedback_ns <= 0:
                raise IntegratedCaptureError("POLICY_EXECUTE_GRIPPER_FEEDBACK_MISSING")
            with self._policy_gripper_condition:
                if self._episode_ending:
                    raise IntegratedCaptureError("POLICY_EXECUTE_EPISODE_END")
                if self._policy_gripper_active or _human_gripper_goal_active(
                    self._shadow_targets, self._shadow_statuses
                ):
                    raise IntegratedCaptureError("POLICY_EXECUTE_GRIPPER_AUTHORITY_BUSY")
                accepted_closed = _completed_gripper_closed_state(
                    self._shadow_targets, self._shadow_statuses
                )
            feedback_closed = (
                float(feedback_width) < 0.0425
                if accepted_closed is None
                else accepted_closed
            )
            if feedback_closed == desired_closed:
                return {
                    **dict(command_context),
                    "command_required": False,
                    "requested_state": "CLOSED" if desired_closed else "OPEN",
                    "requested_width_m": 0.0 if desired_closed else 0.085,
                    "feedback_width_m": float(feedback_width),
                    "feedback_monotonic_ns": feedback_ns,
                    "authority": "existing_accepted_gripper_state",
                }
            if timeout_s <= 0.0 or not self._policy_gripper_client.server_is_ready():
                raise IntegratedCaptureError("POLICY_EXECUTE_GRIPPER_SERVER_UNAVAILABLE")
            with self._policy_gripper_condition:
                if self._episode_ending:
                    raise IntegratedCaptureError("POLICY_EXECUTE_EPISODE_END")
                active_human = _human_gripper_goal_active(
                    self._shadow_targets, self._shadow_statuses
                )
                if self._policy_gripper_active or active_human:
                    raise IntegratedCaptureError("POLICY_EXECUTE_GRIPPER_AUTHORITY_BUSY")
                episode_token = _native_gripper_episode_token(self._shadow_targets)
                sequence = self._policy_gripper_sequence
                self._policy_gripper_sequence += 1
                self._policy_gripper_active = True
            command_id = f"policy-gripper:{sequence}"
            controller_position = (
                self.args.gripper_closed_position
                if desired_closed
                else self.args.gripper_open_position
            )
            metadata = {
                **dict(command_context),
                "command_id": command_id,
                "token": episode_token,
                "local_goal_sequence": sequence,
                "requested_state": "CLOSED" if desired_closed else "OPEN",
                "requested_closed": desired_closed,
                "requested_width_m": 0.0 if desired_closed else 0.085,
                "controller_position": float(controller_position),
                "started_monotonic_ns": time.monotonic_ns(),
                "accepted_monotonic_ns": None,
                "action_goal_id": None,
                "command_required": True,
                "authority": "policy_execution_backend",
            }
            goal = self._policy_gripper_api.GripperCommand.Goal()
            goal.command.position = float(controller_position)
            goal.command.max_effort = float(self.args.gripper_max_effort)
            try:
                future = self._policy_gripper_client.send_goal_async(goal)
                future.add_done_callback(
                    lambda completed: self._policy_gripper_goal_response(
                        completed, command_id, metadata
                    )
                )
            except Exception as error:
                with self._policy_gripper_condition:
                    self._policy_gripper_active = False
                raise IntegratedCaptureError(
                    f"POLICY_EXECUTE_GRIPPER_SEND_FAILED:{type(error).__name__}:{error}"
                ) from error
            deadline = time.monotonic() + timeout_s
            with self._policy_gripper_condition:
                while command_id not in self._policy_gripper_results:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise IntegratedCaptureError(
                            f"POLICY_EXECUTE_GRIPPER_ACK_TIMEOUT:{command_id}"
                        )
                    self._policy_gripper_condition.wait(min(remaining, 0.02))
                result = self._policy_gripper_results.pop(command_id)
            if result.get("outcome") not in {"reached", "stalled"}:
                raise IntegratedCaptureError(
                    f"POLICY_EXECUTE_GRIPPER_TERMINAL_INVALID:{result}"
                )
            return result

        def human_gripper_goal_active(self) -> bool:
            with self._policy_gripper_condition:
                return _human_gripper_goal_active(
                    self._shadow_targets, self._shadow_statuses
                )

        def episode_ending(self) -> bool:
            with self._shadow_lock:
                return self._episode_ending

        def shadow_initial_gripper_origin(
            self, episode_started_ns: int
        ) -> tuple[dict[str, Any], dict[str, Any]] | None:
            with self._shadow_lock:
                targets = list(self._shadow_targets)
                statuses = list(self._shadow_statuses)
            target_by_key = {
                (int(row["local_goal_sequence"]), str(row["action_goal_id"])): row
                for row in targets
                if row.get("action_goal_id")
                and int(row.get("accepted_monotonic_ns") or 0) > 0
            }
            pairs = []
            for status in statuses:
                key = (
                    int(status.get("local_goal_sequence", 0)),
                    str(status.get("action_goal_id", "")),
                )
                target = target_by_key.get(key)
                finished = int(status.get("finished_monotonic_ns", 0))
                if (
                    target is not None
                    and status.get("outcome") in {"reached", "stalled"}
                    and 0 < finished <= episode_started_ns
                ):
                    pairs.append((target, status))
            if not pairs:
                return None
            return max(pairs, key=lambda pair: int(pair[1]["finished_monotonic_ns"]))

        def shadow_stream_ids(self, request: Mapping[str, Any]) -> dict[str, str]:
            provenance = request["provenance"]
            pose_receive = int(provenance["pose_receive_monotonic_ns"])
            gripper_receive = int(provenance["gripper_receive_monotonic_ns"])
            with self._lock:
                pose = next(
                    (row for row in reversed(self.pose_history) if int(row[1]) == pose_receive),
                    None,
                )
            with self._shadow_lock:
                gripper_source = next(
                    (
                        source
                        for receive, source in reversed(self._shadow_gripper_sources)
                        if receive == gripper_receive
                    ),
                    None,
                )
            if pose is None or gripper_source is None:
                raise IntegratedCaptureError("SHADOW_OBSERVATION_STREAM_IDENTITY_MISSING")
            camera1_ns = int(provenance["camera1_receive_monotonic_ns"])
            camera2_ns = int(provenance["camera2_receive_monotonic_ns"])
            return {
                "measured_tcp_pose": f"source:{int(pose[0])}@receive:{pose_receive}",
                "wrench_notch_sensor": (
                    f"source:{int(provenance['wrench_raw_source_stamp_ns'])}"
                    f"@receive:{int(provenance['wrench_receive_monotonic_ns'])}"
                ),
                "gripper_state": f"source:{gripper_source}@receive:{gripper_receive}",
                "external_camera": self.cameras.external.path_for(camera1_ns),
                "wrist_camera": self.cameras.wrist.path_for(camera2_ns),
            }

    return ShadowObservation


def _validate_shadow_contract(contract: IntegratedCaptureContract) -> None:
    if (
        contract.mode != "shadow"
        or contract.actual_action_source != "human"
        or contract.policy_inference is not True
        or contract.policy_execution is not False
        or contract.formal_replay is not False
        or contract.real_online_r is not False
        or contract.controller_owner != "recorder"
        or contract.controller_process_count != 1
        or contract.recorder_controller is not True
        or contract.deploy_controller is not False
        or contract.control_chain_id != RECORDER_CONTROL_CHAIN
        or Path(contract.recorder_entry) != RECORDER_ENTRY
    ):
        raise IntegratedCaptureError("SHADOW_BACKEND_CONTRACT_NOT_AUTHORIZED")


def _validate_policy_execution_contract(contract: IntegratedCaptureContract) -> None:
    if (
        contract.mode != "policy-execute"
        or contract.actual_action_source != "policy"
        or contract.policy_inference is not True
        or contract.policy_execution is not True
        or contract.formal_replay is not False
        or contract.real_online_r is not False
        or contract.development_policy_execution_smoke is not True
        or not contract.deployment_binding
        or contract.controller_owner != "recorder"
        or contract.controller_process_count != 1
        or contract.recorder_controller is not True
        or contract.deploy_controller is not False
        or contract.control_chain_id != RECORDER_CONTROL_CHAIN
        or Path(contract.recorder_entry) != RECORDER_ENTRY
    ):
        raise IntegratedCaptureError("POLICY_EXECUTE_BACKEND_CONTRACT_NOT_AUTHORIZED")


def _validate_policy_execution_profile(
    deploy: Any,
    profile: Mapping[str, Any],
    metadata: Mapping[str, Any],
    contract: IntegratedCaptureContract,
) -> None:
    detector = _json(DETECTOR_CONTRACT)
    detector_smoke = detector.get("reward_gate", {}).get(
        "development_policy_execution_smoke", {}
    )
    profile_binding = Path(profile["deployment_binding"]).resolve()
    checkpoint = Path(profile["checkpoint"]).resolve()
    if (
        profile.get("artifact_status") != "development_only"
        or profile_binding != Path(str(contract.deployment_binding)).resolve()
    ):
        raise IntegratedCaptureError(
            "POLICY_EXECUTE_APPROVED_DEVELOPMENT_DEPLOYMENT_MISMATCH"
        )
    validate_development_policy_package(
        checkpoint, contract.identity.policy_revision
    )
    try:
        deploy.validate_execution_authorization(
            SimpleNamespace(
                allow_development_robot_execution=True,
                execute=True,
                trusted_deployment_binding_sha256=profile[
                    "deployment_binding_sha256"
                ],
                yes=False,
            ),
            dict(metadata),
        )
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as error:
        raise IntegratedCaptureError(
            "POLICY_EXECUTE_SERVER_AUTHORIZATION_MISMATCH"
        ) from error
    expected_metadata = {
        "model_sha256": contract.identity.policy_revision,
        "checkpoint": str(checkpoint),
        "deployment_binding_sha256": profile["deployment_binding_sha256"],
        "rulespec_mode": "development_only",
        "rulespec_approval_status": "approved",
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise IntegratedCaptureError("POLICY_EXECUTE_SERVER_AUTHORIZATION_MISMATCH")
    if (
        detector_smoke.get("approved") is not True
        or detector_smoke.get("scope")
        != "single_episode_cycle210_policy_execution_smoke"
        or detector.get("reward_gate", {}).get(
            "reward_bearing_online_update_authorized"
        )
        is not False
    ):
        raise IntegratedCaptureError("POLICY_EXECUTE_DETECTOR_SCOPE_NOT_APPROVED")


def _selected_chunk_action(
    actions: Any, *, t_ref_ns: int, fps: int, selection_ns: int
) -> tuple[int, Any]:
    if t_ref_ns <= 0 or fps != 30 or selection_ns < t_ref_ns:
        raise IntegratedCaptureError("POLICY_EXECUTE_CHUNK_TIME_INVALID")
    action_index = (
        (selection_ns - t_ref_ns) * fps + 1_000_000_000 - 1
    ) // 1_000_000_000
    if action_index >= 50:
        raise IntegratedCaptureError("POLICY_EXECUTE_CHUNK_EXPIRED")
    return int(action_index), actions[int(action_index)]


def _retryable_camera_error(error: RuntimeError) -> bool:
    return str(error).startswith(RETRYABLE_CAMERA_ERRORS)


def _wait_for_policy_observation_ready(
    observation: Any, process: subprocess.Popen[Any], deadline: float
) -> None:
    while not (observation.ready() and observation.cameras.ready()):
        if observation.shadow_error:
            raise IntegratedCaptureError(observation.shadow_error)
        return_code = process.poll()
        if return_code is not None:
            raise IntegratedCaptureError(
                f"POLICY_EXECUTE_RECORDER_EXITED_BEFORE_OBSERVATION_READY:{return_code}"
            )
        if time.monotonic() >= deadline:
            raise IntegratedCaptureError("POLICY_EXECUTE_OBSERVATION_READY_TIMEOUT")
        time.sleep(0.005)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegratedCaptureError(f"SHADOW_JSON_OBJECT_REQUIRED:{path}")
    return value


def _wait_for_path(path: Path, process: subprocess.Popen[Any], deadline: float) -> None:
    while not path.is_file():
        return_code = process.poll()
        if return_code is not None:
            raise IntegratedCaptureError(
                f"SHADOW_RECORDER_EXITED_BEFORE_READY:{return_code}:{path.name}"
            )
        if time.monotonic() >= deadline:
            raise IntegratedCaptureError(f"SHADOW_BACKEND_START_TIMEOUT:{path.name}")
        time.sleep(0.02)


def _validate_session_binding(
    deploy: Any,
    session: Mapping[str, Any],
    metadata: Mapping[str, Any],
    contract: IntegratedCaptureContract,
    recorder_args: Any,
) -> None:
    if (
        session.get("task") != recorder_args.task
        or session.get("tool_config_hash") != metadata.get("tool_profile_sha256")
        or session.get("controller", {}).get("name") != RECORDER_CONTROL_CHAIN
        or session.get("primary_alignment_clock")
        != "upper_host_receive_monotonic_ns"
        or contract.identity.policy_revision != metadata.get("model_sha256")
    ):
        raise IntegratedCaptureError("SHADOW_SESSION_POLICY_BINDING_MISMATCH")
    workspace = session.get("workspace", {})
    if (
        list(workspace.get("min_xyz_m", ())) != list(recorder_args.workspace_min)
        or list(workspace.get("max_xyz_m", ())) != list(recorder_args.workspace_max)
    ):
        raise IntegratedCaptureError("SHADOW_SESSION_WORKSPACE_MISMATCH")
    frames = session.get("frames", {})
    profile_frames = (
        session.get("tool_profile", {}).get("profile", {}).get("frames", {})
    )
    for key, attribute in (
        ("base", "base_frame"),
        ("tcp", "tcp_frame"),
        ("sensor_body", "sensor_body_frame"),
        ("wrench_measurement", "wrench_measurement_frame"),
    ):
        active_frame = frames.get(key)
        requested_frame = getattr(recorder_args, attribute)
        if (
            not isinstance(active_frame, str)
            or not active_frame
            or profile_frames.get(key) != active_frame
            or (
                requested_frame is not None
                and requested_frame != active_frame
            )
        ):
            raise IntegratedCaptureError("SHADOW_SESSION_FRAME_MISMATCH")
        setattr(recorder_args, attribute, active_frame)
    profile_transform = session["tool_profile"]["profile"]["transforms"][
        "tcp_to_wrench_measurement"
    ]
    checkpoint_transform = metadata["calibration_bundle"][
        "static_transform_tcp_sensor"
    ]
    if not (
        deploy.np.array_equal(
            deploy.np.asarray(profile_transform["xyz_m"], dtype=deploy.np.float64),
            deploy.np.asarray(
                checkpoint_transform["translation_m"], dtype=deploy.np.float64
            ),
        )
        and deploy.np.allclose(
            deploy.Rotation.from_euler(
                "xyz", profile_transform["rpy_rad"]
            ).as_matrix(),
            deploy.quaternion_xyzw_to_matrix(
                deploy.np.asarray(
                    checkpoint_transform["quaternion_xyzw"],
                    dtype=deploy.np.float64,
                )
            ),
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise IntegratedCaptureError("SHADOW_SESSION_CALIBRATION_MISMATCH")


def _initial_gripper_authority(
    observation: Any,
    recorder_args: Any,
    metadata: Mapping[str, Any],
    contract: IntegratedCaptureContract,
    episode_started_ns: int,
) -> dict[str, Any] | None:
    origin = observation.shadow_initial_gripper_origin(episode_started_ns)
    if origin is None:
        return None
    accepted, terminal = origin
    captured_ns = time.monotonic_ns()
    with observation._lock:
        feedback_width = observation.gripper_width_m
        feedback_ns = int(observation.gripper_receive_ns)
    if feedback_width is None or feedback_ns <= 0:
        return None
    state = str(accepted["requested_state"])
    requested_width = (
        recorder_args.gripper_open_width_m
        if state == "OPEN"
        else recorder_args.gripper_closed_width_m
    )
    authority = InitialGripperAuthority(
        episode_id=contract.identity.episode_id,
        origin_local_goal_sequence=int(accepted["local_goal_sequence"]),
        origin_action_goal_id=str(accepted["action_goal_id"]),
        origin_accepted_monotonic_ns=int(accepted["accepted_monotonic_ns"]),
        requested_state=state,
        requested_width_m=float(requested_width),
        terminal_outcome=str(terminal["outcome"]),
        terminal_finished_monotonic_ns=int(terminal["finished_monotonic_ns"]),
        feedback_width_m=float(feedback_width),
        feedback_state="OPEN" if float(feedback_width) >= 0.055 else "CLOSED",
        feedback_monotonic_ns=feedback_ns,
        captured_monotonic_ns=captured_ns,
        feedback_age_ns=captured_ns - feedback_ns,
        clock_domain_id=UPPER_CLOCK_DOMAIN,
        generation=GripperGeneration(
            episode_id=contract.identity.episode_id,
            reset_generation=contract.identity.reset_generation,
            takeover_generation=contract.identity.takeover_generation,
            policy_revision=contract.identity.policy_revision,
            policy_epoch=contract.identity.policy_epoch,
        ),
    )
    configured_max_age_ms = metadata.get("gripper_max_age_ms")
    maximum_age_ns = int(
        min(
            100.0
            if configured_max_age_ms is None
            else float(configured_max_age_ms),
            100.0,
        )
        * 1.0e6
    )
    try:
        return authority.validate(max_feedback_age_ns=maximum_age_ns).to_dict()
    except Exception:
        return None


def _camera_reconciliation(
    native_episode: Path, observations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for role in ("external", "wrist"):
        path = native_episode / "streams" / f"{role}_camera.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                by_path[str(row["rgb_path"])] = row
    result = []
    for observation in observations:
        for stream, role in (
            ("external_camera", "external"), ("wrist_camera", "wrist")
        ):
            relative = observation["stream_ids"][stream]
            native = by_path.get(relative)
            if native is None or native.get("role") != role:
                raise IntegratedCaptureError("SHADOW_NATIVE_CAMERA_RECONCILIATION_FAILED")
            result.append(
                {
                    "observation_id": observation["observation_id"],
                    "role": role,
                    "rgb_path": relative,
                    "policy_receive_monotonic_ns": observation[
                        "stream_timestamps_ns"
                    ][stream],
                    "native_receive_monotonic_ns": int(
                        native["receive_monotonic_ns"]
                    ),
                    "clock_domain_id": UPPER_CLOCK_DOMAIN,
                    "same_recorder_jpeg": True,
                }
            )
    return result


def _record_live_observation(
    observation: Any,
    metadata: Mapping[str, Any],
    contract: IntegratedCaptureContract,
    ledger: IntegratedCaptureLedger,
    store: ShadowArtifactStore,
    observations: list[dict[str, Any]],
    *,
    stream_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    while True:
        try:
            request = observation.request(metadata)
            break
        except RuntimeError as error:
            if not _retryable_camera_error(error):
                raise
            time.sleep(0.005)
    request["provenance"]["session_id"] = contract.identity.session_id
    observation_id = (
        f"{contract.identity.episode_id}:observation:{len(observations):06d}"
    )
    provenance = request["provenance"]
    timestamps = {
        "measured_tcp_pose": int(provenance["pose_receive_monotonic_ns"]),
        "wrench_notch_sensor": int(provenance["wrench_receive_monotonic_ns"]),
        "gripper_state": int(provenance["gripper_receive_monotonic_ns"]),
        "external_camera": int(provenance["camera1_receive_monotonic_ns"]),
        "wrist_camera": int(provenance["camera2_receive_monotonic_ns"]),
    }
    policy_epoch, takeover_generation = ledger.current_policy_generation
    record = ledger.record_observation(
        observation_id=observation_id,
        t_ref_ns=int(provenance["t_ref_ns"]),
        stream_timestamps_ns=timestamps,
        stream_ids=observation.shadow_stream_ids(request),
        policy_epoch=policy_epoch,
        takeover_generation=takeover_generation,
    )
    observations.append(record)
    store.append(stream_name, record)
    return request, record


def _policy_context_is_current(
    ledger: IntegratedCaptureLedger,
    current_observation: Mapping[str, Any] | None,
    *,
    policy_epoch: int,
    takeover_generation: int,
    human_takeover_active: bool,
    observation_id: str | None = None,
) -> bool:
    if current_observation is None or human_takeover_active:
        return False
    if ledger.current_policy_generation != (policy_epoch, takeover_generation):
        return False
    return (
        observation_id is None
        or current_observation.get("observation_id") == observation_id
    )


class IntegratedShadowBackend:
    """One native recorder with shadow inference or explicit policy smoke."""

    capabilities = CaptureBackendCapabilities(
        controller_owner="recorder",
        controller_process_count=1,
        starts_recorder_controller=True,
        starts_deploy_controller=False,
        control_chain_id=RECORDER_CONTROL_CHAIN,
        shares_observation_store=True,
        emits_episode_seal=True,
    )

    def capture(
        self,
        *,
        contract: IntegratedCaptureContract,
        ledger: IntegratedCaptureLedger,
        recorder_arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if contract.mode == "shadow":
            _validate_shadow_contract(contract)
        else:
            _validate_policy_execution_contract(contract)
        if (
            recorder_arguments.get("initial_policy_epoch")
            != contract.identity.policy_epoch
        ):
            raise IntegratedCaptureError(
                "INTEGRATED_CAPTURE_INITIAL_POLICY_EPOCH_MISMATCH"
            )
        command = build_native_recorder_command(recorder_arguments)
        try:
            return self._capture_live(contract, ledger, recorder_arguments, command)
        except IntegratedCaptureError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            raise IntegratedCaptureError(
                f"SHADOW_BACKEND_FAILED:{type(error).__name__}:{error}"
            ) from error

    def _capture_live(
        self,
        contract: IntegratedCaptureContract,
        ledger: IntegratedCaptureLedger,
        arguments: Mapping[str, Any],
        command: list[str],
    ) -> Mapping[str, Any]:
        deploy, recorder = _load_external_modules()
        root = Path(str(arguments["root"])).expanduser().resolve()
        if root.exists():
            raise IntegratedCaptureError("SHADOW_ROOT_MUST_BE_NEW")
        if contract.identity.episode_id != "episode_000000":
            raise IntegratedCaptureError("SHADOW_NEW_ROOT_REQUIRES_EPISODE_000000")
        profile_path = Path(
            str(arguments.get("deployment_profile", DEFAULT_DEPLOYMENT_PROFILE))
        ).resolve()
        profile = deploy.load_deployment_profile(profile_path)
        if str(profile["tool_profile"]) != str(arguments["tool_profile"]):
            raise IntegratedCaptureError("SHADOW_DEPLOYMENT_TOOL_PROFILE_MISMATCH")
        manifest_path = Path(profile["dataset_manifest"]).resolve()
        manifest = _json(manifest_path)
        client = deploy.PolicyHttpClient(
            str(arguments.get("policy_host", "127.0.0.1")),
            int(arguments.get("policy_port", 8000)),
            float(arguments.get("inference_timeout", 30.0)),
        )
        metadata = client.metadata()
        deploy.validate_metadata(metadata)
        if contract.mode == "policy-execute":
            _validate_policy_execution_profile(
                deploy, profile, metadata, contract
            )
        manifest_task = deploy.validate_manifest(manifest, metadata, manifest_path)
        if manifest_task != str(arguments["task"]).strip():
            raise IntegratedCaptureError("SHADOW_DEPLOYMENT_TASK_MISMATCH")
        if metadata.get("model_sha256") != contract.identity.policy_revision:
            raise IntegratedCaptureError("SHADOW_POLICY_REVISION_MISMATCH")

        async_runtime_identity = None
        if bool(arguments.get("async_learner", False)):
            if contract.mode != "policy-execute":
                raise IntegratedCaptureError("ASYNC_LEARNER_REQUIRES_POLICY_EXECUTE")
            async_runtime_identity = _async_runtime_identity(metadata, contract)

        recorder_args = _parse_recorder_arguments(recorder, command)
        start_timeout = float(arguments.get("backend_start_timeout", 180.0))
        inference_period = float(arguments.get("shadow_inference_period", 0.1))
        if start_timeout <= 0.0 or inference_period <= 0.0:
            raise IntegratedCaptureError("SHADOW_BACKEND_TIMING_INVALID")

        process: subprocess.Popen[Any] | None = None
        observation = executor = executor_thread = cameras = None
        executor_stop = threading.Event()
        rclpy_started = False
        async_runtime_started = False
        try:
            if async_runtime_identity is not None:
                client._request(
                    "POST", "/runtime/episode-start", async_runtime_identity
                )
                async_runtime_started = True
            process = subprocess.Popen(command, cwd=EXTERNAL_SCRIPTS)
            deadline = time.monotonic() + start_timeout
            session_path = root / "session.json"
            _wait_for_path(session_path, process, deadline)
            session = _json(session_path)
            _validate_session_binding(
                deploy, session, metadata, contract, recorder_args
            )
            work_episode = root / "episodes/.episode_000000.inprogress"
            final_episode = root / "episodes/episode_000000"
            cameras = _NativeCameraPair(work_episode, deploy.forcevla.cv2)
            cameras.start()

            deploy.rclpy.init()
            rclpy_started = True
            observation_type = _shadow_observation_type(
                deploy, policy_execution=contract.mode == "policy-execute"
            )
            observation = observation_type(
                recorder_args, cameras, metadata, str(arguments["task"]).strip()
            )
            if contract.mode == "shadow":
                if not isinstance(observation.action_publisher, ForbiddenPolicyPublisher):
                    raise IntegratedCaptureError("SHADOW_POLICY_PUBLISHER_WAS_CREATED")
            elif isinstance(observation.action_publisher, ForbiddenPolicyPublisher):
                raise IntegratedCaptureError("POLICY_EXECUTE_PUBLISHER_MISSING")
            executor = deploy.SingleThreadedExecutor()
            executor.add_node(observation)

            def spin() -> None:
                try:
                    while not executor_stop.is_set():
                        executor.spin_once(timeout_sec=0.02)
                except Exception as error:
                    observation.shadow_error = (
                        f"SHADOW_OBSERVATION_EXECUTOR_FAILED:"
                        f"{type(error).__name__}:{error}"
                    )

            executor_thread = threading.Thread(
                target=spin, name="stage3-shadow-observation", daemon=True
            )
            executor_thread.start()
            backend_schema = (
                SHADOW_BACKEND_SCHEMA
                if contract.mode == "shadow"
                else POLICY_EXECUTION_BACKEND_SCHEMA
            )
            integrated_manifest = {
                "schema": backend_schema,
                "contract": contract.to_dict(),
                "native_session": "session.json",
                "native_recorder_entry": str(RECORDER_ENTRY),
                "native_recorder_pid": process.pid,
                "controller_owner": "recorder",
                "controller_process_count": 1,
                "deploy_controller_started": False,
                "policy_server_started_by_backend": False,
                "policy_action_publisher_created": contract.mode == "policy-execute",
                "learner_started": bool(metadata.get("learner_started", False)),
                "learner_resume_checkpoint": metadata.get(
                    "learner_resume_checkpoint"
                ),
                "active_actor_revision": metadata.get("active_actor_revision"),
                "pending_candidate_id": metadata.get("pending_candidate_id"),
                "formal_replay_writer_started": False,
                "policy_revision_publisher_started": False,
                "clock_binding": {
                    "stage3_clock_domain_id": UPPER_CLOCK_DOMAIN,
                    "policy_request_clock_domain_id": "upper_host_monotonic_ns",
                    "native_primary_alignment_clock": session[
                        "primary_alignment_clock"
                    ],
                    "same_upper_host_monotonic_epoch": True,
                },
                "policy_metadata": {
                    "model_sha256": metadata["model_sha256"],
                    "dataset_repo_id": metadata["dataset_repo_id"],
                    "tool_profile_sha256": metadata["tool_profile_sha256"],
                    "calibration_id": metadata["calibration_id"],
                },
            }
            (root / "integrated_capture_session.json").write_text(
                json.dumps(integrated_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            episode_start_path = work_episode / "episode_start.json"
            _wait_for_path(episode_start_path, process, deadline)
            episode_start = _json(episode_start_path)
            episode_started_ns = int(episode_start["started_monotonic_ns"])
            artifact_work = root / "integrated_capture/.episode_000000.inprogress"
            store = ShadowArtifactStore(artifact_work)

            authority = None
            while authority is None:
                if observation.shadow_error:
                    raise IntegratedCaptureError(observation.shadow_error)
                authority = _initial_gripper_authority(
                    observation,
                    recorder_args,
                    metadata,
                    contract,
                    episode_started_ns,
                )
                if authority is not None:
                    break
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise IntegratedCaptureError(
                        "SHADOW_INITIAL_GRIPPER_AUTHORITY_MISSING"
                    )
                time.sleep(0.01)
            initial_lease_name = (
                "policy_shadow_initial_gripper_lease.json"
                if contract.mode == "shadow"
                else "policy_execute_initial_gripper_lease.json"
            )
            store.write(initial_lease_name, authority)

            if contract.mode == "policy-execute":
                result = self._capture_policy_execution_episode(
                    deploy=deploy,
                    process=process,
                    client=client,
                    metadata=metadata,
                    session=session,
                    recorder_args=recorder_args,
                    contract=contract,
                    ledger=ledger,
                    observation=observation,
                    store=store,
                    authority=authority,
                    episode_started_ns=episode_started_ns,
                    work_episode=work_episode,
                    final_episode=final_episode,
                    artifact_work=artifact_work,
                    root=root,
                    start_timeout=start_timeout,
                    arguments=arguments,
                )
                async_runtime_started = False
                return result

            observations: list[dict[str, Any]] = []
            processed_acks: set[int] = set()

            def record_human_acks() -> None:
                ack_rows, safe_by_stamp = observation.shadow_ack_snapshot()
                for ack_row in ack_rows:
                    payload = ack_row["payload"]
                    stamp = int(payload["request_stamp_ns"])
                    receive_ns = int(ack_row["upper_receive_monotonic_ns"])
                    if stamp in processed_acks or receive_ns < episode_started_ns:
                        continue
                    safe = safe_by_stamp.get(stamp)
                    if safe is None:
                        continue
                    if safe["source"] != "human":
                        raise IntegratedCaptureError(
                            "SHADOW_ACTUAL_ACTION_SOURCE_NOT_HUMAN"
                        )
                    eligible = [
                        row for row in observations if row["t_ref_ns"] <= receive_ns
                    ]
                    observation_id = (
                        max(eligible, key=lambda row: row["t_ref_ns"])[
                            "observation_id"
                        ]
                        if eligible
                        else None
                    )
                    row = {
                        "schema": SHADOW_BACKEND_SCHEMA,
                        **asdict(contract.identity),
                        "ack_id": f"human-ack:{stamp}",
                        "observation_id": observation_id,
                        "receive_monotonic_ns": receive_ns,
                        "actual_action_source": "human",
                        "policy_result_id": None,
                        "proposal_id": None,
                        "policy_executed_transition": False,
                        "policy_execution": False,
                        "formal_replay": False,
                        "real_online_r": False,
                        "safe_action": safe["payload"],
                        "reference_ack": payload,
                    }
                    store.append("policy_shadow_human_ack.jsonl", row)
                    processed_acks.add(stamp)
                    if observation_id is not None:
                        ledger.record_actual_action_ack(
                            ack_id=row["ack_id"],
                            observation_id=observation_id,
                            receive_monotonic_ns=receive_ns,
                            actual_action_source="human",
                        )

            next_inference = 0.0
            native_episode_missing_since: float | None = None
            while process.poll() is None and not final_episode.is_dir():
                if observation.shadow_error:
                    raise IntegratedCaptureError(observation.shadow_error)
                if observation.episode_ending():
                    time.sleep(0.005)
                    continue
                if not work_episode.is_dir():
                    if native_episode_missing_since is None:
                        native_episode_missing_since = time.monotonic()
                    elif time.monotonic() - native_episode_missing_since > 0.5:
                        raise IntegratedCaptureError(
                            "SHADOW_NATIVE_EPISODE_ENDED_WITHOUT_SAVE"
                        )
                else:
                    native_episode_missing_since = None
                record_human_acks()
                now = time.monotonic()
                if not (
                    work_episode.is_dir()
                    and observation.ready()
                    and cameras.ready()
                    and now >= next_inference
                ):
                    time.sleep(0.005)
                    continue
                try:
                    request = observation.request(metadata)
                except RuntimeError as error:
                    if not _retryable_camera_error(error):
                        raise
                    next_inference = time.monotonic() + 0.005
                    continue
                request["provenance"]["session_id"] = contract.identity.session_id
                observation_id = (
                    f"{contract.identity.episode_id}:observation:{len(observations):06d}"
                )
                provenance = request["provenance"]
                timestamps = {
                    "measured_tcp_pose": int(
                        provenance["pose_receive_monotonic_ns"]
                    ),
                    "wrench_notch_sensor": int(
                        provenance["wrench_receive_monotonic_ns"]
                    ),
                    "gripper_state": int(
                        provenance["gripper_receive_monotonic_ns"]
                    ),
                    "external_camera": int(
                        provenance["camera1_receive_monotonic_ns"]
                    ),
                    "wrist_camera": int(
                        provenance["camera2_receive_monotonic_ns"]
                    ),
                }
                observation_record = ledger.record_observation(
                    observation_id=observation_id,
                    t_ref_ns=int(provenance["t_ref_ns"]),
                    stream_timestamps_ns=timestamps,
                    stream_ids=observation.shadow_stream_ids(request),
                )
                observations.append(observation_record)
                store.append("policy_shadow_observation.jsonl", observation_record)
                request_record = ledger.record_policy_request(
                    request,
                    observation_id=observation_id,
                    recorded_monotonic_ns=time.monotonic_ns(),
                )
                store.append("policy_shadow_request.jsonl", request_record)
                result = client.infer(request)
                observation.assert_request_generation_current(request)
                actions = deploy.validate_response(
                    result, request, session["workspace"]
                )
                result_record = ledger.record_policy_result(
                    request, result, recorded_monotonic_ns=time.monotonic_ns()
                )
                store.append("policy_shadow_result.jsonl", result_record)
                proposal = {
                    **result_record,
                    "schema": SHADOW_BACKEND_SCHEMA,
                    "actual_action_source": "human",
                    "policy_inference": True,
                    "policy_execution": False,
                    "shadow_proposal": True,
                    "executed": False,
                    "formal_replay": False,
                    "real_online_r": False,
                    "action_semantics": "absolute7",
                    "valid_horizon": int(result["valid_horizon"]),
                    "actions_absolute7": actions.tolist(),
                    "server_timing": {
                        key: result[key]
                        for key in (
                            "server_started_monotonic_ns",
                            "server_completed_monotonic_ns",
                            "inference_latency_ms",
                        )
                    },
                }
                store.append("policy_shadow_proposal.jsonl", proposal)
                record_human_acks()
                next_inference = time.monotonic() + inference_period

            return_code = process.wait(timeout=max(30.0, start_timeout))
            if return_code != 0 or not final_episode.is_dir():
                raise IntegratedCaptureError(
                    f"SHADOW_NATIVE_EPISODE_NOT_SAVED:{return_code}"
                )
            native_result = _json(final_episode / "episode_result.json")
            if native_result.get("saved") is not True:
                raise IntegratedCaptureError("SHADOW_NATIVE_EPISODE_NOT_SAVED")
            time.sleep(0.05)
            record_human_acks()
            if not observations:
                raise IntegratedCaptureError("SHADOW_POLICY_OBSERVATION_MISSING")
            reconciliation = _camera_reconciliation(final_episode, observations)
            store.write(
                "policy_shadow_camera_reconciliation.json",
                {
                    "schema": SHADOW_BACKEND_SCHEMA,
                    "native_episode": str(final_episode),
                    "records": reconciliation,
                },
            )
            seal = ledger.seal_episode(
                seal_id=(
                    f"shadow-seal:{contract.identity.session_id}:"
                    f"{contract.identity.episode_id}"
                ),
                sealed_monotonic_ns=time.monotonic_ns(),
                terminal_observation_id=observations[-1]["observation_id"],
            )
            seal.update(
                {
                    "backend_schema": SHADOW_BACKEND_SCHEMA,
                    "native_episode": str(final_episode),
                    "native_episode_result": native_result,
                    "initial_gripper_lease": authority,
                    "controller_owner": "recorder",
                    "controller_process_count": 1,
                    "deploy_controller_started": False,
                    "policy_action_publisher_created": False,
                    "camera_records_reconciled": len(reconciliation),
                }
            )
            store.write("policy_shadow_episode_seal.json", seal)
            artifact_final = root / "integrated_capture/episode_000000"
            artifact_work.rename(artifact_final)
            return seal
        finally:
            if async_runtime_started and async_runtime_identity is not None:
                try:
                    client._request(
                        "POST",
                        "/runtime/episode-abort",
                        async_runtime_identity,
                    )
                except Exception:
                    pass
            recorder_shutdown_requested = process is not None and process.poll() is None
            if recorder_shutdown_requested:
                process.send_signal(signal.SIGINT)
            executor_stop.set()
            if executor_thread is not None and executor_thread.is_alive():
                executor_thread.join(timeout=2.0)
            if executor is not None:
                executor.shutdown(timeout_sec=1.0)
            if observation is not None:
                observation.destroy_node()
            if rclpy_started and deploy.rclpy.ok():
                deploy.rclpy.shutdown()
            if cameras is not None:
                cameras.stop()
            if recorder_shutdown_requested:
                assert process is not None
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired as error:
                    raise IntegratedCaptureError(
                        "SHADOW_RECORDER_GRACEFUL_SHUTDOWN_TIMEOUT"
                    ) from error

    def _capture_policy_execution_episode(
        self,
        *,
        deploy: Any,
        process: subprocess.Popen[Any],
        client: Any,
        metadata: Mapping[str, Any],
        session: Mapping[str, Any],
        recorder_args: Any,
        contract: IntegratedCaptureContract,
        ledger: IntegratedCaptureLedger,
        observation: Any,
        store: ShadowArtifactStore,
        authority: Mapping[str, Any],
        episode_started_ns: int,
        work_episode: Path,
        final_episode: Path,
        artifact_work: Path,
        root: Path,
        start_timeout: float,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Execute an approved development policy through the recorder's sole bridge."""

        replan_steps = int(arguments.get("policy_replan_steps", 8))
        low_watermark = int(arguments.get("policy_queue_low_watermark", 4))
        max_force_n = float(arguments.get("max_force_n", 25.0))
        max_torque_nm = float(arguments.get("max_torque_nm", 2.0))
        if (
            not 0 < low_watermark < replan_steps <= 50
            or min(max_force_n, max_torque_nm) <= 0.0
        ):
            raise IntegratedCaptureError("POLICY_EXECUTE_RUNTIME_LIMITS_INVALID")

        worker = deploy.LatestOnlyInferenceWorker(client)
        observations: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        chunk_count = 0
        submitted: set[str] = set()
        outstanding: dict[str, dict[str, Any]] = {}
        invalidated_requests: set[str] = set()
        processed_decisions: set[int] = set()
        current_request: dict[str, Any] | None = None
        current_observation: dict[str, Any] | None = None
        current_chunk: dict[str, Any] | None = None
        human_takeover_active = False
        native_episode_missing_since: float | None = None

        def capture_observation() -> None:
            nonlocal current_request, current_observation
            current_request, current_observation = _record_live_observation(
                observation,
                metadata,
                contract,
                ledger,
                store,
                observations,
                stream_name="policy_execute_observation.jsonl",
            )

        def submit_current_request() -> None:
            assert current_request is not None and current_observation is not None
            request_id = str(current_request["request_id"])
            if request_id in submitted or worker.pending_or_busy():
                return
            request_record = ledger.record_policy_request(
                current_request,
                observation_id=current_observation["observation_id"],
                recorded_monotonic_ns=time.monotonic_ns(),
            )
            store.append("policy_execute_request.jsonl", request_record)
            submitted.add(request_id)
            outstanding[request_id] = current_request
            worker.submit(current_request)

        def consume_interventions() -> None:
            nonlocal current_chunk, current_request, current_observation
            nonlocal human_takeover_active
            for audit in observation.policy_audit_snapshot():
                payload = audit["payload"]
                decision_id = int(payload.get("decision_id", -1))
                receive_ns = int(audit["shadow_receive_monotonic_ns"])
                if decision_id in processed_decisions or receive_ns < episode_started_ns:
                    continue
                processed_decisions.add(decision_id)
                arbitration = payload.get("arbitration", {})
                raw_action = arbitration.get("raw_action", {})
                event = str(arbitration.get("event", ""))
                if raw_action.get("source") != "human" or event not in {
                    "intervention_start", "human_action", "intervention_end"
                }:
                    continue
                old_chunk_id = (
                    None if current_chunk is None else current_chunk["result"]["chunk_id"]
                )
                intervention = ledger.record_intervention(
                    event=event,
                    policy_epoch=int(arbitration["policy_epoch"]),
                    receive_monotonic_ns=receive_ns,
                    safe_action=payload,
                )
                intervention["invalidated_chunk_id"] = (
                    old_chunk_id if event == "intervention_start" else None
                )
                store.append("policy_execute_intervention.jsonl", intervention)
                if event == "intervention_start":
                    human_takeover_active = True
                    invalidated_requests.update(outstanding)
                    current_chunk = None
                    current_request = None
                    current_observation = None
                elif event == "intervention_end":
                    human_takeover_active = False

        def yield_to_human_gripper() -> None:
            nonlocal current_chunk, current_request, current_observation
            invalidated_requests.update(outstanding)
            current_chunk = None
            current_request = None
            current_observation = None

        _wait_for_policy_observation_ready(
            observation, process, time.monotonic() + start_timeout
        )
        capture_observation()
        submit_current_request()
        try:
            while process.poll() is None and not final_episode.is_dir():
                if observation.shadow_error:
                    raise IntegratedCaptureError(observation.shadow_error)
                consume_interventions()
                if observation.episode_ending():
                    time.sleep(0.005)
                    continue
                if not work_episode.is_dir():
                    if native_episode_missing_since is None:
                        native_episode_missing_since = time.monotonic()
                    elif time.monotonic() - native_episode_missing_since > 0.5:
                        raise IntegratedCaptureError(
                            "POLICY_EXECUTE_NATIVE_EPISODE_ENDED_WITHOUT_SAVE"
                        )
                else:
                    native_episode_missing_since = None

                if observation.human_gripper_goal_active():
                    yield_to_human_gripper()
                    time.sleep(0.005)
                    continue
                completed = worker.poll()
                if completed is not None:
                    request, result = completed
                    request_id = str(request["request_id"])
                    if request_id not in outstanding:
                        raise IntegratedCaptureError(
                            "POLICY_EXECUTE_RESULT_WITHOUT_REQUEST"
                        )
                    outstanding.pop(request_id)
                    observation.assert_request_generation_current(request)
                    actions = deploy.validate_response(
                        result, request, session["workspace"]
                    )
                    result_record = ledger.record_policy_result(
                        request, result, recorded_monotonic_ns=time.monotonic_ns()
                    )
                    store.append("policy_execute_result.jsonl", result_record)
                    chunk_record = {
                        **result_record,
                        "schema": POLICY_EXECUTION_BACKEND_SCHEMA,
                        "action_semantics": "absolute7",
                        "valid_horizon": int(result["valid_horizon"]),
                        "actions_absolute7": actions.tolist(),
                        "executed_action_source": "policy",
                        "formal_replay": False,
                        "real_online_r": False,
                    }
                    store.append("policy_execute_chunk.jsonl", chunk_record)
                    chunk_count += 1
                    current_result = (
                        request_id == result_record["request_id"]
                        and request_id not in invalidated_requests
                        and any(
                            row["observation_id"] == result_record["observation_id"]
                            for row in observations
                        )
                        and _policy_context_is_current(
                            ledger,
                            current_observation,
                            policy_epoch=int(result_record["policy_epoch"]),
                            takeover_generation=int(
                                result_record["takeover_generation"]
                            ),
                            human_takeover_active=human_takeover_active,
                        )
                    )
                    fresh = current_result
                    proposal = {
                        **result_record,
                        "schema": POLICY_EXECUTION_BACKEND_SCHEMA,
                        "actual_action_source": "policy",
                        "policy_inference": True,
                        "policy_execution": True,
                        "executed": False,
                        "invalidated_by_takeover": not fresh,
                        "formal_replay": False,
                        "real_online_r": False,
                        "action_semantics": "absolute7",
                        "valid_horizon": int(result["valid_horizon"]),
                        "actions_absolute7": actions.tolist(),
                        "server_timing": {
                            key: result[key]
                            for key in (
                                "server_started_monotonic_ns",
                                "server_completed_monotonic_ns",
                                "inference_latency_ms",
                            )
                        },
                    }
                    store.append("policy_execute_proposal.jsonl", proposal)
                    current_chunk = (
                        {
                            "request": request,
                            "result": result,
                            "result_record": result_record,
                            "actions": actions,
                            "dispatched": 0,
                        }
                        if fresh
                        else None
                    )

                if human_takeover_active:
                    time.sleep(0.005)
                    continue
                if current_request is None or current_observation is None:
                    capture_observation()
                if current_chunk is None:
                    if str(current_request["request_id"]) in submitted:
                        capture_observation()
                    submit_current_request()
                    time.sleep(0.005)
                    continue

                remaining = replan_steps - int(current_chunk["dispatched"])
                if remaining <= low_watermark:
                    submit_current_request()
                lineage = ledger.bind_policy_dispatch(
                    str(current_chunk["result_record"]["result_id"])
                )
                wrench = observation.wrench()
                force_norm = float(deploy.np.linalg.norm(wrench[:3]))
                torque_norm = float(deploy.np.linalg.norm(wrench[3:]))
                if force_norm > max_force_n or torque_norm > max_torque_nm:
                    raise IntegratedCaptureError(
                        "POLICY_EXECUTE_WRENCH_GUARD:"
                        f"force={force_norm:.3f}:torque={torque_norm:.3f}"
                    )
                selection_ns = time.monotonic_ns()
                try:
                    action_index, target = _selected_chunk_action(
                        current_chunk["actions"],
                        t_ref_ns=int(current_chunk["result"]["t_ref_ns"]),
                        fps=int(metadata["fps"]),
                        selection_ns=selection_ns,
                    )
                except IntegratedCaptureError as error:
                    if str(error) != "POLICY_EXECUTE_CHUNK_EXPIRED":
                        raise
                    current_chunk = None
                    continue
                position, quaternion = observation.pose()
                deploy.forcevla.check_target_guard(
                    target,
                    position,
                    quaternion,
                    0.08,
                    25.0,
                )
                normalized = deploy.forcevla.absolute_target_to_normalized(
                    target,
                    position,
                    quaternion,
                    recorder_args.hilserl_translation_action_scale_m,
                    recorder_args.hilserl_rotation_action_scale_rad,
                )
                if current_observation is None:
                    current_chunk = None
                    continue
                try:
                    dispatch_observation_id = str(
                        current_observation["observation_id"]
                    )
                    dispatch_generation = (
                        int(lineage["policy_epoch"]),
                        int(lineage["takeover_generation"]),
                    )
                    sequence = observation.publish_policy_action(
                        normalized,
                        policy_epoch=lineage["policy_epoch"],
                        observation_id=dispatch_observation_id,
                    )
                except IntegratedCaptureError as error:
                    if str(error) == "POLICY_EXECUTE_EPISODE_END":
                        current_chunk = None
                        continue
                    raise
                decision = observation.wait_policy_decision(
                    sequence, int(lineage["policy_epoch"]), 0.50
                )
                consume_interventions()
                if decision is None or not _policy_context_is_current(
                    ledger,
                    current_observation,
                    policy_epoch=dispatch_generation[0],
                    takeover_generation=dispatch_generation[1],
                    human_takeover_active=human_takeover_active,
                    observation_id=dispatch_observation_id,
                ):
                    current_chunk = None
                    continue
                arbitration = decision.get("arbitration", {})
                raw_action = arbitration.get("raw_action", {})
                if (
                    raw_action.get("source") != "policy"
                    or int(raw_action.get("sequence", -1)) != sequence
                    or int(raw_action.get("policy_epoch", -1))
                    != int(lineage["policy_epoch"])
                    or raw_action.get("observation_id")
                    != dispatch_observation_id
                    or not deploy.np.array_equal(
                        deploy.np.asarray(raw_action.get("action"), dtype=deploy.np.float64),
                        deploy.np.asarray(normalized, dtype=deploy.np.float64),
                    )
                ):
                    raise IntegratedCaptureError(
                        "POLICY_EXECUTE_ARBITRATION_LINEAGE_MISMATCH"
                    )
                if arbitration.get("accepted") is not True:
                    reason = str(arbitration.get("reason"))
                    if reason in {"human_override", "stale_policy_epoch"}:
                        current_chunk = None
                        continue
                    raise IntegratedCaptureError(
                        f"POLICY_EXECUTE_ACTION_REJECTED:{reason}"
                    )
                if decision.get("equilibrium_published") is not True:
                    raise IntegratedCaptureError("POLICY_EXECUTE_POSE_NOT_PUBLISHED")
                if (
                    arbitration.get("reason") != "accepted_policy"
                    or arbitration.get("event") != "policy_action"
                    or decision.get("execute_enabled") is not True
                ):
                    raise IntegratedCaptureError(
                        "POLICY_EXECUTE_ARBITRATION_ACCEPTANCE_INVALID"
                    )
                selection = {
                    **lineage,
                    "sequence": sequence,
                    "action_index": action_index,
                    "apply_selection_monotonic_ns": selection_ns,
                    "selected_post_adapter_absolute7": target.tolist(),
                    "normalized_action7": normalized.tolist(),
                }
                audited_decision = dict(decision)
                audited_decision["forcesmolvla_chunk_selection"] = selection
                ack_timeout_s = float(metadata["controller_ack_timeout_ms"]) / 1000.0
                pose_ack = observation.wait_reference_ack(
                    int(decision["equilibrium_source_stamp_ns"]), ack_timeout_s
                )
                if pose_ack is None:
                    current_chunk = None
                    continue
                deploy.validate_exact_controller_ack(
                    audited_decision,
                    pose_ack,
                    base_frame=recorder_args.base_frame,
                    max_position_error_m=(
                        math.sqrt(3.0)
                        * recorder_args.hilserl_translation_action_scale_m
                    ),
                    max_rotation_error_rad=(
                        math.sqrt(3.0)
                        * recorder_args.hilserl_rotation_action_scale_rad
                    ),
                )
                try:
                    gripper_authority = observation.ensure_policy_gripper_authority(
                        float(target[6]),
                        command_context={
                            **lineage,
                            "sequence": sequence,
                            "action_index": action_index,
                        },
                        timeout_s=5.0,
                    )
                except IntegratedCaptureError as error:
                    if str(error) == "POLICY_EXECUTE_EPISODE_END":
                        current_chunk = None
                        continue
                    if (
                        str(error) == "POLICY_EXECUTE_GRIPPER_AUTHORITY_BUSY"
                        and observation.human_gripper_goal_active()
                    ):
                        yield_to_human_gripper()
                        continue
                    raise
                store.append(
                    "policy_execute_gripper_authority.jsonl", gripper_authority
                )
                previous_observation = current_observation
                capture_observation()
                transition = ledger.record_actual_action_ack(
                    ack_id=(
                        f"policy-ack:{sequence}:"
                        f"{int(decision['equilibrium_source_stamp_ns'])}"
                    ),
                    observation_id=previous_observation["observation_id"],
                    next_observation_id=current_observation["observation_id"],
                    receive_monotonic_ns=int(
                        pose_ack["upper_receive_monotonic_ns"]
                    ),
                    actual_action_source="policy",
                    policy_result_id=str(lineage["result_id"]),
                    proposal_id=str(lineage["proposal_id"]),
                    accepted_absolute7=target.tolist(),
                )
                transition.update(
                    {
                        "schema": POLICY_EXECUTION_BACKEND_SCHEMA,
                        "executed_action_source": "policy",
                        "policy_executed_transition": True,
                        "selection": selection,
                        "safety_arbitration": arbitration,
                        "pose_command": decision["requested_equilibrium"],
                        "pose_ack": pose_ack,
                        "gripper_authority": gripper_authority,
                        "intervention": False,
                    }
                )
                store.append("policy_execute_transition.jsonl", transition)
                transitions.append(transition)
                current_chunk["dispatched"] = int(current_chunk["dispatched"]) + 1
                if int(current_chunk["dispatched"]) >= replan_steps:
                    current_chunk = None

            for request_id in tuple(outstanding):
                canceled = ledger.cancel_policy_request(
                    request_id,
                    reason="episode_sealed_before_inference_result",
                    recorded_monotonic_ns=time.monotonic_ns(),
                )
                store.append("policy_execute_request_canceled.jsonl", canceled)
                outstanding.pop(request_id, None)
        finally:
            worker.close()

        return_code = process.wait(timeout=max(30.0, start_timeout))
        if return_code != 0 or not final_episode.is_dir():
            raise IntegratedCaptureError(
                f"POLICY_EXECUTE_NATIVE_EPISODE_NOT_SAVED:{return_code}"
            )
        native_result = _json(final_episode / "episode_result.json")
        if native_result.get("saved") is not True:
            raise IntegratedCaptureError("POLICY_EXECUTE_NATIVE_EPISODE_NOT_SAVED")
        if not transitions:
            raise IntegratedCaptureError("POLICY_EXECUTE_ACCEPTED_ACTION_MISSING")
        reconciliation = _camera_reconciliation(final_episode, observations)
        store.write(
            "policy_execute_camera_reconciliation.json",
            {
                "schema": POLICY_EXECUTION_BACKEND_SCHEMA,
                "native_episode": str(final_episode),
                "records": reconciliation,
            },
        )
        async_status = None
        if bool(arguments.get("async_learner", False)):
            async_status = _complete_async_runtime(
                client,
                {
                    "session_id": contract.identity.session_id,
                    "episode_id": contract.identity.episode_id,
                    "policy_revision": contract.identity.policy_revision,
                },
                deadline=time.monotonic() + start_timeout,
            )
        seal = ledger.seal_episode(
            seal_id=(
                f"policy-execute-seal:{contract.identity.session_id}:"
                f"{contract.identity.episode_id}"
            ),
            sealed_monotonic_ns=time.monotonic_ns(),
            terminal_observation_id=observations[-1]["observation_id"],
        )
        seal.update(
            {
                "backend_schema": POLICY_EXECUTION_BACKEND_SCHEMA,
                "technical_seal": "complete",
                "native_episode": str(final_episode),
                "native_episode_result": native_result,
                "initial_gripper_lease": dict(authority),
                "controller_owner": "recorder",
                "controller_process_count": 1,
                "deploy_controller_started": False,
                "policy_action_publisher_created": True,
                "camera_records_reconciled": len(reconciliation),
                "policy_chunk_count": chunk_count,
                "detector_approval_scope": (
                    "single_episode_cycle210_policy_execution_smoke"
                ),
                "formal_training_replay_written": False,
                "checkpoint_written": False,
            }
        )
        if async_status is not None:
            seal.update(
                {
                    "learner_started": True,
                    "learner_resume_checkpoint": async_status[
                        "learner_resume_checkpoint"
                    ],
                    "active_actor_revision": async_status[
                        "active_actor_revision"
                    ],
                    "active_actor_model_revision": async_status[
                        "active_actor_model_revision"
                    ],
                    "learner_critic_steps": int(
                        async_status["learner_critic_steps"]
                    ),
                    "learner_actor_steps": int(
                        async_status["learner_actor_steps"]
                    ),
                    "actor_updates": int(async_status["learner_actor_steps"]),
                    "critic_updates": int(async_status["learner_critic_steps"]),
                    "pending_checkpoint_path": async_status[
                        "pending_checkpoint_path"
                    ],
                    "pending_candidate_id": async_status[
                        "pending_candidate_id"
                    ],
                    "pending_candidate_published": False,
                    "pending_candidate_activated": False,
                    "current_episode_sampled_by_learner": False,
                }
            )
        store.write("policy_execute_episode_seal.json", seal)
        artifact_final = root / "integrated_capture/episode_000000"
        artifact_work.rename(artifact_final)
        return seal


__all__ = [
    "ForbiddenPolicyPublisher",
    "IntegratedShadowBackend",
    "POLICY_EXECUTION_BACKEND_SCHEMA",
    "SHADOW_BACKEND_SCHEMA",
    "ShadowArtifactStore",
    "build_native_recorder_command",
]
