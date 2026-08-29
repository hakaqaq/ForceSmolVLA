"""Recorder-owned live shadow capture with read-only policy inference."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

from .gripper_provenance import GripperGeneration
from .integrated_capture import (
    CaptureBackendCapabilities,
    IntegratedCaptureContract,
    IntegratedCaptureError,
    IntegratedCaptureLedger,
    RECORDER_CONTROL_CHAIN,
    RECORDER_ENTRY,
)
from .policy_lineage import InitialGripperAuthority, UPPER_CLOCK_DOMAIN


SHADOW_BACKEND_SCHEMA = "forcesmolvla-stage3-integrated-shadow-backend-v1"
EXTERNAL_SCRIPTS = Path("/home/rlc123/fr3_client_ws/scripts")
DEFAULT_DEPLOYMENT_PROFILE = Path(
    "/home/rlc123/ForceSmolVLA/configs/deployment.active.development.json"
)


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
    except (KeyError, TypeError, ValueError) as error:
        raise IntegratedCaptureError("SHADOW_RECORDER_ARGUMENTS_INVALID") from error
    if not task or not tool_profile or episodes != 1 or episode_time <= 0.0:
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
    ]


class ShadowArtifactStore:
    """Append-only shadow artifacts; proposals and human ACKs cannot commingle."""

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

    def _update_role(self, role: str) -> None:
        latest: tuple[Path, Any] | None = None
        while True:
            index = self._next[role]
            path = self.episode_dir / "images" / role / f"frame_{index:06d}.jpg"
            if not path.is_file():
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


def _shadow_observation_type(deploy: Any) -> type:
    class ShadowObservation(deploy.LiveForceSmolObservation):
        def __init__(self, args: Any, cameras: Any, metadata: dict, task: str) -> None:
            self._shadow_policy_topic = str(args.policy_action_topic)
            self._shadow_lock = threading.Lock()
            self._shadow_safe_by_stamp: dict[int, dict[str, Any]] = {}
            self._shadow_acks: list[dict[str, Any]] = []
            self._shadow_targets: list[dict[str, Any]] = []
            self._shadow_statuses: list[dict[str, Any]] = []
            self._shadow_gripper_sources: deque[tuple[int, int]] = deque(maxlen=512)
            self.shadow_error: str | None = None
            super().__init__(args, cameras, metadata, task)
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
            if str(topic) == self._shadow_policy_topic:
                return ForbiddenPolicyPublisher(str(topic))
            return super().create_publisher(message_type, topic, *args, **kwargs)

        def _fail_shadow(self, reason: str, error: Exception) -> None:
            self.shadow_error = f"{reason}:{type(error).__name__}:{error}"

        def _safe_action_callback(self, message: Any) -> None:
            super()._safe_action_callback(message)
            try:
                payload = json.loads(message.data)
                raw = payload["arbitration"]["raw_action"]
                source_stamp = payload.get("equilibrium_source_stamp_ns")
                if source_stamp is None:
                    return
                stamp = int(source_stamp)
                if stamp <= 0 or not isinstance(raw, dict):
                    raise ValueError("invalid safe-action identity")
                record = {
                    "shadow_receive_monotonic_ns": time.monotonic_ns(),
                    "source": str(raw["source"]),
                    "payload": payload,
                }
                with self._shadow_lock:
                    self._shadow_safe_by_stamp[stamp] = record
                    for old in list(self._shadow_safe_by_stamp)[:-512]:
                        self._shadow_safe_by_stamp.pop(old, None)
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
    for key, attribute in (
        ("base", "base_frame"),
        ("tcp", "tcp_frame"),
        ("sensor_body", "sensor_body_frame"),
        ("wrench_measurement", "wrench_measurement_frame"),
    ):
        if str(frames.get(key, "")) != str(getattr(recorder_args, attribute)):
            raise IntegratedCaptureError("SHADOW_SESSION_FRAME_MISMATCH")
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
    maximum_age_ns = int(
        min(float(metadata["gripper_max_age_ms"]), 100.0) * 1.0e6
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


class IntegratedShadowBackend:
    """Actual shadow backend: one native recorder plus read-only inference."""

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
        _validate_shadow_contract(contract)
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
        manifest_task = deploy.validate_manifest(manifest, metadata, manifest_path)
        if manifest_task != str(arguments["task"]).strip():
            raise IntegratedCaptureError("SHADOW_DEPLOYMENT_TASK_MISMATCH")
        if metadata.get("model_sha256") != contract.identity.policy_revision:
            raise IntegratedCaptureError("SHADOW_POLICY_REVISION_MISMATCH")

        recorder_args = _parse_recorder_arguments(recorder, command)
        start_timeout = float(arguments.get("backend_start_timeout", 180.0))
        inference_period = float(arguments.get("shadow_inference_period", 0.1))
        if start_timeout <= 0.0 or inference_period <= 0.0:
            raise IntegratedCaptureError("SHADOW_BACKEND_TIMING_INVALID")

        process: subprocess.Popen[Any] | None = None
        observation = executor = executor_thread = cameras = None
        executor_stop = threading.Event()
        rclpy_started = False
        try:
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
            observation_type = _shadow_observation_type(deploy)
            observation = observation_type(
                recorder_args, cameras, metadata, str(arguments["task"]).strip()
            )
            if not isinstance(observation.action_publisher, ForbiddenPolicyPublisher):
                raise IntegratedCaptureError("SHADOW_POLICY_PUBLISHER_WAS_CREATED")
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
            integrated_manifest = {
                "schema": SHADOW_BACKEND_SCHEMA,
                "contract": contract.to_dict(),
                "native_session": "session.json",
                "native_recorder_entry": str(RECORDER_ENTRY),
                "native_recorder_pid": process.pid,
                "controller_owner": "recorder",
                "controller_process_count": 1,
                "deploy_controller_started": False,
                "policy_server_started_by_backend": False,
                "policy_action_publisher_created": False,
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
            store.write("policy_shadow_initial_gripper_lease.json", authority)

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
                request = observation.request(metadata)
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


__all__ = [
    "ForbiddenPolicyPublisher",
    "IntegratedShadowBackend",
    "SHADOW_BACKEND_SCHEMA",
    "ShadowArtifactStore",
    "build_native_recorder_command",
]
