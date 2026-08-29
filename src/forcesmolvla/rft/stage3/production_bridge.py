"""CPU-only recorder-stream bridge to an episode-sealed Stage-3 shadow outbox.

The bridge deliberately stops at a filesystem outbox.  It imports no ROS,
robot, serving, networking, CUDA, or replay implementation and never emits a
robot command.  Recorder receive timestamps are the only timestamps compared
across streams; controller-internal timestamps are retained as provenance only.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .gripper_provenance import (
    GripperAuthorityEvidence,
    GripperAuthorityKind,
    GripperGeneration,
    GripperProvenanceError,
    PoseAcceptedAuthority,
    VALID_TERMINAL_OUTCOMES,
    close_full_action7_authority,
)


SCHEMA_VERSION = "forcesmolvla_stage3_production_bridge_transition.v1"
REPORT_VERSION = "forcesmolvla_stage3_production_bridge_report.v1"
UPPER_CLOCK = "upper_host_monotonic"
POLICY_LINEAGE_FIELDS = frozenset(
    {
        "request_id",
        "result_id",
        "chunk_id",
        "proposal_id",
        "policy_revision",
        "policy_epoch",
        "reset_generation",
        "takeover_generation",
        "t_ref_ns",
        "action_index",
    }
)
REQUIRED_STREAMS = (
    "raw_action",
    "safe_action",
    "requested_equilibrium",
    "accepted_reference",
    "reference_ack",
    "gripper_target",
    "gripper_goal_status",
    "gripper_state",
    "measured_tcp_pose",
    "wrench_notch_sensor",
    "external_camera",
    "wrist_camera",
)


class ProductionBridgeError(RuntimeError):
    """Fail-closed recorder or persistence contract violation."""


class BridgeDigestCollisionError(ProductionBridgeError):
    """A stable UID was observed with different canonical content."""


class InjectedBridgeCrash(ProductionBridgeError):
    """Test-only crash point after an immutable WAL write."""


@dataclass(frozen=True)
class BridgeConfig:
    clock_domain_id: str = UPPER_CLOCK
    max_pose_age_ns: int = 30_000_000
    max_wrench_age_ns: int = 30_000_000
    max_camera_age_ns: int = 100_000_000
    max_gripper_feedback_age_ns: int = 100_000_000
    max_gripper_command_association_ns: int = 50_000_000
    max_pose_ack_latency_ns: int = 50_000_000
    pose_position_tolerance_m: float = 0.02
    pose_quaternion_tolerance_rad: float = 0.2
    requested_width_tolerance_m: float = 1.0e-9
    minimum_full_macro_span_ns: int = 66_666_666
    gripper_close_threshold_m: float = 0.030
    gripper_open_threshold_m: float = 0.055
    gripper_closed_width_m: float = 0.0
    gripper_open_width_m: float = 0.085
    allow_recorded_offline_lineage: bool = True
    hash_camera_files: bool = True

    def validate(self) -> "BridgeConfig":
        if self.clock_domain_id != UPPER_CLOCK:
            raise ProductionBridgeError("BRIDGE_CLOCK_DOMAIN_UNSUPPORTED")
        positive = (
            self.max_pose_age_ns,
            self.max_wrench_age_ns,
            self.max_camera_age_ns,
            self.max_gripper_feedback_age_ns,
            self.max_gripper_command_association_ns,
            self.max_pose_ack_latency_ns,
            self.minimum_full_macro_span_ns,
        )
        if any(isinstance(value, bool) or value <= 0 for value in positive):
            raise ProductionBridgeError("BRIDGE_TEMPORAL_LIMIT_INVALID")
        finite = (
            self.pose_position_tolerance_m,
            self.pose_quaternion_tolerance_rad,
            self.requested_width_tolerance_m,
            self.gripper_close_threshold_m,
            self.gripper_open_threshold_m,
            self.gripper_closed_width_m,
            self.gripper_open_width_m,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in finite):
            raise ProductionBridgeError("BRIDGE_TOLERANCE_INVALID")
        if not (
            self.gripper_closed_width_m
            <= self.gripper_close_threshold_m
            < self.gripper_open_threshold_m
            <= self.gripper_open_width_m
            <= 0.1
        ):
            raise ProductionBridgeError("BRIDGE_GRIPPER_ADAPTER_LIMIT_INVALID")
        return self

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BridgeConfig":
        limits = value.get("limits", value)
        return cls(
            clock_domain_id=str(value.get("clock_domain_id", UPPER_CLOCK)),
            max_pose_age_ns=int(float(limits.get("max_pose_age_ms", 30.0)) * 1e6),
            max_wrench_age_ns=int(float(limits.get("max_wrench_age_ms", 30.0)) * 1e6),
            max_camera_age_ns=int(float(limits.get("max_camera_age_ms", 100.0)) * 1e6),
            max_gripper_feedback_age_ns=int(
                float(limits.get("max_gripper_feedback_age_ms", 100.0)) * 1e6
            ),
            max_gripper_command_association_ns=int(
                float(limits.get("max_gripper_command_association_ms", 50.0))
                * 1e6
            ),
            max_pose_ack_latency_ns=int(
                float(limits.get("max_pose_ack_latency_ms", 50.0)) * 1e6
            ),
            pose_position_tolerance_m=float(
                limits.get("pose_position_tolerance_m", 0.02)
            ),
            pose_quaternion_tolerance_rad=float(
                limits.get("pose_quaternion_tolerance_rad", 0.2)
            ),
            requested_width_tolerance_m=float(
                limits.get("requested_width_tolerance_m", 1.0e-9)
            ),
            minimum_full_macro_span_ns=int(
                limits.get("minimum_full_macro_span_ns", 66_666_666)
            ),
            gripper_close_threshold_m=float(
                limits.get("gripper_close_threshold_m", 0.030)
            ),
            gripper_open_threshold_m=float(
                limits.get("gripper_open_threshold_m", 0.055)
            ),
            gripper_closed_width_m=float(
                limits.get("gripper_closed_width_m", 0.0)
            ),
            gripper_open_width_m=float(limits.get("gripper_open_width_m", 0.085)),
            allow_recorded_offline_lineage=bool(
                value.get("allow_recorded_offline_lineage", True)
            ),
            hash_camera_files=bool(value.get("hash_camera_files", True)),
        ).validate()


@dataclass(frozen=True)
class BridgeReport:
    status: str
    episode_id: str
    sealed: bool
    dry_run: bool
    candidate_count: int
    outbox_eligible_count: int
    quarantined_count: int
    wal_written_count: int
    outbox_written_count: int
    idempotent_count: int
    quarantine_reasons: tuple[str, ...]
    recorded_offline_production_bridge: str
    policy_fixture: bool
    new_command_count: int = 0
    held_command_count: int = 0
    real_online_r_used: bool = False
    formal_training_replay_written: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Goal:
    sequence: int
    action_goal_id: str
    requested_state: str
    requested_width_m: float
    started_ns: int
    accepted_ns: int
    finished_ns: int
    outcome: str


def load_bridge_config(path: Path) -> tuple[BridgeConfig, dict[str, Any]]:
    """Load JSON-compatible YAML without adding a YAML runtime dependency."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProductionBridgeError("BRIDGE_CONFIG_NOT_JSON_COMPATIBLE_YAML") from error
    if not isinstance(value, dict):
        raise ProductionBridgeError("BRIDGE_CONFIG_NOT_OBJECT")
    return BridgeConfig.from_mapping(value), value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProductionBridgeError("BRIDGE_PAYLOAD_NOT_CANONICAL_JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProductionBridgeError(f"BRIDGE_JSON_INVALID:{path.name}") from error
    if not isinstance(value, dict):
        raise ProductionBridgeError(f"BRIDGE_JSON_NOT_OBJECT:{path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("record is not an object")
                records.append(value)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ProductionBridgeError(
            f"BRIDGE_JSONL_INVALID:{path.name}:{line_number if 'line_number' in locals() else 0}"
        ) from error
    return records


def _finite_vector(value: Any, size: int, reason: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ProductionBridgeError(reason) from error
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ProductionBridgeError(reason)
    return result


def _quaternion_to_rotvec(value: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = _finite_vector(value, 4, "BRIDGE_QUATERNION_INVALID")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0 or not math.isclose(norm, 1.0, abs_tol=1.0e-5):
        raise ProductionBridgeError("BRIDGE_QUATERNION_INVALID")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    sine = math.sqrt(x * x + y * y + z * z)
    if sine < 1.0e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(sine, w)
    scale = angle / sine
    return (x * scale, y * scale, z * scale)


def _pose_tcp6(pose: Mapping[str, Any]) -> tuple[float, ...]:
    position = _finite_vector(
        pose.get("position_m"), 3, "BRIDGE_POSE_POSITION_INVALID"
    )
    return position + _quaternion_to_rotvec(pose.get("quaternion_xyzw", ()))


def _quaternion_distance(a: Sequence[float], b: Sequence[float]) -> float:
    qa = _finite_vector(a, 4, "BRIDGE_QUATERNION_INVALID")
    qb = _finite_vector(b, 4, "BRIDGE_QUATERNION_INVALID")
    dot = abs(sum(left * right for left, right in zip(qa, qb, strict=True)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _enum_json(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, tuple):
        return [_enum_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _enum_json(item) for key, item in value.items()}
    return value


class _Timeline:
    def __init__(self, records: Iterable[dict[str, Any]], field: str) -> None:
        ordered = sorted(records, key=lambda item: int(item.get(field, 0)))
        self.records = ordered
        self.timestamps = [int(item.get(field, 0)) for item in ordered]
        if not ordered or self.timestamps[0] <= 0:
            raise ProductionBridgeError(f"BRIDGE_TIMELINE_EMPTY_OR_INVALID:{field}")
        if any(right <= left for left, right in zip(self.timestamps, self.timestamps[1:])):
            raise ProductionBridgeError(f"BRIDGE_TIMELINE_NOT_STRICT:{field}")

    def latest(self, timestamp_ns: int, max_age_ns: int) -> dict[str, Any]:
        index = bisect_right(self.timestamps, timestamp_ns) - 1
        if index < 0:
            raise ProductionBridgeError("BRIDGE_CAUSAL_SAMPLE_MISSING")
        age = timestamp_ns - self.timestamps[index]
        if age < 0 or age > max_age_ns:
            raise ProductionBridgeError("BRIDGE_CAUSAL_SAMPLE_STALE")
        return self.records[index]


class Stage3ProductionBridge:
    """Single-process filesystem development bridge with immutable WAL records."""

    def __init__(self, *, config: BridgeConfig, state_root: Path) -> None:
        self.config = config.validate()
        self.state_root = Path(state_root)
        self._camera_hashes: dict[Path, str] = {}

    def _episode_id(self, episode_dir: Path, start: Mapping[str, Any]) -> str:
        dataset = episode_dir.parent.parent.name
        index = int(start.get("episode_index", -1))
        if index < 0 or episode_dir.name != f"episode_{index:06d}":
            raise ProductionBridgeError("BRIDGE_EPISODE_IDENTITY_INVALID")
        return f"{dataset}/{episode_dir.name}"

    def _load_streams(self, episode_dir: Path) -> dict[str, list[dict[str, Any]]]:
        root = episode_dir / "streams"
        missing = [name for name in REQUIRED_STREAMS if not (root / f"{name}.jsonl").is_file()]
        if missing:
            raise ProductionBridgeError(f"BRIDGE_REQUIRED_STREAM_MISSING:{','.join(missing)}")
        return {name: _read_jsonl(root / f"{name}.jsonl") for name in REQUIRED_STREAMS}

    def _validate_seal(
        self, result: Mapping[str, Any], streams: Mapping[str, list[dict[str, Any]]]
    ) -> None:
        if result.get("saved") is not True or result.get("fatal_reason") is not None:
            raise ProductionBridgeError("BRIDGE_EPISODE_NOT_ACCEPTED_SEALED")
        counts = result.get("stream_counts")
        if not isinstance(counts, dict):
            raise ProductionBridgeError("BRIDGE_SEAL_STREAM_COUNTS_MISSING")
        for name, records in streams.items():
            if int(counts.get(name, -1)) != len(records):
                raise ProductionBridgeError(f"BRIDGE_SEAL_COUNT_MISMATCH:{name}")

    def _goals(self, streams: Mapping[str, list[dict[str, Any]]]) -> tuple[_Goal, ...]:
        targets: dict[int, dict[str, Any]] = {}
        statuses: dict[int, dict[str, Any]] = {}
        for record in streams["gripper_target"]:
            sequence = int(record.get("local_goal_sequence", 0))
            if sequence <= 0 or sequence in targets:
                raise ProductionBridgeError("BRIDGE_GRIPPER_TARGET_IDENTITY_INVALID")
            targets[sequence] = record
        for record in streams["gripper_goal_status"]:
            sequence = int(record.get("local_goal_sequence", 0))
            if sequence <= 0 or sequence in statuses:
                raise ProductionBridgeError("BRIDGE_GRIPPER_TERMINAL_IDENTITY_INVALID")
            statuses[sequence] = record
        if set(targets) != set(statuses):
            raise ProductionBridgeError("BRIDGE_GRIPPER_TERMINAL_PAIRING_INCOMPLETE")
        goals: list[_Goal] = []
        for sequence in sorted(targets):
            target, status = targets[sequence], statuses[sequence]
            identity = (
                str(target.get("action_goal_id", "")),
                str(status.get("action_goal_id", "")),
            )
            outcome = str(status.get("outcome", ""))
            if not identity[0] or identity[0] != identity[1]:
                raise ProductionBridgeError("BRIDGE_GRIPPER_GOAL_ID_MISMATCH")
            if outcome not in VALID_TERMINAL_OUTCOMES:
                raise ProductionBridgeError(f"BRIDGE_GRIPPER_TERMINAL_INVALID:{outcome}")
            started = int(target.get("started_monotonic_ns", 0))
            accepted = int(target.get("accepted_monotonic_ns", 0))
            finished = int(status.get("finished_monotonic_ns", 0))
            if not 0 < started <= accepted <= finished:
                raise ProductionBridgeError("BRIDGE_GRIPPER_GOAL_TIME_INVALID")
            if int(status.get("accepted_monotonic_ns", 0)) != accepted:
                raise ProductionBridgeError("BRIDGE_GRIPPER_ACCEPT_TIME_MISMATCH")
            goals.append(
                _Goal(
                    sequence=sequence,
                    action_goal_id=identity[0],
                    requested_state=str(target.get("requested_state", "")),
                    requested_width_m=float(target.get("target_width_m", -1.0)),
                    started_ns=started,
                    accepted_ns=accepted,
                    finished_ns=finished,
                    outcome=outcome,
                )
            )
        if not goals:
            raise ProductionBridgeError("BRIDGE_GRIPPER_GOAL_STREAM_EMPTY")
        return tuple(goals)

    def _lineage(
        self,
        *,
        episode_id: str,
        raw: Mapping[str, Any],
        safe: Mapping[str, Any],
    ) -> tuple[dict[str, Any], GripperGeneration, bool]:
        source = str(raw.get("source", ""))
        sequence = int(raw.get("sequence", -1))
        policy_epoch = int(raw.get("policy_epoch", -1))
        if sequence < 0 or policy_epoch < 0 or source not in {"human", "policy"}:
            raise ProductionBridgeError("BRIDGE_RAW_ACTION_IDENTITY_INVALID")
        selection = safe.get("forcesmolvla_chunk_selection")
        if source == "policy":
            if not isinstance(selection, dict):
                raise ProductionBridgeError("BRIDGE_POLICY_SELECTION_MISSING")
            missing = sorted(POLICY_LINEAGE_FIELDS - set(selection))
            if missing:
                raise ProductionBridgeError(
                    f"BRIDGE_POLICY_LINEAGE_UNBOUND:{','.join(missing)}"
                )
            lineage = {
                "binding_kind": "recorded_policy_runtime_ledger",
                "request_id": str(selection["request_id"]),
                "result_id": str(selection["result_id"]),
                "chunk_id": str(selection["chunk_id"]),
                "proposal_id": str(selection["proposal_id"]),
                "policy_revision": str(selection["policy_revision"]),
                "policy_epoch": int(selection["policy_epoch"]),
                "reset_generation": int(selection["reset_generation"]),
                "takeover_generation": int(selection["takeover_generation"]),
                "t_ref_ns": int(selection["t_ref_ns"]),
                "selected_index": int(selection["action_index"]),
                "source_sequence": sequence,
                "policy_fixture": True,
            }
            if lineage["policy_epoch"] != policy_epoch:
                raise ProductionBridgeError("BRIDGE_POLICY_EPOCH_MISMATCH")
            policy_fixture = True
        else:
            if not self.config.allow_recorded_offline_lineage:
                raise ProductionBridgeError("BRIDGE_OFFLINE_LINEAGE_DISABLED")
            decision = int(safe.get("decision_id", -1))
            if decision < 0:
                raise ProductionBridgeError("BRIDGE_SAFE_DECISION_ID_INVALID")
            lineage = {
                "binding_kind": "recorded_offline_recorder_identity",
                "request_id": f"raw-action:{source}:{sequence}",
                "result_id": f"safe-action:{decision}",
                "chunk_id": f"offline-dispatch:{episode_id}:{sequence}",
                "proposal_id": f"arbitration:{decision}",
                "policy_revision": "recorded-offline-no-policy-revision",
                "policy_epoch": policy_epoch,
                "reset_generation": 0,
                "takeover_generation": policy_epoch,
                "t_ref_ns": int(raw.get("source_monotonic_ns", 0)),
                "selected_index": None,
                "source_sequence": sequence,
                "policy_fixture": False,
                "derivation": {
                    "synthetic_policy_identity": False,
                    "reset_generation": "episode_scoped_zero",
                    "takeover_generation": "recorded_arbiter_policy_epoch",
                    "chunk_semantics": "single recorder dispatch; not an H50 policy result",
                },
            }
            policy_fixture = False
        generation = GripperGeneration(
            episode_id=episode_id,
            reset_generation=lineage["reset_generation"],
            takeover_generation=lineage["takeover_generation"],
            policy_revision=lineage["policy_revision"],
            policy_epoch=lineage["policy_epoch"],
        ).validate()
        return lineage, generation, policy_fixture

    def _goal_for_transition(
        self,
        *,
        goals: Sequence[_Goal],
        authority_ns: int,
        selected_width_m: float | None,
        used_new: set[int],
    ) -> tuple[_Goal, GripperAuthorityKind]:
        pending = [goal for goal in goals if goal.started_ns <= authority_ns < goal.accepted_ns]
        near = [
            goal
            for goal in goals
            if goal.sequence not in used_new
            and abs(goal.started_ns - authority_ns)
            <= self.config.max_gripper_command_association_ns
            and (
                selected_width_m is None
                or abs(goal.requested_width_m - selected_width_m)
                <= self.config.requested_width_tolerance_m
            )
        ]
        if near:
            goal = min(near, key=lambda item: abs(item.started_ns - authority_ns))
            return goal, GripperAuthorityKind.NEW_COMMAND
        if pending:
            raise ProductionBridgeError("BRIDGE_CONFLICTING_GRIPPER_COMMAND_PENDING")
        accepted = [
            goal
            for goal in goals
            if goal.accepted_ns <= authority_ns
            and (
                selected_width_m is None
                or abs(goal.requested_width_m - selected_width_m)
                <= self.config.requested_width_tolerance_m
            )
        ]
        if not accepted:
            raise ProductionBridgeError("BRIDGE_GRIPPER_ACCEPTED_ORIGIN_MISSING")
        return (
            max(accepted, key=lambda item: item.accepted_ns),
            GripperAuthorityKind.HELD_FROM_ACCEPTED_COMMAND,
        )

    def _camera(self, episode_dir: Path, record: Mapping[str, Any], at_ns: int) -> dict[str, Any]:
        relative = str(record.get("rgb_path", ""))
        path = episode_dir / relative
        if not relative or not path.is_file():
            raise ProductionBridgeError("BRIDGE_CAMERA_BLOB_MISSING")
        if record.get("timestamp_domain") != "host_monotonic_receive":
            raise ProductionBridgeError("BRIDGE_CAMERA_CLOCK_DOMAIN_MISMATCH")
        digest: str | None = None
        if self.config.hash_camera_files:
            digest = self._camera_hashes.get(path)
            if digest is None:
                digest = _sha256_file(path)
                self._camera_hashes[path] = digest
        timestamp = int(record.get("receive_monotonic_ns", 0))
        return {
            "blob_reference": relative,
            "sha256": digest,
            "timestamp_monotonic_ns": timestamp,
            "age_ms": (at_ns - timestamp) / 1.0e6,
            "timestamp_domain": str(record.get("timestamp_domain", "")),
        }

    def _observation(
        self,
        *,
        episode_dir: Path,
        episode_id: str,
        observation_id: str,
        at_ns: int,
        timelines: Mapping[str, _Timeline],
    ) -> dict[str, Any]:
        pose = timelines["pose"].latest(at_ns, self.config.max_pose_age_ns)
        wrench = timelines["wrench"].latest(at_ns, self.config.max_wrench_age_ns)
        gripper = timelines["gripper"].latest(
            at_ns, self.config.max_gripper_feedback_age_ns
        )
        external = timelines["external"].latest(at_ns, self.config.max_camera_age_ns)
        wrist = timelines["wrist"].latest(at_ns, self.config.max_camera_age_ns)
        tcp6 = _pose_tcp6(pose.get("pose", {}))
        width = float(gripper.get("width_m", -1.0))
        if not math.isfinite(width) or not 0.0 <= width <= 0.1:
            raise ProductionBridgeError("BRIDGE_OBSERVATION_GRIPPER_INVALID")
        return {
            "observation_id": observation_id,
            "episode_id": episode_id,
            "timestamp_monotonic_ns": at_ns,
            "clock_domain_id": self.config.clock_domain_id,
            "state7_absolute": list(tcp6 + (width,)),
            "wrench6": list(
                _finite_vector(
                    wrench.get("force_xyz_n_torque_xyz_nm"),
                    6,
                    "BRIDGE_WRENCH_INVALID",
                )
            ),
            "camera_external": self._camera(episode_dir, external, at_ns),
            "camera_wrist": self._camera(episode_dir, wrist, at_ns),
            "normalization": "not_applied_in_absolute_authority_bridge",
        }

    def _selected_pose(
        self,
        *,
        raw: Mapping[str, Any],
        safe: Mapping[str, Any],
        accepted_pose: Mapping[str, Any],
        accepted_reference: Mapping[str, Any],
    ) -> tuple[tuple[float, ...], float | None, dict[str, Any]]:
        if raw.get("source") != "policy":
            width = float(accepted_reference.get("target_gripper_width_m", -1.0))
            if not math.isfinite(width) or not 0.0 <= width <= 0.1:
                raise ProductionBridgeError("BRIDGE_ACCEPTED_REFERENCE_GRIPPER_INVALID")
            return _pose_tcp6(accepted_pose), width, {
                "kind": "recorded_offline_accepted_reference",
                "post_adapter_pose_source": "reference_ack.accepted_pose",
                "gripper_target_source": "accepted_reference.target_gripper_width_m",
            }
        selection = safe.get("forcesmolvla_chunk_selection")
        if not isinstance(selection, dict):
            raise ProductionBridgeError("BRIDGE_POLICY_SELECTION_MISSING")
        selected7 = _finite_vector(
            selection.get("selected_absolute_action7"),
            7,
            "BRIDGE_POLICY_SELECTED_ACTION7_INVALID",
        )
        applied = {
            "position_m": selection.get("applied_position_m"),
            "quaternion_xyzw": selection.get("applied_quaternion_xyzw"),
        }
        selected_tcp6 = _pose_tcp6(applied)
        accepted_tcp6 = _pose_tcp6(accepted_pose)
        position_error = math.dist(selected_tcp6[:3], accepted_tcp6[:3])
        rotation_error = math.dist(selected_tcp6[3:], accepted_tcp6[3:])
        if (
            position_error > self.config.pose_position_tolerance_m
            or rotation_error > self.config.pose_quaternion_tolerance_rad
        ):
            raise ProductionBridgeError("BRIDGE_POST_ADAPTER_POSE_ACK_MISMATCH")
        model_width = selected7[6]
        if model_width <= self.config.gripper_close_threshold_m:
            post_adapter_width: float | None = self.config.gripper_closed_width_m
        elif model_width >= self.config.gripper_open_threshold_m:
            post_adapter_width = self.config.gripper_open_width_m
        else:
            post_adapter_width = None
        return selected_tcp6, post_adapter_width, {
            "kind": "recorded_policy_h50_selection",
            "selected_model_absolute7": list(selected7),
            "post_adapter_pose_source": "forcesmolvla_chunk_selection.applied_pose",
            "post_adapter_gripper_target_width_m": post_adapter_width,
            "gripper_adapter_policy": "close<=0.030; open>=0.055; otherwise held",
        }

    def _finish_transition(self, payload: dict[str, Any]) -> dict[str, Any]:
        stable = {
            "schema_version": payload["schema_version"],
            "episode_id": payload["identity"]["episode_id"],
            "decision_id": payload["identity"]["decision_id"],
            "pose_ack_id": payload["action_authority"]["pose"]["pose_ack_id"],
            "gripper_goal_id": payload["action_authority"]["gripper"][
                "origin_action_goal_id"
            ],
        }
        payload["identity"]["transition_uid"] = _sha256_bytes(_canonical_bytes(stable))
        payload["integrity"] = {
            "canonical_payload_sha256": _sha256_bytes(_canonical_bytes(payload))
        }
        return payload

    def _transition(
        self,
        *,
        episode_dir: Path,
        episode_id: str,
        task: str,
        safe_record: Mapping[str, Any],
        next_safe_record: Mapping[str, Any],
        raw_by_identity: Mapping[tuple[str, int], dict[str, Any]],
        requested: Mapping[tuple[str, int], dict[str, Any]],
        ack_by_stamp: Mapping[int, dict[str, Any]],
        goals: Sequence[_Goal],
        used_new: set[int],
        timelines: Mapping[str, _Timeline],
        episode_result: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        safe = safe_record.get("payload", {})
        arbitration = safe.get("arbitration", {})
        raw = arbitration.get("raw_action", {})
        source, sequence = str(raw.get("source", "")), int(raw.get("sequence", -1))
        key = (source, sequence)
        recorded_raw = raw_by_identity.get(key)
        if recorded_raw is None or recorded_raw != raw:
            raise ProductionBridgeError("BRIDGE_RAW_SAFE_ACTION_IDENTITY_MISMATCH")
        request = requested.get(key)
        if request is None:
            raise ProductionBridgeError("BRIDGE_REQUESTED_EQUILIBRIUM_IDENTITY_MISSING")
        stamp = int(request.get("source_stamp_ns", 0))
        ack_record = ack_by_stamp.get(stamp)
        if ack_record is None:
            raise ProductionBridgeError("BRIDGE_POSE_ACK_MISSING")
        ack = ack_record.get("payload", {})
        if ack.get("accepted") is not True or int(ack.get("request_stamp_ns", 0)) != stamp:
            raise ProductionBridgeError("BRIDGE_POSE_ACK_REJECTED_OR_MISMATCHED")
        publish_ns = int(request.get("equilibrium_publish_monotonic_ns", 0))
        ack_receive_ns = int(ack_record.get("receive_monotonic_ns", 0))
        if not 0 < publish_ns <= ack_receive_ns:
            raise ProductionBridgeError("BRIDGE_POSE_ACK_UPPER_CLOCK_CAUSALITY_INVALID")
        if ack_receive_ns - publish_ns > self.config.max_pose_ack_latency_ns:
            raise ProductionBridgeError("BRIDGE_POSE_ACK_STALE")
        accepted_pose = ack.get("accepted_pose")
        if not isinstance(accepted_pose, dict):
            raise ProductionBridgeError("BRIDGE_POSE_ACK_ACCEPTED_POSE_MISSING")
        requested_pose = request.get("pose")
        if not isinstance(requested_pose, dict):
            raise ProductionBridgeError("BRIDGE_REQUESTED_POSE_MISSING")
        if (
            math.dist(_pose_tcp6(requested_pose)[:3], _pose_tcp6(accepted_pose)[:3])
            > self.config.pose_position_tolerance_m
        ):
            raise ProductionBridgeError("BRIDGE_REQUESTED_ACCEPTED_POSITION_MISMATCH")
        if _quaternion_distance(
            requested_pose.get("quaternion_xyzw", ()),
            accepted_pose.get("quaternion_xyzw", ()),
        ) > self.config.pose_quaternion_tolerance_rad:
            raise ProductionBridgeError("BRIDGE_REQUESTED_ACCEPTED_ROTATION_MISMATCH")
        accepted_reference = timelines["accepted"].latest(
            ack_receive_ns, self.config.max_pose_ack_latency_ns
        )
        accepted_reference_pose = accepted_reference.get("pose")
        if not isinstance(accepted_reference_pose, dict):
            raise ProductionBridgeError("BRIDGE_ACCEPTED_REFERENCE_POSE_MISSING")
        if (
            math.dist(
                _pose_tcp6(accepted_reference_pose)[:3],
                _pose_tcp6(accepted_pose)[:3],
            )
            > self.config.pose_position_tolerance_m
            or _quaternion_distance(
                accepted_reference_pose.get("quaternion_xyzw", ()),
                accepted_pose.get("quaternion_xyzw", ()),
            )
            > self.config.pose_quaternion_tolerance_rad
        ):
            raise ProductionBridgeError("BRIDGE_ACCEPTED_REFERENCE_ACK_MISMATCH")
        lineage, generation, policy_fixture = self._lineage(
            episode_id=episode_id, raw=raw, safe=safe
        )
        selected_tcp6, selected_width, selection_provenance = self._selected_pose(
            raw=raw,
            safe=safe,
            accepted_pose=accepted_pose,
            accepted_reference=accepted_reference,
        )
        goal, kind = self._goal_for_transition(
            goals=goals,
            authority_ns=ack_receive_ns,
            selected_width_m=selected_width,
            used_new=used_new,
        )
        if selected_width is None:
            selected_width = goal.requested_width_m
        transition_id = f"{episode_id}:decision:{int(safe.get('decision_id', -1))}"
        command_id = f"reference-request-stamp:{stamp}"
        pose = PoseAcceptedAuthority(
            transition_id=transition_id,
            pose_command_id=command_id,
            pose_ack_id=command_id,
            pose_ack_monotonic_ns=ack_receive_ns,
            selected_post_adapter_tcp6=selected_tcp6,
            declared_gripper_origin_action_goal_id=goal.action_goal_id,
            clock_domain_id=self.config.clock_domain_id,
            generation=generation,
        ).validate()
        feedback_fields: dict[str, Any] = {}
        authority_ns = max(ack_receive_ns, goal.accepted_ns)
        if kind is GripperAuthorityKind.HELD_FROM_ACCEPTED_COMMAND:
            feedback = timelines["gripper"].latest(
                ack_receive_ns, self.config.max_gripper_feedback_age_ns
            )
            feedback_ns = int(feedback.get("receive_monotonic_ns", 0))
            feedback_fields = {
                "feedback_width_m": float(feedback.get("width_m", -1.0)),
                "feedback_state": (
                    "OPEN" if float(feedback.get("width_m", -1.0)) >= 0.055 else "CLOSED"
                ),
                "feedback_monotonic_ns": feedback_ns,
                "feedback_age_ns": ack_receive_ns - feedback_ns,
            }
        gripper = GripperAuthorityEvidence(
            transition_id=transition_id,
            authority_kind=kind,
            origin_local_goal_sequence=goal.sequence,
            origin_action_goal_id=goal.action_goal_id,
            origin_accepted_monotonic_ns=goal.accepted_ns,
            requested_state=goal.requested_state,
            requested_width_m=goal.requested_width_m,
            authority_monotonic_ns=authority_ns,
            clock_domain_id=self.config.clock_domain_id,
            generation=generation,
            terminal_outcome=goal.outcome,
            terminal_finished_monotonic_ns=goal.finished_ns,
            terminal_sealed=True,
            **feedback_fields,
        ).validate()
        full = close_full_action7_authority(
            pose=pose,
            selected_post_adapter_tcp6=selected_tcp6,
            selected_gripper_width_m=selected_width,
            gripper=gripper,
            requested_width_tolerance_m=self.config.requested_width_tolerance_m,
        )
        if kind is GripperAuthorityKind.NEW_COMMAND:
            used_new.add(goal.sequence)
        at_ns = int(safe.get("accept_monotonic_ns", safe_record.get("receive_monotonic_ns", 0)))
        next_safe = next_safe_record.get("payload", {})
        next_at_ns = int(
            next_safe.get(
                "accept_monotonic_ns", next_safe_record.get("receive_monotonic_ns", 0)
            )
        )
        if next_at_ns <= at_ns:
            raise ProductionBridgeError("BRIDGE_NEXT_OBSERVATION_NOT_CAUSAL")
        macro_span_ns = next_at_ns - at_ns
        if macro_span_ns < self.config.minimum_full_macro_span_ns:
            raise ProductionBridgeError("BRIDGE_PARTIAL_MACRO_AT_BOUNDARY")
        macro_grid_ns = [
            at_ns + (slot * 1_000_000_000) // 30 for slot in range(3)
        ]
        if macro_grid_ns[-1] >= next_at_ns:
            raise ProductionBridgeError("BRIDGE_PARTIAL_MACRO_AT_BOUNDARY")
        decision = int(safe.get("decision_id", -1))
        observation = self._observation(
            episode_dir=episode_dir,
            episode_id=episode_id,
            observation_id=f"{episode_id}:observation:{decision}",
            at_ns=at_ns,
            timelines=timelines,
        )
        next_observation = self._observation(
            episode_dir=episode_dir,
            episode_id=episode_id,
            observation_id=(
                f"{episode_id}:observation:"
                f"{int(next_safe.get('decision_id', decision + 1))}"
            ),
            at_ns=next_at_ns,
            timelines=timelines,
        )
        next_phase = str(
            next_safe.get("arbitration", {}).get("raw_action", {}).get("phase", "")
        )
        reward_available = "reward" in episode_result
        payload = {
            "schema_version": SCHEMA_VERSION,
            "classification": (
                "recorded_live_policy_shadow" if policy_fixture else "recorded_offline_shadow"
            ),
            "identity": {
                "episode_id": episode_id,
                "decision_id": decision,
                "task": task,
                "transition_uid": None,
            },
            "runtime_lineage": lineage,
            "generation": asdict(generation),
            "action_authority": {
                "selected_post_adapter_tcp6": list(selected_tcp6),
                "accepted_absolute_action7": list(full.accepted_absolute_action7),
                "pose": _enum_json(asdict(pose)),
                "accepted_reference": {
                    "source_stamp_ns": accepted_reference.get("source_stamp_ns"),
                    "accepted_receive_monotonic_ns": accepted_reference.get(
                        "accepted_receive_monotonic_ns"
                    ),
                    "clock_domain_id": self.config.clock_domain_id,
                },
                "gripper": _enum_json(asdict(gripper)),
                "full_action7_ack_closure": True,
                "selection_provenance": selection_provenance,
                "remote_pose_ack_timestamps_uncompared": {
                    "request_receive_monotonic_ns": ack.get(
                        "request_receive_monotonic_ns"
                    ),
                    "ack_monotonic_ns": ack.get("ack_monotonic_ns"),
                    "clock_domain_id": "controller_host_monotonic",
                },
            },
            "ack_macro": {
                "projection": "causal_zoh_on_stage3_30hz_grid",
                "projection_grid_hz": 30,
                "controller_internal_servo_hz": "UNVERIFIED",
                "K": 3,
                "grid_monotonic_ns": macro_grid_ns,
                "ack_ids": [pose.pose_ack_id] * 3,
                "gripper_origin_action_goal_ids": [goal.action_goal_id] * 3,
                "accepted_absolute_action_k7": [
                    list(full.accepted_absolute_action7) for _ in range(3)
                ],
                "slot_owner": [
                    "policy" if source == "policy" else "offline_demonstration"
                ]
                * 3,
                "accepted_action_source": [
                    "policy" if source == "policy" else "offline"
                ]
                * 3,
                "intervention_flags": [bool(raw.get("intervention", False))] * 3,
                "partial": False,
                "span_to_next_dispatch_ns": macro_span_ns,
            },
            "observation": observation,
            "next_observation": next_observation,
            "behavior": {
                "recorder_source": source,
                "recorder_owner": arbitration.get("owner"),
                "intervention": bool(raw.get("intervention", False)),
                "phase": raw.get("phase"),
                "workspace_clipped": safe.get("workspace_clipped"),
            },
            "recorder_evidence": {
                "raw_action": {
                    "schema": raw.get("schema"),
                    "source": source,
                    "sequence": sequence,
                    "source_monotonic_ns": raw.get("source_monotonic_ns"),
                    "action": raw.get("action"),
                    "observation_id": raw.get("observation_id"),
                },
                "safe_action": {
                    "schema": safe.get("schema"),
                    "decision_id": decision,
                    "accept_monotonic_ns": safe.get("accept_monotonic_ns"),
                    "safe_action": safe.get("safe_action"),
                    "arbitration_reason": arbitration.get("reason"),
                },
                "requested_equilibrium": {
                    "source_stamp_ns": stamp,
                    "equilibrium_publish_monotonic_ns": publish_ns,
                    "pose": requested_pose,
                },
                "reference_ack": {
                    "request_stamp_ns": ack.get("request_stamp_ns"),
                    "request_sequence": ack.get("request_sequence"),
                    "accepted": ack.get("accepted"),
                    "upper_receive_monotonic_ns": ack_receive_ns,
                    "accepted_pose": accepted_pose,
                },
                "episode_result_payload_sha256": _sha256_bytes(
                    _canonical_bytes(episode_result)
                ),
            },
            "outcome": {
                "reward_available": reward_available,
                "reward": (
                    episode_result.get("reward")
                    if next_phase == "episode_end"
                    else (0.0 if reward_available else None)
                ),
                "reward_source": (
                    "episode_result" if reward_available else "absent_from_recorder"
                ),
                "episode_boundary": next_phase == "episode_end",
                "task_terminated": (
                    bool(episode_result.get("terminated"))
                    if reward_available and next_phase == "episode_end"
                    else None
                ),
                "terminal_source": "accepted_episode_result_seal",
                "episode_saved": True,
            },
            "eligibility": {
                "shadow_outbox_eligible": True,
                "formal_training_replay_eligible": False,
                "recorded_live_policy_fixture": policy_fixture,
                "real_online_r": False,
            },
            "commit": {
                "episode_sealed": True,
                "pose_ack_watermark": int(ack.get("request_sequence", 0)),
                "gripper_terminal_sealed": True,
                "normalizer_invocations": 0,
                "normalization_boundary": "deferred_to_frozen_replay_adapter",
            },
        }
        return self._finish_transition(payload), policy_fixture

    def _episode_quarantine(
        self,
        *,
        episode_id: str,
        episode_key: str,
        result_path: Path,
        error: Exception,
        dry_run: bool,
    ) -> BridgeReport:
        if not dry_run:
            self._immutable_write(
                self.state_root / "quarantine" / f"episode__{episode_key}.json",
                {
                    "schema_version": REPORT_VERSION,
                    "episode_id": episode_id,
                    "reason": str(error),
                    "episode_result_sha256": _sha256_file(result_path),
                },
            )
        return BridgeReport(
            status="SEALED_QUARANTINED",
            episode_id=episode_id,
            sealed=True,
            dry_run=dry_run,
            candidate_count=0,
            outbox_eligible_count=0,
            quarantined_count=1,
            wal_written_count=0,
            outbox_written_count=0,
            idempotent_count=0,
            quarantine_reasons=(str(error),),
            recorded_offline_production_bridge="FAIL",
            policy_fixture=False,
        )

    def _immutable_write(self, path: Path, value: Mapping[str, Any]) -> bool:
        data = _canonical_bytes(value) + b"\n"
        if path.exists():
            existing = path.read_bytes()
            if existing == data:
                return False
            raise BridgeDigestCollisionError(f"BRIDGE_IMMUTABLE_COLLISION:{path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise BridgeDigestCollisionError(
                    f"BRIDGE_IMMUTABLE_COLLISION:{path.name}"
                )
            return False
        finally:
            temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
        return True

    def _stage(self, episode_key: str, value: Mapping[str, Any]) -> None:
        path = self.state_root / "staging" / f"{episode_key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
        data = _canonical_bytes(value) + b"\n"
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)

    def process_episode(
        self,
        episode_dir: Path,
        *,
        dry_run: bool = False,
        inject_crash_after_wal: int | None = None,
    ) -> BridgeReport:
        episode_dir = Path(episode_dir)
        start = _read_json(episode_dir / "episode_start.json")
        episode_id = self._episode_id(episode_dir, start)
        episode_key = episode_id.replace("/", "__")
        result_path = episode_dir / "episode_result.json"
        if not result_path.is_file():
            cursors: dict[str, dict[str, int]] = {}
            candidate_count = 0
            streams_root = episode_dir / "streams"
            if streams_root.is_dir():
                for path in sorted(streams_root.glob("*.jsonl")):
                    records = _read_jsonl(path)
                    cursors[path.stem] = {
                        "record_count": len(records),
                        "byte_offset": path.stat().st_size,
                    }
                    if path.name == "safe_action.jsonl":
                        candidate_count = sum(
                            item.get("payload", {})
                            .get("arbitration", {})
                            .get("raw_action", {})
                            .get("phase")
                            == "control"
                            for item in records
                        )
            if not dry_run:
                self._stage(
                    episode_key,
                    {
                        "schema_version": REPORT_VERSION,
                        "episode_id": episode_id,
                        "status": "ACTIVE_STAGED",
                        "stream_cursors": cursors,
                        "episode_start_sha256": _sha256_file(
                            episode_dir / "episode_start.json"
                        ),
                    },
                )
            return BridgeReport(
                status="ACTIVE_STAGED",
                episode_id=episode_id,
                sealed=False,
                dry_run=dry_run,
                candidate_count=candidate_count,
                outbox_eligible_count=0,
                quarantined_count=0,
                wal_written_count=0,
                outbox_written_count=0,
                idempotent_count=0,
                quarantine_reasons=(),
                recorded_offline_production_bridge="BLOCKED",
                policy_fixture=False,
            )
        result = _read_json(result_path)
        try:
            streams = self._load_streams(episode_dir)
            self._validate_seal(result, streams)
            goals = self._goals(streams)
        except ProductionBridgeError as error:
            return self._episode_quarantine(
                episode_id=episode_id,
                episode_key=episode_key,
                result_path=result_path,
                error=error,
                dry_run=dry_run,
            )
        try:
            requested: dict[tuple[str, int], dict[str, Any]] = {}
            for item in streams["requested_equilibrium"]:
                key = (str(item.get("source", "")), int(item.get("sequence", -1)))
                if key in requested:
                    raise ProductionBridgeError("BRIDGE_REQUESTED_IDENTITY_DUPLICATE")
                requested[key] = item
            raw_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
            for item in streams["raw_action"]:
                raw = item.get("payload", {})
                key = (str(raw.get("source", "")), int(raw.get("sequence", -1)))
                if key in raw_by_identity:
                    raise ProductionBridgeError("BRIDGE_RAW_ACTION_IDENTITY_DUPLICATE")
                raw_by_identity[key] = raw
            ack_by_stamp: dict[int, dict[str, Any]] = {}
            for item in streams["reference_ack"]:
                stamp = int(item.get("payload", {}).get("request_stamp_ns", 0))
                if stamp <= 0 or stamp in ack_by_stamp:
                    raise ProductionBridgeError("BRIDGE_POSE_ACK_IDENTITY_DUPLICATE")
                ack_by_stamp[stamp] = item
            timelines = {
                "pose": _Timeline(streams["measured_tcp_pose"], "receive_monotonic_ns"),
                "wrench": _Timeline(streams["wrench_notch_sensor"], "receive_monotonic_ns"),
                "gripper": _Timeline(streams["gripper_state"], "receive_monotonic_ns"),
                "accepted": _Timeline(
                    streams["accepted_reference"], "accepted_receive_monotonic_ns"
                ),
                "external": _Timeline(streams["external_camera"], "receive_monotonic_ns"),
                "wrist": _Timeline(streams["wrist_camera"], "receive_monotonic_ns"),
            }
        except ProductionBridgeError as error:
            return self._episode_quarantine(
                episode_id=episode_id,
                episode_key=episode_key,
                result_path=result_path,
                error=error,
                dry_run=dry_run,
            )
        safe_records = streams["safe_action"]
        control_indices = [
            index
            for index, item in enumerate(safe_records[:-1])
            if item.get("payload", {})
            .get("arbitration", {})
            .get("raw_action", {})
            .get("phase")
            == "control"
        ]
        used_new: set[int] = set()
        transitions: list[dict[str, Any]] = []
        quarantines: list[dict[str, Any]] = []
        policy_fixture = False
        task = str(result.get("task", start.get("task", "")))
        for index in control_indices:
            item = safe_records[index]
            decision = int(item.get("payload", {}).get("decision_id", -1))
            try:
                transition, is_policy = self._transition(
                    episode_dir=episode_dir,
                    episode_id=episode_id,
                    task=task,
                    safe_record=item,
                    next_safe_record=safe_records[index + 1],
                    raw_by_identity=raw_by_identity,
                    requested=requested,
                    ack_by_stamp=ack_by_stamp,
                    goals=goals,
                    used_new=used_new,
                    timelines=timelines,
                    episode_result=result,
                )
                transitions.append(transition)
                policy_fixture = policy_fixture or is_policy
            except (ProductionBridgeError, GripperProvenanceError) as error:
                quarantines.append(
                    {
                        "schema_version": REPORT_VERSION,
                        "episode_id": episode_id,
                        "decision_id": decision,
                        "reason": str(error),
                        "formal_training_replay_eligible": False,
                    }
                )
        wal_written = outbox_written = idempotent = 0
        if not dry_run:
            for transition in transitions:
                uid = transition["identity"]["transition_uid"]
                digest = transition["integrity"]["canonical_payload_sha256"]
                wal_record = {
                    "transition_uid": uid,
                    "canonical_payload_sha256": digest,
                    "payload": transition,
                }
                if self._immutable_write(self.state_root / "wal" / f"{uid}.json", wal_record):
                    wal_written += 1
                else:
                    idempotent += 1
                if inject_crash_after_wal is not None and wal_written >= inject_crash_after_wal:
                    raise InjectedBridgeCrash("BRIDGE_INJECTED_AFTER_WAL")
                outbox_record = {
                    "transition_uid": uid,
                    "canonical_payload_sha256": digest,
                    "wal_record": f"wal/{uid}.json",
                    "formal_training_replay_written": False,
                }
                if self._immutable_write(
                    self.state_root / "outbox" / f"{uid}.json", outbox_record
                ):
                    outbox_written += 1
                else:
                    idempotent += 1
            for item in quarantines:
                stable = {
                    "episode_id": episode_id,
                    "decision_id": item["decision_id"],
                }
                name = _sha256_bytes(_canonical_bytes(stable))
                self._immutable_write(
                    self.state_root / "quarantine" / f"transition__{name}.json", item
                )
            manifest = {
                "schema_version": REPORT_VERSION,
                "episode_id": episode_id,
                "status": "SEALED_COMMITTED",
                "transition_uids": [
                    item["identity"]["transition_uid"] for item in transitions
                ],
                "quarantined_decisions": [item["decision_id"] for item in quarantines],
                "episode_result_sha256": _sha256_file(result_path),
                "formal_training_replay_written": False,
            }
            self._immutable_write(
                self.state_root / "episodes" / f"{episode_key}.json", manifest
            )
            self._stage(episode_key, manifest)
        status = "DRY_RUN_READY" if dry_run else "SEALED_COMMITTED"
        offline_pass = "PASS" if transitions and not policy_fixture else "BLOCKED"
        authority_kinds = [
            item["action_authority"]["gripper"]["authority_kind"]
            for item in transitions
        ]
        return BridgeReport(
            status=status,
            episode_id=episode_id,
            sealed=True,
            dry_run=dry_run,
            candidate_count=len(control_indices),
            outbox_eligible_count=len(transitions),
            quarantined_count=len(quarantines),
            wal_written_count=wal_written,
            outbox_written_count=outbox_written,
            idempotent_count=idempotent,
            quarantine_reasons=tuple(sorted({item["reason"] for item in quarantines})),
            recorded_offline_production_bridge=offline_pass,
            policy_fixture=policy_fixture,
            new_command_count=authority_kinds.count("NEW_COMMAND"),
            held_command_count=authority_kinds.count("HELD_FROM_ACCEPTED_COMMAND"),
        )
