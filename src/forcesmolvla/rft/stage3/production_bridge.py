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
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from forcesmolvla.raw_to_lerobot_v3 import (
    PreparedEpisode,
    RuntimeContract,
    prepare_episode,
)
from forcesmolvla.rft.detector_reward_transitions import (
    CHECKPOINT_SHA256,
    REQUIRED_CONSECUTIVE_FRAMES,
    REWARD_SOURCE,
    TAU,
    DetectionTrace,
    DetectorMacroTransition,
    causal_detection_trace,
    detector_macro_transitions,
)

from .gripper_provenance import (
    GripperAuthorityEvidence,
    GripperAuthorityKind,
    GripperGeneration,
    GripperProvenanceError,
    PoseAcceptedAuthority,
    VALID_TERMINAL_OUTCOMES,
    close_full_action7_authority,
)
from .policy_lineage import InitialGripperAuthority, POLICY_LINEAGE_SCHEMA


SCHEMA_VERSION = "forcesmolvla_stage3_production_bridge_transition.v1"
REPORT_VERSION = "forcesmolvla_stage3_production_bridge_report.v1"
INTEGRATED_CAPTURE_SCHEMA = "forcesmolvla-stage3-integrated-capture-v1"
INTEGRATED_SHADOW_SCHEMA = "forcesmolvla-stage3-integrated-shadow-backend-v1"
INTEGRATED_POLICY_EXECUTION_SCHEMA = (
    "forcesmolvla-stage3-integrated-policy-execution-backend-v1"
)
POLICY_EXECUTION_SMOKE_CLASSIFICATION = "recorded_live_policy_execution_smoke"
TRAINING_STARTS_UNIQUE_R = 100
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
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PARENT_BINDING = REPO_ROOT / "configs/stage3_parent_binding.v1.development.json"
DEFAULT_G1_CONFIG = REPO_ROOT / "configs/stage2_g1_frozen_detector_transition_view.development.json"
DEFAULT_REWARD_CONTRACT = (
    REPO_ROOT / "configs/stage3_reward_terminal_contract.v1.development.json"
)


class ProductionBridgeError(RuntimeError):
    """Fail-closed recorder or persistence contract violation."""


class BridgeDigestCollisionError(ProductionBridgeError):
    """A stable UID was observed with different canonical content."""


class InjectedBridgeCrash(ProductionBridgeError):
    """Test-only crash point after an immutable WAL write."""


@dataclass(frozen=True)
class FrozenDetectorScores:
    """One frozen-classifier pass over the current prepared episode."""

    probabilities: tuple[float, ...]
    validity: tuple[bool, ...]
    detector_id: str = CHECKPOINT_SHA256
    config_identity: str = "stage2_g1_frozen_detector_transition_view.development"


@dataclass(frozen=True)
class EpisodeMaterialization:
    """Calibrated 30 Hz episode plus its causally detected G1 boundary."""

    prepared: PreparedEpisode
    detector_scores: FrozenDetectorScores
    detection_trace: DetectionTrace
    macros: tuple[DetectorMacroTransition, ...]
    wrench_provenance: Mapping[str, Any]
    outcome_provenance: Mapping[str, Any]

    def validate(self) -> "EpisodeMaterialization":
        count = len(self.prepared.tuple_host_ns)
        if count < 2 or len(self.detector_scores.probabilities) != count:
            raise ProductionBridgeError("BRIDGE_DETECTOR_FRAME_COUNT_MISMATCH")
        if len(self.detector_scores.validity) != count:
            raise ProductionBridgeError("BRIDGE_DETECTOR_VALIDITY_COUNT_MISMATCH")
        if self.detector_scores.detector_id != CHECKPOINT_SHA256:
            raise ProductionBridgeError("BRIDGE_DETECTOR_IDENTITY_MISMATCH")
        if self.detection_trace.trigger_frame is None or not self.macros:
            raise ProductionBridgeError("BRIDGE_FROZEN_G1_DETECTOR_MISS")
        if self.macros[-1].next_frame != self.detection_trace.trigger_frame:
            raise ProductionBridgeError("BRIDGE_DETECTOR_MACRO_CLOSURE_INVALID")
        if not np.all(np.isfinite(self.prepared.wrench6)):
            raise ProductionBridgeError("BRIDGE_MATERIALIZED_WRENCH_NONFINITE")
        return self


EpisodeMaterializer = Callable[[Path], EpisodeMaterialization]
FrozenDetector = Callable[[PreparedEpisode], FrozenDetectorScores]


def _prepare_native_episode(
    episode_dir: Path,
    *,
    parent_binding_path: Path = DEFAULT_PARENT_BINDING,
) -> PreparedEpisode:
    """Materialize calibrated observations without loading the reward model."""

    parent = _read_json(Path(parent_binding_path))
    calibration_payload = _read_json(
        Path(parent["calibration_binding"]["absolute_path"])
    )
    runtime_contract = RuntimeContract.from_development_json(
        Path(parent["runtime_contract_binding"]["absolute_path"])
    )
    episode_dir = Path(episode_dir)
    return prepare_episode(
        episode_dir,
        session=_read_json(episode_dir.parent.parent / "session.json"),
        calibration_payload=calibration_payload,
        contract=runtime_contract,
    )


def frozen_episode_materializer(
    detector: FrozenDetector,
    *,
    parent_binding_path: Path = DEFAULT_PARENT_BINDING,
    detector_config_path: Path = DEFAULT_G1_CONFIG,
) -> EpisodeMaterializer:
    """Bind the existing single-episode converter and frozen G1 detector."""

    parent = _read_json(Path(parent_binding_path))
    detector_config = _read_json(Path(detector_config_path))
    spec = detector_config.get("detector_spec", {})
    reward_contract = detector_config.get("reward_contract", {})
    if (
        spec.get("classifier_checkpoint_sha256") != CHECKPOINT_SHA256
        or float(spec.get("probability_threshold", -1.0)) != TAU
        or int(spec.get("required_consecutive_frames", -1))
        != REQUIRED_CONSECUTIVE_FRAMES
        or int(spec.get("detector_input_rate_hz", -1)) != 30
        or reward_contract.get("reward_source") != REWARD_SOURCE
        or reward_contract.get("detector_miss_policy") != "exclude_without_fallback"
    ):
        raise ProductionBridgeError("BRIDGE_FROZEN_G1_CONFIG_MISMATCH")
    calibration_path = Path(parent["calibration_binding"]["absolute_path"])
    runtime_path = Path(parent["runtime_contract_binding"]["absolute_path"])
    calibration_payload = _read_json(calibration_path)
    runtime_contract = RuntimeContract.from_development_json(runtime_path)
    parent_id = str(parent.get("binding_id", ""))
    if not parent_id:
        raise ProductionBridgeError("BRIDGE_PARENT_BINDING_ID_MISSING")

    def materialize(episode_dir: Path) -> EpisodeMaterialization:
        episode_dir = Path(episode_dir)
        session = _read_json(episode_dir.parent.parent / "session.json")
        try:
            prepared = prepare_episode(
                episode_dir,
                session=session,
                calibration_payload=calibration_payload,
                contract=runtime_contract,
            )
            scores = detector(prepared)
            trace = causal_detection_trace(
                range(len(prepared.tuple_host_ns)),
                scores.probabilities,
                scores.validity,
                tau=TAU,
                required=REQUIRED_CONSECUTIVE_FRAMES,
            )
            macros = (
                ()
                if trace.trigger_frame is None
                else detector_macro_transitions(trace.trigger_frame)
            )
        except ProductionBridgeError:
            raise
        except Exception as error:
            raise ProductionBridgeError(
                f"BRIDGE_EPISODE_MATERIALIZATION_FAILED:{type(error).__name__}:{error}"
            ) from error
        return EpisodeMaterialization(
            prepared=prepared,
            detector_scores=scores,
            detection_trace=trace,
            macros=macros,
            wrench_provenance={
                "source": "raw_to_lerobot_v3.prepare_episode",
                "parent_binding_id": parent_id,
                "calibration_path": str(calibration_path),
                "runtime_contract_path": str(runtime_path),
                "operations": [
                    "sensor_bias",
                    "wrench_sign",
                    "payload_gravity_force_and_moment_compensation",
                    "measurement_to_fr3_link0",
                    "moment_shift_to_tcp",
                    "causal_sos_filter",
                    "rational_30hz_alignment",
                ],
                "normalizer_refit": False,
                "raw_wrench_learner_eligible": False,
                "diagnostics": prepared.diagnostics,
            },
            outcome_provenance={
                "source": REWARD_SOURCE,
                "detector_id": scores.detector_id,
                "detector_config_identity": scores.config_identity,
                "detector_config_path": str(detector_config_path),
                "probability_threshold": TAU,
                "required_consecutive_frames": REQUIRED_CONSECUTIVE_FRAMES,
                "trigger_frame": trace.trigger_frame,
                "streak_start_frame": trace.streak_start_frame,
                "manual_boundary_used": False,
                "episode_result_fallback_used": False,
                "time_limit_fallback_used": False,
            },
        ).validate()

    return materialize


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
    classification: str = "recorded_offline_shadow"
    technical_seal: str = "complete"
    operator_task_outcome: str | None = None
    executed_action_source: str = "human"
    policy_execution: bool = False
    detector_outcome: str = "not_evaluated"
    detector_trigger_frame: int | None = None
    shadow_observation_count: int = 0
    shadow_policy_request_count: int = 0
    shadow_policy_result_count: int = 0
    shadow_policy_proposal_count: int = 0
    shadow_human_ack_count: int = 0
    training_replay_eligible: bool = False
    policy_lineage_complete: bool = False
    policy_chunk_count: int = 0
    policy_action_ack_count: int = 0
    human_override_count: int = 0
    human_override_executed_count: int = 0
    model_update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FormalOnlineRAdmissionReport:
    status: str
    episode_id: str
    classification: str
    policy_execution_smoke_bridge: str
    accepted_unique_r_transition_count: int
    total_unique_r_transition_count: int
    training_starts: int
    training_starts_reached: bool
    human_override_count: int
    human_override_replay_count: int
    invalidated_proposal_replay_count: int
    observation_warmup_excluded_count: int
    wal_written_count: int
    outbox_written_count: int
    replay_written_count: int
    idempotent_transition_count: int
    admission_record_written: bool
    episode_seal_written: bool
    actor_update_count: int = 0
    critic_update_count: int = 0
    optimizer_update_count: int = 0
    checkpoint_update_count: int = 0
    policy_revision_published_count: int = 0

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
    generation: GripperGeneration | None = None
    initial_authority: bool = False


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

    def __init__(
        self,
        *,
        config: BridgeConfig,
        state_root: Path,
        episode_materializer: EpisodeMaterializer | None = None,
    ) -> None:
        self.config = config.validate()
        self.state_root = Path(state_root)
        self.episode_materializer = episode_materializer
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

    @staticmethod
    def _validate_shadow_identity(
        row: Mapping[str, Any], identity: Mapping[str, Any]
    ) -> None:
        for field in (
            "session_id",
            "episode_id",
            "clock_domain_id",
            "policy_revision",
            "policy_epoch",
            "reset_generation",
            "takeover_generation",
        ):
            if row.get(field) != identity.get(field):
                raise ProductionBridgeError(
                    f"BRIDGE_SHADOW_IDENTITY_MISMATCH:{field}"
                )

    @staticmethod
    def _policy_execution_generation(
        row: Mapping[str, Any], identity: Mapping[str, Any]
    ) -> tuple[int, int, int]:
        for field in (
            "session_id",
            "episode_id",
            "clock_domain_id",
            "policy_revision",
        ):
            if row.get(field) != identity.get(field):
                raise ProductionBridgeError(
                    f"BRIDGE_POLICY_EXECUTION_IDENTITY_MISMATCH:{field}"
                )
        try:
            generation = (
                int(row["policy_epoch"]),
                int(row["reset_generation"]),
                int(row["takeover_generation"]),
            )
            initial_generation = (
                int(identity["policy_epoch"]),
                int(identity["reset_generation"]),
                int(identity["takeover_generation"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_GENERATION_MISSING"
            ) from error
        policy_delta = generation[0] - initial_generation[0]
        takeover_delta = generation[2] - initial_generation[2]
        if (
            min(generation) < 0
            or min(initial_generation) < 0
            or generation[1] != initial_generation[1]
            or policy_delta < 0
            or takeover_delta < 0
            or policy_delta != takeover_delta
        ):
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_GENERATION_INVALID"
            )
        return generation

    @staticmethod
    def _validate_policy_execution_generation_step(
        previous: tuple[int, int, int], current: tuple[int, int, int]
    ) -> None:
        if current not in {
            previous,
            (previous[0] + 1, previous[1], previous[2] + 1),
        }:
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_GENERATION_INVALID"
            )

    def _load_integrated_policy_execution(
        self,
        *,
        episode_dir: Path,
        native_result: Mapping[str, Any],
        streams: Mapping[str, list[dict[str, Any]]],
        operator_task_outcome: str | None,
    ) -> dict[str, Any]:
        dataset_root = episode_dir.parent.parent
        manifest = _read_json(dataset_root / "integrated_capture_session.json")
        contract = manifest.get("contract")
        identity = contract.get("identity") if isinstance(contract, Mapping) else None
        async_metadata = (
            manifest.get("learner_resume_checkpoint"),
            manifest.get("active_actor_revision"),
            manifest.get("pending_candidate_id"),
        )
        async_learner = any(value is not None for value in async_metadata)
        if operator_task_outcome != "success":
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_OPERATOR_SUCCESS_REQUIRED"
            )
        if (
            manifest.get("schema") != INTEGRATED_POLICY_EXECUTION_SCHEMA
            or not isinstance(contract, Mapping)
            or contract.get("schema") != INTEGRATED_CAPTURE_SCHEMA
            or not isinstance(identity, Mapping)
            or identity.get("episode_id") != episode_dir.name
            or identity.get("clock_domain_id") != UPPER_CLOCK
            or contract.get("mode") != "policy-execute"
            or contract.get("actual_action_source") != "policy"
            or contract.get("policy_inference") is not True
            or contract.get("policy_execution") is not True
            or contract.get("formal_replay") is not False
            or contract.get("real_online_r") is not False
            or contract.get("development_policy_execution_smoke") is not True
            or contract.get("controller_owner") != "recorder"
            or contract.get("controller_process_count") != 1
            or contract.get("recorder_controller") is not True
            or contract.get("deploy_controller") is not False
            or manifest.get("controller_owner") != "recorder"
            or manifest.get("controller_process_count") != 1
            or manifest.get("deploy_controller_started") is not False
            or manifest.get("policy_action_publisher_created") is not True
            or manifest.get("formal_replay_writer_started") is not False
            or manifest.get("learner_started") is not False
            or manifest.get("policy_revision_publisher_started") is not False
            or (
                async_learner
                and not all(
                    isinstance(value, str) and bool(value)
                    for value in async_metadata
                )
            )
        ):
            raise ProductionBridgeError(
                "BRIDGE_INTEGRATED_POLICY_EXECUTION_CONTRACT_INVALID"
            )
        clock = manifest.get("clock_binding")
        if (
            not isinstance(clock, Mapping)
            or clock.get("stage3_clock_domain_id") != UPPER_CLOCK
            or clock.get("policy_request_clock_domain_id")
            != "upper_host_monotonic_ns"
            or clock.get("native_primary_alignment_clock")
            != "upper_host_receive_monotonic_ns"
            or clock.get("same_upper_host_monotonic_epoch") is not True
        ):
            raise ProductionBridgeError(
                "BRIDGE_INTEGRATED_POLICY_EXECUTION_CLOCK_INVALID"
            )
        session = _read_json(dataset_root / "session.json")
        metadata = manifest.get("policy_metadata")
        cameras = session.get("cameras")
        external_camera = (
            cameras.get("observation.image") if isinstance(cameras, Mapping) else None
        )
        wrist_camera = (
            cameras.get("observation.wrist_image")
            if isinstance(cameras, Mapping)
            else None
        )
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("model_sha256") != identity.get("policy_revision")
            or not str(metadata.get("dataset_repo_id", ""))
            or metadata.get("tool_profile_sha256")
            != session.get("tool_config_hash")
            or not str(metadata.get("calibration_id", ""))
            or not isinstance(external_camera, Mapping)
            or external_camera.get("model") != "D435"
            or external_camera.get("role") != "external"
            or not isinstance(wrist_camera, Mapping)
            or wrist_camera.get("model") != "D405"
            or wrist_camera.get("role") != "wrist"
            or "applied offline" not in str(session.get("force_status", ""))
        ):
            raise ProductionBridgeError(
                "BRIDGE_INTEGRATED_POLICY_EXECUTION_OBSERVATION_BINDING_INVALID"
            )

        stream_root = (
            dataset_root / "integrated_capture" / episode_dir.name / "streams"
        )
        names = {
            "observations": "policy_execute_observation.jsonl",
            "requests": "policy_execute_request.jsonl",
            "results": "policy_execute_result.jsonl",
            "proposals": "policy_execute_proposal.jsonl",
            "chunks": "policy_execute_chunk.jsonl",
            "transitions": "policy_execute_transition.jsonl",
            "gripper_authorities": "policy_execute_gripper_authority.jsonl",
            "interventions": "policy_execute_intervention.jsonl",
        }
        if any(not (stream_root / name).is_file() for name in names.values()):
            raise ProductionBridgeError(
                "BRIDGE_INTEGRATED_POLICY_EXECUTION_STREAM_MISSING"
            )
        rows = {
            key: _read_jsonl(stream_root / name) for key, name in names.items()
        }
        observations = rows["observations"]
        requests = rows["requests"]
        results = rows["results"]
        proposals = rows["proposals"]
        chunks = rows["chunks"]
        transitions = rows["transitions"]
        gripper_authorities = rows["gripper_authorities"]
        interventions = rows["interventions"]
        if (
            not observations
            or not requests
            or not transitions
            or not (
                len(requests) == len(results) == len(proposals) == len(chunks)
            )
            or len(transitions) != len(gripper_authorities)
        ):
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_LINEAGE_COUNT_MISMATCH"
            )

        native_by_source: dict[str, dict[int, dict[str, Any]]] = {}
        for name in ("measured_tcp_pose", "wrench_notch_sensor", "gripper_state"):
            native_by_source[name] = {
                int(item.get("source_stamp_ns", 0)): item for item in streams[name]
            }
        observation_by_id: dict[str, dict[str, Any]] = {}
        previous_t_ref = 0
        previous_generation = self._policy_execution_generation(identity, identity)
        required_observation_streams = set(native_by_source) | {
            "external_camera",
            "wrist_camera",
        }
        receive_skew_limits = {
            "measured_tcp_pose": self.config.max_pose_age_ns,
            "wrench_notch_sensor": self.config.max_wrench_age_ns,
            "gripper_state": self.config.max_gripper_feedback_age_ns,
        }
        for index, observation in enumerate(observations):
            generation = self._policy_execution_generation(observation, identity)
            self._validate_policy_execution_generation_step(
                previous_generation, generation
            )
            observation_id = str(observation.get("observation_id", ""))
            timestamps = observation.get("stream_timestamps_ns")
            stream_ids = observation.get("stream_ids")
            t_ref = int(observation.get("t_ref_ns", 0))
            if (
                observation.get("schema") != INTEGRATED_CAPTURE_SCHEMA
                or observation_id != f"{episode_dir.name}:observation:{index:06d}"
                or observation_id in observation_by_id
                or t_ref <= previous_t_ref
                or not isinstance(timestamps, Mapping)
                or not isinstance(stream_ids, Mapping)
                or set(timestamps) != required_observation_streams
                or set(stream_ids) != required_observation_streams
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_OBSERVATION_INVALID"
                )
            for name, native_index in native_by_source.items():
                policy_receive_ns = int(timestamps.get(name, 0))
                stream_id = str(stream_ids.get(name, ""))
                if not stream_id.startswith("source:") or "@receive:" not in stream_id:
                    raise ProductionBridgeError(
                        f"BRIDGE_POLICY_EXECUTION_NATIVE_STREAM_ID_INVALID:{name}"
                    )
                source_text, receive_text = stream_id[len("source:") :].split(
                    "@receive:", 1
                )
                try:
                    source_ns = int(source_text)
                    identity_receive_ns = int(receive_text)
                except ValueError as error:
                    raise ProductionBridgeError(
                        f"BRIDGE_POLICY_EXECUTION_NATIVE_STREAM_ID_INVALID:{name}"
                    ) from error
                native = native_index.get(source_ns)
                native_receive_ns = int(
                    0 if native is None else native.get("receive_monotonic_ns", 0)
                )
                if (
                    native is None
                    or policy_receive_ns > t_ref
                    or identity_receive_ns != policy_receive_ns
                    or abs(native_receive_ns - policy_receive_ns)
                    > receive_skew_limits[name]
                ):
                    raise ProductionBridgeError(
                        f"BRIDGE_POLICY_EXECUTION_NATIVE_STREAM_MISSING:{name}"
                    )
                if name == "measured_tcp_pose":
                    _pose_tcp6(native.get("pose", {}))
                elif name == "wrench_notch_sensor":
                    _finite_vector(
                        native.get("force_xyz_n_torque_xyz_nm"),
                        6,
                        "BRIDGE_POLICY_EXECUTION_WRENCH6_INVALID",
                    )
                else:
                    width = float(native.get("width_m", -1.0))
                    if not math.isfinite(width) or not 0.0 <= width <= 0.1:
                        raise ProductionBridgeError(
                            "BRIDGE_POLICY_EXECUTION_STATE7_INVALID"
                        )
            observation_by_id[observation_id] = observation
            previous_t_ref = t_ref
            previous_generation = generation

        request_by_id: dict[str, dict[str, Any]] = {}
        result_by_request: dict[str, dict[str, Any]] = {}
        proposal_by_request: dict[str, dict[str, Any]] = {}
        chunk_by_request: dict[str, dict[str, Any]] = {}
        lineage_fields = (
            "request_id",
            "chunk_id",
            "proposal_id",
            "policy_revision",
            "policy_epoch",
            "reset_generation",
            "takeover_generation",
            "t_ref_ns",
            "request_clock_domain_id",
            "clock_domain_id",
            "request_recorded_monotonic_ns",
        )
        for request, result, proposal, chunk in zip(
            requests, results, proposals, chunks, strict=True
        ):
            generations = {
                self._policy_execution_generation(row, identity)
                for row in (request, result, proposal, chunk)
            }
            request_id = str(request.get("request_id", ""))
            result_id = str(result.get("result_id", ""))
            observation = observation_by_id.get(str(request.get("observation_id", "")))
            actions = proposal.get("actions_absolute7")
            valid_horizon = int(proposal.get("valid_horizon", 0))
            if (
                len(generations) != 1
                or request.get("schema") != POLICY_LINEAGE_SCHEMA
                or result.get("schema") != POLICY_LINEAGE_SCHEMA
                or proposal.get("schema") != INTEGRATED_POLICY_EXECUTION_SCHEMA
                or chunk.get("schema") != INTEGRATED_POLICY_EXECUTION_SCHEMA
                or not request_id
                or request_id in request_by_id
                or result_id != f"policy-result:{request_id}"
                or request.get("chunk_id") != f"live-{request_id}"
                or request.get("proposal_id") != f"policy-proposal:{request_id}"
                or observation is None
                or self._policy_execution_generation(observation, identity)
                != next(iter(generations))
                or int(request.get("t_ref_ns", 0))
                != int(observation.get("t_ref_ns", 0))
                or int(request.get("request_recorded_monotonic_ns", 0))
                < int(observation.get("t_ref_ns", 0))
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_REQUEST_INVALID"
                )
            for field in lineage_fields:
                if (
                    result.get(field) != request.get(field)
                    or proposal.get(field) != result.get(field)
                    or chunk.get(field) != result.get(field)
                ):
                    raise ProductionBridgeError(
                        f"BRIDGE_POLICY_EXECUTION_LINEAGE_MISMATCH:{field}"
                    )
            if (
                result.get("result_id") != result_id
                or proposal.get("result_id") != result_id
                or chunk.get("result_id") != result_id
                or result.get("lineage_schema") != POLICY_LINEAGE_SCHEMA
                or proposal.get("lineage_schema") != POLICY_LINEAGE_SCHEMA
                or chunk.get("lineage_schema") != POLICY_LINEAGE_SCHEMA
                or result.get("policy_execution_candidate") is not True
                or proposal.get("policy_execution_candidate") is not True
                or chunk.get("policy_execution_candidate") is not True
                or result.get("executed") is not False
                or proposal.get("executed") is not False
                or chunk.get("executed") is not False
                or result.get("shadow_proposal") is not False
                or proposal.get("shadow_proposal") is not False
                or chunk.get("shadow_proposal") is not False
                or proposal.get("actual_action_source") != "policy"
                or proposal.get("policy_inference") is not True
                or proposal.get("policy_execution") is not True
                or proposal.get("formal_replay") is not False
                or proposal.get("real_online_r") is not False
                or proposal.get("action_semantics") != "absolute7"
                or chunk.get("action_semantics") != "absolute7"
                or not isinstance(proposal.get("invalidated_by_takeover"), bool)
                or not isinstance(actions, list)
                or valid_horizon <= 0
                or len(actions) != valid_horizon
                or chunk.get("actions_absolute7") != actions
                or int(chunk.get("valid_horizon", 0)) != valid_horizon
                or int(result.get("result_recorded_monotonic_ns", 0))
                < int(request.get("request_recorded_monotonic_ns", 0))
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_PROPOSAL_INVALID"
                )
            for action in actions:
                _finite_vector(
                    action, 7, "BRIDGE_POLICY_EXECUTION_PROPOSAL_ACTION7_INVALID"
                )
            request_by_id[request_id] = request
            result_by_request[request_id] = result
            proposal_by_request[request_id] = proposal
            chunk_by_request[request_id] = chunk

        raw_by_identity = {
            (
                str(item.get("payload", {}).get("source", "")),
                int(item.get("payload", {}).get("sequence", -1)),
            ): item.get("payload", {})
            for item in streams["raw_action"]
        }
        safe_by_identity = {
            (
                str(
                    item.get("payload", {})
                    .get("arbitration", {})
                    .get("raw_action", {})
                    .get("source", "")
                ),
                int(
                    item.get("payload", {})
                    .get("arbitration", {})
                    .get("raw_action", {})
                    .get("sequence", -1)
                ),
            ): item.get("payload", {})
            for item in streams["safe_action"]
        }
        requested_by_identity = {
            (str(item.get("source", "")), int(item.get("sequence", -1))): item
            for item in streams["requested_equilibrium"]
        }
        ack_by_stamp = {
            int(item.get("payload", {}).get("request_stamp_ns", 0)): item
            for item in streams["reference_ack"]
        }
        gripper_by_sequence: dict[int, dict[str, Any]] = {}
        for row in gripper_authorities:
            sequence = int(row.get("sequence", -1))
            if sequence < 0 or sequence in gripper_by_sequence:
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_GRIPPER_AUTHORITY_DUPLICATE"
                )
            gripper_by_sequence[sequence] = row

        transition_sequences: set[int] = set()
        transition_generation_by_sequence: dict[int, tuple[int, int, int]] = {}
        transition_by_chunk: dict[str, list[dict[str, Any]]] = {}
        transition_by_request: dict[str, list[dict[str, Any]]] = {}
        lineage_keys = (
            "request_id",
            "result_id",
            "chunk_id",
            "proposal_id",
            "policy_revision",
            "policy_epoch",
            "reset_generation",
            "takeover_generation",
            "t_ref_ns",
        )
        for transition in transitions:
            generation = self._policy_execution_generation(transition, identity)
            request_id = str(transition.get("request_id", ""))
            request = request_by_id.get(request_id)
            result = result_by_request.get(request_id)
            proposal = proposal_by_request.get(request_id)
            chunk = chunk_by_request.get(request_id)
            selection = transition.get("selection")
            gripper = transition.get("gripper_authority")
            arbitration = transition.get("safety_arbitration")
            if not all(
                isinstance(value, Mapping)
                for value in (request, result, proposal, chunk, selection, gripper, arbitration)
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_TRANSITION_LINEAGE_MISSING"
                )
            sequence = int(selection.get("sequence", -1))
            action_index = int(selection.get("action_index", -1))
            native_safe = safe_by_identity.get(("policy", sequence))
            native_raw = raw_by_identity.get(("policy", sequence))
            native_request = requested_by_identity.get(("policy", sequence))
            stamp = int(
                0
                if native_safe is None
                else native_safe.get("equilibrium_source_stamp_ns", 0)
            )
            native_ack = ack_by_stamp.get(stamp)
            sidecar_gripper = gripper_by_sequence.get(sequence)
            current_observation = observation_by_id.get(
                str(transition.get("current_observation_id", ""))
            )
            next_observation = observation_by_id.get(
                str(transition.get("next_observation_id", ""))
            )
            integrated_pose_ack = transition.get("pose_ack")
            native_pose_ack = (
                native_ack.get("payload", {})
                if isinstance(native_ack, Mapping)
                else {}
            )
            pose_command = transition.get("pose_command")
            native_pose_command = (
                native_request.get("pose", {})
                if isinstance(native_request, Mapping)
                else {}
            )
            pose_ack_fields = (
                "schema",
                "accepted",
                "request_stamp_ns",
                "request_sequence",
                "request_receive_monotonic_ns",
                "ack_monotonic_ns",
                "request_frame_id",
                "accepted_pose",
                "reject_reason",
            )
            pose_ack_identity_valid = (
                isinstance(integrated_pose_ack, Mapping)
                and all(
                    integrated_pose_ack.get(field) == native_pose_ack.get(field)
                    for field in pose_ack_fields
                )
                and integrated_pose_ack.get("accepted") is True
            )
            pose_ack_receive_ns = int(
                integrated_pose_ack.get("upper_receive_monotonic_ns", 0)
                if isinstance(integrated_pose_ack, Mapping)
                else 0
            )
            pose_ack_receive_valid = (
                native_ack is not None
                and pose_ack_receive_ns > 0
                and int(transition.get("receive_monotonic_ns", 0))
                == pose_ack_receive_ns
                and abs(
                    int(native_ack.get("receive_monotonic_ns", 0))
                    - pose_ack_receive_ns
                )
                <= self.config.max_pose_ack_latency_ns
            )
            pose_command_valid = (
                isinstance(pose_command, Mapping)
                and isinstance(native_pose_command, Mapping)
                and pose_command.get("position_m")
                == native_pose_command.get("position_m")
                and pose_command.get("quaternion_xyzw")
                == native_pose_command.get("quaternion_xyzw")
                and native_pose_command.get("frame_id")
                == native_pose_ack.get("request_frame_id")
            )
            if not pose_ack_identity_valid:
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_POSE_ACK_IDENTITY_INVALID"
                )
            if not pose_ack_receive_valid:
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_POSE_ACK_RECEIVE_INVALID"
                )
            if not pose_command_valid:
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_POSE_COMMAND_INVALID"
                )
            if (
                transition.get("schema") != INTEGRATED_POLICY_EXECUTION_SCHEMA
                or transition.get("actual_action_source") != "policy"
                or transition.get("executed_action_source") != "policy"
                or transition.get("policy_executed_transition") is not True
                or transition.get("intervention") is not False
                or transition.get("formal_replay") is not False
                or transition.get("real_online_r") is not False
                or sequence < 0
                or sequence in transition_sequences
                or action_index < 0
                or action_index >= int(proposal.get("valid_horizon", 0))
                or self._policy_execution_generation(request, identity) != generation
                or current_observation is None
                or next_observation is None
                or transition.get("observation_id")
                != transition.get("current_observation_id")
                or int(next_observation.get("t_ref_ns", 0))
                <= int(current_observation.get("t_ref_ns", 0))
                or native_safe is None
                or native_raw is None
                or native_request is None
                or native_ack is None
                or arbitration != native_safe.get("arbitration")
                or arbitration.get("accepted") is not True
                or arbitration.get("reason") != "accepted_policy"
                or arbitration.get("raw_action") != native_raw
                or native_safe.get("equilibrium_published") is not True
                or transition.get("ack_id") != f"policy-ack:{sequence}:{stamp}"
                or sidecar_gripper is None
                or gripper != sidecar_gripper
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_ACTION_ACK_INVALID"
                )
            for field in lineage_keys:
                expected = (
                    result.get(field)
                    if field == "result_id"
                    else request.get(field)
                )
                if (
                    transition.get(field) != expected
                    or selection.get(field) != expected
                    or gripper.get(field) != expected
                ):
                    raise ProductionBridgeError(
                        f"BRIDGE_POLICY_EXECUTION_TRANSITION_LINEAGE_MISMATCH:{field}"
                    )
            selected = proposal["actions_absolute7"][action_index]
            if (
                selection.get("selected_post_adapter_absolute7") != selected
                or transition.get("accepted_absolute7") != selected
                or int(gripper.get("action_index", -1)) != action_index
                or gripper.get("requested_state") not in {"OPEN", "CLOSED"}
                or float(gripper.get("requested_width_m", -1.0))
                not in {self.config.gripper_closed_width_m, self.config.gripper_open_width_m}
                or gripper.get("authority")
                not in {"existing_accepted_gripper_state", "policy_execution_backend"}
                or bool(gripper.get("command_required"))
                != (gripper.get("authority") == "policy_execution_backend")
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_ACTION7_AUTHORITY_INVALID"
                )
            _finite_vector(
                transition.get("accepted_absolute7"),
                7,
                "BRIDGE_POLICY_EXECUTION_ACCEPTED_ACTION7_INVALID",
            )
            transition_sequences.add(sequence)
            transition_generation_by_sequence[sequence] = generation
            transition_by_chunk.setdefault(str(transition["chunk_id"]), []).append(
                transition
            )
            transition_by_request.setdefault(request_id, []).append(transition)

        accepted_policy_sequences = {
            sequence
            for (source, sequence), safe in safe_by_identity.items()
            if source == "policy"
            and safe.get("arbitration", {}).get("accepted") is True
            and safe.get("equilibrium_published") is True
        }
        if transition_sequences != accepted_policy_sequences:
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_ACTION_ACK_COVERAGE_MISMATCH"
            )

        override_sequences: set[int] = set()
        for (source, sequence), safe in safe_by_identity.items():
            arbitration = safe.get("arbitration", {})
            if source != "policy" or arbitration.get("accepted") is not False:
                continue
            if (
                arbitration.get("reason") != "human_override"
                or safe.get("reject_reason") != "human_override"
                or safe.get("equilibrium_published") is not False
                or safe.get("equilibrium_source_stamp_ns") is not None
                or (source, sequence) in requested_by_identity
                or sequence in transition_sequences
                or sequence in gripper_by_sequence
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_REJECTED_ACTION_INVALID"
                )
            override_sequences.add(sequence)

        active_intervention: dict[str, Any] | None = None
        completed_takeovers: list[tuple[dict[str, Any], dict[str, Any]]] = []
        previous_intervention_ns = 0
        previous_takeover_generation = int(identity.get("takeover_generation", 0))
        for intervention in interventions:
            generation = self._policy_execution_generation(intervention, identity)
            event = str(intervention.get("event", ""))
            receive_ns = int(intervention.get("receive_monotonic_ns", 0))
            safe = intervention.get("safe_action")
            raw = (
                safe.get("arbitration", {}).get("raw_action", {})
                if isinstance(safe, Mapping)
                else {}
            )
            if (
                intervention.get("schema") != INTEGRATED_CAPTURE_SCHEMA
                or intervention.get("actual_action_source") != "human"
                or intervention.get("policy_execution") is not True
                or event not in {"intervention_start", "human_action", "intervention_end"}
                or receive_ns <= previous_intervention_ns
                or not isinstance(safe, Mapping)
                or raw.get("source") != "human"
                or safe_by_identity.get(("human", int(raw.get("sequence", -1))))
                != safe
                or int(safe.get("arbitration", {}).get("policy_epoch", -1))
                != generation[0]
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_INTERVENTION_INVALID"
                )
            if event == "intervention_start":
                invalidated_chunk = str(intervention.get("invalidated_chunk_id", ""))
                invalidated = next(
                    (
                        chunk
                        for chunk in chunks
                        if chunk.get("chunk_id") == invalidated_chunk
                    ),
                    None,
                )
                if (
                    active_intervention is not None
                    or generation[2] != previous_takeover_generation + 1
                    or intervention.get("old_policy_chunk_invalidated") is not True
                    or invalidated is None
                    or self._policy_execution_generation(invalidated, identity)[2]
                    >= generation[2]
                    or any(
                        int(row.get("receive_monotonic_ns", 0)) >= receive_ns
                        for row in transition_by_chunk.get(invalidated_chunk, ())
                    )
                ):
                    raise ProductionBridgeError(
                        "BRIDGE_POLICY_EXECUTION_TAKEOVER_INVALIDATION_INVALID"
                    )
                active_intervention = intervention
            elif event == "human_action":
                if (
                    active_intervention is None
                    or generation
                    != self._policy_execution_generation(active_intervention, identity)
                ):
                    raise ProductionBridgeError(
                        "BRIDGE_POLICY_EXECUTION_TAKEOVER_GENERATION_INVALID"
                    )
            else:
                if (
                    active_intervention is None
                    or generation
                    != self._policy_execution_generation(active_intervention, identity)
                    or intervention.get("old_policy_chunk_invalidated") is not False
                    or intervention.get("invalidated_chunk_id") is not None
                ):
                    raise ProductionBridgeError(
                        "BRIDGE_POLICY_EXECUTION_TAKEOVER_END_INVALID"
                    )
                completed_takeovers.append((active_intervention, intervention))
                previous_takeover_generation = generation[2]
                active_intervention = None
            previous_intervention_ns = receive_ns
        if active_intervention is not None:
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_TAKEOVER_UNSEALED"
            )
        for start, end in completed_takeovers:
            start_ns = int(start["receive_monotonic_ns"])
            end_ns = int(end["receive_monotonic_ns"])
            generation = self._policy_execution_generation(end, identity)
            if any(
                start_ns <= int(row.get("receive_monotonic_ns", 0)) <= end_ns
                for row in transitions
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_ACTION_DURING_TAKEOVER"
                )
            following = [
                row
                for row in transitions
                if int(row.get("receive_monotonic_ns", 0)) > end_ns
            ]
            following_generations = [
                self._policy_execution_generation(row, identity)
                for row in following
            ]
            if (
                following
                and (
                    following_generations[0] != generation
                    or any(item < generation for item in following_generations)
                )
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_POST_TAKEOVER_GENERATION_INVALID"
                )
        for proposal in proposals:
            if proposal.get("invalidated_by_takeover") is True and transition_by_request.get(
                str(proposal.get("request_id", ""))
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_INVALIDATED_PROPOSAL_EXECUTED"
                )

        lease_payload = _read_json(
            stream_root / "policy_execute_initial_gripper_lease.json"
        )
        try:
            lease = InitialGripperAuthority.from_mapping(lease_payload).validate(
                max_feedback_age_ns=self.config.max_gripper_feedback_age_ns
            )
        except GripperProvenanceError as error:
            raise ProductionBridgeError(
                f"BRIDGE_POLICY_EXECUTION_INITIAL_GRIPPER_LEASE_INVALID:{error}"
            ) from error
        lease_generation = lease.generation
        if (
            lease.episode_id != identity.get("episode_id")
            or lease_generation.policy_revision != identity.get("policy_revision")
            or lease_generation.policy_epoch != identity.get("policy_epoch")
            or lease_generation.reset_generation != identity.get("reset_generation")
            or lease_generation.takeover_generation
            != identity.get("takeover_generation")
        ):
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_INITIAL_GRIPPER_LEASE_IDENTITY_MISMATCH"
            )

        targets = {
            int(row.get("local_goal_sequence", -1)): row
            for row in streams["gripper_target"]
        }
        statuses = {
            int(row.get("local_goal_sequence", -1)): row
            for row in streams["gripper_goal_status"]
        }
        if set(targets) != set(statuses):
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_GRIPPER_TERMINAL_COVERAGE_MISMATCH"
            )
        accepted_gripper_states = [
            {
                "terminal_finished_monotonic_ns": lease.terminal_finished_monotonic_ns,
                "requested_state": lease.requested_state,
                "requested_width_m": lease.requested_width_m,
                "origin_kind": "initial_gripper_lease",
                "origin_local_goal_sequence": lease.origin_local_goal_sequence,
                "origin_action_goal_id": lease.origin_action_goal_id,
                "terminal_outcome": lease.terminal_outcome,
                "generation": {
                    "session_id": identity["session_id"],
                    "clock_domain_id": identity["clock_domain_id"],
                    **asdict(lease_generation),
                },
            }
        ]
        stalled_contact_count = 0
        quality = (
            native_result.get("native_stream_quality", {})
            .get("gripper_state", {})
            .get("goal_aware_validation", {})
        )
        quality_goals = {
            int(row.get("local_goal_sequence", -1)): row
            for row in quality.get("goals", ())
        }
        for sequence, target in targets.items():
            status = statuses[sequence]
            outcome = str(status.get("outcome", ""))
            requested_state = str(target.get("requested_state", ""))
            requested_width = float(target.get("target_width_m", -1.0))
            if (
                sequence < 0
                or target.get("action_goal_id") != status.get("action_goal_id")
                or outcome not in VALID_TERMINAL_OUTCOMES
                or requested_state not in {"OPEN", "CLOSED"}
                or not 0.0 <= requested_width <= 0.1
                or int(target.get("accepted_monotonic_ns", 0))
                != int(status.get("accepted_monotonic_ns", -1))
                or int(status.get("finished_monotonic_ns", 0))
                < int(status.get("accepted_monotonic_ns", 0))
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_GRIPPER_TERMINAL_INVALID"
                )
            accepted_gripper_states.append(
                {
                    "terminal_finished_monotonic_ns": int(
                        status["finished_monotonic_ns"]
                    ),
                    "requested_state": requested_state,
                    "requested_width_m": requested_width,
                    "origin_kind": "native_gripper_terminal",
                    "origin_local_goal_sequence": sequence,
                    "origin_action_goal_id": target["action_goal_id"],
                    "terminal_outcome": outcome,
                    "generation": None,
                }
            )
            if requested_state == "CLOSED" and outcome == "stalled":
                diagnostic = quality_goals.get(sequence)
                if (
                    not isinstance(diagnostic, Mapping)
                    or diagnostic.get("requested_state") != "CLOSED"
                    or diagnostic.get("outcome") != "stalled"
                    or diagnostic.get("direction") != "closing"
                    or not math.isfinite(float(diagnostic.get("settled_width_m", math.nan)))
                ):
                    raise ProductionBridgeError(
                        "BRIDGE_POLICY_EXECUTION_STALLED_CONTACT_PROVENANCE_INVALID"
                    )
                stalled_contact_count += 1
        if quality.get("violations") not in (None, []):
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_GRIPPER_QUALITY_INVALID"
            )
        command_origins: dict[tuple[int, str], dict[str, Any]] = {}
        for authority in gripper_authorities:
            if authority.get("command_required") is not True:
                continue
            sequence = int(authority.get("sequence", -1))
            generation = self._policy_execution_generation(
                {**identity, **authority}, identity
            )
            if transition_generation_by_sequence.get(sequence) != generation:
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_GRIPPER_GENERATION_MISMATCH"
                )
            local_sequence = int(authority.get("local_goal_sequence", -1))
            target = targets.get(local_sequence)
            status = statuses.get(local_sequence)
            key = (local_sequence, str(authority.get("action_goal_id", "")))
            if (
                target is None
                or status is None
                or key in command_origins
                or key[1] != target.get("action_goal_id")
                or authority.get("outcome") != status.get("outcome")
                or authority.get("outcome") not in VALID_TERMINAL_OUTCOMES
                or int(authority.get("accepted_monotonic_ns", 0))
                != int(target.get("accepted_monotonic_ns", -1))
                or int(authority.get("finished_monotonic_ns", 0))
                != int(status.get("finished_monotonic_ns", -1))
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_POLICY_GRIPPER_ACK_INVALID"
                )
            origin = next(
                (
                    item
                    for item in accepted_gripper_states
                    if item["origin_local_goal_sequence"] == local_sequence
                    and item["origin_action_goal_id"] == key[1]
                ),
                None,
            )
            if origin is None:
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_POLICY_GRIPPER_ACK_INVALID"
                )
            command_origin = dict(origin)
            command_origin["generation"] = {
                **{
                    field: identity[field]
                    for field in (
                        "session_id",
                        "episode_id",
                        "clock_domain_id",
                        "policy_revision",
                    )
                },
                "policy_epoch": generation[0],
                "reset_generation": generation[1],
                "takeover_generation": generation[2],
            }
            command_origins[key] = command_origin
        accepted_gripper_states.extend(command_origins.values())

        gripper_origin_by_sequence: dict[int, dict[str, Any]] = {}
        for authority in gripper_authorities:
            sequence = int(authority.get("sequence", -1))
            generation = self._policy_execution_generation(
                {**identity, **authority}, identity
            )
            if transition_generation_by_sequence.get(sequence) != generation:
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_GRIPPER_GENERATION_MISMATCH"
                )
            if authority.get("command_required") is True:
                origin = command_origins.get(
                    (
                        int(authority.get("local_goal_sequence", -1)),
                        str(authority.get("action_goal_id", "")),
                    )
                )
                if origin is None:
                    raise ProductionBridgeError(
                        "BRIDGE_POLICY_EXECUTION_POLICY_GRIPPER_ACK_INVALID"
                    )
                gripper_origin_by_sequence[sequence] = dict(origin)
                continue
            feedback_ns = int(authority.get("feedback_monotonic_ns", 0))
            applicable = [
                state
                for state in accepted_gripper_states
                if int(state["terminal_finished_monotonic_ns"]) <= feedback_ns
                and isinstance(state.get("generation"), Mapping)
                and self._policy_execution_generation(state["generation"], identity)
                == generation
            ]
            if not applicable:
                continue
            origin = max(
                applicable,
                key=lambda item: int(item["terminal_finished_monotonic_ns"]),
            )
            state = str(origin["requested_state"])
            width = float(origin["requested_width_m"])
            if authority.get("command_required") is False and (
                authority.get("requested_state") != state
                or float(authority.get("requested_width_m", -1.0)) != width
                or not 0.0 <= float(authority.get("feedback_width_m", -1.0)) <= 0.1
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_GRIPPER_ORIGIN_INVALID"
                )
            gripper_origin_by_sequence[sequence] = dict(origin)

        accepted_transitions = tuple(
            transition
            for transition in transitions
            if int(transition.get("selection", {}).get("sequence", -1))
            in gripper_origin_by_sequence
        )

        reconciliation = _read_json(
            stream_root / "policy_execute_camera_reconciliation.json"
        )
        camera_records = reconciliation.get("records")
        if (
            reconciliation.get("schema") != INTEGRATED_POLICY_EXECUTION_SCHEMA
            or Path(str(reconciliation.get("native_episode", ""))).resolve()
            != episode_dir.resolve()
            or not isinstance(camera_records, list)
            or len(camera_records) != 2 * len(observations)
        ):
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_CAMERA_RECONCILIATION_INVALID"
            )
        native_cameras = {
            role: {
                str(item.get("rgb_path", "")): item
                for item in streams[f"{role}_camera"]
            }
            for role in ("external", "wrist")
        }
        reconciled: set[tuple[str, str]] = set()
        for record in camera_records:
            observation = observation_by_id.get(str(record.get("observation_id", "")))
            role = str(record.get("role", ""))
            key = (str(record.get("observation_id", "")), role)
            stream_name = f"{role}_camera"
            path = str(record.get("rgb_path", ""))
            native = native_cameras.get(role, {}).get(path)
            if (
                observation is None
                or role not in {"external", "wrist"}
                or key in reconciled
                or record.get("clock_domain_id") != UPPER_CLOCK
                or record.get("same_recorder_jpeg") is not True
                or observation["stream_ids"].get(stream_name) != path
                or int(observation["stream_timestamps_ns"].get(stream_name, 0))
                != int(record.get("policy_receive_monotonic_ns", 0))
                or native is None
                or int(native.get("receive_monotonic_ns", 0))
                != int(record.get("native_receive_monotonic_ns", 0))
                or not (episode_dir / path).is_file()
            ):
                raise ProductionBridgeError(
                    "BRIDGE_POLICY_EXECUTION_CAMERA_BINDING_INVALID"
                )
            reconciled.add(key)

        seal = _read_json(stream_root / "policy_execute_episode_seal.json")
        learner_critic_steps = int(seal.get("learner_critic_steps", -1))
        learner_actor_steps = int(seal.get("learner_actor_steps", -1))
        sync_learner_seal_valid = (
            not async_learner
            and seal.get("learner_started") is False
            and int(seal.get("actor_updates", -1)) == 0
            and int(seal.get("critic_updates", -1)) == 0
        )
        async_learner_seal_valid = (
            async_learner
            and seal.get("learner_started") is True
            and seal.get("learner_resume_checkpoint") == async_metadata[0]
            and seal.get("active_actor_revision") == async_metadata[1]
            and seal.get("active_actor_model_revision")
            == identity.get("policy_revision")
            and seal.get("active_actor_model_revision")
            == manifest.get("policy_metadata", {}).get("model_sha256")
            and learner_critic_steps == 2
            and learner_actor_steps == 1
            and int(seal.get("critic_updates", -1)) == learner_critic_steps
            and int(seal.get("actor_updates", -1)) == learner_actor_steps
            and seal.get("current_episode_sampled_by_learner") is False
            and isinstance(seal.get("pending_checkpoint_path"), str)
            and bool(seal.get("pending_checkpoint_path"))
            and seal.get("pending_candidate_id") == async_metadata[2]
            and seal.get("pending_candidate_published") is False
            and seal.get("pending_candidate_activated") is False
        )
        latest_lineage_ns = max(
            int(observations[-1].get("t_ref_ns", 0)),
            max(int(row.get("result_recorded_monotonic_ns", 0)) for row in results),
            max(int(row.get("receive_monotonic_ns", 0)) for row in transitions),
            max(
                (int(row.get("receive_monotonic_ns", 0)) for row in interventions),
                default=0,
            ),
        )
        if (
            seal.get("schema") != INTEGRATED_CAPTURE_SCHEMA
            or seal.get("backend_schema") != INTEGRATED_POLICY_EXECUTION_SCHEMA
            or seal.get("technical_seal") != "complete"
            or seal.get("actual_action_source") != "policy"
            or seal.get("executed_action_source") != "policy"
            or seal.get("policy_inference") is not True
            or seal.get("policy_execution") is not True
            or seal.get("formal_replay") is not False
            or seal.get("real_online_r") is not False
            or seal.get("formal_training_replay_written") is not False
            or seal.get("policy_revision_published") is not False
            or seal.get("checkpoint_written") is not False
            or not (sync_learner_seal_valid or async_learner_seal_valid)
            or seal.get("controller_owner") != "recorder"
            or seal.get("controller_process_count") != 1
            or seal.get("deploy_controller_started") is not False
            or seal.get("policy_action_publisher_created") is not True
            or Path(str(seal.get("native_episode", ""))).resolve()
            != episode_dir.resolve()
            or seal.get("native_episode_result") != native_result
            or seal.get("initial_gripper_lease") != lease_payload
            or seal.get("terminal_observation_id")
            != observations[-1].get("observation_id")
            or int(seal.get("observation_count", -1)) != len(observations)
            or int(seal.get("policy_request_count", -1)) != len(requests)
            or int(seal.get("policy_result_count", -1)) != len(results)
            or int(seal.get("policy_chunk_count", -1)) != len(chunks)
            or int(seal.get("policy_action_ack_count", -1)) != len(transitions)
            or int(seal.get("human_action_ack_count", -1)) != 0
            or int(seal.get("intervention_count", -1)) != len(interventions)
            or int(seal.get("camera_records_reconciled", -1))
            != len(camera_records)
            or int(seal.get("sealed_monotonic_ns", 0)) < latest_lineage_ns
            or int(seal.get("sealed_monotonic_ns", 0))
            < int(native_result.get("finished_monotonic_ns", 0))
        ):
            raise ProductionBridgeError(
                "BRIDGE_INTEGRATED_POLICY_EXECUTION_SEAL_INVALID"
            )
        self._policy_execution_generation(seal, identity)

        approval = _read_json(DEFAULT_REWARD_CONTRACT).get("reward_gate", {}).get(
            "development_policy_execution_smoke", {}
        )
        basis = approval.get("basis") if isinstance(approval, Mapping) else None
        if (
            not isinstance(approval, Mapping)
            or approval.get("approved") is not True
            or approval.get("scope") != seal.get("detector_approval_scope")
            or not isinstance(basis, Mapping)
            or basis.get("technical_seal") != "complete"
            or basis.get("operator_task_outcome") != "success"
            or basis.get("detector_outcome") != "success"
            or basis.get("executed_action_source") != "human"
            or basis.get("policy_execution") is not False
            or basis.get("formal_replay") is not False
            or basis.get("real_online_r") is not False
        ):
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_DETECTOR_SCOPE_NOT_APPROVED"
            )

        try:
            prepared = _prepare_native_episode(episode_dir)
        except Exception as error:
            raise ProductionBridgeError(
                f"BRIDGE_POLICY_EXECUTION_OBSERVATION_MATERIALIZATION_FAILED:"
                f"{type(error).__name__}:{error}"
            ) from error
        if (
            prepared.raw_episode_id != episode_dir.name
            or prepared.task != str(native_result.get("task", ""))
            or prepared.state7.ndim != 2
            or prepared.state7.shape[1] != 7
            or prepared.wrench6.ndim != 2
            or prepared.wrench6.shape[1] != 6
            or len(prepared.state7) != len(prepared.wrench6)
            or len(prepared.camera1_paths) != len(prepared.state7)
            or len(prepared.camera2_paths) != len(prepared.state7)
            or not np.all(np.isfinite(prepared.state7))
            or not np.all(np.isfinite(prepared.wrench6))
        ):
            raise ProductionBridgeError(
                "BRIDGE_POLICY_EXECUTION_MATERIALIZED_OBSERVATION_INVALID"
            )

        return {
            "classification": POLICY_EXECUTION_SMOKE_CLASSIFICATION,
            "initial_gripper_lease": lease,
            "prepared": prepared,
            "observations": observation_by_id,
            "requests": request_by_id,
            "results": result_by_request,
            "proposals": proposal_by_request,
            "chunks": chunk_by_request,
            "transitions": accepted_transitions,
            "gripper_origins": gripper_origin_by_sequence,
            "seal": seal,
            "summary": {
                "schema": INTEGRATED_POLICY_EXECUTION_SCHEMA,
                "session_id": identity["session_id"],
                "episode_id": identity["episode_id"],
                "policy_revision": identity["policy_revision"],
                "observation_count": len(observations),
                "policy_request_count": len(requests),
                "policy_result_count": len(results),
                "policy_proposal_count": len(proposals),
                "policy_chunk_count": len(chunks),
                "policy_action_ack_count": len(transitions),
                "human_override_count": len(override_sequences),
                "human_override_executed_count": 0,
                "intervention_count": len(interventions),
                "stalled_contact_count": stalled_contact_count,
                "camera_reconciliation_count": len(camera_records),
                "terminal_observation_id": observations[-1]["observation_id"],
                "actual_action_source": "policy",
                "executed_action_source": "policy",
                "policy_execution": True,
                "formal_replay": False,
                "real_online_r": False,
                "training_replay_eligible": False,
                "policy_lineage_complete": True,
                "action7_authority_complete": True,
                "state7_complete": True,
                "calibrated_tcp_wrench6_complete": True,
                "camera_models": ["D435", "D405"],
                "technical_seal": "complete",
                "operator_task_outcome": operator_task_outcome,
                "detector_outcome": "success",
                "detector_outcome_source": (
                    "approved_development_policy_execution_smoke_scope"
                ),
                "model_update_count": 0,
            },
        }

    def _load_integrated_capture(
        self,
        *,
        episode_dir: Path,
        native_result: Mapping[str, Any],
        streams: Mapping[str, list[dict[str, Any]]],
        operator_task_outcome: str | None,
    ) -> dict[str, Any] | None:
        manifest_path = episode_dir.parent.parent / "integrated_capture_session.json"
        stream_root = (
            episode_dir.parent.parent
            / "integrated_capture"
            / episode_dir.name
            / "streams"
        )
        if not manifest_path.exists() and not stream_root.exists():
            return self._load_integrated_shadow(
                episode_dir=episode_dir,
                native_result=native_result,
                streams=streams,
                operator_task_outcome=operator_task_outcome,
            )
        if not manifest_path.is_file() or not stream_root.is_dir():
            raise ProductionBridgeError("BRIDGE_INTEGRATED_CAPTURE_INCOMPLETE")
        if _read_json(manifest_path).get("schema") == INTEGRATED_POLICY_EXECUTION_SCHEMA:
            return self._load_integrated_policy_execution(
                episode_dir=episode_dir,
                native_result=native_result,
                streams=streams,
                operator_task_outcome=operator_task_outcome,
            )
        return self._load_integrated_shadow(
            episode_dir=episode_dir,
            native_result=native_result,
            streams=streams,
            operator_task_outcome=operator_task_outcome,
        )

    def _load_integrated_shadow(
        self,
        *,
        episode_dir: Path,
        native_result: Mapping[str, Any],
        streams: Mapping[str, list[dict[str, Any]]],
        operator_task_outcome: str | None,
    ) -> dict[str, Any] | None:
        dataset_root = episode_dir.parent.parent
        manifest_path = dataset_root / "integrated_capture_session.json"
        shadow_root = dataset_root / "integrated_capture" / episode_dir.name / "streams"
        if not manifest_path.exists() and not shadow_root.exists():
            if operator_task_outcome is not None:
                raise ProductionBridgeError(
                    "BRIDGE_OPERATOR_OUTCOME_WITHOUT_INTEGRATED_SHADOW"
                )
            return None
        if not manifest_path.is_file() or not shadow_root.is_dir():
            raise ProductionBridgeError("BRIDGE_INTEGRATED_SHADOW_INCOMPLETE")
        if operator_task_outcome not in {"success", "failure"}:
            raise ProductionBridgeError("BRIDGE_OPERATOR_TASK_OUTCOME_REQUIRED")

        manifest = _read_json(manifest_path)
        contract = manifest.get("contract")
        identity = contract.get("identity") if isinstance(contract, Mapping) else None
        if (
            manifest.get("schema") != INTEGRATED_SHADOW_SCHEMA
            or not isinstance(contract, Mapping)
            or contract.get("schema") != INTEGRATED_CAPTURE_SCHEMA
            or not isinstance(identity, Mapping)
            or identity.get("episode_id") != episode_dir.name
            or identity.get("clock_domain_id") != UPPER_CLOCK
            or contract.get("mode") != "shadow"
            or contract.get("actual_action_source") != "human"
            or contract.get("policy_inference") is not True
            or contract.get("policy_execution") is not False
            or contract.get("formal_replay") is not False
            or contract.get("real_online_r") is not False
            or contract.get("controller_owner") != "recorder"
            or contract.get("controller_process_count") != 1
            or contract.get("recorder_controller") is not True
            or contract.get("deploy_controller") is not False
            or manifest.get("controller_owner") != "recorder"
            or manifest.get("controller_process_count") != 1
            or manifest.get("deploy_controller_started") is not False
            or manifest.get("policy_action_publisher_created") is not False
        ):
            raise ProductionBridgeError("BRIDGE_INTEGRATED_SHADOW_CONTRACT_INVALID")
        clock = manifest.get("clock_binding")
        if (
            not isinstance(clock, Mapping)
            or clock.get("stage3_clock_domain_id") != UPPER_CLOCK
            or clock.get("policy_request_clock_domain_id")
            != "upper_host_monotonic_ns"
            or clock.get("native_primary_alignment_clock")
            != "upper_host_receive_monotonic_ns"
            or clock.get("same_upper_host_monotonic_epoch") is not True
        ):
            raise ProductionBridgeError("BRIDGE_INTEGRATED_SHADOW_CLOCK_INVALID")
        metadata = manifest.get("policy_metadata")
        session = _read_json(dataset_root / "session.json")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("model_sha256") != identity.get("policy_revision")
            or not str(metadata.get("dataset_repo_id", ""))
            or metadata.get("tool_profile_sha256")
            != session.get("tool_config_hash")
            or not str(metadata.get("calibration_id", ""))
        ):
            raise ProductionBridgeError("BRIDGE_INTEGRATED_SHADOW_POLICY_BINDING_INVALID")

        names = {
            "observations": "policy_shadow_observation.jsonl",
            "requests": "policy_shadow_request.jsonl",
            "results": "policy_shadow_result.jsonl",
            "proposals": "policy_shadow_proposal.jsonl",
            "human_acks": "policy_shadow_human_ack.jsonl",
        }
        rows = {
            key: _read_jsonl(shadow_root / name)
            for key, name in names.items()
            if (shadow_root / name).is_file()
        }
        if set(rows) != set(names):
            raise ProductionBridgeError("BRIDGE_INTEGRATED_SHADOW_STREAM_MISSING")
        observations = rows["observations"]
        requests = rows["requests"]
        results = rows["results"]
        proposals = rows["proposals"]
        human_acks = rows["human_acks"]
        if not observations or not (
            len(observations) == len(requests) == len(results) == len(proposals)
        ):
            raise ProductionBridgeError("BRIDGE_SHADOW_POLICY_LINEAGE_COUNT_MISMATCH")

        native_by_source: dict[str, dict[int, dict[str, Any]]] = {}
        for name in ("measured_tcp_pose", "wrench_notch_sensor", "gripper_state"):
            native_by_source[name] = {
                int(item.get("source_stamp_ns", 0)): item
                for item in streams[name]
            }
        observation_by_id: dict[str, dict[str, Any]] = {}
        previous_t_ref = 0
        required_streams = set(native_by_source) | {"external_camera", "wrist_camera"}
        receive_skew_limits = {
            "measured_tcp_pose": self.config.max_pose_age_ns,
            "wrench_notch_sensor": self.config.max_wrench_age_ns,
            "gripper_state": self.config.max_gripper_feedback_age_ns,
        }
        for index, observation in enumerate(observations):
            self._validate_shadow_identity(observation, identity)
            observation_id = str(observation.get("observation_id", ""))
            if (
                observation.get("schema") != INTEGRATED_CAPTURE_SCHEMA
                or observation_id
                != f"{episode_dir.name}:observation:{index:06d}"
                or observation_id in observation_by_id
            ):
                raise ProductionBridgeError("BRIDGE_SHADOW_OBSERVATION_IDENTITY_INVALID")
            t_ref = int(observation.get("t_ref_ns", 0))
            timestamps = observation.get("stream_timestamps_ns")
            stream_ids = observation.get("stream_ids")
            if (
                t_ref <= previous_t_ref
                or not isinstance(timestamps, Mapping)
                or not isinstance(stream_ids, Mapping)
                or set(timestamps) != required_streams
                or set(stream_ids) != required_streams
            ):
                raise ProductionBridgeError("BRIDGE_SHADOW_OBSERVATION_STREAM_INVALID")
            for name, native_index in native_by_source.items():
                policy_receive_ns = int(timestamps.get(name, 0))
                stream_id = str(stream_ids.get(name, ""))
                prefix = "source:"
                separator = "@receive:"
                if not stream_id.startswith(prefix) or separator not in stream_id:
                    raise ProductionBridgeError(
                        f"BRIDGE_SHADOW_NATIVE_STREAM_ID_MISMATCH:{name}"
                    )
                source_text, receive_text = stream_id[len(prefix) :].split(
                    separator, 1
                )
                try:
                    source_ns = int(source_text)
                    identity_receive_ns = int(receive_text)
                except ValueError as error:
                    raise ProductionBridgeError(
                        f"BRIDGE_SHADOW_NATIVE_STREAM_ID_MISMATCH:{name}"
                    ) from error
                native = native_index.get(source_ns)
                native_receive_ns = int(
                    0 if native is None else native.get("receive_monotonic_ns", 0)
                )
                if (
                    native is None
                    or policy_receive_ns > t_ref
                    or native_receive_ns > t_ref
                    or identity_receive_ns != policy_receive_ns
                    or abs(native_receive_ns - policy_receive_ns)
                    > receive_skew_limits[name]
                ):
                    raise ProductionBridgeError(
                        f"BRIDGE_SHADOW_NATIVE_STREAM_MISSING:{name}"
                    )
                if name == "measured_tcp_pose":
                    _pose_tcp6(native.get("pose", {}))
                elif name == "wrench_notch_sensor":
                    _finite_vector(
                        native.get("force_xyz_n_torque_xyz_nm"),
                        6,
                        "BRIDGE_SHADOW_WRENCH_STREAM_INVALID",
                    )
                else:
                    width = float(native.get("width_m", -1.0))
                    if not math.isfinite(width) or not 0.0 <= width <= 0.1:
                        raise ProductionBridgeError("BRIDGE_SHADOW_STATE_STREAM_INVALID")
            observation_by_id[observation_id] = observation
            previous_t_ref = t_ref

        lineage_fields = (
            "request_id",
            "chunk_id",
            "proposal_id",
            "policy_revision",
            "policy_epoch",
            "reset_generation",
            "takeover_generation",
            "t_ref_ns",
            "request_clock_domain_id",
            "clock_domain_id",
            "request_recorded_monotonic_ns",
        )
        request_ids: set[str] = set()
        result_ids: set[str] = set()
        for observation, request, result, proposal in zip(
            observations, requests, results, proposals, strict=True
        ):
            for row in (request, result, proposal):
                self._validate_shadow_identity(row, identity)
            request_id = str(request.get("request_id", ""))
            result_id = str(result.get("result_id", ""))
            if (
                request.get("schema") != POLICY_LINEAGE_SCHEMA
                or result.get("schema") != POLICY_LINEAGE_SCHEMA
                or proposal.get("schema") != INTEGRATED_SHADOW_SCHEMA
                or not request_id
                or request_id in request_ids
                or result_id != f"policy-result:{request_id}"
                or result_id in result_ids
                or request.get("observation_id") != observation.get("observation_id")
                or int(request.get("t_ref_ns", 0)) != int(observation.get("t_ref_ns", 0))
                or int(request.get("request_recorded_monotonic_ns", 0))
                < int(observation.get("t_ref_ns", 0))
            ):
                raise ProductionBridgeError("BRIDGE_SHADOW_POLICY_REQUEST_INVALID")
            for field in lineage_fields:
                if result.get(field) != request.get(field) or proposal.get(field) != result.get(field):
                    raise ProductionBridgeError(
                        f"BRIDGE_SHADOW_POLICY_LINEAGE_MISMATCH:{field}"
                    )
            if proposal.get("result_id") != result_id:
                raise ProductionBridgeError("BRIDGE_SHADOW_POLICY_RESULT_ID_MISMATCH")
            result_recorded_ns = int(result.get("result_recorded_monotonic_ns", 0))
            actions = proposal.get("actions_absolute7")
            valid_horizon = int(proposal.get("valid_horizon", 0))
            if (
                result.get("lineage_schema") != POLICY_LINEAGE_SCHEMA
                or result.get("shadow_proposal") is not True
                or result.get("executed") is not False
                or result_recorded_ns < int(request["request_recorded_monotonic_ns"])
                or proposal.get("actual_action_source") != "human"
                or proposal.get("policy_inference") is not True
                or proposal.get("policy_execution") is not False
                or proposal.get("shadow_proposal") is not True
                or proposal.get("executed") is not False
                or proposal.get("formal_replay") is not False
                or proposal.get("real_online_r") is not False
                or proposal.get("action_semantics") != "absolute7"
                or not isinstance(actions, list)
                or valid_horizon <= 0
                or len(actions) != valid_horizon
            ):
                raise ProductionBridgeError("BRIDGE_SHADOW_POLICY_PROPOSAL_INVALID")
            for action in actions:
                _finite_vector(action, 7, "BRIDGE_SHADOW_POLICY_ACTION_INVALID")
            request_ids.add(request_id)
            result_ids.add(result_id)

        native_acks = {
            int(item.get("payload", {}).get("request_stamp_ns", 0)): item
            for item in streams["reference_ack"]
        }
        native_safe = {
            int(item.get("payload", {}).get("equilibrium_source_stamp_ns", 0)): item
            for item in streams["safe_action"]
        }
        seen_ack_stamps: set[int] = set()
        observed_ack_count = 0
        for row in human_acks:
            self._validate_shadow_identity(row, identity)
            reference_ack = row.get("reference_ack")
            safe_action = row.get("safe_action")
            if not isinstance(reference_ack, Mapping) or not isinstance(safe_action, Mapping):
                raise ProductionBridgeError("BRIDGE_SHADOW_HUMAN_ACK_PAYLOAD_MISSING")
            stamp = int(reference_ack.get("request_stamp_ns", 0))
            observation_id = row.get("observation_id")
            if (
                row.get("schema") != INTEGRATED_SHADOW_SCHEMA
                or row.get("actual_action_source") != "human"
                or row.get("policy_result_id") is not None
                or row.get("proposal_id") is not None
                or row.get("policy_executed_transition") is not False
                or row.get("policy_execution") is not False
                or row.get("formal_replay") is not False
                or row.get("real_online_r") is not False
                or row.get("ack_id") != f"human-ack:{stamp}"
                or stamp <= 0
                or stamp in seen_ack_stamps
                or reference_ack.get("accepted") is not True
                or native_acks.get(stamp, {}).get("payload") != reference_ack
                or native_safe.get(stamp, {}).get("payload") != safe_action
                or safe_action.get("arbitration", {})
                .get("raw_action", {})
                .get("source")
                != "human"
            ):
                raise ProductionBridgeError("BRIDGE_SHADOW_HUMAN_ACK_INVALID")
            if observation_id is not None:
                observation = observation_by_id.get(str(observation_id))
                if (
                    observation is None
                    or int(row.get("receive_monotonic_ns", 0))
                    < int(observation.get("t_ref_ns", 0))
                ):
                    raise ProductionBridgeError("BRIDGE_SHADOW_HUMAN_ACK_TIME_INVALID")
                observed_ack_count += 1
            seen_ack_stamps.add(stamp)
        if seen_ack_stamps != set(native_acks):
            raise ProductionBridgeError("BRIDGE_SHADOW_HUMAN_ACK_COVERAGE_MISMATCH")

        reconciliation = _read_json(
            shadow_root / "policy_shadow_camera_reconciliation.json"
        )
        records = reconciliation.get("records")
        if (
            reconciliation.get("schema") != INTEGRATED_SHADOW_SCHEMA
            or Path(str(reconciliation.get("native_episode", ""))).resolve()
            != episode_dir.resolve()
            or not isinstance(records, list)
            or len(records) != 2 * len(observations)
        ):
            raise ProductionBridgeError("BRIDGE_SHADOW_CAMERA_RECONCILIATION_INVALID")
        native_cameras = {
            role: {
                str(item.get("rgb_path", "")): item
                for item in streams[f"{role}_camera"]
            }
            for role in ("external", "wrist")
        }
        reconciled: set[tuple[str, str]] = set()
        for record in records:
            observation_id = str(record.get("observation_id", ""))
            role = str(record.get("role", ""))
            observation = observation_by_id.get(observation_id)
            key = (observation_id, role)
            stream_name = f"{role}_camera"
            path = str(record.get("rgb_path", ""))
            native = native_cameras.get(role, {}).get(path)
            if (
                observation is None
                or role not in {"external", "wrist"}
                or key in reconciled
                or record.get("clock_domain_id") != UPPER_CLOCK
                or record.get("same_recorder_jpeg") is not True
                or observation["stream_ids"].get(stream_name) != path
                or int(observation["stream_timestamps_ns"].get(stream_name, 0))
                != int(record.get("policy_receive_monotonic_ns", 0))
                or native is None
                or int(native.get("receive_monotonic_ns", 0))
                != int(record.get("native_receive_monotonic_ns", 0))
                or not (episode_dir / path).is_file()
            ):
                raise ProductionBridgeError("BRIDGE_SHADOW_CAMERA_BINDING_INVALID")
            reconciled.add(key)

        lease_payload = _read_json(
            shadow_root / "policy_shadow_initial_gripper_lease.json"
        )
        try:
            lease = InitialGripperAuthority.from_mapping(lease_payload).validate(
                max_feedback_age_ns=self.config.max_gripper_feedback_age_ns
            )
        except GripperProvenanceError as error:
            raise ProductionBridgeError(
                f"BRIDGE_SHADOW_INITIAL_GRIPPER_LEASE_INVALID:{error}"
            ) from error
        generation = lease.generation
        if (
            lease.episode_id != identity.get("episode_id")
            or generation.policy_revision != identity.get("policy_revision")
            or generation.policy_epoch != identity.get("policy_epoch")
            or generation.reset_generation != identity.get("reset_generation")
            or generation.takeover_generation != identity.get("takeover_generation")
        ):
            raise ProductionBridgeError("BRIDGE_SHADOW_INITIAL_GRIPPER_LEASE_IDENTITY_MISMATCH")

        seal = _read_json(shadow_root / "policy_shadow_episode_seal.json")
        latest_lineage_ns = max(
            int(results[-1].get("result_recorded_monotonic_ns", 0)),
            max(int(row.get("receive_monotonic_ns", 0)) for row in human_acks),
            int(observations[-1].get("t_ref_ns", 0)),
        )
        if (
            seal.get("schema") != INTEGRATED_CAPTURE_SCHEMA
            or seal.get("backend_schema") != INTEGRATED_SHADOW_SCHEMA
            or seal.get("actual_action_source") != "human"
            or seal.get("policy_inference") is not True
            or seal.get("policy_execution") is not False
            or seal.get("formal_replay") is not False
            or seal.get("real_online_r") is not False
            or seal.get("shadow_proposals_executed") is not False
            or seal.get("controller_owner") != "recorder"
            or seal.get("controller_process_count") != 1
            or seal.get("deploy_controller_started") is not False
            or seal.get("policy_action_publisher_created") is not False
            or Path(str(seal.get("native_episode", ""))).resolve()
            != episode_dir.resolve()
            or seal.get("native_episode_result") != native_result
            or seal.get("initial_gripper_lease") != lease_payload
            or seal.get("terminal_observation_id")
            != observations[-1].get("observation_id")
            or int(seal.get("observation_count", -1)) != len(observations)
            or int(seal.get("policy_request_count", -1)) != len(requests)
            or int(seal.get("policy_result_count", -1)) != len(results)
            or int(seal.get("human_action_ack_count", -1)) != observed_ack_count
            or int(seal.get("camera_records_reconciled", -1)) != len(records)
            or int(seal.get("sealed_monotonic_ns", 0)) < latest_lineage_ns
            or int(seal.get("sealed_monotonic_ns", 0))
            < int(native_result.get("finished_monotonic_ns", 0))
        ):
            raise ProductionBridgeError("BRIDGE_INTEGRATED_SHADOW_SEAL_INVALID")
        self._validate_shadow_identity(seal, identity)

        return {
            "initial_gripper_lease": lease,
            "summary": {
                "schema": INTEGRATED_SHADOW_SCHEMA,
                "session_id": identity["session_id"],
                "episode_id": identity["episode_id"],
                "policy_revision": identity["policy_revision"],
                "policy_epoch": identity["policy_epoch"],
                "reset_generation": identity["reset_generation"],
                "takeover_generation": identity["takeover_generation"],
                "observation_count": len(observations),
                "policy_request_count": len(requests),
                "policy_result_count": len(results),
                "policy_proposal_count": len(proposals),
                "human_ack_count": len(human_acks),
                "camera_reconciliation_count": len(records),
                "terminal_observation_id": observations[-1]["observation_id"],
                "actual_action_source": "human",
                "human_ack_policy_binding": None,
                "policy_execution": False,
                "formal_replay": False,
                "real_online_r": False,
                "state_streams_bound": True,
                "calibrated_tcp_wrench_materialized": True,
                "technical_seal": "complete",
                "operator_task_outcome": operator_task_outcome,
            },
        }

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

    def _initial_gripper_goal(
        self,
        streams: Mapping[str, list[dict[str, Any]]],
        *,
        episode_id: str,
    ) -> _Goal | None:
        records = [
            item.get("payload", {})
            for item in streams["safe_action"]
            if item.get("payload", {})
            .get("arbitration", {})
            .get("raw_action", {})
            .get("phase")
            == "episode_start"
        ]
        if len(records) != 1:
            raise ProductionBridgeError("BRIDGE_EPISODE_START_IDENTITY_INVALID")
        value = records[0].get("stage3_initial_gripper_authority")
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ProductionBridgeError("BRIDGE_INITIAL_GRIPPER_AUTHORITY_INVALID")
        try:
            authority = InitialGripperAuthority.from_mapping(value).validate(
                max_feedback_age_ns=self.config.max_gripper_feedback_age_ns
            )
        except GripperProvenanceError as error:
            raise ProductionBridgeError(
                f"BRIDGE_INITIAL_GRIPPER_AUTHORITY_INVALID:{error}"
            ) from error
        if authority.episode_id != episode_id:
            raise ProductionBridgeError("BRIDGE_INITIAL_GRIPPER_EPISODE_MISMATCH")
        return _Goal(
            sequence=authority.origin_local_goal_sequence,
            action_goal_id=authority.origin_action_goal_id,
            requested_state=authority.requested_state,
            requested_width_m=authority.requested_width_m,
            started_ns=authority.origin_accepted_monotonic_ns,
            accepted_ns=authority.origin_accepted_monotonic_ns,
            finished_ns=authority.terminal_finished_monotonic_ns,
            outcome=authority.terminal_outcome,
            generation=authority.generation,
            initial_authority=True,
        )

    def _goals(
        self,
        streams: Mapping[str, list[dict[str, Any]]],
        *,
        episode_id: str,
        integrated_initial_lease: InitialGripperAuthority | None = None,
    ) -> tuple[_Goal, ...]:
        initial = self._initial_gripper_goal(streams, episode_id=episode_id)
        if integrated_initial_lease is not None:
            integrated = _Goal(
                sequence=integrated_initial_lease.origin_local_goal_sequence,
                action_goal_id=integrated_initial_lease.origin_action_goal_id,
                requested_state=integrated_initial_lease.requested_state,
                requested_width_m=integrated_initial_lease.requested_width_m,
                started_ns=integrated_initial_lease.origin_accepted_monotonic_ns,
                accepted_ns=integrated_initial_lease.origin_accepted_monotonic_ns,
                finished_ns=integrated_initial_lease.terminal_finished_monotonic_ns,
                outcome=integrated_initial_lease.terminal_outcome,
                generation=None,
                initial_authority=True,
            )
            if initial is not None and initial != integrated:
                raise ProductionBridgeError(
                    "BRIDGE_INITIAL_GRIPPER_AUTHORITY_SOURCE_CONFLICT"
                )
            initial = integrated
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
        goals: list[_Goal] = [] if initial is None else [initial]
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
        if initial is not None and initial.sequence in targets:
            recorded = targets[initial.sequence]
            if str(recorded.get("action_goal_id", "")) != initial.action_goal_id:
                raise ProductionBridgeError(
                    "BRIDGE_INITIAL_GRIPPER_GOAL_ID_CONFLICT"
                )
            goals = [goal for goal in goals if not goal.initial_authority]
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
            if selection.get("lineage_schema") not in {
                None,
                POLICY_LINEAGE_SCHEMA,
            }:
                raise ProductionBridgeError("BRIDGE_POLICY_LINEAGE_SCHEMA_INVALID")
            for field in (
                "request_id",
                "result_id",
                "chunk_id",
                "proposal_id",
                "policy_revision",
            ):
                if not isinstance(selection[field], str) or not selection[field].strip():
                    raise ProductionBridgeError(
                        f"BRIDGE_POLICY_LINEAGE_IDENTITY_INVALID:{field}"
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
                "dispatch_sequence": int(
                    selection.get("dispatch_sequence", sequence)
                ),
                "source_sequence": sequence,
                "policy_fixture": True,
            }
            if (
                lineage["policy_epoch"] != policy_epoch
                or lineage["dispatch_sequence"] != sequence
                or int(selection.get("selected_index", lineage["selected_index"]))
                != lineage["selected_index"]
                or min(
                    lineage["reset_generation"],
                    lineage["takeover_generation"],
                    lineage["selected_index"],
                )
                < 0
            ):
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
        generation: GripperGeneration,
    ) -> tuple[_Goal, GripperAuthorityKind]:
        pending = [
            goal
            for goal in goals
            if goal.started_ns <= authority_ns < goal.accepted_ns
        ]
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
            and (goal.generation is None or goal.generation == generation)
        ]
        if not accepted:
            if any(
                goal.initial_authority
                and goal.accepted_ns <= authority_ns
                and goal.generation != generation
                for goal in goals
            ):
                raise ProductionBridgeError(
                    "BRIDGE_INITIAL_GRIPPER_GENERATION_MISMATCH"
                )
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

    def _materialized_camera(
        self,
        *,
        episode_dir: Path,
        path: Path,
        timestamp_ns: int,
        at_ns: int,
    ) -> dict[str, Any]:
        path = Path(path)
        if not path.is_file() or not path.is_relative_to(episode_dir):
            raise ProductionBridgeError("BRIDGE_MATERIALIZED_CAMERA_BLOB_INVALID")
        digest: str | None = None
        if self.config.hash_camera_files:
            digest = self._camera_hashes.get(path)
            if digest is None:
                digest = _sha256_file(path)
                self._camera_hashes[path] = digest
        age_ns = at_ns - int(timestamp_ns)
        if age_ns < 0 or age_ns > self.config.max_camera_age_ns:
            raise ProductionBridgeError("BRIDGE_MATERIALIZED_CAMERA_STALE")
        return {
            "blob_reference": path.relative_to(episode_dir).as_posix(),
            "sha256": digest,
            "timestamp_monotonic_ns": int(timestamp_ns),
            "age_ms": age_ns / 1.0e6,
            "timestamp_domain": "host_monotonic_receive",
        }

    def _materialized_observation(
        self,
        *,
        episode_dir: Path,
        episode_id: str,
        observation_id: str,
        materialization: EpisodeMaterialization,
        frame: int,
    ) -> dict[str, Any]:
        prepared = materialization.prepared
        if frame < 0 or frame >= len(prepared.tuple_host_ns):
            raise ProductionBridgeError("BRIDGE_MATERIALIZED_FRAME_INVALID")
        at_ns = int(prepared.tuple_host_ns[frame])
        provenance = prepared.provenance
        state7 = _finite_vector(
            prepared.state7[frame], 7, "BRIDGE_MATERIALIZED_STATE_INVALID"
        )
        wrench6 = _finite_vector(
            prepared.wrench6[frame], 6, "BRIDGE_MATERIALIZED_WRENCH_INVALID"
        )
        return {
            "observation_id": observation_id,
            "episode_id": episode_id,
            "frame_index": frame,
            "timestamp_monotonic_ns": at_ns,
            "clock_domain_id": self.config.clock_domain_id,
            "state7_absolute": list(state7),
            "wrench6": list(wrench6),
            "wrench_materialization": dict(materialization.wrench_provenance),
            "camera_external": self._materialized_camera(
                episode_dir=episode_dir,
                path=prepared.camera1_paths[frame],
                timestamp_ns=int(provenance["camera1_receive_monotonic_ns"][frame]),
                at_ns=at_ns,
            ),
            "camera_wrist": self._materialized_camera(
                episode_dir=episode_dir,
                path=prepared.camera2_paths[frame],
                timestamp_ns=int(provenance["camera2_receive_monotonic_ns"][frame]),
                at_ns=at_ns,
            ),
            "normalization": "not_applied; frozen normalizer deferred to learner adapter",
            "raw_wrench_learner_eligible": False,
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
            "anchor_frame": payload["identity"]["anchor_frame"],
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
        raw_by_identity: Mapping[tuple[str, int], dict[str, Any]],
        requested: Mapping[tuple[str, int], dict[str, Any]],
        ack_by_stamp: Mapping[int, dict[str, Any]],
        ack_by_receive: Mapping[int, dict[str, Any]],
        goals: Sequence[_Goal],
        used_new: set[int],
        timelines: Mapping[str, _Timeline],
        materialization: EpisodeMaterialization,
        macro: DetectorMacroTransition,
        shadow_evidence: Mapping[str, Any] | None = None,
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
        prepared = materialization.prepared
        anchor = macro.anchor_frame
        next_frame = macro.next_frame
        anchor_ack_ns = int(
            prepared.provenance["action_ack_receive_monotonic_ns"][anchor]
        )
        if anchor_ack_ns != ack_receive_ns:
            raise ProductionBridgeError("BRIDGE_MATERIALIZED_ACTION_ACK_MISMATCH")
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
        lineage, generation, executed_policy_fixture = self._lineage(
            episode_id=episode_id, raw=raw, safe=safe
        )
        if shadow_evidence is not None:
            lineage = {
                **lineage,
                "binding_kind": "recorded_live_human_execution",
                "actual_action_source": "human",
                "policy_result_id": None,
                "policy_proposal_id": None,
                "policy_executed_transition": False,
            }
        policy_fixture = executed_policy_fixture or shadow_evidence is not None
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
            generation=generation,
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
        lineage = dict(lineage)
        lineage.update(
            {
                "dispatch_sequence": sequence,
                "selected_post_adapter_absolute7": list(
                    full.accepted_absolute_action7
                ),
                "pose_command_id": pose.pose_command_id,
                "pose_ack_id": pose.pose_ack_id,
                "gripper_authority_reference": {
                    "authority_kind": gripper.authority_kind.value,
                    "origin_local_goal_sequence": (
                        gripper.origin_local_goal_sequence
                    ),
                    "origin_action_goal_id": gripper.origin_action_goal_id,
                },
            }
        )
        if kind is GripperAuthorityKind.NEW_COMMAND:
            used_new.add(goal.sequence)
        at_ns = int(prepared.tuple_host_ns[anchor])
        next_at_ns = int(prepared.tuple_host_ns[next_frame])
        if next_at_ns <= at_ns:
            raise ProductionBridgeError("BRIDGE_NEXT_OBSERVATION_NOT_CAUSAL")
        macro_span_ns = next_at_ns - at_ns
        executed_actions = [
            list(
                _finite_vector(
                    prepared.action7[frame],
                    7,
                    "BRIDGE_MATERIALIZED_ACTION_INVALID",
                )
            )
            for frame in range(anchor, next_frame)
        ]
        if len(executed_actions) != macro.executed_steps or not executed_actions:
            raise ProductionBridgeError("BRIDGE_MATERIALIZED_ACTION_SLICE_INVALID")
        if (
            math.dist(executed_actions[0][:3], list(selected_tcp6[:3]))
            > self.config.pose_position_tolerance_m
            or abs(executed_actions[0][6] - selected_width)
            > self.config.requested_width_tolerance_m
        ):
            raise ProductionBridgeError("BRIDGE_MATERIALIZED_ACTION_AUTHORITY_MISMATCH")
        slot_ack_ids: list[str] = []
        slot_goal_ids: list[str] = []
        slot_authority_kinds: list[str] = []
        for offset, frame in enumerate(range(anchor, next_frame)):
            slot_ack_ns = int(
                prepared.provenance["action_ack_receive_monotonic_ns"][frame]
            )
            slot_ack = ack_by_receive.get(slot_ack_ns)
            if slot_ack is None or slot_ack.get("payload", {}).get("accepted") is not True:
                raise ProductionBridgeError("BRIDGE_MATERIALIZED_SLOT_ACK_MISSING")
            slot_ack_ids.append(
                f"reference-request-stamp:"
                f"{int(slot_ack['payload'].get('request_stamp_ns', 0))}"
            )
            width = executed_actions[offset][6]
            slot_authority_ns = max(
                slot_ack_ns, int(prepared.tuple_host_ns[frame])
            )
            matching_goals = [
                item
                for item in goals
                if item.accepted_ns <= slot_authority_ns
                and abs(item.requested_width_m - width)
                <= self.config.requested_width_tolerance_m
                and (item.generation is None or item.generation == generation)
            ]
            if not matching_goals:
                raise ProductionBridgeError(
                    "BRIDGE_MATERIALIZED_SLOT_GRIPPER_AUTHORITY_MISSING"
                )
            slot_goal = max(matching_goals, key=lambda item: item.accepted_ns)
            slot_kind = (
                GripperAuthorityKind.NEW_COMMAND
                if slot_goal.sequence not in used_new
                and abs(slot_goal.started_ns - slot_ack_ns)
                <= self.config.max_gripper_command_association_ns
                else GripperAuthorityKind.HELD_FROM_ACCEPTED_COMMAND
            )
            if slot_kind is GripperAuthorityKind.NEW_COMMAND:
                used_new.add(slot_goal.sequence)
            slot_goal_ids.append(slot_goal.action_goal_id)
            slot_authority_kinds.append(slot_kind.value)
        padded_actions = list(executed_actions)
        padded_ack_ids = list(slot_ack_ids)
        padded_goal_ids = list(slot_goal_ids)
        padded_authority_kinds = list(slot_authority_kinds)
        while len(padded_actions) < 3:
            padded_actions.append(list(padded_actions[-1]))
            padded_ack_ids.append(padded_ack_ids[-1])
            padded_goal_ids.append(padded_goal_ids[-1])
            padded_authority_kinds.append(padded_authority_kinds[-1])
        macro_grid_ns = [
            int(prepared.tuple_host_ns[min(anchor + slot, next_frame - 1)])
            for slot in range(3)
        ]
        decision = int(safe.get("decision_id", -1))
        observation = self._materialized_observation(
            episode_dir=episode_dir,
            episode_id=episode_id,
            observation_id=f"{episode_id}:frame:{anchor}",
            materialization=materialization,
            frame=anchor,
        )
        next_observation = self._materialized_observation(
            episode_dir=episode_dir,
            episode_id=episode_id,
            observation_id=f"{episode_id}:frame:{next_frame}",
            materialization=materialization,
            frame=next_frame,
        )
        detector_trigger = materialization.detection_trace.trigger_frame
        if detector_trigger is None:
            raise ProductionBridgeError("BRIDGE_FROZEN_G1_DETECTOR_MISS")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "classification": (
                "recorded_live_policy_shadow" if policy_fixture else "recorded_offline_shadow"
            ),
            "identity": {
                "episode_id": episode_id,
                "decision_id": decision,
                "anchor_frame": anchor,
                "next_frame": next_frame,
                "task": task,
                "transition_uid": None,
            },
            "runtime_lineage": lineage,
            "shadow_policy_lineage": (
                None if shadow_evidence is None else dict(shadow_evidence)
            ),
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
                "ack_ids": padded_ack_ids,
                "gripper_origin_action_goal_ids": padded_goal_ids,
                "gripper_authority_kinds": padded_authority_kinds,
                "accepted_absolute_action_k7": padded_actions,
                "executed_action_mask": list(macro.executed_action_mask),
                "executed_steps": macro.executed_steps,
                "slot_owner": [
                    "policy" if source == "policy" else "human"
                ]
                * 3,
                "accepted_action_source": [
                    "policy" if source == "policy" else "human"
                ]
                * 3,
                "intervention_flags": [bool(raw.get("intervention", False))] * 3,
                "partial": macro.executed_steps < 3,
                "span_to_next_dispatch_ns": macro_span_ns,
                "source": "raw_to_lerobot_v3.prepare_episode.action7",
                "normalizer_refit": False,
            },
            "observation": observation,
            "next_observation": next_observation,
            "behavior": {
                "recorder_source": source,
                "actual_action_source": source,
                "recorder_owner": arbitration.get("owner"),
                "intervention": bool(raw.get("intervention", False)),
                "phase": raw.get("phase"),
                "workspace_clipped": safe.get("workspace_clipped"),
                "policy_execution": source == "policy",
                "human_ack_bound_to_policy_proposal": False,
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
                    _canonical_bytes(_read_json(episode_dir / "episode_result.json"))
                ),
            },
            "outcome": {
                "reward_available": True,
                "reward": macro.reward,
                "reward_source": REWARD_SOURCE,
                "episode_boundary": macro.terminated,
                "task_terminated": macro.terminated,
                "done": macro.terminated,
                "success": macro.terminated,
                "failure": False,
                "time_limit": False,
                "terminal_source": "causal_fifth_confirming_frame",
                "bootstrap_mask": macro.bootstrap_mask,
                "discount": macro.discount,
                "mc_return": macro.mc_return,
                "detector_terminal_frame": detector_trigger,
                "detector_streak_start_frame": (
                    materialization.detection_trace.streak_start_frame
                ),
                "detector_probability_at_trigger": (
                    materialization.detector_scores.probabilities[detector_trigger]
                ),
                "provenance": dict(materialization.outcome_provenance),
                "detector_outcome": "success",
                "operator_task_outcome": (
                    None
                    if shadow_evidence is None
                    else shadow_evidence["operator_task_outcome"]
                ),
                "current_observation_frame": anchor,
                "next_observation_frame": next_frame,
                "episode_saved": True,
            },
            "eligibility": {
                "shadow_outbox_eligible": True,
                "formal_training_replay_eligible": False,
                "recorded_live_policy_fixture": policy_fixture,
                "real_online_r": False,
                "formal_replay": False,
            },
            "commit": {
                "episode_sealed": True,
                "integrated_shadow_episode_sealed": shadow_evidence is not None,
                "pose_ack_watermark": int(ack.get("request_sequence", 0)),
                "gripper_terminal_sealed": True,
                "wrench_materialized": True,
                "reward_terminal_materialized": True,
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
        classification: str = "recorded_offline_shadow",
        operator_task_outcome: str | None = None,
        detector_outcome: str = "not_evaluated",
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
            policy_fixture=classification
            in {"recorded_live_policy_shadow", POLICY_EXECUTION_SMOKE_CLASSIFICATION},
            classification=classification,
            operator_task_outcome=operator_task_outcome,
            executed_action_source=(
                "policy"
                if classification == POLICY_EXECUTION_SMOKE_CLASSIFICATION
                else "human"
            ),
            policy_execution=(
                classification == POLICY_EXECUTION_SMOKE_CLASSIFICATION
            ),
            detector_outcome=detector_outcome,
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

    def _formal_online_r_observation(
        self,
        *,
        episode_dir: Path,
        episode_id: str,
        source: Mapping[str, Any],
        prepared: PreparedEpisode,
    ) -> dict[str, Any]:
        t_ref_ns = int(source.get("t_ref_ns", 0))
        frame = bisect_right(prepared.tuple_host_ns, t_ref_ns) - 1
        if frame < 0:
            raise ProductionBridgeError(
                "BRIDGE_FORMAL_R_CAUSAL_OBSERVATION_MISSING"
            )
        materialized_ns = int(prepared.tuple_host_ns[frame])
        state7 = _finite_vector(
            prepared.state7[frame], 7, "BRIDGE_FORMAL_R_STATE7_INVALID"
        )
        wrench6 = _finite_vector(
            prepared.wrench6[frame], 6, "BRIDGE_FORMAL_R_WRENCH6_INVALID"
        )
        stream_ids = source.get("stream_ids")
        stream_timestamps = source.get("stream_timestamps_ns")
        if not isinstance(stream_ids, Mapping) or not isinstance(
            stream_timestamps, Mapping
        ):
            raise ProductionBridgeError(
                "BRIDGE_FORMAL_R_OBSERVATION_STREAMS_MISSING"
            )

        cameras: dict[str, dict[str, Any]] = {}
        for role, model in (("external", "D435"), ("wrist", "D405")):
            stream_name = f"{role}_camera"
            relative = str(stream_ids.get(stream_name, ""))
            camera_path = (episode_dir / relative).resolve()
            camera_ns = int(stream_timestamps.get(stream_name, 0))
            if (
                not relative
                or not camera_path.is_relative_to(episode_dir.resolve())
                or not camera_path.is_file()
                or camera_ns <= 0
                or camera_ns > t_ref_ns
                or t_ref_ns - camera_ns > self.config.max_camera_age_ns
            ):
                raise ProductionBridgeError(
                    f"BRIDGE_FORMAL_R_{role.upper()}_CAMERA_INVALID"
                )
            cameras[role] = {
                "model": model,
                "blob_reference": relative,
                "timestamp_monotonic_ns": camera_ns,
                "age_ms": (t_ref_ns - camera_ns) / 1.0e6,
                "clock_domain_id": self.config.clock_domain_id,
            }

        return {
            "observation_id": str(source["observation_id"]),
            "episode_id": episode_id,
            "source_t_ref_monotonic_ns": t_ref_ns,
            "materialized_frame": frame,
            "materialized_timestamp_monotonic_ns": materialized_ns,
            "materialization_age_ms": (t_ref_ns - materialized_ns) / 1.0e6,
            "clock_domain_id": self.config.clock_domain_id,
            "state7_absolute": list(state7),
            "wrench6_calibrated_tcp": list(wrench6),
            "wrench_materialization": {
                "source": "raw_to_lerobot_v3.prepare_episode",
                "calibrated_tcp_wrench": True,
                "raw_wrench_learner_eligible": False,
            },
            "camera_external": cameras["external"],
            "camera_wrist": cameras["wrist"],
            "normalization": "deferred_to_frozen_replay_adapter",
        }

    def _formal_online_r_transition(
        self,
        *,
        episode_dir: Path,
        episode_id: str,
        task: str,
        integrated: Mapping[str, Any],
        source: Mapping[str, Any],
        terminal: bool,
    ) -> dict[str, Any]:
        request_id = str(source.get("request_id", ""))
        request = integrated["requests"].get(request_id)
        result = integrated["results"].get(request_id)
        proposal = integrated["proposals"].get(request_id)
        chunk = integrated["chunks"].get(request_id)
        current = integrated["observations"].get(
            str(source.get("current_observation_id", ""))
        )
        next_observation = integrated["observations"].get(
            str(source.get("next_observation_id", ""))
        )
        if not all(
            isinstance(value, Mapping)
            for value in (request, result, proposal, chunk, current, next_observation)
        ):
            raise ProductionBridgeError(
                "BRIDGE_FORMAL_R_POLICY_LINEAGE_MISSING"
            )
        if proposal.get("invalidated_by_takeover") is True:
            raise ProductionBridgeError(
                "BRIDGE_FORMAL_R_INVALIDATED_PROPOSAL_SELECTED"
            )
        sequence = int(source.get("selection", {}).get("sequence", -1))
        gripper_origin = integrated["gripper_origins"].get(sequence)
        if not isinstance(gripper_origin, Mapping):
            raise ProductionBridgeError(
                "BRIDGE_FORMAL_R_GRIPPER_ORIGIN_MISSING"
            )
        observation = self._formal_online_r_observation(
            episode_dir=episode_dir,
            episode_id=episode_id,
            source=current,
            prepared=integrated["prepared"],
        )
        materialized_next = self._formal_online_r_observation(
            episode_dir=episode_dir,
            episode_id=episode_id,
            source=next_observation,
            prepared=integrated["prepared"],
        )
        if (
            materialized_next["source_t_ref_monotonic_ns"]
            <= observation["source_t_ref_monotonic_ns"]
        ):
            raise ProductionBridgeError(
                "BRIDGE_FORMAL_R_NEXT_OBSERVATION_NOT_CAUSAL"
            )
        action7 = _finite_vector(
            source.get("accepted_absolute7"),
            7,
            "BRIDGE_FORMAL_R_ACCEPTED_ACTION7_INVALID",
        )
        pose_ack = source.get("pose_ack")
        gripper = source.get("gripper_authority")
        if (
            not isinstance(pose_ack, Mapping)
            or pose_ack.get("accepted") is not True
            or not isinstance(gripper, Mapping)
        ):
            raise ProductionBridgeError(
                "BRIDGE_FORMAL_R_ACTION7_AUTHORITY_INCOMPLETE"
            )
        generation = {
            field: int(source[field])
            for field in (
                "policy_epoch",
                "reset_generation",
                "takeover_generation",
            )
        }
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": POLICY_EXECUTION_SMOKE_CLASSIFICATION,
            "identity": {
                "episode_id": episode_id,
                "session_id": str(source["session_id"]),
                "decision_id": sequence,
                "anchor_frame": observation["materialized_frame"],
                "next_frame": materialized_next["materialized_frame"],
                "source_ack_id": str(source["ack_id"]),
                "task": task,
                "transition_uid": None,
            },
            "generation": generation,
            "policy_lineage": {
                "request": dict(request),
                "result": dict(result),
                "proposal": dict(proposal),
                "chunk": dict(chunk),
                "selection": dict(source["selection"]),
                "revision": str(source["policy_revision"]),
                "generation": generation,
            },
            "action_authority": {
                "accepted_absolute_action7": list(action7),
                "executed_action_source": "policy",
                "pose_command": dict(source["pose_command"]),
                "pose_ack": dict(pose_ack),
                "gripper": dict(gripper),
                "gripper_terminal_provenance": dict(gripper_origin),
                "safety_arbitration": dict(source["safety_arbitration"]),
                "full_action7_ack_closure": True,
            },
            "observation": observation,
            "next_observation": materialized_next,
            "outcome": {
                "reward": 1.0 if terminal else 0.0,
                "terminated": terminal,
                "truncated": False,
                "done": terminal,
                "bootstrap_mask": 0.0 if terminal else 1.0,
                "discount": 0.0 if terminal else 0.99,
                "operator_task_outcome": "success",
                "detector_outcome": "success",
                "terminal_observation_id": integrated["summary"][
                    "terminal_observation_id"
                ],
            },
            "eligibility": {
                "formal_training_replay_eligible": True,
                "formal_replay": True,
                "real_online_r": True,
                "replay_membership": "R_online",
            },
            "commit": {
                "source_episode_technical_seal": "complete",
                "episode_sealed": True,
                "policy_execution_smoke_bridge": "PASS",
                "learner_started": False,
                "actor_updates": 0,
                "critic_updates": 0,
                "optimizer_updates": 0,
                "checkpoint_updates": 0,
                "policy_revision_publications": 0,
            },
        }
        stable = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "source_ack_id": source["ack_id"],
            "sequence": sequence,
            "current_observation_id": source["current_observation_id"],
            "next_observation_id": source["next_observation_id"],
            "policy_revision": source["policy_revision"],
            "generation": generation,
        }
        payload["identity"]["transition_uid"] = _sha256_bytes(
            _canonical_bytes(stable)
        )
        payload["integrity"] = {
            "canonical_payload_sha256": _sha256_bytes(_canonical_bytes(payload))
        }
        return payload

    def admit_policy_execution_smoke(
        self,
        episode_dir: Path,
        *,
        operator_task_outcome: str,
    ) -> FormalOnlineRAdmissionReport:
        """Admit an already bridge-valid smoke episode into durable formal R."""

        episode_dir = Path(episode_dir)
        start = _read_json(episode_dir / "episode_start.json")
        episode_id = self._episode_id(episode_dir, start)
        episode_key = episode_id.replace("/", "__")
        result = _read_json(episode_dir / "episode_result.json")
        streams = self._load_streams(episode_dir)
        self._validate_seal(result, streams)
        integrated = self._load_integrated_capture(
            episode_dir=episode_dir,
            native_result=result,
            streams=streams,
            operator_task_outcome=operator_task_outcome,
        )
        if (
            not isinstance(integrated, Mapping)
            or integrated.get("classification")
            != POLICY_EXECUTION_SMOKE_CLASSIFICATION
        ):
            raise ProductionBridgeError(
                "BRIDGE_FORMAL_R_POLICY_EXECUTION_SMOKE_PASS_REQUIRED"
            )
        summary = integrated["summary"]
        if (
            summary.get("technical_seal") != "complete"
            or summary.get("operator_task_outcome") != "success"
            or summary.get("detector_outcome") != "success"
            or summary.get("executed_action_source") != "policy"
            or summary.get("policy_execution") is not True
            or summary.get("formal_replay") is not False
            or summary.get("real_online_r") is not False
            or summary.get("training_replay_eligible") is not False
            or summary.get("policy_lineage_complete") is not True
            or int(summary.get("human_override_executed_count", -1)) != 0
            or int(summary.get("model_update_count", -1)) != 0
        ):
            raise ProductionBridgeError(
                "BRIDGE_FORMAL_R_POLICY_EXECUTION_SMOKE_PASS_REQUIRED"
            )

        source_transitions = sorted(
            integrated["transitions"],
            key=lambda item: int(item.get("receive_monotonic_ns", 0)),
        )
        first_materialized_ns = int(integrated["prepared"].tuple_host_ns[0])
        replay_sources: list[Mapping[str, Any]] = []
        observation_warmup_excluded_count = 0
        for source in source_transitions:
            current = integrated["observations"].get(
                str(source.get("current_observation_id", ""))
            )
            next_observation = integrated["observations"].get(
                str(source.get("next_observation_id", ""))
            )
            if not isinstance(current, Mapping) or not isinstance(
                next_observation, Mapping
            ):
                raise ProductionBridgeError(
                    "BRIDGE_FORMAL_R_POLICY_LINEAGE_MISSING"
                )
            if int(current.get("t_ref_ns", 0)) < first_materialized_ns:
                observation_warmup_excluded_count += 1
                continue
            replay_sources.append(source)
        if not replay_sources:
            raise ProductionBridgeError(
                "BRIDGE_FORMAL_R_NO_CAUSAL_CALIBRATED_TRANSITIONS"
            )
        terminal_id = str(summary["terminal_observation_id"])
        terminal_indices = [
            index
            for index, item in enumerate(replay_sources)
            if item.get("next_observation_id") == terminal_id
        ]
        if terminal_indices != [len(replay_sources) - 1]:
            raise ProductionBridgeError(
                "BRIDGE_FORMAL_R_TERMINAL_BOUNDARY_INVALID"
            )
        task = str(result.get("task", start.get("task", "")))
        transitions = [
            self._formal_online_r_transition(
                episode_dir=episode_dir,
                episode_id=episode_id,
                task=task,
                integrated=integrated,
                source=source,
                terminal=index == len(replay_sources) - 1,
            )
            for index, source in enumerate(replay_sources)
        ]
        uids = [item["identity"]["transition_uid"] for item in transitions]
        if len(set(uids)) != len(transitions):
            raise ProductionBridgeError("BRIDGE_FORMAL_R_UID_DUPLICATE")
        invalidated_replay_count = sum(
            item["policy_lineage"]["proposal"].get("invalidated_by_takeover")
            is True
            for item in transitions
        )
        if invalidated_replay_count:
            raise ProductionBridgeError(
                "BRIDGE_FORMAL_R_INVALIDATED_PROPOSAL_SELECTED"
            )

        admission_relative = f"admissions/{episode_key}.json"
        admission_record = {
            "kind": "formal_online_r_admission",
            "episode_id": episode_id,
            "source_episode": str(episode_dir.resolve()),
            "source_episode_semantics": {
                "formal_replay": False,
                "real_online_r": False,
            },
            "admitted_replay_semantics": {
                "formal_replay": True,
                "real_online_r": True,
                "membership": "R_online",
            },
            "policy_execution_smoke_bridge": "PASS",
            "classification": POLICY_EXECUTION_SMOKE_CLASSIFICATION,
            "episode_sealed": True,
            "operator_task_outcome": "success",
            "detector_outcome": "success",
            "executed_action_source": "policy",
            "initial_gripper_lease": integrated[
                "initial_gripper_lease"
            ].to_dict(),
            "accepted_unique_r_transition_count": len(transitions),
            "human_override_count": int(summary["human_override_count"]),
            "human_override_replay_count": 0,
            "invalidated_proposal_replay_count": 0,
            "observation_warmup_excluded_count": (
                observation_warmup_excluded_count
            ),
            "transitions": [
                {
                    "transition_uid": item["identity"]["transition_uid"],
                    "canonical_payload_sha256": item["integrity"][
                        "canonical_payload_sha256"
                    ],
                    "source_ack_id": item["identity"]["source_ack_id"],
                }
                for item in transitions
            ],
            "model_updates": {
                "actor": 0,
                "critic": 0,
                "optimizer": 0,
                "checkpoint": 0,
                "policy_revision_publication": 0,
            },
        }
        admission_written = self._immutable_write(
            self.state_root / admission_relative, admission_record
        )

        wal_written = outbox_written = replay_written = idempotent = 0
        for transition in transitions:
            uid = transition["identity"]["transition_uid"]
            digest = transition["integrity"]["canonical_payload_sha256"]
            wal_relative = f"wal/{uid}.json"
            outbox_relative = f"outbox/{uid}.json"
            if self._immutable_write(
                self.state_root / wal_relative,
                {
                    "transition_uid": uid,
                    "canonical_payload_sha256": digest,
                    "admission_record": admission_relative,
                    "episode_sealed": True,
                    "payload": transition,
                },
            ):
                wal_written += 1
            if self._immutable_write(
                self.state_root / outbox_relative,
                {
                    "transition_uid": uid,
                    "canonical_payload_sha256": digest,
                    "admission_record": admission_relative,
                    "wal_record": wal_relative,
                    "episode_sealed": True,
                    "replay_membership": "R_online",
                },
            ):
                outbox_written += 1
            if self._immutable_write(
                self.state_root / "replay" / f"{uid}.json",
                {
                    "transition_uid": uid,
                    "canonical_payload_sha256": digest,
                    "admission_record": admission_relative,
                    "outbox_record": outbox_relative,
                    "episode_sealed": True,
                    "replay_membership": "R_online",
                    "payload": transition,
                },
            ):
                replay_written += 1
            else:
                idempotent += 1

        episode_manifest = {
            "kind": "formal_online_r_episode_seal",
            "episode_id": episode_id,
            "status": "SEALED_COMMITTED",
            "admission_record": admission_relative,
            "replay_membership": "R_online",
            "accepted_unique_r_transition_count": len(transitions),
            "transition_uids": uids,
            "human_override_replay_count": 0,
            "invalidated_proposal_replay_count": 0,
            "observation_warmup_excluded_count": (
                observation_warmup_excluded_count
            ),
            "learner_started": False,
            "actor_updates": 0,
            "critic_updates": 0,
            "optimizer_updates": 0,
            "checkpoint_updates": 0,
            "policy_revision_publications": 0,
        }
        episode_seal_written = self._immutable_write(
            self.state_root / "episodes" / f"{episode_key}.json",
            episode_manifest,
        )
        total_unique = len(list((self.state_root / "replay").glob("*.json")))
        return FormalOnlineRAdmissionReport(
            status="FORMAL_ONLINE_R_ADMITTED",
            episode_id=episode_id,
            classification=POLICY_EXECUTION_SMOKE_CLASSIFICATION,
            policy_execution_smoke_bridge="PASS",
            accepted_unique_r_transition_count=len(transitions),
            total_unique_r_transition_count=total_unique,
            training_starts=TRAINING_STARTS_UNIQUE_R,
            training_starts_reached=total_unique >= TRAINING_STARTS_UNIQUE_R,
            human_override_count=int(summary["human_override_count"]),
            human_override_replay_count=0,
            invalidated_proposal_replay_count=0,
            observation_warmup_excluded_count=(
                observation_warmup_excluded_count
            ),
            wal_written_count=wal_written,
            outbox_written_count=outbox_written,
            replay_written_count=replay_written,
            idempotent_transition_count=idempotent,
            admission_record_written=admission_written,
            episode_seal_written=episode_seal_written,
        )

    def process_episode(
        self,
        episode_dir: Path,
        *,
        dry_run: bool = False,
        inject_crash_after_wal: int | None = None,
        operator_task_outcome: str | None = None,
    ) -> BridgeReport:
        episode_dir = Path(episode_dir)
        start = _read_json(episode_dir / "episode_start.json")
        episode_id = self._episode_id(episode_dir, start)
        episode_key = episode_id.replace("/", "__")
        result_path = episode_dir / "episode_result.json"
        integrated_root = (
            episode_dir.parent.parent
            / "integrated_capture"
            / episode_dir.name
            / "streams"
        )
        integrated_present = integrated_root.is_dir()
        classification = "recorded_offline_shadow"
        if integrated_present:
            classification = (
                POLICY_EXECUTION_SMOKE_CLASSIFICATION
                if (integrated_root / "policy_execute_episode_seal.json").is_file()
                else "recorded_live_policy_shadow"
            )
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
        integrated_capture: dict[str, Any] | None = None
        try:
            streams = self._load_streams(episode_dir)
            self._validate_seal(result, streams)
            integrated_capture = self._load_integrated_capture(
                episode_dir=episode_dir,
                native_result=result,
                streams=streams,
                operator_task_outcome=operator_task_outcome,
            )
            if (
                integrated_capture is not None
                and integrated_capture.get("classification")
                == POLICY_EXECUTION_SMOKE_CLASSIFICATION
            ):
                summary = integrated_capture["summary"]
                return BridgeReport(
                    status=(
                        "DRY_RUN_READY"
                        if dry_run
                        else "SEALED_VALIDATED_NO_REPLAY"
                    ),
                    episode_id=episode_id,
                    sealed=True,
                    dry_run=dry_run,
                    candidate_count=len(integrated_capture["transitions"]),
                    outbox_eligible_count=0,
                    quarantined_count=0,
                    wal_written_count=0,
                    outbox_written_count=0,
                    idempotent_count=0,
                    quarantine_reasons=(),
                    recorded_offline_production_bridge="BLOCKED",
                    policy_fixture=True,
                    real_online_r_used=False,
                    formal_training_replay_written=False,
                    classification=POLICY_EXECUTION_SMOKE_CLASSIFICATION,
                    technical_seal=str(summary["technical_seal"]),
                    operator_task_outcome=str(summary["operator_task_outcome"]),
                    executed_action_source="policy",
                    policy_execution=True,
                    detector_outcome=str(summary["detector_outcome"]),
                    shadow_observation_count=int(summary["observation_count"]),
                    shadow_policy_request_count=int(summary["policy_request_count"]),
                    shadow_policy_result_count=int(summary["policy_result_count"]),
                    shadow_policy_proposal_count=int(summary["policy_proposal_count"]),
                    training_replay_eligible=False,
                    policy_lineage_complete=bool(summary["policy_lineage_complete"]),
                    policy_chunk_count=int(summary["policy_chunk_count"]),
                    policy_action_ack_count=int(summary["policy_action_ack_count"]),
                    human_override_count=int(summary["human_override_count"]),
                    human_override_executed_count=int(
                        summary["human_override_executed_count"]
                    ),
                    model_update_count=int(summary["model_update_count"]),
                )
            goals = self._goals(
                streams,
                episode_id=episode_id,
                integrated_initial_lease=(
                    None
                    if integrated_capture is None
                    else integrated_capture["initial_gripper_lease"]
                ),
            )
            if self.episode_materializer is None:
                raise ProductionBridgeError("BRIDGE_EPISODE_MATERIALIZER_UNBOUND")
            materialization = self.episode_materializer(episode_dir).validate()
            if (
                materialization.prepared.raw_episode_id != episode_dir.name
                or materialization.prepared.task
                != str(result.get("task", start.get("task", "")))
            ):
                raise ProductionBridgeError("BRIDGE_MATERIALIZED_EPISODE_IDENTITY_MISMATCH")
        except Exception as raw_error:
            error = (
                raw_error
                if isinstance(raw_error, ProductionBridgeError)
                else ProductionBridgeError(
                    f"BRIDGE_EPISODE_MATERIALIZATION_FAILED:"
                    f"{type(raw_error).__name__}:{raw_error}"
                )
            )
            return self._episode_quarantine(
                episode_id=episode_id,
                episode_key=episode_key,
                result_path=result_path,
                error=error,
                dry_run=dry_run,
                classification=classification,
                operator_task_outcome=operator_task_outcome,
                detector_outcome=(
                    "miss"
                    if str(error) == "BRIDGE_FROZEN_G1_DETECTOR_MISS"
                    else "not_evaluated"
                ),
            )
        try:
            requested: dict[tuple[str, int], dict[str, Any]] = {}
            for item in streams["requested_equilibrium"]:
                key = (str(item.get("source", "")), int(item.get("sequence", -1)))
                if key in requested:
                    raise ProductionBridgeError("BRIDGE_REQUESTED_IDENTITY_DUPLICATE")
                requested[key] = item
            safe_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
            for item in streams["safe_action"]:
                raw = (
                    item.get("payload", {})
                    .get("arbitration", {})
                    .get("raw_action", {})
                )
                key = (str(raw.get("source", "")), int(raw.get("sequence", -1)))
                if key in safe_by_identity:
                    raise ProductionBridgeError("BRIDGE_SAFE_ACTION_IDENTITY_DUPLICATE")
                safe_by_identity[key] = item
            raw_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
            for item in streams["raw_action"]:
                raw = item.get("payload", {})
                key = (str(raw.get("source", "")), int(raw.get("sequence", -1)))
                if key in raw_by_identity:
                    raise ProductionBridgeError("BRIDGE_RAW_ACTION_IDENTITY_DUPLICATE")
                raw_by_identity[key] = raw
            ack_by_stamp: dict[int, dict[str, Any]] = {}
            ack_by_receive: dict[int, dict[str, Any]] = {}
            for item in streams["reference_ack"]:
                stamp = int(item.get("payload", {}).get("request_stamp_ns", 0))
                if stamp <= 0 or stamp in ack_by_stamp:
                    raise ProductionBridgeError("BRIDGE_POSE_ACK_IDENTITY_DUPLICATE")
                ack_by_stamp[stamp] = item
                receive_ns = int(item.get("receive_monotonic_ns", 0))
                if receive_ns <= 0 or receive_ns in ack_by_receive:
                    raise ProductionBridgeError("BRIDGE_POSE_ACK_RECEIVE_DUPLICATE")
                ack_by_receive[receive_ns] = item
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
                classification=classification,
                operator_task_outcome=operator_task_outcome,
                detector_outcome="success",
            )
        used_new: set[int] = set()
        transitions: list[dict[str, Any]] = []
        quarantines: list[dict[str, Any]] = []
        policy_fixture = integrated_capture is not None
        shadow_evidence = (
            None if integrated_capture is None else integrated_capture["summary"]
        )
        task = str(result.get("task", start.get("task", "")))
        for macro in materialization.macros:
            decision = -1
            try:
                anchor_ack_ns = int(
                    materialization.prepared.provenance[
                        "action_ack_receive_monotonic_ns"
                    ][macro.anchor_frame]
                )
                anchor_ack = ack_by_receive.get(anchor_ack_ns)
                if anchor_ack is None:
                    raise ProductionBridgeError("BRIDGE_MATERIALIZED_ANCHOR_ACK_MISSING")
                stamp = int(anchor_ack.get("payload", {}).get("request_stamp_ns", 0))
                anchor_request = next(
                    (
                        item
                        for item in requested.values()
                        if int(item.get("source_stamp_ns", 0)) == stamp
                    ),
                    None,
                )
                if anchor_request is None:
                    raise ProductionBridgeError(
                        "BRIDGE_MATERIALIZED_ANCHOR_REQUEST_MISSING"
                    )
                key = (
                    str(anchor_request.get("source", "")),
                    int(anchor_request.get("sequence", -1)),
                )
                item = safe_by_identity.get(key)
                if item is None:
                    raise ProductionBridgeError("BRIDGE_MATERIALIZED_SAFE_ACTION_MISSING")
                raw = (
                    item.get("payload", {})
                    .get("arbitration", {})
                    .get("raw_action", {})
                )
                if raw.get("phase") != "control":
                    raise ProductionBridgeError("BRIDGE_MATERIALIZED_ANCHOR_NOT_CONTROL")
                decision = int(item.get("payload", {}).get("decision_id", -1))
                transition, is_policy = self._transition(
                    episode_dir=episode_dir,
                    episode_id=episode_id,
                    task=task,
                    safe_record=item,
                    raw_by_identity=raw_by_identity,
                    requested=requested,
                    ack_by_stamp=ack_by_stamp,
                    ack_by_receive=ack_by_receive,
                    goals=goals,
                    used_new=used_new,
                    timelines=timelines,
                    materialization=materialization,
                    macro=macro,
                    shadow_evidence=shadow_evidence,
                )
                transitions.append(transition)
                policy_fixture = policy_fixture or is_policy
            except (ProductionBridgeError, GripperProvenanceError) as error:
                quarantines.append(
                    {
                        "schema_version": REPORT_VERSION,
                        "episode_id": episode_id,
                        "decision_id": decision,
                        "anchor_frame": macro.anchor_frame,
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
                    "anchor_frame": item["anchor_frame"],
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
                "quarantined_transitions": [
                    {
                        "decision_id": item["decision_id"],
                        "anchor_frame": item["anchor_frame"],
                    }
                    for item in quarantines
                ],
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
            candidate_count=len(materialization.macros),
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
            classification=classification,
            operator_task_outcome=operator_task_outcome,
            detector_outcome="success",
            detector_trigger_frame=materialization.detection_trace.trigger_frame,
            shadow_observation_count=(
                0 if shadow_evidence is None else shadow_evidence["observation_count"]
            ),
            shadow_policy_request_count=(
                0 if shadow_evidence is None else shadow_evidence["policy_request_count"]
            ),
            shadow_policy_result_count=(
                0 if shadow_evidence is None else shadow_evidence["policy_result_count"]
            ),
            shadow_policy_proposal_count=(
                0 if shadow_evidence is None else shadow_evidence["policy_proposal_count"]
            ),
            shadow_human_ack_count=(
                0 if shadow_evidence is None else shadow_evidence["human_ack_count"]
            ),
        )
