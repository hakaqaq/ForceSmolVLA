"""Direct FR3 native-stream to ForceSmolVLA LeRobot v3 conversion.

The public CLI is formal-only: it validates all approval artifacts before it
creates the output directory.  The alignment engine is kept separate and takes
an explicit, approved runtime contract so tests cannot acquire permissive
defaults accidentally.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Sequence

import numpy as np

from .action_delta import ActionDeltaProcessor
from .conversion_gate import formal_conversion_preflight
from .dataset_v3 import create_dataset
from .geometry import (
    StaticWrenchCalibration,
    calibrated_tcp_wrench_conditioned_on_measured_tcp_pose,
)
from .split import split_episodes
from .normalizer import CartesianNormalizerBundle, build_action_target_population
from .temporal import controller_reference_grid, match_measured_tcp_pose_causal_zoh


RAW_FORMAT = "fr3-hilserl-impedance-native-raw-v5"
FPS = 30
CAMERA_ORDER = ("camera1", "camera2")
CAMERA_STREAMS = ("external_camera", "wrist_camera")
CAMERA_ROLES = ("external", "wrist")
CAMERA_MODELS = ("D435", "D405")
SUPPORTED_CLOCK_MAP = "shared-lower-tail-median-v1"
SUPPORTED_GRID_ANCHOR = "first-reference-ack-global-zero-phase-rational-30hz-v1"
SUPPORTED_ACTION_ASSOCIATION = "latest-causal-accepted-reference-with-pose-check-v1"
SUPPORTED_FILTER = "scipy-sosfilt-fixed-500hz-per-valid-source-sample-v1"
LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"


@dataclass(frozen=True)
class RuntimeContract:
    """All semantics that must be approved before formal conversion."""

    max_pose_age_ms: float
    camera_max_age_ms: float
    max_intercamera_skew_ms: float
    clock_map_method: str
    clock_map_lower_fraction: float
    clock_map_min_lower_samples: int
    clock_map_max_callback_delay_p99_ms: float | None
    controller_grid_anchor: str
    action_association: str
    action_pose_tolerance_m: float
    action_quaternion_tolerance_rad: float
    filter_implementation: str
    filter_sos: np.ndarray
    filter_warmup_samples: int
    max_wrench_source_gap_ms: float
    split_ratios: tuple[float, float, float]
    split_seed: str

    def __post_init__(self) -> None:
        positive = {
            "max_pose_age_ms": self.max_pose_age_ms,
            "camera_max_age_ms": self.camera_max_age_ms,
            "max_intercamera_skew_ms": self.max_intercamera_skew_ms,
            "action_pose_tolerance_m": self.action_pose_tolerance_m,
            "action_quaternion_tolerance_rad": self.action_quaternion_tolerance_rad,
            "max_wrench_source_gap_ms": self.max_wrench_source_gap_ms,
        }
        if any(not np.isfinite(value) or value <= 0 for value in positive.values()):
            raise ValueError(f"runtime thresholds must be finite and positive: {positive}")
        if self.clock_map_max_callback_delay_p99_ms is not None and (
            not np.isfinite(self.clock_map_max_callback_delay_p99_ms)
            or self.clock_map_max_callback_delay_p99_ms <= 0
        ):
            raise ValueError("clock_map_max_callback_delay_p99_ms must be positive or null")
        if self.clock_map_method != SUPPORTED_CLOCK_MAP:
            raise ValueError(f"unsupported clock-map semantics: {self.clock_map_method!r}")
        if self.controller_grid_anchor != SUPPORTED_GRID_ANCHOR:
            raise ValueError(f"unsupported grid-anchor semantics: {self.controller_grid_anchor!r}")
        if self.action_association != SUPPORTED_ACTION_ASSOCIATION:
            raise ValueError(f"unsupported action association: {self.action_association!r}")
        if self.filter_implementation != SUPPORTED_FILTER:
            raise ValueError(f"unsupported filter semantics: {self.filter_implementation!r}")
        if not 0 < self.clock_map_lower_fraction <= 0.1:
            raise ValueError("clock_map_lower_fraction must be in (0, 0.1]")
        if self.clock_map_min_lower_samples <= 0 or self.filter_warmup_samples < 0:
            raise ValueError("clock-map sample count and filter warm-up are invalid")
        sos = np.asarray(self.filter_sos, dtype=np.float64)
        if sos.ndim != 2 or sos.shape[1] != 6 or not np.all(np.isfinite(sos)):
            raise ValueError("filter_sos must be finite scipy SOS shape [sections,6]")
        object.__setattr__(self, "filter_sos", sos)
        if len(self.split_ratios) != 3 or not np.isclose(sum(self.split_ratios), 1.0):
            raise ValueError("split_ratios must contain three values summing to one")
        if not self.split_seed:
            raise ValueError("split_seed is required")

    @classmethod
    def from_approved_json(cls, path: Path) -> "RuntimeContract":
        payload = _load_json(path)
        if payload.get("artifact_status") != "approved" or payload.get("formal_ready") is not True:
            raise PermissionError("converter runtime contract is not formally approved")
        return cls(
            max_pose_age_ms=float(payload["pose"]["max_age_ms"]),
            camera_max_age_ms=float(payload["cameras"]["max_age_ms"]),
            max_intercamera_skew_ms=float(payload["cameras"]["max_intercamera_skew_ms"]),
            clock_map_method=str(payload["clock_map"]["method"]),
            clock_map_lower_fraction=float(payload["clock_map"]["lower_fraction"]),
            clock_map_min_lower_samples=int(payload["clock_map"]["min_lower_samples"]),
            clock_map_max_callback_delay_p99_ms=float(
                payload["clock_map"]["max_callback_delay_p99_ms"]
            ),
            controller_grid_anchor=str(payload["controller_grid"]["anchor"]),
            action_association=str(payload["action"]["association"]),
            action_pose_tolerance_m=float(payload["action"]["pose_tolerance_m"]),
            action_quaternion_tolerance_rad=float(
                payload["action"]["quaternion_tolerance_rad"]
            ),
            filter_implementation=str(payload["wrench_filter"]["implementation"]),
            filter_sos=np.asarray(payload["wrench_filter"]["sos"], dtype=np.float64),
            filter_warmup_samples=int(payload["wrench_filter"]["warmup_samples"]),
            max_wrench_source_gap_ms=float(payload["wrench_filter"]["max_source_gap_ms"]),
            split_ratios=tuple(float(value) for value in payload["split"]["ratios"]),
            split_seed=str(payload["split"]["seed"]),
        )

    @classmethod
    def from_development_json(cls, path: Path) -> "RuntimeContract":
        payload = _load_json(path)
        if (
            payload.get("artifact_status") != "development_only"
            or payload.get("formal_ready") is not False
        ):
            raise PermissionError("development conversion requires the development runtime contract")
        maximum_callback_delay = payload["clock_map"]["max_callback_delay_p99_ms"]
        return cls(
            max_pose_age_ms=float(payload["pose"]["max_age_ms"]),
            camera_max_age_ms=float(payload["cameras"]["max_age_ms"]),
            max_intercamera_skew_ms=float(payload["cameras"]["max_intercamera_skew_ms"]),
            clock_map_method=str(payload["clock_map"]["method"]),
            clock_map_lower_fraction=float(payload["clock_map"]["lower_fraction"]),
            clock_map_min_lower_samples=int(payload["clock_map"]["min_lower_samples"]),
            clock_map_max_callback_delay_p99_ms=(
                None if maximum_callback_delay is None else float(maximum_callback_delay)
            ),
            controller_grid_anchor=str(payload["controller_grid"]["anchor"]),
            action_association=str(payload["action"]["association"]),
            action_pose_tolerance_m=float(payload["action"]["pose_tolerance_m"]),
            action_quaternion_tolerance_rad=float(
                payload["action"]["quaternion_tolerance_rad"]
            ),
            filter_implementation=str(payload["wrench_filter"]["implementation"]),
            filter_sos=np.asarray(payload["wrench_filter"]["sos"], dtype=np.float64),
            filter_warmup_samples=int(payload["wrench_filter"]["warmup_samples"]),
            max_wrench_source_gap_ms=float(payload["wrench_filter"]["max_source_gap_ms"]),
            split_ratios=tuple(float(value) for value in payload["split"]["ratios"]),
            split_seed=str(payload["split"]["seed"]),
        )


@dataclass(frozen=True)
class ClockMap:
    offset_ns: int
    map_id: str
    sha256: str
    callback_delay_p99_ms: float
    callback_delay_max_ms: float

    def source_to_host(self, stamps_ns: np.ndarray) -> np.ndarray:
        stamps = np.asarray(stamps_ns, dtype=np.int64)
        return stamps + np.int64(self.offset_ns)


@dataclass(frozen=True)
class PreparedEpisode:
    raw_episode_id: str
    task: str
    tuple_host_ns: np.ndarray
    state7: np.ndarray
    wrench6: np.ndarray
    action7: np.ndarray
    camera1_paths: tuple[Path, ...]
    camera2_paths: tuple[Path, ...]
    provenance: dict[str, np.ndarray]
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.tuple_host_ns)
        if self.state7.shape != (count, 7):
            raise ValueError("prepared state must have shape [N,7]")
        if self.wrench6.shape != (count, 6):
            raise ValueError("prepared wrench must have shape [N,6]")
        if self.action7.shape != (count, 7):
            raise ValueError("prepared action must have shape [N,7]")
        if len(self.camera1_paths) != count or len(self.camera2_paths) != count:
            raise ValueError("prepared cameras must contain one path per tuple")
        if any(np.asarray(value).shape != (count,) for value in self.provenance.values()):
            raise ValueError("each provenance vector must have shape [N]")
        if not all(np.all(np.isfinite(value)) for value in (self.state7, self.wrench6, self.action7)):
            raise ValueError("prepared numeric features must be finite")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    if not records:
        raise ValueError(f"empty source stream: {path}")
    return records


def _timestamps(records: Sequence[dict[str, Any]], key: str, label: str) -> np.ndarray:
    try:
        values = np.asarray([int(record[key]) for record in records], dtype=np.int64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} has a missing/noninteger {key}") from error
    if np.any(values <= 0) or (len(values) > 1 and np.any(np.diff(values) <= 0)):
        raise ValueError(f"{label}.{key} must be positive and strictly increasing")
    return values


def _stream(episode_dir: Path, result: dict[str, Any], name: str) -> list[dict[str, Any]]:
    records = _load_jsonl(episode_dir / "streams" / f"{name}.jsonl")
    declared = int(result.get("stream_counts", {}).get(name, -1))
    if declared != len(records):
        raise ValueError(f"{name}: declared {declared} records but found {len(records)}")
    return records


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_manifest(
    raw_root: Path, *, progress_every: int | None = None
) -> tuple[list[dict[str, Any]], str]:
    """Hash every original source file without following links or writing raw data."""

    entries: list[dict[str, Any]] = []
    root_digest = hashlib.sha256()
    for path in sorted(raw_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"raw source tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(raw_root).as_posix()
        entry = {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        entries.append(entry)
        root_digest.update(
            json.dumps(entry, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        if progress_every is not None and len(entries) % progress_every == 0:
            print(f"hashed {len(entries)} raw source files", flush=True)
    if not entries:
        raise ValueError("raw source tree contains no files")
    return entries, root_digest.hexdigest()


def estimate_clock_map(
    source_streams: Iterable[tuple[str, Sequence[dict[str, Any]]]],
    contract: RuntimeContract,
) -> ClockMap:
    """Apply only the exact clock-map estimator named by the approved contract."""

    source_streams = tuple(source_streams)
    offsets: list[np.ndarray] = []
    labels: list[str] = []
    for label, records in source_streams:
        source = _timestamps(records, "source_stamp_ns", label)
        receive = _timestamps(records, "receive_monotonic_ns", label)
        offsets.append(receive - source)
        labels.append(label)
    combined = np.concatenate(offsets)
    lower_count = min(
        len(combined),
        max(contract.clock_map_min_lower_samples, int(len(combined) * contract.clock_map_lower_fraction)),
    )
    lower = np.partition(combined, lower_count - 1)[:lower_count]
    offset_ns = int(np.median(lower))
    delays = np.concatenate(
        [
            _timestamps(records, "receive_monotonic_ns", label)
            - (_timestamps(records, "source_stamp_ns", label) + offset_ns)
            for label, records in source_streams
        ]
    )
    p99_ms = float(np.percentile(delays, 99)) / 1e6
    max_ms = float(np.max(delays)) / 1e6
    if (
        contract.clock_map_max_callback_delay_p99_ms is not None
        and p99_ms > contract.clock_map_max_callback_delay_p99_ms
    ):
        raise ValueError("CLOCK_MAP_CALLBACK_DELAY_P99_EXCEEDED")
    payload = {
        "method": contract.clock_map_method,
        "streams": labels,
        "lower_fraction": contract.clock_map_lower_fraction,
        "min_lower_samples": contract.clock_map_min_lower_samples,
        "offset_ns": offset_ns,
        "callback_delay_p99_ms": p99_ms,
        "callback_delay_max_ms": max_ms,
    }
    digest = _canonical_json_sha256(payload)
    return ClockMap(offset_ns, f"sha256:{digest}", digest, p99_ms, max_ms)


def _calibration(payload: dict[str, Any]) -> StaticWrenchCalibration:
    transform = payload["static_transform_tcp_sensor"]
    return StaticWrenchCalibration(
        calibration_id=str(payload["calibration_id"]),
        translation_tcp_sensor_m=transform["translation_m"],
        quaternion_tcp_sensor_xyzw=transform["quaternion_xyzw"],
        sensor_bias6=payload["sensor_bias6"],
        wrench_sign6=payload["wrench_sign6"],
        downstream_mass_kg=float(payload["downstream_mass_kg"]),
        downstream_com_sensor_m=payload["downstream_com_sensor_m"],
        gravity_base_m_s2=payload["gravity_base_m_s2"],
    )


def _latest_indices(source_ns: np.ndarray, target_ns: np.ndarray, label: str) -> np.ndarray:
    indices = np.searchsorted(source_ns, target_ns, side="right") - 1
    if np.any(indices < 0):
        raise ValueError(f"{label}: target has no causal source")
    if np.any(source_ns[indices] > target_ns):
        raise AssertionError(f"{label}: future source selected")
    return indices


def _normalize_quaternions(values: np.ndarray, label: str) -> np.ndarray:
    quaternions = np.asarray(values, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4 or not np.all(np.isfinite(quaternions)):
        raise ValueError(f"{label} quaternions must be finite [N,4]")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError(f"{label} contains a zero quaternion")
    return quaternions / norms[:, None]


def _quaternion_geodesic(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(abs(np.dot(first, second)))
    return float(2 * np.arccos(np.clip(dot, -1.0, 1.0)))


def _associate_acknowledged_actions(
    references: Sequence[dict[str, Any]],
    acknowledgements: Sequence[dict[str, Any]],
    contract: RuntimeContract,
) -> tuple[np.ndarray, np.ndarray]:
    reference_times = _timestamps(references, "accepted_receive_monotonic_ns", "accepted_reference")
    action_times: list[int] = []
    actions: list[np.ndarray] = []
    for acknowledgement in acknowledgements:
        payload = acknowledgement.get("payload", {})
        if payload.get("accepted") is not True:
            raise ValueError("REFERENCE_ACK_REJECTED")
        ack_time = int(acknowledgement["receive_monotonic_ns"])
        index = int(np.searchsorted(reference_times, ack_time, side="right")) - 1
        if index < 0:
            raise ValueError("REFERENCE_ACK_HAS_NO_CAUSAL_ACCEPTED_REFERENCE")
        reference = references[index]
        reference_pose = reference["pose"]
        ack_pose = payload["accepted_pose"]
        reference_position = np.asarray(reference_pose["position_m"], dtype=np.float64)
        ack_position = np.asarray(ack_pose["position_m"], dtype=np.float64)
        if np.linalg.norm(reference_position - ack_position) > contract.action_pose_tolerance_m:
            raise ValueError("REFERENCE_ACK_POSITION_MISMATCH")
        reference_q = _normalize_quaternions(
            np.asarray([reference_pose["quaternion_xyzw"]]), "accepted_reference"
        )[0]
        ack_q = _normalize_quaternions(
            np.asarray([ack_pose["quaternion_xyzw"]]), "reference_ack"
        )[0]
        if _quaternion_geodesic(reference_q, ack_q) > contract.action_quaternion_tolerance_rad:
            raise ValueError("REFERENCE_ACK_QUATERNION_MISMATCH")
        width = float(reference["target_gripper_width_m"])
        if not np.isfinite(width) or not 0 <= width <= 0.1:
            raise ValueError("REFERENCE_ACK_GRIPPER_TARGET_INVALID")
        action_times.append(ack_time)
        actions.append(np.concatenate((reference_position, reference_q, [width])))
    times = np.asarray(action_times, dtype=np.int64)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("reference acknowledgements must be strictly increasing")
    return times, np.asarray(actions, dtype=np.float64)


def _filter_calibrated_wrench(
    calibrated: np.ndarray, contract: RuntimeContract
) -> np.ndarray:
    try:
        import scipy
        from scipy.signal import sosfilt, sosfilt_zi
    except ImportError as error:
        raise RuntimeError("formal wrench filtering requires scipy==1.16.3") from error
    if scipy.__version__ != "1.16.3":
        raise RuntimeError(f"formal wrench filtering requires scipy==1.16.3, got {scipy.__version__}")
    output = np.empty_like(calibrated)
    zi = sosfilt_zi(contract.filter_sos)
    for axis in range(6):
        output[:, axis], _ = sosfilt(
            contract.filter_sos,
            calibrated[:, axis],
            zi=zi * calibrated[0, axis],
        )
    return output


def _rpy_unwrapped(quaternions: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as error:
        raise RuntimeError("quaternion conversion requires scipy==1.16.3") from error
    return np.unwrap(Rotation.from_quat(quaternions).as_euler("xyz"), axis=0)


def prepare_episode(
    episode_dir: Path,
    *,
    session: dict[str, Any],
    calibration_payload: dict[str, Any],
    contract: RuntimeContract,
) -> PreparedEpisode:
    """Prepare one episode without writing anything."""

    result = _load_json(episode_dir / "episode_result.json")
    if result.get("saved") is not True or result.get("fatal_reason") is not None:
        raise ValueError("RAW_EPISODE_NOT_SAVED_OR_FATAL")
    task = str(result.get("task", "")).strip()
    if not task:
        raise ValueError("TASK_PROMPT_MISSING")

    poses = _stream(episode_dir, result, "measured_tcp_pose")
    raw_wrenches = _stream(episode_dir, result, "wrench_notch_sensor")
    grippers = _stream(episode_dir, result, "gripper_state")
    camera1 = _stream(episode_dir, result, "external_camera")
    camera2 = _stream(episode_dir, result, "wrist_camera")
    references = _stream(episode_dir, result, "accepted_reference")
    acknowledgements = _stream(episode_dir, result, "reference_ack")

    pose_source_ns = _timestamps(poses, "source_stamp_ns", "measured_tcp_pose")
    wrench_source_ns = _timestamps(raw_wrenches, "source_stamp_ns", "wrench_notch_sensor")
    gripper_source_ns = _timestamps(grippers, "source_stamp_ns", "gripper_state")
    camera1_ns = _timestamps(camera1, "receive_monotonic_ns", "external_camera")
    camera2_ns = _timestamps(camera2, "receive_monotonic_ns", "wrist_camera")
    if float(np.max(np.diff(wrench_source_ns))) / 1e6 > contract.max_wrench_source_gap_ms:
        raise ValueError("WRENCH_SOURCE_GAP_EXCEEDED")

    clock_map = estimate_clock_map(
        (
            ("measured_tcp_pose", poses),
            ("wrench_notch_sensor", raw_wrenches),
            ("gripper_state", grippers),
        ),
        contract,
    )
    pose_host_ns = clock_map.source_to_host(pose_source_ns)
    gripper_host_ns = clock_map.source_to_host(gripper_source_ns)

    pose_matches = match_measured_tcp_pose_causal_zoh(
        pose_source_ns,
        wrench_source_ns,
        max_pose_age_ms=contract.max_pose_age_ms,
    )
    valid_wrench = pose_matches.valid.copy()
    raw_values = np.asarray(
        [record.get("force_xyz_n_torque_xyz_nm") for record in raw_wrenches], dtype=np.float64
    )
    valid_wrench &= np.all(np.isfinite(raw_values), axis=1)
    if int(valid_wrench.sum()) <= contract.filter_warmup_samples:
        raise ValueError("WRENCH_VALID_SAMPLES_INSUFFICIENT_FOR_FILTER_WARMUP")
    calibration = _calibration(calibration_payload)
    calibrated_values: list[np.ndarray] = []
    valid_raw_indices = np.flatnonzero(valid_wrench)
    for raw_index in valid_raw_indices:
        pose_index = int(pose_matches.pose_indices[raw_index])
        pose = poses[pose_index]["pose"]
        transformed = calibrated_tcp_wrench_conditioned_on_measured_tcp_pose(
            raw_values[raw_index],
            pose["position_m"],
            pose["quaternion_xyzw"],
            calibration,
        ).wrench_base_at_tcp6
        if not np.all(np.isfinite(transformed)):
            raise ValueError("WRENCH_CALIBRATED_NONFINITE")
        calibrated_values.append(transformed)
    filtered = _filter_calibrated_wrench(np.asarray(calibrated_values), contract)
    usable_raw_indices = valid_raw_indices[contract.filter_warmup_samples :]
    filtered = filtered[contract.filter_warmup_samples :]
    filtered_source_ns = wrench_source_ns[usable_raw_indices]
    filtered_host_ns = clock_map.source_to_host(filtered_source_ns)
    geometry_pose_indices = pose_matches.pose_indices[usable_raw_indices]
    geometry_pose_source_ns = pose_source_ns[geometry_pose_indices]
    geometry_pose_age_ms = pose_matches.pose_age_ms[usable_raw_indices]

    action_event_ns, action_pose_quat_width = _associate_acknowledged_actions(
        references, acknowledgements, contract
    )
    start_ns = max(
        int(result["started_monotonic_ns"]),
        int(pose_host_ns[0]),
        int(gripper_host_ns[0]),
        int(camera1_ns[0]),
        int(camera2_ns[0]),
        int(filtered_host_ns[0]),
        int(action_event_ns[0]),
    )
    end_ns = min(
        int(result["finished_monotonic_ns"]),
        int(pose_host_ns[-1]),
        int(gripper_host_ns[-1]),
        int(camera1_ns[-1]),
        int(camera2_ns[-1]),
        int(filtered_host_ns[-1]),
        int(action_event_ns[-1]),
    )
    grid = controller_reference_grid(session_start_ack_ns=start_ns, episode_end_ns=end_ns, fps=FPS)
    if len(grid) < 2:
        raise ValueError("EPISODE_COMMON_INTERVAL_TOO_SHORT")

    pose_grid_indices = _latest_indices(pose_host_ns, grid, "state pose")
    pose_grid_age_ms = (grid - pose_host_ns[pose_grid_indices]) / 1e6
    camera1_indices = _latest_indices(camera1_ns, grid, "camera1")
    camera2_indices = _latest_indices(camera2_ns, grid, "camera2")
    camera1_age_ms = (grid - camera1_ns[camera1_indices]) / 1e6
    camera2_age_ms = (grid - camera2_ns[camera2_indices]) / 1e6
    intercamera_skew_ms = np.abs(
        camera1_ns[camera1_indices] - camera2_ns[camera2_indices]
    ) / 1e6
    if np.any(pose_grid_age_ms > contract.max_pose_age_ms):
        raise ValueError("STATE_POSE_AGE_EXCEEDED")
    if np.any(camera1_age_ms > contract.camera_max_age_ms) or np.any(
        camera2_age_ms > contract.camera_max_age_ms
    ):
        raise ValueError("CAMERA_AGE_EXCEEDED")
    if np.any(intercamera_skew_ms > contract.max_intercamera_skew_ms):
        raise ValueError("INTERCAMERA_SKEW_EXCEEDED")

    gripper_indices = _latest_indices(gripper_host_ns, grid, "gripper")
    wrench_indices = _latest_indices(filtered_host_ns, grid, "filtered wrench")
    action_indices = _latest_indices(action_event_ns, grid, "acknowledged action")
    pose_positions = np.asarray([record["pose"]["position_m"] for record in poses], dtype=np.float64)
    pose_quaternions = _normalize_quaternions(
        np.asarray([record["pose"]["quaternion_xyzw"] for record in poses]), "state pose"
    )
    state_rpy = _rpy_unwrapped(pose_quaternions[pose_grid_indices])
    gripper_width = np.asarray([record["width_m"] for record in grippers], dtype=np.float64)
    if np.any(~np.isfinite(gripper_width)) or np.any((gripper_width < 0) | (gripper_width > 0.1)):
        raise ValueError("GRIPPER_WIDTH_INVALID")
    state7 = np.column_stack(
        (pose_positions[pose_grid_indices], state_rpy, gripper_width[gripper_indices])
    ).astype(np.float32)
    action_quaternions = _normalize_quaternions(
        action_pose_quat_width[:, 3:7], "acknowledged action"
    )
    action_rpy = _rpy_unwrapped(action_quaternions[action_indices])
    action7 = np.column_stack(
        (
            action_pose_quat_width[action_indices, :3],
            action_rpy,
            action_pose_quat_width[action_indices, 7],
        )
    ).astype(np.float32)
    wrench6 = filtered[wrench_indices].astype(np.float32)

    for records, indices, role, model in (
        (camera1, camera1_indices, CAMERA_ROLES[0], CAMERA_MODELS[0]),
        (camera2, camera2_indices, CAMERA_ROLES[1], CAMERA_MODELS[1]),
    ):
        for record in (records[int(index)] for index in np.unique(indices)):
            if record.get("role") != role or model not in str(record.get("model", "")):
                raise ValueError(f"CAMERA_IDENTITY_MISMATCH:{role}/{model}")

    camera1_paths = tuple(episode_dir / camera1[index]["rgb_path"] for index in camera1_indices)
    camera2_paths = tuple(episode_dir / camera2[index]["rgb_path"] for index in camera2_indices)
    if any(not path.is_file() for path in camera1_paths + camera2_paths):
        raise FileNotFoundError("selected camera image is missing")
    selected_raw_indices = usable_raw_indices[wrench_indices]
    selected_geometry_pose_indices = geometry_pose_indices[wrench_indices]
    provenance = {
        "tuple_host_monotonic_ns": grid,
        "state_pose_source_stamp_ns": pose_source_ns[pose_grid_indices],
        "state_pose_age_ms": pose_grid_age_ms.astype(np.float32),
        "camera1_receive_monotonic_ns": camera1_ns[camera1_indices],
        "camera1_age_ms": camera1_age_ms.astype(np.float32),
        "camera2_receive_monotonic_ns": camera2_ns[camera2_indices],
        "camera2_age_ms": camera2_age_ms.astype(np.float32),
        "intercamera_skew_ms": intercamera_skew_ms.astype(np.float32),
        "gripper_source_stamp_ns": gripper_source_ns[gripper_indices],
        "pose_source_stamp_ns": pose_source_ns[selected_geometry_pose_indices],
        "pose_age_ms": geometry_pose_age_ms[wrench_indices].astype(np.float32),
        "wrench_raw_source_stamp_ns": wrench_source_ns[selected_raw_indices],
        "wrench_filter_output_stamp_ns": filtered_source_ns[wrench_indices],
        "action_ack_receive_monotonic_ns": action_event_ns[action_indices],
        "action_ack_age_ms": ((grid - action_event_ns[action_indices]) / 1e6).astype(np.float32),
        "calibration_index": np.zeros(len(grid), dtype=np.int64),
        "validity_bits": np.full(len(grid), 0xFF, dtype=np.int64),
    }
    diagnostics = {
        "frames": len(grid),
        "clock_map_id": clock_map.map_id,
        "clock_map_sha256": clock_map.sha256,
        "clock_offset_ns": clock_map.offset_ns,
        "clock_callback_delay_p99_ms": clock_map.callback_delay_p99_ms,
        "pose_age_max_ms": float(np.max(geometry_pose_age_ms[wrench_indices])),
        "camera1_age_max_ms": float(np.max(camera1_age_ms)),
        "camera2_age_max_ms": float(np.max(camera2_age_ms)),
        "intercamera_skew_max_ms": float(np.max(intercamera_skew_ms)),
        "filter_warmup_samples_excluded": contract.filter_warmup_samples,
        "raw_wrench_samples": len(raw_wrenches),
        "usable_filtered_samples": len(filtered),
    }
    return PreparedEpisode(
        raw_episode_id=episode_dir.name,
        task=task,
        tuple_host_ns=grid,
        state7=state7,
        wrench6=wrench6,
        action7=action7,
        camera1_paths=camera1_paths,
        camera2_paths=camera2_paths,
        provenance=provenance,
        diagnostics=diagnostics,
    )


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (480, 640, 3):
        raise ValueError(f"unexpected RGB shape {rgb.shape}: {path}")
    return rgb


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def convert_dataset(
    *,
    raw_root: Path,
    output_root: Path,
    repo_id: str,
    project_root: Path,
    contract: RuntimeContract,
    runtime_contract_path: Path,
    formal: bool,
) -> dict[str, Any]:
    """Convert all eligible raw episodes under an explicit runtime contract."""

    lerobot_head = subprocess.run(
        ["git", "-C", str(project_root / "vendor/lerobot"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if lerobot_head != LEROBOT_COMMIT:
        raise RuntimeError(f"LeRobot source drift: {lerobot_head} != {LEROBOT_COMMIT}")
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION

    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(f"ForceSmolVLA requires LeRobot codebase v3.0, got {CODEBASE_VERSION}")
    session = _load_json(raw_root / "session.json")
    if session.get("raw_format_version") != RAW_FORMAT or int(session.get("canonical_fps", -1)) != FPS:
        raise ValueError("raw session format/fps does not match ForceSmolVLA v4.1")
    cameras = session.get("cameras", {})
    expected = (
        ("observation.image", "external", "D435"),
        ("observation.wrist_image", "wrist", "D405"),
    )
    for key, role, model in expected:
        record = cameras.get(key, {})
        if record.get("role") != role or record.get("model") != model:
            raise ValueError(f"session camera contract mismatch: {key}")
    calibration_path = project_root / "configs" / (
        "calibration_bundle.approved.json" if formal else "calibration_bundle.development.json"
    )
    geometry_path = project_root / "configs" / (
        "wrench_geometry_spec.approved.json" if formal else "wrench_geometry_spec.development.json"
    )
    runtime_path = runtime_contract_path.resolve()
    if not runtime_path.is_file():
        raise FileNotFoundError(f"runtime contract does not exist: {runtime_path}")
    calibration_payload = _load_json(calibration_path)
    expected_status = "approved" if formal else "development_only"
    expected_ready = formal
    if (
        calibration_payload.get("artifact_status") != expected_status
        or calibration_payload.get("formal_ready") is not expected_ready
    ):
        raise PermissionError("calibration bundle status does not match conversion mode")

    prepared: list[PreparedEpisode] = []
    excluded: list[dict[str, str]] = []
    episode_dirs = sorted((raw_root / "episodes").glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if not episode_dirs:
        raise FileNotFoundError("raw root has no episodes")
    for episode_dir in episode_dirs:
        try:
            prepared.append(
                prepare_episode(
                    episode_dir,
                    session=session,
                    calibration_payload=calibration_payload,
                    contract=contract,
                )
            )
            print(
                f"prepared {episode_dir.name}: {len(prepared[-1].tuple_host_ns)} frames",
                flush=True,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as error:
            excluded.append({"raw_episode_id": episode_dir.name, "reason": str(error)})
            print(f"excluded {episode_dir.name}: {error}", flush=True)
    split = split_episodes(
        (episode.raw_episode_id for episode in prepared),
        ratios=contract.split_ratios,
        seed=contract.split_seed,
    )

    split_payload = {
        "method": "sha256-ranked-episode-disjoint-v1",
        "seed": contract.split_seed,
        "ratios": list(contract.split_ratios),
        "train": list(split.train),
        "val": list(split.val),
        "test": list(split.test),
    }
    split_sha256 = _canonical_json_sha256(split_payload)
    train_episodes = [episode for episode in prepared if episode.raw_episode_id in split.train]
    train_state = np.concatenate([episode.state7 for episode in train_episodes])
    train_wrench = np.concatenate([episode.wrench6 for episode in train_episodes])
    action_target_population = build_action_target_population(
        (
            (episode.raw_episode_id, episode.state7, episode.action7)
            for episode in train_episodes
        )
    )
    train_delta = action_target_population.action_target7
    train_delta_episode_ids = action_target_population.episode_ids
    train_episode_ids = tuple(
        episode.raw_episode_id
        for episode in train_episodes
        for _ in range(len(episode.state7))
    )
    if not geometry_path.is_file():
        raise FileNotFoundError("wrench geometry spec is required")
    normalizer = CartesianNormalizerBundle.fit(
        state7=train_state,
        wrench6=train_wrench,
        delta_action7=train_delta,
        sample_episode_ids=train_episode_ids,
        delta_action_episode_ids=train_delta_episode_ids,
        split=split,
        split_sha256=split_sha256,
        calibration_bundle_sha256=_sha256_file(calibration_path),
        wrench_geometry_spec_sha256=_sha256_file(geometry_path),
        action_target_population=action_target_population.manifest(
            split_sha256=split_sha256,
            builder_source_sha256=_sha256_file(
                Path(__file__).with_name("normalizer.py")
            ),
        ),
    )
    normalizer_payload = normalizer.manifest()
    print("hashing immutable raw source tree", flush=True)
    source_entries, source_tree_sha256 = source_tree_manifest(raw_root, progress_every=10000)
    print(f"raw source hash complete: {source_tree_sha256}", flush=True)

    dataset = create_dataset(output_root, repo_id=repo_id)
    episode_manifest: list[dict[str, Any]] = []
    try:
        for output_index, episode in enumerate(prepared):
            for frame_index in range(len(episode.tuple_host_ns)):
                frame = {
                    "task": episode.task,
                    "observation.images.camera1": _read_rgb(episode.camera1_paths[frame_index]),
                    "observation.images.camera2": _read_rgb(episode.camera2_paths[frame_index]),
                    "observation.state": episode.state7[frame_index],
                    "observation.wrench": episode.wrench6[frame_index],
                    "action": episode.action7[frame_index],
                }
                frame.update(
                    {
                        f"provenance.{name}": np.asarray([values[frame_index]])
                        for name, values in episode.provenance.items()
                    }
                )
                dataset.add_frame(frame)
            dataset.save_episode()
            episode_manifest.append(
                {
                    "output_episode_index": output_index,
                    "raw_episode_id": episode.raw_episode_id,
                    "split": (
                        "train"
                        if episode.raw_episode_id in split.train
                        else "val"
                        if episode.raw_episode_id in split.val
                        else "test"
                    ),
                    "frames": len(episode.tuple_host_ns),
                    "task": episode.task,
                    "diagnostics": episode.diagnostics,
                }
            )
            print(
                f"wrote {episode.raw_episode_id} as v3 episode {output_index:06d} "
                f"({len(episode.tuple_host_ns)} frames)",
                flush=True,
            )
        dataset.finalize()
    finally:
        if getattr(dataset, "image_writer", None) is not None:
            dataset.stop_image_writer()

    manifest = {
        "artifact_status": "approved" if formal else "development_only",
        "formal_ready": formal,
        "schema": "forcesmolvla-v4.1-available-sensor-lerobot-v3",
        "repo_id": repo_id,
        "raw_root": str(raw_root),
        "raw_source_tree_sha256": source_tree_sha256,
        "output_root": str(output_root),
        "lerobot_commit": LEROBOT_COMMIT,
        "converter_source": {
            "entrypoint": "tools/convert_franka_raw_to_lerobot_v3.py",
            "entrypoint_sha256": _sha256_file(
                project_root / "tools/convert_franka_raw_to_lerobot_v3.py"
            ),
            "implementation": "src/forcesmolvla/raw_to_lerobot_v3.py",
            "implementation_sha256": _sha256_file(Path(__file__)),
            "runtime_contract_sha256": _sha256_file(runtime_path),
        },
        "camera_order": list(CAMERA_ORDER),
        "camera_roles": {"camera1": "D435 third-person", "camera2": "D405 wrist"},
        "state": "7D measured TCP xyz+rpy plus gripper width",
        "wrench": "calibrated TCP wrench conditioned on measured TCP pose",
        "action": "7D absolute acknowledged TCP xyz+rpy plus gripper width",
        "fps": FPS,
        "evaluation_scope": "within-session offline fine-tuning; not cross-session generalization",
        "episodes": episode_manifest,
        "excluded_episodes": excluded,
        "split": split_payload,
        "split_sha256": split_sha256,
        "normalizer_stats_sha256": normalizer_payload["normalizer_stats_sha256"],
        "normalizer_fit_contract": normalizer_payload["fit_contract"],
        "calibration_id_by_index": {"0": calibration_payload["calibration_id"]},
        "formal_quality_gates": (
            "verified"
            if formal
            else "not evaluated: RuleSpec thresholds/signatures remain approval_pending"
        ),
    }
    _write_jsonl(output_root / "source_files.sha256.jsonl", source_entries)
    _write_json(output_root / "split_manifest.json", split_payload)
    _write_json(output_root / "normalizer_manifest.json", normalizer_payload)
    _write_json(output_root / "conversion_manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert native FR3 data to the ForceSmolVLA LeRobot v3 contract"
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/rlc123/fr3_client_ws/datasets/task1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/rlc123/ForceSmolVLA/datasets/task1_forcesmolvla_v4_1"),
    )
    parser.add_argument("--repo-id", default="local/task1_forcesmolvla_v4_1")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--runtime-spec",
        type=Path,
        help="development-only runtime contract; defaults to the task1 development spec",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="write an explicitly non-formal audit dataset from development candidate semantics",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    if args.preflight_only and args.development_only:
        raise SystemExit("--preflight-only and --development-only are mutually exclusive")
    if args.development_only:
        if args.output_root.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {args.output_root}")
        runtime_path = (
            args.runtime_spec.resolve()
            if args.runtime_spec is not None
            else project_root / "configs/converter_runtime_spec.development.json"
        )
        contract = RuntimeContract.from_development_json(runtime_path)
        manifest = convert_dataset(
            raw_root=args.raw_root.resolve(),
            output_root=args.output_root.resolve(),
            repo_id=args.repo_id,
            project_root=project_root,
            contract=contract,
            runtime_contract_path=runtime_path,
            formal=False,
        )
        print(
            json.dumps(
                {
                    "status": "converted_development_only",
                    "episodes": len(manifest["episodes"]),
                    "excluded_episodes": len(manifest["excluded_episodes"]),
                },
                indent=2,
            )
        )
        return
    if args.runtime_spec is not None:
        raise SystemExit("--runtime-spec is permitted only with --development-only")
    try:
        result = formal_conversion_preflight(
            raw_root=args.raw_root,
            output_root=args.output_root,
            project_root=project_root,
        )
    except (FileNotFoundError, PermissionError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "blocked_fail_closed",
                    "reason": str(error),
                    "output_created": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2) from None
    if args.preflight_only:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    runtime_path = project_root / "configs/converter_runtime_spec.approved.json"
    contract = RuntimeContract.from_approved_json(runtime_path)
    manifest = convert_dataset(
        raw_root=args.raw_root.resolve(),
        output_root=args.output_root.resolve(),
        repo_id=args.repo_id,
        project_root=project_root,
        contract=contract,
        runtime_contract_path=runtime_path,
        formal=True,
    )
    print(json.dumps({"status": "converted", "episodes": len(manifest["episodes"])}, indent=2))
