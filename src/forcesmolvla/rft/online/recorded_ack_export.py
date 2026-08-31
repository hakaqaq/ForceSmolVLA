"""Export a provenance-checked recorded ACK parity fixture from native streams."""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from forcesmolvla.raw_to_lerobot_v3 import RuntimeContract, prepare_episode

from forcesmolvla.rft.online.temporal_parity import (
    DEFAULT_RECORDED_FIXTURE,
    ROOT,
    directory_tree_sha256,
    sha256_file,
    validate_recorded_ack_fixture,
)


DEFAULT_RAW_SESSION = Path("/home/rlc123/fr3_client_ws/datasets/task2")
DEFAULT_RAW_EPISODE = DEFAULT_RAW_SESSION / "episodes/episode_000018"
DEFAULT_CAPTURE_MANIFEST = (
    ROOT / "golden_fixtures/stage3_recorded_ack_fixture.v1.capture_manifest.json"
)
DEFAULT_PARENT_BINDING: Path | None = None
CANONICAL_MANIFEST_ROOT = (
    ROOT / "outputs/task2/sft/checkpoints/forcesmolvla_sft_step_010000/manifests"
)
DEFAULT_TERMINAL_INDEX = (
    ROOT
    / "artifacts/development/stage2/g1_frozen_detector_transition_view.v1"
    / "transition_index.parquet"
)
FIXTURE_ID = "task2-episode_000018-recorded-ack-k3-terminal"


class RecordedAckExportError(ValueError):
    """A fail-closed native-source or fixture requirement failure."""

    def __init__(self, *missing_fields: str) -> None:
        self.missing_fields = tuple(str(value) for value in missing_fields)
        super().__init__("; ".join(self.missing_fields))


def _require(condition: bool, missing_field: str) -> None:
    if not condition:
        raise RecordedAckExportError(missing_field)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label}: missing file {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RecordedAckExportError(f"{label}: invalid JSON {path}") from error
    _require(isinstance(value, dict), f"{label}: expected JSON object")
    return value


def _load_stream(episode: Path, name: str) -> list[dict[str, Any]]:
    path = episode / "streams" / f"{name}.jsonl"
    _require(path.is_file(), f"native stream missing: {name}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, TypeError, ValueError) as error:
        raise RecordedAckExportError(f"native stream invalid: {name}") from error
    _require(bool(rows), f"native stream empty: {name}")
    return rows


def _binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def _fixture_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _runtime_contract(path: Path) -> RuntimeContract:
    payload = _load_json(path, "Stage-2 runtime contract")
    status = payload.get("artifact_status")
    if status == "approved":
        return RuntimeContract.from_approved_json(path)
    if status == "development_only":
        return RuntimeContract.from_development_json(path)
    raise RecordedAckExportError("Stage-2 runtime contract artifact_status missing or invalid")


def _pose(value: Mapping[str, Any]) -> dict[str, list[float]]:
    try:
        position = [float(item) for item in value["position_m"]]
        quaternion = [float(item) for item in value["quaternion_xyzw"]]
    except (KeyError, TypeError, ValueError) as error:
        raise RecordedAckExportError("recorded Pose ACK pose fields missing") from error
    _require(len(position) == 3 and len(quaternion) == 4, "recorded Pose ACK pose shape invalid")
    return {"position_m": position, "quaternion_xyzw": quaternion}


def recorded_ack_id(*, request_sequence: int, request_stamp_ns: int) -> str:
    """Use the native request natural key; do not mint a replacement UUID."""

    return f"reference-ack:{request_sequence}:{request_stamp_ns}"


def _derive_transition_selection(
    terminal_index: Path, episode_id: str,
) -> dict[str, int]:
    try:
        import pandas as pd
    except ImportError as error:
        raise RecordedAckExportError(
            "terminal transition parquet reader missing: pandas"
        ) from error
    _require(terminal_index.is_file(), f"terminal transition index missing: {terminal_index}")
    try:
        frame = pd.read_parquet(terminal_index)
        episode = frame[frame["episode_id"] == episode_id]
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RecordedAckExportError("terminal transition index unreadable") from error
    terminal = episode[
        (episode["terminated"] == True)  # noqa: E712 - pandas scalar comparison
        & (episode["executed_steps"] > 0)
        & (episode["executed_steps"] < 3)
    ]
    _require(len(terminal) == 1, "unique terminal partial K=3 transition missing")
    terminal_row = terminal.iloc[0]
    full = episode[
        (episode["next_frame"] == int(terminal_row["anchor_frame"]))
        & (episode["executed_steps"] == 3)
        & (episode["terminated"] == False)  # noqa: E712 - pandas scalar comparison
    ]
    _require(len(full) == 1, "full K=3 macro immediately before terminal transition missing")
    full_row = full.iloc[0]
    full_mask = [bool(value) for value in full_row["executed_action_mask"]]
    terminal_mask = [bool(value) for value in terminal_row["executed_action_mask"]]
    _require(full_mask == [True, True, True], "full K=3 macro action mask incomplete")
    _require(terminal_mask == [True, True, False], "terminal partial macro mask is not [T,T,F]")
    _require(
        int(terminal_row["next_frame"]) == int(terminal_row["detector_terminal_frame"]),
        "terminal observation boundary identity mismatch",
    )
    start = int(full_row["anchor_frame"])
    terminal_anchor = int(terminal_row["anchor_frame"])
    terminal_observation = int(terminal_row["next_frame"])
    _require(terminal_anchor == start + 3, "10 Hz next anchor is not one K=3 macro after current")
    _require(
        terminal_observation == terminal_anchor + 2,
        "terminal partial macro length is not two",
    )
    return {
        "prepared_grid_start_index": start,
        "prepared_grid_stop_index_exclusive": terminal_observation + 1,
        "current_observation_grid_index": 0,
        "next_observation_grid_index": 3,
        "terminal_observation_grid_index": terminal_observation - start,
        "last_executable_grid_index": terminal_observation - start - 1,
        "full_macro_transition_index": int(full_row["transition_index"]),
        "terminal_transition_index": int(terminal_row["transition_index"]),
    }


def _gripper_authority(
    episode: Path, *, established_before_ns: int,
) -> dict[str, Any]:
    targets = _load_stream(episode, "gripper_target")
    statuses = _load_stream(episode, "gripper_goal_status")
    status_by_id = {row.get("action_goal_id"): row for row in statuses}
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for target in targets:
        status = status_by_id.get(target.get("action_goal_id"))
        if (
            status is not None
            and int(status.get("finished_monotonic_ns", 0)) <= established_before_ns
        ):
            candidates.append((target, status))
    _require(bool(candidates), "initial gripper authority completed before selected macro missing")
    target, status = max(candidates, key=lambda pair: int(pair[1]["finished_monotonic_ns"]))
    _require(
        target.get("action_goal_id") == status.get("action_goal_id")
        and target.get("local_goal_sequence") == status.get("local_goal_sequence"),
        "gripper target/status identity mismatch",
    )
    _require(status.get("outcome") in {"reached", "stalled"}, "gripper terminal outcome missing")
    return {
        "action_goal_id": str(target["action_goal_id"]),
        "local_goal_sequence": int(target["local_goal_sequence"]),
        "accepted_monotonic_ns": int(target["accepted_monotonic_ns"]),
        "target_receive_monotonic_ns": int(target["receive_monotonic_ns"]),
        "finished_monotonic_ns": int(status["finished_monotonic_ns"]),
        "status_receive_monotonic_ns": int(status["receive_monotonic_ns"]),
        "requested_state": str(target["requested_state"]),
        "target_width_m": float(target["target_width_m"]),
        "outcome": str(status["outcome"]),
    }


def _accepted_ack_rows(
    episode: Path,
    *,
    selected_ack_times: Sequence[int],
    authority: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    raw_references = _load_stream(episode, "accepted_reference")
    raw_acks = _load_stream(episode, "reference_ack")
    safe_actions = _load_stream(episode, "safe_action")
    ack_by_receive = {int(row["receive_monotonic_ns"]): row for row in raw_acks}
    reference_times = [int(row["accepted_receive_monotonic_ns"]) for row in raw_references]
    _require(
        reference_times == sorted(reference_times),
        "accepted_reference timestamps not monotonic",
    )

    references: list[dict[str, Any]] = []
    acknowledgements: list[dict[str, Any]] = []
    age_limits: set[float] = set()
    goal_id = str(authority["action_goal_id"])
    for receive_ns in sorted(set(int(value) for value in selected_ack_times)):
        raw_ack = ack_by_receive.get(receive_ns)
        _require(raw_ack is not None, f"recorded reference_ack missing at {receive_ns}")
        payload = raw_ack.get("payload")
        _require(
            isinstance(payload, Mapping),
            f"recorded reference_ack payload missing at {receive_ns}",
        )
        _require(payload.get("accepted") is True, f"positive Pose ACK missing at {receive_ns}")
        request_sequence = int(payload.get("request_sequence", -1))
        request_stamp_ns = int(payload.get("request_stamp_ns", 0))
        controller_ack_ns = int(payload.get("ack_monotonic_ns", 0))
        _require(
            request_sequence >= 0 and request_stamp_ns > 0 and controller_ack_ns > 0,
            f"recorded ACK natural identity/timestamp missing at {receive_ns}",
        )
        reference_index = bisect.bisect_right(reference_times, receive_ns) - 1
        _require(reference_index >= 0, f"causal accepted_reference missing for ACK {receive_ns}")
        raw_reference = raw_references[reference_index]
        accepted_pose = _pose(payload.get("accepted_pose", {}))
        _require(
            _pose(raw_reference.get("pose", {})) == accepted_pose,
            f"causal accepted_reference pose does not equal ACK pose at {receive_ns}",
        )
        _require(
            float(raw_reference.get("target_gripper_width_m", -1.0))
            == float(authority["target_width_m"]),
            f"accepted_reference gripper width has no selected goal origin at {receive_ns}",
        )
        safe_matches = [
            row for row in safe_actions
            if int(row.get("payload", {}).get("equilibrium_source_stamp_ns", 0))
            == request_stamp_ns
        ]
        _require(len(safe_matches) == 1, f"unique safe_action lineage missing for ACK {receive_ns}")
        safe = safe_matches[0]
        safe_payload = safe.get("payload", {})
        arbitration = safe_payload.get("arbitration", {})
        raw_action = arbitration.get("raw_action", {})
        _require(
            arbitration.get("accepted") is True
            and safe_payload.get("equilibrium_published") is True
            and raw_action.get("source") == "human",
            f"human accepted action lineage missing for ACK {receive_ns}",
        )
        intervention = bool(raw_action.get("intervention"))
        _require(intervention, f"human intervention ownership missing for ACK {receive_ns}")
        workspace = safe_payload.get("workspace_clipped")
        _require(
            isinstance(workspace, list),
            f"workspace clip provenance missing for ACK {receive_ns}",
        )
        age_limits.add(float(safe_payload.get("action_step_gap_limit_ms", 0.0)))

        references.append({
            "accepted_receive_monotonic_ns": int(raw_reference["accepted_receive_monotonic_ns"]),
            "source_stamp_ns": int(raw_reference["source_stamp_ns"]),
            "frame_id": str(raw_reference["frame_id"]),
            "pose": accepted_pose,
            "target_gripper_width_m": float(raw_reference["target_gripper_width_m"]),
            "gripper_command_id": goal_id,
            "gripper_command_id_origin": "recorded_gripper_target_and_goal_status",
        })
        acknowledgements.append({
            "ack_id": recorded_ack_id(
                request_sequence=request_sequence, request_stamp_ns=request_stamp_ns,
            ),
            "ack_id_origin": "recorded_request_sequence_and_stamp",
            "receive_monotonic_ns": receive_ns,
            "request_sequence": request_sequence,
            "request_stamp_ns": request_stamp_ns,
            "controller_ack_monotonic_ns": controller_ack_ns,
            "action_decision_id": int(safe_payload["decision_id"]),
            "action_source_receive_monotonic_ns": int(safe["receive_monotonic_ns"]),
            "gripper_command_id": goal_id,
            "gripper_ack_command_id": goal_id,
            "slot_owner": "human_intervention",
            "accepted_action_source": "human",
            "intervention": True,
            "workspace_clipped": bool(any(workspace)),
            "payload": {"accepted": True, "accepted_pose": accepted_pose},
        })
    _require(len(references) >= 2, "selected K=3 macro has fewer than two recorded ACK events")
    _require(
        len(age_limits) == 1 and next(iter(age_limits)) > 0,
        "recorded action ACK age limit missing",
    )
    return references, acknowledgements, next(iter(age_limits))


def _observation(
    prepared: Any,
    episode: Path,
    *,
    role: str,
    global_index: int,
    local_index: int,
) -> dict[str, Any]:
    provenance = prepared.provenance
    try:
        external = prepared.camera1_paths[global_index].relative_to(episode).as_posix()
        wrist = prepared.camera2_paths[global_index].relative_to(episode).as_posix()
    except ValueError as error:
        raise RecordedAckExportError(f"{role} observation camera path outside episode") from error
    return {
        "role": role,
        "local_grid_index": local_index,
        "global_grid_index": global_index,
        "grid_monotonic_ns": int(prepared.tuple_host_ns[global_index]),
        "state7": prepared.state7[global_index].tolist(),
        "wrench6": prepared.wrench6[global_index].tolist(),
        "external_camera_relative_path": external,
        "wrist_camera_relative_path": wrist,
        "state_pose_source_stamp_ns": int(provenance["state_pose_source_stamp_ns"][global_index]),
        "camera1_receive_monotonic_ns": int(
            provenance["camera1_receive_monotonic_ns"][global_index]
        ),
        "camera2_receive_monotonic_ns": int(
            provenance["camera2_receive_monotonic_ns"][global_index]
        ),
        "gripper_source_stamp_ns": int(provenance["gripper_source_stamp_ns"][global_index]),
        "wrench_filter_output_stamp_ns": int(
            provenance["wrench_filter_output_stamp_ns"][global_index]
        ),
        "action_ack_receive_monotonic_ns": int(
            provenance["action_ack_receive_monotonic_ns"][global_index]
        ),
        "validity_bits": int(provenance["validity_bits"][global_index]),
    }


def export_recorded_ack_fixture(
    *,
    raw_session: Path = DEFAULT_RAW_SESSION,
    raw_episode: Path = DEFAULT_RAW_EPISODE,
    output: Path = DEFAULT_RECORDED_FIXTURE,
    capture_manifest_output: Path = DEFAULT_CAPTURE_MANIFEST,
    parent_binding_path: Path | None = DEFAULT_PARENT_BINDING,
    terminal_index_path: Path = DEFAULT_TERMINAL_INDEX,
) -> dict[str, Any]:
    raw_session = raw_session.resolve()
    raw_episode = raw_episode.resolve()
    output = output.resolve()
    capture_manifest_output = capture_manifest_output.resolve()
    _require(raw_session.is_dir(), f"raw session missing: {raw_session}")
    _require(raw_episode.is_dir(), f"raw episode missing: {raw_episode}")
    try:
        raw_episode.relative_to(raw_session)
    except ValueError as error:
        raise RecordedAckExportError("raw episode is outside raw session") from error

    session = _load_json(raw_session / "session.json", "native session manifest")
    result = _load_json(raw_episode / "episode_result.json", "native episode result")
    start_record = _load_json(raw_episode / "episode_start.json", "native episode start")
    _require(result.get("saved") is True, "episode_result.saved is not true")
    _require(result.get("fatal_reason") is None, "episode_result.fatal_reason is not null")
    _require(result.get("task") == session.get("task"), "episode/session task identity mismatch")
    _require(
        session.get("raw_format_version") == "fr3-hilserl-impedance-native-raw-v5",
        "native v5 format identity missing",
    )
    _require(
        session.get("tool_profile_name") == "onrobot_robotiq",
        "onrobot_robotiq tool profile identity missing",
    )

    if parent_binding_path is None:
        calibration_path = CANONICAL_MANIFEST_ROOT / "calibration_bundle.development.json"
        runtime_path = CANONICAL_MANIFEST_ROOT / "converter_runtime_spec.task2.development.json"
        normalizer_path = ROOT / "datasets/task2_lerobotv3/normalizer_manifest.json"
        action_contract_path = CANONICAL_MANIFEST_ROOT / "action_delta_spec.json"
    else:
        parent = _load_json(parent_binding_path.resolve(), "online bootstrap parent binding")
        try:
            calibration_path = Path(parent["calibration_binding"]["absolute_path"]).resolve()
            runtime_path = Path(parent["runtime_contract_binding"]["absolute_path"]).resolve()
            normalizer_path = Path(parent["normalizer_binding"]["absolute_path"]).resolve()
            action_contract_path = Path(parent["action_contract_binding"]["absolute_path"]).resolve()
        except (KeyError, TypeError, ValueError) as error:
            raise RecordedAckExportError("online bootstrap parent converter bindings missing") from error
    calibration = _load_json(calibration_path, "Stage-2 calibration bundle")
    prepared = prepare_episode(
        raw_episode,
        session=session,
        calibration_payload=calibration,
        contract=_runtime_contract(runtime_path),
    )
    _require(prepared.raw_episode_id == raw_episode.name, "prepared episode identity mismatch")
    selection = _derive_transition_selection(terminal_index_path.resolve(), raw_episode.name)
    start = selection["prepared_grid_start_index"]
    stop = selection["prepared_grid_stop_index_exclusive"]
    _require(
        0 <= start < stop <= len(prepared.tuple_host_ns),
        "selected terminal window outside prepared episode",
    )
    authority = _gripper_authority(
        raw_episode, established_before_ns=int(prepared.tuple_host_ns[start]),
    )
    selected_ack_times = prepared.provenance["action_ack_receive_monotonic_ns"][start:start + 3]
    references, acknowledgements, max_ack_age_ms = _accepted_ack_rows(
        raw_episode,
        selected_ack_times=[int(value) for value in selected_ack_times],
        authority=authority,
    )
    observation_indices = (
        ("current", 0),
        ("next", selection["next_observation_grid_index"]),
        ("terminal", selection["terminal_observation_grid_index"]),
    )
    observations = [
        _observation(
            prepared, raw_episode, role=role,
            global_index=start + local, local_index=local,
        )
        for role, local in observation_indices
    ]
    selection = {
        **selection,
        "observation_provenance": observations,
        "gripper_authority": authority,
    }

    capture_manifest = {
        "schema_version": "forcesmolvla_stage3_recorded_ack_capture_manifest.v1",
        "fixture_id": FIXTURE_ID,
        "fixture_kind": "recorded_live",
        "synthetic": False,
        "action_source": "human",
        "capture_origin": "historical_native_real",
        "raw_session_path": str(raw_session),
        "raw_episode_path": str(raw_episode),
        "native_raw_format_version": session["raw_format_version"],
        "episode_index": int(start_record["episode_index"]),
        "episode_saved": True,
        "episode_fatal_reason": None,
        "task": str(result["task"]),
        "selection": {
            key: selection[key]
            for key in (
                "prepared_grid_start_index", "prepared_grid_stop_index_exclusive",
                "current_observation_grid_index", "next_observation_grid_index",
                "terminal_observation_grid_index", "last_executable_grid_index",
                "full_macro_transition_index", "terminal_transition_index",
            )
        },
        "gripper_authority": authority,
        "ack_natural_identities": [
            {
                "ack_id": row["ack_id"],
                "request_sequence": row["request_sequence"],
                "request_stamp_ns": row["request_stamp_ns"],
                "receive_monotonic_ns": row["receive_monotonic_ns"],
            }
            for row in acknowledgements
        ],
        "observation_boundaries": observations,
    }
    manifest_encoded = json.dumps(
        capture_manifest, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n"
    manifest_sha256 = hashlib.sha256(manifest_encoded.encode("utf-8")).hexdigest()

    grid = np.asarray(prepared.tuple_host_ns[start:stop], dtype=np.int64)
    fixture = {
        "schema_version": "forcesmolvla_stage3_recorded_ack_fixture.v1",
        "fixture_id": FIXTURE_ID,
        "fixture_kind": "recorded_live",
        "synthetic": False,
        "action_source": "human",
        "capture_origin": "historical_native_real",
        "provenance": {
            "recorded_live_evidence": True,
            "raw_session_path": str(raw_session),
            "raw_episode_path": str(raw_episode),
            "raw_session_tree_sha256": directory_tree_sha256(raw_session),
            "capture_manifest_path": _fixture_path(capture_manifest_output),
            "capture_manifest_sha256": manifest_sha256,
        },
        "bindings": {
            "stage2_ack_converter": _binding(ROOT / "src/forcesmolvla/raw_to_lerobot_v3.py"),
            "stage2_temporal": _binding(ROOT / "src/forcesmolvla/temporal.py"),
            "action_delta": _binding(ROOT / "src/forcesmolvla/action_delta.py"),
            "normalizer_source": _binding(ROOT / "src/forcesmolvla/normalizer.py"),
            "normalizer_manifest": _binding(normalizer_path),
            "action_contract_v2": _binding(action_contract_path),
            "stage2_runtime_contract": _binding(runtime_path),
            "calibration_bundle": _binding(calibration_path),
            "terminal_transition_index": _binding(terminal_index_path),
        },
        "selection": selection,
        "temporal": {
            "session_start_ack_ns": int(grid[0]),
            "episode_end_ns": int(grid[-1]),
            "terminal_boundary_ns": int(grid[selection["last_executable_grid_index"]]),
            "data_grid_hz": 30,
            "policy_hz": 10,
            "K": 3,
            "macro_duration_ms": 100,
            "policy_anchor_phase_on_30hz_grid": 0,
            "max_ack_age_ms": max_ack_age_ms,
        },
        "accepted_references": references,
        "reference_acks": acknowledgements,
        "anchor_states": [
            {"grid_index": local, "state7": prepared.state7[start + local].tolist()}
            for local in (0, 3)
        ],
    }
    validated = validate_recorded_ack_fixture(fixture)
    output.parent.mkdir(parents=True, exist_ok=True)
    capture_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    capture_manifest_output.write_text(manifest_encoded, encoding="utf-8")
    output.write_text(
        json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return validated


__all__ = [
    "DEFAULT_CAPTURE_MANIFEST",
    "DEFAULT_PARENT_BINDING",
    "DEFAULT_RAW_EPISODE",
    "DEFAULT_RAW_SESSION",
    "DEFAULT_TERMINAL_INDEX",
    "RecordedAckExportError",
    "export_recorded_ack_fixture",
    "recorded_ack_id",
]
