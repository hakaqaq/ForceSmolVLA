from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from forcesmolvla.rft.online import production_bridge as bridge_module
from forcesmolvla.raw_to_lerobot_v3 import PreparedEpisode
from forcesmolvla.rft.detector_reward_transitions import (
    causal_detection_trace,
    detector_macro_transitions,
)
from forcesmolvla.rft.online.gripper_authority import GripperGeneration
from forcesmolvla.rft.online.policy_lineage import InitialGripperAuthority
from forcesmolvla.rft.online.production_bridge import (
    BridgeConfig,
    BridgeDigestCollisionError,
    EpisodeMaterialization,
    FrozenDetectorScores,
    InjectedBridgeCrash,
    ProductionBridgeError,
    ProductionBridge,
    frozen_episode_materializer,
    load_bridge_config,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/online_replay_production_bridge.v1.development.yaml"
REAL_EPISODE = Path(
    "/home/rlc123/fr3_client_ws/datasets/task2/episodes/episode_000018"
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def _offset_policy_epoch(value, offset: int):
    if isinstance(value, dict):
        return {
            key: (
                int(item) + offset
                if key == "policy_epoch"
                else _offset_policy_epoch(item, offset)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_offset_policy_epoch(item, offset) for item in value]
    return value


def _offset_episode_policy_epoch(episode: Path, offset: int) -> None:
    dataset = episode.parent.parent
    paths = [dataset / "integrated_capture_session.json"]
    paths.extend((dataset / "integrated_capture" / episode.name / "streams").glob("*.json*"))
    paths.extend((episode / "streams").glob("*.jsonl"))
    for path in paths:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            _write_jsonl(path, [_offset_policy_epoch(row, offset) for row in rows])
        else:
            _write_json(path, _offset_policy_epoch(json.loads(path.read_text()), offset))


def _pose() -> dict:
    return {
        "frame_id": "fr3_link0",
        "position_m": [0.5, 0.0, 0.2],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _fixture(tmp_path: Path, *, sealed: bool = True) -> Path:
    episode = tmp_path / "dataset" / "episodes" / "episode_000000"
    _write_json(
        episode / "episode_start.json",
        {"episode_index": 0, "task": "fixture", "started_monotonic_ns": 900_000_000},
    )
    if not sealed:
        return episode
    times = [1_000_000_000 + index * 100_000_000 for index in range(5)]
    phases = ["episode_start", "control", "control", "control", "episode_end"]
    raw_records, safe_records, requested, accepted, acks = [], [], [], [], []
    for sequence, (timestamp, phase) in enumerate(zip(times, phases, strict=True)):
        raw = {
            "schema": "fr3-hilserl-raw-action-v1",
            "source": "human",
            "sequence": sequence,
            "source_monotonic_ns": timestamp - 2_000_000,
            "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            "intervention": phase == "control",
            "phase": phase,
            "policy_epoch": 1,
            "observation_id": None,
        }
        stamp = 10_000 + sequence
        payload = {
            "schema": "fr3-hilserl-safe-action-v1",
            "accept_monotonic_ns": timestamp,
            "decision_id": sequence,
            "arbitration": {
                "accepted": True,
                "owner": "human" if phase == "control" else "none",
                "raw_action": raw,
            },
            "equilibrium_published": True,
            "equilibrium_source_stamp_ns": stamp,
            "requested_equilibrium": _pose(),
            "workspace_clipped": [False, False, False],
        }
        raw_records.append({"receive_monotonic_ns": timestamp, "payload": raw})
        safe_records.append({"receive_monotonic_ns": timestamp, "payload": payload})
        requested.append(
            {
                "source_stamp_ns": stamp,
                "equilibrium_publish_monotonic_ns": timestamp + 1_000_000,
                "receive_monotonic_ns": timestamp,
                "source": "human",
                "sequence": sequence,
                "pose": _pose(),
            }
        )
        accepted.append(
            {
                "source_stamp_ns": stamp,
                "accepted_receive_monotonic_ns": timestamp + 3_000_000,
                "pose": _pose(),
                "target_gripper_width_m": 0.085,
            }
        )
        acks.append(
            {
                "receive_monotonic_ns": timestamp + 5_000_000,
                "payload": {
                    "schema": "fr3-hilserl-reference-ack-v1",
                    "accepted": True,
                    "request_stamp_ns": stamp,
                    "request_sequence": 100 + sequence,
                    "request_receive_monotonic_ns": 9_000_000_000 + sequence,
                    "ack_monotonic_ns": 9_000_000_100 + sequence,
                    "accepted_pose": _pose(),
                },
            }
        )
    stream_values = {
        "raw_action": raw_records,
        "safe_action": safe_records,
        "requested_equilibrium": requested,
        "accepted_reference": accepted,
        "reference_ack": acks,
        "gripper_target": [
            {
                "receive_monotonic_ns": 1_106_000_000,
                "local_goal_sequence": 1,
                "action_goal_id": "real-goal-1",
                "requested_state": "OPEN",
                "started_monotonic_ns": 1_095_000_000,
                "accepted_monotonic_ns": 1_106_000_000,
                "target_width_m": 0.085,
            }
        ],
        "gripper_goal_status": [
            {
                "receive_monotonic_ns": 1_251_000_000,
                "local_goal_sequence": 1,
                "action_goal_id": "real-goal-1",
                "accepted_monotonic_ns": 1_106_000_000,
                "finished_monotonic_ns": 1_250_000_000,
                "outcome": "reached",
            }
        ],
    }
    sensor_times = [timestamp - 1_000_000 for timestamp in times]
    stream_values["measured_tcp_pose"] = [
        {"receive_monotonic_ns": timestamp, "source_stamp_ns": timestamp, "pose": _pose()}
        for timestamp in sensor_times
    ]
    stream_values["wrench_notch_sensor"] = [
        {
            "receive_monotonic_ns": timestamp,
            "source_stamp_ns": timestamp,
            "force_xyz_n_torque_xyz_nm": [0.0] * 6,
        }
        for timestamp in sensor_times
    ]
    stream_values["gripper_state"] = [
        {
            "receive_monotonic_ns": timestamp,
            "source_stamp_ns": timestamp,
            "width_m": 0.085,
        }
        for timestamp in sensor_times
    ]
    for role in ("external", "wrist"):
        camera = []
        for index, timestamp in enumerate(sensor_times):
            relative = f"images/{role}/frame_{index:06d}.jpg"
            blob = episode / relative
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(f"{role}-{index}".encode())
            camera.append(
                {
                    "receive_monotonic_ns": timestamp,
                    "timestamp_domain": "host_monotonic_receive",
                    "rgb_path": relative,
                }
            )
        stream_values[f"{role}_camera"] = camera
    streams = episode / "streams"
    for name, values in stream_values.items():
        _write_jsonl(streams / f"{name}.jsonl", values)
    result = {
        "episode_index": 0,
        "task": "fixture",
        "saved": True,
        "fatal_reason": None,
        "reward": 1.0,
        "terminated": True,
        "stream_counts": {name: len(values) for name, values in stream_values.items()},
    }
    _write_json(episode / "episode_result.json", result)
    return episode


def _integrated_shadow_fixture(episode: Path) -> None:
    dataset = episode.parent.parent
    revision = "e24c1d6bb0a778921659514ac47c692b952178aa39af2601ccf0fc32bf94774d"
    identity = {
        "session_id": "stage3-shadow-fixture",
        "episode_id": episode.name,
        "clock_domain_id": "upper_host_monotonic",
        "policy_revision": revision,
        "policy_epoch": 0,
        "reset_generation": 0,
        "takeover_generation": 0,
    }
    _write_json(
        dataset / "session.json",
        {
            "primary_alignment_clock": "upper_host_receive_monotonic_ns",
            "tool_config_hash": "tool-profile-fixture",
        },
    )
    _write_json(
        dataset / "integrated_capture_session.json",
        {
            "schema": "forcesmolvla-stage3-integrated-shadow-backend-v1",
            "contract": {
                "schema": "forcesmolvla-stage3-integrated-capture-v1",
                "mode": "shadow",
                "identity": identity,
                "actual_action_source": "human",
                "policy_inference": True,
                "policy_execution": False,
                "formal_replay": False,
                "real_online_r": False,
                "controller_owner": "recorder",
                "controller_process_count": 1,
                "recorder_controller": True,
                "deploy_controller": False,
            },
            "clock_binding": {
                "native_primary_alignment_clock": "upper_host_receive_monotonic_ns",
                "policy_request_clock_domain_id": "upper_host_monotonic_ns",
                "same_upper_host_monotonic_epoch": True,
                "stage3_clock_domain_id": "upper_host_monotonic",
            },
            "controller_owner": "recorder",
            "controller_process_count": 1,
            "deploy_controller_started": False,
            "policy_action_publisher_created": False,
            "policy_metadata": {
                "model_sha256": revision,
                "dataset_repo_id": "local/task2_lerobotv3",
                "tool_profile_sha256": "tool-profile-fixture",
                "calibration_id": "calibration-fixture",
            },
        },
    )
    stream_root = dataset / "integrated_capture" / episode.name / "streams"
    native_streams = episode / "streams"
    native_pose = json.loads(
        (native_streams / "measured_tcp_pose.jsonl").read_text().splitlines()[1]
    )
    native_wrench = json.loads(
        (native_streams / "wrench_notch_sensor.jsonl").read_text().splitlines()[1]
    )
    native_gripper = json.loads(
        (native_streams / "gripper_state.jsonl").read_text().splitlines()[1]
    )
    t_ref_ns = 1_101_000_000
    observation_id = f"{episode.name}:observation:000000"
    observation = {
        "schema": "forcesmolvla-stage3-integrated-capture-v1",
        **identity,
        "observation_id": observation_id,
        "t_ref_ns": t_ref_ns,
        "stream_timestamps_ns": {
            "measured_tcp_pose": native_pose["receive_monotonic_ns"] + 100_000,
            "wrench_notch_sensor": native_wrench["receive_monotonic_ns"] + 100_000,
            "gripper_state": native_gripper["receive_monotonic_ns"] + 100_000,
            "external_camera": 1_100_000_000,
            "wrist_camera": 1_100_000_000,
        },
        "stream_ids": {
            "measured_tcp_pose": (
                f"source:{native_pose['source_stamp_ns']}@receive:"
                f"{native_pose['receive_monotonic_ns'] + 100_000}"
            ),
            "wrench_notch_sensor": (
                f"source:{native_wrench['source_stamp_ns']}@receive:"
                f"{native_wrench['receive_monotonic_ns'] + 100_000}"
            ),
            "gripper_state": (
                f"source:{native_gripper['source_stamp_ns']}@receive:"
                f"{native_gripper['receive_monotonic_ns'] + 100_000}"
            ),
            "external_camera": "images/external/frame_000001.jpg",
            "wrist_camera": "images/wrist/frame_000001.jpg",
        },
    }
    request = {
        "schema": "forcesmolvla-stage3-policy-lineage-v1",
        **identity,
        "observation_id": observation_id,
        "request_id": "request-1",
        "chunk_id": "live-request-1",
        "proposal_id": "policy-proposal:request-1",
        "t_ref_ns": t_ref_ns,
        "request_clock_domain_id": "upper_host_monotonic_ns",
        "request_recorded_monotonic_ns": 1_102_000_000,
    }
    result = {
        **request,
        "lineage_schema": "forcesmolvla-stage3-policy-lineage-v1",
        "result_id": "policy-result:request-1",
        "result_recorded_monotonic_ns": 1_103_000_000,
        "shadow_proposal": True,
        "executed": False,
    }
    proposal = {
        **result,
        "schema": "forcesmolvla-stage3-integrated-shadow-backend-v1",
        "actual_action_source": "human",
        "policy_inference": True,
        "policy_execution": False,
        "formal_replay": False,
        "real_online_r": False,
        "action_semantics": "absolute7",
        "valid_horizon": 1,
        "actions_absolute7": [[0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 0.085]],
    }
    _write_jsonl(stream_root / "policy_shadow_observation.jsonl", [observation])
    _write_jsonl(stream_root / "policy_shadow_request.jsonl", [request])
    _write_jsonl(stream_root / "policy_shadow_result.jsonl", [result])
    _write_jsonl(stream_root / "policy_shadow_proposal.jsonl", [proposal])

    safe_rows = [
        json.loads(line)
        for line in (native_streams / "safe_action.jsonl").read_text().splitlines()
    ]
    ack_rows = [
        json.loads(line)
        for line in (native_streams / "reference_ack.jsonl").read_text().splitlines()
    ]
    human_acks = []
    observed_ack_count = 0
    for safe, ack in zip(safe_rows, ack_rows, strict=True):
        receive_ns = int(ack["receive_monotonic_ns"])
        ack_observation = observation_id if receive_ns >= t_ref_ns else None
        observed_ack_count += ack_observation is not None
        stamp = int(ack["payload"]["request_stamp_ns"])
        human_acks.append(
            {
                "schema": "forcesmolvla-stage3-integrated-shadow-backend-v1",
                **identity,
                "ack_id": f"human-ack:{stamp}",
                "observation_id": ack_observation,
                "receive_monotonic_ns": receive_ns,
                "actual_action_source": "human",
                "policy_result_id": None,
                "proposal_id": None,
                "policy_executed_transition": False,
                "policy_execution": False,
                "formal_replay": False,
                "real_online_r": False,
                "safe_action": safe["payload"],
                "reference_ack": ack["payload"],
            }
        )
    _write_jsonl(stream_root / "policy_shadow_human_ack.jsonl", human_acks)

    generation = GripperGeneration(
        episode_id=episode.name,
        reset_generation=0,
        takeover_generation=0,
        policy_revision=revision,
        policy_epoch=0,
    )
    lease = InitialGripperAuthority(
        episode_id=episode.name,
        origin_local_goal_sequence=9,
        origin_action_goal_id="integrated-startup-open",
        origin_accepted_monotonic_ns=800_000_000,
        requested_state="OPEN",
        requested_width_m=0.085,
        terminal_outcome="reached",
        terminal_finished_monotonic_ns=850_000_000,
        feedback_width_m=0.085,
        feedback_state="OPEN",
        feedback_monotonic_ns=990_000_000,
        captured_monotonic_ns=995_000_000,
        feedback_age_ns=5_000_000,
        clock_domain_id="upper_host_monotonic",
        generation=generation,
    ).validate(max_feedback_age_ns=100_000_000).to_dict()
    _write_json(stream_root / "policy_shadow_initial_gripper_lease.json", lease)
    camera_records = []
    for role in ("external", "wrist"):
        camera_records.append(
            {
                "clock_domain_id": "upper_host_monotonic",
                "native_receive_monotonic_ns": 1_099_000_000,
                "observation_id": observation_id,
                "policy_receive_monotonic_ns": 1_100_000_000,
                "rgb_path": f"images/{role}/frame_000001.jpg",
                "role": role,
                "same_recorder_jpeg": True,
            }
        )
    _write_json(
        stream_root / "policy_shadow_camera_reconciliation.json",
        {
            "schema": "forcesmolvla-stage3-integrated-shadow-backend-v1",
            "native_episode": str(episode),
            "records": camera_records,
        },
    )
    native_result = json.loads((episode / "episode_result.json").read_text())
    _write_json(
        stream_root / "policy_shadow_episode_seal.json",
        {
            "schema": "forcesmolvla-stage3-integrated-capture-v1",
            **identity,
            "backend_schema": "forcesmolvla-stage3-integrated-shadow-backend-v1",
            "seal_id": "shadow-seal:fixture",
            "sealed_monotonic_ns": 1_500_000_000,
            "terminal_observation_id": observation_id,
            "observation_count": 1,
            "policy_request_count": 1,
            "policy_result_count": 1,
            "human_action_ack_count": observed_ack_count,
            "actual_action_source": "human",
            "policy_inference": True,
            "policy_execution": False,
            "formal_replay": False,
            "real_online_r": False,
            "shadow_proposals_executed": False,
            "controller_owner": "recorder",
            "controller_process_count": 1,
            "deploy_controller_started": False,
            "policy_action_publisher_created": False,
            "native_episode": str(episode),
            "native_episode_result": native_result,
            "initial_gripper_lease": lease,
            "camera_records_reconciled": 2,
        },
    )


def _integrated_policy_execution_fixture(episode: Path) -> None:
    dataset = episode.parent.parent
    native_root = episode / "streams"
    revision = "e24c1d6bb0a778921659514ac47c692b952178aa39af2601ccf0fc32bf94774d"
    identity = {
        "session_id": "stage3-policy-execute-fixture",
        "episode_id": episode.name,
        "clock_domain_id": "upper_host_monotonic",
        "policy_revision": revision,
        "policy_epoch": 0,
        "reset_generation": 0,
        "takeover_generation": 0,
    }
    _write_json(
        dataset / "session.json",
        {
            "primary_alignment_clock": "upper_host_receive_monotonic_ns",
            "tool_config_hash": "tool-profile-fixture",
            "force_status": (
                "notch-filtered measurement-frame wrench retained; calibration, "
                "gravity compensation and TCP moment shift applied offline"
            ),
            "cameras": {
                "observation.image": {"role": "external", "model": "D435"},
                "observation.wrist_image": {"role": "wrist", "model": "D405"},
            },
        },
    )
    _write_json(
        dataset / "integrated_capture_session.json",
        {
            "schema": "forcesmolvla-stage3-integrated-policy-execution-backend-v1",
            "contract": {
                "schema": "forcesmolvla-stage3-integrated-capture-v1",
                "mode": "policy-execute",
                "identity": identity,
                "actual_action_source": "policy",
                "policy_inference": True,
                "policy_execution": True,
                "formal_replay": False,
                "real_online_r": False,
                "development_policy_execution_smoke": True,
                "controller_owner": "recorder",
                "controller_process_count": 1,
                "recorder_controller": True,
                "deploy_controller": False,
            },
            "clock_binding": {
                "native_primary_alignment_clock": "upper_host_receive_monotonic_ns",
                "policy_request_clock_domain_id": "upper_host_monotonic_ns",
                "same_upper_host_monotonic_epoch": True,
                "stage3_clock_domain_id": "upper_host_monotonic",
            },
            "controller_owner": "recorder",
            "controller_process_count": 1,
            "deploy_controller_started": False,
            "formal_replay_writer_started": False,
            "learner_started": False,
            "policy_action_publisher_created": True,
            "policy_revision_publisher_started": False,
            "policy_metadata": {
                "model_sha256": revision,
                "dataset_repo_id": "local/task2_lerobotv3",
                "tool_profile_sha256": "tool-profile-fixture",
                "calibration_id": "calibration-fixture",
            },
        },
    )

    raw_rows = [json.loads(line) for line in (native_root / "raw_action.jsonl").read_text().splitlines()]
    safe_rows = [json.loads(line) for line in (native_root / "safe_action.jsonl").read_text().splitlines()]
    requested_rows = [
        json.loads(line)
        for line in (native_root / "requested_equilibrium.jsonl").read_text().splitlines()
    ]
    accepted_rows = [
        json.loads(line)
        for line in (native_root / "accepted_reference.jsonl").read_text().splitlines()
    ]
    ack_rows = [json.loads(line) for line in (native_root / "reference_ack.jsonl").read_text().splitlines()]

    lineage_by_sequence = {
        1: {
            "request_id": "request-a",
            "result_id": "policy-result:request-a",
            "chunk_id": "live-request-a",
            "proposal_id": "policy-proposal:request-a",
            "policy_revision": revision,
            "policy_epoch": 0,
            "reset_generation": 0,
            "takeover_generation": 0,
            "t_ref_ns": 1_101_000_000,
        },
        3: {
            "request_id": "request-b",
            "result_id": "policy-result:request-b",
            "chunk_id": "live-request-b",
            "proposal_id": "policy-proposal:request-b",
            "policy_revision": revision,
            "policy_epoch": 1,
            "reset_generation": 0,
            "takeover_generation": 1,
            "t_ref_ns": 1_301_000_000,
        },
    }
    selected_action7 = [0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 0.085]
    for sequence, lineage in lineage_by_sequence.items():
        raw = raw_rows[sequence]["payload"]
        raw.update(
            {
                "source": "policy",
                "policy_epoch": lineage["policy_epoch"],
                "observation_id": (
                    f"{episode.name}:observation:{0 if sequence == 1 else 2:06d}"
                ),
                "intervention": False,
            }
        )
        safe = safe_rows[sequence]["payload"]
        safe["arbitration"].update(
            {
                "accepted": True,
                "event": "policy_action",
                "owner": "policy",
                "policy_epoch": lineage["policy_epoch"],
                "reason": "accepted_policy",
                "raw_action": raw,
            }
        )
        safe["forcesmolvla_chunk_selection"] = {
            **lineage,
            "lineage_schema": "forcesmolvla-stage3-policy-lineage-v1",
            "request_clock_domain_id": "upper_host_monotonic_ns",
            "request_recorded_monotonic_ns": lineage["t_ref_ns"] + 1_000_000,
            "result_recorded_monotonic_ns": lineage["t_ref_ns"] + 2_000_000,
            "action_index": 0,
            "sequence": sequence,
            "normalized_action7": [0.0] * 7,
            "selected_post_adapter_absolute7": selected_action7,
        }
        requested_rows[sequence]["source"] = "policy"
        ack_rows[sequence]["payload"]["request_frame_id"] = "fr3_link0"

    takeover_raw = raw_rows[2]["payload"]
    takeover_raw.update({"intervention": True, "policy_epoch": 0})
    takeover_safe = safe_rows[2]["payload"]
    takeover_safe["arbitration"].update(
        {
            "accepted": True,
            "event": "intervention_start",
            "owner": "human",
            "policy_epoch": 1,
            "reason": "accepted_human",
            "raw_action": takeover_raw,
        }
    )
    override_raw = {
        **deepcopy(raw_rows[1]["payload"]),
        "source": "policy",
        "sequence": 99,
        "policy_epoch": 0,
        "observation_id": f"{episode.name}:observation:000001",
    }
    override_safe = {
        **deepcopy(safe_rows[1]["payload"]),
        "decision_id": 99,
        "arbitration": {
            "accepted": False,
            "event": "rejected",
            "owner": "human",
            "policy_epoch": 1,
            "reason": "human_override",
            "raw_action": override_raw,
        },
        "equilibrium_published": False,
        "equilibrium_source_stamp_ns": None,
        "requested_equilibrium": None,
        "reject_reason": "human_override",
    }
    release_raw = {
        **deepcopy(raw_rows[2]["payload"]),
        "sequence": 50,
        "source_monotonic_ns": 1_248_000_000,
        "intervention": False,
        "phase": "release",
        "policy_epoch": 1,
    }
    release_safe = {
        **deepcopy(safe_rows[2]["payload"]),
        "accept_monotonic_ns": 1_250_000_000,
        "decision_id": 50,
        "arbitration": {
            "accepted": True,
            "event": "intervention_end",
            "owner": "none",
            "policy_epoch": 1,
            "reason": "accepted_human_hold",
            "raw_action": release_raw,
        },
        "equilibrium_published": False,
        "equilibrium_source_stamp_ns": None,
        "requested_equilibrium": None,
    }
    raw_rows.extend(
        [
            {"receive_monotonic_ns": 1_201_000_000, "payload": override_raw},
            {"receive_monotonic_ns": 1_250_000_000, "payload": release_raw},
        ]
    )
    safe_rows.extend(
        [
            {"receive_monotonic_ns": 1_201_000_000, "payload": override_safe},
            {"receive_monotonic_ns": 1_250_000_000, "payload": release_safe},
        ]
    )
    _write_jsonl(native_root / "raw_action.jsonl", raw_rows)
    _write_jsonl(native_root / "safe_action.jsonl", safe_rows)
    _write_jsonl(native_root / "requested_equilibrium.jsonl", requested_rows)
    _write_jsonl(native_root / "reference_ack.jsonl", ack_rows)

    native_result = json.loads((episode / "episode_result.json").read_text())
    native_result["stream_counts"]["raw_action"] = len(raw_rows)
    native_result["stream_counts"]["safe_action"] = len(safe_rows)
    _write_json(episode / "episode_result.json", native_result)

    stream_root = dataset / "integrated_capture" / episode.name / "streams"
    sensor_rows = {
        name: [json.loads(line) for line in (native_root / f"{name}.jsonl").read_text().splitlines()]
        for name in ("measured_tcp_pose", "wrench_notch_sensor", "gripper_state")
    }
    camera_rows = {
        role: [json.loads(line) for line in (native_root / f"{role}_camera.jsonl").read_text().splitlines()]
        for role in ("external", "wrist")
    }
    observations = []
    observation_indices = (1, 2, 3, 4)
    for index, native_index in enumerate(observation_indices):
        generation = 0 if index < 2 else 1
        t_ref_ns = 1_101_000_000 + index * 100_000_000
        observations.append(
            {
                "schema": "forcesmolvla-stage3-integrated-capture-v1",
                **identity,
                "policy_epoch": generation,
                "takeover_generation": generation,
                "observation_id": f"{episode.name}:observation:{index:06d}",
                "t_ref_ns": t_ref_ns,
                "stream_timestamps_ns": {
                    **{
                        name: sensor_rows[name][native_index]["receive_monotonic_ns"]
                        for name in sensor_rows
                    },
                    **{
                        f"{role}_camera": camera_rows[role][native_index]["receive_monotonic_ns"]
                        for role in camera_rows
                    },
                },
                "stream_ids": {
                    **{
                        name: (
                            f"source:{sensor_rows[name][native_index]['source_stamp_ns']}"
                            f"@receive:{sensor_rows[name][native_index]['receive_monotonic_ns']}"
                        )
                        for name in sensor_rows
                    },
                    **{
                        f"{role}_camera": camera_rows[role][native_index]["rgb_path"]
                        for role in camera_rows
                    },
                },
            }
        )
    _write_jsonl(stream_root / "policy_execute_observation.jsonl", observations)

    requests, results, proposals, chunks = [], [], [], []
    for sequence, lineage in lineage_by_sequence.items():
        observation_index = 0 if sequence == 1 else 2
        request = {
            "schema": "forcesmolvla-stage3-policy-lineage-v1",
            **identity,
            **lineage,
            "observation_id": f"{episode.name}:observation:{observation_index:06d}",
            "request_clock_domain_id": "upper_host_monotonic_ns",
            "request_recorded_monotonic_ns": lineage["t_ref_ns"] + 1_000_000,
        }
        result = {
            **request,
            "result_id": lineage["result_id"],
            "lineage_schema": "forcesmolvla-stage3-policy-lineage-v1",
            "result_recorded_monotonic_ns": lineage["t_ref_ns"] + 2_000_000,
            "policy_execution_candidate": True,
            "shadow_proposal": False,
            "executed": False,
        }
        proposal = {
            **result,
            "schema": "forcesmolvla-stage3-integrated-policy-execution-backend-v1",
            "actual_action_source": "policy",
            "policy_inference": True,
            "policy_execution": True,
            "formal_replay": False,
            "real_online_r": False,
            "invalidated_by_takeover": False,
            "action_semantics": "absolute7",
            "valid_horizon": 1,
            "actions_absolute7": [selected_action7],
        }
        chunk = {
            **result,
            "schema": "forcesmolvla-stage3-integrated-policy-execution-backend-v1",
            "executed_action_source": "policy",
            "formal_replay": False,
            "real_online_r": False,
            "action_semantics": "absolute7",
            "valid_horizon": 1,
            "actions_absolute7": [selected_action7],
        }
        requests.append(request)
        results.append(result)
        proposals.append(proposal)
        chunks.append(chunk)
    _write_jsonl(stream_root / "policy_execute_request.jsonl", requests)
    _write_jsonl(stream_root / "policy_execute_result.jsonl", results)
    _write_jsonl(stream_root / "policy_execute_proposal.jsonl", proposals)
    _write_jsonl(stream_root / "policy_execute_chunk.jsonl", chunks)

    gripper_authorities, transitions = [], []
    for offset, sequence in enumerate((1, 3)):
        lineage = lineage_by_sequence[sequence]
        observation_index = offset * 2
        selection = safe_rows[sequence]["payload"]["forcesmolvla_chunk_selection"]
        feedback = sensor_rows["gripper_state"][observation_indices[observation_index]]
        gripper = {
            **lineage,
            "lineage_schema": "forcesmolvla-stage3-policy-lineage-v1",
            "request_clock_domain_id": "upper_host_monotonic_ns",
            "request_recorded_monotonic_ns": lineage["t_ref_ns"] + 1_000_000,
            "result_recorded_monotonic_ns": lineage["t_ref_ns"] + 2_000_000,
            "sequence": sequence,
            "action_index": 0,
            "command_required": False,
            "requested_state": "OPEN",
            "requested_width_m": 0.085,
            "feedback_width_m": 0.085,
            "feedback_monotonic_ns": feedback["receive_monotonic_ns"],
            "authority": "existing_accepted_gripper_state",
        }
        if sequence == 3:
            target_path = native_root / "gripper_target.jsonl"
            status_path = native_root / "gripper_goal_status.jsonl"
            target = json.loads(target_path.read_text().splitlines()[0])
            status = json.loads(status_path.read_text().splitlines()[0])
            target.update(
                {
                    "started_monotonic_ns": 1_305_000_000,
                    "accepted_monotonic_ns": 1_306_000_000,
                    "receive_monotonic_ns": 1_306_100_000,
                }
            )
            status.update(
                {
                    "accepted_monotonic_ns": 1_306_000_000,
                    "finished_monotonic_ns": 1_320_000_000,
                    "receive_monotonic_ns": 1_320_100_000,
                }
            )
            _write_jsonl(target_path, [target])
            _write_jsonl(status_path, [status])
            gripper.update(
                {
                    "command_required": True,
                    "authority": "policy_execution_backend",
                    "command_id": "policy-gripper:1",
                    "local_goal_sequence": 1,
                    "action_goal_id": "real-goal-1",
                    "started_monotonic_ns": 1_305_000_000,
                    "accepted_monotonic_ns": 1_306_000_000,
                    "finished_monotonic_ns": 1_320_000_000,
                    "outcome": "reached",
                }
            )
            gripper.pop("feedback_width_m")
            gripper.pop("feedback_monotonic_ns")
        native_safe = safe_rows[sequence]["payload"]
        native_ack = ack_rows[sequence]
        integrated_ack_receive_ns = native_ack["receive_monotonic_ns"] - 100_000
        integrated_pose_ack = {
            **native_ack["payload"],
            "upper_receive_monotonic_ns": integrated_ack_receive_ns,
        }
        transition = {
            "schema": "forcesmolvla-stage3-integrated-policy-execution-backend-v1",
            **identity,
            **lineage,
            "lineage_schema": "forcesmolvla-stage3-policy-lineage-v1",
            "request_clock_domain_id": "upper_host_monotonic_ns",
            "request_recorded_monotonic_ns": lineage["t_ref_ns"] + 1_000_000,
            "result_recorded_monotonic_ns": lineage["t_ref_ns"] + 2_000_000,
            "observation_id": f"{episode.name}:observation:{observation_index:06d}",
            "current_observation_id": f"{episode.name}:observation:{observation_index:06d}",
            "next_observation_id": f"{episode.name}:observation:{observation_index + 1:06d}",
            "ack_id": (
                f"policy-ack:{sequence}:"
                f"{native_safe['equilibrium_source_stamp_ns']}"
            ),
            "receive_monotonic_ns": integrated_ack_receive_ns,
            "actual_action_source": "policy",
            "executed_action_source": "policy",
            "policy_executed_transition": True,
            "intervention": False,
            "formal_replay": False,
            "real_online_r": False,
            "policy_result_id": lineage["result_id"],
            "accepted_absolute7": selected_action7,
            "selection": selection,
            "safety_arbitration": native_safe["arbitration"],
            "pose_command": {
                "position_m": requested_rows[sequence]["pose"]["position_m"],
                "quaternion_xyzw": requested_rows[sequence]["pose"][
                    "quaternion_xyzw"
                ],
            },
            "pose_ack": integrated_pose_ack,
            "gripper_authority": gripper,
        }
        gripper_authorities.append(gripper)
        transitions.append(transition)
    _write_jsonl(stream_root / "policy_execute_gripper_authority.jsonl", gripper_authorities)
    _write_jsonl(stream_root / "policy_execute_transition.jsonl", transitions)

    intervention_rows = [
        {
            "schema": "forcesmolvla-stage3-integrated-capture-v1",
            **identity,
            "policy_epoch": 1,
            "takeover_generation": 1,
            "receive_monotonic_ns": 1_202_000_000,
            "event": "intervention_start",
            "actual_action_source": "human",
            "policy_execution": True,
            "invalidated_chunk_id": "live-request-a",
            "old_policy_chunk_invalidated": True,
            "safe_action": takeover_safe,
        },
        {
            "schema": "forcesmolvla-stage3-integrated-capture-v1",
            **identity,
            "policy_epoch": 1,
            "takeover_generation": 1,
            "receive_monotonic_ns": 1_251_000_000,
            "event": "intervention_end",
            "actual_action_source": "human",
            "policy_execution": True,
            "invalidated_chunk_id": None,
            "old_policy_chunk_invalidated": False,
            "safe_action": release_safe,
        },
    ]
    _write_jsonl(stream_root / "policy_execute_intervention.jsonl", intervention_rows)

    lease = InitialGripperAuthority(
        episode_id=episode.name,
        origin_local_goal_sequence=9,
        origin_action_goal_id="integrated-startup-open",
        origin_accepted_monotonic_ns=800_000_000,
        requested_state="OPEN",
        requested_width_m=0.085,
        terminal_outcome="reached",
        terminal_finished_monotonic_ns=850_000_000,
        feedback_width_m=0.085,
        feedback_state="OPEN",
        feedback_monotonic_ns=990_000_000,
        captured_monotonic_ns=995_000_000,
        feedback_age_ns=5_000_000,
        clock_domain_id="upper_host_monotonic",
        generation=GripperGeneration(
            episode_id=episode.name,
            reset_generation=0,
            takeover_generation=0,
            policy_revision=revision,
            policy_epoch=0,
        ),
    ).validate(max_feedback_age_ns=100_000_000).to_dict()
    _write_json(stream_root / "policy_execute_initial_gripper_lease.json", lease)
    camera_records = []
    for observation, native_index in zip(observations, observation_indices, strict=True):
        for role in ("external", "wrist"):
            native_camera = camera_rows[role][native_index]
            camera_records.append(
                {
                    "clock_domain_id": "upper_host_monotonic",
                    "native_receive_monotonic_ns": native_camera["receive_monotonic_ns"],
                    "observation_id": observation["observation_id"],
                    "policy_receive_monotonic_ns": native_camera["receive_monotonic_ns"],
                    "rgb_path": native_camera["rgb_path"],
                    "role": role,
                    "same_recorder_jpeg": True,
                }
            )
    _write_json(
        stream_root / "policy_execute_camera_reconciliation.json",
        {
            "schema": "forcesmolvla-stage3-integrated-policy-execution-backend-v1",
            "native_episode": str(episode),
            "records": camera_records,
        },
    )
    _write_json(
        stream_root / "policy_execute_episode_seal.json",
        {
            "schema": "forcesmolvla-stage3-integrated-capture-v1",
            **identity,
            "backend_schema": "forcesmolvla-stage3-integrated-policy-execution-backend-v1",
            "technical_seal": "complete",
            "seal_id": "policy-execute-seal:fixture",
            "sealed_monotonic_ns": 1_600_000_000,
            "terminal_observation_id": observations[-1]["observation_id"],
            "observation_count": len(observations),
            "policy_request_count": len(requests),
            "policy_result_count": len(results),
            "policy_chunk_count": len(chunks),
            "policy_action_ack_count": len(transitions),
            "human_action_ack_count": 0,
            "intervention_count": len(intervention_rows),
            "camera_records_reconciled": len(camera_records),
            "actual_action_source": "policy",
            "executed_action_source": "policy",
            "policy_inference": True,
            "policy_execution": True,
            "formal_replay": False,
            "real_online_r": False,
            "formal_training_replay_written": False,
            "learner_started": False,
            "actor_updates": 0,
            "critic_updates": 0,
            "checkpoint_written": False,
            "policy_revision_published": False,
            "controller_owner": "recorder",
            "controller_process_count": 1,
            "deploy_controller_started": False,
            "policy_action_publisher_created": True,
            "detector_approval_scope": "single_episode_cycle210_policy_execution_smoke",
            "native_episode": str(episode),
            "native_episode_result": native_result,
            "initial_gripper_lease": lease,
        },
    )


def _add_canceled_policy_request(
    episode: Path, *, completed_request_count: int = 0
) -> str:
    stream_root = (
        episode.parent.parent
        / "integrated_capture"
        / episode.name
        / "streams"
    )
    paths = {
        name: stream_root / f"policy_execute_{name}.jsonl"
        for name in ("request", "result", "proposal", "chunk")
    }
    rows = {
        name: [json.loads(line) for line in path.read_text().splitlines()]
        for name, path in paths.items()
    }

    def rebound(row: dict, request_id: str) -> dict:
        value = deepcopy(row)
        value.update(
            {
                "request_id": request_id,
                "result_id": f"policy-result:{request_id}",
                "chunk_id": f"live-{request_id}",
                "proposal_id": f"policy-proposal:{request_id}",
            }
        )
        if value.get("schema") == "forcesmolvla-stage3-policy-lineage-v1" and (
            "result_recorded_monotonic_ns" not in value
        ):
            value.pop("result_id")
        return value

    for index in range(completed_request_count):
        request_id = f"completed-extra-{index:03d}"
        for name in rows:
            rows[name].append(rebound(rows[name][-1], request_id))

    canceled_request_id = "canceled-request"
    canceled_request = rebound(rows["request"][-1], canceled_request_id)
    rows["request"].append(canceled_request)
    for name, path in paths.items():
        _write_jsonl(path, rows[name])
    _write_jsonl(
        stream_root / "policy_execute_request_canceled.jsonl",
        [
            {
                **canceled_request,
                "canceled_monotonic_ns": 1_500_000_000,
                "cancel_reason": "episode_sealed_before_inference_result",
                "executed": False,
            }
        ],
    )
    seal_path = stream_root / "policy_execute_episode_seal.json"
    seal = json.loads(seal_path.read_text())
    seal.update(
        {
            "policy_request_count": len(rows["request"]),
            "policy_result_count": len(rows["result"]),
            "policy_chunk_count": len(rows["chunk"]),
            "policy_request_canceled_count": 1,
        }
    )
    _write_json(seal_path, seal)
    return canceled_request_id


def _make_policy_execution_fixture_async(episode: Path) -> Path:
    dataset = episode.parent.parent
    manifest_path = dataset / "integrated_capture_session.json"
    manifest = json.loads(manifest_path.read_text())
    resume = "/tmp/offline_actor_critic_cycle_000210"
    active = "offline-actor-critic-cycle-000210"
    manifest.update(
        {
            "learner_started": True,
            "learner_resume_checkpoint": resume,
            "active_actor_revision": active,
            "pending_candidate_id": None,
        }
    )
    _write_json(manifest_path, manifest)

    seal_path = (
        dataset
        / "integrated_capture"
        / episode.name
        / "streams/policy_execute_episode_seal.json"
    )
    seal = json.loads(seal_path.read_text())
    seal.update(
        {
            "learner_started": True,
            "learner_resume_checkpoint": resume,
            "active_actor_revision": active,
            "active_actor_model_revision": manifest["contract"]["identity"][
                "policy_revision"
            ],
            "learner_critic_steps": 0,
            "learner_actor_steps": 0,
            "critic_updates": 0,
            "actor_updates": 0,
            "current_episode_sampled_by_learner": False,
            "online_checkpoint_path": None,
            "actor_parameter_broadcast_count": 0,
        }
    )
    _write_json(seal_path, seal)
    return seal_path


def _fake_materialization(episode: Path, *, trigger_frame: int = 9) -> EpisodeMaterialization:
    count = max(10, trigger_frame + 1)
    grid = np.asarray(
        [1_107_000_000 + (frame * 1_000_000_000) // 30 for frame in range(count)],
        dtype=np.int64,
    )
    state7 = np.tile(
        np.asarray([0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 0.085], dtype=np.float32),
        (count, 1),
    )
    wrench6 = np.tile(
        np.asarray([9.0, 8.0, 7.0, 6.0, 5.0, 4.0], dtype=np.float32),
        (count, 1),
    )
    action7 = state7.copy()
    images = sorted((episode / "images/external").glob("*.jpg"))
    wrists = sorted((episode / "images/wrist").glob("*.jpg"))
    ack_times = np.asarray(
        [
            1_105_000_000
            if frame < 3
            else 1_205_000_000
            if frame < 6
            else 1_305_000_000
            if frame < 9
            else 1_405_000_000
            for frame in range(count)
        ],
        dtype=np.int64,
    )
    prepared = PreparedEpisode(
        raw_episode_id=episode.name,
        task=json.loads((episode / "episode_result.json").read_text())["task"],
        tuple_host_ns=grid,
        state7=state7,
        wrench6=wrench6,
        action7=action7,
        camera1_paths=tuple(images[frame % len(images)] for frame in range(count)),
        camera2_paths=tuple(wrists[frame % len(wrists)] for frame in range(count)),
        provenance={
            "camera1_receive_monotonic_ns": grid.copy(),
            "camera2_receive_monotonic_ns": grid.copy(),
            "action_ack_receive_monotonic_ns": ack_times,
        },
        diagnostics={"fixture": True},
    )
    probabilities = np.zeros(count, dtype=np.float64)
    probabilities[trigger_frame - 4 : trigger_frame + 1] = 0.9
    scores = FrozenDetectorScores(
        probabilities=tuple(probabilities), validity=(True,) * count
    )
    trace = causal_detection_trace(range(count), scores.probabilities, scores.validity)
    return EpisodeMaterialization(
        prepared=prepared,
        detector_scores=scores,
        detection_trace=trace,
        macros=detector_macro_transitions(trigger_frame),
        wrench_provenance={
            "source": "test_frozen_single_episode_preparer",
            "raw_wrench_learner_eligible": False,
            "normalizer_refit": False,
        },
        outcome_provenance={"source": "frozen_classifier_detector"},
    ).validate()


def _fake_failure_materialization(episode: Path) -> EpisodeMaterialization:
    successful = _fake_materialization(episode)
    count = len(successful.prepared.tuple_host_ns)
    scores = FrozenDetectorScores(
        probabilities=(0.0,) * count,
        validity=(True,) * count,
    )
    return EpisodeMaterialization(
        prepared=successful.prepared,
        detector_scores=scores,
        detection_trace=causal_detection_trace(
            range(count), scores.probabilities, scores.validity
        ),
        macros=(),
        wrench_provenance=successful.wrench_provenance,
        outcome_provenance=successful.outcome_provenance,
    ).validate(allow_detector_miss=True)


def test_frozen_episode_materializer_defers_detector_miss_for_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    prepared = _fake_materialization(episode).prepared
    monkeypatch.setattr(
        bridge_module,
        "prepare_episode",
        lambda *_args, **_kwargs: prepared,
    )

    def detector(value: PreparedEpisode) -> FrozenDetectorScores:
        count = len(value.tuple_host_ns)
        return FrozenDetectorScores(
            probabilities=(0.0,) * count,
            validity=(True,) * count,
        )

    materialization = frozen_episode_materializer(detector)(episode)

    assert materialization.detection_trace.trigger_frame is None
    with pytest.raises(
        ProductionBridgeError,
        match="BRIDGE_FROZEN_DETECTOR_DETECTOR_MISS",
    ):
        materialization.validate()
    assert materialization.validate(allow_detector_miss=True) is materialization


def _bridge(state: Path, **overrides) -> ProductionBridge:
    return ProductionBridge(
        config=BridgeConfig(**overrides),
        state_root=state,
        episode_materializer=_fake_materialization,
    )


def _wal_payloads(state: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))["payload"]
        for path in sorted((state / "wal").glob("*.json"))
    ]


def test_config_is_json_compatible_yaml_and_development_only() -> None:
    config, raw = load_bridge_config(CONFIG)
    assert config.clock_domain_id == "upper_host_monotonic"
    assert config.max_camera_age_ns == 100_000_000
    assert raw["status"] == "filesystem_shadow_only_not_production_integrated"
    assert raw["persistence"]["formal_training_replay_written"] is False


def test_online_materialization_uses_100ms_camera_age_boundary(tmp_path: Path) -> None:
    payload = json.loads(
        (ROOT / "configs/converter_runtime_spec.task2.development.json").read_text(
            encoding="utf-8"
        )
    )
    payload["cameras"]["max_age_ms"] = 34.0
    frozen_checkpoint_contract = tmp_path / "frozen-runtime.json"
    _write_json(frozen_checkpoint_contract, payload)

    contract = bridge_module._online_runtime_contract(frozen_checkpoint_contract)
    assert contract.camera_max_age_ms == 100.0
    assert 46.144793 <= contract.camera_max_age_ms
    assert 100.000001 > contract.camera_max_age_ms


def test_core_source_has_no_ros_network_robot_or_cuda_imports() -> None:
    source = (ROOT / "src/forcesmolvla/rft/online/production_bridge.py").read_text()
    for forbidden in ("import rclpy", "import requests", "import torch", "import socket"):
        assert forbidden not in source


def test_integrated_shadow_keeps_human_execution_separate_from_policy_lineage(
    tmp_path: Path,
) -> None:
    episode = _fixture(tmp_path)
    _integrated_shadow_fixture(episode)
    state = tmp_path / "state"
    report = _bridge(state).process_episode(
        episode, operator_task_outcome="success"
    )
    assert report.status == "SEALED_COMMITTED"
    assert report.classification == "recorded_live_policy_shadow"
    assert report.technical_seal == "complete"
    assert report.operator_task_outcome == "success"
    assert report.executed_action_source == "human"
    assert report.policy_execution is False
    assert report.real_online_r_used is False
    assert report.formal_training_replay_written is False
    assert report.detector_outcome == "success"
    assert report.shadow_observation_count == 1
    assert report.shadow_policy_request_count == 1
    assert report.shadow_policy_result_count == 1
    assert report.shadow_policy_proposal_count == 1
    assert report.shadow_human_ack_count == 5
    assert report.outbox_eligible_count == 3
    assert report.quarantined_count == 0
    payloads = _wal_payloads(state)
    assert all(item["classification"] == "recorded_live_policy_shadow" for item in payloads)
    assert all(
        item["runtime_lineage"]["binding_kind"]
        == "recorded_live_human_execution"
        and item["runtime_lineage"]["policy_result_id"] is None
        and item["runtime_lineage"]["policy_proposal_id"] is None
        and item["runtime_lineage"]["policy_executed_transition"] is False
        and item["behavior"]["actual_action_source"] == "human"
        and item["behavior"]["human_ack_bound_to_policy_proposal"] is False
        and item["eligibility"]["formal_replay"] is False
        and item["eligibility"]["real_online_r"] is False
        for item in payloads
    )
    assert all(
        item["shadow_policy_lineage"]["policy_request_count"] == 1
        and item["shadow_policy_lineage"]["policy_execution"] is False
        and item["shadow_policy_lineage"]["human_ack_policy_binding"] is None
        and item["commit"]["integrated_shadow_episode_sealed"] is True
        for item in payloads
    )


def test_integrated_shadow_dry_run_is_read_only_and_rejects_ack_rebinding(
    tmp_path: Path,
) -> None:
    episode = _fixture(tmp_path)
    _integrated_shadow_fixture(episode)
    path = (
        episode.parent.parent
        / "integrated_capture"
        / episode.name
        / "streams/policy_shadow_human_ack.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["proposal_id"] = "policy-proposal:forged"
    _write_jsonl(path, rows)
    state = tmp_path / "dry-run-state"
    report = _bridge(state).process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )
    assert report.status == "SEALED_QUARANTINED"
    assert report.classification == "recorded_live_policy_shadow"
    assert report.quarantine_reasons == ("BRIDGE_SHADOW_HUMAN_ACK_INVALID",)
    assert not state.exists()


def test_integrated_policy_execution_smoke_is_read_only_and_not_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    observation_path = (
        episode.parent.parent
        / "integrated_capture"
        / episode.name
        / "streams/policy_execute_observation.jsonl"
    )
    observation_rows = [
        json.loads(line) for line in observation_path.read_text().splitlines()
    ]
    pose_path = episode / "streams/measured_tcp_pose.jsonl"
    pose_rows = [json.loads(line) for line in pose_path.read_text().splitlines()]
    source_ns = int(
        observation_rows[0]["stream_ids"]["measured_tcp_pose"]
        .split("source:", 1)[1]
        .split("@receive:", 1)[0]
    )
    policy_receive_ns = observation_rows[0]["t_ref_ns"] - 100_000
    observation_rows[0]["stream_timestamps_ns"]["measured_tcp_pose"] = (
        policy_receive_ns
    )
    observation_rows[0]["stream_ids"]["measured_tcp_pose"] = (
        f"source:{source_ns}@receive:{policy_receive_ns}"
    )
    next(
        row for row in pose_rows if int(row["source_stamp_ns"]) == source_ns
    )["receive_monotonic_ns"] = observation_rows[0]["t_ref_ns"] + 100_000
    _write_jsonl(observation_path, observation_rows)
    _write_jsonl(pose_path, pose_rows)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )
    state = tmp_path / "dry-run-state"

    report = _bridge(state).process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "DRY_RUN_READY"
    assert report.classification == "recorded_live_policy_execution_smoke"
    assert report.executed_action_source == "policy"
    assert report.policy_execution is True
    assert report.policy_lineage_complete is True
    assert report.policy_action_ack_count == 2
    assert report.candidate_count == 2
    assert report.human_override_count == 1
    assert report.human_override_executed_count == 1
    assert report.quarantined_count == 0
    assert report.operator_task_outcome == "success"
    assert report.training_replay_eligible is False
    assert report.formal_training_replay_written is False
    assert report.real_online_r_used is False
    assert report.model_update_count == 0
    assert not state.exists()


def test_integrated_async_policy_execution_seal_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    _offset_episode_policy_epoch(episode, 1)
    _make_policy_execution_fixture_async(episode)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )
    state = tmp_path / "dry-run-state"

    report = _bridge(state).process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "DRY_RUN_READY"
    assert report.policy_action_ack_count == 2
    assert report.human_override_count == 1
    assert report.quarantined_count == 0
    assert report.model_update_count == 0
    assert not state.exists()


def test_policy_execution_request_cancellation_partitions_63_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    _add_canceled_policy_request(episode, completed_request_count=60)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )

    report = _bridge(tmp_path / "dry-run-state").process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "DRY_RUN_READY"
    assert report.shadow_policy_request_count == 63
    assert report.shadow_policy_result_count == 62
    assert report.policy_chunk_count == 62
    assert report.quarantined_count == 0


def test_policy_execution_request_cancellation_rejects_result_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    request_id = _add_canceled_policy_request(episode)
    stream_root = episode.parent.parent / "integrated_capture" / episode.name / "streams"
    for name in ("result", "proposal", "chunk"):
        path = stream_root / f"policy_execute_{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        overlap = deepcopy(rows[-1])
        overlap.update(
            {
                "request_id": request_id,
                "result_id": f"policy-result:{request_id}",
                "chunk_id": f"live-{request_id}",
                "proposal_id": f"policy-proposal:{request_id}",
            }
        )
        rows.append(overlap)
        _write_jsonl(path, rows)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )

    report = _bridge(tmp_path / "dry-run-state").process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "SEALED_QUARANTINED"
    assert report.quarantine_reasons == (
        "BRIDGE_POLICY_EXECUTION_LINEAGE_COUNT_MISMATCH",
    )


@pytest.mark.parametrize("artifact", ["proposal", "chunk", "ack"])
def test_policy_execution_request_cancellation_rejects_execution_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    request_id = _add_canceled_policy_request(episode)
    stream_root = episode.parent.parent / "integrated_capture" / episode.name / "streams"
    if artifact in {"proposal", "chunk"}:
        path = stream_root / f"policy_execute_{artifact}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        executed = deepcopy(rows[-1])
        executed.update(
            {
                "request_id": request_id,
                "result_id": f"policy-result:{request_id}",
                "chunk_id": f"live-{request_id}",
                "proposal_id": f"policy-proposal:{request_id}",
            }
        )
        rows.append(executed)
        _write_jsonl(path, rows)
    else:
        path = episode / "streams/safe_action.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        override = next(
            row
            for row in rows
            if row.get("payload", {}).get("arbitration", {}).get("reason")
            == "human_override"
        )
        override["payload"]["forcesmolvla_chunk_selection"] = {
            "request_id": request_id
        }
        _write_jsonl(path, rows)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )

    report = _bridge(tmp_path / "dry-run-state").process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "SEALED_QUARANTINED"


@pytest.mark.parametrize("invalid", ["unknown_request", "generation_mismatch"])
def test_policy_execution_request_cancellation_rejects_invalid_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    _add_canceled_policy_request(episode)
    path = (
        episode.parent.parent
        / "integrated_capture"
        / episode.name
        / "streams/policy_execute_request_canceled.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if invalid == "unknown_request":
        rows[0]["request_id"] = "unknown-request"
    else:
        rows[0]["policy_epoch"] += 1
        rows[0]["takeover_generation"] += 1
    _write_jsonl(path, rows)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )

    report = _bridge(tmp_path / "dry-run-state").process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "SEALED_QUARANTINED"
    assert report.quarantine_reasons == (
        "BRIDGE_POLICY_EXECUTION_CANCELED_REQUEST_INVALID",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learner_resume_checkpoint", None),
        ("current_episode_sampled_by_learner", True),
        ("active_actor_revision", "changed-active-revision"),
        ("learner_critic_steps", 1),
    ],
)
def test_integrated_async_policy_execution_rejects_invalid_runtime_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value,
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    _make_policy_execution_fixture_async(episode)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )
    seal_path = (
        episode.parent.parent
        / "integrated_capture"
        / episode.name
        / "streams/policy_execute_episode_seal.json"
    )
    seal = json.loads(seal_path.read_text())
    if value is None:
        seal.pop(field)
    else:
        seal[field] = value
    _write_json(seal_path, seal)
    state = tmp_path / "dry-run-state"

    report = _bridge(state).process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "SEALED_QUARANTINED"
    assert report.quarantine_reasons == (
        "BRIDGE_INTEGRATED_POLICY_EXECUTION_SEAL_INVALID",
    )
    assert not state.exists()


def _replace_post_takeover_gripper_command_with_held(episode: Path) -> None:
    path = (
        episode.parent.parent
        / "integrated_capture"
        / episode.name
        / "streams/policy_execute_gripper_authority.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1] = {
        key: value
        for key, value in rows[-1].items()
        if key
        not in {
            "command_id",
            "local_goal_sequence",
            "action_goal_id",
            "started_monotonic_ns",
            "accepted_monotonic_ns",
            "finished_monotonic_ns",
            "outcome",
        }
    }
    rows[-1].update(
        {
            "command_required": False,
            "authority": "existing_accepted_gripper_state",
            "feedback_width_m": 0.085,
            "feedback_monotonic_ns": 1_300_000_000,
        }
    )
    _write_jsonl(path, rows)
    transition_path = path.with_name("policy_execute_transition.jsonl")
    transitions = [
        json.loads(line) for line in transition_path.read_text().splitlines()
    ]
    transitions[-1]["gripper_authority"] = rows[-1]
    _write_jsonl(transition_path, transitions)


def _takeover_gripper_transfer_inputs() -> dict:
    identity = {
        "session_id": "session",
        "episode_id": "episode_000000",
        "clock_domain_id": "upper_host_monotonic",
        "policy_revision": "revision",
        "policy_epoch": 1,
        "reset_generation": 0,
        "takeover_generation": 0,
    }
    return {
        "origin": {
            "terminal_finished_monotonic_ns": 80,
            "requested_state": "OPEN",
            "requested_width_m": 0.085,
            "origin_kind": "native_gripper_terminal",
            "origin_local_goal_sequence": 7,
            "origin_action_goal_id": "real-goal-7",
            "origin_accepted_monotonic_ns": 70,
            "terminal_outcome": "reached",
            "generation": dict(identity),
        },
        "sync_event": {
            **identity,
            "policy_epoch": 2,
            "takeover_generation": 1,
            "event": "intervention_start",
            "receive_monotonic_ns": 100,
            "safe_action": {
                "arbitration": {"raw_action": {"action": [0.0] * 6 + [-1.0]}}
            },
        },
        "feedback": {"width_m": 0.085, "receive_monotonic_ns": 110},
        "new_generation": (2, 0, 1),
        "identity": identity,
        "conflicting_pending_command": False,
        "stalled_settled_width_m": None,
        "settled_width_tolerance_m": 0.001,
    }


def test_takeover_gripper_held_authority_transfer_preserves_real_origin(
    tmp_path: Path,
) -> None:
    bridge = _bridge(tmp_path / "state")
    transfer = bridge._transfer_takeover_gripper_held_authority(
        **_takeover_gripper_transfer_inputs()
    )

    assert transfer is not None
    assert transfer["authority_kind"] == "HELD_FROM_ACCEPTED_COMMAND"
    assert transfer["origin_local_goal_sequence"] == 7
    assert transfer["origin_action_goal_id"] == "real-goal-7"
    assert transfer["origin_accepted_monotonic_ns"] == 70
    assert transfer["terminal_outcome"] == "reached"
    assert transfer["generation"]["policy_epoch"] == 2
    assert transfer["generation"]["takeover_generation"] == 1
    assert not set(transfer) & {"ack_id", "action_goal_ack", "terminal_ack"}


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_sync",
        "stale_feedback",
        "mismatched_state",
        "mismatched_width",
        "pending_command",
        "invalid_terminal",
    ],
)
def test_takeover_gripper_held_authority_transfer_rejects_invalid_evidence(
    tmp_path: Path, invalid: str
) -> None:
    values = _takeover_gripper_transfer_inputs()
    if invalid == "missing_sync":
        values["sync_event"] = None
    elif invalid == "stale_feedback":
        values["feedback"]["receive_monotonic_ns"] = 100_000_101
    elif invalid == "mismatched_state":
        values["sync_event"]["safe_action"]["arbitration"]["raw_action"][
            "action"
        ][-1] = 1.0
    elif invalid == "mismatched_width":
        values["feedback"]["width_m"] = 0.04
    elif invalid == "pending_command":
        values["conflicting_pending_command"] = True
    else:
        values["origin"]["terminal_outcome"] = "rejected"

    transfer = _bridge(
        tmp_path / "state"
    )._transfer_takeover_gripper_held_authority(**values)

    assert transfer is None


def test_repeated_takeover_gripper_transfers_bind_each_new_generation(
    tmp_path: Path,
) -> None:
    bridge = _bridge(tmp_path / "state")
    first = bridge._transfer_takeover_gripper_held_authority(
        **_takeover_gripper_transfer_inputs()
    )
    assert first is not None
    values = _takeover_gripper_transfer_inputs()
    values.update(
        {
            "origin": first,
            "new_generation": (3, 0, 2),
            "sync_event": {
                **values["sync_event"],
                "policy_epoch": 3,
                "takeover_generation": 2,
                "receive_monotonic_ns": 200,
            },
            "feedback": {"width_m": 0.085, "receive_monotonic_ns": 210},
        }
    )

    second = bridge._transfer_takeover_gripper_held_authority(**values)

    assert second is not None
    assert first["generation"]["takeover_generation"] == 1
    assert second["generation"]["takeover_generation"] == 2
    assert second["origin_action_goal_id"] == first["origin_action_goal_id"]


def test_policy_execution_accepts_explicit_takeover_gripper_held_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    _replace_post_takeover_gripper_command_with_held(episode)
    original_ack = (episode / "streams/reference_ack.jsonl").read_bytes()
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )

    report = _bridge(tmp_path / "dry-run-state").process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "DRY_RUN_READY"
    assert report.candidate_count == 2
    assert (episode / "streams/reference_ack.jsonl").read_bytes() == original_ack


def test_policy_execution_excludes_held_action_after_takeover_until_new_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    _replace_post_takeover_gripper_command_with_held(episode)
    target_path = episode / "streams/gripper_target.jsonl"
    target_rows = [json.loads(line) for line in target_path.read_text().splitlines()]
    target_rows[0]["started_monotonic_ns"] = 1_190_000_000
    _write_jsonl(target_path, target_rows)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )

    report = _bridge(tmp_path / "dry-run-state").process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "DRY_RUN_READY"
    assert report.policy_action_ack_count == 2
    assert report.candidate_count == 1
    assert report.quarantined_count == 0


def test_policy_execution_recovers_held_gripper_when_pending_goal_finishes_during_takeover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    _replace_post_takeover_gripper_command_with_held(episode)
    target_path = episode / "streams/gripper_target.jsonl"
    status_path = episode / "streams/gripper_goal_status.jsonl"
    targets = [json.loads(line) for line in target_path.read_text().splitlines()]
    statuses = [json.loads(line) for line in status_path.read_text().splitlines()]
    targets[0].update(
        started_monotonic_ns=1_190_000_000,
        accepted_monotonic_ns=1_195_000_000,
    )
    statuses[0].update(
        accepted_monotonic_ns=1_195_000_000,
        finished_monotonic_ns=1_220_000_000,
    )
    _write_jsonl(target_path, targets)
    _write_jsonl(status_path, statuses)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )

    report = _bridge(tmp_path / "dry-run-state").process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "DRY_RUN_READY"
    assert report.candidate_count == 2


def _admission_prepared(episode: Path) -> PreparedEpisode:
    prepared = _fake_materialization(episode).prepared
    prepared.tuple_host_ns[:] -= 20_000_000
    prepared.provenance["camera1_receive_monotonic_ns"][:] -= 20_000_000
    prepared.provenance["camera2_receive_monotonic_ns"][:] -= 20_000_000
    return prepared


def test_formal_online_r_admission_materializes_policy_and_human_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    monkeypatch.setattr(bridge_module, "_prepare_native_episode", _admission_prepared)
    stream_root = (
        episode.parent.parent / "integrated_capture" / episode.name / "streams"
    )
    transition_path = stream_root / "policy_execute_transition.jsonl"
    seal_path = stream_root / "policy_execute_episode_seal.json"
    original_transition_bytes = transition_path.read_bytes()
    original_seal_bytes = seal_path.read_bytes()
    state = tmp_path / "formal-r"

    report = _bridge(state).admit_policy_execution_smoke(
        episode,
        operator_task_outcome="success",
    )

    assert report.status == "FORMAL_ONLINE_R_ADMITTED"
    assert report.policy_execution_smoke_bridge == "PASS"
    assert report.accepted_unique_r_transition_count == 3
    assert report.total_unique_r_transition_count == 3
    assert report.training_starts == 100
    assert report.training_starts_reached is False
    assert report.human_override_count == 1
    assert report.human_override_replay_count == 1
    assert report.invalidated_proposal_replay_count == 0
    assert report.observation_warmup_excluded_count == 0
    assert report.wal_written_count == 3
    assert report.outbox_written_count == 3
    assert report.replay_written_count == 3
    assert report.actor_update_count == 0
    assert report.critic_update_count == 0
    assert report.optimizer_update_count == 0
    assert report.checkpoint_update_count == 0
    assert len(list((state / "wal").glob("*.json"))) == 3
    assert len(list((state / "outbox").glob("*.json"))) == 3
    replay_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((state / "replay").glob("*.json"))
    ]
    payloads = sorted(
        (record["payload"] for record in replay_records),
        key=lambda item: item["identity"]["decision_id"],
    )
    assert [item["identity"]["decision_id"] for item in payloads] == [1, 2, 3]
    assert all(
        item["classification"] == "recorded_live_policy_execution_smoke"
        and item["eligibility"]["formal_replay"] is True
        and item["eligibility"]["real_online_r"] is True
        and item["eligibility"]["replay_membership"] == "R_online"
        and item["action_authority"]["full_action7_ack_closure"] is True
        and item["action_authority"]["pose_ack"]["accepted"] is True
        and len(item["action_authority"]["accepted_absolute_action7"]) == 7
        and len(item["observation"]["state7_absolute"]) == 7
        and len(item["observation"]["wrench6_calibrated_tcp"]) == 6
        and item["observation"]["camera_external"]["model"] == "D435"
        and item["observation"]["camera_wrist"]["model"] == "D405"
        and item["action_authority"]["gripper_terminal_provenance"]
        for item in payloads
    )
    policy_payloads = [item for item in payloads if item["action_source"] == "policy"]
    human = next(item for item in payloads if item["action_source"] == "human")
    assert all(
        item["expert"] is False
        and item["intervention"] is False
        and item["policy_lineage"].keys()
        >= {"request", "result", "proposal", "chunk", "revision", "generation"}
        for item in policy_payloads
    )
    assert human["expert"] is True and human["intervention"] is True
    assert "policy_lineage" not in human
    assert np.asarray(human["human_action_target_h50"]).shape == (50, 7)
    human_mask = np.asarray(
        human["human_action_valid_mask_h50"], dtype=np.bool_
    )
    assert human_mask.shape == (50, 7) and human_mask.sum() == 7
    assert human["action_authority"]["pose_ack"]["accepted"] is True
    assert [item["outcome"]["reward"] for item in payloads] == [0.0, 0.0, 1.0]
    assert [item["outcome"]["terminated"] for item in payloads] == [False, False, True]
    assert [item["outcome"]["truncated"] for item in payloads] == [True, True, False]
    assert [item["outcome"]["bootstrap_mask"] for item in payloads] == [0.0, 0.0, 0.0]
    assert [item["outcome"]["discount"] for item in payloads] == [0.0, 0.0, 0.0]
    assert all(
        item["policy_lineage"]["proposal"]["invalidated_by_takeover"] is False
        for item in policy_payloads
    )
    admission = json.loads(next((state / "admissions").glob("*.json")).read_text())
    assert admission["source_episode_semantics"] == {
        "formal_replay": False,
        "real_online_r": False,
    }
    assert admission["admitted_replay_semantics"] == {
        "formal_replay": True,
        "membership": "R_online",
        "real_online_r": True,
    }
    episode_seal = json.loads(next((state / "episodes").glob("*.json")).read_text())
    assert episode_seal["accepted_unique_r_transition_count"] == 3
    assert episode_seal["human_override_replay_count"] == 1
    assert episode_seal["learner_started"] is False
    assert episode_seal["actor_updates"] == 0
    assert episode_seal["critic_updates"] == 0
    assert episode_seal["optimizer_updates"] == 0
    assert episode_seal["checkpoint_updates"] == 0
    assert transition_path.read_bytes() == original_transition_bytes
    assert seal_path.read_bytes() == original_seal_bytes


def test_formal_online_r_admission_accepts_exact_resume_waiting_for_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    _make_policy_execution_fixture_async(episode)
    monkeypatch.setattr(bridge_module, "_prepare_native_episode", _admission_prepared)

    report = _bridge(tmp_path / "formal-r").admit_policy_execution_smoke(
        episode,
        operator_task_outcome="success",
    )

    assert report.status == "FORMAL_ONLINE_R_ADMITTED"
    assert report.accepted_unique_r_transition_count == 3
    assert report.training_starts_reached is False
    assert report.actor_update_count == 0
    assert report.critic_update_count == 0


def test_formal_online_r_admission_is_uid_digest_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    monkeypatch.setattr(bridge_module, "_prepare_native_episode", _admission_prepared)
    state = tmp_path / "formal-r"
    bridge = _bridge(state)
    first = bridge.admit_policy_execution_smoke(
        episode, operator_task_outcome="success"
    )
    before = {
        path.relative_to(state): path.read_bytes()
        for path in state.rglob("*.json")
    }

    second = bridge.admit_policy_execution_smoke(
        episode, operator_task_outcome="success"
    )

    assert first.accepted_unique_r_transition_count == 3
    assert second.accepted_unique_r_transition_count == 3
    assert second.wal_written_count == 0
    assert second.outbox_written_count == 0
    assert second.replay_written_count == 0
    assert second.idempotent_transition_count == 3
    assert second.admission_record_written is False
    assert second.episode_seal_written is False
    assert before == {
        path.relative_to(state): path.read_bytes()
        for path in state.rglob("*.json")
    }


@pytest.mark.parametrize("missing", [True, False])
def test_human_action_without_accepted_ack_is_not_expert_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: bool,
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    ack_path = episode / "streams/reference_ack.jsonl"
    rows = [json.loads(line) for line in ack_path.read_text().splitlines()]
    if missing:
        rows.pop(2)
        result_path = episode / "episode_result.json"
        result = json.loads(result_path.read_text())
        result["stream_counts"]["reference_ack"] = len(rows)
        _write_json(result_path, result)
        seal_path = (
            episode.parent.parent
            / "integrated_capture"
            / episode.name
            / "streams/policy_execute_episode_seal.json"
        )
        seal = json.loads(seal_path.read_text())
        seal["native_episode_result"] = result
        _write_json(seal_path, seal)
    else:
        rows[2]["payload"]["accepted"] = False
    _write_jsonl(ack_path, rows)
    monkeypatch.setattr(bridge_module, "_prepare_native_episode", _admission_prepared)

    report = _bridge(tmp_path / "formal-r").admit_policy_execution_smoke(
        episode, operator_task_outcome="success"
    )

    assert report.accepted_unique_r_transition_count == 2
    assert report.human_override_replay_count == 0


def test_human_action_uses_first_causal_tracker_reference_when_stamp_is_sparse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    intervention_path = (
        episode.parent.parent
        / "integrated_capture"
        / episode.name
        / "streams/policy_execute_intervention.jsonl"
    )
    intervention = json.loads(intervention_path.read_text().splitlines()[0])
    stamp = intervention["safe_action"]["equilibrium_source_stamp_ns"]
    ack_rows = [
        json.loads(line)
        for line in (episode / "streams/reference_ack.jsonl").read_text().splitlines()
    ]
    ack_receive_ns = next(
        row["receive_monotonic_ns"]
        for row in ack_rows
        if row["payload"]["request_stamp_ns"] == stamp
    )
    ack = next(
        row for row in ack_rows if row["payload"]["request_stamp_ns"] == stamp
    )
    ack["payload"]["accepted_pose"]["position_m"][0] += 0.002
    _write_jsonl(episode / "streams/reference_ack.jsonl", ack_rows)
    reference_path = episode / "streams/accepted_reference.jsonl"
    references = [
        json.loads(line) for line in reference_path.read_text().splitlines()
    ]
    reference = next(row for row in references if row["source_stamp_ns"] == stamp)
    reference["source_stamp_ns"] += 1
    reference["accepted_receive_monotonic_ns"] = ack_receive_ns + 10_000_000
    reference["pose"]["position_m"][0] += 0.001
    _write_jsonl(reference_path, references)
    monkeypatch.setattr(bridge_module, "_prepare_native_episode", _admission_prepared)

    report = _bridge(tmp_path / "formal-r").admit_policy_execution_smoke(
        episode,
        operator_task_outcome="success",
    )

    assert report.status == "FORMAL_ONLINE_R_ADMITTED"
    assert report.human_override_replay_count == 1


def test_formal_online_r_admission_excludes_pre_warmup_current_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )
    state = tmp_path / "formal-r"

    report = _bridge(state).admit_policy_execution_smoke(
        episode,
        operator_task_outcome="success",
    )

    assert report.accepted_unique_r_transition_count == 2
    assert report.observation_warmup_excluded_count == 1
    payloads = [
        json.loads(path.read_text())["payload"]
        for path in (state / "replay").glob("*.json")
    ]
    assert {payload["action_source"] for payload in payloads} == {
        "human",
        "policy",
    }
    terminal = next(payload for payload in payloads if payload["outcome"]["terminated"])
    assert terminal["identity"]["decision_id"] == 3


def test_failure_detector_success_is_outcome_conflict_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    monkeypatch.setattr(bridge_module, "_prepare_native_episode", _admission_prepared)
    state = tmp_path / "formal-r"

    with pytest.raises(
        ProductionBridgeError,
        match="BRIDGE_FORMAL_R_OUTCOME_CONFLICT",
    ):
        _bridge(state).admit_policy_execution_smoke(
            episode,
            operator_task_outcome="failure",
        )

    assert not state.exists()


def _admit_complete_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, list[dict]]:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    monkeypatch.setattr(bridge_module, "_prepare_native_episode", _admission_prepared)
    state = tmp_path / "formal-r"
    bridge = ProductionBridge(
        config=BridgeConfig(),
        state_root=state,
        episode_materializer=_fake_failure_materialization,
    )
    report = bridge.admit_policy_execution_smoke(
        episode,
        operator_task_outcome="failure",
    )
    payloads = sorted(
        (
            json.loads(path.read_text(encoding="utf-8"))["payload"]
            for path in (state / "replay").glob("*.json")
        ),
        key=lambda item: item["identity"]["decision_id"],
    )
    return report, payloads


def test_complete_failure_episode_is_td_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, payloads = _admit_complete_failure(tmp_path, monkeypatch)

    assert report.status == "FORMAL_ONLINE_R_ADMITTED"
    assert report.accepted_unique_r_transition_count == len(payloads) == 3
    assert all(item["eligibility"]["td_eligible"] is True for item in payloads)


def test_failure_terminal_reward_is_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _report, payloads = _admit_complete_failure(tmp_path, monkeypatch)

    assert [item["outcome"]["reward"] for item in payloads] == [0.0, 0.0, 0.0]


def test_failure_terminal_is_terminated_without_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _report, payloads = _admit_complete_failure(tmp_path, monkeypatch)
    terminal = payloads[-1]["outcome"]

    assert terminal["terminated"] is True
    assert terminal["truncated"] is False
    assert terminal["bootstrap_mask"] == 0.0
    assert terminal["discount"] == 0.0


def test_failure_policy_rows_have_zero_fm_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _report, payloads = _admit_complete_failure(tmp_path, monkeypatch)
    policy = [item for item in payloads if item["action_source"] == "policy"]

    assert policy
    assert all(item["eligibility"]["fm_eligible"] is False for item in policy)
    assert all(item["expert"] is False for item in policy)


def test_failed_episode_human_rows_are_td_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _report, payloads = _admit_complete_failure(tmp_path, monkeypatch)
    human = next(item for item in payloads if item["action_source"] == "human")

    assert human["eligibility"]["td_eligible"] is True
    assert human["eligibility"]["fm_eligible"] is False
    assert human["expert"] is False


def test_policy_execution_generation_mismatch_is_quarantined_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )
    path = (
        episode.parent.parent
        / "integrated_capture"
        / episode.name
        / "streams/policy_execute_transition.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["takeover_generation"] = 0
    _write_jsonl(path, rows)
    state = tmp_path / "dry-run-state"

    report = _bridge(state).process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "SEALED_QUARANTINED"
    assert report.classification == "recorded_live_policy_execution_smoke"
    assert report.quarantine_reasons == (
        "BRIDGE_POLICY_EXECUTION_GENERATION_INVALID",
    )
    assert not state.exists()


def test_policy_execution_generation_uses_independent_initial_offsets() -> None:
    identity = {
        "session_id": "stage3-policy-execute-fixture",
        "episode_id": "episode_000000",
        "clock_domain_id": "upper_host_monotonic",
        "policy_revision": "cycle10",
        "policy_epoch": 1,
        "reset_generation": 0,
        "takeover_generation": 0,
    }

    generations = [
        ProductionBridge._policy_execution_generation(
            {**identity, "policy_epoch": epoch, "takeover_generation": takeover},
            identity,
        )
        for epoch, takeover in ((1, 0), (2, 1), (3, 2))
    ]
    for previous, current in zip(generations[:-1], generations[1:], strict=True):
        ProductionBridge._validate_policy_execution_generation_step(
            previous, current
        )

    for epoch, takeover in ((2, 0), (1, 1)):
        with pytest.raises(
            ProductionBridgeError,
            match="BRIDGE_POLICY_EXECUTION_GENERATION_INVALID",
        ):
            ProductionBridge._policy_execution_generation(
                {
                    **identity,
                    "policy_epoch": epoch,
                    "takeover_generation": takeover,
                },
                identity,
            )
    for previous, current in (
        (generations[0], generations[2]),
        (generations[1], generations[0]),
    ):
        with pytest.raises(
            ProductionBridgeError,
            match="BRIDGE_POLICY_EXECUTION_GENERATION_INVALID",
        ):
            ProductionBridge._validate_policy_execution_generation_step(
                previous, current
            )


def test_policy_execution_accepts_nonzero_initial_policy_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    _offset_episode_policy_epoch(episode, 1)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )
    state = tmp_path / "dry-run-state"

    report = _bridge(state).process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "DRY_RUN_READY"
    assert report.policy_action_ack_count == 2
    assert report.quarantined_count == 0
    assert not state.exists()


def test_policy_execution_invalidated_old_proposal_cannot_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _fixture(tmp_path)
    _integrated_policy_execution_fixture(episode)
    monkeypatch.setattr(
        bridge_module,
        "_prepare_native_episode",
        lambda path: _fake_materialization(path).prepared,
    )
    path = (
        episode.parent.parent
        / "integrated_capture"
        / episode.name
        / "streams/policy_execute_proposal.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["invalidated_by_takeover"] = True
    _write_jsonl(path, rows)
    state = tmp_path / "dry-run-state"

    report = _bridge(state).process_episode(
        episode,
        dry_run=True,
        operator_task_outcome="success",
    )

    assert report.status == "SEALED_QUARANTINED"
    assert report.quarantine_reasons == (
        "BRIDGE_POLICY_EXECUTION_INVALIDATED_PROPOSAL_EXECUTED",
    )
    assert not state.exists()


def test_active_episode_only_updates_staging(tmp_path: Path) -> None:
    episode = _fixture(tmp_path, sealed=False)
    _write_jsonl(
        episode / "streams/safe_action.jsonl",
        [
            {
                "receive_monotonic_ns": 1,
                "payload": {
                    "arbitration": {"raw_action": {"phase": "control"}}
                },
            }
        ],
    )
    state = tmp_path / "state"
    report = _bridge(state).process_episode(episode)
    assert report.status == "ACTIVE_STAGED"
    assert report.candidate_count == 1
    staged_path = next((state / "staging").glob("*.json"))
    staged = json.loads(staged_path.read_text())
    assert staged["stream_cursors"]["safe_action"]["record_count"] == 1
    assert staged["stream_cursors"]["safe_action"]["byte_offset"] > 0
    assert not (state / "wal").exists()
    assert not (state / "outbox").exists()


def test_sealed_episode_closes_new_and_held_authority_into_wal_outbox(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    state = tmp_path / "state"
    report = _bridge(state).process_episode(episode)
    assert report.status == "SEALED_COMMITTED"
    assert report.outbox_eligible_count == 3
    assert report.quarantined_count == 0
    assert report.new_command_count == 1
    assert report.held_command_count == 2
    assert len(list((state / "wal").glob("*.json"))) == 3
    assert len(list((state / "outbox").glob("*.json"))) == 3
    payloads = sorted(_wal_payloads(state), key=lambda item: item["identity"]["decision_id"])
    kinds = [item["action_authority"]["gripper"]["authority_kind"] for item in payloads]
    assert kinds == ["NEW_COMMAND", "HELD_FROM_ACCEPTED_COMMAND", "HELD_FROM_ACCEPTED_COMMAND"]
    assert all(item["action_authority"]["full_action7_ack_closure"] for item in payloads)
    assert all(item["ack_macro"]["K"] == 3 for item in payloads)
    assert all(item["ack_macro"]["projection_grid_hz"] == 30 for item in payloads)
    assert all(item["ack_macro"]["partial"] is False for item in payloads)
    assert all(
        item["action_authority"]["gripper"]["origin_action_goal_id"] == "real-goal-1"
        for item in payloads
    )
    assert all(item["commit"]["normalizer_invocations"] == 0 for item in payloads)
    assert all(
        item["observation"]["wrench6"] == [9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
        and item["observation"]["raw_wrench_learner_eligible"] is False
        and item["commit"]["wrench_materialized"] is True
        and item["commit"]["reward_terminal_materialized"] is True
        for item in payloads
    )
    assert payloads[-1]["outcome"]["episode_boundary"] is True
    assert all(item["behavior"]["phase"] == "control" for item in payloads)
    assert not (state / "replay").exists()


def test_sealed_episode_without_materializer_is_quarantined(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    state = tmp_path / "state"
    report = ProductionBridge(
        config=BridgeConfig(), state_root=state
    ).process_episode(episode)
    assert report.status == "SEALED_QUARANTINED"
    assert report.outbox_eligible_count == 0
    assert report.quarantine_reasons == ("BRIDGE_EPISODE_MATERIALIZER_UNBOUND",)
    assert not (state / "wal").exists()
    assert not (state / "outbox").exists()


def test_reward_terminal_comes_from_frozen_detector_not_episode_result(
    tmp_path: Path,
) -> None:
    episode = _fixture(tmp_path)
    result_path = episode / "episode_result.json"
    result = json.loads(result_path.read_text())
    result["reward"] = 0.0
    result["terminated"] = False
    _write_json(result_path, result)
    state = tmp_path / "state"
    report = _bridge(state).process_episode(episode)
    assert report.outbox_eligible_count == 3
    terminal = next(
        item for item in _wal_payloads(state) if item["outcome"]["task_terminated"]
    )
    assert terminal["outcome"]["reward"] == 1.0
    assert terminal["outcome"]["done"] is True
    assert terminal["outcome"]["reward_source"] == "frozen_classifier_detector"
    assert terminal["outcome"]["provenance"]["source"] == "frozen_classifier_detector"


def test_detector_miss_quarantines_episode_before_wal(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)

    def detector_miss(_: Path) -> EpisodeMaterialization:
        raise ProductionBridgeError("BRIDGE_FROZEN_DETECTOR_DETECTOR_MISS")

    state = tmp_path / "state"
    report = ProductionBridge(
        config=BridgeConfig(),
        state_root=state,
        episode_materializer=detector_miss,
    ).process_episode(episode)
    assert report.status == "SEALED_QUARANTINED"
    assert report.quarantine_reasons == ("BRIDGE_FROZEN_DETECTOR_DETECTOR_MISS",)
    assert not (state / "wal").exists()
    assert not (state / "outbox").exists()


def test_held_authority_has_fresh_feedback_and_no_fake_command_id(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    state = tmp_path / "state"
    _bridge(state).process_episode(episode)
    held = [
        item
        for item in _wal_payloads(state)
        if item["action_authority"]["gripper"]["authority_kind"]
        == "HELD_FROM_ACCEPTED_COMMAND"
    ]
    assert held
    for item in held:
        gripper = item["action_authority"]["gripper"]
        assert gripper["origin_action_goal_id"] == "real-goal-1"
        assert gripper["feedback_age_ns"] <= 100_000_000
        assert "new_command_id" not in gripper


def test_terminal_failure_quarantines_whole_episode_and_writes_no_wal(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    status = episode / "streams/gripper_goal_status.jsonl"
    record = json.loads(status.read_text())
    record["outcome"] = "result_error"
    _write_jsonl(status, [record])
    state = tmp_path / "state"
    report = _bridge(state).process_episode(episode)
    assert report.status == "SEALED_QUARANTINED"
    assert "BRIDGE_GRIPPER_TERMINAL_INVALID" in report.quarantine_reasons[0]
    assert not (state / "wal").exists()
    assert list((state / "quarantine").glob("*.json"))


def test_stale_feedback_quarantines_transition_and_never_enters_outbox(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    state = tmp_path / "state"
    report = _bridge(state, max_gripper_feedback_age_ns=1).process_episode(episode)
    assert report.outbox_eligible_count == 1
    assert report.quarantined_count == 2
    assert "BRIDGE_CAUSAL_SAMPLE_STALE" in report.quarantine_reasons
    assert len(list((state / "outbox").glob("*.json"))) == 1


def test_same_uid_same_digest_is_idempotent(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    state = tmp_path / "state"
    first = _bridge(state).process_episode(episode)
    second = _bridge(state).process_episode(episode)
    assert first.wal_written_count == first.outbox_written_count == 3
    assert second.wal_written_count == second.outbox_written_count == 0
    assert second.idempotent_count == 6


def test_same_uid_different_digest_fails_closed(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    state = tmp_path / "state"
    _bridge(state).process_episode(episode)
    result_path = episode / "episode_result.json"
    result = json.loads(result_path.read_text())
    result["task"] = "different-content-same-transition-identity"
    _write_json(result_path, result)
    with pytest.raises(BridgeDigestCollisionError, match="IMMUTABLE_COLLISION"):
        _bridge(state).process_episode(episode)


def test_crash_after_wal_recovers_idempotently_from_sealed_episode(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    state = tmp_path / "state"
    with pytest.raises(InjectedBridgeCrash, match="INJECTED_AFTER_WAL"):
        _bridge(state).process_episode(episode, inject_crash_after_wal=1)
    assert len(list((state / "wal").glob("*.json"))) == 1
    assert not list((state / "outbox").glob("*.json"))
    recovered = _bridge(state).process_episode(episode)
    assert recovered.outbox_written_count == 3
    assert len(list((state / "wal").glob("*.json"))) == 3
    assert len(list((state / "outbox").glob("*.json"))) == 3


def test_policy_source_without_full_runtime_lineage_is_quarantined(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    safe_path = episode / "streams/safe_action.jsonl"
    raw_path = episode / "streams/raw_action.jsonl"
    safe_records = [json.loads(line) for line in safe_path.read_text().splitlines()]
    raw_records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    requested_path = episode / "streams/requested_equilibrium.jsonl"
    requested = [json.loads(line) for line in requested_path.read_text().splitlines()]
    for index in (1, 2, 3):
        raw = safe_records[index]["payload"]["arbitration"]["raw_action"]
        raw["source"] = "policy"
        safe_records[index]["payload"]["forcesmolvla_chunk_selection"] = {
            "chunk_id": "current-production-only",
            "t_ref_ns": raw["source_monotonic_ns"],
            "action_index": index,
        }
        raw_records[index]["payload"]["source"] = "policy"
        requested[index]["source"] = "policy"
    _write_jsonl(safe_path, safe_records)
    _write_jsonl(raw_path, raw_records)
    _write_jsonl(requested_path, requested)
    state = tmp_path / "state"
    report = _bridge(state).process_episode(episode)
    assert report.outbox_eligible_count == 0
    assert report.quarantined_count == 3
    assert any("BRIDGE_POLICY_LINEAGE_UNBOUND" in reason for reason in report.quarantine_reasons)


def test_complete_policy_runtime_lineage_closes_cpu_shadow_action7(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    safe_path = episode / "streams/safe_action.jsonl"
    raw_path = episode / "streams/raw_action.jsonl"
    requested_path = episode / "streams/requested_equilibrium.jsonl"
    safe_records = [json.loads(line) for line in safe_path.read_text().splitlines()]
    raw_records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    requested = [json.loads(line) for line in requested_path.read_text().splitlines()]
    for index in (1, 2, 3):
        raw = safe_records[index]["payload"]["arbitration"]["raw_action"]
        raw["source"] = "policy"
        selection = {
            "request_id": f"request-{index}",
            "result_id": f"result-{index}",
            "chunk_id": "chunk-1",
            "proposal_id": "proposal-1",
            "policy_revision": "revision-1",
            "policy_epoch": 1,
            "reset_generation": 0,
            "takeover_generation": 1,
            "t_ref_ns": raw["source_monotonic_ns"],
            "action_index": index * 3,
            "selected_absolute_action7": [0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 0.085],
            "applied_position_m": [0.5, 0.0, 0.2],
            "applied_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        safe_records[index]["payload"]["forcesmolvla_chunk_selection"] = selection
        raw_records[index]["payload"]["source"] = "policy"
        requested[index]["source"] = "policy"
    _write_jsonl(safe_path, safe_records)
    _write_jsonl(raw_path, raw_records)
    _write_jsonl(requested_path, requested)
    state = tmp_path / "state"
    report = _bridge(state).process_episode(episode)
    assert report.outbox_eligible_count == 3
    assert report.policy_fixture is True
    assert report.recorded_offline_production_bridge == "BLOCKED"
    payloads = _wal_payloads(state)
    assert all(
        item["runtime_lineage"]["binding_kind"]
        == "recorded_policy_runtime_ledger"
        for item in payloads
    )
    assert all(item["action_authority"]["full_action7_ack_closure"] for item in payloads)
    assert all(
        item["runtime_lineage"]["dispatch_sequence"]
        == item["runtime_lineage"]["source_sequence"]
        for item in payloads
    )
    assert all(
        len(item["runtime_lineage"]["selected_post_adapter_absolute7"]) == 7
        and item["runtime_lineage"]["pose_command_id"]
        == item["runtime_lineage"]["pose_ack_id"]
        and item["runtime_lineage"]["gripper_authority_reference"][
            "origin_action_goal_id"
        ]
        == item["action_authority"]["gripper"]["origin_action_goal_id"]
        for item in payloads
    )


def test_episode_initial_real_gripper_lease_closes_early_policy_transitions(
    tmp_path: Path,
) -> None:
    episode = _fixture(tmp_path)
    safe_path = episode / "streams/safe_action.jsonl"
    raw_path = episode / "streams/raw_action.jsonl"
    requested_path = episode / "streams/requested_equilibrium.jsonl"
    safe_records = [json.loads(line) for line in safe_path.read_text().splitlines()]
    raw_records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    requested = [json.loads(line) for line in requested_path.read_text().splitlines()]
    episode_id = "dataset/episode_000000"
    generation = GripperGeneration(
        episode_id=episode_id,
        reset_generation=1,
        takeover_generation=1,
        policy_revision="revision-1",
        policy_epoch=1,
    )
    initial = InitialGripperAuthority(
        episode_id=episode_id,
        origin_local_goal_sequence=1,
        origin_action_goal_id="startup-open-real-goal",
        origin_accepted_monotonic_ns=850_000_000,
        requested_state="OPEN",
        requested_width_m=0.085,
        terminal_outcome="reached",
        terminal_finished_monotonic_ns=900_000_000,
        feedback_width_m=0.085,
        feedback_state="OPEN",
        feedback_monotonic_ns=999_000_000,
        captured_monotonic_ns=1_000_000_000,
        feedback_age_ns=1_000_000,
        clock_domain_id="upper_host_monotonic",
        generation=generation,
    ).validate(max_feedback_age_ns=100_000_000)
    safe_records[0]["payload"]["stage3_initial_gripper_authority"] = initial.to_dict()
    for index in (1, 2, 3):
        raw = safe_records[index]["payload"]["arbitration"]["raw_action"]
        raw["source"] = "policy"
        safe_records[index]["payload"]["forcesmolvla_chunk_selection"] = {
            "lineage_schema": "forcesmolvla-stage3-policy-lineage-v1",
            "request_id": f"request-{index}",
            "result_id": f"result-{index}",
            "chunk_id": "chunk-1",
            "proposal_id": "proposal-1",
            "policy_revision": "revision-1",
            "policy_epoch": 1,
            "reset_generation": 1,
            "takeover_generation": 1,
            "t_ref_ns": raw["source_monotonic_ns"],
            "dispatch_sequence": index,
            "selected_index": index * 3,
            "action_index": index * 3,
            "selected_absolute_action7": [
                0.5,
                0.0,
                0.2,
                0.0,
                0.0,
                0.0,
                0.085,
            ],
            "applied_position_m": [0.5, 0.0, 0.2],
            "applied_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        raw_records[index]["payload"]["source"] = "policy"
        requested[index]["source"] = "policy"
    _write_jsonl(safe_path, safe_records)
    _write_jsonl(raw_path, raw_records)
    _write_jsonl(requested_path, requested)
    _write_jsonl(episode / "streams/gripper_target.jsonl", [])
    _write_jsonl(episode / "streams/gripper_goal_status.jsonl", [])
    result_path = episode / "episode_result.json"
    result = json.loads(result_path.read_text())
    result["stream_counts"]["gripper_target"] = 0
    result["stream_counts"]["gripper_goal_status"] = 0
    _write_json(result_path, result)

    state = tmp_path / "state"
    report = _bridge(state).process_episode(episode)
    assert report.outbox_eligible_count == 3
    assert report.quarantined_count == 0
    assert report.held_command_count == 3
    payloads = _wal_payloads(state)
    assert all(
        item["action_authority"]["gripper"]["authority_kind"]
        == "HELD_FROM_ACCEPTED_COMMAND"
        and item["action_authority"]["gripper"]["origin_action_goal_id"]
        == "startup-open-real-goal"
        for item in payloads
    )


def test_initial_gripper_generation_mismatch_quarantines_instead_of_rebinding(
    tmp_path: Path,
) -> None:
    episode = _fixture(tmp_path)
    safe_path = episode / "streams/safe_action.jsonl"
    records = [json.loads(line) for line in safe_path.read_text().splitlines()]
    generation = GripperGeneration(
        episode_id="dataset/episode_000000",
        reset_generation=7,
        takeover_generation=7,
        policy_revision="old-revision",
        policy_epoch=7,
    )
    records[0]["payload"]["stage3_initial_gripper_authority"] = (
        InitialGripperAuthority(
            episode_id=generation.episode_id,
            origin_local_goal_sequence=9,
            origin_action_goal_id="old-real-goal",
            origin_accepted_monotonic_ns=850_000_000,
            requested_state="OPEN",
            requested_width_m=0.085,
            terminal_outcome="reached",
            terminal_finished_monotonic_ns=900_000_000,
            feedback_width_m=0.085,
            feedback_state="OPEN",
            feedback_monotonic_ns=999_000_000,
            captured_monotonic_ns=1_000_000_000,
            feedback_age_ns=1_000_000,
            clock_domain_id="upper_host_monotonic",
            generation=generation,
        )
        .validate(max_feedback_age_ns=100_000_000)
        .to_dict()
    )
    _write_jsonl(safe_path, records)
    _write_jsonl(episode / "streams/gripper_target.jsonl", [])
    _write_jsonl(episode / "streams/gripper_goal_status.jsonl", [])
    result_path = episode / "episode_result.json"
    result = json.loads(result_path.read_text())
    result["stream_counts"]["gripper_target"] = 0
    result["stream_counts"]["gripper_goal_status"] = 0
    _write_json(result_path, result)
    report = _bridge(tmp_path / "state").process_episode(episode)
    assert report.quarantined_count >= 1
    assert "BRIDGE_INITIAL_GRIPPER_GENERATION_MISMATCH" in report.quarantine_reasons


def test_invalid_initial_gripper_origin_quarantines_whole_episode(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    path = episode / "streams/safe_action.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["payload"]["stage3_initial_gripper_authority"] = {
        "schema": "forcesmolvla-stage3-initial-gripper-authority-v1",
        "origin_action_goal_id": "",
    }
    _write_jsonl(path, records)
    report = _bridge(tmp_path / "state").process_episode(episode)
    assert report.status == "SEALED_QUARANTINED"
    assert report.outbox_eligible_count == 0
    assert "BRIDGE_INITIAL_GRIPPER_AUTHORITY_INVALID" in report.quarantine_reasons[0]


def test_rejected_pose_ack_is_quarantined_and_never_outboxed(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    path = episode / "streams/reference_ack.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[2]["payload"]["accepted"] = False
    _write_jsonl(path, records)
    state = tmp_path / "state"
    report = _bridge(state).process_episode(episode)
    assert report.outbox_eligible_count == 2
    assert report.quarantined_count == 1
    assert "BRIDGE_POSE_ACK_REJECTED_OR_MISMATCHED" in report.quarantine_reasons
    outboxed = {
        item["identity"]["decision_id"] for item in _wal_payloads(state)
    }
    assert 2 not in outboxed


def test_frozen_g1_partial_terminal_macro_is_materialized_with_mask(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    state = tmp_path / "state"
    report = ProductionBridge(
        config=BridgeConfig(),
        state_root=state,
        episode_materializer=lambda path: _fake_materialization(
            path, trigger_frame=8
        ),
    ).process_episode(episode)
    assert report.outbox_eligible_count == 3
    terminal = next(
        item for item in _wal_payloads(state) if item["outcome"]["task_terminated"]
    )
    assert terminal["ack_macro"]["partial"] is True
    assert terminal["ack_macro"]["executed_steps"] == 2
    assert terminal["ack_macro"]["executed_action_mask"] == [True, True, False]


def test_conflicting_pending_gripper_command_blocks_held_authority(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    target_path = episode / "streams/gripper_target.jsonl"
    status_path = episode / "streams/gripper_goal_status.jsonl"
    targets = [json.loads(line) for line in target_path.read_text().splitlines()]
    statuses = [json.loads(line) for line in status_path.read_text().splitlines()]
    targets.append(
        {
            "receive_monotonic_ns": 1_260_000_000,
            "local_goal_sequence": 2,
            "action_goal_id": "conflicting-goal-2",
            "requested_state": "CLOSED",
            "started_monotonic_ns": 1_195_000_000,
            "accepted_monotonic_ns": 1_260_000_000,
            "target_width_m": 0.0,
        }
    )
    statuses.append(
        {
            "receive_monotonic_ns": 1_280_000_000,
            "local_goal_sequence": 2,
            "action_goal_id": "conflicting-goal-2",
            "accepted_monotonic_ns": 1_260_000_000,
            "finished_monotonic_ns": 1_280_000_000,
            "outcome": "reached",
        }
    )
    _write_jsonl(target_path, targets)
    _write_jsonl(status_path, statuses)
    result_path = episode / "episode_result.json"
    result = json.loads(result_path.read_text())
    result["stream_counts"]["gripper_target"] = 2
    result["stream_counts"]["gripper_goal_status"] = 2
    _write_json(result_path, result)
    state = tmp_path / "state"
    report = _bridge(state).process_episode(episode)
    assert report.quarantined_count >= 1
    assert any(
        reason == "BRIDGE_CONFLICTING_GRIPPER_COMMAND_PENDING"
        for reason in report.quarantine_reasons
    )


def test_remote_pose_clock_is_retained_but_not_compared_to_upper_clock(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    state = tmp_path / "state"
    report = _bridge(state).process_episode(episode)
    assert report.outbox_eligible_count == 3
    remote = _wal_payloads(state)[0]["action_authority"][
        "remote_pose_ack_timestamps_uncompared"
    ]
    assert remote["clock_domain_id"] == "controller_host_monotonic"
    assert remote["ack_monotonic_ns"] > 9_000_000_000


@pytest.mark.skipif(
    not REAL_EPISODE.is_dir(), reason="accepted offline recorder episode unavailable"
)
def test_task2_episode18_materialization_matches_existing_g1_row_4922(
    tmp_path: Path,
) -> None:
    import pyarrow.parquet as pq

    def detector(prepared: PreparedEpisode) -> FrozenDetectorScores:
        probabilities = np.zeros(len(prepared.tuple_host_ns), dtype=np.float64)
        probabilities[850:855] = 0.9
        return FrozenDetectorScores(
            probabilities=tuple(probabilities),
            validity=(True,) * len(probabilities),
        )

    config, _ = load_bridge_config(CONFIG)
    state = tmp_path / "state"
    report = ProductionBridge(
        config=config,
        state_root=state,
        episode_materializer=frozen_episode_materializer(detector),
    ).process_episode(REAL_EPISODE)
    assert report.status == "SEALED_COMMITTED"
    assert report.recorded_offline_production_bridge == "PASS"
    assert report.outbox_eligible_count > 0
    assert report.quarantined_count > 0
    assert report.held_command_count > 0
    assert report.policy_fixture is False
    assert report.real_online_r_used is False
    assert report.formal_training_replay_written is False
    payload = next(
        item for item in _wal_payloads(state) if item["identity"]["anchor_frame"] == 837
    )
    g1 = pq.read_table(
        ROOT
        / "artifacts/development/stage2/g1_frozen_detector_transition_view.v1/transition_index.parquet"
    ).slice(4922, 1).to_pylist()[0]
    source = pq.read_table(
        ROOT / "datasets/task2_lerobotv3/data/chunk-000/file-018.parquet",
        columns=["observation.state", "observation.wrench", "action"],
    ).slice(837, 4).to_pylist()
    np.testing.assert_array_equal(
        payload["observation"]["state7_absolute"], source[0]["observation.state"]
    )
    np.testing.assert_array_equal(
        payload["observation"]["wrench6"], source[0]["observation.wrench"]
    )
    np.testing.assert_array_equal(
        payload["next_observation"]["state7_absolute"],
        source[3]["observation.state"],
    )
    np.testing.assert_array_equal(
        payload["ack_macro"]["accepted_absolute_action_k7"],
        [row["action"] for row in source[:3]],
    )
    assert payload["outcome"]["reward"] == g1["reward"]
    assert payload["outcome"]["task_terminated"] == g1["terminated"]
    assert payload["identity"]["next_frame"] == g1["next_frame"] == 840
    assert payload["commit"]["wrench_materialized"] is True
    assert payload["commit"]["reward_terminal_materialized"] is True
