from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from forcesmolvla.rft.stage3.production_bridge import (
    BridgeConfig,
    BridgeDigestCollisionError,
    InjectedBridgeCrash,
    Stage3ProductionBridge,
    load_bridge_config,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/stage3_production_bridge.v1.development.yaml"
REAL_EPISODE = Path(
    "/home/rlc123/fr3_client_ws/datasets/task1/episodes/episode_000017"
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


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


def _bridge(state: Path, **overrides) -> Stage3ProductionBridge:
    return Stage3ProductionBridge(config=BridgeConfig(**overrides), state_root=state)


def _wal_payloads(state: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))["payload"]
        for path in sorted((state / "wal").glob("*.json"))
    ]


def test_config_is_json_compatible_yaml_and_development_only() -> None:
    config, raw = load_bridge_config(CONFIG)
    assert config.clock_domain_id == "upper_host_monotonic"
    assert raw["status"] == "filesystem_shadow_only_not_production_integrated"
    assert raw["persistence"]["formal_training_replay_written"] is False


def test_core_source_has_no_ros_network_robot_or_cuda_imports() -> None:
    source = (ROOT / "src/forcesmolvla/rft/stage3/production_bridge.py").read_text()
    for forbidden in ("import rclpy", "import requests", "import torch", "import socket"):
        assert forbidden not in source


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
    assert payloads[-1]["outcome"]["episode_boundary"] is True
    assert all(item["behavior"]["phase"] == "control" for item in payloads)
    assert not (state / "replay").exists()


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
    assert report.outbox_eligible_count == 0
    assert report.quarantined_count == 3
    assert "BRIDGE_CAUSAL_SAMPLE_STALE" in report.quarantine_reasons
    assert not list((state / "outbox").glob("*.json"))


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


def test_partial_macro_at_terminal_boundary_is_quarantined(tmp_path: Path) -> None:
    episode = _fixture(tmp_path)
    path = episode / "streams/safe_action.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[4]["payload"]["accept_monotonic_ns"] = 1_350_000_000
    records[4]["receive_monotonic_ns"] = 1_350_000_000
    _write_jsonl(path, records)
    state = tmp_path / "state"
    report = _bridge(state).process_episode(episode)
    assert report.outbox_eligible_count == 2
    assert "BRIDGE_PARTIAL_MACRO_AT_BOUNDARY" in report.quarantine_reasons


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
def test_real_accepted_offline_episode_dry_run() -> None:
    config, _ = load_bridge_config(CONFIG)
    report = Stage3ProductionBridge(
        config=config, state_root=Path("/tmp/unused-stage3-production-bridge-dry-run")
    ).process_episode(REAL_EPISODE, dry_run=True)
    assert report.status == "DRY_RUN_READY"
    assert report.recorded_offline_production_bridge == "PASS"
    assert report.outbox_eligible_count > 0
    assert report.quarantined_count > 0
    assert report.new_command_count > 0
    assert report.held_command_count > 0
    assert report.policy_fixture is False
    assert report.real_online_r_used is False
    assert report.formal_training_replay_written is False
