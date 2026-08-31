from __future__ import annotations

import json
from pathlib import Path

import pytest

from forcesmolvla.rft.online.recorded_ack_export import (
    DEFAULT_TERMINAL_INDEX,
    RecordedAckExportError,
    _accepted_ack_rows,
    _derive_transition_selection,
    recorded_ack_id,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_real_terminal_index_selects_full_k3_and_terminal_partial_boundary() -> None:
    selection = _derive_transition_selection(DEFAULT_TERMINAL_INDEX, "episode_000018")
    assert selection == {
        "prepared_grid_start_index": 849,
        "prepared_grid_stop_index_exclusive": 855,
        "current_observation_grid_index": 0,
        "next_observation_grid_index": 3,
        "terminal_observation_grid_index": 5,
        "last_executable_grid_index": 4,
        "full_macro_transition_index": 4926,
        "terminal_transition_index": 4927,
    }


def test_recorded_ack_id_is_only_the_recorded_request_natural_key() -> None:
    assert recorded_ack_id(request_sequence=287, request_stamp_ns=1787145538074728303) == (
        "reference-ack:287:1787145538074728303"
    )


def test_exporter_reports_missing_recorded_ack_identity_instead_of_minting_one(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode_000001"
    pose = {
        "position_m": [0.4, 0.0, 0.3],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    ack_times = [1_000_000_000, 1_100_000_000]
    _write_jsonl(episode / "streams/accepted_reference.jsonl", [
        {
            "accepted_receive_monotonic_ns": value - 1,
            "source_stamp_ns": value - 2,
            "frame_id": "fr3_link0",
            "pose": pose,
            "target_gripper_width_m": 0.085,
        }
        for value in ack_times
    ])
    _write_jsonl(episode / "streams/reference_ack.jsonl", [
        {
            "receive_monotonic_ns": value,
            "payload": {
                "accepted": True,
                "accepted_pose": pose,
                "request_sequence": index,
                "request_stamp_ns": 0,
                "ack_monotonic_ns": value - 3,
            },
        }
        for index, value in enumerate(ack_times)
    ])
    _write_jsonl(episode / "streams/safe_action.jsonl", [{"receive_monotonic_ns": 1}])
    authority = {
        "action_goal_id": "real-goal-id",
        "target_width_m": 0.085,
    }
    with pytest.raises(RecordedAckExportError, match="natural identity/timestamp missing"):
        _accepted_ack_rows(
            episode, selected_ack_times=ack_times, authority=authority,
        )
