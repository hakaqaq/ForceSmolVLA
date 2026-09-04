from __future__ import annotations

import numpy as np

from forcesmolvla.rft.online.action_representation import (
    ABSOLUTE_ACTION_ROTATION_REPRESENTATION,
    legacy_absolute_action7_to_rpy_xyz,
    quaternion_xyzw_to_rpy_xyz,
    rotation_vector_to_rpy_xyz,
)
from forcesmolvla.rft.online import replay_training


def _ack_row(
    rotation: list[float], *, source: str, marked_rpy: bool
) -> dict:
    row = {
        "action_source": source,
        "identity": {
            "episode_id": "episode_000000",
            "decision_id": 1,
            "source_ack_id": "ack-1",
        },
        "generation": {
            "policy_epoch": 0,
            "takeover_generation": 0,
            "reset_generation": 0,
        },
        "observation": {"clock_domain_id": "upper_host_monotonic"},
        "action_authority": {
            "accepted_absolute_action7": [0.5, 0.0, 0.2, *rotation, 0.085],
            "executed_action_source": source,
            "pose_ack": {
                "accepted": True,
                "upper_receive_monotonic_ns": 1_000_000_000,
                "command_id": "command-1",
            },
            "gripper_terminal_provenance": {
                "origin_action_goal_id": "gripper-1"
            },
            "safety_arbitration": {"workspace_clipped": False},
        },
    }
    if marked_rpy:
        row["absolute_action_rotation_representation"] = (
            ABSOLUTE_ACTION_ROTATION_REPRESENTATION
        )
    return row


def test_online_rotation_conversions_match_model_rpy_chart() -> None:
    half = np.sqrt(0.5)
    assert np.allclose(
        quaternion_xyzw_to_rpy_xyz(np.asarray([0.0, 0.0, half, half])),
        [0.0, 0.0, np.pi / 2.0],
    )
    assert np.allclose(
        rotation_vector_to_rpy_xyz(np.asarray([0.0, 0.0, np.pi / 2.0])),
        [0.0, 0.0, np.pi / 2.0],
    )
    actions = np.zeros((2, 7), dtype=np.float64)
    actions[:, 5] = np.pi / 2.0
    assert np.allclose(
        legacy_absolute_action7_to_rpy_xyz(actions)[:, 3:6],
        [[0.0, 0.0, np.pi / 2.0]] * 2,
    )


def test_legacy_replay_is_converted_and_marked_rpy_is_not() -> None:
    legacy = replay_training._accepted_ack(
        _ack_row(
            [0.0, 0.0, np.pi / 2.0], source="human", marked_rpy=False
        )
    )
    rpy = [0.2, -0.3, 0.4]
    legacy_policy = replay_training._accepted_ack(
        _ack_row(rpy, source="policy", marked_rpy=False)
    )
    current = replay_training._accepted_ack(
        _ack_row(rpy, source="human", marked_rpy=True)
    )

    assert np.allclose(legacy.accepted_absolute_action7[3:6], [0.0, 0.0, np.pi / 2.0])
    assert np.allclose(legacy_policy.accepted_absolute_action7[3:6], rpy)
    assert np.allclose(current.accepted_absolute_action7[3:6], rpy)
