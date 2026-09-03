from __future__ import annotations

from forcesmolvla.rft.online.replay_training import DemoReplay


def test_offline_demo_materialization_marks_actor_q_invalid() -> None:
    replay = DemoReplay.__new__(DemoReplay)
    replay.rows = (
        {
            "episode_id": "episode_000000",
            "transition_index": 0,
            "anchor_frame": 0,
            "next_frame": 3,
            "executed_steps": 3,
            "executed_action_mask": [True, True, True],
            "normalized_delta_action_exec_flat": [0.0] * 21,
            "reward": 0.0,
            "discount": 0.99,
            "terminated": False,
            "truncated": False,
            "bootstrap_mask": True,
            "mc_return": 0.0,
            "split": "train",
            "observation_row_reference": {
                "data_relative_path": "data.parquet",
                "row_index": 0,
            },
            "next_observation_row_reference": {
                "data_relative_path": "data.parquet",
                "row_index": 3,
            },
        },
    )
    replay.tasks = {"episode_000000": "task"}
    replay._sample = lambda *_args: {"sample_identity": "demo"}

    row = replay.materialize(0)

    assert row["action_source"] == "offline_demonstration"
    assert row["actor_q_valid"] is False
    assert row["actor_q_eligibility_reason"] == (
        "offline_demonstration_not_ack_deployment_semantics"
    )
