from __future__ import annotations

from forceprior.rft.online.replay_training import (
    algorithm_hyperparameters,
    load_common_actor_critic_config,
)


def test_task2_and_task3_share_one_algorithm_contract() -> None:
    task2 = load_common_actor_critic_config("task2")
    task3 = load_common_actor_critic_config("task3")

    assert task2["task"]["task_id"] == "task2"
    assert task3["task"]["task_id"] == "task3"
    assert task2["paths"]["lerobot_v3_root"] != task3["paths"]["lerobot_v3_root"]
    assert algorithm_hyperparameters(task2) == algorithm_hyperparameters(task3)


def test_task_profiles_cannot_override_algorithm_hyperparameters() -> None:
    task2 = load_common_actor_critic_config("task2")

    assert task2["optimizer"]["residual_actor"]["lr"] == 3.0e-5
    assert task2["ack_critic_warmup"] == {
        "minimum_ack_transitions": 1000,
        "minimum_admitted_episodes": 3,
        "optimizer_steps": 256,
    }
    assert task2["residual_actor_critic_training"]["new_td_rows_per_cycle"] == 8
    assert "q_gradient_controller" not in task2
    assert "actor_unlock" not in task2
