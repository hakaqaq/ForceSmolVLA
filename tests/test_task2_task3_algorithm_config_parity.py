from __future__ import annotations

from forcesmolvla.rft.online.replay_training import (
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

    assert task2["optimizer"]["actor"]["lr"] == 1.0e-4
    assert task2["warmup"] == {
        "training_starts": 100,
        "critic_burnin_updates": 256,
    }
    assert "q_gradient_controller" not in task2
    assert "actor_unlock" not in task2
