from __future__ import annotations

import inspect
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_forcerft_online_residual_bootstrap as seed_tool  # noqa: E402
from forcesmolvla.rft.online.residual_actor_critic_runtime import (  # noqa: E402
    exact_resume_checkpoint_is_recoverable,
    prepare_learner,
)
from forcesmolvla.rft.online.residual_actor_critic_checkpoint import (  # noqa: E402
    RESIDUAL_ACTOR_CRITIC_CHECKPOINT_FILES,
)


class TinyBaseActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))


def test_online_residual_bootstrap_needs_no_critic_parent_and_starts_zero(
    tmp_path: Path, monkeypatch,
) -> None:
    base = TinyBaseActor()
    monkeypatch.setattr(seed_tool, "_load_base_actor", lambda _path: base)
    checkpoint = tmp_path / seed_tool.BOOTSTRAP_DIRECTORY_NAME
    result = seed_tool.build_online_residual_bootstrap(
        task_id="task3",
        output_root=tmp_path / "outputs/task3",
        dataset_root=tmp_path / "datasets/task3_lerobotv3",
        frozen_base_policy_checkpoint=tmp_path / "sft-base",
        checkpoint=checkpoint,
        online_residual_config=ROOT
        / "configs/forcerft/online_ack_residual_actor_critic.yaml",
    )

    assert result == checkpoint.resolve()
    assert "critic_checkpoint" not in inspect.signature(
        seed_tool.build_online_residual_bootstrap
    ).parameters
    assert "reward_transition_root" not in inspect.signature(
        seed_tool.build_online_residual_bootstrap
    ).parameters
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert exact_resume_checkpoint_is_recoverable(
        checkpoint, expected_kind="online_residual_bootstrap"
    )
    assert {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file()
    } == set(RESIDUAL_ACTOR_CRITIC_CHECKPOINT_FILES)

    learner = prepare_learner(torch.device("cpu"), resume_checkpoint=checkpoint)
    zeros = learner["residual_actor"](
        normalized_state7=torch.randn(5, 7),
        normalized_wrench6=torch.randn(5, 6),
        normalized_wrench_delta6=torch.randn(5, 6),
        base_action6=torch.randn(5, 6),
    )
    assert torch.equal(zeros, torch.zeros_like(zeros))
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            learner["q1"].state_dict().values(),
            learner["q1_target"].state_dict().values(),
            strict=True,
        )
    )
    runtime = learner["runtime"]
    assert runtime["learner_state"] == "ack_replay_collection"
    assert runtime["ack_critic_warmup_complete"] is False
    assert runtime["online_adaptation_id"].startswith("task3-ack-residual-")
    assert runtime["counters"] == {
        "twin_q_optimizer_steps": 0,
        "residual_actor_optimizer_steps": 0,
        "twin_q_target_update_steps": 0,
    }
    assert runtime["replay"]["loaded_episode_keys"] == []
    assert runtime["replay"]["per_episode_critic_row_counts"] == {}
    assert runtime["replay"]["admission_cycle_budgets"] == {}
    assert runtime["replay"]["replay_generation"] == 0
