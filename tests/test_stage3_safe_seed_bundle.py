from __future__ import annotations

import inspect
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_forcerft_stage3_seed_bundle as seed_tool  # noqa: E402
from forcesmolvla.rft.online.actor_learner_runtime import (  # noqa: E402
    exact_resume_checkpoint_is_recoverable,
    prepare_learner,
)
from forcesmolvla.rft.online.learner_checkpoint import (  # noqa: E402
    RESIDUAL_CHECKPOINT_FILES,
)


class TinyBaseActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))


def test_stage3_seed_needs_no_critic_parent_and_starts_zero(
    tmp_path: Path, monkeypatch,
) -> None:
    base = TinyBaseActor()
    monkeypatch.setattr(seed_tool, "_load_base_actor", lambda _path: base)
    checkpoint = tmp_path / seed_tool.SEED_DIRECTORY_NAME
    result = seed_tool.build_stage3_seed_bundle(
        task_id="task3",
        output_root=tmp_path / "outputs/task3",
        dataset_root=tmp_path / "datasets/task3_lerobotv3",
        base_actor_checkpoint=tmp_path / "sft-base",
        checkpoint=checkpoint,
        common_online_config=ROOT / "configs/forcerft/actor_critic_common.yaml",
    )

    assert result == checkpoint.resolve()
    assert "critic_checkpoint" not in inspect.signature(
        seed_tool.build_stage3_seed_bundle
    ).parameters
    assert "reward_transition_root" not in inspect.signature(
        seed_tool.build_stage3_seed_bundle
    ).parameters
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert exact_resume_checkpoint_is_recoverable(
        checkpoint, expected_kind="stage3_seed"
    )
    assert {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file()
    } == set(RESIDUAL_CHECKPOINT_FILES)

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
    assert runtime["phase"] == "collecting"
    assert runtime["critic_burnin_complete"] is False
    assert runtime["counters"] == {
        "critic_optimizer_steps": 0,
        "actor_optimizer_steps": 0,
        "target_polyak_steps": 0,
    }
