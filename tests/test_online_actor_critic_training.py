from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_forcerft_actor_critic import (  # noqa: E402
    assert_optimizer_ownership,
    load_joint_checkpoint_once,
    load_resume_modules,
    make_schedules,
    save_joint_checkpoint,
)
from forcesmolvla.rft.online.sample_credit import UpdateCreditLedger  # noqa: E402
from forcesmolvla.training_runtime import resolve_task_output_root  # noqa: E402


def test_task_output_root_is_canonical(tmp_path: Path) -> None:
    assert resolve_task_output_root(tmp_path, task_id="task2") == (
        tmp_path / "outputs/task2"
    ).resolve()
    explicit = tmp_path / "custom"
    assert resolve_task_output_root(
        tmp_path, task_id="task2", output_root=explicit
    ) == explicit.resolve()


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = torch.nn.Linear(3, 3)
        self.frozen = torch.nn.Linear(3, 3)
        for parameter in self.frozen.parameters():
            parameter.requires_grad_(False)

    def save_pretrained(self, path: Path) -> None:
        from safetensors.torch import save_file

        path.mkdir(parents=True)
        save_file({name: value.detach().contiguous() for name, value in self.state_dict().items()}, path / "model.safetensors")
        (path / "config.json").write_text("{}\n")


class TinyCritic(torch.nn.Linear):
    def make_permanent_eval_target(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def _step(module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    module(torch.ones(1, 3)).sum().backward()
    optimizer.step()


def test_joint_sampler_continues_rng_state() -> None:
    r_rng = random.Random(7)
    d_rng = random.Random(8)
    r_rng.sample(range(100), 32)
    d_rng.sample(tuple(range(200)), 32)
    r_state, d_state = r_rng.getstate(), d_rng.getstate()

    schedules = make_schedules(
        r_rng,
        d_rng,
        r_population_size=100,
        d_population=tuple(range(200)),
        cycles=2,
    )
    restored_r, restored_d = random.Random(), random.Random()
    restored_r.setstate(r_state)
    restored_d.setstate(d_state)
    repeated = make_schedules(
        restored_r,
        restored_d,
        r_population_size=100,
        d_population=tuple(range(200)),
        cycles=2,
    )

    assert schedules == repeated
    assert [len(batch) for batch in schedules[0]] == [32] * 4
    assert [len(batch) for batch in schedules[2]] == [12] * 2


def test_optimizer_ownership_is_disjoint() -> None:
    actor = TinyPolicy()
    critic = torch.nn.Linear(3, 1)
    actor_optimizer = torch.optim.AdamW(actor.trainable.parameters(), lr=1e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    assert_optimizer_ownership(
        actor_optimizer,
        critic_optimizer,
        frozen_parameters=actor.frozen.parameters(),
    )


def test_resume_modules_reads_exact_resume_actor_and_canonical_training_config(
    tmp_path: Path, monkeypatch,
) -> None:
    checkpoint = tmp_path / "joint"
    actor_package = checkpoint / "actor"
    actor_package.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(json.dumps({
        "kind": "offline_actor_critic_exact_resume",
        "actor_directory": "actor",
    }))
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        "data:\n  critic_backbone_npz: backbone.npz\n"
        "  critic_backbone_manifest: backbone.json\n"
    )
    monkeypatch.setattr(
        "train_forcerft_actor_critic.warmup.TRAINING_CONFIG", config_path
    )
    monkeypatch.setattr(
        "forcesmolvla.modeling_forcesmolvla.ForceSmolVLAPolicy.from_pretrained",
        lambda *_args, **_kwargs: TinyPolicy(),
    )
    monkeypatch.setattr(
        "forcesmolvla.rft.critic.build_twin_q",
        lambda *_args, **_kwargs: (
            TinyCritic(3, 1), TinyCritic(3, 1),
            TinyCritic(3, 1), TinyCritic(3, 1), None,
        ),
    )

    *_, binding, config = load_resume_modules(
        checkpoint,
        actor_package,
        torch.device("cpu"),
    )

    assert binding["normalizer_binding"]["absolute_path"] == str(
        checkpoint / "artifacts/normalizer_manifest.json"
    )
    assert config["data"]["critic_backbone_npz"] == "backbone.npz"


def test_joint_checkpoint_round_trip(tmp_path: Path, monkeypatch) -> None:
    def export_tiny_actor(**kwargs) -> None:
        kwargs["policy"].save_pretrained(kwargs["destination"])
        (kwargs["destination"] / "candidate.json").write_text(
            json.dumps(
                {
                    "revision_id": kwargs["candidate_revision_id"],
                    "state": "candidate",
                    "published": False,
                    "activated": False,
                }
            )
        )

    monkeypatch.setattr(
        "forcesmolvla.checkpoint.export_development_actor_checkpoint",
        export_tiny_actor,
    )
    actor = TinyPolicy()
    modules = {
        "q1": torch.nn.Linear(3, 1),
        "q2": torch.nn.Linear(3, 1),
        "q1_target": torch.nn.Linear(3, 1),
        "q2_target": torch.nn.Linear(3, 1),
    }
    critic_optimizer = torch.optim.Adam(
        tuple(modules["q1"].parameters()) + tuple(modules["q2"].parameters()), lr=3e-4,
    )
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        critic_optimizer, lambda _step: 1.0
    )
    actor_optimizer = torch.optim.AdamW(actor.trainable.parameters(), lr=1e-5)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lambda _step: 1.0)
    _step(modules["q1"], critic_optimizer)
    critic_scheduler.step()
    _step(actor.trainable, actor_optimizer)
    actor_scheduler.step()
    credits = UpdateCreditLedger(credits_per_transition=1, credits_per_joint_cycle=1)
    for uid in ("a", "b", "c"):
        credits.mint_for_unique_online_transition(uid)
    credits.consume_joint_cycle()
    runtime = {
        "sample_credit": credits.state_dict(),
        "sampler_state": {
            "r_rng": random.Random(1).getstate(),
            "d_rng": random.Random(2).getstate(),
        },
        "counters": {"joint_cycles": 10},
        "runtime_artifacts": {
            "normalizer": str(tmp_path / "normalizer.json"),
            "action_contract": str(tmp_path / "action_contract.json"),
        },
        "step_metrics": {"critic_td_loss": [float(index) for index in range(20)]},
    }
    (tmp_path / "normalizer.json").write_text("{}\n")
    (tmp_path / "action_contract.json").write_text("{}\n")
    checkpoint = tmp_path / "joint"

    save_joint_checkpoint(
        checkpoint,
        actor=actor,
        modules=modules,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        runtime_state=runtime,
        parent_binding=None,
        actor_parent_path=tmp_path,
        parent_binding_id="task2-offline-exact-resume",
        source_checkpoint=tmp_path / "offline_parent",
        total_joint_cycles=10,
        actor_checkpoint_id="offline-actor-critic-cycle-000210",
        checkpoint_kind="offline_actor_critic_exact_resume",
        actor_directory="actor",
    )
    restored = load_joint_checkpoint_once(
        checkpoint,
        actor=actor,
        modules=modules,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        device=torch.device("cpu"),
    )

    assert restored["counters"]["joint_cycles"] == 10
    assert restored["step_metrics"]["critic_td_loss"] == [
        float(index) for index in range(20)
    ]
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["kind"] == "offline_actor_critic_exact_resume"
    assert metadata["actor_directory"] == "actor"
    assert metadata["source_checkpoint"].endswith("offline_parent")
    assert metadata["joint_cycles"] == 10
    assert metadata["actor_optimizer_restored"] is True
    assert (checkpoint / "artifacts/normalizer_manifest.json").is_file()
    assert (checkpoint / "artifacts/action_delta_spec.json").is_file()
    assert critic_optimizer.state and actor_optimizer.state
    assert critic_scheduler.last_epoch == 1 and actor_scheduler.last_epoch == 1
