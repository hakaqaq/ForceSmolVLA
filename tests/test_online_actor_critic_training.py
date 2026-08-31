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


def test_resume_modules_reads_canonical_training_config(
    tmp_path: Path, monkeypatch,
) -> None:
    checkpoint = tmp_path / "joint"
    actor_package = checkpoint / "candidate_policy"
    actor_package.mkdir(parents=True)
    (actor_package / "candidate.json").write_text(json.dumps({
        "source_joint_checkpoint": str(checkpoint),
        "state": "candidate",
        "published": False,
        "activated": False,
    }))
    binding_path = tmp_path / "binding.json"
    binding_path.write_text("{}\n")
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        "data:\n  critic_backbone_npz: backbone.npz\n"
        "  critic_backbone_manifest: backbone.json\n"
    )
    monkeypatch.setattr(
        "train_forcerft_actor_critic.warmup.PARENT_BINDING", binding_path
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
        allow_checkpoint_candidate=True,
    )

    assert binding == {}
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
    actor_optimizer = torch.optim.AdamW(actor.trainable.parameters(), lr=1e-5)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lambda _step: 1.0)
    _step(modules["q1"], critic_optimizer)
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
        "step_metrics": {"critic_td_loss": [float(index) for index in range(20)]},
    }
    binding = {
        "binding_id": "approved_hybrid_cycle210_actor_g7a_r2_twin_q.v1",
        "actor_parent": {
            "architecture_binding": {"container_path": str(tmp_path / "unused")}
        },
    }
    checkpoint = tmp_path / "joint"

    save_joint_checkpoint(
        checkpoint,
        actor=actor,
        modules=modules,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        runtime_state=runtime,
        parent_binding=binding,
        source_checkpoint=tmp_path / "cycle_000010",
        total_joint_cycles=20,
        candidate_revision_id="stage3-online-r-joint-cycle-000020-candidate",
    )
    restored = load_joint_checkpoint_once(
        checkpoint,
        actor=actor,
        modules=modules,
        critic_optimizer=critic_optimizer,
        actor_optimizer=actor_optimizer,
        actor_scheduler=actor_scheduler,
        device=torch.device("cpu"),
    )

    assert restored["counters"]["joint_cycles"] == 10
    assert restored["step_metrics"]["critic_td_loss"] == [
        float(index) for index in range(20)
    ]
    candidate = json.loads((checkpoint / "candidate_policy/candidate.json").read_text())
    assert candidate["revision_id"] == "stage3-online-r-joint-cycle-000020-candidate"
    assert candidate["state"] == "candidate"
    assert candidate["activated"] is False
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["source_checkpoint"].endswith("cycle_000010")
    assert metadata["joint_cycles"] == 20
    assert metadata["actor_optimizer_restored"] is True
