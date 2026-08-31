from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
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
from forcesmolvla.rft.online import replay_training  # noqa: E402
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
    assert [len(batch) for batch in schedules[1]] == [32] * 4
    assert [len(batch) for batch in schedules[2]] == [12] * 2
    assert [len(batch) for batch in schedules[3]] == [12] * 2


def test_loader_partitions_human_expert_from_policy_training_start(
    tmp_path: Path,
) -> None:
    root = tmp_path / "online"
    (root / "admissions").mkdir(parents=True)
    (root / "replay").mkdir()
    (root / "episodes").mkdir()
    episode = tmp_path / "episode_000000"
    episode.mkdir()
    admission = {
        "policy_execution_smoke_bridge": "PASS",
        "source_episode_semantics": {
            "formal_replay": False,
            "real_online_r": False,
        },
        "episode_id": episode.name,
        "source_episode": str(episode),
        "accepted_unique_r_transition_count": 101,
    }
    (root / "admissions/episode.json").write_text(json.dumps(admission))
    (root / "episodes/episode.json").write_text(
        json.dumps(
            {"episode_id": episode.name, "status": "SEALED_COMMITTED"}
        )
    )
    eligibility = {
        "formal_replay": True,
        "formal_training_replay_eligible": True,
        "real_online_r": True,
        "replay_membership": "R_online",
    }
    policy_rows = []
    for sequence in range(100):
        policy_rows.append(
            {
                "classification": "recorded_live_policy_execution_smoke",
                "action_source": "policy",
                "expert": False,
                "intervention": False,
                "identity": {
                    "episode_id": episode.name,
                    "decision_id": sequence,
                    "transition_uid": f"policy-{sequence}",
                },
                "generation": {
                    "policy_epoch": 0,
                    "takeover_generation": 0,
                    "reset_generation": 0,
                },
                "policy_lineage": {"selection": {"sequence": sequence}},
                "action_authority": {"executed_action_source": "policy"},
                "outcome": {"terminated": sequence == 99},
                "eligibility": eligibility,
            }
        )
    mask = [[False] * 7 for _ in range(50)]
    mask[1] = [True] * 7
    human = {
        "classification": "recorded_live_policy_execution_smoke",
        "action_source": "human",
        "expert": True,
        "intervention": True,
        "human_action_target": [[0.0] * 7 for _ in range(50)],
        "human_action_valid_mask": mask,
        "identity": {
            "episode_id": episode.name,
            "decision_id": 100,
            "transition_uid": "human-100",
        },
        "generation": {
            "policy_epoch": 1,
            "takeover_generation": 1,
            "reset_generation": 0,
        },
        "action_authority": {"executed_action_source": "human"},
        "outcome": {"terminated": False},
        "eligibility": eligibility,
    }
    for row in [*policy_rows, human]:
        envelope = {"episode_sealed": True, "payload": row}
        (root / "replay" / f"{row['identity']['transition_uid']}.json").write_text(
            json.dumps(envelope)
        )

    policies, macros, sources, humans = replay_training.load_formal_online_r(root)

    assert len(policies) == 100 and len(macros) == 98
    assert policies[0]["expert"] is False
    assert np.asarray(policies[0]["action_target"]).shape == (50, 7)
    assert not np.asarray(policies[0]["action_valid_mask"]).any()
    assert sources == {episode.name: episode}
    assert len(humans) == 1
    assert humans[0]["action_source"] == "human"
    assert humans[0]["expert"] is True
    assert np.asarray(humans[0]["action_target"]).shape == (50, 7)
    assert np.asarray(humans[0]["action_valid_mask"]).sum() == 7
    assert replay_training.count_sealed_autonomous_policy_transitions(root) == 100
    (root / "replay/unsealed-policy.json").write_text(
        json.dumps({"episode_sealed": False, "payload": policy_rows[0]})
    )
    assert replay_training.count_sealed_autonomous_policy_transitions(root) == 100


def test_human_replay_builds_masked_h50_target_with_offline_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    class IdentityNormalizer:
        @staticmethod
        def apply(value):
            return np.asarray(value)

    monkeypatch.setattr(
        replay_training,
        "_decode_path",
        lambda _path: np.zeros((3, 480, 640), dtype=np.uint8),
    )
    state = np.asarray([0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 0.085])
    target = np.zeros((50, 7), dtype=np.float64)
    target[1] = state + np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mask = np.zeros((50, 7), dtype=np.bool_)
    mask[1] = True
    observation = {
        "camera_external": {"blob_reference": "external.jpg"},
        "camera_wrist": {"blob_reference": "wrist.jpg"},
        "state7_absolute": state.tolist(),
        "wrench6_calibrated_tcp": [0.0] * 6,
    }
    row = {
        "identity": {
            "episode_id": "episode_000000",
            "transition_uid": "human-1",
        },
        "observation": observation,
        "next_observation": observation,
        "action_target": target.tolist(),
        "action_valid_mask": mask.tolist(),
        "outcome": {
            "reward": 0.0,
            "terminated": False,
            "bootstrap_mask": 1.0,
            "discount": 0.99,
        },
    }
    normalizer = SimpleNamespace(
        state7=IdentityNormalizer(),
        wrench6=IdentityNormalizer(),
        delta_action7=IdentityNormalizer(),
    )
    replay = replay_training.HumanCorrectionReplay(
        [row], {"episode_000000": tmp_path}, normalizer
    )

    sample = replay.materialize(0)

    assert sample["expert"] is True and sample["action_source"] == "human"
    assert sample["action_target"].shape == (50, 7)
    assert sample["action_valid_mask"].sum() == 7
    assert sample["behavior_mask"].tolist() == [False, True, False]
    assert sample["action_target"][1, 0] == pytest.approx(0.01)


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
