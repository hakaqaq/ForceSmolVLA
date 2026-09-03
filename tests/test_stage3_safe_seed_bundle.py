from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_forcerft_stage3_seed_bundle as seed_tool  # noqa: E402
from forcesmolvla.rft.online.actor_learner_runtime import (  # noqa: E402
    exact_resume_checkpoint_is_recoverable,
)


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = torch.nn.Linear(2, 2)

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_bytes(b"tiny")
        (path / "config.json").write_text("{}\n", encoding="utf-8")
        (path / "artifact_manifest.json").write_text("{}\n", encoding="utf-8")


def test_safe_seed_is_explicitly_sft_actor_and_critic_first(
    tmp_path: Path, monkeypatch,
) -> None:
    actor = TinyPolicy()
    modules = {
        name: torch.nn.Linear(2, 1)
        for name in ("q1", "q2", "q1_target", "q2_target")
    }
    critic_optimizer = torch.optim.Adam(
        tuple(modules["q1"].parameters()) + tuple(modules["q2"].parameters()),
        lr=3e-4,
    )
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        critic_optimizer, lambda _step: 1.0
    )
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
        actor_optimizer, lambda _step: 1.0
    )
    monkeypatch.setattr(
        seed_tool.joint,
        "load_offline_training_parents",
        lambda **_kwargs: (
            actor,
            modules["q1"], modules["q2"], modules["q1_target"],
            modules["q2_target"], modules, critic_optimizer, critic_scheduler,
            actor_optimizer, actor_scheduler, {}, {},
        ),
    )
    monkeypatch.setattr(
        "forcesmolvla.checkpoint.export_development_actor_checkpoint",
        lambda **kwargs: kwargs["policy"].save_pretrained(kwargs["destination"]),
    )
    actor_parent = tmp_path / "sft"
    critic_parent = tmp_path / "critic"
    (critic_parent / "state").mkdir(parents=True)
    torch.save(
        {
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": [],
            "named_generator_states": {
                "td_next_action_flow_noise": torch.Generator().get_state()
            },
        },
        critic_parent / "state/rng_states.pt",
    )
    actor_parent.mkdir()
    normalizer = tmp_path / "task_normalizer.json"
    action_contract = tmp_path / "critic_action_contract.json"
    common_config = tmp_path / "actor_critic_common.yaml"
    for path in (normalizer, action_contract, common_config):
        path.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / seed_tool.SEED_DIRECTORY_NAME

    seed_tool.build_stage3_seed_bundle(
        task_id="task3",
        output_root=tmp_path / "outputs/task3",
        dataset_root=tmp_path / "datasets/task3_lerobotv3",
        reward_transition_root=tmp_path / "datasets/task3_reward_transitions",
        actor_checkpoint=actor_parent,
        critic_checkpoint=critic_parent,
        checkpoint=checkpoint,
        normalizer=normalizer,
        action_contract=action_contract,
        common_online_config=common_config,
    )

    metadata = json.loads((checkpoint / "metadata.json").read_text())
    runtime = torch.load(
        checkpoint / "state/runtime_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert metadata["kind"] == "stage3_safe_seed_v1"
    assert metadata["actor_source_kind"] == "sft"
    assert metadata["actor_equal_to_sft"] is True
    assert metadata["actor_updates_enabled"] is False
    assert metadata["legacy_actor210_parent"] is False
    assert runtime["counters"]["actor_optimizer_steps"] == 0
    assert runtime["flags"]["critic_updates_enabled"] is True
    assert exact_resume_checkpoint_is_recoverable(
        checkpoint, expected_kind="stage3_safe_seed_v1"
    )

