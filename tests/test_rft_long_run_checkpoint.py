from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from forcesmolvla.rft.long_run_checkpoint import (
    MARKERS,
    counters_for_cycle,
    hardlink_milestone,
    save_cycle_checkpoint,
    validate_cycle_checkpoint,
)


ROOT = Path(__file__).parents[1]


def test_long_run_recipe_and_parent_are_frozen() -> None:
    config = yaml.safe_load((ROOT / "configs/stage2_g7_long_run_stage1.development.yaml").read_text())
    assert config["parent"]["path"].endswith("g7a_r2_critic_warmup_checkpoint")
    assert config["parent"]["actor_optimizer_updates"] == 0
    assert config["parent"]["g7b_smoke_checkpoint_used_as_parent"] is False
    assert config["recipe"] == {
        "joint_cycles": 256,
        "critic_updates_per_cycle": 2,
        "actor_updates_per_cycle": 1,
        "expected_critic_updates": 512,
        "expected_actor_updates": 256,
        "expected_polyak_updates_per_target": 512,
        "eta_q": 10.0,
        "eta_status": "development_only",
        "beta_flow": 1.0,
        "frozen_cycle_config": "configs/stage2_g5_single_cycle.v2.development.yaml",
        "frozen_cycle_config_sha256": "a728c4544c11f3ff15ba2b3b7ceca9cea7a068169ddc3913fa5707127f0f0fd0",
    }
    assert config["diagnostics"]["validation_cycles"] == [0, 64, 128, 256]
    assert config["gradient_dominance_fail_closed"]["rolling_window_cycles"] == 32
    assert config["gradient_dominance_fail_closed"]["rolling_median_maximum"] == 1.0


def test_counter_formula() -> None:
    assert counters_for_cycle(0)["critic_optimizer_updates"] == 0
    assert counters_for_cycle(256) == {
        "joint_cycles": 256,
        "critic_optimizer_updates": 512,
        "actor_optimizer_updates": 256,
        "q1_target_polyak_updates": 512,
        "q2_target_polyak_updates": 512,
        "critic_scheduler_steps": 512,
        "actor_scheduler_steps": 256,
        "actor_target_updates": 0,
    }
    with pytest.raises(ValueError, match="G7_LONG_RUN_CYCLE_OUT_OF_RANGE"):
        counters_for_cycle(257)


def _target() -> torch.nn.Module:
    module = torch.nn.Linear(2, 1).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def test_rolling_replace_preserves_hardlinked_milestone(tmp_path: Path) -> None:
    actor, q1, q2 = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    modules = {"actor": actor, "q1": q1, "q2": q2, "q1_target": _target(), "q2_target": _target()}
    actor_optimizer = torch.optim.AdamW(actor.parameters())
    critic_optimizer = torch.optim.Adam((*q1.parameters(), *q2.parameters()))
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lambda _: 1.0)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lambda _: 1.0)
    rolling = tmp_path / "recovery_latest"

    def save(cycle: int) -> None:
        save_cycle_checkpoint(
            rolling, cycle=cycle, modules=modules,
            actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
            actor_scheduler=actor_scheduler, critic_scheduler=critic_scheduler,
            sampler_states={"draws": cycle}, rng_states={"fixed": cycle},
            ownership_manifest={"intersection": 0}, protected_snapshot={"frozen": True},
            startup_snapshot_bytes={"config/fixed.txt": b"fixed\n"}, replace_rolling=True,
        )

    save(0)
    milestone = tmp_path / "milestone_cycle_000000"
    hardlink_milestone(rolling, milestone, expected_cycle=0)
    original = validate_cycle_checkpoint(milestone, expected_cycle=0)["manifest_payload_sha256"]
    save(32)
    assert validate_cycle_checkpoint(rolling, expected_cycle=32)["cycle"] == 32
    assert validate_cycle_checkpoint(milestone, expected_cycle=0)["manifest_payload_sha256"] == original
    assert all(validate_cycle_checkpoint(milestone, expected_cycle=0)[key] == value for key, value in MARKERS.items())
