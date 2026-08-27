from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from forcesmolvla.rft.g7b import (
    G7B_CHECKPOINT_MARKERS,
    G7B_COUNTERS,
    describe_p95,
    save_g7b_checkpoint,
    validate_optimizer_step_sets,
    validate_g7b_checkpoint,
)


ROOT = Path(__file__).parents[1]


def test_frozen_g7b_recipe_is_exact() -> None:
    config = yaml.safe_load((ROOT / "configs/stage2_g7b_joint_smoke.development.yaml").read_text())
    joint = config["joint_smoke"]
    assert joint["eta_actor_q"] == 10.0
    assert joint["beta_flow"] == 1.0
    assert joint["joint_cycles"] == 8
    assert joint["expected_critic_updates"] == 16
    assert joint["expected_actor_updates"] == 8
    assert joint["expected_polyak_updates_per_target"] == 16
    assert joint["target_actor"] is None
    assert config["data"]["allowed_split"] == "train"
    assert all(config["data"][key] is False for key in (
        "validation_reads_allowed", "test_reads_allowed", "manual_g1_allowed",
        "manual_labels_allowed", "reward_classifier_inference_allowed",
    ))


def test_statistics_report_exact_median_p95_and_maximum() -> None:
    summary = describe_p95([1.0, 2.0, 3.0, 4.0])
    assert summary == {"count": 4, "median": 2.5, "p95": 3.8499999999999996, "maximum": 4.0}
    with pytest.raises(ValueError, match="G7B_STATISTIC_INPUT_INVALID"):
        describe_p95([float("nan")])


def test_sparse_actor_optimizer_steps_do_not_fake_global_counter_failure() -> None:
    validate_optimizer_step_sets({272}, {3, 8})
    with pytest.raises(RuntimeError, match="G7B_OPTIMIZER_COUNTER_DRIFT"):
        validate_optimizer_step_sets({272}, {3, 7})
    with pytest.raises(RuntimeError, match="G7B_OPTIMIZER_COUNTER_DRIFT"):
        validate_optimizer_step_sets({271}, {8})


def _target() -> torch.nn.Module:
    module = torch.nn.Linear(2, 1)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def test_atomic_checkpoint_is_append_only_and_tamper_evident(tmp_path: Path) -> None:
    actor = torch.nn.Linear(2, 2)
    q1, q2 = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    modules = {"actor": actor, "q1": q1, "q2": q2, "q1_target": _target(), "q2_target": _target()}
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-5)
    critic_optimizer = torch.optim.Adam((*q1.parameters(), *q2.parameters()), lr=3e-4)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lambda _: 1.0)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lambda _: 1.0)
    target = tmp_path / "checkpoint"
    manifest = save_g7b_checkpoint(
        target, modules=modules, actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer, actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler, counters=G7B_COUNTERS,
        parent_counters={"critic_optimizer_updates": 256}, sampler_states={"x": 1},
        rng_states={"x": 2}, ownership_manifest={"intersection": 0},
        protected_snapshot={"frozen": True}, startup_snapshot_bytes={"config/test.txt": b"fixed\n"},
    )
    assert all(manifest[key] == value for key, value in G7B_CHECKPOINT_MARKERS.items())
    assert validate_g7b_checkpoint(target)["counters"] == G7B_COUNTERS
    with pytest.raises(RuntimeError, match="G7B_CHECKPOINT_TARGET_OR_COUNTER_INVALID"):
        save_g7b_checkpoint(
            target, modules=modules, actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer, actor_scheduler=actor_scheduler,
            critic_scheduler=critic_scheduler, counters=G7B_COUNTERS,
            parent_counters={"critic_optimizer_updates": 256}, sampler_states={}, rng_states={},
            ownership_manifest={}, protected_snapshot={}, startup_snapshot_bytes={},
        )
    counter = target / "state/counters.json"
    payload = json.loads(counter.read_text())
    payload["joint_cycles"] = 7
    counter.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="G7B_CHECKPOINT_INTERNAL_FILE_SHA_MISMATCH"):
        validate_g7b_checkpoint(target)
