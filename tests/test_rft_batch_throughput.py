from __future__ import annotations

import pytest

from forcesmolvla.rft.batch_throughput import (
    GIB,
    epoch_budget,
    expanded_actor_candidates,
    select_best,
    stage_a_candidates,
)


def _result(name: str, joint: float, reserved_gib: float, *, finite: bool = True) -> dict:
    return {
        "candidate_id": name,
        "status": "pass",
        "all_finite": finite,
        "action_contract_v2": {"passed": True},
        "peak_reserved_bytes": int(reserved_gib * GIB),
        "throughput": {
            "joint_training_sample_memberships_per_second": joint,
            "actor_samples_per_second": joint / 4,
            "critic_sample_memberships_per_second": joint * 3 / 4,
        },
    }


def test_staged_candidates_preserve_effective_batch() -> None:
    assert stage_a_candidates(4, 16) == [
        {"actor_microbatch": 1, "actor_accumulation": 4, "effective_actor_batch": 4, "critic_batch": 16},
        {"actor_microbatch": 2, "actor_accumulation": 2, "effective_actor_batch": 4, "critic_batch": 16},
        {"actor_microbatch": 4, "actor_accumulation": 1, "effective_actor_batch": 4, "critic_batch": 16},
    ]
    expanded = expanded_actor_candidates(2, [8, 16, 32], 16)
    assert [(item["actor_microbatch"], item["actor_accumulation"]) for item in expanded] == [(2, 4), (2, 8), (2, 16)]
    with pytest.raises(ValueError, match="NOT_DIVISIBLE"):
        expanded_actor_candidates(3, [8], 16)


def test_selection_uses_throughput_with_hard_vram_and_finite_gates() -> None:
    best = select_best([
        _result("slow", 10, 10),
        _result("too_large", 100, 21.1),
        _result("nonfinite", 200, 10, finite=False),
        _result("winner", 20, 20.5),
    ])
    assert best["candidate_id"] == "winner"


def test_epoch_budget_uses_ceil_and_reports_both_critic_batches() -> None:
    budgets = epoch_budget(10075, actor_batch=32, critic_batch=64, cycle_seconds=10.0)
    assert budgets[0]["joint_cycles"] == 315
    assert budgets[1]["joint_cycles"] == 630
    assert budgets[2]["joint_cycles"] == 945
    assert budgets[0]["critic_td_sample_memberships"] == 315 * 2 * 64
    assert budgets[0]["critic_total_sample_memberships"] == 315 * 4 * 64

