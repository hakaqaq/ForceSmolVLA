"""Pure planning and selection rules for the Stage-2 batch benchmark."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Any


GIB = 1024 ** 3


def stage_a_candidates(effective_actor_batch: int, critic_batch: int) -> list[dict[str, int]]:
    layouts = []
    for microbatch in (1, 2, 4):
        if effective_actor_batch % microbatch == 0:
            layouts.append({
                "actor_microbatch": microbatch,
                "actor_accumulation": effective_actor_batch // microbatch,
                "effective_actor_batch": effective_actor_batch,
                "critic_batch": critic_batch,
            })
    return layouts


def expanded_actor_candidates(
    physical_microbatch: int, effective_batches: Iterable[int], critic_batch: int
) -> list[dict[str, int]]:
    candidates = []
    for effective in effective_batches:
        if effective % physical_microbatch:
            raise ValueError("ACTOR_EFFECTIVE_BATCH_NOT_DIVISIBLE_BY_MICROBATCH")
        candidates.append({
            "actor_microbatch": physical_microbatch,
            "actor_accumulation": effective // physical_microbatch,
            "effective_actor_batch": effective,
            "critic_batch": critic_batch,
        })
    return candidates


def eligible(result: Mapping[str, Any], *, maximum_reserved_gib: float = 21.0) -> bool:
    return bool(
        result.get("status") == "pass"
        and result.get("all_finite") is True
        and result.get("action_contract_v2", {}).get("passed") is True
        and int(result.get("peak_reserved_bytes", 1 << 63))
        <= int(maximum_reserved_gib * GIB)
    )


def select_best(results: Iterable[Mapping[str, Any]], *, maximum_reserved_gib: float = 21.0) -> Mapping[str, Any]:
    possible = [result for result in results if eligible(result, maximum_reserved_gib=maximum_reserved_gib)]
    if not possible:
        raise RuntimeError("NO_ELIGIBLE_BATCH_BENCHMARK_CANDIDATE")
    return max(possible, key=lambda result: (
        float(result["throughput"]["joint_training_sample_memberships_per_second"]),
        float(result["throughput"]["actor_samples_per_second"]),
        float(result["throughput"]["critic_sample_memberships_per_second"]),
        -int(result["peak_reserved_bytes"]),
    ))


def epoch_budget(train_rows: int, actor_batch: int, critic_batch: int, cycle_seconds: float) -> list[dict[str, Any]]:
    if min(train_rows, actor_batch, critic_batch) <= 0 or not math.isfinite(cycle_seconds) or cycle_seconds <= 0:
        raise ValueError("INVALID_EPOCH_BUDGET_INPUT")
    one_epoch_cycles = math.ceil(train_rows / actor_batch)
    budgets = []
    for epochs in (1, 2, 3):
        cycles = epochs * one_epoch_cycles
        budgets.append({
            "actor_epochs": epochs,
            "joint_cycles": cycles,
            "estimated_seconds": cycles * cycle_seconds,
            "estimated_hours": cycles * cycle_seconds / 3600.0,
            "actor_sample_memberships": cycles * actor_batch,
            "critic_td_sample_memberships": cycles * 2 * critic_batch,
            "critic_calql_sample_memberships": cycles * 2 * critic_batch,
            "critic_total_sample_memberships": cycles * 4 * critic_batch,
        })
    return budgets

