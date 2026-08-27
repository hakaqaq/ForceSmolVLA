from __future__ import annotations

from forcesmolvla.rft.batch_scaling import actor_pass_table, aggregate_repeats, select_by_samples_per_second


def _repeat(batch: int, rate: float, path: str) -> dict:
    return {
        "physical_batch_size": batch, "status": "pass", "all_finite": True,
        "contract_valid": True, "samples_per_second": rate,
        "seconds_per_update": {"median": batch / rate},
        "peak_allocated_bytes": 1, "peak_reserved_bytes": 2,
        "gpu_utilization_percent": {"mean": 90}, "gpu_power_watts": {"mean": 300},
        "result_path": path,
    }


def test_repeat_aggregation_and_five_percent_selection() -> None:
    aggregates = []
    for batch, rate in ((4, 10.0), (8, 10.4), (16, 12.0)):
        aggregates.append(aggregate_repeats(batch, [_repeat(batch, rate, f"{batch}-{i}") for i in range(3)], "samples_per_second"))
    selected = select_by_samples_per_second(aggregates, "samples_per_second", total_memory_bytes=100)
    assert selected["physical_batch_size"] == 16
    assert aggregates[1]["incremental_gain_over_previous"] < 0.05
    assert aggregates[2]["incremental_gain_over_previous"] > 0.05


def test_actor_pass_table_matches_frozen_contract() -> None:
    assert actor_pass_table() == [
        {"effective_actor_batch": 4, "half_pass_updates": 1260, "one_pass_updates": 2519, "two_pass_updates": 5038},
        {"effective_actor_batch": 8, "half_pass_updates": 630, "one_pass_updates": 1260, "two_pass_updates": 2519},
        {"effective_actor_batch": 16, "half_pass_updates": 315, "one_pass_updates": 630, "two_pass_updates": 1260},
        {"effective_actor_batch": 24, "half_pass_updates": 210, "one_pass_updates": 420, "two_pass_updates": 840},
        {"effective_actor_batch": 32, "half_pass_updates": 158, "one_pass_updates": 315, "two_pass_updates": 630},
        {"effective_actor_batch": 48, "half_pass_updates": 105, "one_pass_updates": 210, "two_pass_updates": 420},
        {"effective_actor_batch": 64, "half_pass_updates": 79, "one_pass_updates": 158, "two_pass_updates": 315},
    ]
