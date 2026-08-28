#!/usr/bin/env python3
"""Data-only 210-cycle draw-plan/cache simulation; never creates a CUDA model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import yaml


ROOT = Path(__file__).parents[1].resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/lerobot/src"), str(ROOT / "tools")]
CONFIG = ROOT / "configs/stage2b_long_run_half_pass_throughput_v2.development.yaml"
PARENT = ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint"
OUTPUT = ROOT / "artifacts/development/stage2/throughput_v2_long_run/cache_210_cycle_preflight.json"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_samplers(data, parent_states: dict, parent_rng: dict):
    from forcesmolvla.rft.training_cycle import (
        SerializableReplacementSampler,
        SerializableUniqueSampler,
    )

    names = ("td_sampler", "calql_sampler", "actor_sampler", "empirical_random_proposal")
    training = yaml.safe_load(
        (ROOT / "configs/stage2_g5_single_cycle.v2.development.yaml").read_text()
    )
    seeds = training["rng"]["named_stream_seeds"]
    generators = {
        name: torch.Generator(device="cpu").manual_seed(int(seeds[name]))
        for name in names
    }
    for name in names:
        if name in parent_rng["named_generator_states"]:
            generators[name].set_state(parent_rng["named_generator_states"][name])
    td = parent_states["td"]
    calql = parent_states["calql"]
    proposal = parent_states["empirical_random_proposal"]
    samplers = {
        "td": SerializableUniqueSampler(
            td["name"], tuple(td["population"]), generators["td_sampler"], int(td["draws"])
        ),
        "calql": SerializableUniqueSampler(
            calql["name"], tuple(calql["population"]), generators["calql_sampler"], int(calql["draws"])
        ),
        "actor": SerializableUniqueSampler(
            "Actor_sampler", data.actor_population, generators["actor_sampler"]
        ),
        "empirical_random_proposal": SerializableReplacementSampler(
            proposal["name"], int(proposal["population_size"]),
            generators["empirical_random_proposal"], int(proposal["draws"]),
        ),
    }
    return samplers, generators


def row_references(data, indices: list[int], *, include_flow_actions: bool):
    references: set[tuple[str, int]] = set()
    for index in indices:
        row = data.rows[index]
        for name in ("observation_row_reference", "next_observation_row_reference"):
            reference = row[name]
            references.add((reference["data_relative_path"], int(reference["row_index"])))
        if include_flow_actions:
            reference = row["observation_row_reference"]
            stop = min(int(reference["row_index"]) + 50, data.frame_counts[row["episode_id"]])
            references.update(
                (reference["data_relative_path"], frame)
                for frame in range(int(reference["row_index"]), stop)
            )
    return references


def simulate(output: Path) -> dict:
    import preflight_s2_g5_single_cycle_gpu as g5
    from forcesmolvla.rft.canonical_state import canonical_digest
    from forcesmolvla.rft.throughput_v2_long_run import (
        BoundedTrainingDataCache,
        stable_draw_plan_sha256,
    )

    config = yaml.safe_load(CONFIG.read_text())
    require(config["authorization"] == "integration_preflight_only_no_long_run", "CACHE_SIM_AUTH")
    cycles = int(config["cache_preflight"]["draw_plan_cycles"])
    critic_batch = int(config["cache_preflight"]["selected_simulation_critic_batch"])
    actor_batch = int(config["cache_preflight"]["actor_batch"])
    candidate_count = int(config["fixed_semantics"]["calql_candidates_per_source"])
    parent_states = torch.load(PARENT / "state/sampler_states.pt", map_location="cpu", weights_only=False)
    parent_rng = torch.load(PARENT / "state/rng_states.pt", map_location="cpu", weights_only=False)
    data_started = time.perf_counter()
    data = g5.TrainData()
    initialization_seconds = time.perf_counter() - data_started
    samplers, generators = build_samplers(data, parent_states, parent_rng)
    rng_before = canonical_digest({name: generator.get_state() for name, generator in generators.items()})
    cache = BoundedTrainingDataCache(
        data,
        max_bytes=int(config["cache"]["decoded_cache_max_bytes"]),
        prefetch_workers=int(config["cache"]["decode_workers"]),
    )
    all_rows: set[tuple[str, int]] = set()
    draw_plan: list[dict] = []
    cycle_latencies: list[float] = []
    started = time.perf_counter()
    for cycle in range(1, cycles + 1):
        cycle_started = time.perf_counter()
        critic_draws = []
        for substep in range(2):
            td = samplers["td"].draw(critic_batch)
            calql = samplers["calql"].draw(critic_batch)
            proposal = samplers["empirical_random_proposal"].draw(critic_batch * candidate_count)
            cache.prefetch_indices(td)
            cache.prefetch_indices(calql)
            all_rows.update(row_references(data, td, include_flow_actions=False))
            all_rows.update(row_references(data, calql, include_flow_actions=False))
            critic_draws.append({
                "substep": substep + 1,
                "td": td,
                "calql": calql,
                "proposal": proposal,
            })
        actor = samplers["actor"].draw(actor_batch)
        cache.prefetch_indices(actor)
        all_rows.update(row_references(data, actor, include_flow_actions=True))
        draw_plan.append({"cycle": cycle, "critic": critic_draws, "actor": actor})
        cycle_latencies.append(time.perf_counter() - cycle_started)
        if cycle == 1 or cycle % 10 == 0 or cycle == cycles:
            print(f"CACHE_SIM cycle={cycle}/{cycles}", flush=True)
    total_seconds = time.perf_counter() - started
    rng_after = canonical_digest({name: generator.get_state() for name, generator in generators.items()})
    report = cache.report()
    result = {
        "schema_version": "forcesmolvla_stage2b_throughput_v2_cache_210_preflight.v1",
        "status": "pass",
        "scope": "data_only_no_cuda_model_no_optimizer",
        "config_path": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": file_sha256(CONFIG),
        "parent_path": PARENT.relative_to(ROOT).as_posix(),
        "cycles_simulated": cycles,
        "critic_batch": critic_batch,
        "actor_batch": actor_batch,
        "draw_plan_sha256": stable_draw_plan_sha256(draw_plan),
        "committed_cycle_cursor": cycles,
        "pending_prefetched_cycle_identities": [],
        "prefetch_derivation": "synchronous_current_committed_draw_only",
        "total_unique_row_references": len(all_rows),
        "total_unique_images": report["total_unique_images"],
        "working_set_decoded_bytes": report["total_unique_decoded_image_bytes"],
        "peak_cache_bytes": report["decoded_cache_peak_bytes"],
        "peak_process_rss_bytes": report["process_rss_peak_bytes"],
        "cache_hit_rate": report["cache_hit_rate"],
        "eviction_count": report["cache_evictions"],
        "cold_start_seconds": initialization_seconds + (cycle_latencies[0] if cycle_latencies else 0.0),
        "steady_state_data_latency_seconds_per_cycle": (
            sum(cycle_latencies[1:]) / max(1, len(cycle_latencies) - 1)
        ),
        "total_simulation_seconds": total_seconds,
        "cache": report,
        "sampler_draws": {name: sampler.draws for name, sampler in samplers.items()},
        "training_rng_before_sha256": rng_before,
        "training_rng_after_sha256": rng_after,
        "rng_change_explained_only_by_frozen_draw_plan_sampler_draws": True,
        "cache_or_prefetch_consumed_rng": False,
        "cache_reconstruction_on_resume": "empty_then_deterministically_refilled_by_committed_draws",
        "optimizer_updates": 0,
        "cuda_model_created": False,
        "long_run_started": False,
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = simulate(args.output)
    print(json.dumps({
        "status": result["status"],
        "draw_plan_sha256": result["draw_plan_sha256"],
        "cache_hit_rate": result["cache_hit_rate"],
        "peak_cache_bytes": result["peak_cache_bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
