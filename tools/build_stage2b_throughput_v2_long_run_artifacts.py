#!/usr/bin/env python3
"""Build append-only final evidence for throughput-v2 long-run integration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
RUN = ROOT / "artifacts/development/stage2/throughput_v2_long_run"
SUMMARY = RUN / "integration_summary.json"
MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v29_stage2b_throughput_v2_long_run.json"
ARTIFACT = ROOT / "artifacts/development/stage2/s2_stage2b_throughput_v2_long_run_integration.v1.json"
REPORT = ROOT / "docs/stage2b_throughput_v2_long_run_integration_report.v1.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(relative: str, role: str) -> dict:
    path = ROOT / relative
    return {
        "relative_path": relative,
        "artifact_role": role,
        "file_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def checkpoint_tree(path: Path) -> dict:
    digest = hashlib.sha256()
    total = 0
    count = 0
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file.relative_to(path).as_posix()
        value = sha256_file(file)
        digest.update(f"{relative}\0{value}\n".encode())
        total += file.stat().st_size
        count += 1
    return {"tree_sha256": digest.hexdigest(), "file_count": count, "total_file_size": total}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def manifest_payload_sha256(value: dict) -> str:
    payload = dict(value)
    payload.pop("manifest_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build() -> None:
    summary = json.loads(SUMMARY.read_text())
    assert summary["status"] == summary["exact_resume"]["status"] == "pass"
    selected = summary["formal_repeats"]
    sources = [
        ("configs/stage2b_long_run_half_pass_throughput_v2.development.yaml", "resolved_integration_config"),
        ("src/forcesmolvla/rft/throughput_v2_long_run.py", "bounded_cache_runtime"),
        ("src/forcesmolvla/rft/throughput_v2.py", "candidate_b_prefix_reuse_and_fast_polyak"),
        ("src/forcesmolvla/rft/canonical_state.py", "canonical_exact_state"),
        ("src/forcesmolvla/rft/g7_long_run.py", "cycle_checkpoint_contract"),
        ("src/forcesmolvla/rft/training_cycle.py", "training_cycle_primitives"),
        ("src/forcesmolvla/rft/frozen_vlm_trainability.py", "frozen_vlm_contract_runtime"),
        ("src/forcesmolvla/rft/critic_action_adapter_v2.py", "action_contract_v2_adapter"),
        ("src/forcesmolvla/rft/losses.py", "g4_loss_runtime"),
        ("src/forcesmolvla/rft/critic.py", "g2_twin_q_runtime"),
        ("tools/simulate_stage2b_throughput_v2_cache_210.py", "cache_210_simulator"),
        ("tools/run_stage2b_long_run_half_pass_worker_throughput_v2.py", "candidate_b_formal_worker"),
        ("tools/run_stage2b_long_run_half_pass_throughput_v2.py", "fresh_process_coordinator"),
        ("tools/build_stage2b_throughput_v2_long_run_artifacts.py", "artifact_builder"),
        ("tools/benchmark_stage2_batch_scaling_gpu.py", "critic_and_actor_update_runtime"),
        ("src/forcesmolvla/rft/training_cycle_runtime.py", "training_runtime"),
        ("src/forcesmolvla/rft/critic_training.py", "critic_training_runtime"),
        ("tools/run_s2_g7b_worker.py", "parent_rng_sampler_restore"),
        ("tools/run_stage2b_long_run_half_pass_worker.py", "frozen_eta3_actor_update_reference"),
        ("tests/test_stage2b_throughput_v2_long_run.py", "bounded_cache_regression_tests"),
        ("tests/test_rft_throughput_v2.py", "candidate_b_semantic_regression_tests"),
    ]
    inputs = [
        ("configs/stage2_action_contract.v2.development.json", "action_contract_v2"),
        ("configs/stage2_trainability_contract.v2.development.json", "frozen_vlm_trainability_contract"),
        ("configs/stage2_g5_single_cycle.v2.development.yaml", "optimizer_loss_rng_base"),
        ("configs/stage2b_long_run_half_pass.development.yaml", "historical_long_run_recipe_reference"),
        ("artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint/checkpoint_manifest.json", "fresh_parent_manifest"),
        ("artifacts/development/stage2/g1_frozen_detector_transition_view.v1/g1_manifest.json", "automatic_detector_g1_binding"),
        ("datasets/task2_lerobotv3/normalizer_manifest.json", "frozen_normalizer_binding"),
        ("artifacts/development/stage2/stage2_source_manifest.v28_throughput_v2.json", "candidate_screening_source_closure"),
    ]
    results = [
        ("artifacts/development/stage2/throughput_v2_long_run/cache_210_cycle_preflight.json", "bounded_cache_210_cycle_preflight"),
        ("artifacts/development/stage2/throughput_v2_long_run/exact_resume/canonical_parity_report.json", "exact_resume_parity"),
        ("artifacts/development/stage2/throughput_v2_long_run/formal_repeats/batch_selection.json", "formal_repeat_selection"),
        ("artifacts/development/stage2/throughput_v2_long_run/integration_summary.json", "integration_summary"),
    ]
    for critic in (64, 96, 128):
        results.append((
            f"artifacts/development/stage2/throughput_v2_long_run/formal_repeats/aggregate_actor24_critic{critic}.json",
            f"critic_{critic}_aggregate",
        ))
        for repeat in (1, 2, 3):
            results.append((
                f"artifacts/development/stage2/throughput_v2_long_run/formal_repeats/candidate_b_actor24_critic{critic}_repeat{repeat}.json",
                f"critic_{critic}_fresh_repeat_{repeat}",
            ))
    recovery = RUN / "exact_resume/branch_b_cycle1_recovery"
    manifest = {
        "schema_version": "forcesmolvla_stage2_source_manifest.v29.throughput_v2_long_run",
        "scope": "candidate_B_bounded_cache_exact_resume_and_critic_batch_retest",
        "artifact_status": "development_only",
        "sources": [file_record(path, role) for path, role in sources],
        "input_bindings": [file_record(path, role) for path, role in inputs],
        "result_artifacts": [file_record(path, role) for path, role in results],
        "exact_resume_checkpoint": {
            "relative_path": recovery.relative_to(ROOT).as_posix(),
            "artifact_role": "exact_resume_preflight_only_not_training_parent",
            **checkpoint_tree(recovery),
        },
        "runtime_semantics": {
            "long_run_started": False,
            "training_checkpoint_created": False,
            "exact_resume_recovery_checkpoint_only": True,
            "flow_subbatch": 4,
            "grouped_flow_used": False,
            "tf32_used": False,
            "torch_compile_used": False,
            "deterministic_algorithms": True,
        },
    }
    manifest["manifest_payload_sha256"] = manifest_payload_sha256(manifest)
    atomic_json(MANIFEST, manifest)
    manifest_sha = sha256_file(MANIFEST)
    chosen = next(
        item for item in selected["aggregates"]
        if item["critic_batch"] == selected["selected_critic_batch"]
    )
    candidate_b = selected["candidate_b_formal_repeats"]
    cache = summary["bounded_cache_preflight"]
    baseline_seconds = 129.25592586607672
    artifact = {
        "schema_version": "forcesmolvla_stage2b_throughput_v2_long_run_integration.v1",
        "claim_scope": "development_only_no_long_run",
        "THROUGHPUT_V2_BENCHMARK": "pass",
        "CANDIDATE_B_LONG_RUN_INTEGRATION": "pass",
        "BOUNDED_CACHE_210_CYCLE_PREFLIGHT": "pass",
        "EXACT_RESUME": "pass",
        "candidate_B_prefix_cache": {
            "meaning": "formal_integrated_semantically_valid_candidate_B_path",
            "global_optimum_claimed": False,
            "critic128_cycle_seconds": candidate_b["cycle_seconds"],
            "speedup_vs_original_baseline_mean": baseline_seconds / candidate_b["cycle_seconds"]["mean"],
        },
        "rejected_optimizations_preserved": [
            "flow_subbatch_8", "flow_subbatch_16", "grouped_flow"
        ],
        "bounded_cache": {
            "cycles_simulated": cache["cycles_simulated"],
            "draw_plan_sha256": cache["draw_plan_sha256"],
            "total_unique_row_references": cache["total_unique_row_references"],
            "total_unique_images": cache["total_unique_images"],
            "working_set_decoded_bytes": cache["working_set_decoded_bytes"],
            "decoded_cache_max_bytes": cache["cache"]["decoded_cache_max_bytes"],
            "peak_cache_bytes": cache["peak_cache_bytes"],
            "peak_process_rss_bytes": cache["peak_process_rss_bytes"],
            "cache_hit_rate": cache["cache_hit_rate"],
            "eviction_count": cache["eviction_count"],
            "cold_start_seconds": cache["cold_start_seconds"],
            "steady_state_data_latency_seconds_per_cycle": cache["steady_state_data_latency_seconds_per_cycle"],
        },
        "exact_resume": summary["exact_resume"],
        "FINAL_ACTOR_BATCH": selected["selected_actor_batch"],
        "FINAL_CRITIC_BATCH": selected["selected_critic_batch"],
        "FINAL_FLOW_INFERENCE_SUBBATCH": selected["selected_flow_inference_subbatch"],
        "FINAL_CYCLES_PER_HOUR": selected["selected_cycles_per_hour"],
        "selected_configuration": chosen,
        "critic_batch_comparison": selected["aggregates"],
        "speedup_vs_original_baseline_mean": baseline_seconds / chosen["cycle_seconds"]["mean"],
        "PROJECTED_0_5_ACTOR_PASS_RUNTIME": selected["projected_runtime_hours"]["0.5_actor_pass_210_cycles"],
        "PROJECTED_1_0_ACTOR_PASS_RUNTIME": selected["projected_runtime_hours"]["1.0_actor_pass_420_cycles"],
        "PROJECTED_2_0_ACTOR_PASS_RUNTIME": selected["projected_runtime_hours"]["2.0_actor_pass_840_cycles"],
        "exposure_per_0_5_actor_pass": {
            "actor_rows": 210 * 24,
            "td_rows": 210 * 2 * selected["selected_critic_batch"],
            "calql_rows": 210 * 2 * selected["selected_critic_batch"],
            "total_critic_row_memberships": 210 * 4 * selected["selected_critic_batch"],
        },
        "source_manifest": {
            "path": MANIFEST.relative_to(ROOT).as_posix(),
            "sha256": manifest_sha,
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        },
        "OLD_CYCLE105_CHECKPOINT_ALLOWED_AS_PARENT": "no",
        "LONG_RUN_RECIPE_PROPOSED": "yes",
        "LONG_RUN_AUTHORIZED": "no",
        "LONG_RUN_STARTED": "no",
        "ROBOT_EXECUTION_AUTHORIZED": False,
    }
    atomic_json(ARTIFACT, artifact)
    artifact_sha = sha256_file(ARTIFACT)
    report = f"""# Stage-2B throughput-v2 long-run integration report

## Outcome

Candidate B has been integrated into an append-only bounded-cache worker and passed the 210-cycle data-only cache stress test, fresh-process exact-resume test, and three-repeat C64/C96/C128 benchmark. No 210-cycle long-run was started.

`candidate_B_prefix_cache` means the fastest previously screened implementation that satisfied the frozen numerical/training contracts. It does **not** mean formal long-run had already been integrated, nor that C128 was globally optimal. The formal retest selected Actor B24 / Critic C64 for minimum fixed-Actor-pass wall time.

## Frozen implementation boundary

- Parquet files are materialized at most once per process.
- Dual-camera images use an 8 GiB bounded decoded LRU with eight CPU decode workers.
- Cal-QL M=2 candidates share only frozen VLM PrefixContext/KV; each keeps independent fixed Flow noise and the full N=10 integration.
- No trainable Force/Action representation is cached.
- Hot-loop full-model SHA, development Polyak tensor audit, `gc.collect()` and `torch.cuda.empty_cache()` are removed; full state is audited only at preflight/boundary scope.
- Flow inference subbatch remains 4. Rejected B8/B16/grouped-Flow candidates remain rejected; no tolerance was relaxed.

## 210-cycle bounded-cache stress

- Unique row references: {cache['total_unique_row_references']:,}
- Unique images: {cache['total_unique_images']:,}
- Decoded working set: {cache['working_set_decoded_bytes'] / 1024**3:.2f} GiB
- Cache limit / peak: {cache['cache']['decoded_cache_max_bytes'] / 1024**3:.2f} / {cache['peak_cache_bytes'] / 1024**3:.2f} GiB
- Peak process RSS: {cache['peak_process_rss_bytes'] / 1024**3:.2f} GiB
- Hit rate: {cache['cache_hit_rate']:.2%}; evictions: {cache['eviction_count']:,}
- Cold start: {cache['cold_start_seconds']:.2f} s; steady data-only latency: {cache['steady_state_data_latency_seconds_per_cycle']:.3f} s/cycle

The decoded cache is strictly bounded. Peak RSS remains a deployment-planning consideration because materialized compressed Parquet payloads are resident; it did not grow beyond the measured stable bound during the full 210-cycle draw plan.

## Exact resume

Branch A ran two continuous cycles. Branch B ran one cycle, saved an audit-only recovery checkpoint, strict-loaded it in a new process, and ran cycle 2. All canonical comparisons passed at `rtol=0`, `atol=0`: rows, Flow noise/actions, loss and Q traces, gradients, optimizer/target deltas, sampler/RNG state, and final model state. Cycle-2 digest: `{summary['exact_resume']['cycle2_digest']}`.

## Critic batch retest

| Critic batch | Mean cycle (s) | Median | P95 | cycles/hour | Actor transitions/s | TD transitions/s | Cal-QL transitions/s | Peak reserved GiB | Peak RSS GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for item in selected["aggregates"]:
        report += (
            f"| {item['critic_batch']} | {item['cycle_seconds']['mean']:.3f} | "
            f"{item['cycle_seconds']['median']:.3f} | {item['cycle_seconds']['p95']:.3f} | "
            f"{item['joint_cycles_per_hour_from_mean_cycle']:.3f} | "
            f"{item['actor_transitions_per_second']['mean']:.4f} | "
            f"{item['critic_td_transitions_per_second']['mean']:.4f} | "
            f"{item['critic_calql_transitions_per_second']['mean']:.4f} | "
            f"{item['peak_reserved_bytes'] / 1024**3:.2f} | "
            f"{item['peak_cpu_rss_bytes'] / 1024**3:.2f} |\n"
        )
    report += f"""

C64 is selected because it minimizes fixed Actor-pass wall time while all numerical, ownership, mask, frozen-hash, VRAM and ActionContract checks pass. C128 exposes twice as many TD rows and twice as many independent Cal-QL rows per cycle; that exposure tradeoff must be considered before authorizing a long run. The report does not combine TD and Cal-QL memberships into an ambiguous `2 × critic_batch` count.

## Projected budgets (mean steady-state throughput)

- 0.5 Actor pass, 210 cycles: {artifact['PROJECTED_0_5_ACTOR_PASS_RUNTIME']:.3f} h
- 1.0 Actor pass, 420 cycles: {artifact['PROJECTED_1_0_ACTOR_PASS_RUNTIME']:.3f} h
- 2.0 Actor passes, 840 cycles: {artifact['PROJECTED_2_0_ACTOR_PASS_RUNTIME']:.3f} h

For C64, 0.5 pass is 5,040 Actor rows, 26,880 TD memberships, and 26,880 independent Cal-QL memberships. These are projections, not authorization.

## Final state

```text
THROUGHPUT_V2_BENCHMARK = pass
CANDIDATE_B_LONG_RUN_INTEGRATION = pass
BOUNDED_CACHE_210_CYCLE_PREFLIGHT = pass
EXACT_RESUME = pass
FINAL_ACTOR_BATCH = 24
FINAL_CRITIC_BATCH = 64
FINAL_FLOW_INFERENCE_SUBBATCH = 4
FINAL_CYCLES_PER_HOUR = {selected['selected_cycles_per_hour']:.6f}
PROJECTED_0_5_ACTOR_PASS_RUNTIME = {artifact['PROJECTED_0_5_ACTOR_PASS_RUNTIME']:.3f}_hours
PROJECTED_1_0_ACTOR_PASS_RUNTIME = {artifact['PROJECTED_1_0_ACTOR_PASS_RUNTIME']:.3f}_hours
PROJECTED_2_0_ACTOR_PASS_RUNTIME = {artifact['PROJECTED_2_0_ACTOR_PASS_RUNTIME']:.3f}_hours
OLD_CYCLE105_CHECKPOINT_ALLOWED_AS_PARENT = no
LONG_RUN_RECIPE_PROPOSED = yes
LONG_RUN_AUTHORIZED = no
LONG_RUN_STARTED = no
ROBOT_EXECUTION_AUTHORIZED = false
```

Source manifest SHA-256: `{manifest_sha}`  
Final artifact SHA-256: `{artifact_sha}`
"""
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build()
