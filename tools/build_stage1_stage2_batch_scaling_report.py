#!/usr/bin/env python3
"""Build append-only Trainability-v2 and batch-scaling evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).parents[1].resolve()
STAGE1 = ROOT / "artifacts/development/stage2/batch_scaling/stage1/stage1_summary.json"
STAGE2 = ROOT / "artifacts/development/stage2/batch_scaling/stage2/stage2_summary.json"
PREFLIGHT = ROOT / "artifacts/development/stage2/stage2_frozen_vlm_trainability_preflight.json"
GRADIENT = ROOT / "artifacts/development/stage2/batch_scaling/stage2/frozen_vlm_gradient_scale.json"
OUTPUT = ROOT / "artifacts/development/stage2/batch_scaling_report.json"
REPORT = ROOT / "docs/stage1_stage2_batch_scaling_report.md"
MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v20_trainability_batch_scaling.json"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dist(values) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    require(array.size and np.isfinite(array).all(), "BATCH_REPORT_STAT_INVALID")
    return {
        "count": int(array.size), "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)), "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()), "maximum": float(array.max()),
        "range": float(array.max() - array.min()),
    }


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def raw_results(paths: list[str]) -> list[dict]:
    return [load(ROOT / value) for value in paths]


def cycle_breakdown(results: list[dict]) -> dict:
    cycles = [cycle for result in results for cycle in result["measured_records"]]
    critic_first = [cycle["critic"][0] for cycle in cycles]
    critic_second = [cycle["critic"][1] for cycle in cycles]
    actors = [cycle["actor"] for cycle in cycles]

    def critic(items: list[dict]) -> dict:
        names = (
            "data_loading", "td_next_action_sampling",
            "calql_current_policy_sampling", "calql_next_policy_sampling",
            "calql_empirical_proposal_and_overhead",
            "q_forward_backward_excluding_optimizer_polyak", "optimizer", "polyak", "scheduler",
        )
        return {name: dist(item["timing"][name] for item in items) for name in names}

    actor_names = (
        "data_loading", "flow_matching_forward_backward",
        "differentiable_n10_flow_twin_q_actor_q_backward", "actor_optimizer",
    )
    prefix = [cycle.get("frozen_prefix_prefill_seconds_embedded") for cycle in cycles]
    prefix = [value for value in prefix if value is not None]
    return {
        "steady_state_training_time_seconds": dist(cycle["cycle_seconds"] for cycle in cycles),
        "data_loading_total_seconds": dist(
            cycle["actor"]["timing"]["data_loading"]
            + sum(item["timing"]["data_loading"] for item in cycle["critic"])
            for cycle in cycles
        ),
        "frozen_observation_encoding_prefix_prefill_seconds_embedded": (
            dist(prefix) if prefix else {
                "status": "embedded_in_flow_components_not_separately_timed_in_selected_runs",
                "call_count_is_recorded_per_flow_counter": True,
            }
        ),
        "critic_update_1": critic(critic_first),
        "critic_update_2": critic(critic_second),
        "actor_update": {
            name: dist(item["timing"][name] for item in actors) for name in actor_names
        },
        "polyak_update_seconds": dist(
            sum(item["timing"]["polyak"] for item in cycle["critic"])
            for cycle in cycles
        ),
        "development_diagnostics_seconds": {"steady_state": 0.0, "excluded": True},
        "public_inference_audit_seconds": dist(
            result["public_inference"]["before"]["latency_seconds"]
            + result["public_inference"]["after"]["latency_seconds"]
            for result in results
        ),
        "checkpoint_and_report_seconds": 0.0,
        "checkpoint_created": False,
        "note": "prefix timing is embedded in FM/Flow sampling and must not be added twice",
    }


def hours(seconds: float) -> float:
    return seconds / 3600.0


def main() -> None:
    require(not OUTPUT.exists() and not REPORT.exists() and not MANIFEST.exists(), "BATCH_REPORT_OUTPUT_EXISTS")
    stage1, stage2, preflight, gradient = map(load, (STAGE1, STAGE2, PREFLIGHT, GRADIENT))
    require(stage1["status"] == stage2["status"] == preflight["status"] == gradient["status"] == "pass", "BATCH_REPORT_PARENT_NOT_PASS")
    selected = stage2["recommended_joint_result"]
    selected_results = raw_results(selected["result_paths"])
    baseline_paths = [
        f"artifacts/development/stage2/batch_scaling/stage2/candidate_results/baseline_frozen_joint_a4_c16_repeat{repeat}.json"
        for repeat in (1, 2, 3)
    ]
    baseline_results = raw_results(baseline_paths)
    require(all(item["status"] == "pass" for item in baseline_results), "BATCH_REPORT_BASELINE_NOT_PASS")
    baseline_cycle = dist(item["seconds_per_cycle"]["median"] for item in baseline_results)
    baseline_reserved = max(item["peak_reserved_bytes"] for item in baseline_results)
    historical_g7b = load(ROOT / "artifacts/development/stage2/s2_g7b_joint_smoke_preflight.json")
    historical_cycle = 41.58
    historical_artifact_cycle = historical_g7b["train"]["runtime"]["total_seconds"] / 8.0
    historical_reserved = historical_g7b["train"]["runtime"]["peak_reserved_bytes"]
    total_memory = int(stage2["total_gpu_memory_bytes"])
    selected_cycles_per_hour = float(stage2["recommended_joint_cycles_per_hour"])
    conservative = next(
        item for item in stage2["candidate_aggregates"]["joint"]
        if item["actor_physical_batch_size"] == 24 and item["critic_physical_batch_size"] == 64
    )
    conservative_cycle = float(conservative["seconds_per_cycle"]["median"])
    pass_table = []
    for batch in (4, 8, 16, 24, 32, 48, 64):
        pass_table.append({
            "effective_actor_batch": batch,
            "half_pass_updates": math.ceil(0.5 * 10075 / batch),
            "one_pass_updates": math.ceil(10075 / batch),
            "two_pass_updates": math.ceil(2.0 * 10075 / batch),
        })
    selected_budgets = []
    for label, actor_passes, updates in (("0.5", 0.5, 210), ("1.0", 1.0, 420), ("2.0", 2.0, 840)):
        actor_samples = updates * 24
        critic_samples = updates * 2 * 128
        projected_hours = updates / selected_cycles_per_hour
        selected_budgets.append({
            "actor_transition_passes_target": actor_passes,
            "joint_cycles_actor_updates": updates,
            "critic_optimizer_updates": updates * 2,
            "actor_sample_exposure": actor_samples,
            "actor_passes_actual": actor_samples / 10075,
            "critic_sample_exposure": critic_samples,
            "critic_transition_passes": critic_samples / 10075,
            "projection_basis": "reported_average_steady_state_cycles_per_hour",
            "projection_cycles_per_hour": selected_cycles_per_hour,
            "projected_seconds": projected_hours * 3600.0,
            "projected_hours": projected_hours,
            "conservative_b24_c64_projected_hours": hours(updates * conservative_cycle),
        })
    raw_ratio = gradient["gradient_scale"]["global"]["raw_q_over_fm"]
    proposed_eta = 3.0
    artifact = {
        "schema_version": "forcesmolvla_stage1_stage2_batch_scaling.v2",
        "status": "pass",
        "scope": "development_only_temporary_updates_discarded",
        "trainability_contract": {
            "status": "frozen_vlm_force_action_trainable",
            "config": "configs/stage2_trainability_contract.v2.development.json",
            "preflight": PREFLIGHT.relative_to(ROOT).as_posix(),
            "FROZEN_VLM_FORWARD_PARITY": "pass",
            "FROZEN_PARAMETER_HASH_UNCHANGED": "yes",
            "parameter_counts": preflight["parameter_counts"],
            "historical_g7b": preflight["historical_status"],
        },
        "stage1": {
            "summary_path": STAGE1.relative_to(ROOT).as_posix(),
            "recommended_physical_batch": stage1["recommended_physical_batch"],
            "samples_per_second": stage1["recommended_samples_per_second"],
            "projected_40000_runtime_seconds": stage1["projected_40000_sample_runtime_seconds"],
            "projected_40000_runtime_hours": hours(stage1["projected_40000_sample_runtime_seconds"]),
            "candidates": stage1["candidate_aggregates"],
            "b24_b32_not_run_reason": "B16 throughput gain over B8 was below 5 percent and reserved memory was near the safety boundary",
            "training_performed": False,
            "sample_budget_note": "equal sample exposure does not imply an equivalent optimizer trajectory; LR, warmup, scheduler, and batch-local MoE losses need separate approval",
        },
        "stage2": {
            "summary_path": STAGE2.relative_to(ROOT).as_posix(),
            "recommended_offline": {
                "actor_physical_batch": 24,
                "critic_physical_batch": 128,
                "gradient_accumulation": 1,
                "actor_transitions_per_second_joint": stage2["recommended_actor_transitions_per_second"],
                "critic_transitions_per_second_joint": stage2["recommended_critic_transitions_per_second"],
                "joint_cycles_per_hour": stage2["recommended_joint_cycles_per_hour"],
                "peak_reserved_bytes": selected["peak_reserved_bytes"],
                "peak_reserved_gib": selected["peak_reserved_bytes"] / 2**30,
                "reserved_fraction": selected["peak_reserved_bytes"] / total_memory,
                "free_headroom_gib": (total_memory - selected["peak_reserved_bytes"]) / 2**30,
            },
            "recommended_same_gpu_online_coexistence_candidate": {
                "status": "requires_separate_concurrent_stress_test",
                "actor_physical_batch": 24,
                "critic_physical_batch": 64,
                "actor_transitions_per_second_joint": conservative["actor_transitions_per_second"]["median"],
                "critic_transitions_per_second_joint": conservative["critic_transitions_per_second"]["median"],
                "joint_cycles_per_hour": conservative["joint_cycles_per_hour"]["median"],
                "peak_reserved_gib": conservative["peak_reserved_bytes"] / 2**30,
                "free_headroom_gib": (total_memory - conservative["peak_reserved_bytes"]) / 2**30,
                "online_execution_authorized": False,
            },
            "candidate_aggregates": stage2["candidate_aggregates"],
            "oom_rejections": {"actor_batch_32": "3/3 OOM", "critic_batch_256": "3/3 OOM"},
            "critic_policy_flow_inference_subbatch": 4,
            "critic_policy_flow_inference_subbatch_note": "held fixed across critic candidates; not the Critic physical Q batch",
            "same_config_frozen_vlm_baseline_b4_c16": {
                "cycle_seconds": baseline_cycle,
                "peak_reserved_gib": baseline_reserved / 2**30,
                "speedup_vs_user_baseline_41p58": historical_cycle / baseline_cycle["median"],
                "speedup_vs_g7b_artifact_mean": historical_artifact_cycle / baseline_cycle["median"],
                "memory_reduction_vs_g7b": 1.0 - baseline_reserved / historical_reserved,
            },
            "cycle_decomposition_same_config_b4_c16": cycle_breakdown(baseline_results),
            "cycle_decomposition_selected_b24_c128": cycle_breakdown(selected_results),
            "actor_pass_table": pass_table,
            "projected_budgets": selected_budgets,
            "gradient_scale_measurement": gradient,
        },
        "long_run_recipe_proposal": {
            "status": "proposal_only_not_authorized",
            "parent": "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint",
            "old_g7b_checkpoint_parent_allowed": False,
            "actor_physical_batch_size": 24,
            "critic_physical_batch_size": 128,
            "gradient_accumulation": 1,
            "critic_to_actor_update_ratio": 2,
            "critic_warmup_transition_passes": 0.5,
            "critic_warmup_samples": 5038,
            "critic_warmup_updates_at_b128": 40,
            "existing_g7a_r2_warmup_exposure": {"samples": 4096, "passes": 4096 / 10075},
            "existing_parent_topup_requires_explicit_decision": True,
            "recommended_starting_actor_joint_training_passes": 0.5,
            "allowed_budget_options": [0.5, 1.0, 2.0],
            "actor_learning_rate": 1e-5,
            "force_module_learning_rate": 1e-5,
            "action_expert_learning_rate": 1e-5,
            "action_io_learning_rate": 1e-5,
            "critic_learning_rate": 3e-4,
            "linear_lr_scaling_used": False,
            "beta": 1.0,
            "eta_measured": 10.0,
            "eta_measured_weighted_ratio": gradient["gradient_scale"]["global"]["weighted_eta10_q_over_beta1_fm"],
            "eta_proposed_for_next_approval": proposed_eta,
            "eta3_analytical_weighted_ratio": proposed_eta * raw_ratio,
            "eta_note": "eta=10 produced a 0.19085 weighted norm ratio with mildly opposing global cosine; eta=3 is an analytical proposal only and was not run",
            "calql_alpha": 0.1,
            "polyak_tau": 0.005,
            "polyak_tau_status": "retain_as_proposal_pending_sample_timescale_review; not automatically batch-scaled",
            "actor_lr_warmup_samples": 0,
            "actor_scheduler": "constant",
            "scheduler_total_samples_by_budget": {item["actor_transition_passes_target"]: item["actor_sample_exposure"] for item in selected_budgets},
            "validation_interval_in_actor_pass": 0.25,
            "validation_interval_actor_updates": 105,
            "recovery_checkpoint_interval_actor_pass": 0.25,
            "recovery_checkpoint_interval_cycles": 105,
            "validation_is_read_only": True,
            "test_or_manual_data_for_selection": False,
        },
        "conrft_relationship": {
            "batch_or_20k_steps_directly_transferable": False,
            "steps_per_update_50_interpretation": "asynchronous learner update/publication cadence, not 50 ForceRFT updates per new batch",
            "force_rft_stage1": "full-model force-conditioned behavior adaptation",
            "force_rft_stage2": "frozen-backbone value-guided force-action refinement",
            "stage2_twin_q_updates_vlm": False,
        },
        "access_audit": {
            "validation_reads": 0, "test_reads": 0,
            "manual_g1_opens": 0, "manual_label_opens": 0,
            "reward_classifier_inference": 0, "reward_classifier_updates": 0,
            "candidate_checkpoint_count": 0,
        },
        "protected_bindings": {
            "r5_artifact_manifest_sha256": sha(ROOT / "outputs/development/task2_lerobotv3_full_sft_10k_r5/checkpoints/step_010000/artifact_manifest.json"),
            "g1_manifest_sha256": sha(ROOT / "artifacts/development/stage2/g1_frozen_detector_transition_view.v1/g1_manifest.json"),
            "g7a_r2_checkpoint_manifest_sha256": sha(ROOT / "artifacts/development/stage2/g7a_r2_critic_warmup_checkpoint/checkpoint_manifest.json"),
            "g7b_artifact_sha256": sha(ROOT / "artifacts/development/stage2/s2_g7b_joint_smoke_preflight.json"),
            "g7b_source_manifest_sha256": sha(ROOT / "artifacts/development/stage2/stage2_source_manifest.v17_g7b.json"),
            "action_contract_v2_config_sha256": sha(ROOT / "configs/stage2_action_contract.v2.development.json"),
            "action_contract_v2_source_sha256": sha(ROOT / "src/forcesmolvla/rft/critic_action_adapter_v2.py"),
            "reward_classifier_checkpoint_sha256": "6b4e366baa55993d150cb3dd86e67a1d708e58d836b123a0c433190835021510",
            "protected_paths_written": 0,
        },
        "STAGE2_TRAINABILITY_CONTRACT": "frozen_vlm_force_action_trainable",
        "FROZEN_VLM_FORWARD_PARITY": "pass",
        "FROZEN_PARAMETER_HASH_UNCHANGED": "yes",
        "STAGE1_RECOMMENDED_PHYSICAL_BATCH": 8,
        "STAGE1_SAMPLES_PER_SECOND": stage1["recommended_samples_per_second"],
        "STAGE1_PROJECTED_40K_RUNTIME": stage1["projected_40000_sample_runtime_seconds"],
        "STAGE2_RECOMMENDED_ACTOR_PHYSICAL_BATCH": 24,
        "STAGE2_RECOMMENDED_CRITIC_PHYSICAL_BATCH": 128,
        "STAGE2_ACTOR_TRANSITIONS_PER_SECOND": stage2["recommended_actor_transitions_per_second"],
        "STAGE2_CRITIC_TRANSITIONS_PER_SECOND": stage2["recommended_critic_transitions_per_second"],
        "STAGE2_JOINT_CYCLES_PER_HOUR": stage2["recommended_joint_cycles_per_hour"],
        "STAGE2_SPEEDUP_VS_FULL_ACTOR_G7B": historical_cycle / baseline_cycle["median"],
        "PROJECTED_0_5_ACTOR_PASS_RUNTIME": selected_budgets[0]["projected_seconds"],
        "PROJECTED_1_ACTOR_PASS_RUNTIME": selected_budgets[1]["projected_seconds"],
        "PROJECTED_2_ACTOR_PASS_RUNTIME": selected_budgets[2]["projected_seconds"],
        "PROJECTED_ACTOR_PASS_RUNTIME_BASIS": "reported_average_steady_state_cycles_per_hour",
        "PROJECTED_ACTOR_PASS_RUNTIME_UNIT": "seconds",
        "LONG_RUN_RECIPE_PROPOSED": "yes",
        "LONG_RUN_AUTHORIZED": "no",
        "LONG_RUN_STARTED": "no",
        "ROBOT_EXECUTION_AUTHORIZED": False,
    }
    atomic_json(OUTPUT, artifact)

    report = f"""# Stage-1 / Stage-2 Trainability and Batch Scaling\n\n## Outcome\n\nFrozen-VLM TrainabilityContract v2 and its GPU preflight **passed**. The recommended offline throughput configuration is Actor B24 / Critic B128. Actor B32 and Critic B256 each OOMed in all three independent processes and are rejected. No benchmark state was checkpointed or retained.\n\nThe same-GPU online-coexistence candidate is B24/B64, pending a separate concurrent stress test. It leaves {artifact['stage2']['recommended_same_gpu_online_coexistence_candidate']['free_headroom_gib']:.2f} GiB versus {artifact['stage2']['recommended_offline']['free_headroom_gib']:.2f} GiB for B24/B128. Neither authorizes online or robot execution.\n\n## TrainabilityContract v2\n\n- Frozen: Vision Encoder, SmolVLM/token embeddings, state-to-prefix projection; always eval and excluded from the Actor optimizer.\n- Trainable: ForceMLP, Fusion/MoE/router, Force Action Adapter, Action Expert and Action I/O.\n- Frozen/trainable Actor parameters: {preflight['parameter_counts']['frozen_parameter_count']:,} / {preflight['parameter_counts']['trainable_actor_parameter_count']:,}.\n- Exact frozen/full forward parity before updates: pass; frozen parameter/buffer hashes after temporary updates: unchanged.\n- Prefix representation/cache detached; Force K/V prepared once per chunk.\n- ActionContract-v2 and public execution behavior are unchanged.\n\nExisting full-Actor G7-B remains `historical_valid_development_mechanics`; its checkpoint is not a long-run parent.\n\n## Stage-1 scaling\n\n| Batch | Median samples/s | Peak reserved GiB | Decision |\n|---:|---:|---:|---|\n"""
    for item in stage1["candidate_aggregates"]:
        decision = "recommended" if item["physical_batch_size"] == 8 else "measured"
        report += f"| {item['physical_batch_size']} | {item['samples_per_second']['median']:.3f} | {item['peak_reserved_bytes']/2**30:.2f} | {decision} |\n"
    report += f"""\nB16 improved median throughput only {(stage1['candidate_aggregates'][2]['samples_per_second']['median']/stage1['candidate_aggregates'][1]['samples_per_second']['median']-1)*100:.2f}% over B8, below the frozen 5% rule, while reaching {stage1['candidate_aggregates'][2]['peak_reserved_bytes']/2**30:.2f} GiB. Therefore B24/B32 were not run. At B8, 40,000 exposures are projected to take {hours(stage1['projected_40000_sample_runtime_seconds']):.2f} h. Equal sample exposure does not make the optimization trajectory equivalent; LR, warmup, scheduler and batch-local MoE losses require separate approval.\n\n## Stage-2 scaling\n\n### Actor-only\n\n| Batch | Median transitions/s | Peak reserved GiB | Result |\n|---:|---:|---:|---|\n"""
    for item in stage2["candidate_aggregates"]["actor"]:
        speed = item.get("actor_transitions_per_second")
        report += f"| {item['actor_physical_batch_size']} | {speed['median']:.3f} | {item['peak_reserved_bytes']/2**30:.2f} | {'PASS' if item['all_pass'] else 'OOM 3/3'} |\n" if speed else f"| {item['actor_physical_batch_size']} | — | — | OOM 3/3 |\n"
    report += "\n### Critic-only\n\n| Batch | Median transitions/s | Peak reserved GiB | Result |\n|---:|---:|---:|---|\n"
    for item in stage2["candidate_aggregates"]["critic"]:
        speed = item.get("critic_transitions_per_second")
        report += f"| {item['critic_physical_batch_size']} | {speed['median']:.3f} | {item['peak_reserved_bytes']/2**30:.2f} | {'PASS' if item['all_pass'] else 'OOM 3/3'} |\n" if speed else f"| {item['critic_physical_batch_size']} | — | — | OOM 3/3 |\n"
    report += "\n### Joint combinations\n\n| Actor/Critic | Cycle s | Actor tr/s | Critic tr/s | Reserved GiB |\n|---|---:|---:|---:|---:|\n"
    for item in stage2["candidate_aggregates"]["joint"]:
        report += f"| {item['actor_physical_batch_size']}/{item['critic_physical_batch_size']} | {item['seconds_per_cycle']['median']:.2f} | {item['actor_transitions_per_second']['median']:.3f} | {item['critic_transitions_per_second']['median']:.3f} | {item['peak_reserved_bytes']/2**30:.2f} |\n"
    report += f"""\nAt the same historical B4/C16 layout, Frozen-VLM uses {baseline_cycle['median']:.2f} s/cycle versus the supplied 41.58 s/cycle full-Actor baseline: {historical_cycle/baseline_cycle['median']:.2f}x speedup and {(1-baseline_reserved/historical_reserved)*100:.1f}% lower reserved memory. Steady-state timing excludes public audit, checkpoint, process load and report generation. Prefix timing is embedded in the Flow components and is separately identified in the JSON artifact to prevent double counting.\n\n## Frozen-VLM gradient scale\n\nAt eta=10, beta=1 on the selected B24 physical Actor batch:\n\n- raw `||g_Q|| / ||g_FM||`: {raw_ratio:.6f}\n- weighted ratio: {gradient['gradient_scale']['global']['weighted_eta10_q_over_beta1_fm']:.6f}\n- cosine similarity: {gradient['gradient_scale']['global']['cosine_similarity']:.6f}\n- TCP6 Q gradient: {gradient['tcp6_q_gradient_norm']:.8g}\n- gripper Q gradient: exact {gradient['gripper_q_gradient_max_abs']:.1f}\n- gripper FM gradient: {gradient['gripper_fm_gradient_norm']:.8g}\n- one discarded step normalized TCP drift: {gradient['fixed_action_diagnostic']['normalized_tcp_action_drift_mean_l2']:.6f}\n- binary gripper change rate: {gradient['fixed_action_diagnostic']['binary_gripper_change_rate']:.3f}\n- raw-gripper out-of-public-tolerance rate before/after: {gradient['fixed_action_diagnostic']['raw_gripper_out_of_public_tolerance_rate_before']:.3f}/{gradient['fixed_action_diagnostic']['raw_gripper_out_of_public_tolerance_rate_after']:.3f}\n\nBecause eta=10 yields a 0.19085 weighted ratio and mildly opposing global cosine, the proposed next approval value is eta=3 (analytical expected ratio {proposed_eta*raw_ratio:.5f}); eta=3 was not run or approved.\n\n## Actor-transition budgets (B24/C128)\n\nBudget projections use the reported average steady-state throughput of {selected_cycles_per_hour:.4f} cycles/hour (about {3600.0/selected_cycles_per_hour:.2f} seconds/cycle), while all original mean, median and range measurements remain unchanged.\n\n| Target Actor passes | Cycles | Actor exposure | Critic exposure | Critic passes | Projected time |\n|---:|---:|---:|---:|---:|---:|\n"""
    for item in selected_budgets:
        report += f"| {item['actor_transition_passes_target']:.1f} | {item['joint_cycles_actor_updates']} | {item['actor_sample_exposure']} | {item['critic_sample_exposure']} | {item['critic_transition_passes']:.2f} | {item['projected_hours']:.2f} h |\n"
    report += """\nThe proposed starting budget is 0.5 Actor pass, followed by review; 1 and 2 passes are projections, not convergence claims. G7-A's 256 B16 Critic updates equal 4,096 samples (0.407 transition pass); whether to accept that parent as-is or top it up toward 0.5 pass needs explicit approval.\n\n## ConRFT boundary\n\nConRFT batch sizes, 20k pretraining steps, and online `steps_per_update=50` are not directly transferable. ConRFT updates a lighter consistency policy/critic while freezing larger VLA representations. ForceRFT performs full-model force-conditioned behavior adaptation in Stage-1, then frozen-backbone value-guided force-action refinement with native N=10 Flow in Stage-2. `steps_per_update=50` is an asynchronous learner publication cadence, not a requirement to train ForceRFT 50 times per new batch. Stage-2 Twin-Q does not update VLM parameters.\n\n## Stop state\n\n`LONG_RUN_RECIPE_PROPOSED=yes`; `LONG_RUN_AUTHORIZED=no`; `LONG_RUN_STARTED=no`; `ROBOT_EXECUTION_AUTHORIZED=false`. Validation/test/manual G1/manual labels/Reward Classifier reads were all zero.\n"""
    atomic(REPORT, report)

    source_paths = [
        "configs/stage2_trainability_contract.v2.development.json",
        "configs/stage1_batch_scaling.development.yaml",
        "configs/stage2_batch_scaling.development.yaml",
        "src/forcesmolvla/rft/frozen_vlm_trainability.py",
        "src/forcesmolvla/rft/batch_scaling.py",
        "src/forcesmolvla/rft/critic_action_adapter_v2.py",
        "src/forcesmolvla/rft/critic.py",
        "src/forcesmolvla/rft/losses.py",
        "src/forcesmolvla/rft/flow_sampling.py",
        "src/forcesmolvla/rft/training_cycle.py",
        "src/forcesmolvla/modeling_forcesmolvla.py",
        "src/forcesmolvla/router_training.py",
        "tools/preflight_stage2_frozen_vlm_trainability_gpu.py",
        "tools/benchmark_stage1_batch_scaling_gpu.py",
        "tools/benchmark_stage2_batch_scaling_gpu.py",
        "tools/measure_stage2_frozen_vlm_gradient_scale_gpu.py",
        "tools/build_stage1_stage2_batch_scaling_report.py",
        "tools/preflight_s2_g5_single_cycle_gpu.py",
        "tools/run_s2_g7a_worker.py",
        "tools/run_s2_g7a_r2_worker.py",
        "tools/run_s2_g7b_worker.py",
        "tests/test_stage2_frozen_vlm_trainability.py",
        "tests/test_rft_batch_scaling.py",
    ]
    evidence_paths = [
        PREFLIGHT.relative_to(ROOT).as_posix(), STAGE1.relative_to(ROOT).as_posix(),
        STAGE2.relative_to(ROOT).as_posix(), GRADIENT.relative_to(ROOT).as_posix(),
        OUTPUT.relative_to(ROOT).as_posix(), REPORT.relative_to(ROOT).as_posix(),
        *baseline_paths,
    ]
    manifest = {
        "schema_version": "forcesmolvla_stage2_source_manifest.v20_trainability_batch_scaling",
        "scope": "append_only_frozen_vlm_trainability_and_temporary_batch_scaling",
        "files": [
            {"relative_path": value, "sha256": sha(ROOT / value), "file_size": (ROOT / value).stat().st_size}
            for value in source_paths
        ],
        "evidence": [
            {"relative_path": value, "sha256": sha(ROOT / value), "file_size": (ROOT / value).stat().st_size}
            for value in evidence_paths
        ],
        "candidate_result_inventory": {
            "stage1": len(list((ROOT / "artifacts/development/stage2/batch_scaling/stage1/candidate_results").glob("*.json"))),
            "stage2": len(list((ROOT / "artifacts/development/stage2/batch_scaling/stage2/candidate_results").glob("*.json"))),
        },
        "manual_g1_or_manual_label_in_runtime_closure": False,
        "checkpoint_created": False,
        "long_run_started": False,
    }
    atomic_json(MANIFEST, manifest)
    print("BATCH_SCALING_REPORT pass")


if __name__ == "__main__":
    main()
