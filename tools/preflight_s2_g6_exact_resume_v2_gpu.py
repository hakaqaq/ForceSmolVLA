#!/usr/bin/env python3
"""Coordinate append-only G6-v2 exact-resume branches without creating CUDA here."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from forcesmolvla.rft.exact_resume_v2 import G5_MANIFEST_SHA256, install_exact_resume_v2

import preflight_s2_g5_single_cycle_gpu as g5
import preflight_s2_g6_exact_resume_gpu as legacy


ROOT = Path(__file__).parents[1].resolve()
install_exact_resume_v2()

legacy.CONFIG = ROOT / "configs/stage2_g6_exact_resume.v2.development.yaml"
legacy.SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v14_g6_v2.json"
legacy.G5_CHECKPOINT = ROOT / "artifacts/development/stage2/g5_single_cycle_checkpoint.v2.development"
legacy.G6_ROOT = ROOT / "artifacts/development/stage2/g6_exact_resume.v2"
legacy.ARTIFACT = ROOT / "artifacts/development/stage2/s2_g6_exact_resume_preflight.v2.json"
legacy.REPORT = ROOT / "docs/s2_g6_exact_resume_preflight_report.v2.md"
legacy.WORKER = ROOT / "tools/run_s2_g6_branch_worker_v2.py"
g5.CONFIG = ROOT / "configs/stage2_g5_single_cycle.v2.development.yaml"
g5.SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v13_g5_v2.json"
g5.CHECKPOINT = legacy.G5_CHECKPOINT


def _protected_snapshot_v2() -> dict:
    from forcesmolvla.rft.exact_resume import checkpoint_tree

    base = g5.protected_snapshot()
    paths = {
        "action_contract_v2": ROOT / "configs/stage2_action_contract.v2.development.json",
        "action_adapter_v2": ROOT / "src/forcesmolvla/rft/critic_action_adapter_v2.py",
        "g5_v2_config": g5.CONFIG,
        "g5_v2_source_manifest": g5.SOURCE_MANIFEST,
        "g5_v2_artifact": ROOT / "artifacts/development/stage2/s2_g5_single_cycle_preflight.v2.json",
        "g5_v2_checkpoint_manifest": legacy.G5_CHECKPOINT / "checkpoint_manifest.json",
        "g6_v2_config": legacy.CONFIG,
        "g6_v2_source_manifest": legacy.SOURCE_MANIFEST,
    }
    return {
        "g5_protected": base,
        "files": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": legacy.sha256_file(path),
                "file_size": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "g5_checkpoint": checkpoint_tree(legacy.G5_CHECKPOINT),
    }


legacy.protected_snapshot = _protected_snapshot_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    legacy.require(args.run, "pass --run for authorized G6-v2 exact-resume")
    legacy.require(not torch.cuda.is_initialized(), "G6_V2_COORDINATOR_MUST_NOT_INITIALIZE_CUDA")
    legacy.require(legacy.CONFIG.is_file() and legacy.SOURCE_MANIFEST.is_file(), "G6_V2_CONFIG_OR_SOURCE_MISSING")
    for target in (legacy.G6_ROOT, legacy.ARTIFACT, legacy.REPORT):
        legacy.require(not target.exists(), f"G6_V2_APPEND_ONLY_TARGET_EXISTS:{target}")

    from forcesmolvla.rft.canonical_state import assert_payload_exact
    from forcesmolvla.rft.exact_resume import (
        G6_CHECKPOINT_MARKERS,
        checkpoint_training_payload,
        checkpoint_tree,
        preflight_g5_checkpoint,
        validate_checkpoint_files,
    )

    preflight = preflight_g5_checkpoint(ROOT, legacy.G5_CHECKPOINT)
    before = legacy.protected_snapshot()
    negatives = legacy.negative_tests()
    legacy.require(
        legacy.sha256_file(legacy.G5_CHECKPOINT / "checkpoint_manifest.json")
        == G5_MANIFEST_SHA256,
        "G6_V2_NEGATIVE_TEST_MUTATED_G5",
    )
    work = Path(tempfile.mkdtemp(prefix=".g6_v2_work.", dir=legacy.G6_ROOT.parent))
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONHASHSEED": "42",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HOME": "/tmp/forcesmolvla_g6_v2_hf_cache",
            "HF_DATASETS_CACHE": "/tmp/forcesmolvla_g6_v2_hf_cache/datasets",
            "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'vendor/lerobot/src'}:{ROOT / 'tools'}",
        }
    )
    try:
        branch_a, parameter_map = legacy.launch_branch_a(work, environment)
        legacy.require(branch_a["cycle1_g5_exact"] and not branch_a["loaded_g5_training_state"], "G6_V2_BRANCH_A_S1_FAILED")
        branch_b = legacy.launch_branch_b(work, environment)
        legacy.require(branch_b["restored_s1_exact"] and branch_b["rng_restored_last"], "G6_V2_BRANCH_B_RESTORE_FAILED")
        legacy.require(branch_a["environment"]["pid"] != branch_b["environment"]["pid"], "G6_V2_PID_REUSED")
        a2 = checkpoint_training_payload(work / "branch_a_cycle2_checkpoint", parameter_map, g5=False)
        b2 = checkpoint_training_payload(work / "branch_b_cycle2_checkpoint", parameter_map, g5=False)
        assert_payload_exact(a2, b2, "g6_v2_branch_A_vs_B_cycle2")
        legacy.require(branch_a["cycle2_trace"] == branch_b["cycle2_trace"], "G6_V2_TRACE_NOT_EXACT")
        expected = {
            "training_cycles": 2,
            "critic_optimizer_updates": 4,
            "actor_optimizer_updates": 2,
            "q1_target_polyak_updates": 4,
            "q2_target_polyak_updates": 4,
            "actor_target_updates": 0,
            "critic_scheduler_steps": 4,
            "actor_scheduler_steps": 2,
        }
        legacy.require(branch_a["final_counters"] == branch_b["final_counters"] == expected, "G6_V2_COUNTERS")
        for name in ("branch_a_cycle1_checkpoint", "branch_a_cycle2_checkpoint", "branch_b_cycle2_checkpoint"):
            validate_checkpoint_files(work / name, expected_markers=G6_CHECKPOINT_MARKERS)
        pid_a, pid_b = branch_a["environment"]["pid"], branch_b["environment"]["pid"]
        legacy.atomic_json(work / "branch_a_cycle1_manifest.json", {
            "branch_pid": pid_a, "cycle1_g5_exact": True,
            "state_digest": branch_a["cycle1_state_digest"],
            "checkpoint_save_side_effect": branch_a["cycle1_checkpoint_save_side_effect"],
            "checkpoint": checkpoint_tree(work / "branch_a_cycle1_checkpoint"),
        })
        legacy.atomic_json(work / "branch_a_cycle2_manifest.json", {
            "branch_pid": pid_a, "continuous_in_memory_from_cycle1": True,
            "state_digest": a2["training_state_digest"],
            "trace_digest": branch_a["cycle2_trace"]["canonical_trace_digest"],
            "checkpoint": checkpoint_tree(work / "branch_a_cycle2_checkpoint"),
        })
        legacy.atomic_json(work / "branch_b_cycle2_manifest.json", {
            "branch_pid": pid_b, "fresh_process_strict_resume": True,
            "state_digest": b2["training_state_digest"],
            "trace_digest": branch_b["cycle2_trace"]["canonical_trace_digest"],
            "checkpoint": checkpoint_tree(work / "branch_b_cycle2_checkpoint"),
        })
        parity = {
            "schema_version": "forcesmolvla_g6_canonical_parity.v2",
            "action_contract": "v2",
            "s1_branch_a_equals_g5": True,
            "s2_branch_a_equals_branch_b": True,
            "cycle2_trace_exact": True,
            "cycle2_trace_digest": branch_a["cycle2_trace"]["canonical_trace_digest"],
            "cycle2_training_state_digest": a2["training_state_digest"],
            "comparison": {"rtol": 0.0, "atol": 0.0, "equal_nan": False},
        }
        legacy.atomic_json(work / "canonical_parity_report.json", parity)
        os.replace(work, legacy.G6_ROOT)
    except BaseException:
        failure = legacy.G6_ROOT.parent / f"g6_v2_failed_{os.getpid()}"
        if work.exists():
            os.replace(work, failure)
        raise

    after = legacy.protected_snapshot()
    legacy.require(before == after, "G6_V2_FROZEN_INPUT_MUTATION")
    legacy.require(not torch.cuda.is_initialized(), "G6_V2_COORDINATOR_CREATED_CUDA")
    branch_a = json.loads((legacy.G6_ROOT / "branch_a_result.json").read_text())
    branch_b = json.loads((legacy.G6_ROOT / "branch_b_result.json").read_text())
    parity = json.loads((legacy.G6_ROOT / "canonical_parity_report.json").read_text())
    acceptance = {
        "g5_v2_checkpoint_and_bindings_valid": True,
        "distinct_fresh_processes": branch_a["environment"]["pid"] != branch_b["environment"]["pid"],
        "branch_a_no_g5_state_load": not branch_a["loaded_g5_training_state"],
        "branch_a_s1_exact": branch_a["cycle1_g5_exact"],
        "checkpoint_save_side_effect_free": branch_a["cycle1_checkpoint_save_side_effect"]["exact_unchanged"],
        "branch_b_strict_rng_last": branch_b["loaded_g5_training_state_strict"] and branch_b["rng_restored_last"],
        "branch_b_no_predraw": branch_b["random_sanity_forward_after_rng_restore"] == branch_b["sampler_draws_before_cycle2"] == 0,
        "cycle2_trace_exact": parity["cycle2_trace_exact"],
        "cycle2_state_exact": parity["s2_branch_a_equals_branch_b"],
        "final_counters_exact": branch_a["final_counters"] == branch_b["final_counters"],
        "frozen_inputs_unchanged": before == after,
        "heldout_manual_reads_zero": all(
            item["data_access_audit"]["validation_transition_reads"]
            == item["data_access_audit"]["test_transition_reads"]
            == item["data_access_audit"]["manual_g1_files_opened"]
            == item["data_access_audit"]["manual_label_files_opened"]
            == 0
            for item in (branch_a, branch_b)
        ),
        "five_negative_tests_fail_closed": len(negatives) == 5 and all(item["rejected"] for item in negatives),
        "cycle3_and_g7b_not_run": branch_a["final_counters"]["training_cycles"] == 2,
    }
    legacy.require(all(acceptance.values()), f"G6_V2_ACCEPTANCE_FAILED:{acceptance}")
    artifact = {
        "schema_version": "forcesmolvla_s2_g6_exact_resume_preflight.v2",
        "G6_FRESH_PROCESS_EXACT_RESUME": "pass",
        "G5_CHECKPOINT_RESUME_EXACTNESS": "development_verified",
        "action_contract": "v2",
        "coordinator": {"pid": os.getpid(), "cuda_initialized": False},
        "parent_preflight": preflight,
        "protected_before": before,
        "protected_after": after,
        "branches": {"A": branch_a, "B": branch_b},
        "parity": parity,
        "negative_tests": negatives,
        "acceptance": acceptance,
        "data_access_audit": {
            "validation_transition_reads": 0,
            "test_transition_reads": 0,
            "manual_g1_files_opened": 0,
            "manual_label_files_opened": 0,
            "reward_classifier_inference_calls": 0,
            "reward_classifier_optimizer_updates": 0,
        },
        "checkpoint_markers": G6_CHECKPOINT_MARKERS,
        "terminal_status": {
            "G6_V2": "pass",
            "G7A_R2_STARTED": "no",
            "G7B_STARTED": "no",
        },
    }
    report = legacy.report_markdown(artifact) + "\n\nThis v2 replay binds total-binary internal gripper canonicalization; public execution behavior remains unchanged.\n"
    legacy.atomic_text(legacy.REPORT, report)
    artifact["report_sha256"] = legacy.sha256_file(legacy.REPORT)
    artifact["g6_output_tree"] = checkpoint_tree(legacy.G6_ROOT)
    artifact["artifact_payload_sha256"] = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    legacy.atomic_json(legacy.ARTIFACT, artifact)
    print(json.dumps({
        "status": "pass",
        "branch_a_pid": branch_a["environment"]["pid"],
        "branch_b_pid": branch_b["environment"]["pid"],
        "cycle2_trace_digest": parity["cycle2_trace_digest"],
        "cycle2_training_state_digest": parity["cycle2_training_state_digest"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
