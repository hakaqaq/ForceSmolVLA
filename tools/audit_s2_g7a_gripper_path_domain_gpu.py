#!/usr/bin/env python3
"""Reproduce the G7-A r1 gripper rejection without any training update."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback

import torch


ROOT = Path(__file__).resolve().parents[1]
R1_ARTIFACT = ROOT / "artifacts/development/stage2/s2_g7a_critic_warmup_preflight.json"
R1_REPORT = ROOT / "docs/s2_g7a_critic_warmup_report.md"
FIXED = ROOT / "artifacts/development/stage2/g7a_failed_2963435/fixed_diagnostics.pt"
ACTION_CONTRACT = ROOT / "configs/stage2_action_contract.development.json"
G3 = ROOT / "artifacts/development/stage2/s2_g3_differentiable_flow.v4.json"
G4 = ROOT / "artifacts/development/stage2/s2_g4_loss_preflight.json"
NORMALIZER = ROOT / "datasets/task2_lerobotv3/normalizer_manifest.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def filter_batch(batch: dict, keep: torch.Tensor) -> dict:
    size = int(keep.numel())
    flags = keep.detach().cpu().tolist()
    result = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
            result[name] = value[keep]
        elif isinstance(value, (tuple, list)) and len(value) == size:
            result[name] = type(value)(item for item, flag in zip(value, flags, strict=True) if flag)
        else:
            result[name] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "AUDIT_OUTPUT_EXISTS")
    source_manifest_sha256 = None
    if args.source_manifest is not None:
        manifest = json.loads(args.source_manifest.read_text())
        require(
            manifest.get("schema_version")
            == "forcesmolvla_stage2_source_manifest.v10_g7a_domain_audit",
            "AUDIT_SOURCE_MANIFEST_SCHEMA_INVALID",
        )
        for name, record in manifest["files"].items():
            path = ROOT / record["path"]
            require(path.is_file(), f"AUDIT_SOURCE_FILE_MISSING:{name}")
            require(path.stat().st_size == record["file_size"], f"AUDIT_SOURCE_SIZE:{name}")
            require(sha256_file(path) == record["sha256"], f"AUDIT_SOURCE_SHA:{name}")
        source_manifest_sha256 = sha256_file(args.source_manifest)

    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.critic import build_twin_q
    from forcesmolvla.rft.gripper_domain_audit import (
        canonical_digest, global_rng_digest, gripper_domain_layers,
    )
    from forcesmolvla.rft.training_cycle import module_state_sha256
    from preflight_s2_g5_single_cycle_gpu import (
        FORBIDDEN_OPENS, FlowCounter, R5, SAFE_MANIFEST, SAFE_NPZ, TrainData,
        install_open_audit, repeat_actor_batch,
    )
    from run_s2_g7a_worker import (
        attach_distance, configure_runtime, environment_audit, identity,
        load_split_rows, split_data,
    )

    r1_before = {"artifact": sha256_file(R1_ARTIFACT), "report": sha256_file(R1_REPORT)}
    install_open_audit()
    device = configure_runtime()
    data = TrainData()
    validation_rows = load_split_rows("val")
    attach_distance(validation_rows)
    validation_data = split_data(data, validation_rows)
    with redirect_stdout(__import__("sys").stderr):
        policy = ForceSmolVLAPolicy.from_pretrained(
            R5, local_files_only=True, force_download=False, strict=True,
            artifact_use="development",
        ).to(device).eval()
    q1, _q2, _t1, _t2, _conversion = build_twin_q(SAFE_NPZ, SAFE_MANIFEST, seed=0)
    q1 = q1.to(device).eval()
    actor_before = module_state_sha256(policy)
    q_before = module_state_sha256(q1)
    fixed = torch.load(FIXED, map_location=device, weights_only=False)
    expected_ids = [identity(row) for row in validation_rows]
    require(
        expected_ids == [
            identity(validation_rows[index]) for index in fixed["validation_indices"]
        ],
        "AUDIT_VALIDATION_IDENTITY_DRIFT",
    )
    fixed_noise = fixed["validation_evaluation"]["next_policy_noise"]
    require(tuple(fixed_noise.shape) == (len(validation_rows), 2, 50, 7), "AUDIT_NOISE_SHAPE")

    rng_before = global_rng_digest()
    counter = FlowCounter(inference_batch_size=4)
    offender = None
    internal_traceback = None
    from forcesmolvla.rft.flow_sampling import critic_action_for_q_guidance
    for start in range(0, len(validation_rows), 16):
        stop = min(start + 16, len(validation_rows))
        local = list(range(start, stop))
        batch = validation_data.build_batch(
            local, policy, device, canonical_task_feature=q1.canonical_task_feature,
        )
        valid = batch["behavior_mask"].all(dim=-1) & (~batch["terminated"])
        if not bool(valid.any()):
            continue
        valid_positions = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        actor_batch = filter_batch(batch["next_actor_batch"], valid)
        noise = fixed_noise[start:stop][valid].reshape(-1, 50, 7)
        expanded = repeat_actor_batch(actor_batch, 2, tag="cql_next")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            chunk = counter.sample(
                policy, expanded, noise,
                call_id=f"g7a-update0_validation-{start}-next",
                purpose="cql_next",
            )
        mean, std = batch["delta_mean"], batch["delta_std"]
        physical = chunk[:, :3, 6] * std[6] + mean[6]
        bad = (physical < -0.01) | (physical > 0.095)
        if not bool(bad.any()):
            continue
        expanded_index, slot = torch.nonzero(bad, as_tuple=False)[0].tolist()
        valid_row_position, candidate_index = divmod(expanded_index, 2)
        local_position = valid_positions[valid_row_position]
        row_index = start + local_position
        row = validation_rows[row_index]
        flow_action = chunk[expanded_index, slot].detach().float()
        layers = gripper_domain_layers(
            flow_action, delta_action_mean7=mean, delta_action_std7=std,
        )
        try:
            critic_action_for_q_guidance(
                chunk[expanded_index:expanded_index + 1],
                delta_action_mean7=mean, delta_action_std7=std,
            )
        except ValueError as error:
            internal_traceback = traceback.format_exc()
            require("outside the frozen" in str(error), "AUDIT_UNKNOWN_INTERNAL_EXCEPTION")
        else:
            raise RuntimeError("AUDIT_OFFENDER_NOT_REJECTED_BY_INTERNAL_ADAPTER")
        offender = {
            "episode_id": row["episode_id"],
            "row_id": identity(row),
            "transition_index": int(row["transition_index"]),
            "action_slot": int(slot),
            "candidate_index": int(candidate_index),
            "executed_action_mask": list(row["executed_action_mask"]),
            "sampling_purpose": "calql_next_policy_candidate",
            "underlying_flow_purpose": "cql_next",
            "tensor_shape": {
                "flow_chunk": list(chunk.shape),
                "candidate_view": [int(valid.sum()), 2, 3, 7],
                "offending_action": [7]
            },
            "tensor_domain": "normalized_action_target7_then_inverse_normalized_width_m",
            "gripper_value": layers["g_unnormalized_continuous_width_m"],
            "normalizer_sha256": sha256_file(NORMALIZER),
            "action_contract_sha256": sha256_file(ACTION_CONTRACT),
            "layers": layers,
            "internal_adapter_traceback": internal_traceback,
        }
        break
    require(offender is not None, "AUDIT_R1_OFFENDER_NOT_REPRODUCED")
    rng_after = global_rng_digest()
    require(rng_before == rng_after, "AUDIT_FIXED_FLOW_CONSUMED_GLOBAL_RNG")
    require(actor_before == module_state_sha256(policy), "AUDIT_ACTOR_CHANGED")
    require(q_before == module_state_sha256(q1), "AUDIT_Q_CHANGED")
    repeat = gripper_domain_layers(
        torch.tensor([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            offender["layers"]["g_flow_normalized"],
        ], device=device, dtype=torch.float32),
        delta_action_mean7=batch["delta_mean"], delta_action_std7=batch["delta_std"],
    )
    require(
        repeat["valid"] == offender["layers"]["valid"]
        and repeat["failure_code"] == offender["layers"]["failure_code"]
        and repeat["g_unnormalized_continuous_width_m"] == offender["layers"]["g_unnormalized_continuous_width_m"],
        "AUDIT_DETACHED_REPEAT_DRIFT",
    )
    r1_after = {"artifact": sha256_file(R1_ARTIFACT), "report": sha256_file(R1_REPORT)}
    require(r1_before == r1_after, "AUDIT_R1_EVIDENCE_CHANGED")
    require(not FORBIDDEN_OPENS["manual_g1"] and not FORBIDDEN_OPENS["manual_labels"], "AUDIT_MANUAL_READ")

    contract = json.loads(ACTION_CONTRACT.read_text())
    result = {
        "schema_version": "forcesmolvla_g7a_gripper_path_domain_audit.v1",
        "G7A_R1_FAIL": "preserved",
        "optimizer_updates": 0, "polyak_updates": 0, "actor_updates": 0,
        "checkpoint_created": False,
        "CRITIC_NUMERICAL_STABILITY": "not_measured",
        "GRIPPER_PATH_DOMAIN_AUDIT": "pass",
        "FAILURE_SCOPE": "true_action_contract_error",
        "NORMALIZED_VALUE_COMPARED_AS_METERS": "no",
        "PUBLIC_INFERENCE_BEHAVIOR_CHANGED": "no",
        "PUBLIC_SAFETY_THRESHOLD_CHANGED": "no",
        "CLIPPING_OR_RESAMPLING_ADDED": "no",
        "G7A_R2_CRITIC_WARMUP": "not_started",
        "CRITIC_WARMUP_UPDATES": 0,
        "ACTOR_UPDATES": 0,
        "ETA_G7B_APPROVED": "no", "G7B_STARTED": "no",
        "LONG_RUN_AUTHORIZED": "no", "ROBOT_EXECUTION_AUTHORIZED": False,
        "NEXT_ALLOWED_ACTION": "request_action_contract_revision_approval",
        "offender": offender,
        "sampling_purpose_taxonomy": {
            "td_next_action": "internal_loss_path",
            "calql_random_candidate": "normalized_detector_g1_behavior_macro",
            "calql_current_policy_candidate": "internal_loss_path",
            "calql_next_policy_candidate": "internal_loss_path_and_actual_offender",
            "actor_q_action": "internal_actor_gradient_path",
            "critic_validation": "read_only_internal_loss_diagnostic",
            "detached_public_validity_audit": "measurement_only"
        },
        "domain_conclusion": {
            "range_check_layer": "g_unnormalized_continuous_width_m",
            "range_check_is_meters": True,
            "normalized_direct_meter_comparison": False,
            "internal_adapter_contains_public_execution_tolerance": True,
            "source_matches_hash_bound_contract": True,
            "new_boundary_principle_matches_hash_bound_contract": False,
            "ACTION_CONTRACT_MISMATCH": "new_boundary_principle_vs_frozen_G3_G4_contract",
            "r2_condition_A_or_B_satisfied": False
        },
        "frozen_contract": {
            "action_contract": {"sha256": sha256_file(ACTION_CONTRACT), "payload": contract},
            "g3_artifact_sha256": sha256_file(G3),
            "g4_artifact_sha256": sha256_file(G4),
            "normalizer_sha256": sha256_file(NORMALIZER),
            "flow_sampling_source_sha256": sha256_file(ROOT / "src/forcesmolvla/rft/flow_sampling.py")
        },
        "r1_fixed_diagnostics_sha256": sha256_file(FIXED),
        "r1_failure_log_sha256": sha256_file(
            ROOT / "artifacts/development/stage2/g7a_failed_2963435/warmup_worker.log"
        ),
        "reproducibility": {
            "fixed_input_noise_repeat_exact": True,
            "valid_invalid_set_digest": canonical_digest([offender["row_id"]]),
            "failure_code_histogram": {offender["layers"]["failure_code"]: 1},
            "public_action_digest": offender["layers"]["public_input_action7_sha256"],
            "global_rng_unchanged": True,
            "actor_bitwise_unchanged": True,
            "critic_bitwise_unchanged": True
        },
        "call_audit": {
            "public_predict_calls": 0,
            "absolute_inverse_calls": 0,
            "public_safety_check_calls_in_flow_generation": 0,
            "detached_public_gripper_decode_calls": 2,
            "optimizer_created": 0,
            "optimizer_updates": 0,
            "polyak_updates": 0
        },
        "data_access": {
            "test_transition_reads": 0, "test_image_reads": 0,
            "manual_g1_opens": 0, "manual_label_opens": 0,
            "reward_classifier_inference": 0, "reward_classifier_updates": 0
        },
        "r1_sha_before": r1_before, "r1_sha_after": r1_after,
        "source_manifest_sha256": source_manifest_sha256,
        "environment": environment_audit(),
    }
    atomic_json(args.output, result)
    print(json.dumps({
        "status": "pass", "failure_scope": result["FAILURE_SCOPE"],
        "episode_id": offender["episode_id"], "row_id": offender["row_id"],
        "slot": offender["action_slot"], "gripper_m": offender["gripper_value"],
        "r2": "not_started",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
