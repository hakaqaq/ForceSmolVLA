#!/usr/bin/env python3
"""S2-G0 zero-update bridge from the frozen r5 Actor to Stage-2 sidecars."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).parents[1].resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _head_changes(root: Path, base: str, head: str) -> list[dict]:
    changes = []
    output = _git(root, "diff", "--name-status", "--find-renames", base, head)
    for line in output.splitlines():
        fields = line.split("\t")
        if fields[0].startswith("R"):
            if len(fields) != 3:
                raise RuntimeError("S2_G0_GIT_RENAME_RECORD_INVALID")
            changes.append(
                {"status": fields[0], "old_path": fields[1], "new_path": fields[2]}
            )
        else:
            if len(fields) != 2:
                raise RuntimeError("S2_G0_GIT_CHANGE_RECORD_INVALID")
            changes.append({"status": fields[0], "path": fields[1]})
    return changes


def _validate_worktree(root: Path, prefixes: list[str]) -> list[str]:
    lines = _git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    unauthorized = []
    for line in lines:
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if not any(path == prefix or path.startswith(prefix) for prefix in prefixes):
            unauthorized.append(line)
    if unauthorized:
        raise RuntimeError(f"S2_G0_UNAUTHORIZED_WORKTREE_CHANGE:{unauthorized}")
    return lines


def _validate_source_bridge(root: Path, config: dict) -> dict:
    if (
        config.get("schema_version") != "1.0"
        or config.get("gate") != "S2-G0"
        or config.get("acceptance_status") != "development_only"
        or config.get("formal_eligible") is not False
    ):
        raise RuntimeError("S2_G0_CONFIG_STATUS_INVALID")

    source_entry = config["parent_source_binding"]
    source_path = root / source_entry["path"]
    if _sha256(source_path) != source_entry["sha256"]:
        raise RuntimeError("S2_G0_PARENT_SOURCE_BINDING_HASH_MISMATCH")
    source = json.loads(source_path.read_text(encoding="utf-8"))

    resolved_entry = config["parent_resolved_training_config"]
    resolved_path = root / resolved_entry["path"]
    if _sha256(resolved_path) != resolved_entry["sha256"]:
        raise RuntimeError("S2_G0_PARENT_RESOLVED_CONFIG_HASH_MISMATCH")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if (
        resolved.get("training_stage") != "offline_full_finetune"
        or resolved.get("force_variant") != "force_token_moe"
        or resolved.get("training_budget", {}).get("target_samples") != 40_000
        or resolved.get("training_budget", {}).get("derived_optimizer_updates") != 10_000
        or resolved.get("all_parameters_trainable") is not True
    ):
        raise RuntimeError("S2_G0_PARENT_RESOLVED_CONFIG_SEMANTICS_DRIFT")

    git_config = config["current_git"]
    actual_head = _git(root, "rev-parse", "HEAD")
    if actual_head != git_config["head"]:
        raise RuntimeError("S2_G0_CURRENT_GIT_HEAD_MISMATCH")
    if _git(root, "rev-parse", f"{actual_head}^") != git_config["head_parent"]:
        raise RuntimeError("S2_G0_CURRENT_GIT_PARENT_MISMATCH")
    actual_changes = _head_changes(root, git_config["head_parent"], actual_head)
    if actual_changes != config["current_head_change_allowlist"]:
        raise RuntimeError("S2_G0_CURRENT_HEAD_CHANGE_ALLOWLIST_MISMATCH")

    vendor_root = root / "vendor/lerobot"
    if (
        _git(vendor_root, "rev-parse", "HEAD") != source["lerobot_commit"]
        or _git(vendor_root, "status", "--porcelain")
    ):
        raise RuntimeError("S2_G0_LEROBOT_COMMIT_OR_CLEANLINESS_MISMATCH")
    for relative, expected in source["lerobot_file_sha256"].items():
        if _sha256(vendor_root / relative) != expected:
            raise RuntimeError(f"S2_G0_LEROBOT_SOURCE_HASH_MISMATCH:{relative}")

    allowlist = config["parent_snapshot_changed_file_allowlist"]
    observed_changed = []
    for relative, expected in source["project_file_sha256"].items():
        path = root / relative
        actual = _sha256(path) if path.is_file() else None
        if actual == expected:
            continue
        observed_changed.append(relative)
        allowed = allowlist.get(relative)
        if not isinstance(allowed, dict) or allowed.get("parent_sha256") != expected:
            raise RuntimeError(f"S2_G0_UNAPPROVED_PARENT_SOURCE_DRIFT:{relative}")
        current = root / allowed["current_path"]
        if not current.is_file() or _sha256(current) != allowed["current_sha256"]:
            raise RuntimeError(f"S2_G0_ALLOWLIST_CURRENT_HASH_MISMATCH:{relative}")
        if allowed["status"] == "modified":
            if allowed["current_path"] != relative:
                raise RuntimeError(f"S2_G0_ALLOWLIST_MODIFIED_PATH_MISMATCH:{relative}")
        elif allowed["status"] == "renamed_and_refactored":
            if path.exists() or allowed["current_path"] == relative:
                raise RuntimeError(f"S2_G0_ALLOWLIST_RENAME_CONTRACT_INVALID:{relative}")
        else:
            raise RuntimeError(f"S2_G0_ALLOWLIST_STATUS_INVALID:{relative}")
    if sorted(observed_changed) != sorted(allowlist):
        raise RuntimeError("S2_G0_PARENT_SOURCE_ALLOWLIST_SCOPE_MISMATCH")

    dataset_root = Path(source["dataset_root"])
    for name, expected in source["dataset_manifest_sha256"].items():
        if _sha256(dataset_root / name) != expected:
            raise RuntimeError(f"S2_G0_PARENT_DATASET_MANIFEST_DRIFT:{name}")

    qualification = []
    for entry in config["parent_p4_to_p8_qualification_artifacts"]:
        path = root / entry["path"]
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"S2_G0_PARENT_QUALIFICATION_DRIFT:{entry['path']}")
        qualification.append({**entry, "file_size": path.stat().st_size})

    worktree = _validate_worktree(root, config["authorized_worktree_prefixes"])
    return {
        "parent_source_binding_sha256": source_entry["sha256"],
        "parent_resolved_training_config_sha256": resolved_entry["sha256"],
        "parent_training_samples": 40_000,
        "parent_optimizer_updates": 10_000,
        "current_git_head": actual_head,
        "current_head_changes": actual_changes,
        "parent_snapshot_changed_files": observed_changed,
        "parent_p4_to_p8_qualification_artifacts": qualification,
        "worktree_changes": worktree,
    }


def _tensor_record(tensor) -> dict:
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.view(__import__("torch").uint8).numpy()).hexdigest(),
    }


def _training_terms_record(policy, batch, noise7, timestep) -> dict:
    import torch

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        losses, feature_mask, router = policy.forward_single_pass_training_terms(
            batch, noise=noise7, time=timestep
        )
    return {
        "losses": _tensor_record(losses),
        "feature_mask": _tensor_record(feature_mask),
        "router_logits_fp32": _tensor_record(router.logits_fp32),
        "router_probabilities_fp32": _tensor_record(router.probabilities_fp32),
        "router_route_ids": _tensor_record(router.route_ids),
        "router_valid_mask": _tensor_record(router.valid_mask),
        "validation_scalar": float((losses.sum() / feature_mask.sum()).cpu()),
    }


def _public_error_record(policy, batch, context, invalid_noise) -> dict:
    from forcesmolvla.modeling_forcesmolvla import ActionInferenceError

    try:
        policy.predict_action_chunk(batch, chunk_context=context, noise=invalid_noise)
    except ActionInferenceError as error:
        return {"type": type(error).__name__, "code": error.code, "message": str(error)}
    raise RuntimeError("S2_G0_PUBLIC_ERROR_FIXTURE_UNEXPECTEDLY_SUCCEEDED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/stage2_parent_bridge.development.json",
    )
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/task2_lerobotv3")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite S2-G0 artifact: {args.output}")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 required")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE_NO_CPU_FALLBACK")
    gpu_name = torch.cuda.get_device_name(0)
    if "4090 D" not in gpu_name and "4090D" not in gpu_name:
        raise RuntimeError(f"S2_G0_REQUIRES_RTX_4090D:{gpu_name}")

    sys.path.insert(0, str(ROOT / "tools"))
    from forcesmolvla.checkpoint import validate_force_artifact_manifest
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.flow_sampling import sample_normalized_action_chunk_with_grad
    from forcesmolvla.rft.source_manifest import stage2_source_manifest_binding
    from p8_checkpoint_common import chunk_context_from_fixture, load_fixed_validation_inputs
    from preflight_s2_common import module_state_dict_sha256

    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    source_evidence = _validate_source_bridge(ROOT, config)
    source_manifest_path = args.source_manifest or ROOT / config["stage2_source_manifest"]
    stage2_source_manifest = stage2_source_manifest_binding(ROOT, source_manifest_path)
    checkpoint = ROOT / config["parent_checkpoint"]
    manifest_path = checkpoint / "artifact_manifest.json"
    if _sha256(manifest_path) != config["parent_checkpoint_artifact_manifest_sha256"]:
        raise RuntimeError("S2_G0_PARENT_CHECKPOINT_MANIFEST_HASH_MISMATCH")

    with contextlib.redirect_stdout(sys.stderr):
        policy = ForceSmolVLAPolicy.from_pretrained(
            checkpoint,
            local_files_only=True,
            force_download=False,
            strict=True,
            artifact_use="development",
        ).to("cuda:0")
    if not all(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("S2_G0_PARENT_ACTOR_NOT_FULLY_TRAINABLE")
    state_before = module_state_dict_sha256(policy)

    fixture = json.loads(
        (checkpoint / "manifests/fixed_validation_fixture.json").read_text(encoding="utf-8")
    )
    batch, _raw, runtime_artifacts = load_fixed_validation_inputs(
        policy, args.dataset_root.resolve(), fixture, torch.device("cuda:0")
    )
    policy.bind_runtime_artifacts(runtime_artifacts)
    batch["sample_identity"] = tuple(
        f"episode={row['episode_index']}/frame={row['frame_index']}"
        for row in fixture["tuple_list"]
    )
    noise7 = torch.tensor(fixture["epsilon7"]["tensor"], dtype=torch.float32, device="cuda:0")
    timestep = torch.tensor(
        fixture["time"]["tensor"], dtype=torch.float32, device="cuda:0"
    )
    context_before = chunk_context_from_fixture(
        fixture, policy_generation=policy._context_generation
    )
    context_after = replace(
        context_before,
        chunk_id=tuple(f"s2-g0-after-{index}" for index in range(noise7.shape[0])),
    )
    error_context_before = replace(
        context_before,
        chunk_id=tuple(f"s2-g0-error-before-{index}" for index in range(noise7.shape[0])),
    )
    error_context_after = replace(
        context_before,
        chunk_id=tuple(f"s2-g0-error-after-{index}" for index in range(noise7.shape[0])),
    )

    policy.eval()
    training_terms_before = _training_terms_record(policy, batch, noise7, timestep)
    error_before = _public_error_record(
        policy, batch, error_context_before, noise7[..., :6]
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        normalized_before = policy._predict_normalized_delta_chunk(
            batch, chunk_context=context_before, noise=noise7
        )
        absolute_before = policy.predict_action_chunk(
            batch, chunk_context=context_before, noise=noise7
        )
        normalized_after_graph = sample_normalized_action_chunk_with_grad(
            policy,
            batch,
            noise7,
            call_id="parent-zero-update",
            purpose="actor_guidance",
        )
        normalized_after = normalized_after_graph.detach()
        del normalized_after_graph
        absolute_after = policy.predict_action_chunk(
            batch, chunk_context=context_after, noise=noise7
        )
    training_terms_after = _training_terms_record(policy, batch, noise7, timestep)
    error_after = _public_error_record(
        policy, batch, error_context_after, noise7[..., :6]
    )

    normalized_exact = torch.equal(normalized_before, normalized_after)
    absolute_exact = torch.equal(absolute_before, absolute_after)
    state_after = module_state_dict_sha256(policy)
    state_exact = state_before == state_after
    fixed_training_terms_exact = training_terms_before == training_terms_after
    public_error_exact = error_before == error_after
    validate_force_artifact_manifest(checkpoint, artifact_use="development")
    if (
        not normalized_exact
        or not absolute_exact
        or not state_exact
        or not fixed_training_terms_exact
        or not public_error_exact
    ):
        raise RuntimeError("S2_G0_ZERO_UPDATE_PARITY_FAILED")

    trainable_parameters = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in policy.parameters())

    result = {
        "schema_version": "1.0",
        "gate": "S2-G0",
        "gate_status": "pass",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "checkpoint": str(checkpoint),
        "checkpoint_artifact_manifest_sha256": _sha256(manifest_path),
        "source_bridge": source_evidence,
        "stage2_source_manifest": stage2_source_manifest,
        "state_dict": {"before_sha256": state_before, "after_sha256": state_after, "exact": True},
        "fixed_noise_normalized_action": {
            "before": _tensor_record(normalized_before),
            "after": _tensor_record(normalized_after),
            "max_abs_error": float((normalized_before - normalized_after).abs().max().cpu()),
            "exact": True,
        },
        "public_absolute_action": {
            "before": _tensor_record(absolute_before),
            "after": _tensor_record(absolute_after),
            "max_abs_error": float((absolute_before - absolute_after).abs().max().cpu()),
            "exact": True,
        },
        "public_error_code": {
            "before": error_before,
            "after": error_after,
            "exact": public_error_exact,
        },
        "fixed_forward_single_pass_training_terms": {
            "before": training_terms_before,
            "after": training_terms_after,
            "historical_expected_validation_scalar": None,
            "historical_expected_validation_scalar_status": (
                "not_present_in_r5_fixed_validation_fixture"
            ),
            "fixture_sha256": _sha256(
                checkpoint / "manifests/fixed_validation_fixture.json"
            ),
            "exact": fixed_training_terms_exact,
        },
        "strict_r5_reload": True,
        "all_parameters_trainable": True,
        "trainable_parameter_count": trainable_parameters,
        "total_parameter_count": total_parameters,
        "trainability_matches_parent_config": trainable_parameters == total_parameters,
        "optimizer_created": False,
        "optimizer_state_restored": False,
        "optimizer_steps": 0,
        "robot_actions_sent": 0,
        "real_rft_training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
