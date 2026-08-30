#!/usr/bin/env python3
"""Export, validate, and development-publish the Stage-3 joint Actor candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import validate_forcerft_candidate as validator  # noqa: E402


JOINT_CHECKPOINT = validator.JOINT_CHECKPOINT
DESTINATION = validator.PACKAGED_CHECKPOINT
PROFILE = ROOT / "configs/deployment.stage3_joint_cycle000010_candidate.development.json"
BINDING = (
    ROOT
    / "artifacts/development/live"
    / "task2_stage3_joint_cycle000010_candidate_deployment_binding.v1.json"
)
PARENT_BINDING = validator.PARENT_BINDING
CYCLE210_PROFILE = ROOT / "configs/deployment.stage3_cycle210_shadow.development.json"
CYCLE210_BINDING = validator.EXECUTION_BINDING
RULESPEC = validator.RULESPEC


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_new_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite deployment artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _deployment_artifacts(
    *,
    checkpoint: Path,
    model_revision: str,
    profile_path: Path,
    binding_path: Path,
    deployment_id: str,
    approval_id: str,
) -> tuple[Path, Path]:
    from forcesmolvla.checkpoint import sha256_file
    from export_stage2b_cycle210_evaluation_smoke import client_source_sha256
    from serve_policy import (
        load_deployment_binding,
        load_deployment_profile,
        source_tree_sha256,
    )

    base_binding = json.loads(CYCLE210_BINDING.read_text(encoding="utf-8"))
    rulespec_sha = sha256_file(RULESPEC)
    server_sha = source_tree_sha256(ROOT)
    binding = {
        "schema_version": "forcesmolvla-live-deployment-binding-v1",
        "artifact_status": "approved",
        "model_sha256": model_revision,
        "rulespec_sha256": rulespec_sha,
        "server_source_sha256": server_sha,
        "client_source_sha256": client_source_sha256(),
        "state_pose_max_age_ms": base_binding["state_pose_max_age_ms"],
        "camera_max_age_ms": base_binding["camera_max_age_ms"],
        "max_intercamera_skew_ms": base_binding["max_intercamera_skew_ms"],
        "gripper_max_age_ms": base_binding["gripper_max_age_ms"],
        "controller_ack_timeout_ms": base_binding["controller_ack_timeout_ms"],
        "approval": {
            "status": "approved",
            "approval_id": approval_id,
            "approver_identity": "rlc123",
            "approver_role": "experiment_lead",
            "approved_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        },
    }
    _write_new_json(binding_path, binding)
    binding_sha = sha256_file(binding_path)
    loaded, actual_sha = load_deployment_binding(
        binding_path,
        binding_sha,
        model_sha256=model_revision,
        rulespec_sha256=rulespec_sha,
        server_source_sha256=server_sha,
    )
    require(loaded == binding and actual_sha == binding_sha, "STAGE3_CANDIDATE_BINDING_RELOAD")

    base_profile = json.loads(CYCLE210_PROFILE.read_text(encoding="utf-8"))
    profile = {
        "schema_version": "forcesmolvla-deployment-profile-v1",
        "artifact_status": "development_only",
        "deployment_id": deployment_id,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "rulespec": str(RULESPEC.relative_to(ROOT)),
        "deployment_binding": str(binding_path.relative_to(ROOT)),
        "deployment_binding_sha256": binding_sha,
        "dataset_manifest": base_profile["dataset_manifest"],
        "raw_session": base_profile["raw_session"],
        "tool_profile": base_profile["tool_profile"],
    }
    _write_new_json(profile_path, profile)
    loaded_profile = load_deployment_profile(profile_path, ROOT)
    require(
        loaded_profile["checkpoint"].resolve() == checkpoint.resolve()
        and loaded_profile["deployment_binding"].resolve() == binding_path.resolve(),
        "STAGE3_CANDIDATE_PROFILE_RELOAD",
    )
    return profile_path, binding_path


def run(
    *,
    joint_checkpoint: Path,
    destination: Path,
    profile_path: Path,
    binding_path: Path,
    candidate_revision_id: str = validator.EXPECTED_REVISION,
    deployment_id: str = "task2-stage3-joint-cycle000010-candidate",
    approval_id: str = "forcesmolvla-stage3-joint-cycle000010-candidate-20260830-001",
) -> dict[str, Any]:
    from forcesmolvla.checkpoint import export_development_actor_checkpoint

    joint_checkpoint = joint_checkpoint.resolve()
    destination = destination.resolve()
    require(not destination.exists(), "STAGE3_CANDIDATE_EXPORT_DESTINATION_EXISTS")
    require(not profile_path.exists(), "STAGE3_CANDIDATE_PROFILE_EXISTS")
    require(not binding_path.exists(), "STAGE3_CANDIDATE_BINDING_EXISTS")
    require(torch.cuda.is_available(), "STAGE3_CANDIDATE_EXPORT_CUDA_UNAVAILABLE")
    device = torch.device("cuda:0")
    parent_binding = json.loads(PARENT_BINDING.read_text(encoding="utf-8"))
    parent_path = Path(
        parent_binding["actor_parent"]["architecture_binding"]["container_path"]
    )
    actor = validator._load_direct_candidate(joint_checkpoint, parent_path, device)
    staging = destination.with_name(f".{destination.name}.validation-{os.getpid()}")
    try:
        manifest = export_development_actor_checkpoint(
            policy=actor,
            destination=staging,
            runtime_parent=parent_path,
            source_joint_checkpoint=joint_checkpoint,
            candidate_revision_id=candidate_revision_id,
            parent_binding_id=parent_binding["binding_id"],
            published=True,
        )
        del actor
        torch.cuda.empty_cache()
        validation = validator.run(
            joint_checkpoint,
            staging,
            expected_revision=candidate_revision_id,
        )
        require(
            validation["CANDIDATE_OFFLINE_VALIDATION"] == "PASS",
            f"STAGE3_CANDIDATE_OFFLINE_VALIDATION:{validation['HARD_ERRORS']}",
        )
        os.replace(staging, destination)
        model_revision = manifest["payloads"]["model.safetensors"]["sha256"]
        profile_path, binding_path = _deployment_artifacts(
            checkpoint=destination,
            model_revision=model_revision,
            profile_path=profile_path,
            binding_path=binding_path,
            deployment_id=deployment_id,
            approval_id=approval_id,
        )
        validation.update(
            {
                "CANDIDATE_EXPORT": "PASS",
                "CANDIDATE_PUBLISHED": True,
                "CANDIDATE_ACTIVATED": False,
                "DEPLOYMENT_PROFILE_PATH": str(profile_path.resolve()),
                "DEPLOYMENT_BINDING_PATH": str(binding_path.resolve()),
                "MODEL_UPDATE_COUNT": 0,
            }
        )
        return validation
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, default=JOINT_CHECKPOINT)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--deployment-profile", type=Path, default=PROFILE)
    parser.add_argument("--deployment-binding", type=Path, default=BINDING)
    parser.add_argument("--candidate-revision-id", default=validator.EXPECTED_REVISION)
    parser.add_argument(
        "--deployment-id", default="task2-stage3-joint-cycle000010-candidate"
    )
    parser.add_argument(
        "--approval-id",
        default="forcesmolvla-stage3-joint-cycle000010-candidate-20260830-001",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        joint_checkpoint=args.joint_checkpoint,
        destination=args.destination,
        profile_path=args.deployment_profile,
        binding_path=args.deployment_binding,
        candidate_revision_id=args.candidate_revision_id,
        deployment_id=args.deployment_id,
        approval_id=args.approval_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
