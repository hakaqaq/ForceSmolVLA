#!/usr/bin/env python3
"""Build the non-recursive integrity closure for uncommitted Stage-2 files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from forcesmolvla.rft.source_manifest import canonical_sha256, sha256_file


ROOT = Path(__file__).parents[1].resolve()
CONRFT_ROOT = Path("/home/rlc123/conrft")
V4_SPEC = "docs/ForceRFT_Stage2_Offline_TwinQ_Implementation_Spec_v4.md"
V4_SPEC_SHA256 = "0d0ad0312e9758ede7b6910b232096dcaeed338d3a7d4b5aa96347d988ecdce4"
HISTORICAL_V3_SPEC = (
    "docs/ForceRFT_Stage2_Offline_TwinQ_Implementation_Spec_v3_v4_2_aligned.md"
)
HISTORICAL_V3_SPEC_SHA256 = (
    "14cb57537e7c30547ca7f220fdc232f4ca61b314ee8d2cd58dbcd75d4a1058d8"
)
V4_SOURCE_ENTRIES = {
    "configs/stage2_action_contract.development.json": ("action_contract", True),
    "configs/stage2_g3_differentiable_flow.development.json": ("gate_config", True),
    "configs/stage2_g3_gradient_matrix.development.json": ("gate_config", True),
    "configs/stage2_parent_bridge.development.json": ("gate_config", True),
    "configs/stage2_reward_spec.development.yaml": ("g1_unapproved_input", False),
    "docs/ForceRFT_Stage2_Offline_TwinQ_Implementation_Spec_v3_v4_2_aligned.md": (
        "historical_v3_specification",
        False,
    ),
    "docs/ForceRFT_Stage2_Offline_TwinQ_Implementation_Spec_v4.md": (
        "active_v4_specification",
        False,
    ),
    "labels/task2_episode_outcomes.v1.json": ("g1_unapproved_input", False),
    "src/forcesmolvla/rft/__init__.py": ("runtime_source", True),
    "src/forcesmolvla/rft/flow_sampling.py": ("runtime_source", True),
    "src/forcesmolvla/rft/offline_transitions.py": ("g1_framework", False),
    "src/forcesmolvla/rft/source_manifest.py": ("integrity_source", True),
    "tests/test_rft_flow_sampling.py": ("test_source", False),
    "tests/test_rft_offline_transitions.py": ("test_source", False),
    "tests/test_s2_parent_bridge.py": ("test_source", False),
    "tools/build_stage2_source_manifest.py": ("manifest_builder", True),
    "tools/build_task2_offline_rl_transitions.py": ("g1_blocked_builder", False),
    "tools/preflight_s2_common.py": ("preflight_library", True),
    "tools/preflight_s2_differentiable_flow_gpu.py": ("g3_preflight", True),
    "tools/preflight_s2_g3_gradient_matrix_gpu.py": ("g3_measurement", True),
    "tools/preflight_s2_parent_bridge.py": ("g0_preflight", True),
}
CONRFT_ENTRIES = {
    "LICENSE": ("conrft_license", False),
    "examples/experiments/resnet10_params.pkl": ("pretrained_resnet10", False),
    "examples/train_reward_classifier.py": ("reward_classifier_training_source", False),
    "serl_launcher/serl_launcher/common/encoding.py": ("encoding_wrapper_source", False),
    "serl_launcher/serl_launcher/networks/classifier.py": ("classifier_source", False),
    "serl_launcher/serl_launcher/networks/reward_classifier.py": (
        "reward_classifier_source",
        False,
    ),
    "serl_launcher/serl_launcher/vision/resnet_v1.py": ("resnet_encoder_source", False),
}


def _discover_stage2_files(root: Path) -> set[str]:
    files = set()
    patterns = (
        "configs/stage2_*",
        "labels/task2_episode_outcomes.v1.json",
        "src/forcesmolvla/rft/*.py",
        "tests/test_rft_*.py",
        "tests/test_s2_*.py",
        "tools/build_stage2_source_manifest.py",
        "tools/build_task2_offline_rl_transitions.py",
        "tools/preflight_s2_*.py",
    )
    for pattern in patterns:
        files.update(
            path.relative_to(root).as_posix()
            for path in root.glob(pattern)
            if path.is_file()
        )
    files.add(
        "docs/ForceRFT_Stage2_Offline_TwinQ_Implementation_Spec_v3_v4_2_aligned.md"
    )
    files.add("docs/ForceRFT_Stage2_Offline_TwinQ_Implementation_Spec_v4.md")
    return files


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _entry(base: Path, relative: str, role: str, runtime_imported: bool) -> dict:
    path = base / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return {
        "relative_path": relative,
        "sha256": sha256_file(path),
        "file_size": path.stat().st_size,
        "artifact_role": role,
        "runtime_imported": runtime_imported,
    }


def build_manifest(root: Path) -> dict:
    discovered = _discover_stage2_files(root)
    expected = set(V4_SOURCE_ENTRIES)
    if discovered != expected:
        raise RuntimeError(
            f"STAGE2_SOURCE_CLOSURE_REGISTRY_MISMATCH:"
            f"missing={sorted(expected - discovered)}:unlisted={sorted(discovered - expected)}"
        )
    files = [
        _entry(root, relative, role, runtime_imported)
        for relative, (role, runtime_imported) in sorted(V4_SOURCE_ENTRIES.items())
    ]
    active = next(entry for entry in files if entry["relative_path"] == V4_SPEC)
    historical = next(
        entry for entry in files if entry["relative_path"] == HISTORICAL_V3_SPEC
    )
    if active["sha256"] != V4_SPEC_SHA256:
        raise RuntimeError("STAGE2_V4_ACTIVE_SPEC_SHA256_MISMATCH")
    if historical["sha256"] != HISTORICAL_V3_SPEC_SHA256:
        raise RuntimeError("STAGE2_HISTORICAL_V3_SPEC_SHA256_MISMATCH")

    parent_config = json.loads(
        (root / "configs/stage2_parent_bridge.development.json").read_text(
            encoding="utf-8"
        )
    )
    qualification = [
        _entry(root, item["path"], "stage1_p4_to_p8_qualification", False)
        for item in parent_config["parent_p4_to_p8_qualification_artifacts"]
    ]
    qualification.sort(key=lambda item: item["relative_path"])
    expected_qualification = {
        item["path"]: item["sha256"]
        for item in parent_config["parent_p4_to_p8_qualification_artifacts"]
    }
    if any(
        entry["sha256"] != expected_qualification[entry["relative_path"]]
        for entry in qualification
    ):
        raise RuntimeError("STAGE2_PARENT_QUALIFICATION_CONFIG_SHA256_MISMATCH")

    checkpoint_relative = parent_config["parent_checkpoint"]
    checkpoint_root = root / checkpoint_relative
    artifact_manifest_path = checkpoint_root / "artifact_manifest.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    model_payload = artifact_manifest["payloads"]["model.safetensors"]
    model_path = checkpoint_root / "model.safetensors"
    if (
        sha256_file(artifact_manifest_path)
        != parent_config["parent_checkpoint_artifact_manifest_sha256"]
        or sha256_file(model_path) != model_payload["sha256"]
        or model_path.stat().st_size != model_payload["size_bytes"]
    ):
        raise RuntimeError("STAGE2_PARENT_CHECKPOINT_BINDING_DRIFT")
    parent_checkpoint = {
        "relative_path": checkpoint_relative,
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        "artifact_manifest_file_size": artifact_manifest_path.stat().st_size,
        "model_safetensors_sha256": model_payload["sha256"],
        "model_safetensors_file_size": model_payload["size_bytes"],
        "strict_reload_required": True,
    }

    conrft_root = CONRFT_ROOT.resolve()
    conrft_status = _git(conrft_root, "status", "--porcelain")
    if conrft_status:
        raise RuntimeError(f"STAGE2_CONRFT_WORKTREE_NOT_CLEAN:{conrft_status.splitlines()}")
    conrft_files = [
        _entry(conrft_root, relative, role, runtime_imported)
        for relative, (role, runtime_imported) in sorted(CONRFT_ENTRIES.items())
    ]
    conrft = {
        "repository_path": str(conrft_root),
        "git_remote_url": _git(conrft_root, "remote", "get-url", "origin"),
        "git_head_sha": _git(conrft_root, "rev-parse", "HEAD"),
        "git_describe": _git(conrft_root, "describe", "--always", "--dirty", "--tags"),
        "git_diff_status": "clean",
        "license_sha256": next(
            entry["sha256"] for entry in conrft_files if entry["relative_path"] == "LICENSE"
        ),
        "runtime_imported": False,
        "environment_binding_status": "pending_R0",
        "jax_flax_optax_versions": "pending_R0",
        "files": conrft_files,
        "files_sha256": canonical_sha256(conrft_files),
    }
    closure = {
        "files": files,
        "qualification_files": qualification,
        "parent_checkpoint": parent_checkpoint,
        "conrft_repository": conrft,
    }
    return {
        "schema_version": "2.0",
        "artifact_status": "development_only",
        "active_specification": active,
        "historical_specification": historical,
        "git_head": _git(root, "rev-parse", "HEAD"),
        "self_included": False,
        "self_exclusion_reason": "a content-addressed manifest cannot include its own SHA256",
        "files": files,
        "files_sha256": canonical_sha256(files),
        "qualification_files": qualification,
        "qualification_files_sha256": canonical_sha256(qualification),
        "parent_checkpoint": parent_checkpoint,
        "conrft_repository": conrft,
        "closure_sha256": canonical_sha256(closure),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/development/stage2/stage2_source_manifest.v4.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite source manifest: {output}")
    payload = build_manifest(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    print(json.dumps({"path": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
