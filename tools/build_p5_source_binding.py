#!/usr/bin/env python3
"""Build a versioned, development-only P5 source binding without overwriting evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from preflight_p5_dense_compute_gpu import (
    _sha256,
    _tree_sha256,
    _validate_action_target_population_prerequisite,
    _validate_p4_prerequisite,
    _validate_static_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P5 source binding: {output}")

    spec_path = root / "configs/p5_force_token_dense_compute.development.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    _validate_static_spec(spec)
    _validate_p4_prerequisite(root, spec)
    action_target_population_prerequisite = (
        _validate_action_target_population_prerequisite(root, dataset_root)
    )

    vendor_root = root / "vendor/lerobot"
    commit = subprocess.run(
        ["git", "-C", str(vendor_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(vendor_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("P5_LEROBOT_VENDOR_DIRTY_WORKTREE")

    conversion = json.loads(
        (dataset_root / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    repo_id = conversion.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("P5_CONVERSION_REPO_ID_MISSING")

    vendor_files = (
        "pyproject.toml",
        "src/lerobot/policies/pretrained.py",
        "src/lerobot/policies/smolvla/configuration_smolvla.py",
        "src/lerobot/policies/smolvla/modeling_smolvla.py",
        "src/lerobot/policies/smolvla/smolvlm_with_expert.py",
    )
    project_files = [
        "ForceSmolVLA_Implementation_Spec_v4_2.md",
        "artifacts/development/action_delta_spec.json",
        "artifacts/development/action_target_population_parity_r1.json",
        "configs/calibration_bundle.development.json",
        "configs/p5_force_token_dense_compute.development.json",
        "configs/parity_acceptance.development.json",
        "configs/training_stage.development.json",
        "configs/wrench_geometry_spec.development.json",
        "tools/build_p5_source_binding.py",
        "tools/action_target_population_parity_gate.py",
        "tools/preflight_p4_bare_parity.py",
        "tools/preflight_p5_dense_compute_gpu.py",
        "tools/refit_chunk_delta_normalizer.py",
    ]
    project_files.extend(
        path.relative_to(root).as_posix()
        for path in sorted((root / "src/forcesmolvla").glob("*.py"))
    )
    binding = {
        "schema_version": "1.0",
        "status": "development_only",
        "formal_eligible": False,
        "stage": "P5",
        "lerobot": {
            "repository": "https://github.com/huggingface/lerobot.git",
            "tag": "v0.6.0",
            "commit": commit,
            "dirty_worktree": False,
            "files": {
                relative: _sha256(vendor_root / relative) for relative in vendor_files
            },
        },
        "base_assets": {
            "smolvla_revision": "d5ef92b547b2bf36bdd50f18ea6ed6463cb5c5af",
            "base_checkpoint_files": {
                relative: _sha256(root / "assets/base_checkpoint" / relative)
                for relative in ("config.json", "model.safetensors")
            },
            "constructor_tree_sha256": _tree_sha256(
                root / "assets/smolvlm_constructor"
            ),
        },
        "dataset": {
            "repo_id": repo_id,
            "split": "train",
            "split_semantics": "episode_disjoint_within_session_development",
            "manifest_files": {
                relative: _sha256(dataset_root / relative)
                for relative in (
                    "conversion_manifest.json",
                    "normalizer_manifest.json",
                    "split_manifest.json",
                )
            },
        },
        "p4_prerequisite": spec["p4_prerequisite"],
        "action_target_population_prerequisite": action_target_population_prerequisite,
        "forcesmolvla_files": {
            relative: _sha256(root / relative) for relative in sorted(set(project_files))
        },
        "detached_signature": None,
        "signature_status": "development_only_untrusted",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "file_sha256": _sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
