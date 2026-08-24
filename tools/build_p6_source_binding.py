#!/usr/bin/env python3
"""Build a versioned P6 binding over source, tests, P5 evidence, and dataset shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from preflight_p5_dense_compute_gpu import _sha256, _tree_sha256
from preflight_p6_variants_gpu import (
    _dataset_storage_binding,
    _pytest_evidence_summary,
    _validate_p5_prerequisite,
    _validate_runtime_import_roots,
    _validate_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pytest-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    dataset_root = args.dataset_root.resolve()
    pytest_report = args.pytest_report.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P6 source binding: {output}")

    spec_path = root / "configs/p6_dense_param_moe.development.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    _validate_spec(spec)
    runtime_imports = _validate_runtime_import_roots(root)
    conversion = json.loads(
        (dataset_root / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    repo_id = conversion.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("P6_CONVERSION_REPO_ID_MISSING")
    _validate_p5_prerequisite(
        root, spec, dataset_root=dataset_root, repo_id=repo_id
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
        raise RuntimeError("P6_LEROBOT_VENDOR_DIRTY_WORKTREE")

    excluded_downstream_prefixes = ["test_p7_", "test_p8_", "test_p9_"]
    test_files = [
        path
        for path in sorted((root / "tests").glob("test_*.py"))
        if not any(path.name.startswith(prefix) for prefix in excluded_downstream_prefixes)
    ]
    included_test_files = [path.relative_to(root).as_posix() for path in test_files]
    test_evidence = {
        "format": "junit_xml",
        "report": {
            "path": pytest_report.relative_to(root).as_posix(),
            "sha256": _sha256(pytest_report),
        },
        "selection": {
            "stage": "P6_and_upstream",
            "included_files": included_test_files,
            "excluded_downstream_prefixes": excluded_downstream_prefixes,
        },
        "test_files": {
            path.relative_to(root).as_posix(): _sha256(path) for path in test_files
        },
    }
    _pytest_evidence_summary(root, test_evidence)
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
        "configs/calibration_bundle.development.json",
        "configs/p5_force_token_dense_compute.development.json",
        "configs/p6_dense_param_moe.development.json",
        "configs/parity_acceptance.development.json",
        "configs/training_stage.development.json",
        "configs/wrench_geometry_spec.development.json",
        "tools/build_p5_source_binding.py",
        "tools/build_p6_source_binding.py",
        "tools/preflight_p4_bare_parity.py",
        "tools/preflight_p5_dense_compute_gpu.py",
        "tools/preflight_p6_variants_gpu.py",
    ]
    project_files.extend(
        path.relative_to(root).as_posix()
        for path in sorted((root / "src/forcesmolvla").glob("*.py"))
    )
    binding = {
        "schema_version": "1.0",
        "status": "development_only",
        "formal_eligible": False,
        "stage": "P6",
        "runtime_imports": runtime_imports,
        "p5_prerequisite": spec["p5_prerequisite"],
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
            "storage_tree": _dataset_storage_binding(dataset_root),
        },
        "test_evidence": test_evidence,
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
