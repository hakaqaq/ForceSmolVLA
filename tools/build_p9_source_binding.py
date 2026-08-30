#!/usr/bin/env python3
"""Build a versioned P9 binding over P8 r4, P9 code/tests, and task2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preflight_p5_dense_compute_gpu import _sha256
from preflight_p6_variants_gpu import (
    _dataset_storage_binding,
    _pytest_evidence_summary,
    _validate_runtime_import_roots,
)
from preflight_p9_offline_replay import _load_scope_amendment, _validate_contract


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
        raise FileExistsError(f"refusing to overwrite P9 source binding: {output}")

    config_path = root / "configs/p9_shadow_replay.development.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_contract(config)
    scope_amendment, data_scope = _load_scope_amendment(root, config)
    if dataset_root != (root / config["dataset"]).resolve():
        raise RuntimeError("P9_BINDING_DATASET_OUTSIDE_FROZEN_SCOPE")
    conversion = json.loads(
        (dataset_root / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    repo_id = conversion.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise RuntimeError("P9_DATASET_REPO_ID_MISSING")

    p8_binding = json.loads(
        (root / config["p8_prerequisite"]["source_binding"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    runtime_imports = _validate_runtime_import_roots(root)
    test_files = sorted((root / "tests").glob("test_*.py"))
    test_evidence = {
        "format": "junit_xml",
        "report": {
            "path": pytest_report.relative_to(root).as_posix(),
            "sha256": _sha256(pytest_report),
        },
        "selection": {
            "stage": "P9_and_upstream",
            "included_files": [path.relative_to(root).as_posix() for path in test_files],
            "excluded_downstream_prefixes": [],
        },
        "test_files": {
            path.relative_to(root).as_posix(): _sha256(path) for path in test_files
        },
    }
    _pytest_evidence_summary(root, test_evidence)

    project_files = [
        "configs/offline_sft_training_recipe.development.yaml",
        "configs/p9_shadow_replay.development.json",
        "configs/p9_task2_scope_amendment.development.json",
        "configs/shadow_safety_thresholds.development.yaml",
        "configs/task2_development_data_scope.json",
        "schemas/rulespec.schema.json",
        "tests/fixtures/p9_shadow_cases.test_only.json",
        "tests/fixtures/shadow_clock_map.test_only.json",
        "tests/fixtures/shadow_safety_thresholds.test_only.yaml",
        "tests/test_p9_shadow.py",
        "tools/build_p9_source_binding.py",
        "tools/p8_checkpoint_common.py",
        "tools/preflight_p7_two_pass_gpu.py",
        "tools/preflight_p9_offline_replay.py",
        "tools/train_forcesmolvla_sft.py",
    ]
    bound_inputs = {
        item["path"]: item["sha256"]
        for item in config["p8_prerequisite"].values()
        if isinstance(item, dict) and set(item) == {"path", "sha256"}
    }
    bound_inputs[config["scope_amendment"]["path"]] = config["scope_amendment"][
        "sha256"
    ]
    bound_inputs[scope_amendment["task2_data_scope"]["path"]] = scope_amendment[
        "task2_data_scope"
    ]["sha256"]
    bound_inputs[data_scope["training_budget"]["recipe_path"]] = data_scope[
        "training_budget"
    ]["recipe_sha256"]
    binding = {
        "schema_version": "1.0",
        "status": "development_only",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "stage": "P9",
        "runtime_imports": runtime_imports,
        "p8_prerequisite": config["p8_prerequisite"],
        "scope_amendment": config["scope_amendment"],
        "task2_data_scope": scope_amendment["task2_data_scope"],
        "session_provenance": data_scope["session_provenance"],
        "training_budget": data_scope["training_budget"],
        "lerobot": p8_binding["lerobot"],
        "base_assets": p8_binding["base_assets"],
        "dataset": {
            "repo_id": repo_id,
            "split": "val",
            "split_semantics": (
                "episode_disjoint_within_single_collection_scope_development; "
                "physical_session_id_not_recorded"
            ),
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
            relative: _sha256(root / relative) for relative in project_files
        },
        "bound_inputs": bound_inputs,
        "test_only_artifacts_never_formal": True,
        "detached_signature": None,
        "signature_status": "development_only_untrusted",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "file_sha256": _sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
