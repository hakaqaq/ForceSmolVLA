#!/usr/bin/env python3
"""Reconcile the frozen P8 dataset hash with the R0 whole-tree hash.

This utility is deliberately read-only with respect to the dataset.  It imports
the original Stage-1 implementation instead of reimplementing the P8 digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forcesmolvla.dataset_binding import dataset_storage_binding  # noqa: E402


EXPECTED_P8_SHA256 = "f9935b6479dc851e49444669065d20b8aef8cb3ad382f77f53391f701a55a58d"
EXPECTED_R0_SHA256 = "daa3d3b876cddc25caa4effa1e7ac8c55e875738367304c4d51a18653118aa01"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def r0_records(dataset_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(dataset_root).as_posix(),
            "file_size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(candidate for candidate in dataset_root.rglob("*") if candidate.is_file())
    ]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/task2_lerobotv3")
    parser.add_argument(
        "--p8-binding",
        type=Path,
        default=ROOT / "artifacts/development/p8_v4_2_r7_source_binding.json",
    )
    parser.add_argument(
        "--r0-artifact",
        type=Path,
        default=ROOT / "artifacts/development/stage2/s2_r0_preparation.v4.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/development/stage2/dataset_hash_bridge.v4.json",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    p8 = dataset_storage_binding(dataset_root)
    if p8["tree_sha256"] != EXPECTED_P8_SHA256:
        raise RuntimeError(f"STAGE1_DATA_DRIFT: {p8['tree_sha256']}")

    frozen_p8 = json.loads(args.p8_binding.read_text())["dataset"]["storage_tree"]
    if p8 != frozen_p8:
        raise RuntimeError("STAGE1_DATA_DRIFT: current P8 records differ from frozen binding")

    records = r0_records(dataset_root)
    r0_sha = canonical_sha(records)
    if r0_sha != EXPECTED_R0_SHA256:
        raise RuntimeError(f"R0_DATASET_TREE_DRIFT: {r0_sha}")

    p8_records = [
        {
            "relative_path": relative,
            "file_size": (dataset_root / relative).stat().st_size,
            "sha256": value,
        }
        for relative, value in p8["files"].items()
    ]
    p8_by_path = {record["relative_path"]: record for record in p8_records}
    r0_by_path = {record["relative_path"]: record for record in records}
    common = sorted(p8_by_path)
    mismatches = [relative for relative in common if p8_by_path[relative] != r0_by_path.get(relative)]
    if mismatches:
        raise RuntimeError(f"DATA_PAYLOAD_FILE_MISMATCH: {mismatches}")
    r0_only = [r0_by_path[relative] for relative in sorted(set(r0_by_path) - set(p8_by_path))]

    payload = {
        "schema_version": "force_rft_dataset_hash_bridge.v4",
        "status": "PASS_DIFFERENT_HASH_DEFINITIONS_SAME_FROZEN_PAYLOAD",
        "dataset_root": str(dataset_root),
        "p8_original_implementation": {
            "source_path": "src/forcesmolvla/dataset_binding.py",
            "source_sha256": sha256(ROOT / "src/forcesmolvla/dataset_binding.py"),
            "function": "dataset_storage_binding",
            "algorithm": "sha256(concat(sorted(relative_path + NUL + file_sha256 + LF)))",
            "included_paths": ["data/**", "videos/**", "meta/**"],
            "excluded_paths": ["all dataset-root files outside data/, videos/, meta/"],
            "file_count": len(p8_records),
            "total_file_size": sum(record["file_size"] for record in p8_records),
            "tree_sha256": p8["tree_sha256"],
            "files": p8_records,
            "frozen_binding_path": args.p8_binding.relative_to(ROOT).as_posix(),
            "frozen_binding_sha256": sha256(args.p8_binding),
            "frozen_binding_exact_match": True,
        },
        "r0_preparation_implementation": {
            "source_path": "tools/preflight_s2_r0_preparation.py",
            "source_sha256": sha256(ROOT / "tools/preflight_s2_r0_preparation.py"),
            "function": "_tree_sha",
            "algorithm": "sha256(canonical_json(sorted(relative_path,file_size,sha256)))",
            "included_paths": ["all regular files recursively below dataset root"],
            "excluded_paths": [],
            "file_count": len(records),
            "total_file_size": sum(record["file_size"] for record in records),
            "tree_sha256": r0_sha,
            "files": records,
            "r0_artifact_path": args.r0_artifact.relative_to(ROOT).as_posix(),
            "r0_artifact_sha256": sha256(args.r0_artifact),
        },
        "bridge_proof": {
            "p8_payload_is_exact_subset_of_r0_tree": True,
            "common_payload_file_count": len(common),
            "common_payload_total_file_size": sum(p8_by_path[path]["file_size"] for path in common),
            "common_relative_paths": common,
            "common_records_match_path_size_sha256": True,
            "r0_only_file_count": len(r0_only),
            "r0_only_files": r0_only,
            "p8_only_file_count": 0,
            "p8_only_files": [],
            "interpretation": (
                "Both digests bind the identical immutable data/videos/meta payload. "
                "The R0 digest additionally binds six dataset-root manifests/auxiliary files "
                "and uses a different aggregate serialization."
            ),
            "stage1_data_drift": False,
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "p8": p8["tree_sha256"], "r0": r0_sha}))


if __name__ == "__main__":
    main()
