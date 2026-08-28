#!/usr/bin/env python3
"""Extend the frozen v27 closure with append-only throughput-v2 sources."""

from __future__ import annotations

import json
from pathlib import Path

from forcesmolvla.rft.source_manifest import canonical_sha256, sha256_file


ROOT = Path(__file__).parents[1].resolve()
PARENT = ROOT / "artifacts/development/stage2/stage2_source_manifest.v27_stage2b_interrupted_pilot.json"
OUTPUT = ROOT / "artifacts/development/stage2/stage2_source_manifest.v28_throughput_v2.json"
NEW = {
    "configs/stage2_throughput_v2.development.yaml": ("resolved_benchmark_config", True),
    "src/forcesmolvla/rft/throughput_v2.py": ("throughput_runtime", True),
    "tests/test_rft_throughput_v2.py": ("throughput_test", False),
    "tools/benchmark_stage2_throughput_v2_gpu.py": ("throughput_worker_and_coordinator", True),
    "tools/build_stage2_throughput_v2_artifacts_v1.py": ("artifact_builder", False),
    "tools/build_stage2_throughput_v2_source_manifest_v1.py": ("manifest_builder", False),
}


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("THROUGHPUT_V2_SOURCE_MANIFEST_APPEND_ONLY")
    payload = json.loads(PARENT.read_text())
    entries = {item["relative_path"]: item for item in payload["files"]}
    for relative, (role, runtime) in NEW.items():
        path = ROOT / relative
        entries[relative] = {
            "relative_path": relative,
            "sha256": sha256_file(path),
            "file_size": path.stat().st_size,
            "artifact_role": role,
            "runtime_imported": runtime,
        }
    payload["files"] = [entries[name] for name in sorted(entries)]
    payload["files_sha256"] = canonical_sha256(payload["files"])
    payload["supersedes_source_manifest"] = PARENT.relative_to(ROOT).as_posix()
    payload["throughput_v2_scope"] = "benchmark_only_no_training_checkpoint"
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
