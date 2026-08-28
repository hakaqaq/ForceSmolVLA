#!/usr/bin/env python3
"""Extend the frozen v26 closure with the append-only interruption tools."""

import json
from pathlib import Path

from forcesmolvla.rft.source_manifest import canonical_sha256, sha256_file


ROOT = Path(__file__).parents[1].resolve()
PARENT = ROOT / (
    "artifacts/development/stage2/"
    "stage2_source_manifest.v26_stage2b_long_run_recovery.json"
)
OUTPUT = ROOT / (
    "artifacts/development/stage2/"
    "stage2_source_manifest.v27_stage2b_interrupted_pilot.json"
)
NEW = {
    "tests/test_stage2b_interrupted_pilot_v1.py": ("interruption_test", False),
    "tools/build_stage2b_interrupted_source_manifest_v1.py": ("manifest_builder", False),
    "tools/recover_stage2b_interrupted_pilot_v1.py": ("interruption_coordinator", True),
    "tools/run_stage2b_interrupted_pilot_worker_v1.py": ("interruption_worker", True),
}


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("INTERRUPTED_SOURCE_MANIFEST_APPEND_ONLY")
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
    payload["interruption_scope"] = "cycle136_audit_only_checkpoint"
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
