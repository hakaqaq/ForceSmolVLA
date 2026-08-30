"""Append-only Critic source-manifest verification."""

from __future__ import annotations

from pathlib import Path

from forcesmolvla.rft.source_manifest import validate_stage2_source_manifest


def verify_critic_source_manifest(root: Path, manifest_path: Path) -> dict:
    payload = validate_stage2_source_manifest(root, manifest_path)
    paths = [entry["relative_path"] for entry in payload["files"]]
    if any("manual_reward" in path or path.startswith("labels/") for path in paths):
        raise RuntimeError("G7A_R2_MANUAL_SOURCE_IN_RUNTIME_CLOSURE")
    if payload.get("scope") != "G7A-r2_ActionContract-v2_critic_warmup":
        raise RuntimeError("G7A_R2_SOURCE_SCOPE_INVALID")
    return payload
