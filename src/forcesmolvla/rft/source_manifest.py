"""Integrity checks for the uncommitted Stage-2 development source closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_ENTRY_FIELDS = {
    "relative_path",
    "sha256",
    "file_size",
    "artifact_role",
    "runtime_imported",
}


def _validate_entries(base: Path, entries, *, manifest_relative: str | None = None) -> None:
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("STAGE2_SOURCE_MANIFEST_EMPTY")
    paths = [entry.get("relative_path") for entry in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RuntimeError("STAGE2_SOURCE_MANIFEST_PATH_ORDER_OR_DUPLICATE")
    if manifest_relative is not None and manifest_relative in paths:
        raise RuntimeError("STAGE2_SOURCE_MANIFEST_RECURSIVELY_INCLUDES_SELF")
    for entry in entries:
        if set(entry) != _ENTRY_FIELDS:
            raise RuntimeError("STAGE2_SOURCE_MANIFEST_ENTRY_FIELDS_INVALID")
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("STAGE2_SOURCE_MANIFEST_PATH_INVALID")
        path = base / relative
        if (
            not path.is_file()
            or path.stat().st_size != entry["file_size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise RuntimeError(
                f"STAGE2_SOURCE_MANIFEST_FILE_DRIFT:{entry['relative_path']}"
            )
        if not isinstance(entry["artifact_role"], str) or not isinstance(
            entry["runtime_imported"], bool
        ):
            raise RuntimeError("STAGE2_SOURCE_MANIFEST_ENTRY_TYPE_INVALID")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_stage2_source_manifest(root: Path, manifest_path: Path) -> dict:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") not in {"1.0", "2.0"}
        or payload.get("artifact_status") != "development_only"
        or payload.get("self_included") is not False
    ):
        raise RuntimeError("STAGE2_SOURCE_MANIFEST_HEADER_INVALID")
    entries = payload.get("files")
    try:
        manifest_relative = manifest_path.relative_to(root).as_posix()
    except ValueError:
        manifest_relative = None
    _validate_entries(root, entries, manifest_relative=manifest_relative)
    if payload.get("files_sha256") != canonical_sha256(entries):
        raise RuntimeError("STAGE2_SOURCE_MANIFEST_FILES_HASH_MISMATCH")
    if payload["schema_version"] == "2.0":
        active = payload.get("active_specification")
        active_entry = next(
            (entry for entry in entries if entry["relative_path"] == active.get("relative_path")),
            None,
        ) if isinstance(active, dict) else None
        if active_entry != active or active.get("artifact_role") != "active_v4_specification":
            raise RuntimeError("STAGE2_V4_ACTIVE_SPEC_BINDING_INVALID")

        qualification = payload.get("qualification_files")
        _validate_entries(root, qualification)
        if payload.get("qualification_files_sha256") != canonical_sha256(qualification):
            raise RuntimeError("STAGE2_V4_QUALIFICATION_HASH_MISMATCH")
        checkpoint = payload.get("parent_checkpoint")
        if not isinstance(checkpoint, dict):
            raise RuntimeError("STAGE2_V4_PARENT_CHECKPOINT_BINDING_INVALID")
        checkpoint_root = root / checkpoint["relative_path"]
        artifact_manifest = checkpoint_root / "artifact_manifest.json"
        model = checkpoint_root / "model.safetensors"
        if (
            sha256_file(artifact_manifest) != checkpoint["artifact_manifest_sha256"]
            or artifact_manifest.stat().st_size != checkpoint["artifact_manifest_file_size"]
            or sha256_file(model) != checkpoint["model_safetensors_sha256"]
            or model.stat().st_size != checkpoint["model_safetensors_file_size"]
        ):
            raise RuntimeError("STAGE2_V4_PARENT_CHECKPOINT_DRIFT")

        conrft = payload.get("conrft_repository")
        if not isinstance(conrft, dict):
            raise RuntimeError("STAGE2_V4_CONRFT_BINDING_INVALID")
        conrft_root = Path(conrft["repository_path"]).resolve()
        if (
            conrft.get("runtime_imported") is not False
            or conrft.get("environment_binding_status") != "pending_R0"
            or _git(conrft_root, "rev-parse", "HEAD") != conrft.get("git_head_sha")
            or _git(conrft_root, "remote", "get-url", "origin")
            != conrft.get("git_remote_url")
            or _git(conrft_root, "status", "--porcelain")
            or conrft.get("git_diff_status") != "clean"
        ):
            raise RuntimeError("STAGE2_V4_CONRFT_REPOSITORY_DRIFT")
        external_entries = conrft.get("files")
        _validate_entries(conrft_root, external_entries)
        if conrft.get("files_sha256") != canonical_sha256(external_entries):
            raise RuntimeError("STAGE2_V4_CONRFT_FILES_HASH_MISMATCH")
        license_entry = next(
            (entry for entry in external_entries if entry["relative_path"] == "LICENSE"),
            None,
        )
        if license_entry is None or license_entry["sha256"] != conrft.get("license_sha256"):
            raise RuntimeError("STAGE2_V4_CONRFT_LICENSE_BINDING_INVALID")
        closure = {
            "files": entries,
            "qualification_files": qualification,
            "parent_checkpoint": checkpoint,
            "conrft_repository": conrft,
        }
        if payload.get("closure_sha256") != canonical_sha256(closure):
            raise RuntimeError("STAGE2_V4_SOURCE_CLOSURE_HASH_MISMATCH")
    return payload


def stage2_source_manifest_binding(root: Path, manifest_path: Path) -> dict:
    payload = validate_stage2_source_manifest(root, manifest_path)
    binding = {
        "relative_path": (
            manifest_path.resolve().relative_to(root.resolve()).as_posix()
            if manifest_path.resolve().is_relative_to(root.resolve())
            else str(manifest_path.resolve())
        ),
        "sha256": sha256_file(manifest_path),
        "file_size": manifest_path.stat().st_size,
        "file_count": len(payload["files"]),
        "files_sha256": payload["files_sha256"],
    }
    if payload["schema_version"] == "2.0":
        binding.update(
            {
                "schema_version": "2.0",
                "qualification_file_count": len(payload["qualification_files"]),
                "conrft_file_count": len(payload["conrft_repository"]["files"]),
                "closure_sha256": payload["closure_sha256"],
            }
        )
    return binding
