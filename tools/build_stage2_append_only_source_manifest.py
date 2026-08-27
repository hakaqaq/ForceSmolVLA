#!/usr/bin/env python3
"""Build a small content-addressed append-only Stage-2 source closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).parents[1].resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _parse_entry(text: str) -> tuple[str, str, bool]:
    try:
        relative, role, runtime = text.rsplit(":", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected PATH:ROLE:true|false") from error
    if runtime not in {"true", "false"}:
        raise argparse.ArgumentTypeError("runtime flag must be true or false")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError("path must be repository-relative")
    return relative, role, runtime == "true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry", action="append", type=_parse_entry, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite source manifest: {output}")
    entries = []
    for relative, role, runtime_imported in sorted(args.entry):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        entries.append(
            {
                "relative_path": relative,
                "sha256": _sha256_file(path),
                "file_size": path.stat().st_size,
                "artifact_role": role,
                "runtime_imported": runtime_imported,
            }
        )
    if len({entry["relative_path"] for entry in entries}) != len(entries):
        raise RuntimeError("duplicate source-manifest path")
    payload = {
        "schema_version": "1.0",
        "artifact_status": "development_only",
        "scope": args.scope,
        "self_included": False,
        "self_exclusion_reason": "content-addressed manifest cannot include itself",
        "files": entries,
        "files_sha256": _canonical_sha256(entries),
    }
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
    print(json.dumps({"path": str(output), "sha256": _sha256_file(output)}))


if __name__ == "__main__":
    main()
