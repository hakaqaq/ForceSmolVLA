#!/usr/bin/env python3
"""Fetch ConRFT's source-declared public ResNet-10 asset outside its repo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickletools
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
URL = "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl"
CONRFT_LOCAL = Path("/home/rlc123/conrft/examples/experiments/resnet10_params.pkl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_params.pkl",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    provenance = output.with_suffix(".provenance.json")
    if output.exists() or provenance.exists():
        raise FileExistsError("refusing to overwrite prepared ResNet-10 asset")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("wb", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        with urlopen(URL, timeout=60) as response:
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
    try:
        last_opcode = None
        with temporary.open("rb") as stream:
            for opcode, _, _ in pickletools.genops(stream):
                last_opcode = opcode.name
        if last_opcode != "STOP":
            raise RuntimeError("downloaded ResNet-10 pickle is structurally incomplete")
        payload = {
            "schema_version": "1.0.0",
            "artifact_status": "frozen_external_asset",
            "source_authority": "ConRFT fixed commit train_utils.py public release URL",
            "source_url": URL,
            "relative_path": output.relative_to(ROOT).as_posix(),
            "sha256": sha256(temporary),
            "file_size": temporary.stat().st_size,
            "pickle_last_opcode": last_opcode,
            "conrft_local_repository_copy": {
                "path": str(CONRFT_LOCAL),
                "sha256": sha256(CONRFT_LOCAL),
                "file_size": CONRFT_LOCAL.stat().st_size,
                "load_status": "invalid_truncated_pickle",
                "modified": False,
            },
        }
        with tempfile.NamedTemporaryFile(
            "w", dir=output.parent, delete=False, encoding="utf-8"
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            provenance_temporary = Path(stream.name)
        temporary.replace(output)
        provenance_temporary.replace(provenance)
        print(json.dumps(payload, sort_keys=True))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
