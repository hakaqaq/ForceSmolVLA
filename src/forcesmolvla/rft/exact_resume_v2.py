"""Append-only strict G5-v2 binding for the frozen G6 implementation."""

from __future__ import annotations

import json
from pathlib import Path

from forcesmolvla.rft import exact_resume


G5_CONFIG = Path("configs/stage2_g5_single_cycle.v2.development.yaml")
G5_SOURCE = Path("artifacts/development/stage2/stage2_source_manifest.v13_g5_v2.json")
G5_CONFIG_SHA256 = "a728c4544c11f3ff15ba2b3b7ceca9cea7a068169ddc3913fa5707127f0f0fd0"
G5_SOURCE_SHA256 = "7a4565d1896b93ffad69eda8fff89c548c8e80aa328c498bf62de090a50b5ccf"
G5_MANIFEST_SHA256 = "12dd0087ac6e1abee527901283fa79219f0a7b4cf1109b244be7bfa52b862c95"
G5_TREE_SHA256 = "e33dde34072d2c3fff36d28480c8cb1ee590673e360061e26cf44435dfcc46ce"


def validate_g5_v2_bindings(root: Path, checkpoint: Path) -> dict:
    root, checkpoint = Path(root), Path(checkpoint)
    config = root / G5_CONFIG
    source = root / G5_SOURCE
    if exact_resume.sha256_file(config) != G5_CONFIG_SHA256:
        raise RuntimeError("G6_V2_G5_CONFIG_SHA_MISMATCH")
    if exact_resume.sha256_file(source) != G5_SOURCE_SHA256:
        raise RuntimeError("G6_V2_G5_SOURCE_MANIFEST_SHA_MISMATCH")
    config_snapshot = (
        checkpoint
        / "startup_snapshot/resolved_config/stage2_g5_single_cycle.development.yaml"
    )
    source_snapshot = checkpoint / "startup_snapshot/source/stage2_source_manifest.v7_g5.json"
    if config_snapshot.read_bytes() != config.read_bytes():
        raise RuntimeError("G6_V2_G5_CHECKPOINT_CONFIG_SNAPSHOT_MISMATCH")
    if source_snapshot.read_bytes() != source.read_bytes():
        raise RuntimeError("G6_V2_G5_CHECKPOINT_SOURCE_SNAPSHOT_MISMATCH")
    frozen_path = checkpoint / "startup_snapshot/bindings/frozen_inputs_startup.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    for item in frozen["files"].values():
        path = root / item["path"]
        if (
            exact_resume.sha256_file(path) != item["sha256"]
            or path.stat().st_size != item["file_size"]
        ):
            raise RuntimeError(f"G6_V2_FROZEN_FILE_BINDING_MISMATCH:{item['path']}")
    return {
        "g5_config_sha256": G5_CONFIG_SHA256,
        "g5_source_manifest_sha256": G5_SOURCE_SHA256,
        "frozen_file_binding_count": len(frozen["files"]),
        "frozen_bindings_sha256": exact_resume.sha256_file(frozen_path),
        "action_contract": "v2",
    }


def install_exact_resume_v2() -> None:
    exact_resume.G5_CONFIG_SHA256 = G5_CONFIG_SHA256
    exact_resume.G5_SOURCE_MANIFEST_SHA256 = G5_SOURCE_SHA256
    exact_resume.G5_CHECKPOINT_MANIFEST_SHA256 = G5_MANIFEST_SHA256
    exact_resume.G5_CHECKPOINT_TREE_SHA256 = G5_TREE_SHA256
    exact_resume.validate_g5_bindings = validate_g5_v2_bindings
