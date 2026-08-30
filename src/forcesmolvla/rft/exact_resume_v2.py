"""Append-only strict binding for the frozen training-cycle resume format."""

from __future__ import annotations

import json
from pathlib import Path

from forcesmolvla.rft import exact_resume


TRAINING_CYCLE_CONFIG = Path("configs/forcerft_training_cycle.development.yaml")
TRAINING_CYCLE_SOURCE = Path(
    "artifacts/development/stage2/stage2_source_manifest.v13_g5_v2.json"
)
TRAINING_CYCLE_CONFIG_SHA256 = "a728c4544c11f3ff15ba2b3b7ceca9cea7a068169ddc3913fa5707127f0f0fd0"
TRAINING_CYCLE_SOURCE_SHA256 = "7a4565d1896b93ffad69eda8fff89c548c8e80aa328c498bf62de090a50b5ccf"
TRAINING_CYCLE_MANIFEST_SHA256 = "12dd0087ac6e1abee527901283fa79219f0a7b4cf1109b244be7bfa52b862c95"
TRAINING_CYCLE_TREE_SHA256 = "e33dde34072d2c3fff36d28480c8cb1ee590673e360061e26cf44435dfcc46ce"


def validate_training_cycle_bindings(root: Path, checkpoint: Path) -> dict:
    root, checkpoint = Path(root), Path(checkpoint)
    config = root / TRAINING_CYCLE_CONFIG
    source = root / TRAINING_CYCLE_SOURCE
    if exact_resume.sha256_file(config) != TRAINING_CYCLE_CONFIG_SHA256:
        raise RuntimeError("G6_V2_G5_CONFIG_SHA_MISMATCH")
    if exact_resume.sha256_file(source) != TRAINING_CYCLE_SOURCE_SHA256:
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
        "training_cycle_config_sha256": TRAINING_CYCLE_CONFIG_SHA256,
        "training_cycle_source_manifest_sha256": TRAINING_CYCLE_SOURCE_SHA256,
        "frozen_file_binding_count": len(frozen["files"]),
        "frozen_bindings_sha256": exact_resume.sha256_file(frozen_path),
        "action_contract": "v2",
    }


def install_exact_resume_v2() -> None:
    # The target module names are part of the persisted legacy checkpoint ABI.
    exact_resume.G5_CONFIG_SHA256 = TRAINING_CYCLE_CONFIG_SHA256
    exact_resume.G5_SOURCE_MANIFEST_SHA256 = TRAINING_CYCLE_SOURCE_SHA256
    exact_resume.G5_CHECKPOINT_MANIFEST_SHA256 = TRAINING_CYCLE_MANIFEST_SHA256
    exact_resume.G5_CHECKPOINT_TREE_SHA256 = TRAINING_CYCLE_TREE_SHA256
    exact_resume.validate_g5_bindings = validate_training_cycle_bindings
