import copy
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from preflight_s2_parent_bridge import _validate_source_bridge  # noqa: E402
from preflight_s2_common import module_state_dict_sha256  # noqa: E402
from build_stage2_source_manifest import build_manifest  # noqa: E402


def _config():
    return json.loads(
        (ROOT / "configs/stage2_parent_bridge.development.json").read_text(encoding="utf-8")
    )


def test_s2_parent_bridge_binds_r5_snapshot_current_head_and_exact_allowlists():
    evidence = _validate_source_bridge(ROOT, _config())

    assert evidence["parent_training_samples"] == 40_000
    assert evidence["parent_optimizer_updates"] == 10_000
    assert evidence["current_git_head"] == "8cd99bff895d46c3f5334acfc308a1571673c483"
    assert evidence["parent_snapshot_changed_files"] == [
        "configs/training_checkpoint_contract.development.json",
        "src/forcesmolvla/inference.py",
        "tools/train_task2_full_gpu.py",
    ]


def test_s2_parent_bridge_rejects_allowlisted_current_hash_drift():
    config = copy.deepcopy(_config())
    config["parent_snapshot_changed_file_allowlist"]["src/forcesmolvla/inference.py"][
        "current_sha256"
    ] = "0" * 64

    with pytest.raises(RuntimeError, match="S2_G0_ALLOWLIST_CURRENT_HASH_MISMATCH"):
        _validate_source_bridge(ROOT, config)


def test_s2_state_dict_hash_supports_scalar_bfloat16_without_mutation():
    module = torch.nn.Module()
    module.register_buffer("scalar", torch.tensor(1.0, dtype=torch.bfloat16))

    before = module.scalar.clone()
    first = module_state_dict_sha256(module)
    second = module_state_dict_sha256(module)

    assert len(first) == 64
    assert first == second
    assert torch.equal(module.scalar, before)


def test_stage2_source_manifest_closes_all_sidecars_without_self_reference():
    payload = build_manifest(ROOT)
    entries = payload["files"]

    assert payload["schema_version"] == "2.0"
    assert payload["self_included"] is False
    assert len(entries) == 21
    assert [entry["relative_path"] for entry in entries] == sorted(
        entry["relative_path"] for entry in entries
    )
    assert "artifacts/development/stage2/stage2_source_manifest.v4.json" not in {
        entry["relative_path"] for entry in entries
    }
    assert all(
        set(entry)
        == {
            "relative_path",
            "sha256",
            "file_size",
            "artifact_role",
            "runtime_imported",
        }
        for entry in entries
    )
    assert payload["active_specification"]["sha256"] == (
        "0d0ad0312e9758ede7b6910b232096dcaeed338d3a7d4b5aa96347d988ecdce4"
    )
    assert len(payload["qualification_files"]) == 10
    assert payload["conrft_repository"]["git_diff_status"] == "clean"
    assert payload["conrft_repository"]["runtime_imported"] is False
    assert payload["conrft_repository"]["environment_binding_status"] == "pending_R0"
    assert len(payload["conrft_repository"]["files"]) == 7
