import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from preflight_p8_checkpoint_gpu import (  # noqa: E402
    _validate_contract,
    _validate_p7_prerequisite,
)


def _contract():
    return json.loads(
        (ROOT / "configs/p8_checkpoint_contract.development.json").read_text()
    )


def test_p8_freezes_all_p7_evidence_and_fixture_hashes():
    contract = _contract()
    _validate_contract(contract)
    observed = _validate_p7_prerequisite(
        ROOT,
        contract,
        dataset_root=ROOT / "datasets/task2_lerobotv3",
        repo_id="local/task2_lerobotv3",
    )
    assert observed == {
        "training_recipe": "bdaba473fce1d132d2ab8d481a2a8056f32bf142a3159e222476c6db9fd1e9f4",
        "source_binding": "c276405aaed1f3a3f70dff344ac3cbd988037dc87ebf3d9d2db302834223760b",
        "resolved_config": "6c510aabcb8cbdb3bd4ae3596dd455ba8ef117f310986f3a88ed83f828ac2026",
        "gate_result": "120745433fd8e73b0e6427b0b8940daec5a365a900005978fc42c8d02cbbea5e",
        "validation_fixture": "5589500e80dfc9b3a8607cc27aae7237f9672b10037c00029b151d33ca2def50",
    }


def test_p8_rejects_any_parent_p7_hash_drift():
    contract = _contract()
    contract["p7_prerequisite"]["gate_result"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="P8_P7_GATE_RESULT_HASH_MISMATCH"):
        _validate_p7_prerequisite(
            ROOT,
            contract,
            dataset_root=ROOT / "datasets/task2_lerobotv3",
            repo_id="local/task2_lerobotv3",
        )
