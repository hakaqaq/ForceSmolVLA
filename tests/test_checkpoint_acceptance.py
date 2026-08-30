import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from prefix_parity import P8_EXACT_CONTRACTS, load_p8_threshold  # noqa: E402


def test_p8_development_thresholds_are_scoped_and_formal_remains_null():
    path = ROOT / "configs/p8_parity_acceptance.development.json"
    fp32 = load_p8_threshold(path, "fp32")
    bf16 = load_p8_threshold(path, "bf16")
    assert fp32["prefix_hidden_atol"] == fp32["velocity_cache_atol"] == 1e-5
    assert bf16["atol"] is None
    assert bf16["prefix_hidden_atol"] == 0.3
    assert bf16["velocity_cache_atol"] == 0.1
    payload = json.loads(path.read_text())
    assert payload["thresholds"]["P8"]["structural_contracts_exact"] == (
        P8_EXACT_CONTRACTS
    )
    assert all(
        value is None
        for precision in payload["formal_thresholds"]["P8"].values()
        for key, value in precision.items()
        if key != "approval_status"
    )


def test_p8_rejects_non_null_formal_threshold(tmp_path):
    source = ROOT / "configs/p8_parity_acceptance.development.json"
    payload = json.loads(source.read_text())
    payload["formal_thresholds"]["P8"]["fp32"]["atol"] = 1e-5
    path = tmp_path / "p8_acceptance.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(
        RuntimeError, match="P8_FORMAL_THRESHOLDS_MUST_REMAIN_NULL_UNAPPROVED"
    ):
        load_p8_threshold(path, "fp32")
