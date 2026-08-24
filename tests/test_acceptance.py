import json

import pytest

from forcesmolvla.acceptance import load_development_parity_threshold


def _payload():
    return {
        "schema_version": "1.0",
        "config_id": "test-only",
        "mode": "development_only",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "production_allowed": False,
        "operator_overrides_allowed": False,
        "thresholds": {
            "P4": {
                "fp32": {"atol": 1e-5, "rtol": 0.0},
                "bf16": {
                    "prefix_hidden_atol": 0.3,
                    "velocity_cache_atol": 0.1,
                    "rtol": 0.0,
                    "approval_scope": "P4_development_only",
                    "comparisons": {
                        "prefix_hidden": {
                            "applies_to": [
                                "prefill_prefix_vs_full_prefix_valid_tokens"
                            ],
                            "dtype": (
                                "bfloat16_autocast_compared_after_float32_cast"
                            ),
                            "lhs": (
                                "prefill_prefix_hidden_at_valid_physical_prefix_tokens"
                            ),
                            "rhs": (
                                "full_forward_prefix_hidden_at_same_valid_physical_"
                                "prefix_tokens"
                            ),
                            "direction": (
                                "max_abs(lhs-rhs) <= prefix_hidden_atol + "
                                "rtol*prefix_reference_scale"
                            ),
                            "observed_prefix_hidden_max_abs": 0.25,
                        },
                        "velocity_cache": {
                            "applies_to": [
                                "cached_batch_vs_full_batch",
                                "cached_batch_vs_cached_single",
                                "full_batch_vs_full_single",
                                "cached_single_vs_full_single",
                                "cached_10_step_vs_uncached_10_step",
                            ],
                            "dtype": (
                                "bfloat16_autocast_compared_after_float32_cast"
                            ),
                            "direction": (
                                "each max_abs(lhs-rhs) <= velocity_cache_atol + "
                                "rtol*velocity_reference_scale"
                            ),
                        },
                    },
                    "structural_contracts_exact": [
                        "prefix_layout",
                        "prefix_mask",
                        "prefix_physical_length",
                        "invalid_suffix_velocity_zero",
                        "cache_append_crop_restoration",
                        "cache_snapshot_unchanged",
                    ],
                },
            },
            "P8": {
                "fp32": {"atol": 2e-5, "rtol": 0.0},
                "bf16": {"atol": 0.1, "rtol": 0.0},
            },
        },
        "formal_thresholds": {
            "P4": {
                "bf16": {
                    "prefix_hidden_atol": None,
                    "velocity_cache_atol": None,
                    "rtol": None,
                    "approval_status": "unapproved",
                }
            },
            "P8": {
                "bf16": {
                    "atol": None,
                    "rtol": None,
                    "approval_status": "unapproved",
                }
            },
        },
        "detached_signature": None,
    }


def test_loads_hash_bound_development_threshold(tmp_path):
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    result = load_development_parity_threshold(path, gate="P4", precision="fp32")
    assert result.atol == 1e-5
    assert result.prefix_hidden_atol == 1e-5
    assert result.velocity_cache_atol == 1e-5
    assert result.rtol == 0.0
    assert len(result.config_sha256) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "production"),
        ("acceptance_status", "approved"),
        ("formal_eligible", True),
        ("production_allowed", True),
        ("operator_overrides_allowed", True),
        ("detached_signature", "dev-key"),
    ],
)
def test_rejects_non_development_contract(tmp_path, field, value):
    payload = _payload()
    payload[field] = value
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="PARITY_ACCEPTANCE_CONFIG_CONTRACT_DRIFT"):
        load_development_parity_threshold(path, gate="P4", precision="fp32")


def test_loads_scoped_p4_bf16_threshold_without_changing_p8(tmp_path):
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    p4 = load_development_parity_threshold(path, gate="P4", precision="bf16")
    p8 = load_development_parity_threshold(path, gate="P8", precision="bf16")
    assert p4.atol is None
    assert p4.prefix_hidden_atol == 0.3
    assert p4.velocity_cache_atol == 0.1
    assert p8.atol == 0.1
    assert p8.prefix_hidden_atol == 0.1
    assert p8.velocity_cache_atol == 0.1


def test_rejects_p4_bf16_scope_drift(tmp_path):
    payload = _payload()
    payload["thresholds"]["P4"]["bf16"]["comparisons"]["prefix_hidden"][
        "observed_prefix_hidden_max_abs"
    ] = 0.24
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="P4_BF16_COMPARISON_SCOPE_DRIFT"):
        load_development_parity_threshold(path, gate="P4", precision="bf16")


def test_rejects_non_null_formal_bf16_threshold(tmp_path):
    payload = _payload()
    payload["formal_thresholds"]["P4"]["bf16"]["prefix_hidden_atol"] = 0.3
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="PARITY_FORMAL_BF16_THRESHOLDS_MUST_REMAIN_NULL_UNAPPROVED"
    ):
        load_development_parity_threshold(path, gate="P4", precision="bf16")
