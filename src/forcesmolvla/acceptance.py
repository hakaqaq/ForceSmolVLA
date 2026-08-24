"""Fail-closed loading for versioned development acceptance thresholds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class ParityThreshold:
    gate: str
    precision: str
    atol: float | None
    prefix_hidden_atol: float
    velocity_cache_atol: float
    rtol: float
    config_id: str
    config_sha256: str


_P4_BF16_PREFIX_COMPARISON = {
    "applies_to": ["prefill_prefix_vs_full_prefix_valid_tokens"],
    "dtype": "bfloat16_autocast_compared_after_float32_cast",
    "lhs": "prefill_prefix_hidden_at_valid_physical_prefix_tokens",
    "rhs": "full_forward_prefix_hidden_at_same_valid_physical_prefix_tokens",
    "direction": (
        "max_abs(lhs-rhs) <= prefix_hidden_atol + rtol*prefix_reference_scale"
    ),
    "observed_prefix_hidden_max_abs": 0.25,
}
_P4_BF16_VELOCITY_COMPARISON = {
    "applies_to": [
        "cached_batch_vs_full_batch",
        "cached_batch_vs_cached_single",
        "full_batch_vs_full_single",
        "cached_single_vs_full_single",
        "cached_10_step_vs_uncached_10_step",
    ],
    "dtype": "bfloat16_autocast_compared_after_float32_cast",
    "direction": (
        "each max_abs(lhs-rhs) <= velocity_cache_atol + "
        "rtol*velocity_reference_scale"
    ),
}
_P4_BF16_EXACT_CONTRACTS = [
    "prefix_layout",
    "prefix_mask",
    "prefix_physical_length",
    "invalid_suffix_velocity_zero",
    "cache_append_crop_restoration",
    "cache_snapshot_unchanged",
]
_FORMAL_BF16_UNAPPROVED = {
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
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_development_parity_threshold(
    path: Path, *, gate: str, precision: str
) -> ParityThreshold:
    """Load immutable development-only parity limits; reject formal/override semantics."""
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    expected_contract = {
        "schema_version": "1.0",
        "mode": "development_only",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "production_allowed": False,
        "operator_overrides_allowed": False,
        "detached_signature": None,
    }
    drift = {
        key: {"actual": payload.get(key), "expected": expected}
        for key, expected in expected_contract.items()
        if payload.get(key) != expected
    }
    if drift:
        raise RuntimeError(f"PARITY_ACCEPTANCE_CONFIG_CONTRACT_DRIFT: {drift}")
    if payload.get("formal_thresholds") != _FORMAL_BF16_UNAPPROVED:
        raise RuntimeError("PARITY_FORMAL_BF16_THRESHOLDS_MUST_REMAIN_NULL_UNAPPROVED")
    config_id = payload.get("config_id")
    if not isinstance(config_id, str) or not config_id:
        raise RuntimeError("PARITY_ACCEPTANCE_CONFIG_ID_MISSING")
    if gate not in {"P4", "P8"} or precision not in {"fp32", "bf16"}:
        raise ValueError("unsupported parity gate or precision")
    try:
        values = payload["thresholds"][gate][precision]
        rtol = float(values["rtol"])
        if gate == "P4" and precision == "bf16":
            if set(values) != {
                "prefix_hidden_atol",
                "velocity_cache_atol",
                "rtol",
                "approval_scope",
                "comparisons",
                "structural_contracts_exact",
            }:
                raise RuntimeError("PARITY_ACCEPTANCE_THRESHOLD_UNKNOWN_FIELD")
            if values["approval_scope"] != "P4_development_only":
                raise RuntimeError("P4_BF16_APPROVAL_SCOPE_DRIFT")
            if values["comparisons"] != {
                "prefix_hidden": _P4_BF16_PREFIX_COMPARISON,
                "velocity_cache": _P4_BF16_VELOCITY_COMPARISON,
            }:
                raise RuntimeError("P4_BF16_COMPARISON_SCOPE_DRIFT")
            if values["structural_contracts_exact"] != _P4_BF16_EXACT_CONTRACTS:
                raise RuntimeError("P4_BF16_EXACT_CONTRACT_SCOPE_DRIFT")
            atol = None
            prefix_hidden_atol = float(values["prefix_hidden_atol"])
            velocity_cache_atol = float(values["velocity_cache_atol"])
        else:
            if set(values) != {"atol", "rtol"}:
                raise RuntimeError("PARITY_ACCEPTANCE_THRESHOLD_UNKNOWN_FIELD")
            atol = float(values["atol"])
            prefix_hidden_atol = atol
            velocity_cache_atol = atol
    except RuntimeError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("PARITY_ACCEPTANCE_THRESHOLD_MISSING") from error
    if prefix_hidden_atol < 0 or velocity_cache_atol < 0 or rtol < 0:
        raise RuntimeError("PARITY_ACCEPTANCE_THRESHOLD_NEGATIVE")
    return ParityThreshold(
        gate=gate,
        precision=precision,
        atol=atol,
        prefix_hidden_atol=prefix_hidden_atol,
        velocity_cache_atol=velocity_cache_atol,
        rtol=rtol,
        config_id=config_id,
        config_sha256=_sha256(resolved),
    )
