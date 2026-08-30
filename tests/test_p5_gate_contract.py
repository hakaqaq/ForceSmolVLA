import json
from pathlib import Path
import pytest


ROOT = Path(__file__).parents[1]
from forcesmolvla.training_runtime import (
    validate_dense_compute_spec as _validate_static_spec,
    validate_action_target_prerequisite as _validate_action_target_prerequisite,
)


def test_p5_entry_is_bound_to_current_p4_r7_evidence():
    spec = json.loads(
        (ROOT / "configs/p5_force_token_dense_compute.development.json").read_text()
    )
    _validate_static_spec(spec)
    observed = _validate_action_target_prerequisite(ROOT, spec)
    assert observed["acceptance_config_sha256"] == (
        "fefe35a92bcbf22de67c7c7b43e9f97d2658afef41745182b2b8207f750592f4"
    )
    assert observed["source_binding_sha256"] == (
        "dd205856db61de6ed5f0b5406a706ddc127ca7a928406a756645cede73f8fec5"
    )
    assert set(observed["artifacts"]) == {"fp32", "bf16"}


def test_p5_entry_rejects_p4_artifact_hash_drift():
    spec = json.loads(
        (ROOT / "configs/p5_force_token_dense_compute.development.json").read_text()
    )
    spec["p4_prerequisite"]["artifacts"]["bf16"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="P5_P4_BF16_ARTIFACT_HASH_MISMATCH"):
        _validate_action_target_prerequisite(ROOT, spec)
