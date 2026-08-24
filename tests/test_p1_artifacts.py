import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("schema_name", "artifact_name"),
    [
        ("wrench_geometry_spec.schema.json", "wrench_geometry_spec.development.json"),
        ("calibration_bundle.schema.json", "calibration_bundle.development.json"),
    ],
)
def test_p1_development_artifacts_validate(schema_name, artifact_name):
    schema = json.loads((ROOT / "schemas" / schema_name).read_text())
    artifact = json.loads((ROOT / "configs" / artifact_name).read_text())
    Draft202012Validator(schema).validate(artifact)
    assert artifact["artifact_status"] == "development_only"
    assert artifact["formal_ready"] is False


def test_geometry_approved_cannot_keep_pending_threshold():
    schema = json.loads((ROOT / "schemas" / "wrench_geometry_spec.schema.json").read_text())
    artifact = json.loads(
        (ROOT / "configs" / "wrench_geometry_spec.development.json").read_text()
    )
    artifact["artifact_status"] = "approved"
    artifact["formal_ready"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(artifact)


def test_filter_candidate_is_explicitly_incomplete():
    artifact = json.loads(
        (ROOT / "configs" / "wrench_filter_resample_spec.development.json").read_text()
    )
    assert artifact["formal_ready"] is False
    assert artifact["filter_candidate"]["candidate_status"] == "approval_pending"
    assert len(artifact["filter_candidate"]["sos_coefficients"]) == 2
    assert artifact["filter_candidate"]["warmup_samples"] == 250
    assert artifact["resampler"]["future_interpolation"] == "forbidden"


def test_training_stage_amendment_is_explicit():
    artifact = json.loads((ROOT / "configs" / "training_stage.development.json").read_text())
    assert artifact["current_stage"] == "offline_full_finetune"
    assert artifact["offline_full_finetune"]["all_existing_model_parameters_require_grad"] is True
    assert artifact["online_hil_vlm_frozen"]["implementation_status"] == "not_implemented"
    assert "cross-stage restore forbidden" in artifact["stage_transition"][
        "optimizer_scheduler_scaler_accumulation_state"
    ]
