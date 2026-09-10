import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("schema_name", "artifact_name"),
    [
        ("wrench_geometry_spec.schema.json", "wrench_geometry_spec.json"),
        ("calibration_bundle.schema.json", "calibration_bundle.json"),
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
        (ROOT / "configs" / "wrench_geometry_spec.json").read_text()
    )
    artifact["artifact_status"] = "approved"
    artifact["formal_ready"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(artifact)


def test_filter_candidate_is_explicitly_incomplete():
    artifact = json.loads(
        (ROOT / "configs" / "wrench_filter_resample_spec.json").read_text()
    )
    assert artifact["formal_ready"] is False
    assert artifact["filter_candidate"]["candidate_status"] == "approval_pending"
    assert len(artifact["filter_candidate"]["sos_coefficients"]) == 2
    assert artifact["filter_candidate"]["warmup_samples"] == 250
    assert artifact["resampler"]["future_interpolation"] == "forbidden"
