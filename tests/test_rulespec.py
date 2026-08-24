from pathlib import Path

import pytest

from forcesmolvla.rules import load_and_validate_rulespec


ROOT = Path(__file__).parents[1]
SCHEMA = ROOT / "schemas" / "rulespec.schema.json"


@pytest.mark.parametrize(
    "name",
    [
        "force_quality_thresholds.development.yaml",
        "shadow_safety_thresholds.development.yaml",
    ],
)
def test_development_rulespec_validates(name):
    artifact = load_and_validate_rulespec(ROOT / "configs" / name, SCHEMA, formal=False)
    assert artifact["artifact_status"] == "development_only"


def test_formal_mode_fails_closed_before_approval():
    with pytest.raises(PermissionError, match="approved RuleSpec"):
        load_and_validate_rulespec(
            ROOT / "configs" / "force_quality_thresholds.development.yaml",
            SCHEMA,
            formal=True,
        )


def test_test_only_shadow_rulespec_is_schema_valid_but_not_formal():
    path = ROOT / "tests/fixtures/shadow_safety_thresholds.test_only.yaml"
    artifact = load_and_validate_rulespec(path, SCHEMA, formal=False)
    assert artifact["mode"] == "test_only"
    with pytest.raises(PermissionError):
        load_and_validate_rulespec(path, SCHEMA, formal=True)
