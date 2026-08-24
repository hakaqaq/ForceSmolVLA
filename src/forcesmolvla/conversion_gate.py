"""Fail-closed formal raw-to-v3 conversion preflight."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml

from .rules import load_and_validate_rulespec


def _formal_artifact(config_root: Path, approved_name: str, development_name: str) -> Path:
    approved = config_root / approved_name
    return approved if approved.is_file() else config_root / development_name


def _validate_json_artifact(path: Path, schema_path: Path) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(artifact)
    return artifact


def formal_conversion_preflight(
    *,
    raw_root: Path,
    output_root: Path,
    project_root: Path,
) -> dict:
    raw_root = raw_root.resolve()
    output_root = output_root.resolve()
    if not (raw_root / "session.json").is_file():
        raise FileNotFoundError("raw root has no session.json")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")

    schema_root = project_root / "schemas"
    config_root = project_root / "configs"
    rules = load_and_validate_rulespec(
        _formal_artifact(
            config_root,
            "force_quality_thresholds.approved.yaml",
            "force_quality_thresholds.development.yaml",
        ),
        schema_root / "rulespec.schema.json",
        formal=True,
    )
    geometry = _validate_json_artifact(
        _formal_artifact(
            config_root,
            "wrench_geometry_spec.approved.json",
            "wrench_geometry_spec.development.json",
        ),
        schema_root / "wrench_geometry_spec.schema.json",
    )
    calibration = _validate_json_artifact(
        _formal_artifact(
            config_root,
            "calibration_bundle.approved.json",
            "calibration_bundle.development.json",
        ),
        schema_root / "calibration_bundle.schema.json",
    )
    for name, artifact in (("geometry", geometry), ("calibration", calibration)):
        if artifact["artifact_status"] != "approved" or artifact["formal_ready"] is not True:
            raise PermissionError(f"formal conversion requires approved {name} artifact")

    filter_spec = json.loads(
        _formal_artifact(
            config_root,
            "wrench_filter_resample_spec.approved.json",
            "wrench_filter_resample_spec.development.json",
        ).read_text()
    )
    if (
        filter_spec["artifact_status"] != "approved"
        or filter_spec["formal_ready"] is not True
        or filter_spec["unresolved_fields"]
    ):
        raise PermissionError("formal conversion requires a fully resolved approved filter spec")

    checklist = yaml.safe_load(
        _formal_artifact(
            config_root,
            "approval_checklist.approved.yaml",
            "approval_checklist.yaml",
        ).read_text()
    )
    if checklist["status"] != "approved":
        raise PermissionError("formal conversion approval checklist is pending")
    missing_decisions = [
        item["field"]
        for item in checklist["thresholds_requiring_experiment_lead_decision"]
        if item.get("approved_value") is None
    ]
    if missing_decisions:
        raise PermissionError(f"formal conversion has unapproved decisions: {missing_decisions}")
    runtime_path = config_root / "converter_runtime_spec.approved.json"
    if not runtime_path.is_file():
        raise PermissionError("formal conversion requires converter_runtime_spec.approved.json")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if (
        runtime.get("artifact_status") != "approved"
        or runtime.get("formal_ready") is not True
        or runtime.get("unresolved_fields")
    ):
        raise PermissionError("formal conversion requires a fully approved runtime contract")
    return {
        "status": "pass",
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "rules": len(rules["rules"]),
        "formal_gate": "all approved artifacts and detached signatures verified",
    }
