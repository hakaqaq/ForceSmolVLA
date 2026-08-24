"""RuleSpec validation and formal fail-closed enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml


def load_and_validate_rulespec(path: Path, schema_path: Path, *, formal: bool) -> dict:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    artifact = yaml.safe_load(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(artifact)
    if formal:
        if artifact["artifact_status"] != "approved":
            raise PermissionError("formal mode requires an approved RuleSpec")
        if artifact["approval"]["status"] != "approved":
            raise PermissionError("formal mode requires experiment-lead approval")
        if artifact["signature"]["status"] != "verified":
            raise PermissionError("formal mode requires a verified detached signature")
        unresolved = [
            rule["rule_id"]
            for rule in artifact["rules"]
            if rule["threshold"]["value"] is None
            or rule["threshold"]["approval_status"] != "approved"
        ]
        if unresolved:
            raise PermissionError(f"formal mode has unresolved rules: {unresolved}")
        raise PermissionError("formal RuleSpec detached-signature verifier is not configured")
    return artifact
