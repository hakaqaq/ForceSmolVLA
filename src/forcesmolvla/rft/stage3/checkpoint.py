"""Stage-3 checkpoint metadata schema validation and JSON CPU round-trip only."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping

from jsonschema import Draft202012Validator


ROOT = Path(__file__).parents[4]
SCHEMA_PATH = ROOT / "schemas/stage3_online_checkpoint.v1.schema.json"


class Stage3CheckpointSchemaError(ValueError):
    pass


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_online_checkpoint_metadata(payload: Mapping) -> dict:
    value = deepcopy(dict(payload))
    errors = sorted(
        Draft202012Validator(_schema()).iter_errors(value),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        path = ".".join(str(item) for item in errors[0].absolute_path)
        raise Stage3CheckpointSchemaError(
            f"STAGE3_CHECKPOINT_SCHEMA:{path}:{errors[0].message}"
        )
    credits = value["credits"]
    if credits["available"] != credits["minted"] - credits["consumed"]:
        raise Stage3CheckpointSchemaError("STAGE3_CHECKPOINT_CREDIT_LEDGER_MISMATCH")
    counters = value["counters"]
    if counters["critic_updates"] != 2 * counters["learner_cycles"]:
        raise Stage3CheckpointSchemaError("STAGE3_CHECKPOINT_CRITIC_COUNTER_MISMATCH")
    if counters["actor_updates"] != counters["learner_cycles"]:
        raise Stage3CheckpointSchemaError("STAGE3_CHECKPOINT_ACTOR_COUNTER_MISMATCH")
    if counters["polyak_updates_per_target"] != counters["critic_updates"]:
        raise Stage3CheckpointSchemaError("STAGE3_CHECKPOINT_POLYAK_COUNTER_MISMATCH")
    return value


def cpu_round_trip_online_checkpoint(payload: Mapping) -> tuple[dict, bytes]:
    value = validate_online_checkpoint_metadata(payload)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    decoded = json.loads(encoded)
    validate_online_checkpoint_metadata(decoded)
    if decoded != value:
        raise Stage3CheckpointSchemaError("STAGE3_CHECKPOINT_CPU_ROUND_TRIP_MISMATCH")
    return decoded, encoded
