#!/usr/bin/env python3
"""Export the task2/episode_000018 native recorded ACK fixture (CPU-only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forcesmolvla.rft.stage3.recorded_ack_export import (  # noqa: E402
    DEFAULT_CAPTURE_MANIFEST,
    DEFAULT_PARENT_BINDING,
    DEFAULT_RAW_EPISODE,
    DEFAULT_RAW_SESSION,
    DEFAULT_TERMINAL_INDEX,
    RecordedAckExportError,
    export_recorded_ack_fixture,
)
from forcesmolvla.rft.stage3.temporal_parity import DEFAULT_RECORDED_FIXTURE  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one provenance-checked real K=3 ACK parity fixture.",
    )
    parser.add_argument("--raw-session", type=Path, default=DEFAULT_RAW_SESSION)
    parser.add_argument("--raw-episode", type=Path, default=DEFAULT_RAW_EPISODE)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECORDED_FIXTURE)
    parser.add_argument("--capture-manifest-output", type=Path, default=DEFAULT_CAPTURE_MANIFEST)
    parser.add_argument("--parent-binding", type=Path, default=DEFAULT_PARENT_BINDING)
    parser.add_argument("--terminal-index", type=Path, default=DEFAULT_TERMINAL_INDEX)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        fixture = export_recorded_ack_fixture(
            raw_session=args.raw_session,
            raw_episode=args.raw_episode,
            output=args.output,
            capture_manifest_output=args.capture_manifest_output,
            parent_binding_path=args.parent_binding,
            terminal_index_path=args.terminal_index,
        )
    except (OSError, KeyError, TypeError, ValueError, RecordedAckExportError) as error:
        missing = getattr(error, "missing_fields", (f"{type(error).__name__}: {error}",))
        print(json.dumps({
            "status": "BLOCKED",
            "fixture_written": False,
            "missing_required_fields": list(missing),
            "ROBOT_COMMAND_COUNT": 0,
        }, indent=2, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "EXPORTED",
        "fixture_written": True,
        "fixture_path": str(args.output.resolve()),
        "fixture_id": fixture["fixture_id"],
        "fixture_kind": fixture["fixture_kind"],
        "synthetic": fixture["synthetic"],
        "action_source": fixture["action_source"],
        "capture_origin": fixture["capture_origin"],
        "ROBOT_COMMAND_COUNT": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
