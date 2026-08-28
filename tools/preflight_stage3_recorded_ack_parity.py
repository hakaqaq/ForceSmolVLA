#!/usr/bin/env python3
"""CPU-only recorded-live ACK temporal parity gate for Stage-3 G1B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forcesmolvla.rft.stage3.temporal_parity import (  # noqa: E402
    DEFAULT_RECORDED_FIXTURE,
    TemporalParityError,
    blocked_temporal_parity_report,
    run_recorded_ack_parity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_RECORDED_FIXTURE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixture_path = args.fixture.resolve()
    if not fixture_path.is_file():
        report = blocked_temporal_parity_report(fixture_path)
    else:
        try:
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            report = run_recorded_ack_parity(fixture, fixture_path=fixture_path)
        except (OSError, KeyError, TypeError, ValueError, TemporalParityError) as error:
            report = blocked_temporal_parity_report(
                fixture_path,
                missing_fields=(f"fixture validation/parity error: {type(error).__name__}: {error}",),
            )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if not args.require_pass or report["G1_GATE_PASSED"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
