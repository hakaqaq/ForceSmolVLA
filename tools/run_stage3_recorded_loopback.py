#!/usr/bin/env python3
"""Run the offline Stage-3 loopback without connecting to external systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forcesmolvla.rft.stage3.loopback import (
    recorded_fixture_blocked_report,
    run_synthetic_loopback,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-kind",
        required=True,
        choices=("synthetic_tool_test", "recorded_live"),
        help="Synthetic is a tool test only; recorded_live remains blocked in G3P.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Optional recorded-live fixture path; never synthesized by this tool.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fixture_kind == "synthetic_tool_test":
        if args.fixture is not None:
            raise SystemExit("--fixture is only valid with --fixture-kind recorded_live")
        report = run_synthetic_loopback(seed=args.seed)
    else:
        report = recorded_fixture_blocked_report(args.fixture)
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
