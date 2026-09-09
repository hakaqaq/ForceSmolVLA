#!/usr/bin/env python3
"""Migrate one bounded-schedule ForceRFT checkpoint to continuous_async."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forcesmolvla.rft.online.schedule_migration import (  # noqa: E402
    migrate_schedule_checkpoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    migrated = migrate_schedule_checkpoint(
        args.source_checkpoint, args.target_config, args.destination
    )
    print(migrated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
