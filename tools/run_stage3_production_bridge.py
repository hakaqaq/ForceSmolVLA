#!/usr/bin/env python3
"""Run the CPU-only Stage-3 filesystem shadow bridge for one recorder episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forcesmolvla.rft.stage3.production_bridge import (
    Stage3ProductionBridge,
    load_bridge_config,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/stage3_production_bridge.v1.development.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, raw = load_bridge_config(args.config)
    episode = args.episode or Path(raw["recorded_offline_fixture"]["episode_dir"])
    if not args.dry_run and args.state_root is None:
        raise SystemExit("--state-root is required unless --dry-run is used")
    state_root = args.state_root or Path("/tmp/forcesmolvla_stage3_bridge_dry_run")
    report = Stage3ProductionBridge(
        config=config, state_root=state_root
    ).process_episode(episode, dry_run=args.dry_run)
    print(json.dumps(report.to_dict(), sort_keys=True, indent=2))
    return 0 if report.status in {"DRY_RUN_READY", "SEALED_COMMITTED", "ACTIVE_STAGED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
