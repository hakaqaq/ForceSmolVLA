#!/usr/bin/env python3
"""Run the CPU-only approved-hybrid Stage-3 parent preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forcesmolvla.rft.stage3.parent import (
    DEFAULT_CONFIG,
    ParentBindingError,
    preflight_parent_binding,
    render_parent_binding_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/development/stage3/stage3_parent_binding_preflight.v1.json"
DEFAULT_REPORT = ROOT / "docs/stage3_parent_binding_report.v1.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = preflight_parent_binding(args.config)
    except (ParentBindingError, FileNotFoundError, json.JSONDecodeError) as error:
        print(json.dumps({"tool_status": "FAIL", "error": str(error)}, sort_keys=True))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(render_parent_binding_markdown(result), encoding="utf-8")
    print(json.dumps({
        "tool_status": result["tool_status"],
        "binding": result["G0_FINAL_PARENT_BINDING"],
        "output": str(args.output),
        "report": str(args.report),
        "canonical_report_sha256": result["canonical_report_sha256"],
        "CUDA_INITIALIZED": result["CUDA_INITIALIZED"],
        "ROBOT_COMMAND_COUNT": result["ROBOT_COMMAND_COUNT"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
