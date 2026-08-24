#!/usr/bin/env python3
"""Run formal conversion gates without creating the target directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forcesmolvla.conversion_gate import formal_conversion_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    try:
        result = formal_conversion_preflight(
            raw_root=args.raw_root,
            output_root=args.output_root,
            project_root=args.project_root,
        )
    except (FileNotFoundError, FileExistsError, PermissionError, ValueError) as error:
        print(json.dumps({
            "status": "fail_closed",
            "error_type": type(error).__name__,
            "reason": str(error),
            "output_created": False,
        }, indent=2, sort_keys=True))
        sys.exit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
