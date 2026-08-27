#!/usr/bin/env python3
"""Final append-only coordinator binding for the Stage-2B half pass."""

from pathlib import Path

import run_stage2b_long_run_half_pass as coordinator


ROOT = Path(__file__).parents[1].resolve()
coordinator.WORKER = ROOT / "tools/run_stage2b_long_run_half_pass_worker_v3.py"
coordinator.SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v23_stage2b_long_run_half_pass.json"
coordinator.ARTIFACT = ROOT / "artifacts/development/stage2/s2_stage2b_long_run_half_pass.v3.json"
coordinator.REPORT = ROOT / "docs/stage2b_long_run_half_pass_report.v3.md"


if __name__ == "__main__":
    coordinator.main()
