#!/usr/bin/env python3
"""Append-only coordinator for the corrected Stage-2B half-pass worker."""

from pathlib import Path

import run_stage2b_long_run_half_pass as coordinator


ROOT = Path(__file__).parents[1].resolve()
coordinator.WORKER = ROOT / "tools/run_stage2b_long_run_half_pass_worker_v2.py"
coordinator.SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v22_stage2b_long_run_half_pass.json"
coordinator.ARTIFACT = ROOT / "artifacts/development/stage2/s2_stage2b_long_run_half_pass.v2.json"
coordinator.REPORT = ROOT / "docs/stage2b_long_run_half_pass_report.v2.md"


if __name__ == "__main__":
    coordinator.main()
