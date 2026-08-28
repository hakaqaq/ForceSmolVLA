#!/usr/bin/env python3
"""Bind Stage-2B recovery coordination to canonical sampler auditing."""

from pathlib import Path

import resume_stage2b_long_run_half_pass_v5 as coordinator


ROOT = Path(__file__).parents[1].resolve()
coordinator.WORKER = ROOT / "tools/run_stage2b_long_run_half_pass_worker_v6.py"
coordinator.AUDITOR = ROOT / "tools/audit_stage2b_long_run_recovery_boundary_v6.py"
coordinator.SOURCE = ROOT / (
    "artifacts/development/stage2/"
    "stage2_source_manifest.v26_stage2b_long_run_recovery.json"
)
coordinator.ARTIFACT = ROOT / (
    "artifacts/development/stage2/"
    "s2_stage2b_long_run_half_pass.recovered.v6.json"
)
coordinator.REPORT = ROOT / "docs/stage2b_long_run_half_pass_report.recovered.v6.md"


if __name__ == "__main__":
    coordinator.main()
