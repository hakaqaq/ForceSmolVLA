#!/usr/bin/env python3
"""Bind Stage-2B recovery execution to the v26 source closure."""

from pathlib import Path

import run_stage2b_long_run_half_pass_worker_v5
import run_stage2b_long_run_half_pass_worker as worker


ROOT = Path(__file__).parents[1].resolve()
worker.SOURCE = ROOT / (
    "artifacts/development/stage2/"
    "stage2_source_manifest.v26_stage2b_long_run_recovery.json"
)


if __name__ == "__main__":
    worker.main()
