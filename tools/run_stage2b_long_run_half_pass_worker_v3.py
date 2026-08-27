#!/usr/bin/env python3
"""Final append-only Stage-2B worker binding after the zero-update attempts."""

from pathlib import Path

import run_stage2b_long_run_half_pass_worker_v2  # installs the mask-audit correction
import run_stage2b_long_run_half_pass_worker as worker


ROOT = Path(__file__).parents[1].resolve()
worker.SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v23_stage2b_long_run_half_pass.json"


if __name__ == "__main__":
    worker.main()
