#!/usr/bin/env python3
"""Append-only wiring correction for the zero-update Stage-2B attempt."""

from pathlib import Path

import benchmark_stage2_batch_scaling_gpu as benchmark
from run_s2_g7_batch_candidate import partial_mask_audit
import run_stage2b_long_run_half_pass_worker as worker


ROOT = Path(__file__).parents[1].resolve()
benchmark.partial_mask_audit = partial_mask_audit
worker.SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v22_stage2b_long_run_half_pass.json"


if __name__ == "__main__":
    worker.main()
