#!/usr/bin/env python3
"""Append-only outer-batch compatibility binding for Stage-2B diagnostics."""

from pathlib import Path

import run_s2_g7b_worker as g7b
import run_stage2b_long_run_half_pass_worker_v3  # installs prior corrections
import run_stage2b_long_run_half_pass_worker as worker


ROOT = Path(__file__).parents[1].resolve()
_internal_action_diagnostic = g7b.internal_action_diagnostic


def internal_action_diagnostic(policy, batch, noise, delta_mean, delta_std):
    outer = batch if "current_actor_batch" in batch else {"current_actor_batch": batch}
    return _internal_action_diagnostic(policy, outer, noise, delta_mean, delta_std)


g7b.internal_action_diagnostic = internal_action_diagnostic
worker.SOURCE = ROOT / "artifacts/development/stage2/stage2_source_manifest.v24_stage2b_long_run_half_pass.json"


if __name__ == "__main__":
    worker.main()
