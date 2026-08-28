#!/usr/bin/env python3
"""Append-only report-field compatibility binding for Stage-2B recovery."""

from pathlib import Path

import run_stage2b_long_run_half_pass_worker_v4  # installs v2 mask/action bindings
import run_stage2b_long_run_half_pass_worker as worker
from forcesmolvla.rft.frozen_vlm_trainability import TrainabilityManifest


ROOT = Path(__file__).parents[1].resolve()

# The v4 process completed and checkpointed cycle 105 before report serialization
# referenced the pre-contract attribute name. Keep the frozen implementation
# untouched and expose the old read-only spelling only inside this recovery worker.
if not hasattr(TrainabilityManifest, "trainable_parameter_count"):
    TrainabilityManifest.trainable_parameter_count = property(  # type: ignore[attr-defined]
        lambda self: self.trainable_actor_parameter_count
    )

worker.SOURCE = ROOT / (
    "artifacts/development/stage2/"
    "stage2_source_manifest.v25_stage2b_long_run_recovery.json"
)


if __name__ == "__main__":
    worker.main()
