from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def test_interrupted_worker_only_relaxes_segment_endpoint() -> None:
    import run_stage2b_interrupted_pilot_worker_v1 as worker

    worker._interrupted_require(False, "STAGE2B_SEGMENT_RANGE")
    with pytest.raises(RuntimeError, match="OTHER_FAILURE"):
        worker._interrupted_require(False, "OTHER_FAILURE")


def test_semantic_cycle_excludes_runtime_only_fields() -> None:
    from recover_stage2b_interrupted_pilot_v1 import semantic_cycle

    record = {
        "cycle": 136,
        "cycle_seconds": 99.0,
        "critic_updates": [
            {"loss": {"L_critic": 1.0}},
            {"loss": {"L_critic": 2.0}},
        ],
        "actor_update": {"loss": {
            "flow_matching": 3.0,
            "actor_q_min_twin": 4.0,
            "weighted_total": 5.0,
        }},
    }
    assert semantic_cycle(record) == {
        "cycle": 136,
        "critic_loss": [1.0, 2.0],
        "fm_loss": 3.0,
        "actor_q_loss": 4.0,
        "actor_total_loss": 5.0,
    }
