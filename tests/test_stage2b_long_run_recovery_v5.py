from forcesmolvla.rft.frozen_vlm_trainability import TrainabilityManifest

import run_stage2b_long_run_half_pass_worker_v5


def test_v5_exposes_only_the_historical_report_field_alias() -> None:
    manifest = TrainabilityManifest(1, 2, 3, 4, ("frozen",), ("trainable",))
    assert manifest.trainable_actor_parameter_count == 2
    assert manifest.trainable_parameter_count == 2
