import benchmark_stage2_batch_scaling_gpu as benchmark
from run_s2_g7_batch_candidate import partial_mask_audit


def test_correct_mask_audit_is_reused() -> None:
    assert not hasattr(benchmark, "partial_mask_audit")
    assert callable(partial_mask_audit)
