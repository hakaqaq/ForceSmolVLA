import torch

import audit_stage2b_long_run_recovery_boundary_v6
from forcesmolvla.rft.training_cycle import SerializableUniqueSampler


def test_sampler_audit_state_uses_digest_not_tensor_equality() -> None:
    sampler = SerializableUniqueSampler(
        "test", (1, 2, 3), torch.Generator().manual_seed(7)
    )
    state = sampler.state_dict()
    assert "generator_state" not in state
    assert len(state["generator_state_sha256"]) == 64
