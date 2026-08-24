import pytest
import torch

from forcesmolvla.context import ChunkContext


def make_context(generation=0):
    batch, horizon = 2, 50
    valid = torch.ones(batch, horizon, dtype=torch.bool)
    digest = "a" * 64
    return ChunkContext(
        policy_generation=generation,
        raw_state_snapshot=torch.zeros(batch, 7),
        t_ref_ns=torch.tensor([1, 2], dtype=torch.int64),
        tau0_ns=torch.tensor([3, 4], dtype=torch.int64),
        clock_domain_id=("controller", "controller"),
        episode_id=("e0", "e1"),
        session_id=("s0", "s0"),
        sample_id=("x0", "x1"),
        chunk_id=("c0", "c1"),
        action_valid_mask=valid,
        suffix_valid_mask=valid.clone(),
        calibration_bundle_hash=(digest, digest),
        wrench_geometry_spec_hash=(digest, digest),
        normalizer_hash=(digest, digest),
        calibration_mapping_hash_or_none=(None, None),
        wrench_geometry_valid=torch.ones(batch, dtype=torch.bool),
        runtime_artifact_compatible=torch.ones(batch, dtype=torch.bool),
        selected_provenance=({"state_id": 1}, {"state_id": 2}),
    )


def test_chunk_context_is_batch_bound():
    make_context().validate(batch_size=2, horizon=50, policy_generation=0)


def test_reset_generation_invalidates_context():
    with pytest.raises(RuntimeError, match="INVALIDATED_BY_RESET"):
        make_context(generation=0).validate(batch_size=2, horizon=50, policy_generation=1)


def test_context_rejects_short_tail():
    context = make_context()
    context.action_valid_mask[0] = False
    context.suffix_valid_mask[0] = False
    context.action_valid_mask[0, :2] = True
    context.suffix_valid_mask[0, :2] = True
    with pytest.raises(ValueError, match="at least three"):
        context.validate(batch_size=2, horizon=50, policy_generation=0)
