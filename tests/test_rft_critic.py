from __future__ import annotations

import inspect

import pytest
import torch

from forceprior.rft.critic import (
    CRITIC_INPUT_DIM,
    RESIDUAL_ACTION_OFFSET,
    RESIDUAL_ACTION_WIDTH,
    ResidualQHead,
    build_twin_q,
    modules_storage_independent,
    polyak_update,
    state_exact,
)


def inputs(batch: int = 4) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    return (
        torch.randn(batch, 7, generator=generator),
        torch.randn(batch, 6, generator=generator),
        torch.randn(batch, 6, generator=generator),
        torch.randn(batch, 6, generator=generator),
        torch.zeros(batch, 1),
        torch.randn(batch, 6, generator=generator),
    )


def test_residual_q_is_a_32_dimensional_proposal_space_mlp() -> None:
    q = ResidualQHead(hidden_dim=32)
    assert CRITIC_INPUT_DIM == 32
    assert tuple(inspect.signature(q.forward).parameters) == (
        "normalized_state7",
        "normalized_wrench6",
        "normalized_wrench_delta6",
        "base_action6",
        "base_gripper",
        "residual_proposal6",
    )
    assert not any("camera" in name or "image" in name for name in q.state_dict())
    result = q(*inputs())
    assert result.shape == (4,) and result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_twin_q_heads_are_independent_and_targets_are_exact_copies() -> None:
    q1, q2, q1_target, q2_target = build_twin_q(hidden_dim=32, seed=9)
    assert modules_storage_independent(q1, q2)
    assert modules_storage_independent(q1, q1_target)
    assert modules_storage_independent(q2, q2_target)
    assert state_exact(q1, q1_target)
    assert state_exact(q2, q2_target)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(q1.parameters(), q2.parameters(), strict=True)
    )
    assert all(
        not parameter.requires_grad
        for target in (q1_target, q2_target)
        for parameter in target.parameters()
    )


def test_residual_action_input_columns_start_at_zero() -> None:
    q = ResidualQHead(hidden_dim=32)
    first = q.layers[0]
    action_columns = first.weight[
        :, RESIDUAL_ACTION_OFFSET : RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH
    ]
    other_columns = torch.cat(
        (
            first.weight[:, :RESIDUAL_ACTION_OFFSET],
            first.weight[:, RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH :],
        ),
        dim=1,
    )
    assert torch.count_nonzero(action_columns) == 0
    assert torch.count_nonzero(other_columns) > 0
    values = list(inputs(batch=1))
    first_output = q(*values)
    values[5] = values[5] + 1000.0
    assert torch.equal(first_output, q(*values))


def test_base_gripper_and_proposal_are_separate_critic_inputs() -> None:
    q = ResidualQHead(hidden_dim=32)
    values = list(inputs(batch=1))
    assert q(*values).shape == (1,)
    values[4] = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="BASE_GRIPPER"):
        q(*values)


def test_polyak_update_is_simple_in_place_interpolation() -> None:
    source = ResidualQHead(hidden_dim=8)
    target = ResidualQHead(hidden_dim=8)
    before = next(target.parameters()).detach().clone()
    online = next(source.parameters()).detach().clone()
    polyak_update(source, target, 0.005)
    assert torch.allclose(
        next(target.parameters()), before * 0.995 + online * 0.005
    )
