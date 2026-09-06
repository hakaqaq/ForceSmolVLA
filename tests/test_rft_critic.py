from __future__ import annotations

import inspect

import pytest
import torch

from forcesmolvla.rft.critic import (
    CRITIC_INPUT_DIM,
    RESIDUAL_ACTION_OFFSET,
    RESIDUAL_ACTION_WIDTH,
    ResidualQHead,
    build_twin_q,
    modules_storage_independent,
    polyak_update,
    state_exact,
)


def inputs(batch: int = 4, mask=(True, True, True)) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    return (
        torch.randn(batch, 7, generator=generator),
        torch.randn(batch, 6, generator=generator),
        torch.randn(batch, 6, generator=generator),
        torch.randn(batch, 3, 6, generator=generator),
        torch.randn(batch, 3, 6, generator=generator),
        torch.tensor(mask, dtype=torch.bool).repeat(batch, 1),
        torch.zeros(batch, 1),
        torch.randn(batch, 1, generator=generator),
    )


def test_residual_q_is_a_60_dimensional_image_free_mlp() -> None:
    q = ResidualQHead(hidden_dim=32)
    assert CRITIC_INPUT_DIM == 60
    assert tuple(inspect.signature(q.forward).parameters) == (
        "normalized_state7",
        "normalized_wrench6",
        "normalized_wrench_delta6",
        "base_action_k6",
        "residual_action_k6",
        "action_mask_k",
        "control_source",
        "gripper_command",
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
    values[4] = values[4] + 1000.0
    assert torch.equal(first_output, q(*values))


def test_masked_slots_do_not_affect_q_and_empty_masks_fail() -> None:
    q = ResidualQHead(hidden_dim=32)
    values = list(inputs(batch=1, mask=(True, False, False)))
    expected = q(*values)
    values[3][:, 1:] = 1000.0
    values[4][:, 1:] = -1000.0
    assert torch.equal(expected, q(*values))
    with pytest.raises(ValueError, match="ACTION_MASK_EMPTY"):
        q(*inputs(batch=1, mask=(False, False, False)))


def test_polyak_update_is_simple_in_place_interpolation() -> None:
    source = ResidualQHead(hidden_dim=8)
    target = ResidualQHead(hidden_dim=8)
    before = next(target.parameters()).detach().clone()
    online = next(source.parameters()).detach().clone()
    polyak_update(source, target, 0.005)
    assert torch.allclose(
        next(target.parameters()), before * 0.995 + online * 0.005
    )
