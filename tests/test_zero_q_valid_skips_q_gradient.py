from __future__ import annotations

from types import SimpleNamespace

import torch

from forcesmolvla.rft.online.training_losses import residual_actor_loss


class Residual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, **kwargs):
        return self.value.expand(len(kwargs["normalized_state7"]), 6)


class NeverQ(torch.nn.Module):
    def forward(self, *_args):
        raise AssertionError("actor_q_valid=false row reached Q")


def test_zero_q_valid_rows_force_zero_q_contribution() -> None:
    batch = SimpleNamespace(
        state7=torch.zeros(2, 7),
        wrench6=torch.zeros(2, 6),
        wrench_delta6=torch.zeros(2, 6),
        base_action_k6=torch.zeros(2, 3, 6),
        action_mask=torch.ones(2, 3, dtype=torch.bool),
        actor_q_valid=torch.zeros(2, dtype=torch.bool),
    )
    loss = residual_actor_loss(
        NeverQ(),
        NeverQ(),
        Residual(),
        batch,
        None,
        actor_q_weight=1.0,
        residual_l2_weight=0.01,
        human_residual_weight=1.0,
    )
    assert loss.actor_q_valid_count == 0
    assert torch.equal(loss.value, torch.zeros_like(loss.value))
