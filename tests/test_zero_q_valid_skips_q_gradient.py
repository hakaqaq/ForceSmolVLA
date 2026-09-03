from __future__ import annotations

import torch

from forcesmolvla.rft.online.q_gradient_controller import (
    QGradientRatioController,
)


def test_zero_q_valid_rows_force_zero_q_contribution() -> None:
    decision = QGradientRatioController().update(
        [torch.tensor([1.0])],
        [torch.tensor([100.0])],
        actor_q_valid_count=0,
    )

    assert decision.eta == 0.0
    assert decision.applied_ratio == 0.0
    assert decision.skipped_reason == "no_actor_q_valid_rows"

