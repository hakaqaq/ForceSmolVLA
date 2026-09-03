from __future__ import annotations

import pytest
import torch

from forcesmolvla.rft.online.q_gradient_controller import (
    QGradientRatioController,
)


def test_current_batch_q_ratio_never_exceeds_hard_cap() -> None:
    controller = QGradientRatioController(
        target_ratio=0.10,
        hard_max_ratio=0.10,
    )
    controller.update(
        [torch.tensor([100.0])], [torch.tensor([1.0])], actor_q_valid_count=1
    )
    decision = controller.update(
        [torch.tensor([1.0])], [torch.tensor([100.0])], actor_q_valid_count=1
    )

    assert decision.applied_ratio == pytest.approx(0.10)
    assert decision.eta == pytest.approx(0.001)
    assert decision.hard_cap_applied is True
