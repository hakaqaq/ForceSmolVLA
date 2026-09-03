from __future__ import annotations

import pytest
import torch

from forcesmolvla.rft.online.q_gradient_controller import (
    QGradientRatioController,
)


def test_controller_targets_parameter_gradient_ratio() -> None:
    controller = QGradientRatioController(target_ratio=0.03)
    decision = controller.update(
        [torch.tensor([6.0, 8.0])],
        [torch.tensor([0.0, 20.0])],
        actor_q_valid_count=4,
    )

    assert decision.eta == pytest.approx(0.015)
    assert decision.applied_ratio == pytest.approx(0.03)
    assert decision.q_grad_norm_weighted == pytest.approx(0.3)


def test_controller_records_gradient_cosine_without_using_it() -> None:
    controller = QGradientRatioController(target_ratio=0.03)
    decision = controller.update(
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([-1.0, 0.0])],
        actor_q_valid_count=1,
    )

    assert decision.cosine == pytest.approx(-1.0)
    assert decision.eta > 0.0

