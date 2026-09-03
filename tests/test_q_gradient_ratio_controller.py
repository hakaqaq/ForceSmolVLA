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


def test_controller_audits_periodically_and_holds_eta_between_audits() -> None:
    controller = QGradientRatioController(
        target_ratio=0.03,
        calibration_interval=3,
    )

    assert controller.should_audit()
    calibrated = controller.update(
        [torch.tensor([2.0])],
        [torch.tensor([4.0])],
        actor_q_valid_count=1,
    )
    assert calibrated.audited
    assert not controller.should_audit()

    held_first = controller.hold(actor_q_valid_count=1)
    held_second = controller.hold(actor_q_valid_count=1)
    assert not held_first.audited
    assert not held_second.audited
    assert held_first.eta == calibrated.eta
    assert held_second.eta == calibrated.eta
    assert controller.should_audit()
