from __future__ import annotations

import torch

from forcesmolvla.rft.online.q_gradient_controller import (
    QGradientRatioController,
)


def test_controller_resume_matches_uninterrupted_next_update() -> None:
    uninterrupted = QGradientRatioController()
    uninterrupted.update(
        [torch.tensor([2.0])], [torch.tensor([5.0])], actor_q_valid_count=1
    )
    state = uninterrupted.state_dict()

    resumed = QGradientRatioController()
    resumed.load_state_dict(state)
    expected = uninterrupted.update(
        [torch.tensor([3.0])], [torch.tensor([4.0])], actor_q_valid_count=1
    )
    actual = resumed.update(
        [torch.tensor([3.0])], [torch.tensor([4.0])], actor_q_valid_count=1
    )

    assert actual == expected
    assert resumed.state_dict() == uninterrupted.state_dict()

