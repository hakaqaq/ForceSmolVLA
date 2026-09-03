from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from train_forcerft_actor_critic import (  # noqa: E402
    _accumulate_parameter_grads,
    _install_combined_parameter_grads,
)


def test_microbatch_parameter_gradients_are_combined_once() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    preservation = [None]
    q_gradient = [None]
    _accumulate_parameter_grads(preservation, [torch.tensor([1.0, 2.0])])
    _accumulate_parameter_grads(preservation, [torch.tensor([3.0, 4.0])])
    _accumulate_parameter_grads(q_gradient, [torch.tensor([2.0, 0.0])])
    _accumulate_parameter_grads(q_gradient, [torch.tensor([0.0, 2.0])])

    _install_combined_parameter_grads(
        [parameter], preservation, q_gradient, eta=0.25
    )

    torch.testing.assert_close(parameter.grad, torch.tensor([4.5, 6.5]))

