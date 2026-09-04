"""Small wrist-wrench residual policy for online ForceRFT."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn


class WristWrenchResidualActor(nn.Module):
    """Predict one normalized TCP6 residual; gripper is intentionally absent."""

    def __init__(
        self, hidden_dim: int = 256, max_normalized_residual: float = 0.5
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or not 0.0 < max_normalized_residual <= 1.0:
            raise ValueError("FORCERFT_RESIDUAL_ACTOR_CONFIG_INVALID")
        self.max_normalized_residual = float(max_normalized_residual)
        self.layers = nn.Sequential(
            nn.Linear(25, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    @staticmethod
    def _input(value: Tensor, batch: int, width: int, name: str) -> Tensor:
        if (
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or tuple(value.shape) != (batch, width)
        ):
            raise ValueError(f"FORCERFT_RESIDUAL_ACTOR_{name}_INVALID")
        value = value.float()
        if not torch.isfinite(value).all():
            raise ValueError(f"FORCERFT_RESIDUAL_ACTOR_{name}_NONFINITE")
        return value

    def forward(
        self,
        *,
        normalized_state7: Tensor,
        normalized_wrench6: Tensor,
        normalized_wrench_delta6: Tensor,
        base_action6: Tensor,
    ) -> Tensor:
        batch = int(normalized_state7.shape[0])
        if batch < 1:
            raise ValueError("FORCERFT_RESIDUAL_ACTOR_EMPTY_BATCH")
        features = torch.cat(
            (
                self._input(normalized_state7, batch, 7, "STATE7"),
                self._input(normalized_wrench6, batch, 6, "WRENCH6"),
                self._input(normalized_wrench_delta6, batch, 6, "WRENCH_DELTA6"),
                self._input(base_action6, batch, 6, "BASE_ACTION6"),
            ),
            dim=1,
        )
        output = torch.tanh(self.layers(features)) * self.max_normalized_residual
        if output.shape != (batch, 6) or not torch.isfinite(output).all():
            raise RuntimeError("FORCERFT_RESIDUAL_ACTOR_OUTPUT_INVALID")
        return output.float()


def make_residual_actor_pair(
    *, hidden_dim: int = 256, max_normalized_residual: float = 0.5
) -> tuple[WristWrenchResidualActor, WristWrenchResidualActor]:
    actor = WristWrenchResidualActor(hidden_dim, max_normalized_residual)
    target = deepcopy(actor).eval()
    target.requires_grad_(False)
    return actor, target
