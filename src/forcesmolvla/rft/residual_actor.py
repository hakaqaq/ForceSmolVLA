"""Small wrist-wrench residual policy for online ForceRFT."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


def resolve_residual_cap6(
    normalizer: object, actor_config: Mapping[str, object]
) -> Tensor:
    """Resolve the normalized per-axis proposal cap from frozen action units."""

    sigma6 = np.asarray(normalizer.delta_action7.std[:6], dtype=np.float64)
    physical_limit6 = np.asarray(
        [
            *(
                [float(actor_config["max_translation_residual_per_axis_m"])]
                * 3
            ),
            *([float(actor_config["max_rpy_residual_per_axis_rad"])] * 3),
        ],
        dtype=np.float64,
    )
    normalized_limit = float(actor_config["max_normalized_residual"])
    if (
        sigma6.shape != (6,)
        or not np.isfinite(sigma6).all()
        or not np.all(sigma6 > 0.0)
        or not np.isfinite(physical_limit6).all()
        or not np.all(physical_limit6 > 0.0)
        or not 0.0 < normalized_limit <= 1.0
    ):
        raise ValueError("FORCERFT_RESIDUAL_ACTOR_CAP_CONFIG_INVALID")
    cap6 = np.minimum(normalized_limit, physical_limit6 / sigma6)
    return torch.tensor(cap6, dtype=torch.float32)


class WristWrenchResidualActor(nn.Module):
    """Predict one normalized TCP6 residual; gripper is intentionally absent."""

    def __init__(
        self,
        hidden_dim: int = 256,
        max_normalized_residual: float = 0.5,
        residual_cap6: Sequence[float] | Tensor | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or not 0.0 < max_normalized_residual <= 1.0:
            raise ValueError("FORCERFT_RESIDUAL_ACTOR_CONFIG_INVALID")
        self.max_normalized_residual = float(max_normalized_residual)
        cap6 = torch.as_tensor(
            [self.max_normalized_residual] * 6
            if residual_cap6 is None
            else residual_cap6,
            dtype=torch.float32,
        )
        if (
            tuple(cap6.shape) != (6,)
            or not torch.isfinite(cap6).all()
            or not bool((cap6 > 0.0).all())
            or not bool((cap6 <= self.max_normalized_residual).all())
        ):
            raise ValueError("FORCERFT_RESIDUAL_ACTOR_CAP6_INVALID")
        self.register_buffer("residual_cap6", cap6.detach().clone())
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
        output = torch.tanh(self.layers(features)) * self.residual_cap6
        if output.shape != (batch, 6) or not torch.isfinite(output).all():
            raise RuntimeError("FORCERFT_RESIDUAL_ACTOR_OUTPUT_INVALID")
        return output.float()


def make_residual_actor_pair(
    *,
    hidden_dim: int = 256,
    max_normalized_residual: float = 0.5,
    residual_cap6: Sequence[float] | Tensor | None = None,
) -> tuple[WristWrenchResidualActor, WristWrenchResidualActor]:
    actor = WristWrenchResidualActor(
        hidden_dim, max_normalized_residual, residual_cap6
    )
    target = deepcopy(actor).eval()
    target.requires_grad_(False)
    return actor, target
