"""Small image-free Twin-Q used by online ForceRFT."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch
from torch import Tensor, nn


TCP_DIM = 6
CRITIC_CONTEXT_DIM = 26
CRITIC_INPUT_DIM = 32
CRITIC_INPUT_SPEC = (
    "state7+wrench6+wrench_delta6+base_tcp6+base_gripper1+"
    "policy_residual_proposal6=32"
)
CRITIC_CONTEXT_ORDER = (
    "state7",
    "wrench6",
    "wrench_delta6",
    "base_tcp6",
    "base_gripper1",
)
CRITIC_ACTION_REPRESENTATION = "normalized_policy_residual_proposal6"
CRITIC_TD_SOURCE_MODE = "policy_only"
CRITIC_CANDIDATE_FEASIBILITY = "deployed_policy_single_action_guard"


def require_critic_input_config(config: Mapping[str, Any]) -> None:
    if (
        int(config.get("input_dim", -1)) != CRITIC_INPUT_DIM
        or tuple(config.get("context_order", ())) != CRITIC_CONTEXT_ORDER
        or config.get("action_representation")
        != CRITIC_ACTION_REPRESENTATION
        or config.get("td_source_mode") != CRITIC_TD_SOURCE_MODE
        or config.get("candidate_feasibility")
        != CRITIC_CANDIDATE_FEASIBILITY
    ):
        raise ValueError("FORCERFT_CRITIC_INPUT_SPEC_MISMATCH")
RESIDUAL_ACTION_OFFSET = CRITIC_CONTEXT_DIM
RESIDUAL_ACTION_WIDTH = TCP_DIM


def _float_tensor(value: Tensor, shape: tuple[int, ...], name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or tuple(value.shape) != shape
    ):
        raise ValueError(f"FORCERFT_CRITIC_{name}_SHAPE_OR_DTYPE_INVALID")
    value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError(f"FORCERFT_CRITIC_{name}_NONFINITE")
    return value


class ResidualQHead(nn.Module):
    """Proposal value conditioned on the compact decision context."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("FORCERFT_CRITIC_HIDDEN_DIM_INVALID")
        self.layers = nn.Sequential(
            nn.Linear(CRITIC_INPUT_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        # A random Critic must not inject an arbitrary action gradient into the
        # initially-zero residual policy. These columns learn normally afterward.
        with torch.no_grad():
            self.layers[0].weight[
                :, RESIDUAL_ACTION_OFFSET : RESIDUAL_ACTION_OFFSET + RESIDUAL_ACTION_WIDTH
            ].zero_()

    def forward(
        self,
        normalized_state7: Tensor,
        normalized_wrench6: Tensor,
        normalized_wrench_delta6: Tensor,
        base_action6: Tensor,
        base_gripper: Tensor,
        residual_proposal6: Tensor,
    ) -> Tensor:
        batch = int(normalized_state7.shape[0])
        if batch < 1:
            raise ValueError("FORCERFT_CRITIC_EMPTY_BATCH")
        state = _float_tensor(normalized_state7, (batch, 7), "STATE7")
        wrench = _float_tensor(normalized_wrench6, (batch, 6), "WRENCH6")
        wrench_delta = _float_tensor(
            normalized_wrench_delta6, (batch, 6), "WRENCH_DELTA6"
        )
        base = _float_tensor(base_action6, (batch, TCP_DIM), "BASE_ACTION")
        gripper = _float_tensor(base_gripper, (batch, 1), "BASE_GRIPPER")
        proposal = _float_tensor(
            residual_proposal6, (batch, TCP_DIM), "RESIDUAL_PROPOSAL"
        )
        features = torch.cat(
            (
                state,
                wrench,
                wrench_delta,
                base,
                gripper,
                proposal,
            ),
            dim=1,
        )
        if features.shape[1] != CRITIC_INPUT_DIM:
            raise RuntimeError("FORCERFT_CRITIC_INPUT_DIMENSION_DRIFT")
        output = self.layers(features).squeeze(-1).float()
        if output.shape != (batch,) or not torch.isfinite(output).all():
            raise RuntimeError("FORCERFT_CRITIC_OUTPUT_INVALID")
        return output


def build_twin_q(
    *, hidden_dim: int = 256, seed: int = 0
) -> tuple[ResidualQHead, ResidualQHead, ResidualQHead, ResidualQHead]:
    """Create independent online heads and exact frozen target copies."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        q1 = ResidualQHead(hidden_dim)
        torch.manual_seed(seed + 1)
        q2 = ResidualQHead(hidden_dim)
    q1_target, q2_target = deepcopy(q1), deepcopy(q2)
    for target in (q1_target, q2_target):
        target.eval()
        target.requires_grad_(False)
    return q1, q2, q1_target, q2_target


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    if not 0.0 <= tau <= 1.0:
        raise ValueError("FORCERFT_POLYAK_TAU_INVALID")
    source_parameters = tuple(source.parameters())
    target_parameters = tuple(target.parameters())
    if len(source_parameters) != len(target_parameters):
        raise ValueError("FORCERFT_POLYAK_MODULE_MISMATCH")
    for source_value, target_value in zip(
        source_parameters, target_parameters, strict=True
    ):
        if source_value.shape != target_value.shape:
            raise ValueError("FORCERFT_POLYAK_MODULE_MISMATCH")
        target_value.mul_(1.0 - tau).add_(source_value, alpha=tau)


def state_exact(left: nn.Module, right: nn.Module) -> bool:
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)


def modules_storage_independent(left: nn.Module, right: nn.Module) -> bool:
    left_values = [*left.parameters(), *left.buffers()]
    right_values = [*right.parameters(), *right.buffers()]
    return not (
        {value.untyped_storage().data_ptr() for value in left_values}
        & {value.untyped_storage().data_ptr() for value in right_values}
    )
