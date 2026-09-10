from __future__ import annotations

from types import SimpleNamespace

import torch

from forceprior.rft.online.controller_acceptance import CandidateGuardBatch
from forceprior.rft.online.training_losses import residual_actor_loss


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
    valid = torch.ones(2, dtype=torch.bool)
    batch = SimpleNamespace(
        state7=torch.zeros(2, 7),
        wrench6=torch.zeros(2, 6),
        wrench_delta6=torch.zeros(2, 6),
        base_action6=torch.zeros(2, 6),
        base_gripper=torch.zeros(2, 1),
        behavior_proposal6=torch.zeros(2, 6),
        candidate_guard=CandidateGuardBatch(
            valid=valid,
            decision_state7=torch.zeros(2, 7),
            upper_position3=torch.zeros(2, 3),
            upper_quaternion4=torch.tensor(
                [[0.0, 0.0, 0.0, 1.0]] * 2
            ),
            policy_workspace_min3=torch.full((2, 3), -1.0),
            policy_workspace_max3=torch.full((2, 3), 1.0),
            policy_orientation_min3=torch.full((2, 3), -3.0),
            policy_orientation_max3=torch.full((2, 3), 3.0),
            policy_gimbal_margin_rad=torch.full((2, 1), 0.01),
            policy_gripper_width_m=torch.zeros(2, 1),
            policy_gripper_min_m=torch.zeros(2, 1),
            policy_gripper_max_m=torch.ones(2, 1),
            policy_continuity_max_xyz_m=torch.full((2, 1), 0.08),
            policy_continuity_max_rotation_rad=torch.full(
                (2, 1), 0.4363323129985824
            ),
            policy_continuity_max_gripper_delta_m=torch.ones(2, 1),
            delta_action_mean6=torch.zeros(2, 6),
            delta_action_std6=torch.ones(2, 6),
        ),
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
