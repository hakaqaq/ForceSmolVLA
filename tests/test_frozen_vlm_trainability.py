from __future__ import annotations

import torch
from torch import nn

from forcesmolvla.rft.frozen_vlm_trainability import (
    apply_frozen_vlm_trainability,
    build_frozen_vlm_actor_optimizer,
    compute_min_twin_q_actor_loss,
)
from forcesmolvla.rft.losses import CriticObservation


class _VLMWithExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm = nn.Linear(2, 2)
        self.lm_expert = nn.Linear(2, 2)


class _ForceBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.force_mlp = nn.Linear(2, 2)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm_with_expert = _VLMWithExpert()
        self.state_proj = nn.Linear(2, 2)
        self.force_branch = _ForceBranch()
        self.force_adapter = nn.Linear(2, 2)
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.action_time_mlp_in = nn.Linear(2, 2)
        self.action_time_mlp_out = nn.Linear(2, 2)


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Model()


class _ActionSensitiveQ(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, _c1, _c2, _task, _state, _wrench, action, mask):
        return self.scale * (action * mask[..., None]).sum(dim=(1, 2))


def test_trainability_freezes_prefix_and_optimizer_excludes_it() -> None:
    policy = _Policy()
    manifest = apply_frozen_vlm_trainability(policy)
    policy.train(True)
    assert manifest.frozen_parameter_count > 0
    assert not policy.model.vlm_with_expert.vlm.training
    assert not policy.model.state_proj.training
    assert all(
        not parameter.requires_grad
        for name, parameter in policy.named_parameters()
        if name.startswith(("model.vlm_with_expert.vlm.", "model.state_proj."))
    )
    optimizer, _scheduler, ownership = build_frozen_vlm_actor_optimizer(policy)
    owned = {id(value) for group in optimizer.param_groups for value in group["params"]}
    assert ownership["frozen_parameter_in_optimizer"] == 0
    assert not owned.intersection(
        id(parameter)
        for name, parameter in policy.named_parameters()
        if name.startswith(("model.vlm_with_expert.vlm.", "model.state_proj."))
    )


def test_min_twin_q_actor_loss_stops_gripper_gradient() -> None:
    batch = 2
    observation = CriticObservation(
        camera1=torch.zeros(batch, 1), camera2=torch.zeros(batch, 1),
        task_feature=torch.zeros(batch, 1), normalized_state7=torch.zeros(batch, 7),
        normalized_wrench6=torch.zeros(batch, 6),
    )
    action = torch.randn(batch, 50, 7, requires_grad=True)
    loss, _q1, _q2, critic_action = compute_min_twin_q_actor_loss(
        q1=_ActionSensitiveQ(1.0), q2=_ActionSensitiveQ(2.0),
        observation=observation, normalized_flow_action_chunk7=action,
        delta_action_mean7=torch.zeros(7), delta_action_std7=torch.ones(7),
    )
    endpoints = torch.tensor([0.0, 0.085], dtype=critic_action.dtype)
    assert torch.all((critic_action[..., 6, None].detach() == endpoints).any(dim=-1))
    loss.backward()
    assert action.grad[:, :3, :6].abs().sum() > 0
    assert torch.equal(action.grad[:, :3, 6], torch.zeros_like(action.grad[:, :3, 6]))
