from __future__ import annotations

import torch
from torch import nn

from forcesmolvla.rft.online.training_contracts import (
    apply_online_trainability,
    load_online_contracts,
    validate_online_contracts,
)


class DummyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.vlm_with_expert = nn.Module()
        self.model.vlm_with_expert.vlm = nn.Linear(2, 2)
        self.model.vlm_with_expert.lm_expert = nn.Linear(2, 2)
        self.model.state_proj = nn.Linear(2, 2)
        self.model.force_branch = nn.Linear(2, 2)
        self.model.force_adapter = nn.Linear(2, 2)
        self.model.action_in_proj = nn.Linear(2, 2)
        self.model.action_out_proj = nn.Linear(2, 2)
        self.model.action_time_mlp_in = nn.Linear(2, 2)
        self.model.action_time_mlp_out = nn.Linear(2, 2)


def test_g1_contracts_are_cross_consistent_and_locked() -> None:
    result = validate_online_contracts()
    assert result == {
        "bootstrap_parent_binding": "PENDING",
        "bootstrap_optimizer_rebuilt": "NOT_RUN",
        "temporal_parity": "BLOCKED",
        "critic_ready": False,
        "actor_q_guidance_enabled": False,
        "robot_execution_authorized": False,
    }
    values = load_online_contracts()
    assert values["transition"]["temporal_parity"]["recorded_live_fixture"] is None
    assert values["online_hil"]["readiness"]["recorded_loopback"] == "NOT_RUN"


def test_online_trainability_reuses_frozen_vlm_contract() -> None:
    policy = DummyPolicy().train()
    manifest = apply_online_trainability(policy)
    named = dict(policy.named_parameters())
    assert manifest.frozen_parameter_count > 0
    assert manifest.trainable_actor_parameter_count > 0
    assert all(not named[name].requires_grad for name in manifest.frozen_names)
    assert all(named[name].requires_grad for name in manifest.trainable_names)
    assert not policy.model.vlm_with_expert.vlm.training
    assert not policy.model.state_proj.training
    policy.train()
    assert not policy.model.vlm_with_expert.vlm.training
    assert not policy.model.state_proj.training
    assert all(parameter.grad is None for parameter in policy.parameters())
