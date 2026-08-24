from types import SimpleNamespace

import pytest
import torch

from forcesmolvla.checkpoint import validate_resume_training_stage
from forcesmolvla.configuration_forcesmolvla import (
    OFFLINE_FULL_FINETUNE,
    ONLINE_HIL_VLM_FROZEN,
)
from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy


class _FakeVLMWithExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = torch.nn.Linear(2, 2)
        self.lm_expert = torch.nn.Linear(2, 2)


class _FakeFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_with_expert = _FakeVLMWithExpert()
        self.projection = torch.nn.Linear(2, 2)


def _policy(stage):
    policy = object.__new__(ForceSmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(training_stage=stage)
    policy.model = _FakeFlow()
    return policy


def test_offline_stage_unfreezes_every_parameter_and_trains_vlm():
    policy = _policy(OFFLINE_FULL_FINETUNE)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.apply_training_stage()
    policy.train()
    assert all(parameter.requires_grad for parameter in policy.parameters())
    assert policy.model.vlm_with_expert.vlm.training


def test_online_hil_stage_freezes_only_vlm_and_keeps_it_eval():
    policy = _policy(ONLINE_HIL_VLM_FROZEN)
    policy.apply_training_stage()
    policy.train()
    assert not any(parameter.requires_grad for parameter in policy.model.vlm_with_expert.vlm.parameters())
    assert all(parameter.requires_grad for parameter in policy.model.vlm_with_expert.lm_expert.parameters())
    assert not policy.model.vlm_with_expert.vlm.training
    assert policy.model.vlm_with_expert.lm_expert.training


def test_optimizer_state_cannot_cross_training_stage():
    with pytest.raises(RuntimeError, match="rebuild optimizer"):
        validate_resume_training_stage(
            checkpoint_stage=OFFLINE_FULL_FINETUNE,
            runtime_stage=ONLINE_HIL_VLM_FROZEN,
            restore_optimizer_state=True,
        )
    validate_resume_training_stage(
        checkpoint_stage=OFFLINE_FULL_FINETUNE,
        runtime_stage=ONLINE_HIL_VLM_FROZEN,
        restore_optimizer_state=False,
    )
