from types import SimpleNamespace

import pytest
import torch
from torch import nn

from forcesmolvla.action_delta import decode_binary_gripper_width
from forcesmolvla.rft.flow_sampling import (
    critic_action_for_q_guidance,
    sample_normalized_action_chunk_with_grad,
)


class FakeFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.last_call = None

    def parameter_generation(self):
        return 7

    def sample_actions_masked(
        self,
        images,
        image_masks,
        tokens,
        language_mask,
        state,
        noise32,
        *,
        action_feature_mask,
        suffix_valid_mask,
        wrench,
        force_context_binding,
    ):
        self.last_call = {
            "noise32": noise32.detach().clone(),
            "feature_mask": action_feature_mask.detach().clone(),
            "suffix_valid": suffix_valid_mask.detach().clone(),
            "binding": force_context_binding,
        }
        value = noise32
        mask = action_feature_mask.to(dtype=value.dtype)
        for _ in range(10):
            value = (value - 0.1 * self.scale * value) * mask
        return value


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            chunk_size=50,
            num_steps=10,
            max_state_dim=32,
            max_action_dim=32,
        )
        self.model = FakeFlowModel()
        self._context_generation = 3

    def prepare_images(self, batch):
        return [batch["camera1"], batch["camera2"]], [
            torch.ones(len(batch["camera1"]), dtype=torch.bool),
            torch.ones(len(batch["camera2"]), dtype=torch.bool),
        ]

    def prepare_state(self, batch):
        result = torch.zeros(len(batch["observation.state"]), 32)
        result[:, :7] = batch["observation.state"]
        return result

    def _prepare_wrench(self, batch, *, device):
        return batch["observation.wrench"].to(device=device, dtype=torch.float32)


def _batch(batch_size=2):
    return {
        "camera1": torch.zeros(batch_size, 3, 8, 8),
        "camera2": torch.zeros(batch_size, 3, 8, 8),
        "observation.state": torch.zeros(batch_size, 7),
        "observation.wrench": torch.zeros(batch_size, 6),
        "observation.language.tokens": torch.ones(batch_size, 48, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(batch_size, 48, dtype=torch.bool),
        "sample_identity": tuple(f"sample-{index}" for index in range(batch_size)),
    }


def test_differentiable_wrapper_uses_native_h50_masks_and_keeps_actor_gradient():
    policy = FakePolicy().eval()
    noise = torch.randn(2, 50, 7)

    action = sample_normalized_action_chunk_with_grad(
        policy, noise7=noise, batch=_batch(), call_id="unit", purpose="actor_guidance"
    )
    action[:, 0, :6].sum().backward()

    call = policy.model.last_call
    assert action.shape == (2, 50, 7)
    assert action.dtype == torch.float32
    assert torch.count_nonzero(call["noise32"][..., 7:]) == 0
    assert torch.all(call["suffix_valid"])
    assert torch.all(call["feature_mask"][..., :7])
    assert not torch.any(call["feature_mask"][..., 7:])
    assert call["binding"].chunk_id == (
        "rft:actor_guidance:unit:0",
        "rft:actor_guidance:unit:1",
    )
    assert call["binding"].sample_id == ("sample-0", "sample-1")
    assert policy.model.scale.grad is not None
    assert torch.isfinite(policy.model.scale.grad)
    assert policy.model.scale.grad != 0


def test_actor_guidance_requires_eval_with_autograd_but_td_allows_no_grad():
    policy = FakePolicy()
    with pytest.raises(RuntimeError, match="RFT_FLOW_SAMPLING_REQUIRES_ACTOR_EVAL_MODE"):
        sample_normalized_action_chunk_with_grad(
            policy, _batch(), torch.zeros(2, 50, 7), call_id="train", purpose="actor_guidance"
        )

    policy.eval()
    with torch.no_grad(), pytest.raises(
        RuntimeError, match="RFT_ACTOR_GUIDANCE_REQUIRES_AUTOGRAD"
    ):
        sample_normalized_action_chunk_with_grad(
            policy, _batch(), torch.zeros(2, 50, 7), call_id="nograd", purpose="actor_guidance"
        )
    with torch.no_grad():
        action = sample_normalized_action_chunk_with_grad(
            policy, _batch(), torch.zeros(2, 50, 7), call_id="td", purpose="td_next"
        )
    assert not action.requires_grad


def test_wrapper_rejects_missing_identity_and_non_7d_noise():
    policy = FakePolicy().eval()
    batch = _batch()
    del batch["sample_identity"]
    with pytest.raises(ValueError, match="sample_identity"):
        sample_normalized_action_chunk_with_grad(
            policy, batch, torch.zeros(2, 50, 7), call_id="missing", purpose="actor_guidance"
        )
    with pytest.raises(ValueError, match="noise7"):
        sample_normalized_action_chunk_with_grad(
            policy, _batch(), torch.zeros(2, 50, 32), call_id="padded", purpose="actor_guidance"
        )


def test_k3_q_action_observes_discrete_gripper_but_only_guides_tcp():
    mean = torch.tensor([0.0] * 6 + [0.028491082421846097])
    std = torch.tensor([1.0] * 6 + [0.04012480845771951])
    chunk = torch.randn(2, 50, 7)
    chunk[..., 6] = (0.0 - mean[6]) / std[6]
    chunk.requires_grad_()
    weights = torch.arange(1, 22, dtype=torch.float32).view(3, 7) / 21

    critic_action = critic_action_for_q_guidance(
        chunk,
        delta_action_mean7=mean,
        delta_action_std7=std,
    )
    closed_q = (critic_action * weights).sum()
    closed_q.backward()

    expected_closed = (torch.tensor(0.0) - mean[6]) / std[6]
    assert critic_action.shape == (2, 3, 7)
    assert torch.all(critic_action[:, :, 6] == expected_closed)
    assert torch.all(chunk.grad[:, :3, :6] != 0)
    assert torch.count_nonzero(chunk.grad[:, :3, 6]) == 0
    assert torch.count_nonzero(chunk.grad[:, 3:]) == 0

    open_chunk = chunk.detach().clone()
    open_chunk[..., 6] = (0.085 - mean[6]) / std[6]
    open_action = critic_action_for_q_guidance(
        open_chunk,
        delta_action_mean7=mean,
        delta_action_std7=std,
    )
    open_q = (open_action * weights).sum()
    assert open_q != closed_q.detach()

    physical = open_chunk[:, :3].numpy() * std.numpy() + mean.numpy()
    public = decode_binary_gripper_width(physical)
    public_normalized = (
        torch.from_numpy(public[..., 6]).to(torch.float32) - mean[6]
    ) / std[6]
    torch.testing.assert_close(
        open_action[..., 6], public_normalized, rtol=0.0, atol=0.0
    )


def test_k3_q_action_rejects_non_h50_chunk_and_invalid_gripper_candidate():
    mean = torch.zeros(7)
    std = torch.ones(7)
    with pytest.raises(ValueError, match=r"\[B,50,7\]"):
        critic_action_for_q_guidance(
            torch.zeros(2, 7),
            delta_action_mean7=mean,
            delta_action_std7=std,
        )
    invalid = torch.zeros(1, 50, 7)
    invalid[..., 6] = 0.2
    with pytest.raises(ValueError, match="frozen"):
        critic_action_for_q_guidance(
            invalid,
            delta_action_mean7=mean,
            delta_action_std7=std,
        )


def test_k3_q_action_rejects_removed_partial_action_mask_api():
    with pytest.raises(TypeError, match="positional"):
        critic_action_for_q_guidance(
            torch.zeros(1, 50, 7),
            torch.ones(1, 3, dtype=torch.bool),
            delta_action_mean7=torch.zeros(7),
            delta_action_std7=torch.ones(7),
        )
