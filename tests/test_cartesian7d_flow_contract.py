from types import MethodType, SimpleNamespace

import pytest
import torch

from forcesmolvla.configuration_forcesmolvla import CAMERA1, CAMERA2
from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


class _FakeMaskedFlow(torch.nn.Module):
    def __init__(self, scales=None):
        super().__init__()
        self.last = None
        self.scales = scales

    def forward(self, images, image_masks, tokens, language_masks, state, actions, noise, time, **kwargs):
        self.last = {"state": state, "actions": actions, "noise": noise, **kwargs}
        # Masked flow returns one unit error for every valid feature token.
        result = kwargs["action_feature_mask"].to(dtype=torch.float32)
        if self.scales is not None:
            result = result * self.scales.view(-1, 1, 1)
        return result


def _fake_policy(scales=None):
    policy = object.__new__(ForceSmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(max_action_dim=32)
    policy.model = _FakeMaskedFlow(scales=scales)

    def prepare_images(self, batch):
        b = batch[CAMERA1].shape[0]
        return [batch[CAMERA1], batch[CAMERA2]], [torch.ones(b, dtype=torch.bool)] * 2

    def prepare_state(self, batch):
        state = batch["observation.state"]
        return torch.nn.functional.pad(state, (0, 25))

    policy.prepare_images = MethodType(prepare_images, policy)
    policy.prepare_state = MethodType(prepare_state, policy)
    return policy


def _batch():
    return {
        CAMERA1: torch.zeros(2, 3, 8, 8),
        CAMERA2: torch.zeros(2, 3, 8, 8),
        "observation.state": torch.ones(2, 7),
        ACTION: torch.ones(2, 4, 7),
        "action_valid_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 48, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 48, dtype=torch.bool),
    }


def test_cartesian7d_global_feature_mask_and_per_sample_reduction():
    policy = _fake_policy()
    loss, report = policy.forward(_batch())
    assert loss.item() == 1.0
    assert report["valid_feature_tokens"] == (2 + 3) * 7
    assert torch.count_nonzero(policy.model.last["state"][:, 7:]) == 0
    assert torch.count_nonzero(policy.model.last["actions"][:, :, 7:]) == 0
    feature = policy.model.last["action_feature_mask"]
    assert not feature[0, 2:].any()
    assert not feature[..., 7:].any()


def test_mean_reduction_uses_global_valid_feature_denominator():
    policy = _fake_policy(scales=torch.tensor([1.0, 3.0]))
    loss, _ = policy.forward(_batch())
    # Sample 0 has 14 valid tokens, sample 1 has 21; never average chunk means.
    assert loss.item() == pytest.approx((14 * 1 + 21 * 3) / 35)
    per_sample, _ = policy.forward(_batch(), reduction="none")
    torch.testing.assert_close(per_sample, torch.tensor([1.0, 3.0]))


def test_noise_contract_accepts_7d_and_rejects_nonzero_padding():
    policy = _fake_policy()
    batch = _batch()
    policy.forward(batch, noise=torch.zeros(2, 4, 7))
    assert policy.model.last["noise"].shape == (2, 4, 32)
    bad = torch.zeros(2, 4, 32)
    bad[..., 9] = 1
    with pytest.raises(ValueError, match="zero-padded"):
        policy.forward(batch, noise=bad)


def test_action_valid_mask_is_mandatory():
    policy = _fake_policy()
    batch = _batch()
    del batch["action_valid_mask"]
    with pytest.raises(KeyError, match="action_valid_mask"):
        policy.forward(batch)
