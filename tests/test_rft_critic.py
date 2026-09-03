from pathlib import Path

import pytest
import torch
from torch import nn

from forcesmolvla.rft.critic import (
    ACTION_SLOTS,
    PROJECT_ROOT,
    build_twin_q,
    frozen_task_feature,
    load_authorized_critic_train_transitions,
    modules_storage_independent,
    polyak_blend_state,
    state_exact,
)


SAFE = PROJECT_ROOT / "assets/reward_classifier/resnet10_parameters.npz"
SAFE_MANIFEST = PROJECT_ROOT / "assets/reward_classifier/resnet10_manifest.json"


@pytest.fixture(scope="module")
def topology():
    return build_twin_q(SAFE, SAFE_MANIFEST, seed=0)


def inputs(batch=1, mask=(True, True, True)):
    generator = torch.Generator().manual_seed(7)
    camera1 = torch.randint(0, 256, (batch, 3, 128, 128), dtype=torch.uint8, generator=generator)
    camera2 = torch.randint(0, 256, (batch, 3, 128, 128), dtype=torch.uint8, generator=generator)
    task = torch.from_numpy(frozen_task_feature()).repeat(batch, 1)
    state = torch.randn(batch, 7, generator=generator)
    wrench = torch.randn(batch, 6, generator=generator)
    action = torch.randn(batch, ACTION_SLOTS, 7, generator=generator)
    action_mask = torch.tensor(mask, dtype=torch.bool).repeat(batch, 1)
    return [camera1, camera2, task, state, wrench, action, action_mask]


def test_twin_q_topology_targets_mapping_and_polyak(topology):
    q1, q2, q1_target, q2_target, conversion = topology
    assert conversion["mapped_backbone_key_count"] == 36
    assert conversion["mapped_shape_coverage"] == 1.0
    assert conversion["all_tensor_roundtrip_parity"]
    assert conversion["random_backbone_parameter_count"] == 0
    assert modules_storage_independent(q1, q2)
    assert modules_storage_independent(q1, q1_target)
    assert modules_storage_independent(q2, q2_target)
    assert state_exact(q1, q1_target) and state_exact(q2, q2_target)
    assert not q1_target.training and not q2_target.training
    assert all(not parameter.requires_grad for target in (q1_target, q2_target) for parameter in target.parameters())
    q1_target.train(True)
    assert not q1_target.training
    assert not any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)) for module in q1.modules())
    assert q1.q_output.weight.abs().max() <= 1e-3 and torch.count_nonzero(q1.q_output.weight)

    online = {"x": torch.tensor([1.0, 3.0])}
    target = {"x": torch.tensor([5.0, 7.0])}
    for tau in (0.0, 0.005, 1.0):
        result = polyak_blend_state(online, target, tau)["x"]
        expected = target["x"] if tau == 0 else online["x"] if tau == 1 else (1 - tau) * target["x"] + tau * online["x"]
        assert torch.equal(result, expected)
        assert torch.equal(online["x"], torch.tensor([1.0, 3.0]))
        assert torch.equal(target["x"], torch.tensor([5.0, 7.0]))


def test_mask_contract_sensitivity_and_gradients(topology):
    q1 = topology[0].eval()
    base = inputs(mask=(True, False, False))
    output = q1(*base)
    assert output.shape == (1,) and output.dtype == torch.float32 and torch.isfinite(output).all()

    invalid_changed = [value.clone() for value in base]
    invalid_changed[5][:, 1:, :] = torch.randn_like(invalid_changed[5][:, 1:, :]) * 1000
    assert torch.equal(output, q1(*invalid_changed))

    for index in (0, 1, 3, 4):
        changed = [value.clone() for value in base]
        if index < 2:
            changed[index][:, :, :16, :16] = 255 - changed[index][:, :, :16, :16]
        else:
            changed[index][:, 0] += 1.0
        assert not torch.equal(output, q1(*changed))

    valid_action_changed = [value.clone() for value in base]
    valid_action_changed[5][:, 0, 0] += 1.0
    assert not torch.equal(output, q1(*valid_action_changed))
    gripper_changed = [value.clone() for value in base]
    gripper_changed[5][:, 0, 6] += 1.0
    assert not torch.equal(output, q1(*gripper_changed))

    for mask in ((True, False, False), (True, True, False), (True, True, True)):
        values = inputs(mask=mask)
        values[5].requires_grad_(True)
        q = q1(*values)
        q.sum().backward()
        valid = values[6][..., None].expand_as(values[5])
        assert torch.all(values[5].grad[valid] != 0)
        assert torch.all(values[5].grad[~valid] == 0)
        q1.zero_grad(set_to_none=True)


@pytest.mark.parametrize("mask", [(False, False, False), (True, False, True)])
def test_invalid_masks_fail_closed(topology, mask):
    with pytest.raises(ValueError):
        topology[0](*inputs(mask=mask))


def test_wrong_shapes_and_non_detector_root_rejected_before_open(topology, tmp_path):
    values = inputs()
    values[5] = torch.zeros(1, 50, 7)
    with pytest.raises(ValueError, match="ACTION_K7_SHAPE"):
        topology[0](*values)
    with pytest.raises(RuntimeError, match="BEFORE_OPEN"):
        load_authorized_critic_train_transitions(tmp_path / "does-not-exist-manual-reward")
