from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from forcesmolvla.rft.training_checkpoint import (
    G5_CHECKPOINT_MARKERS,
    save_g5_cycle_checkpoint,
    validate_g5_checkpoint,
)
from forcesmolvla.rft.training_cycle import (
    SerializableReplacementSampler,
    SerializableUniqueSampler,
    calql_unclipped_details,
    generator_state_sha256,
    global_gradient_norm,
    gradients_finite,
    optimizer_state_storage_independent,
    parameter_change_matrix,
    polyak_update_verified,
)


def generator(seed):
    return torch.Generator().manual_seed(seed)


def test_named_samplers_unique_independent_and_serializable():
    left = SerializableUniqueSampler("td", tuple(range(20)), generator(1))
    right = SerializableUniqueSampler("calql", tuple(range(20)), generator(2))
    before_right = generator_state_sha256(right.generator)
    first = left.draw(16)
    assert len(first) == len(set(first)) == 16
    assert generator_state_sha256(right.generator) == before_right
    assert left.state_dict()["draws"] == 1 and right.state_dict()["draws"] == 0
    replacement = SerializableReplacementSampler("proposal", 3, generator(3))
    values = replacement.draw(100)
    assert len(values) == 100 and set(values) <= {0, 1, 2}


class TinyTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.online = nn.Parameter(torch.tensor([1.0, 3.0]))
        self.frozen = nn.Parameter(torch.tensor([5.0]), requires_grad=False)
        self.register_buffer("floating", torch.tensor([2.0]))
        self.register_buffer("counter", torch.tensor(4, dtype=torch.int64))
        self._permanent = False

    def train(self, mode=True):
        return super().train(False if self._permanent else mode)


def frozen_target(module):
    target = TinyTarget()
    target.load_state_dict(module.state_dict())
    target._permanent = True
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.eval()
    return target


def test_polyak_is_post_online_formula_frozen_float_preserve_and_nonfloat_copy():
    online = TinyTarget()
    target = frozen_target(online)
    before = target.online.detach().clone()
    floating_before = target.floating.detach().clone()
    online.online.data.add_(torch.tensor([2.0, -1.0]))
    online.counter.add_(1)
    report = polyak_update_verified(online, target, tau=0.005, target_name="target")
    assert torch.equal(target.online, before * 0.995 + online.online * 0.005)
    assert torch.equal(target.floating, floating_before)
    assert torch.equal(target.counter, online.counter)
    assert report["maximum_formula_abs_error"] == 0.0
    assert report["frozen_floating_buffer_changed_count"] == 0
    assert not target.training and all(not parameter.requires_grad for parameter in target.parameters())


def test_calql_sidecar_is_unclipped_and_activation_visible():
    qd = torch.tensor([1.0, -1.0])
    qc = torch.tensor([[0.0] * 6, [2.0] * 6])
    mc = torch.tensor([0.5, 3.0])
    result = calql_unclipped_details(qd, qc, mc, temperature=1.0)
    assert result["difference"].shape == (2,)
    assert result["mc_lower_bound_activation"].shape == (2, 6)
    assert result["mc_lower_bound_activation"].all()


def test_gradient_helpers_and_change_matrix():
    model = nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    assert global_gradient_norm(model.parameters()) > 0
    assert gradients_finite(model.parameters())
    assert parameter_change_matrix({"a": "x", "b": "y"}, {"a": "z", "b": "y"}) == {"a": True, "b": False}


def test_atomic_checkpoint_boundary_and_markers(tmp_path):
    modules = [nn.Linear(2, 2) for _ in range(5)]
    actor, q1, q2, q1_target, q2_target = modules
    for target in (q1_target, q2_target):
        target.eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-5)
    critic_optimizer = torch.optim.Adam([*q1.parameters(), *q2.parameters()], lr=3e-4)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lambda _: 1.0)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lambda _: 1.0)
    counters = {
        "training_cycles": 1,
        "critic_optimizer_updates": 2,
        "actor_optimizer_updates": 1,
        "q1_target_polyak_updates": 2,
        "q2_target_polyak_updates": 2,
        "actor_target_updates": 0,
        "critic_scheduler_steps": 2,
        "actor_scheduler_steps": 1,
    }
    destination = tmp_path / "checkpoint"
    save_g5_cycle_checkpoint(
        destination,
        actor=actor,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        counters=counters,
        sampler_states={"td": {"draws": 2}},
        rng_states={"python": (1, 2)},
        startup_snapshot_bytes={"config.yaml": b"frozen\n"},
        parameter_ownership_manifest={"pass": True},
        trainability_manifest={"pass": True},
        proposal_population_manifest={"pass": True},
    )
    manifest = validate_g5_checkpoint(destination)
    assert all(manifest[name] == value for name, value in G5_CHECKPOINT_MARKERS.items())
    assert manifest["counters"] == counters
    with pytest.raises(FileExistsError):
        save_g5_cycle_checkpoint(
            destination,
            actor=actor, q1=q1, q2=q2, q1_target=q1_target, q2_target=q2_target,
            actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
            actor_scheduler=actor_scheduler, critic_scheduler=critic_scheduler,
            counters=counters, sampler_states={}, rng_states={},
            startup_snapshot_bytes={"x": b"x"}, parameter_ownership_manifest={},
            trainability_manifest={}, proposal_population_manifest={},
        )
