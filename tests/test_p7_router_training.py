from types import SimpleNamespace

import pytest
import torch

from forcesmolvla.configuration_forcesmolvla import FORCE_TOKEN_MOE
from forcesmolvla.force_token import RouterState
from forcesmolvla.router_training import (
    MoEMicrobatch,
    SerializableUniformSampler,
    build_p7_optimizer_and_scheduler,
    collect_pass_a_statistics,
    derive_optimizer_updates,
    microbatch_two_pass_terms,
    single_pass_optimizer_update,
    two_pass_optimizer_update,
)
from lerobot.optim import CosineDecayWithWarmupSchedulerConfig


def _router_state(logits: torch.Tensor, valid: torch.Tensor) -> RouterState:
    probabilities = torch.softmax(logits, dim=-1) * valid.unsqueeze(-1)
    routes = torch.argmax(probabilities, dim=-1).masked_fill(~valid, -1)
    return RouterState(logits * valid.unsqueeze(-1), probabilities, routes, valid)


def test_two_pass_terms_sum_to_exact_global_definition_and_gradients():
    torch.manual_seed(42)
    valid_masks = [
        torch.tensor([[1, 1, 0]], dtype=torch.bool),
        torch.tensor([[1, 0, 1]], dtype=torch.bool),
    ]
    pass_b_logits = [torch.randn(1, 3, 4, requires_grad=True) for _ in range(2)]
    pass_b_states = [_router_state(value, valid) for value, valid in zip(pass_b_logits, valid_masks)]
    pass_a_states = [
        _router_state(value.detach(), valid)
        for value, valid in zip(pass_b_logits, valid_masks)
    ]
    flow_masks = [valid.unsqueeze(-1).expand(-1, -1, 7) for valid in valid_masks]
    statistics = collect_pass_a_statistics(pass_a_states, flow_masks)

    flow_losses = []
    terms = []
    for index, valid in enumerate(valid_masks):
        raw = torch.randn(1, 3, 7, requires_grad=True)
        masked = raw.square() * valid.unsqueeze(-1)
        flow_losses.append(masked)
        terms.append(microbatch_two_pass_terms(masked, pass_b_states[index], statistics))

    total = sum(item.total for item in terms)
    all_probabilities = torch.cat(
        [state.probabilities_fp32[state.valid_mask] for state in pass_b_states], dim=0
    )
    all_logits = torch.cat([state.logits_fp32[state.valid_mask] for state in pass_b_states], dim=0)
    expected_flow = sum(value.sum() for value in flow_losses) / statistics.valid_flow_features
    expected_balance = 4 * torch.sum(statistics.rbar * all_probabilities.sum(dim=0) / 4)
    expected_z = torch.logsumexp(all_logits, dim=-1).square().sum() / 4
    expected_total = expected_flow + 0.01 * expected_balance + 0.001 * expected_z
    torch.testing.assert_close(total, expected_total, rtol=0, atol=0)

    gradients = torch.autograd.grad(total, pass_b_logits, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_total, pass_b_logits)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert statistics.valid_router_tokens == 4
    assert statistics.valid_flow_features == 28
    assert sum(statistics.route_counts.tolist()) == 4


def test_zero_router_and_flow_denominators_return_graph_connected_zero():
    valid = torch.zeros(1, 2, dtype=torch.bool)
    logits = torch.randn(1, 2, 4, requires_grad=True)
    pass_b = _router_state(logits, valid)
    pass_a = _router_state(logits.detach(), valid)
    statistics = collect_pass_a_statistics(
        [pass_a], [torch.zeros(1, 2, 7, dtype=torch.bool)]
    )
    flow = torch.randn(1, 2, 7, requires_grad=True)
    terms = microbatch_two_pass_terms(flow * 0, pass_b, statistics)
    assert terms.flow.item() == 0
    assert terms.balance.item() == 0
    assert terms.z.item() == 0
    terms.total.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0
    assert flow.grad is not None and torch.count_nonzero(flow.grad) == 0


class _FakeTwoPassPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(force_variant=FORCE_TOKEN_MOE)
        self.router_logits = torch.nn.Parameter(torch.tensor([0.3, -0.1, 0.2, -0.4]))
        self.flow_scale = torch.nn.Parameter(torch.tensor(0.5))
        self.pass_a_grad_enabled = []
        self.pass_b_calls = 0
        self.single_pass_calls = 0

    def _state(self, batch):
        valid = torch.ones(batch["action"].shape[0], 3, dtype=torch.bool)
        logits = self.router_logits.view(1, 1, 4).expand(valid.shape[0], 3, 4)
        return _router_state(logits, valid)

    def router_pass_a(self, batch):
        self.pass_a_grad_enabled.append(torch.is_grad_enabled())
        return self._state(batch)

    def forward_training_terms(self, batch, *, noise, time):
        self.pass_b_calls += 1
        valid = batch["action_valid_mask"]
        feature = valid.unsqueeze(-1) & (torch.arange(32).view(1, 1, 32) < 7)
        losses = self.flow_scale.square().expand_as(feature) * feature
        return losses, feature, self._state(batch)

    def forward_single_pass_training_terms(self, batch, *, noise, time):
        self.single_pass_calls += 1
        valid = batch["action_valid_mask"]
        feature = valid.unsqueeze(-1) & (torch.arange(32).view(1, 1, 32) < 7)
        losses = self.flow_scale.square().expand_as(feature) * feature
        return losses, feature, self._state(batch)


def test_two_pass_window_uses_eight_microbatches_and_exactly_one_update():
    policy = _FakeTwoPassPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    microbatches = []
    for index in range(8):
        batch = {
            "action": torch.zeros(1, 2, 7),
            "action_valid_mask": torch.tensor([[1, index % 3 != 0]], dtype=torch.bool),
        }
        microbatches.append(
            MoEMicrobatch(
                batch=batch,
                noise7=torch.zeros(1, 2, 7, dtype=torch.float32),
                time=torch.tensor([0.5], dtype=torch.float32),
                identity=f"microbatch-{index}",
            )
        )
    before = {name: value.detach().clone() for name, value in policy.named_parameters()}
    report = two_pass_optimizer_update(
        policy, microbatches, optimizer, oracle_mode=True, scheduler=scheduler
    )
    assert policy.pass_a_grad_enabled == [False] * 8
    assert policy.pass_b_calls == 8
    assert report["microbatch_count"] == 8
    assert report["optimizer_steps"] == 1
    assert report["scheduler_steps"] == 1
    assert report["valid_router_tokens"] == 24
    assert report["valid_flow_features"] == 91
    assert report["max_router_probability_replay_error"] == 0
    assert any(not torch.equal(before[name], value) for name, value in policy.named_parameters())


def test_two_pass_rejects_wrong_window_size_and_duplicate_identity():
    policy = _FakeTwoPassPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    batch = {"action": torch.zeros(1, 2, 7), "action_valid_mask": torch.ones(1, 2, dtype=torch.bool)}
    microbatch = MoEMicrobatch(batch, torch.zeros(1, 2, 7), torch.ones(1), "same")
    with pytest.raises(RuntimeError, match="ACCEPTANCE_ORACLE_ONLY"):
        two_pass_optimizer_update(policy, [microbatch] * 8, optimizer)
    with pytest.raises(ValueError, match="requires 8"):
        two_pass_optimizer_update(policy, [microbatch], optimizer, oracle_mode=True)
    with pytest.raises(ValueError, match="unique"):
        two_pass_optimizer_update(policy, [microbatch] * 8, optimizer, oracle_mode=True)


def test_optimizer_updates_are_derived_from_the_primary_sample_budget():
    assert derive_optimizer_updates(80_000, 4) == 20_000
    with pytest.raises(ValueError, match="divisible"):
        derive_optimizer_updates(10, 4)


def test_single_pass_update_uses_one_forward_backward_and_optimizer_step():
    policy = _FakeTwoPassPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    batch = {
        "action": torch.zeros(4, 2, 7),
        "action_valid_mask": torch.ones(4, 2, dtype=torch.bool),
    }
    microbatch = MoEMicrobatch(
        batch,
        torch.zeros(4, 2, 7),
        torch.full((4,), 0.5),
        "single-batch",
    )
    before = {name: value.detach().clone() for name, value in policy.named_parameters()}
    report = single_pass_optimizer_update(
        policy, microbatch, optimizer, scheduler=scheduler
    )
    assert policy.single_pass_calls == 1
    assert policy.pass_a_grad_enabled == []
    assert policy.pass_b_calls == 0
    assert report["training_update_algorithm"] == "single_pass_batch_local"
    assert report["microbatch_count"] == 1
    assert report["optimizer_steps"] == 1
    assert report["scheduler_steps"] == 1
    assert report["valid_router_tokens"] == 12
    assert report["valid_flow_features"] == 56
    assert any(not torch.equal(before[name], value) for name, value in policy.named_parameters())


def test_uniform_sampler_restores_rng_and_cursor_exactly():
    sampler = SerializableUniformSampler([2, 5, 9, 11], seed=42)
    assert sampler.draw(3) == [2, 2, 9]
    state = sampler.state_dict()
    expected = sampler.draw(12)
    restored = SerializableUniformSampler([2, 5, 9, 11], seed=42)
    restored.load_state_dict(state)
    assert restored.cursor == 3
    assert restored.draw(12) == expected
    with pytest.raises(RuntimeError, match="BINDING"):
        SerializableUniformSampler([2, 5], seed=42).load_state_dict(state)


class _OptimizerGroupingFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.bias = torch.nn.Parameter(torch.ones(2))
        self.alpha = torch.nn.Parameter(torch.ones(()))
        self.learned_action_slot = torch.nn.Parameter(torch.ones(2, 2))
        self.norm = torch.nn.LayerNorm(2)
        self.embedding = torch.nn.Embedding(3, 2)
        self.config = SimpleNamespace(
            get_scheduler_preset=lambda: CosineDecayWithWarmupSchedulerConfig(
                peak_lr=1e-4,
                decay_lr=2.5e-6,
                num_warmup_steps=1000,
                num_decay_steps=20000,
            )
        )


def test_p7_optimizer_groups_and_frozen_scheduler_semantics():
    policy = _OptimizerGroupingFixture()
    optimizer, scheduler, manifest = build_p7_optimizer_and_scheduler(policy)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [1e-10, 0.0]
    assert manifest["decay_parameter_count"] == policy.weight.numel()
    assert manifest["no_decay_parameter_count"] == sum(
        parameter.numel() for name, parameter in policy.named_parameters() if name != "weight"
    )
    assert manifest["trainable_tensor_count"] == len(list(policy.parameters()))
    assert manifest["optimizer_tensor_count"] == len(list(policy.parameters()))
    assert manifest["each_trainable_parameter_exactly_once"] is True
    action_slot = dict(policy.named_parameters())["learned_action_slot"]
    assert any(
        action_slot is parameter
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.0
        for parameter in group["params"]
    )
    schedule = scheduler.lr_lambdas[0]
    assert schedule(0) == 1 / 1001
    cosine = 0.5 * (1 + __import__("math").cos(__import__("math").pi * 1000 / 20000))
    assert schedule(1000) == pytest.approx((1 - 0.025) * cosine + 0.025)
    assert schedule(20000) == pytest.approx(0.025)
