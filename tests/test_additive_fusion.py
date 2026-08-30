from pathlib import Path
import math

import torch

from forcesmolvla.configuration_forcesmolvla import (
    FORCE_TOKEN_MOE,
    FORCE_TOKEN_MOE_ADDITIVE,
    load_force_config,
)
from forcesmolvla.force_token import ForceActionAdapter, ForceContext, module_state_sha256


ROOT = Path(__file__).parents[1]


def _adapter(mode: str) -> ForceActionAdapter:
    torch.manual_seed(42)
    return ForceActionAdapter(8, 3, query_mode=mode)


def _inputs():
    suffix = torch.randn(2, 3, 8)
    noisy = torch.randn(2, 3, 32)
    time = torch.tensor([0.2, 0.8])
    valid = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    context = ForceContext(
        z_action_fp32=torch.randn(2, 177, 8),
        fused_valid_mask=torch.ones(2, 177, dtype=torch.bool),
    )
    feature = valid.unsqueeze(-1).expand(-1, -1, 8)
    action_out = torch.nn.Linear(8, 8)
    return suffix, noisy, time, valid, context, feature, action_out


def test_additive_and_main_have_identical_parameter_names_shapes_counts_and_init():
    main = _adapter("action_query")
    additive = _adapter("additive")
    assert list(main.state_dict()) == list(additive.state_dict())
    assert sum(value.numel() for value in main.parameters()) == sum(
        value.numel() for value in additive.parameters()
    )
    for name, value in main.state_dict().items():
        assert value.shape == additive.state_dict()[name].shape
        assert torch.equal(value, additive.state_dict()[name])
    assert module_state_sha256(main) == module_state_sha256(additive)
    assert main.alpha.item() == __import__("pytest").approx(math.atanh(1e-3))
    assert torch.tanh(main.alpha).item() == __import__("pytest").approx(1e-3)


def test_zero_initialized_wout_makes_native_output_exactly_equal_at_step_zero():
    main = _adapter("action_query")
    additive = _adapter("additive")
    inputs = _inputs()
    main_output = main.velocity(*inputs[:3], inputs[4], suffix_valid_mask=inputs[3], action_feature_mask=inputs[5], action_out_proj=inputs[6])
    additive_output = additive.velocity(*inputs[:3], inputs[4], suffix_valid_mask=inputs[3], action_feature_mask=inputs[5], action_out_proj=inputs[6])
    assert torch.equal(main_output, additive_output)


class _ReturnQuery(torch.nn.Module):
    def forward(self, queries, _context, query_valid_mask):
        return queries * query_valid_mask.unsqueeze(-1)


class _CaptureQuery(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = None

    def forward(self, queries, _context, query_valid_mask):
        self.query = queries.detach().clone()
        return torch.zeros_like(queries) * query_valid_mask.unsqueeze(-1)


def test_action_query_exactly_adds_suffix_position_noisy_action_and_timestep():
    adapter = _adapter("action_query")
    capture = _CaptureQuery()
    adapter.cross_attention = capture
    suffix, noisy, time, valid, context, feature, action_out = _inputs()
    adapter.velocity(
        suffix,
        noisy,
        time,
        context,
        suffix_valid_mask=valid,
        action_feature_mask=feature,
        action_out_proj=action_out,
    )
    sanitized = noisy[..., :7] * valid.unsqueeze(-1)
    expected = (
        suffix.float()
        + adapter.learned_action_slot.unsqueeze(0)
        + adapter.noisy_action_projection(sanitized.float())
        + adapter.time_projection(time.float()[:, None, None])
    )
    torch.testing.assert_close(capture.query, expected)


def test_only_additive_structural_difference_is_query_excluding_suffix():
    main = _adapter("action_query")
    additive = _adapter("additive")
    additive.load_state_dict(main.state_dict(), strict=True)
    main.cross_attention = _ReturnQuery()
    additive.cross_attention = _ReturnQuery()
    for adapter in (main, additive):
        torch.nn.init.eye_(adapter.w_out.weight)
        torch.nn.init.zeros_(adapter.w_out.bias)
        adapter.alpha.data.fill_(1.0)
    inputs = _inputs()
    main_output = main.velocity(*inputs[:3], inputs[4], suffix_valid_mask=inputs[3], action_feature_mask=inputs[5], action_out_proj=inputs[6])
    additive_output = additive.velocity(*inputs[:3], inputs[4], suffix_valid_mask=inputs[3], action_feature_mask=inputs[5], action_out_proj=inputs[6])
    assert torch.count_nonzero(main_output - additive_output) > 0
    assert torch.count_nonzero(main_output[~inputs[3]]) == 0
    assert torch.count_nonzero(additive_output[~inputs[3]]) == 0


def test_additive_config_is_explicit_and_recipe_presets_are_frozen():
    common = dict(
        base_checkpoint=ROOT / "assets/base_checkpoint",
        constructor_assets=ROOT / "assets/smolvlm_constructor",
        device="cpu",
    )
    main = load_force_config(**common, force_variant=FORCE_TOKEN_MOE)
    additive = load_force_config(**common, force_variant=FORCE_TOKEN_MOE_ADDITIVE)
    assert main.adapter_query_mode == "action_query"
    assert additive.adapter_query_mode == "additive"
    assert main.get_optimizer_preset() == additive.get_optimizer_preset()
    assert main.get_scheduler_preset() == additive.get_scheduler_preset()
