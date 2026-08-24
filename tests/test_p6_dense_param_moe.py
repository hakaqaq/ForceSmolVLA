import torch

from forcesmolvla.force_token import (
    MOE_NUM_EXPERTS,
    ROUTER_INIT_STD,
    DenseParamRefiner,
    ForceTokenDenseParam,
    ForceTokenMoE,
    Top1MoERefiner,
    solve_dense_param_hidden_dim,
)


def test_router_uses_deterministic_forcevla_flaxformer_initialization():
    torch.manual_seed(42)
    first = Top1MoERefiner(960)
    torch.manual_seed(42)
    second = Top1MoERefiner(960)
    assert torch.equal(first.router.weight, second.router.weight)
    assert torch.count_nonzero(first.router.bias) == 0
    assert first.router.weight.mean().abs() < 0.001
    assert abs(float(first.router.weight.std().detach()) - ROUTER_INIT_STD) < 0.001


def test_dense_param_hidden_dim_is_unique_nearest_moe_budget():
    d_vlm = 960
    hidden = solve_dense_param_hidden_dim(d_vlm)
    assert hidden == 15364
    target = 32 * d_vlm**2 + 24 * d_vlm + 4
    actual = hidden * (2 * d_vlm + 1) + d_vlm
    assert abs(actual - target) == 960
    assert abs(actual - target) / target < 0.001
    assert abs(actual - target) < abs((hidden - 1) * (2 * d_vlm + 1) + d_vlm - target)


def test_dense_param_and_moe_resolved_parameter_and_active_compute_budgets():
    d_vlm = 960
    with torch.device("meta"):
        dense = DenseParamRefiner(d_vlm)
        moe = Top1MoERefiner(d_vlm)
    dense_parameters = sum(parameter.numel() for parameter in dense.parameters())
    moe_parameters = sum(parameter.numel() for parameter in moe.parameters())
    assert dense_parameters == 29_517_124
    assert moe_parameters == 29_516_164
    assert abs(dense_parameters - moe_parameters) == 960
    assert abs(dense_parameters - moe_parameters) / moe_parameters < 0.001

    dense_compute_active_macs = 8 * d_vlm**2
    moe_active_macs = 8 * d_vlm**2 + 4 * d_vlm + d_vlm
    assert dense_compute_active_macs == 7_372_800
    assert moe_active_macs == 7_377_600
    assert abs(dense_compute_active_macs - moe_active_macs) / moe_active_macs < 0.01


def test_dense_param_refiner_resolves_hidden_and_masks_invalid_tokens():
    torch.manual_seed(42)
    refiner = DenseParamRefiner(8)
    assert refiner.hidden_dim == solve_dense_param_hidden_dim(8)
    values = torch.randn(2, 5, 8)
    valid = torch.tensor([[1, 1, 0, 1, 1], [1, 0, 0, 1, 1]], dtype=torch.bool)
    output = refiner(values, valid)
    assert torch.count_nonzero(output[~valid]) == 0


def test_moe_capacity_free_top1_has_no_drop_and_only_one_active_expert_per_token():
    torch.manual_seed(42)
    moe = Top1MoERefiner(8)
    values = torch.randn(3, 7, 8)
    valid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    active_rows = [0] * MOE_NUM_EXPERTS
    handles = [
        expert.register_forward_hook(
            lambda _module, inputs, _output, expert_id=expert_id: active_rows.__setitem__(
                expert_id, active_rows[expert_id] + inputs[0].shape[0]
            )
        )
        for expert_id, expert in enumerate(moe.experts)
    ]
    output, state = moe(values, valid)
    for handle in handles:
        handle.remove()

    assert sum(active_rows) == int(valid.sum())
    assert state.route_ids[valid].numel() == int(valid.sum())
    assert torch.all((0 <= state.route_ids[valid]) & (state.route_ids[valid] < 4))
    assert torch.all(state.route_ids[~valid] == -1)
    assert torch.count_nonzero(state.probabilities_fp32[~valid]) == 0
    assert torch.count_nonzero(output[~valid]) == 0
    torch.testing.assert_close(
        state.probabilities_fp32[valid].sum(dim=-1),
        torch.ones(int(valid.sum())),
    )


def test_moe_routing_and_output_are_batch_and_permutation_invariant():
    torch.manual_seed(9)
    moe = Top1MoERefiner(8).eval()
    fixed = torch.randn(1, 6, 8)
    fixed_valid = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    fixed_output, fixed_state = moe(fixed, fixed_valid)

    competitors = torch.randn(3, 6, 8)
    competitor_valid = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 0]],
        dtype=torch.bool,
    )
    combined = torch.cat([competitors[:2], fixed, competitors[2:]], dim=0)
    combined_valid = torch.cat([competitor_valid[:2], fixed_valid, competitor_valid[2:]], dim=0)
    combined_output, combined_state = moe(combined, combined_valid)
    torch.testing.assert_close(combined_output[2], fixed_output[0])
    torch.testing.assert_close(
        combined_state.probabilities_fp32[2], fixed_state.probabilities_fp32[0]
    )
    assert torch.equal(combined_state.route_ids[2], fixed_state.route_ids[0])

    permutation = torch.tensor([3, 1, 0, 2])
    permuted_output, permuted_state = moe(combined[permutation], combined_valid[permutation])
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(permuted_output[inverse], combined_output)
    torch.testing.assert_close(
        permuted_state.probabilities_fp32[inverse], combined_state.probabilities_fp32
    )
    assert torch.equal(permuted_state.route_ids[inverse], combined_state.route_ids)


def test_moe_tie_breaks_to_minimum_expert_and_router_is_fp32_under_autocast():
    torch.manual_seed(42)
    moe = Top1MoERefiner(8)
    torch.nn.init.zeros_(moe.router.weight)
    torch.nn.init.zeros_(moe.router.bias)
    values = torch.randn(2, 4, 8)
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _output, state = moe(values, valid)
    assert state.logits_fp32.dtype == torch.float32
    assert state.probabilities_fp32.dtype == torch.float32
    assert torch.all(state.route_ids[valid] == 0)


def test_moe_state_dict_names_are_explicit_and_force_context_exposes_router_state():
    torch.manual_seed(42)
    branch = ForceTokenMoE(d_vlm=16, d_expert=8).eval()
    names = set(branch.state_dict())
    assert "refiner.norm.weight" in names
    assert "refiner.router.weight" in names
    for expert_id in range(4):
        assert f"refiner.experts.{expert_id}.linear_in.weight" in names
        assert f"refiner.experts.{expert_id}.linear_out.weight" in names

    prefix = torch.randn(2, 177, 16)
    valid = torch.ones(2, 177, dtype=torch.bool)
    valid[1, 160:176] = False
    context = branch(prefix, valid, torch.randn(2, 6))
    assert context.router_state is not None
    assert context.router_state.route_ids.shape == (2, 177)
    assert context.router_state.route_ids[1, 160:176].eq(-1).all()
    assert context.router_state.route_ids[:, 176].ge(0).all()


def test_dense_and_moe_common_modules_have_identical_seeded_initialization():
    dense = ForceTokenDenseParam(16, 8, initialization_seed=42)
    torch.manual_seed(999)
    moe = ForceTokenMoE(16, 8, initialization_seed=42)
    dense_common = {
        name: value
        for name, value in dense.state_dict().items()
        if not name.startswith("refiner.")
    }
    moe_common = {
        name: value
        for name, value in moe.state_dict().items()
        if not name.startswith("refiner.")
    }
    assert dense_common.keys() == moe_common.keys()
    for name in dense_common:
        assert torch.equal(dense_common[name], moe_common[name]), name
