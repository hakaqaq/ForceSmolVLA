import math
from dataclasses import replace

import torch

from forcesmolvla.force_token import (
    CAMERA1_SPAN,
    CAMERA2_SPAN,
    FORCE_SLOT_INDEX,
    FUSION_SELECTION_STOP,
    LANGUAGE_SPAN,
    N_FUSED_PHYSICAL,
    ForceActionAdapter,
    ForceContext,
    ForceCrossAttention,
    PreparedForceContextBinding,
    ForceTokenDenseCompute,
    module_state_sha256,
)
from forcesmolvla.modeling_forcesmolvla import ForceVLAFlowMatching


def _context(values: torch.Tensor, valid: torch.Tensor | None = None) -> ForceContext:
    if valid is None:
        valid = torch.ones(values.shape[:2], dtype=torch.bool)
    return ForceContext(values.float(), valid)


def _binding(batch_size: int = 2) -> PreparedForceContextBinding:
    return PreparedForceContextBinding(
        chunk_id=tuple(f"chunk-{index}" for index in range(batch_size)),
        sample_id=tuple(f"sample-{index}" for index in range(batch_size)),
        context_generation=3,
        model_generation=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_frozen_physical_layout_and_segment_boundaries():
    assert CAMERA1_SPAN == (0, 64)
    assert CAMERA2_SPAN == (64, 128)
    assert LANGUAGE_SPAN == (128, 176)
    assert FUSION_SELECTION_STOP == 176
    assert FORCE_SLOT_INDEX == 176
    assert N_FUSED_PHYSICAL == 177
    segments = ForceTokenDenseCompute.selection_segment_ids(device=torch.device("cpu"))
    assert torch.equal(segments[:64], torch.zeros(64, dtype=torch.long))
    assert torch.equal(segments[64:128], torch.ones(64, dtype=torch.long))
    assert torch.equal(segments[128:176], torch.full((48,), 2, dtype=torch.long))
    assert segments[176].item() == 3


def test_force_fusion_excludes_state_and_masks_right_padded_language():
    torch.manual_seed(42)
    module = ForceTokenDenseCompute(d_vlm=16, d_expert=8).eval()
    prefix = torch.randn(1, 177, 16)
    valid = torch.ones(1, 177, dtype=torch.bool)
    valid[:, 170:176] = False
    wrench = torch.randn(1, 6)
    expected = module(prefix, valid, wrench)

    changed = prefix.clone()
    changed[:, 176] = 1e6  # Physical state token is excluded from fusion.
    changed[:, 170:176] = -1e6  # Invalid right-padding cannot affect valid outputs.
    actual = module(changed, valid, wrench)
    torch.testing.assert_close(actual.z_action_fp32, expected.z_action_fp32)
    assert actual.fused_valid_mask.shape == (1, 177)
    assert actual.fused_valid_mask[:, 176].all()
    assert not actual.fused_valid_mask[:, 170:176].any()
    assert torch.count_nonzero(actual.z_action_fp32[:, 170:176]) == 0


def test_force_cross_attention_is_exact_single_head_formula_and_masks_keys_queries():
    torch.manual_seed(7)
    module = ForceCrossAttention(d_expert=8)
    assert module.scale == 1 / math.sqrt(8)
    assert not hasattr(module, "out_proj")
    assert not any(isinstance(child, torch.nn.MultiheadAttention) for child in module.modules())

    queries = torch.randn(2, 3, 8, dtype=torch.float32)
    values = torch.randn(2, 177, 8, dtype=torch.float32)
    valid_keys = torch.ones(2, 177, dtype=torch.bool)
    valid_keys[:, 10] = False
    valid_queries = torch.tensor([[True, False, True], [False, True, True]])
    context = _context(values, valid_keys)
    actual = module(queries, context, valid_queries)

    q = module.q_proj(queries)
    k = module.k_proj(values)
    v = module.v_proj(values)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(8)
    logits = logits.masked_fill(~valid_keys[:, None, :], -torch.inf)
    expected = torch.softmax(logits, dim=-1) @ v
    expected = expected * valid_queries.unsqueeze(-1)
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[~valid_queries]) == 0

    changed_values = values.clone()
    changed_values[:, 10] = 1e9
    changed = module(queries, _context(changed_values, valid_keys), valid_queries)
    torch.testing.assert_close(changed, actual)


def test_prepared_force_context_projects_kv_once_for_repeated_queries():
    module = ForceCrossAttention(d_expert=8)
    context = _context(torch.randn(2, 177, 8))
    calls = {"k": 0, "v": 0}
    handles = [
        module.k_proj.register_forward_hook(
            lambda *_args: calls.__setitem__("k", calls["k"] + 1)
        ),
        module.v_proj.register_forward_hook(
            lambda *_args: calls.__setitem__("v", calls["v"] + 1)
        ),
    ]
    binding = _binding()
    prepared = module.prepare(context, binding=binding)
    valid = torch.ones(2, 3, dtype=torch.bool)
    module(torch.randn(2, 3, 8), prepared, valid, prepared_binding=binding)
    module(torch.randn(2, 3, 8), prepared, valid, prepared_binding=binding)
    for handle in handles:
        handle.remove()
    assert calls == {"k": 1, "v": 1}


def test_prepared_force_context_rejects_any_lifecycle_binding_change():
    module = ForceCrossAttention(d_expert=8)
    binding = _binding()
    prepared = module.prepare(_context(torch.randn(2, 177, 8)), binding=binding)
    valid = torch.ones(2, 3, dtype=torch.bool)
    query = torch.randn(2, 3, 8)
    changes = (
        {"chunk_id": ("new-0", "new-1")},
        {"sample_id": ("other-0", "other-1")},
        {"context_generation": 4},
        {"model_generation": 8},
        {"device": torch.device("meta")},
        {"dtype": torch.float64},
    )
    for change in changes:
        with __import__("pytest").raises(RuntimeError, match="STALE"):
            module(
                query,
                prepared,
                valid,
                prepared_binding=replace(binding, **change),
            )


def test_model_generation_changes_after_optimizer_step():
    module = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    before = ForceVLAFlowMatching.parameter_generation(module)
    module(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    after = ForceVLAFlowMatching.parameter_generation(module)
    assert after != before


def test_adapter_has_unique_zero_output_projection_and_first_step_wout_gradient():
    torch.manual_seed(42)
    adapter = ForceActionAdapter(d_expert=8, horizon=3)
    assert torch.count_nonzero(adapter.w_out.weight) == 0
    assert torch.count_nonzero(adapter.w_out.bias) == 0
    assert not hasattr(adapter.cross_attention, "out_proj")
    assert [name for name, _ in adapter.named_modules() if name.endswith("w_out")] == ["w_out"]

    suffix = torch.randn(2, 3, 8, dtype=torch.float32, requires_grad=True)
    noisy = torch.randn(2, 3, 32, dtype=torch.float32)
    timestep = torch.tensor([0.2, 0.8], dtype=torch.float32)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    feature = valid[:, :, None] & (torch.arange(32).view(1, 1, 32) < 7)
    action_out = torch.nn.Linear(8, 32)
    context = _context(torch.randn(2, 177, 8))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        velocity = adapter.velocity(
            suffix,
            noisy,
            timestep,
            context,
            suffix_valid_mask=valid,
            action_feature_mask=feature,
            action_out_proj=action_out,
        )
    expected = action_out(suffix.float()) * feature
    assert velocity.dtype == torch.float32
    torch.testing.assert_close(velocity, expected)
    assert torch.count_nonzero(velocity[~feature]) == 0

    velocity.sum().backward()
    assert adapter.w_out.weight.grad is not None
    assert torch.count_nonzero(adapter.w_out.weight.grad) > 0


def test_seed42_force_initialization_tensor_hash_is_deterministic():
    torch.manual_seed(42)
    first = ForceActionAdapter(d_expert=8, horizon=3)
    torch.manual_seed(42)
    second = ForceActionAdapter(d_expert=8, horizon=3)
    assert module_state_sha256(first) == module_state_sha256(second)
