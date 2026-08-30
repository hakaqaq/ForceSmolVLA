import torch

from forcesmolvla.force_token import (
    ForceContext,
    ForceCrossAttention,
    PreparedForceContextBinding,
)


def test_prepared_force_kv_stays_fp32_inside_outer_bf16_autocast():
    attention = ForceCrossAttention(8)
    context = ForceContext(
        z_action_fp32=torch.randn(2, 177, 8, dtype=torch.float32),
        fused_valid_mask=torch.ones(2, 177, dtype=torch.bool),
    )
    binding = PreparedForceContextBinding(
        chunk_id=("chunk-0", "chunk-1"),
        sample_id=("sample-0", "sample-1"),
        context_generation=0,
        model_generation=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        prepared = attention.prepare(context, binding=binding)
    assert prepared.key_fp32.dtype == torch.float32
    assert prepared.value_fp32.dtype == torch.float32
