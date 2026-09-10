"""P5/P6 ForceToken Dense and deterministic top-1 MoE modules."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F


CAMERA1_SPAN = (0, 64)
CAMERA2_SPAN = (64, 128)
LANGUAGE_SPAN = (128, 176)
FUSION_SELECTION_STOP = 176
FORCE_SLOT_INDEX = 176
N_FUSED_PHYSICAL = 177
FORCE_SEGMENT_ID = 3
MOE_NUM_EXPERTS = 4
ROUTER_INIT_STD = 0.02


@contextmanager
def _forked_cpu_seed(seed: int | None):
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def _xavier_linear(module: nn.Linear, *, zero: bool = False) -> None:
    if zero:
        nn.init.zeros_(module.weight)
    else:
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _masked_softmax(logits: torch.Tensor, valid_keys: torch.Tensor) -> torch.Tensor:
    if valid_keys.dtype != torch.bool or valid_keys.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("valid_keys must have bool shape [B,K]")
    if not torch.all(valid_keys.any(dim=-1)):
        raise ValueError("attention requires at least one valid key per sample")
    return torch.softmax(logits.masked_fill(~valid_keys[:, None, :], -torch.inf), dim=-1)


@dataclass(frozen=True)
class RouterState:
    logits_fp32: torch.Tensor
    probabilities_fp32: torch.Tensor
    route_ids: torch.Tensor
    valid_mask: torch.Tensor

    def validate(self) -> None:
        if self.logits_fp32.dtype != torch.float32 or self.probabilities_fp32.dtype != torch.float32:
            raise ValueError("router logits and probabilities must be fp32")
        if self.logits_fp32.shape != self.probabilities_fp32.shape:
            raise ValueError("router logits/probability shape mismatch")
        if self.logits_fp32.shape[:2] != self.valid_mask.shape:
            raise ValueError("router valid mask shape mismatch")
        if self.logits_fp32.shape[-1] != MOE_NUM_EXPERTS:
            raise ValueError("router expert count mismatch")
        if self.route_ids.shape != self.valid_mask.shape or self.route_ids.dtype != torch.long:
            raise ValueError("router route id shape or dtype mismatch")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("router valid mask must be bool")
        if not torch.all(self.route_ids[self.valid_mask] >= 0):
            raise ValueError("valid router token is not assigned")
        if not torch.all(self.route_ids[~self.valid_mask] == -1):
            raise ValueError("invalid router token must have route id -1")
        valid_probabilities = self.probabilities_fp32[self.valid_mask]
        if not torch.allclose(
            valid_probabilities.sum(dim=-1),
            torch.ones_like(valid_probabilities[:, 0]),
        ):
            raise ValueError("valid router probabilities must sum to one")
        if torch.count_nonzero(self.probabilities_fp32[~self.valid_mask]):
            raise ValueError("invalid router probabilities must be zero")


@dataclass(frozen=True)
class ForceContext:
    z_action_fp32: torch.Tensor
    fused_valid_mask: torch.Tensor
    router_state: RouterState | None = None

    def validate(self) -> None:
        if self.z_action_fp32.dtype != torch.float32 or self.z_action_fp32.ndim != 3:
            raise ValueError("z_action_fp32 must be fp32 [B,N,D_expert]")
        if self.fused_valid_mask.dtype != torch.bool:
            raise ValueError("fused_valid_mask must be bool")
        if self.fused_valid_mask.shape != self.z_action_fp32.shape[:2]:
            raise ValueError("ForceContext mask shape mismatch")
        if self.z_action_fp32.shape[1] != N_FUSED_PHYSICAL:
            raise ValueError("ForceContext physical length mismatch")
        if not torch.all(self.fused_valid_mask[:, FORCE_SLOT_INDEX]):
            raise ValueError("Force slot must always be valid")
        if not torch.all(torch.isfinite(self.z_action_fp32)):
            raise ValueError("ForceContext contains nonfinite values")
        if self.router_state is not None:
            self.router_state.validate()
            if not torch.equal(self.router_state.valid_mask, self.fused_valid_mask):
                raise ValueError("router and fused valid masks differ")


@dataclass(frozen=True)
class PreparedForceContextBinding:
    chunk_id: tuple[str, ...]
    sample_id: tuple[str, ...]
    context_generation: int
    model_generation: int
    device: torch.device
    dtype: torch.dtype

    def validate(self, *, batch_size: int) -> None:
        if len(self.chunk_id) != batch_size or len(self.sample_id) != batch_size:
            raise ValueError("prepared force binding must be batch-bound")
        if len(set(self.chunk_id)) != batch_size or any(
            not value for value in (*self.chunk_id, *self.sample_id)
        ):
            raise ValueError("prepared force binding IDs must be nonempty and chunk-unique")
        if self.context_generation < 0 or self.model_generation < 0:
            raise ValueError("prepared force generations must be nonnegative")
        if self.dtype != torch.float32:
            raise ValueError("prepared force binding dtype must be torch.float32")


@dataclass(frozen=True)
class PreparedForceContext:
    key_fp32: torch.Tensor
    value_fp32: torch.Tensor
    fused_valid_mask: torch.Tensor
    binding: PreparedForceContextBinding

    def validate(self, *, expected_binding: PreparedForceContextBinding | None = None) -> None:
        if self.key_fp32.dtype != torch.float32 or self.value_fp32.dtype != torch.float32:
            raise ValueError("prepared force K/V must be fp32")
        if self.key_fp32.shape != self.value_fp32.shape or self.key_fp32.ndim != 3:
            raise ValueError("prepared force K/V shape mismatch")
        if self.fused_valid_mask.shape != self.key_fp32.shape[:2]:
            raise ValueError("prepared force mask shape mismatch")
        self.binding.validate(batch_size=self.key_fp32.shape[0])
        if (
            self.key_fp32.device != self.binding.device
            or self.value_fp32.device != self.binding.device
            or self.fused_valid_mask.device != self.binding.device
            or self.key_fp32.dtype != self.binding.dtype
            or self.value_fp32.dtype != self.binding.dtype
        ):
            raise RuntimeError("PREPARED_FORCE_CONTEXT_DEVICE_OR_DTYPE_MISMATCH")
        if expected_binding is not None and self.binding != expected_binding:
            raise RuntimeError("PREPARED_FORCE_CONTEXT_STALE")
        if not torch.all(self.fused_valid_mask[:, FORCE_SLOT_INDEX]):
            raise ValueError("prepared force slot must always be valid")
        if not torch.all(torch.isfinite(self.key_fp32)) or not torch.all(
            torch.isfinite(self.value_fp32)
        ):
            raise ValueError("prepared force K/V contains nonfinite values")


def fp32_action_projection(
    action_out_proj: nn.Linear, hidden: torch.Tensor, action_feature_mask: torch.Tensor
) -> torch.Tensor:
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return action_out_proj(hidden.float()) * action_feature_mask.to(dtype=torch.float32)


class ForceMLP(nn.Module):
    def __init__(self, d_vlm: int):
        super().__init__()
        self.linear_in = nn.Linear(6, d_vlm, bias=True)
        self.linear_out = nn.Linear(d_vlm, d_vlm, bias=True)
        _xavier_linear(self.linear_in)
        _xavier_linear(self.linear_out)

    def forward(self, normalized_wrench6: torch.Tensor) -> torch.Tensor:
        if normalized_wrench6.ndim != 2 or normalized_wrench6.shape[-1] != 6:
            raise ValueError("normalized wrench must have shape [B,6]")
        if not torch.all(torch.isfinite(normalized_wrench6)):
            raise ValueError("normalized wrench contains nonfinite values")
        return self.linear_out(F.silu(self.linear_in(normalized_wrench6)))


class MaskedMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("fusion width must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            _xavier_linear(module)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = values.shape
        if width != self.d_model or valid.shape != (batch, tokens) or valid.dtype != torch.bool:
            raise ValueError("fusion attention shape or mask mismatch")

        def heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        q = heads(self.q_proj(values))
        k = heads(self.k_proj(values))
        v = heads(self.v_proj(values))
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weights = torch.softmax(
            logits.masked_fill(~valid[:, None, None, :], -torch.inf), dim=-1
        )
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, tokens, width)
        output = self.out_proj(output)
        return output * valid.unsqueeze(-1).to(dtype=output.dtype)


class FusionBlock(nn.Module):
    def __init__(self, d_vlm: int, *, num_heads: int = 8):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_vlm, eps=1e-5)
        self.attention = MaskedMultiheadSelfAttention(d_vlm, num_heads)
        self.ln_ffn = nn.LayerNorm(d_vlm, eps=1e-5)
        self.ffn_in = nn.Linear(d_vlm, 4 * d_vlm, bias=True)
        self.ffn_out = nn.Linear(4 * d_vlm, d_vlm, bias=True)
        _xavier_linear(self.ffn_in)
        _xavier_linear(self.ffn_out)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid.unsqueeze(-1).to(dtype=values.dtype)
        values = (values + self.attention(self.ln_attn(values), valid)) * mask
        update = self.ffn_out(F.gelu(self.ffn_in(self.ln_ffn(values))))
        return (values + update) * mask


class DenseComputeRefiner(nn.Module):
    def __init__(self, d_vlm: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_vlm, eps=1e-5)
        self.linear_in = nn.Linear(d_vlm, 4 * d_vlm, bias=True)
        self.linear_out = nn.Linear(4 * d_vlm, d_vlm, bias=True)
        _xavier_linear(self.linear_in)
        _xavier_linear(self.linear_out)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        update = self.linear_out(F.gelu(self.linear_in(self.norm(values))))
        return (values + update) * valid.unsqueeze(-1).to(dtype=values.dtype)


def solve_dense_param_hidden_dim(d_vlm: int) -> int:
    target = 32 * d_vlm**2 + 24 * d_vlm + 4
    exact = (target - d_vlm) / (2 * d_vlm + 1)
    candidates = {max(0, math.floor(exact)), max(0, math.ceil(exact))}
    return min(
        candidates,
        key=lambda hidden: (abs(hidden * (2 * d_vlm + 1) + d_vlm - target), hidden),
    )


class DenseParamRefiner(nn.Module):
    def __init__(self, d_vlm: int):
        super().__init__()
        self.hidden_dim = solve_dense_param_hidden_dim(d_vlm)
        self.norm = nn.LayerNorm(d_vlm, eps=1e-5)
        self.linear_in = nn.Linear(d_vlm, self.hidden_dim, bias=True)
        self.linear_out = nn.Linear(self.hidden_dim, d_vlm, bias=True)
        _xavier_linear(self.linear_in)
        _xavier_linear(self.linear_out)
        target = 32 * d_vlm**2 + 24 * d_vlm + 4
        actual = self.hidden_dim * (2 * d_vlm + 1) + d_vlm
        if d_vlm == 960 and abs(actual - target) / target > 1e-3:
            raise RuntimeError("Dense-Param refiner exceeds the 0.1% MoE parameter budget")

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        update = self.linear_out(F.gelu(self.linear_in(self.norm(values))))
        return (values + update) * valid.unsqueeze(-1).to(dtype=values.dtype)


class ExpertMLP(nn.Module):
    def __init__(self, d_vlm: int):
        super().__init__()
        self.linear_in = nn.Linear(d_vlm, 4 * d_vlm, bias=True)
        self.linear_out = nn.Linear(4 * d_vlm, d_vlm, bias=True)
        _xavier_linear(self.linear_in)
        _xavier_linear(self.linear_out)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear_out(F.gelu(self.linear_in(values)))


class Top1MoERefiner(nn.Module):
    """Capacity-free deterministic top-1 routing with exactly one active expert."""

    def __init__(self, d_vlm: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_vlm, eps=1e-5)
        self.router = nn.Linear(d_vlm, MOE_NUM_EXPERTS, bias=True)
        self.experts = nn.ModuleList([ExpertMLP(d_vlm) for _ in range(MOE_NUM_EXPERTS)])
        # Match ForceVLA/FlaxFormer RouterWeights while preserving the frozen
        # deterministic no-jitter routing contract.
        nn.init.normal_(self.router.weight, mean=0.0, std=ROUTER_INIT_STD)
        nn.init.zeros_(self.router.bias)

    def forward(
        self, values: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, RouterState]:
        if values.ndim != 3 or valid.shape != values.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("MoE values/valid shape mismatch")
        normalized = self.norm(values)
        with torch.autocast(device_type=values.device.type, enabled=False):
            logits = self.router(normalized.float())
            probabilities = torch.softmax(logits, dim=-1)
            routes = torch.argmax(probabilities, dim=-1)
            masked_probabilities = probabilities * valid.unsqueeze(-1)
            routes = routes.masked_fill(~valid, -1)

        flat_normalized = normalized.reshape(-1, normalized.shape[-1])
        flat_routes = routes.reshape(-1)
        flat_probabilities = masked_probabilities.reshape(-1, MOE_NUM_EXPERTS)
        update = torch.zeros_like(flat_normalized)
        for expert_id, expert in enumerate(self.experts):
            token_indices = torch.nonzero(flat_routes == expert_id, as_tuple=False).squeeze(-1)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(flat_normalized.index_select(0, token_indices))
            gate = flat_probabilities.index_select(0, token_indices)[:, expert_id]
            selected = (expert_output.float() * gate.unsqueeze(-1)).to(dtype=update.dtype)
            update = update.index_copy(0, token_indices, selected)
        update = update.view_as(values)
        output = (values + update) * valid.unsqueeze(-1).to(dtype=values.dtype)
        router_state = RouterState(
            logits_fp32=logits * valid.unsqueeze(-1),
            probabilities_fp32=masked_probabilities,
            route_ids=routes,
            valid_mask=valid,
        )
        router_state.validate()
        return output, router_state


class ForceTokenFusion(nn.Module):
    def __init__(
        self,
        d_vlm: int,
        d_expert: int,
        refiner_type: type[nn.Module],
        *,
        initialization_seed: int | None = None,
    ):
        super().__init__()
        self.d_vlm = d_vlm
        self.d_expert = d_expert
        with _forked_cpu_seed(None if initialization_seed is None else initialization_seed + 1):
            self.force_mlp = ForceMLP(d_vlm)
        with _forked_cpu_seed(None if initialization_seed is None else initialization_seed + 2):
            self.segment_embedding = nn.Embedding(4, d_vlm)
            self.fusion_position_embedding = nn.Embedding(N_FUSED_PHYSICAL, d_vlm)
            nn.init.xavier_uniform_(self.segment_embedding.weight)
            nn.init.xavier_uniform_(self.fusion_position_embedding.weight)
        with _forked_cpu_seed(None if initialization_seed is None else initialization_seed + 3):
            self.fusion_blocks = nn.ModuleList([FusionBlock(d_vlm) for _ in range(2)])
        with _forked_cpu_seed(None if initialization_seed is None else initialization_seed + 4):
            self.refiner = refiner_type(d_vlm)
        with _forked_cpu_seed(None if initialization_seed is None else initialization_seed + 5):
            self.guidance_projection = nn.Linear(d_vlm, d_expert, bias=True)
            _xavier_linear(self.guidance_projection)

    @staticmethod
    def selection_segment_ids(*, device: torch.device) -> torch.Tensor:
        result = torch.empty(N_FUSED_PHYSICAL, dtype=torch.long, device=device)
        result[CAMERA1_SPAN[0] : CAMERA1_SPAN[1]] = 0
        result[CAMERA2_SPAN[0] : CAMERA2_SPAN[1]] = 1
        result[LANGUAGE_SPAN[0] : LANGUAGE_SPAN[1]] = 2
        result[FORCE_SLOT_INDEX] = FORCE_SEGMENT_ID
        return result

    def forward(
        self,
        prefix_out: torch.Tensor,
        prefix_valid_mask: torch.Tensor,
        normalized_wrench6: torch.Tensor,
    ) -> ForceContext:
        batch, physical, width = prefix_out.shape
        if physical != 177 or width != self.d_vlm:
            raise ValueError("P5 requires prefix_out [B,177,D_vlm]")
        if prefix_valid_mask.shape != (batch, physical) or prefix_valid_mask.dtype != torch.bool:
            raise ValueError("prefix valid mask mismatch")
        if normalized_wrench6.shape != (batch, 6):
            raise ValueError("normalized wrench must have shape [B,6]")

        selected = prefix_out[:, :FUSION_SELECTION_STOP]
        force = self.force_mlp(normalized_wrench6.to(dtype=prefix_out.dtype)).unsqueeze(1)
        values = torch.cat([selected, force], dim=1)
        valid = torch.cat(
            [prefix_valid_mask[:, :FUSION_SELECTION_STOP], torch.ones(batch, 1, dtype=torch.bool, device=prefix_out.device)],
            dim=1,
        )
        segment_ids = self.selection_segment_ids(device=prefix_out.device)
        positions = torch.arange(N_FUSED_PHYSICAL, device=prefix_out.device)
        values = (
            values
            + self.segment_embedding(segment_ids).unsqueeze(0)
            + self.fusion_position_embedding(positions).unsqueeze(0)
        ) * valid.unsqueeze(-1).to(dtype=values.dtype)
        for block in self.fusion_blocks:
            values = block(values, valid)
        refined = self.refiner(values, valid)
        if isinstance(refined, tuple):
            values, router_state = refined
        else:
            values, router_state = refined, None
        with torch.autocast(device_type=values.device.type, enabled=False):
            z_action = self.guidance_projection(values.float())
        context = ForceContext(
            z_action_fp32=z_action,
            fused_valid_mask=valid,
            router_state=router_state,
        )
        context.validate()
        return context


class ForceTokenDenseCompute(ForceTokenFusion):
    def __init__(self, d_vlm: int, d_expert: int, *, initialization_seed: int | None = None):
        super().__init__(
            d_vlm, d_expert, DenseComputeRefiner, initialization_seed=initialization_seed
        )


class ForceTokenDenseParam(ForceTokenFusion):
    def __init__(self, d_vlm: int, d_expert: int, *, initialization_seed: int | None = None):
        super().__init__(
            d_vlm, d_expert, DenseParamRefiner, initialization_seed=initialization_seed
        )


class ForceTokenMoE(ForceTokenFusion):
    def __init__(self, d_vlm: int, d_expert: int, *, initialization_seed: int | None = None):
        super().__init__(d_vlm, d_expert, Top1MoERefiner, initialization_seed=initialization_seed)


class ForceCrossAttention(nn.Module):
    """Frozen single-head fp32 cross-attention; W_out is the only output projection."""

    def __init__(self, d_expert: int):
        super().__init__()
        self.d_expert = d_expert
        self.scale = 1.0 / math.sqrt(d_expert)
        self.q_proj = nn.Linear(d_expert, d_expert, bias=True)
        self.k_proj = nn.Linear(d_expert, d_expert, bias=True)
        self.v_proj = nn.Linear(d_expert, d_expert, bias=True)
        for module in (self.q_proj, self.k_proj, self.v_proj):
            _xavier_linear(module)

    def prepare(
        self, context: ForceContext, *, binding: PreparedForceContextBinding
    ) -> PreparedForceContext:
        context.validate()
        with torch.autocast(device_type=context.z_action_fp32.device.type, enabled=False):
            prepared = PreparedForceContext(
                key_fp32=self.k_proj(context.z_action_fp32.float()),
                value_fp32=self.v_proj(context.z_action_fp32.float()),
                fused_valid_mask=context.fused_valid_mask,
                binding=binding,
            )
        prepared.validate(expected_binding=binding)
        return prepared

    def forward(
        self,
        queries: torch.Tensor,
        context: ForceContext | PreparedForceContext,
        query_valid_mask: torch.Tensor,
        *,
        prepared_binding: PreparedForceContextBinding | None = None,
    ) -> torch.Tensor:
        if isinstance(context, ForceContext):
            if prepared_binding is not None:
                raise ValueError("prepared binding is only valid for PreparedForceContext")
            context.validate()
            key_fp32 = self.k_proj(context.z_action_fp32)
            value_fp32 = self.v_proj(context.z_action_fp32)
            fused_valid_mask = context.fused_valid_mask
        else:
            if prepared_binding is None:
                raise RuntimeError("PREPARED_FORCE_CONTEXT_BINDING_REQUIRED")
            context.validate(expected_binding=prepared_binding)
            key_fp32 = context.key_fp32
            value_fp32 = context.value_fp32
            fused_valid_mask = context.fused_valid_mask
        if queries.dtype != torch.float32 or queries.ndim != 3:
            raise ValueError("ForceCrossAttention queries must be fp32 [B,H,D_expert]")
        if queries.shape[0] != key_fp32.shape[0] or queries.shape[2] != self.d_expert:
            raise ValueError("ForceCrossAttention query shape mismatch")
        if query_valid_mask.shape != queries.shape[:2] or query_valid_mask.dtype != torch.bool:
            raise ValueError("query_valid_mask mismatch")
        q = self.q_proj(queries)
        logits = torch.matmul(q, key_fp32.transpose(-1, -2)) * self.scale
        weights = _masked_softmax(logits, fused_valid_mask)
        output = torch.matmul(weights, value_fp32)
        return output * query_valid_mask.unsqueeze(-1).to(dtype=output.dtype)


class ForceActionAdapter(nn.Module):
    def __init__(
        self,
        d_expert: int,
        horizon: int,
        *,
        query_mode: str = "action_query",
        initialization_seed: int | None = None,
    ):
        super().__init__()
        if query_mode not in {"action_query", "additive"}:
            raise ValueError(f"unsupported adapter query mode: {query_mode!r}")
        self.d_expert = d_expert
        self.horizon = horizon
        self.query_mode = query_mode
        with _forked_cpu_seed(
            None if initialization_seed is None else initialization_seed + 101
        ):
            self.learned_action_slot = nn.Parameter(
                torch.empty(horizon, d_expert, dtype=torch.float32)
            )
            nn.init.normal_(self.learned_action_slot, mean=0.0, std=0.02)
            self.time_projection = nn.Linear(1, d_expert, bias=True)
            self.noisy_action_projection = nn.Linear(7, d_expert, bias=True)
            _xavier_linear(self.time_projection)
            _xavier_linear(self.noisy_action_projection)
            self.cross_attention = ForceCrossAttention(d_expert)
            self.w_out = nn.Linear(d_expert, d_expert, bias=True)
            _xavier_linear(self.w_out, zero=True)
            self.alpha = nn.Parameter(torch.tensor(math.atanh(1e-3), dtype=torch.float32))

    def velocity(
        self,
        suffix_out: torch.Tensor,
        noisy_actions32: torch.Tensor,
        timestep: torch.Tensor,
        context: ForceContext | PreparedForceContext,
        *,
        suffix_valid_mask: torch.Tensor,
        action_feature_mask: torch.Tensor,
        action_out_proj: nn.Linear,
        prepared_binding: PreparedForceContextBinding | None = None,
    ) -> torch.Tensor:
        batch, horizon, width = suffix_out.shape
        if horizon != self.horizon or width != self.d_expert:
            raise ValueError("adapter suffix shape mismatch")
        if noisy_actions32.shape[:2] != (batch, horizon) or noisy_actions32.shape[-1] < 7:
            raise ValueError("adapter noisy action shape mismatch")
        if timestep.shape != (batch,):
            raise ValueError("adapter timestep must have shape [B]")
        if suffix_valid_mask.shape != (batch, horizon) or suffix_valid_mask.dtype != torch.bool:
            raise ValueError("adapter suffix mask mismatch")
        if action_feature_mask.shape != (batch, horizon, action_out_proj.out_features):
            raise ValueError("adapter action feature mask mismatch")

        with torch.autocast(device_type=suffix_out.device.type, enabled=False):
            valid = suffix_valid_mask
            sanitized_action7 = noisy_actions32[:, :, :7].float() * valid.unsqueeze(-1)
            conditioner = (
                self.learned_action_slot.unsqueeze(0)
                + self.time_projection(timestep.float()[:, None, None])
                + self.noisy_action_projection(sanitized_action7)
            )
            suffix_fp32 = suffix_out.float()
            query_main = suffix_fp32 + conditioner if self.query_mode == "action_query" else conditioner
            aggregate = (
                self.cross_attention(
                    query_main, context, valid, prepared_binding=prepared_binding
                )
                if isinstance(context, PreparedForceContext)
                else self.cross_attention(query_main, context, valid)
            )
            residual = torch.tanh(self.alpha) * self.w_out(aggregate)
            refined = (suffix_fp32 + residual) * valid.unsqueeze(-1)
            velocity = fp32_action_projection(action_out_proj, refined, action_feature_mask)
        return velocity


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
