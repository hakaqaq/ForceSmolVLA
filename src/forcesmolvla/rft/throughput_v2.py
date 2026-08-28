"""Append-only helpers for the Stage-2 throughput-v2 benchmark.

These helpers change scheduling only.  They do not change the Flow topology,
the action contract, loss definitions, optimizer ownership, or update counts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import re
import time
from typing import Any, Iterable

import torch

from forcesmolvla.force_token import PreparedForceContextBinding
from forcesmolvla.prefix import PrefixContext
from lerobot.policies.smolvla.modeling_smolvla import pad_vector
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


_CANDIDATE_SUFFIX = re.compile(r"/(?:cql_current|cql_next)=\d+$")


def _batch_size(batch: dict[str, Any]) -> int:
    return next(
        int(value.shape[0])
        for value in batch.values()
        if isinstance(value, torch.Tensor) and value.ndim
    )


def index_actor_batch(batch: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    """Select ordered Actor rows without changing non-batch metadata."""

    size = _batch_size(batch)
    index = torch.tensor(indices, dtype=torch.long, device=next(
        value.device for value in batch.values()
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size
    ))
    result: dict[str, Any] = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
            result[name] = value.index_select(0, index)
        elif isinstance(value, (tuple, list)) and len(value) == size:
            result[name] = type(value)(value[position] for position in indices)
        else:
            result[name] = value
    return result


def concat_actor_batches(batches: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate batches while preserving sample identity order."""

    batches = tuple(batches)
    if not batches:
        raise ValueError("THROUGHPUT_V2_EMPTY_BATCH_SEQUENCE")
    sizes = tuple(_batch_size(batch) for batch in batches)
    keys = set(batches[0])
    if any(set(batch) != keys for batch in batches[1:]):
        raise ValueError("THROUGHPUT_V2_BATCH_SCHEMA_MISMATCH")
    result: dict[str, Any] = {}
    for name in sorted(keys):
        values = [batch[name] for batch in batches]
        if all(isinstance(value, torch.Tensor) and value.ndim for value in values):
            result[name] = torch.cat(values, dim=0)
        elif all(isinstance(value, (tuple, list)) for value in values):
            flattened = [item for value in values for item in value]
            result[name] = type(values[0])(flattened)
        else:
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"THROUGHPUT_V2_NONBATCH_VALUE_MISMATCH:{name}")
            result[name] = values[0]
    if _batch_size(result) != sum(sizes):
        raise RuntimeError("THROUGHPUT_V2_CONCAT_BATCH_SIZE")
    return result


def slice_prefix_context(context: PrefixContext, indices: list[int]) -> PrefixContext:
    """Select PrefixContext rows, including every cached K/V tensor."""

    device = context.prefix_out.device
    index = torch.tensor(indices, dtype=torch.long, device=device)
    cache = {
        layer: {
            name: value.index_select(0, index)
            for name, value in states.items()
        }
        for layer, states in context.past_key_values.items()
    }
    result = replace(
        context,
        prefix_out=context.prefix_out.index_select(0, index),
        prefix_valid_mask=context.prefix_valid_mask.index_select(0, index),
        prefix_segment_ids=context.prefix_segment_ids.index_select(0, index),
        prefix_position_ids=context.prefix_position_ids.index_select(0, index),
        past_key_values=cache,
        cache_snapshot=None,
    )
    result.validate(check_cache=False)
    return result


@torch.no_grad()
def prepare_frozen_prefix(policy, batch: dict[str, Any]) -> tuple[PrefixContext, torch.Tensor]:
    """Create the frozen, detached prefix once for a unique observation batch."""

    if policy.training:
        raise RuntimeError("THROUGHPUT_V2_PREFIX_REQUIRES_ACTOR_EVAL")
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    state = state * (
        torch.arange(32, device=state.device).view(1, 32) < 7
    ).to(dtype=state.dtype)
    wrench = policy._prepare_wrench(batch, device=state.device)
    context = policy.model.encode_prefix(
        images,
        image_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        state,
        audit_cache=False,
    )
    tensors = [
        context.prefix_out,
        context.prefix_valid_mask,
        context.prefix_segment_ids,
        context.prefix_position_ids,
        *(
            value
            for states in context.past_key_values.values()
            for value in states.values()
        ),
    ]
    if any(value.requires_grad for value in tensors):
        raise RuntimeError("THROUGHPUT_V2_PREFIX_NOT_DETACHED")
    return context, wrench


def sample_from_frozen_prefix(
    policy,
    batch: dict[str, Any],
    noise7: torch.Tensor,
    context: PrefixContext,
    wrench: torch.Tensor,
    *,
    call_id: str,
    purpose: str,
) -> torch.Tensor:
    """Run the unchanged N=10 Flow integration from a detached prefix."""

    if policy.training:
        raise RuntimeError("THROUGHPUT_V2_FLOW_REQUIRES_ACTOR_EVAL")
    batch_size = _batch_size(batch)
    if noise7.shape != (batch_size, 50, 7) or noise7.dtype != torch.float32:
        raise ValueError("THROUGHPUT_V2_NOISE_SHAPE_OR_DTYPE")
    if policy.config.num_steps != 10 or policy.config.chunk_size != 50:
        raise RuntimeError("THROUGHPUT_V2_FLOW_TOPOLOGY_DRIFT")
    feature_mask = torch.ones(batch_size, 50, 32, dtype=torch.bool, device=noise7.device)
    feature_mask[..., 7:] = False
    suffix_valid = torch.ones(batch_size, 50, dtype=torch.bool, device=noise7.device)
    noise32 = pad_vector(noise7, 32) * feature_mask.to(dtype=noise7.dtype)
    identities = tuple(str(value) for value in batch["sample_identity"])
    binding = PreparedForceContextBinding(
        chunk_id=tuple(f"throughput-v2:{purpose}:{call_id}:{row}" for row in range(batch_size)),
        sample_id=identities,
        context_generation=policy._context_generation,
        model_generation=policy.model.parameter_generation(),
        device=noise7.device,
        dtype=torch.float32,
    )
    force_context = policy.model.build_force_context(
        context.prefix_out, context.prefix_valid_mask, wrench
    )
    if force_context is None:
        raise RuntimeError("THROUGHPUT_V2_FORCE_CONTEXT_MISSING")
    prepared = policy.model.force_adapter.cross_attention.prepare(
        force_context, binding=binding
    )
    mask = feature_mask.to(dtype=noise32.dtype)
    x_t = noise32 * mask
    dt = -1.0 / 10.0
    for step in range(10):
        timestep = torch.full(
            (batch_size,), 1.0 + step * dt,
            dtype=torch.float32, device=noise7.device,
        )
        velocity = policy.model.velocity_cached(
            context,
            x_t,
            timestep,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            force_context=prepared,
            force_context_binding=binding,
        )
        x_t = (x_t + dt * velocity) * mask
    result = x_t[..., :7].float()
    if result.shape != (batch_size, 50, 7) or not torch.isfinite(result).all():
        raise FloatingPointError("THROUGHPUT_V2_FLOW_OUTPUT_INVALID")
    return result


def canonical_observation_identity(value: str) -> str:
    """Remove candidate-only spelling while retaining the physical frame identity."""

    value = _CANDIDATE_SUFFIX.sub("", value)
    return value.replace("/next=", "/frame=")


def sample_grouped_flow_requests(
    policy,
    requests: list[tuple[str, dict[str, Any], torch.Tensor]],
    *,
    unique_observation_subbatch: int,
    call_id: str,
    capture: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, torch.Tensor]]:
    """Batch TD/current/next requests while preserving every sample/noise mapping.

    The scheduling unit is at most ``unique_observation_subbatch`` distinct
    observations.  Repeated M=2 candidate requests share one frozen prefix but
    remain separate Flow trajectories with their original noises.
    """

    if not requests or unique_observation_subbatch < 1:
        raise ValueError("THROUGHPUT_V2_GROUPED_REQUESTS_INVALID")
    batches = [item[1] for item in requests]
    noises = [item[2] for item in requests]
    combined_batch = concat_actor_batches(batches)
    combined_noise = torch.cat(noises, dim=0)
    labels = [label for label, batch, _noise in requests for _ in range(_batch_size(batch))]
    identities = tuple(combined_batch["sample_identity"])
    keys = [canonical_observation_identity(value) for value in identities]
    unique_keys = list(dict.fromkeys(keys))
    positions_by_key = {
        key: [index for index, value in enumerate(keys) if value == key]
        for key in unique_keys
    }
    outputs = torch.empty_like(combined_noise)
    captured: dict[str, torch.Tensor] = {}
    prefix_prefills = 0
    flow_calls = 0
    maximum_action_batch = 0
    started = time.perf_counter()
    for start in range(0, len(unique_keys), unique_observation_subbatch):
        selected_keys = unique_keys[start : start + unique_observation_subbatch]
        first_positions = [positions_by_key[key][0] for key in selected_keys]
        request_positions = [
            position
            for key in selected_keys
            for position in positions_by_key[key]
        ]
        unique_batch = index_actor_batch(combined_batch, first_positions)
        context, wrench = prepare_frozen_prefix(policy, unique_batch)
        key_to_unique = {key: index for index, key in enumerate(selected_keys)}
        inverse = [key_to_unique[keys[position]] for position in request_positions]
        request_batch = index_actor_batch(combined_batch, request_positions)
        expanded_context = slice_prefix_context(context, inverse)
        expanded_wrench = wrench.index_select(
            0, torch.tensor(inverse, device=wrench.device, dtype=torch.long)
        )
        value = sample_from_frozen_prefix(
            policy,
            request_batch,
            combined_noise[request_positions],
            expanded_context,
            expanded_wrench,
            call_id=f"{call_id}/unique={start}:{start + len(selected_keys)}",
            purpose="grouped_td_calql",
        )
        outputs[request_positions] = value
        prefix_prefills += 1
        flow_calls += 1
        maximum_action_batch = max(maximum_action_batch, len(request_positions))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    split: dict[str, list[torch.Tensor]] = {}
    offset = 0
    for label, batch, _noise in requests:
        count = _batch_size(batch)
        split.setdefault(label, []).append(outputs[offset : offset + count])
        offset += count
    result = {
        label: torch.cat(values, dim=0)
        for label, values in split.items()
    }
    if capture:
        for label, value in result.items():
            captured[f"action|{call_id}|{label}"] = value.detach().cpu()
        for label, batch, noise in requests:
            captured[f"noise|{call_id}|{label}"] = noise.detach().cpu()
    return result, {
        "seconds": elapsed,
        "flow_chunks_sampled": flow_calls,
        "euler_velocity_evaluations": 10 * flow_calls,
        "prefix_prefill_count": prefix_prefills,
        "policy_action_chunks": int(combined_noise.shape[0]),
        "unique_observation_count": len(unique_keys),
        "maximum_flow_action_batch": maximum_action_batch,
        "request_counts": dict(Counter(labels)),
        "sample_to_noise_mapping_preserved": True,
        "candidate_order_preserved": True,
    }, captured


class FrozenPrefixFlowCounter:
    """Flow dispatcher that reuses one prefix for repeated Cal-QL candidates."""

    def __init__(self, inference_batch_size: int, *, capture: bool = False) -> None:
        if inference_batch_size < 1:
            raise ValueError("THROUGHPUT_V2_FLOW_SUBBATCH")
        self.inference_batch_size = inference_batch_size
        self.capture = capture
        self.seconds = {name: 0.0 for name in ("td_next", "cql_current", "cql_next", "actor_guidance")}
        self.flow_chunks_sampled = 0
        self.euler_velocity_evaluations = 0
        self.prefix_prefill_count = 0
        self.policy_action_chunks = 0
        self.by_purpose: Counter[str] = Counter()
        self.captured: dict[str, torch.Tensor] = {}

    def sample(self, policy, batch, noise7, *, call_id: str, purpose: str):
        torch.cuda.synchronize()
        started = time.perf_counter()
        if purpose in {"cql_current", "cql_next"}:
            result = self._sample_repeated_candidates(policy, batch, noise7, call_id, purpose)
        else:
            result = self._sample_plain(policy, batch, noise7, call_id, purpose)
        torch.cuda.synchronize()
        self.seconds[purpose] += time.perf_counter() - started
        if self.capture:
            self.captured[f"action|{call_id}|{purpose}"] = result.detach().cpu()
            self.captured[f"noise|{call_id}|{purpose}"] = noise7.detach().cpu()
        return result

    def _sample_plain(self, policy, batch, noise7, call_id: str, purpose: str):
        outputs = []
        for start in range(0, noise7.shape[0], self.inference_batch_size):
            stop = min(start + self.inference_batch_size, noise7.shape[0])
            selected = list(range(start, stop))
            actor_batch = index_actor_batch(batch, selected)
            context, wrench = prepare_frozen_prefix(policy, actor_batch)
            outputs.append(sample_from_frozen_prefix(
                policy, actor_batch, noise7[start:stop], context, wrench,
                call_id=f"{call_id}/chunk={start}:{stop}", purpose=purpose,
            ))
            self.prefix_prefill_count += 1
            self.flow_chunks_sampled += 1
            self.euler_velocity_evaluations += 10
        result = torch.cat(outputs, dim=0)
        self.policy_action_chunks += result.shape[0]
        self.by_purpose[purpose] += result.shape[0]
        return result

    def _sample_repeated_candidates(self, policy, batch, noise7, call_id: str, purpose: str):
        identities = tuple(batch["sample_identity"])
        if len(identities) % 2:
            raise ValueError("THROUGHPUT_V2_CALQL_CANDIDATE_COUNT")
        base = tuple(_CANDIDATE_SUFFIX.sub("", value) for value in identities)
        if any(base[index] != base[index + 1] for index in range(0, len(base), 2)):
            raise RuntimeError("THROUGHPUT_V2_CALQL_REPEAT_ORDER")
        unique_positions = list(range(0, len(base), 2))
        grouped_noise = noise7.reshape(len(unique_positions), 2, 50, 7)
        grouped_outputs = []
        for start in range(0, len(unique_positions), self.inference_batch_size):
            stop = min(start + self.inference_batch_size, len(unique_positions))
            positions = unique_positions[start:stop]
            actor_batch = index_actor_batch(batch, positions)
            context, wrench = prepare_frozen_prefix(policy, actor_batch)
            candidates = []
            for candidate in range(2):
                candidate_batch = dict(actor_batch)
                candidate_batch["sample_identity"] = tuple(
                    identities[2 * row + candidate]
                    for row in range(start, stop)
                )
                candidates.append(sample_from_frozen_prefix(
                    policy, candidate_batch, grouped_noise[start:stop, candidate], context, wrench,
                    call_id=f"{call_id}/unique={start}:{stop}/candidate={candidate}",
                    purpose=purpose,
                ))
                self.flow_chunks_sampled += 1
                self.euler_velocity_evaluations += 10
            grouped_outputs.append(torch.stack(candidates, dim=1))
            self.prefix_prefill_count += 1
        result = torch.cat(grouped_outputs, dim=0).reshape(len(identities), 50, 7)
        self.policy_action_chunks += result.shape[0]
        self.by_purpose[purpose] += result.shape[0]
        return result

    def report(self) -> dict[str, Any]:
        return {
            "flow_chunks_sampled": self.flow_chunks_sampled,
            "euler_velocity_evaluations": self.euler_velocity_evaluations,
            "prefix_prefill_count": self.prefix_prefill_count,
            "policy_action_chunks": self.policy_action_chunks,
            "policy_action_chunks_by_purpose": dict(sorted(self.by_purpose.items())),
            "frozen_prefix_candidate_reuse": True,
            "flow_subbatch_size": self.inference_batch_size,
        }


def lightweight_state_token(module: torch.nn.Module) -> str:
    """Cheap mutation token for inner-loop ownership checks only."""

    digest = hashlib.sha256()
    for name, value in module.state_dict(keep_vars=True).items():
        digest.update(name.encode())
        digest.update(str(value._version).encode())
    return digest.hexdigest()


@torch.no_grad()
def fast_polyak_update(
    online: torch.nn.Module,
    target: torch.nn.Module,
    *,
    tau: float,
    target_name: str,
) -> dict[str, Any]:
    """Same fp32 Polyak formula without the development per-tensor SHA audit."""

    if tau != 0.005 or target.training or any(value.requires_grad for value in target.parameters()):
        raise RuntimeError("THROUGHPUT_V2_POLYAK_CONTRACT")
    online_parameters = dict(online.named_parameters())
    target_parameters = dict(target.named_parameters())
    if online_parameters.keys() != target_parameters.keys():
        raise RuntimeError("THROUGHPUT_V2_POLYAK_SCHEMA")
    for name, source in online_parameters.items():
        destination = target_parameters[name]
        if source.requires_grad:
            if source.dtype != torch.float32 or destination.dtype != torch.float32:
                raise TypeError("THROUGHPUT_V2_POLYAK_DTYPE")
            destination.mul_(1.0 - tau).add_(source.detach(), alpha=tau)
        elif not torch.equal(source, destination):
            raise RuntimeError(f"THROUGHPUT_V2_FROZEN_TARGET_DRIFT:{name}")
    for name, source in online.named_buffers():
        destination = dict(target.named_buffers())[name]
        if source.is_floating_point():
            if not torch.equal(source, destination):
                raise RuntimeError(f"THROUGHPUT_V2_BUFFER_DRIFT:{name}")
        else:
            destination.copy_(source)
    target.eval()
    return {
        "target": target_name,
        "tau": tau,
        "formula": "target=(1-tau)*target_before+tau*online_after_optimizer",
        "inner_loop_tensor_sha_disabled": True,
        "boundary_exact_audit_required": True,
    }
