"""ForceToken-MoE training updates and the P7 exact-two-pass oracle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any

import torch
from torch import nn

from .configuration_forcesmolvla import FORCE_TOKEN_MOE, FORCE_TOKEN_MOE_ADDITIVE
from .force_token import MOE_NUM_EXPERTS, RouterState


L_BALANCE_WEIGHT = 0.01
L_Z_WEIGHT = 0.001
EXPECTED_MICROBATCHES = 8
DEFAULT_TARGET_SAMPLES = 80_000
DEFAULT_EFFECTIVE_SAMPLES_PER_UPDATE = 4


def derive_optimizer_updates(target_samples: int, effective_samples_per_update: int) -> int:
    """Derive an exact update count from the primary sample budget."""
    if target_samples <= 0 or effective_samples_per_update <= 0:
        raise ValueError("sample budget and effective batch must be positive")
    updates, remainder = divmod(target_samples, effective_samples_per_update)
    if remainder:
        raise ValueError("sample budget must be divisible by effective samples per update")
    return updates


DEFAULT_DERIVED_OPTIMIZER_UPDATES = derive_optimizer_updates(
    DEFAULT_TARGET_SAMPLES, DEFAULT_EFFECTIVE_SAMPLES_PER_UPDATE
)


@dataclass(frozen=True)
class MoEMicrobatch:
    batch: dict[str, Any]
    noise7: torch.Tensor
    time: torch.Tensor
    identity: str

    def validate(self) -> None:
        actions = self.batch.get("action")
        valid = self.batch.get("action_valid_mask")
        if not isinstance(actions, torch.Tensor) or actions.ndim != 3 or actions.shape[-1] != 7:
            raise ValueError("microbatch action must be [B,H,7]")
        if self.noise7.shape != actions.shape or self.noise7.dtype != torch.float32:
            raise ValueError("microbatch noise7 must be fp32 and match action")
        if self.time.shape != (actions.shape[0],) or self.time.dtype != torch.float32:
            raise ValueError("microbatch time must be fp32 [B]")
        if not isinstance(valid, torch.Tensor) or valid.shape != actions.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("microbatch action_valid_mask must be bool [B,H]")
        if not self.identity:
            raise ValueError("microbatch identity is required")


@dataclass(frozen=True)
class TwoPassWindowStatistics:
    sum_probabilities: torch.Tensor
    route_counts: torch.Tensor
    valid_router_tokens: int
    valid_flow_features: int
    world_size: int

    @property
    def pbar(self) -> torch.Tensor:
        if self.valid_router_tokens == 0:
            return torch.zeros_like(self.sum_probabilities)
        return self.sum_probabilities / self.valid_router_tokens

    @property
    def rbar(self) -> torch.Tensor:
        if self.valid_router_tokens == 0:
            return torch.zeros_like(self.sum_probabilities)
        return self.route_counts.to(dtype=torch.float32) / self.valid_router_tokens

    @property
    def logged_balance(self) -> torch.Tensor:
        return MOE_NUM_EXPERTS * torch.sum(self.pbar * self.rbar)


@dataclass(frozen=True)
class MicrobatchLossTerms:
    total: torch.Tensor
    flow: torch.Tensor
    balance: torch.Tensor
    z: torch.Tensor


def _no_decay_parameter_names(policy: nn.Module) -> set[str]:
    modules = dict(policy.named_modules())
    result = set()
    for name, _parameter in policy.named_parameters():
        parent_name, _, leaf_name = name.rpartition(".")
        parent = modules[parent_name]
        parent_type = type(parent).__name__.lower()
        if (
            leaf_name == "bias"
            or leaf_name == "alpha"
            or isinstance(parent, nn.Embedding)
            or "norm" in parent_type
            or "embed" in name.lower()
            or name.endswith("learned_action_slot")
        ):
            result.add(name)
    return result


def build_sft_optimizer_and_scheduler(
    policy: nn.Module,
    *,
    derived_optimizer_updates: int = DEFAULT_DERIVED_OPTIMIZER_UPDATES,
):
    if not all(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("P7_OFFLINE_REQUIRES_ALL_PARAMETERS_TRAINABLE")
    named = dict(policy.named_parameters())
    no_decay_names = _no_decay_parameter_names(policy)
    decay_names = set(named) - no_decay_names
    if decay_names & no_decay_names or decay_names | no_decay_names != set(named):
        raise RuntimeError("P7_OPTIMIZER_GROUP_PARTITION_INVALID")
    optimizer = torch.optim.AdamW(
        [
            {"params": [named[name] for name in sorted(decay_names)], "weight_decay": 1e-10},
            {"params": [named[name] for name in sorted(no_decay_names)], "weight_decay": 0.0},
        ],
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    trainable_ids = [id(parameter) for parameter in named.values()]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise RuntimeError("P7_OPTIMIZER_PARAMETER_DUPLICATED")
    if set(optimizer_ids) != set(trainable_ids) or len(optimizer_ids) != len(trainable_ids):
        raise RuntimeError("P7_OPTIMIZER_PARAMETER_MISSING_OR_EXTRA")
    if derived_optimizer_updates <= 0:
        raise ValueError("derived optimizer updates must be positive")
    scheduler = policy.config.get_scheduler_preset().build(
        optimizer, derived_optimizer_updates
    )
    digest = hashlib.sha256()
    for group, names in (("decay", decay_names), ("no_decay", no_decay_names)):
        for name in sorted(names):
            digest.update(f"{group}\0{name}\n".encode())
    manifest = {
        "decay_parameter_count": sum(named[name].numel() for name in decay_names),
        "no_decay_parameter_count": sum(named[name].numel() for name in no_decay_names),
        "decay_tensor_count": len(decay_names),
        "no_decay_tensor_count": len(no_decay_names),
        "group_name_sha256": digest.hexdigest(),
        "trainable_tensor_count": len(trainable_ids),
        "optimizer_tensor_count": len(optimizer_ids),
        "each_trainable_parameter_exactly_once": True,
    }
    return optimizer, scheduler, manifest


def _distributed_sum(value: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value, 1
    result = value.clone()
    torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result, torch.distributed.get_world_size()


def collect_pass_a_statistics(
    router_states: list[RouterState], flow_feature_masks: list[torch.Tensor]
) -> TwoPassWindowStatistics:
    if not router_states or len(router_states) != len(flow_feature_masks):
        raise ValueError("Pass A router states and flow masks must be nonempty and aligned")
    device = router_states[0].probabilities_fp32.device
    sum_probabilities = torch.zeros(MOE_NUM_EXPERTS, dtype=torch.float32, device=device)
    route_counts = torch.zeros(MOE_NUM_EXPERTS, dtype=torch.int64, device=device)
    valid_router_tokens = torch.zeros((), dtype=torch.int64, device=device)
    valid_flow_features = torch.zeros((), dtype=torch.int64, device=device)
    for state, flow_mask in zip(router_states, flow_feature_masks, strict=True):
        state.validate()
        if state.logits_fp32.requires_grad or state.probabilities_fp32.requires_grad:
            raise RuntimeError("PASS_A_ROUTER_STATE_MUST_BE_NO_GRAD")
        if flow_mask.dtype != torch.bool:
            raise ValueError("flow feature mask must be bool")
        valid_probabilities = state.probabilities_fp32[state.valid_mask]
        valid_routes = state.route_ids[state.valid_mask]
        sum_probabilities += valid_probabilities.sum(dim=0)
        route_counts += torch.bincount(valid_routes, minlength=MOE_NUM_EXPERTS)
        valid_router_tokens += state.valid_mask.sum()
        valid_flow_features += flow_mask.sum()

    sum_probabilities, world_size_a = _distributed_sum(sum_probabilities)
    route_counts, world_size_b = _distributed_sum(route_counts)
    valid_router_tokens, world_size_c = _distributed_sum(valid_router_tokens)
    valid_flow_features, world_size_d = _distributed_sum(valid_flow_features)
    if len({world_size_a, world_size_b, world_size_c, world_size_d}) != 1:
        raise RuntimeError("DISTRIBUTED_WORLD_SIZE_DRIFT")
    token_count = int(valid_router_tokens.item())
    route_count = int(route_counts.sum().item())
    if route_count != token_count:
        raise RuntimeError(
            f"GLOBAL_ROUTER_TOKEN_ACCOUNTING_MISMATCH: routes={route_count}, valid={token_count}"
        )
    return TwoPassWindowStatistics(
        sum_probabilities=sum_probabilities.detach(),
        route_counts=route_counts.detach(),
        valid_router_tokens=token_count,
        valid_flow_features=int(valid_flow_features.item()),
        world_size=world_size_a,
    )


def microbatch_two_pass_terms(
    flow_losses: torch.Tensor,
    router_state: RouterState,
    statistics: TwoPassWindowStatistics,
) -> MicrobatchLossTerms:
    router_state.validate()
    world_size = statistics.world_size
    if statistics.valid_flow_features == 0:
        flow = flow_losses.sum() * 0.0
    else:
        flow = world_size * flow_losses.sum() / statistics.valid_flow_features
    if statistics.valid_router_tokens == 0:
        balance = router_state.probabilities_fp32.sum() * 0.0
        z_loss = router_state.logits_fp32.sum() * 0.0
    else:
        probabilities = router_state.probabilities_fp32[router_state.valid_mask]
        logits = router_state.logits_fp32[router_state.valid_mask]
        balance = (
            world_size
            * MOE_NUM_EXPERTS
            * torch.sum(
                statistics.rbar.to(device=probabilities.device)
                * probabilities.sum(dim=0)
                / statistics.valid_router_tokens
            )
        )
        z_loss = (
            world_size
            * torch.sum(torch.logsumexp(logits, dim=-1).square())
            / statistics.valid_router_tokens
        )
    total = flow + L_BALANCE_WEIGHT * balance + L_Z_WEIGHT * z_loss
    return MicrobatchLossTerms(total=total, flow=flow, balance=balance, z=z_loss)


def two_pass_optimizer_update(
    policy: nn.Module,
    microbatches: list[MoEMicrobatch],
    optimizer: torch.optim.Optimizer,
    *,
    oracle_mode: bool = False,
    scheduler=None,
    expected_microbatches: int = EXPECTED_MICROBATCHES,
    grad_clip_norm: float = 10.0,
) -> dict:
    if not oracle_mode:
        raise RuntimeError("EXACT_TWO_PASS_IS_P7_ACCEPTANCE_ORACLE_ONLY")
    if getattr(policy.config, "force_variant", None) not in {
        FORCE_TOKEN_MOE,
        FORCE_TOKEN_MOE_ADDITIVE,
    }:
        raise RuntimeError("TWO_PASS_UPDATE_REQUIRES_MOE_VARIANT")
    if len(microbatches) != expected_microbatches:
        raise ValueError(
            f"two-pass window requires {expected_microbatches} microbatches, got {len(microbatches)}"
        )
    for microbatch in microbatches:
        microbatch.validate()
    identities = [microbatch.identity for microbatch in microbatches]
    if len(set(identities)) != len(identities):
        raise ValueError("two-pass microbatch identities must be unique")

    optimizer.zero_grad(set_to_none=True)
    parameter_versions = {name: parameter._version for name, parameter in policy.named_parameters()}
    pass_a_states = []
    flow_masks = []
    for microbatch in microbatches:
        device_type = microbatch.noise7.device.type
        with torch.no_grad(), torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type in {"cpu", "cuda"},
        ):
            state = policy.router_pass_a(microbatch.batch)
        pass_a_states.append(
            RouterState(
                logits_fp32=state.logits_fp32.detach(),
                probabilities_fp32=state.probabilities_fp32.detach(),
                route_ids=state.route_ids.detach(),
                valid_mask=state.valid_mask.detach(),
            )
        )
        valid = microbatch.batch["action_valid_mask"]
        active7 = torch.arange(32, device=valid.device).view(1, 1, 32) < 7
        flow_masks.append(valid.unsqueeze(-1) & active7)
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise RuntimeError("PASS_A_CREATED_PARAMETER_GRADIENT")
    if any(
        parameter._version != parameter_versions[name]
        for name, parameter in policy.named_parameters()
    ):
        raise RuntimeError("PARAMETER_UPDATED_BETWEEN_PASS_A_AND_PASS_B")
    statistics = collect_pass_a_statistics(pass_a_states, flow_masks)

    flow_value = balance_value = z_value = total_value = 0.0
    max_router_probability_replay_error = 0.0
    for index, (microbatch, pass_a_state) in enumerate(
        zip(microbatches, pass_a_states, strict=True)
    ):
        device_type = microbatch.noise7.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type in {"cpu", "cuda"},
        ):
            flow_losses, feature_mask, pass_b_state = policy.forward_training_terms(
                microbatch.batch,
                noise=microbatch.noise7,
                time=microbatch.time,
            )
            if not torch.equal(feature_mask, flow_masks[index]):
                raise RuntimeError("PASS_A_B_FLOW_MASK_MISMATCH")
            if not torch.equal(pass_b_state.valid_mask, pass_a_state.valid_mask):
                raise RuntimeError("PASS_A_B_VALID_MASK_MISMATCH")
            if not torch.equal(pass_b_state.route_ids, pass_a_state.route_ids):
                raise RuntimeError("PASS_A_B_ROUTE_MISMATCH")
            replay_error = torch.max(
                torch.abs(pass_b_state.probabilities_fp32 - pass_a_state.probabilities_fp32)
            )
            max_router_probability_replay_error = max(
                max_router_probability_replay_error, float(replay_error.detach().cpu())
            )
            if replay_error != 0:
                raise RuntimeError(
                    "PASS_A_B_ROUTER_PROBABILITY_NOT_BITWISE_EQUAL: "
                    f"max_abs={float(replay_error.detach())}"
                )
            terms = microbatch_two_pass_terms(flow_losses, pass_b_state, statistics)
        terms.total.backward()
        flow_value += float(terms.flow.detach().cpu())
        balance_value += float(terms.balance.detach().cpu())
        z_value += float(terms.z.detach().cpu())
        total_value += float(terms.total.detach().cpu())

    gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("NONFINITE_TWO_PASS_GRADIENT_NORM")
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {
        "microbatch_count": len(microbatches),
        "microbatch_identities": identities,
        "world_size": statistics.world_size,
        "valid_router_tokens": statistics.valid_router_tokens,
        "valid_flow_features": statistics.valid_flow_features,
        "pbar": statistics.pbar.cpu().tolist(),
        "rbar": statistics.rbar.cpu().tolist(),
        "route_counts": statistics.route_counts.cpu().tolist(),
        "logged_balance": float(statistics.logged_balance.cpu()),
        "backward_flow_sum": flow_value,
        "backward_balance_sum": balance_value,
        "backward_z_sum": z_value,
        "backward_total_sum": total_value,
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "max_router_probability_replay_error": max_router_probability_replay_error,
        "optimizer_steps": 1,
        "scheduler_steps": int(scheduler is not None),
    }


def single_pass_optimizer_update(
    policy: nn.Module,
    microbatch: MoEMicrobatch,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler=None,
    grad_clip_norm: float = 10.0,
) -> dict:
    """Official-style single-GPU SFT update with in-batch MoE auxiliary losses."""
    if getattr(policy.config, "force_variant", None) not in {
        FORCE_TOKEN_MOE,
        FORCE_TOKEN_MOE_ADDITIVE,
    }:
        raise RuntimeError("SINGLE_PASS_UPDATE_REQUIRES_MOE_VARIANT")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("SINGLE_PASS_BATCH_LOCAL_REQUIRES_ONE_PROCESS")
    microbatch.validate()
    optimizer.zero_grad(set_to_none=True)
    device_type = microbatch.noise7.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type in {"cpu", "cuda"},
    ):
        flow_losses, feature_mask, router_state = policy.forward_single_pass_training_terms(
            microbatch.batch,
            noise=microbatch.noise7,
            time=microbatch.time,
        )
        detached_state = RouterState(
            logits_fp32=router_state.logits_fp32.detach(),
            probabilities_fp32=router_state.probabilities_fp32.detach(),
            route_ids=router_state.route_ids.detach(),
            valid_mask=router_state.valid_mask.detach(),
        )
        statistics = collect_pass_a_statistics([detached_state], [feature_mask])
        terms = microbatch_two_pass_terms(flow_losses, router_state, statistics)
    terms.total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("NONFINITE_SINGLE_PASS_GRADIENT_NORM")
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {
        "training_update_algorithm": "single_pass_batch_local",
        "microbatch_count": 1,
        "microbatch_identities": [microbatch.identity],
        "world_size": 1,
        "valid_router_tokens": statistics.valid_router_tokens,
        "valid_flow_features": statistics.valid_flow_features,
        "pbar": statistics.pbar.cpu().tolist(),
        "rbar": statistics.rbar.cpu().tolist(),
        "route_counts": statistics.route_counts.cpu().tolist(),
        "logged_balance": float(statistics.logged_balance.cpu()),
        "backward_flow_sum": float(terms.flow.detach().cpu()),
        "backward_balance_sum": float(terms.balance.detach().cpu()),
        "backward_z_sum": float(terms.z.detach().cpu()),
        "backward_total_sum": float(terms.total.detach().cpu()),
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "optimizer_steps": 1,
        "scheduler_steps": int(scheduler is not None),
    }


class SerializableUniformSampler:
    def __init__(self, eligible_indices: list[int] | tuple[int, ...], *, seed: int):
        if not eligible_indices:
            raise ValueError("eligible sampler indices cannot be empty")
        self.eligible_indices = tuple(int(index) for index in eligible_indices)
        self.seed = int(seed)
        self.cursor = 0
        self._rng = random.Random(self.seed)

    def draw(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("sampler draw count must be positive")
        result = [self.eligible_indices[self._rng.randrange(len(self.eligible_indices))] for _ in range(count)]
        self.cursor += count
        return result

    def state_dict(self) -> dict:
        return {
            "eligible_indices": self.eligible_indices,
            "seed": self.seed,
            "cursor": self.cursor,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        if tuple(state["eligible_indices"]) != self.eligible_indices or int(state["seed"]) != self.seed:
            raise RuntimeError("SAMPLER_STATE_BINDING_MISMATCH")
        self.cursor = int(state["cursor"])
        self._rng.setstate(state["rng_state"])
