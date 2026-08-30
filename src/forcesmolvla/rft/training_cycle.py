"""Reusable mechanics for one 2-Critic:1-Actor ForceRFT training cycle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


@dataclass
class SerializableUniqueSampler:
    """Independent uniform-without-replacement batches from a fixed population."""

    name: str
    population: tuple[int, ...]
    generator: torch.Generator
    draws: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.population or len(set(self.population)) != len(self.population):
            raise ValueError("G5_SAMPLER_NAME_OR_POPULATION_INVALID")

    def draw(self, count: int) -> list[int]:
        if count < 1 or count > len(self.population):
            raise ValueError("G5_UNIQUE_SAMPLER_COUNT_INVALID")
        order = torch.randperm(len(self.population), generator=self.generator)[:count].tolist()
        result = [self.population[index] for index in order]
        if len(set(result)) != count:
            raise RuntimeError("G5_SAMPLER_DUPLICATE_WITHIN_BATCH")
        self.draws += 1
        return result

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "population": self.population,
            "generator_state": self.generator.get_state(),
            "draws": self.draws,
        }


@dataclass
class SerializableReplacementSampler:
    """Independent empirical macro bootstrap over whole Kx7 population rows."""

    name: str
    population_size: int
    generator: torch.Generator
    draws: int = 0

    def draw(self, count: int) -> list[int]:
        if count < 1 or self.population_size < 1:
            raise ValueError("G5_REPLACEMENT_SAMPLER_COUNT_INVALID")
        result = torch.randint(
            self.population_size, (count,), generator=self.generator
        ).tolist()
        self.draws += 1
        return result

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "population_size": self.population_size,
            "generator_state": self.generator.get_state(),
            "draws": self.draws,
        }


def tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def generator_state_sha256(generator: torch.Generator) -> str:
    return tensor_sha256(generator.get_state())


def build_stage2_optimizers(actor: nn.Module, q1: nn.Module, q2: nn.Module):
    """Create exactly the authorized fresh Stage-2 optimizer ownership."""

    from forcesmolvla.router_training import _no_decay_parameter_names

    actor_named = dict(actor.named_parameters())
    if not actor_named or not all(parameter.requires_grad for parameter in actor_named.values()):
        raise RuntimeError("G5_ACTOR_TRAINABILITY_INVALID")
    no_decay = _no_decay_parameter_names(actor)
    decay = set(actor_named) - no_decay
    if decay & no_decay or decay | no_decay != set(actor_named):
        raise RuntimeError("G5_ACTOR_OPTIMIZER_PARTITION_INVALID")
    actor_optimizer = torch.optim.AdamW(
        [
            {"params": [actor_named[name] for name in sorted(decay)], "weight_decay": 1e-10},
            {"params": [actor_named[name] for name in sorted(no_decay)], "weight_decay": 0.0},
        ],
        lr=1e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    q1_trainable = [(name, parameter) for name, parameter in q1.named_parameters() if parameter.requires_grad]
    q2_trainable = [(name, parameter) for name, parameter in q2.named_parameters() if parameter.requires_grad]
    if not q1_trainable or not q2_trainable:
        raise RuntimeError("G5_CRITIC_TRAINABILITY_INVALID")
    critic_parameters = [parameter for _name, parameter in q1_trainable + q2_trainable]
    critic_optimizer = torch.optim.Adam(
        critic_parameters,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lambda _step: 1.0)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lambda _step: 1.0)

    actor_ids = [id(parameter) for group in actor_optimizer.param_groups for parameter in group["params"]]
    critic_ids = [id(parameter) for group in critic_optimizer.param_groups for parameter in group["params"]]
    q1_ids = [id(parameter) for _name, parameter in q1_trainable]
    q2_ids = [id(parameter) for _name, parameter in q2_trainable]
    forbidden_ids = {
        id(parameter)
        for critic in (q1, q2)
        for backbone in (critic.camera1_backbone, critic.camera2_backbone)
        for parameter in backbone.parameters()
    }
    if (
        len(actor_ids) != len(set(actor_ids))
        or len(critic_ids) != len(set(critic_ids))
        or set(actor_ids) & set(critic_ids)
        or set(q1_ids) & set(q2_ids)
        or set(critic_ids) & forbidden_ids
        or set(critic_ids) != set(q1_ids) | set(q2_ids)
        or set(actor_ids) != {id(parameter) for parameter in actor.parameters() if parameter.requires_grad}
    ):
        raise RuntimeError("G5_OPTIMIZER_PARAMETER_OWNERSHIP_FAILED")

    ownership = {
        "actor_optimizer": {
            "type": "AdamW",
            "trainable_tensor_count": len(actor_ids),
            "parameter_count": sum(parameter.numel() for parameter in actor_named.values()),
            "decay_tensor_count": len(decay),
            "no_decay_tensor_count": len(no_decay),
            "decay_parameter_count": sum(actor_named[name].numel() for name in decay),
            "no_decay_parameter_count": sum(actor_named[name].numel() for name in no_decay),
            "group_name_sha256": hashlib.sha256(
                "".join(
                    f"{group}\0{name}\n"
                    for group, names in (("decay", decay), ("no_decay", no_decay))
                    for name in sorted(names)
                ).encode()
            ).hexdigest(),
        },
        "critic_optimizer": {
            "type": "Adam",
            "trainable_tensor_count": len(critic_ids),
            "q1_tensor_count": len(q1_ids),
            "q2_tensor_count": len(q2_ids),
            "parameter_count": sum(parameter.numel() for parameter in critic_parameters),
        },
        "actor_critic_parameter_id_intersection": 0,
        "q1_q2_parameter_id_intersection": 0,
        "frozen_backbone_in_critic_optimizer": 0,
        "targets_in_any_optimizer": 0,
        "each_approved_trainable_parameter_exactly_once": True,
    }
    return actor_optimizer, critic_optimizer, actor_scheduler, critic_scheduler, ownership


def global_gradient_norm(parameters: Iterable[nn.Parameter]) -> Tensor:
    gradients = [parameter.grad.detach().float() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.zeros((), dtype=torch.float32)
    device = gradients[0].device
    return torch.sqrt(
        torch.stack([gradient.square().sum().to(device=device) for gradient in gradients]).sum()
    )


def gradients_finite(parameters: Iterable[nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


@torch.no_grad()
def polyak_update_verified(
    online: nn.Module,
    target: nn.Module,
    *,
    tau: float,
    target_name: str,
) -> dict:
    """Apply one fp32 Polyak update and verify each target tensor immediately."""

    if tau != 0.005 or target.training or any(parameter.requires_grad for parameter in target.parameters()):
        raise RuntimeError("G5_POLYAK_TARGET_OR_TAU_INVALID")
    online_parameters = dict(online.named_parameters())
    target_parameters = dict(target.named_parameters())
    online_buffers = dict(online.named_buffers())
    target_buffers = dict(target.named_buffers())
    if online_parameters.keys() != target_parameters.keys() or online_buffers.keys() != target_buffers.keys():
        raise RuntimeError("G5_POLYAK_STATE_SCHEMA_MISMATCH")

    before_sha = module_state_sha256(target)
    tensor_records = []
    maximum_error = 0.0
    frozen_changed = 0
    for name, online_value in online_parameters.items():
        target_value = target_parameters[name]
        before = target_value.detach().clone()
        if online_value.requires_grad:
            if online_value.dtype != torch.float32 or target_value.dtype != torch.float32:
                raise TypeError("G5_POLYAK_TRAINABLE_PARAMETER_NOT_FP32")
            expected = before.mul(1.0 - tau).add(online_value.detach(), alpha=tau)
            target_value.mul_(1.0 - tau).add_(online_value.detach(), alpha=tau)
            error = float((target_value - expected).abs().max().cpu())
        else:
            if not torch.equal(target_value, online_value):
                raise RuntimeError(f"G5_FROZEN_ONLINE_TARGET_DRIFT:{name}")
            error = 0.0
            frozen_changed += int(not torch.equal(before, target_value))
        maximum_error = max(maximum_error, error)
        tensor_records.append(
            {
                "name": name,
                "kind": "trainable_parameter" if online_value.requires_grad else "frozen_parameter",
                "before_sha256": tensor_sha256(before),
                "online_sha256": tensor_sha256(online_value),
                "after_sha256": tensor_sha256(target_value),
                "formula_max_abs_error": error,
            }
        )
    frozen_buffer_changed = 0
    for name, online_value in online_buffers.items():
        target_value = target_buffers[name]
        before = target_value.detach().clone()
        if online_value.is_floating_point():
            if not torch.equal(target_value, online_value):
                raise RuntimeError(f"G5_FROZEN_FLOAT_BUFFER_DRIFT:{name}")
            error = 0.0
            kind = "frozen_floating_buffer_preserved_exact"
            frozen_buffer_changed += int(not torch.equal(before, target_value))
        else:
            target_value.copy_(online_value)
            error = 0.0
            kind = "nonfloating_buffer_copy"
        maximum_error = max(maximum_error, error)
        tensor_records.append(
            {
                "name": name,
                "kind": kind,
                "before_sha256": tensor_sha256(before),
                "online_sha256": tensor_sha256(online_value),
                "after_sha256": tensor_sha256(target_value),
                "formula_max_abs_error": error,
            }
        )
    target.eval()
    if maximum_error != 0.0 or frozen_changed or frozen_buffer_changed or any(parameter.requires_grad for parameter in target.parameters()):
        raise RuntimeError("G5_POLYAK_FORMULA_OR_TARGET_OWNERSHIP_FAILED")
    return {
        "target": target_name,
        "tau": tau,
        "before_state_sha256": before_sha,
        "after_state_sha256": module_state_sha256(target),
        "maximum_formula_abs_error": maximum_error,
        "frozen_parameter_changed_count": frozen_changed,
        "frozen_floating_buffer_changed_count": frozen_buffer_changed,
        "tensor_count": len(tensor_records),
        "tensors": tensor_records,
    }


def calql_unclipped_details(
    q_dataset: Tensor,
    q_candidates: Tensor,
    mc_return: Tensor,
    *,
    temperature: float,
) -> dict[str, Tensor]:
    """Reporting sidecar for the unclipped G4 finite-candidate objective."""

    for value in (q_dataset, q_candidates, mc_return):
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise FloatingPointError("G5_CALQL_DETAIL_INPUT_INVALID")
    calibrated = torch.maximum(q_candidates, mc_return[:, None])
    values = torch.cat((q_dataset[:, None], calibrated), dim=1)
    lse = temperature * (
        torch.logsumexp(values / temperature, dim=1)
        - math.log(values.shape[1])
    )
    difference = lse - q_dataset
    return {
        "difference": difference,
        "mc_lower_bound_activation": q_candidates < mc_return[:, None],
        "calibrated_candidates": calibrated,
    }


def parameter_change_matrix(before: dict[str, str], after: dict[str, str]) -> dict[str, bool]:
    if before.keys() != after.keys():
        raise ValueError("G5_CHANGE_MATRIX_KEYS_MISMATCH")
    return {name: before[name] != after[name] for name in before}


def ensure_all_gradients_none(*modules: nn.Module) -> None:
    if any(parameter.grad is not None for module in modules for parameter in module.parameters()):
        raise RuntimeError("G5_PENDING_GRADIENT_AT_CYCLE_BOUNDARY")


def optimizer_state_storage_independent(optimizer: torch.optim.Optimizer, q1: nn.Module, q2: nn.Module) -> bool:
    q1_ids = {id(parameter) for parameter in q1.parameters() if parameter.requires_grad}
    q2_ids = {id(parameter) for parameter in q2.parameters() if parameter.requires_grad}
    q1_storage = set()
    q2_storage = set()
    for parameter, state in optimizer.state.items():
        destination = q1_storage if id(parameter) in q1_ids else q2_storage if id(parameter) in q2_ids else None
        if destination is None:
            return False
        destination.update(
            value.untyped_storage().data_ptr()
            for value in state.values()
            if isinstance(value, Tensor)
        )
    return bool(q1_storage) and bool(q2_storage) and not (q1_storage & q2_storage)


# Stable production API for the materialization and update primitives formerly
# embedded in the historical single-cycle runner.
from forcesmolvla.rft.training_cycle_runtime import (  # noqa: E402
    FORBIDDEN_OPENS,
    DATASET,
    PARENT_ACTOR_CHECKPOINT,
    REWARD_BACKBONE_MANIFEST,
    REWARD_BACKBONE_PARAMETERS,
    FlowCounter,
    TrainData,
    actor_module_gradient_norms,
    actor_gradient_scale_probe,
    actor_update,
    capture_rng_states,
    critic_update,
    flow_microbatch_terms,
    install_open_audit,
    named_generator,
    parameter_group_gradient_norm,
    protected_snapshot,
    repeat_actor_batch,
    sample_policy_candidates,
    slice_actor_batch,
    verify_config,
)
