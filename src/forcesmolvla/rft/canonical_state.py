"""Canonical, process-independent hashing for G6 exact-resume evidence."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_record(value: Tensor) -> dict:
    tensor = value.detach().cpu().contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "kind": "tensor",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "byte_count": len(raw),
        "sha256": _sha256(raw),
    }


def ndarray_record(value: np.ndarray) -> dict:
    array = np.ascontiguousarray(value)
    raw = array.tobytes(order="C")
    return {
        "kind": "ndarray",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "byte_count": len(raw),
        "sha256": _sha256(raw),
    }


def canonicalize(value: Any) -> Any:
    """Encode values without dtype conversion, tolerance, or process-local IDs."""

    if isinstance(value, Tensor):
        return tensor_record(value)
    if isinstance(value, np.ndarray):
        return ndarray_record(value)
    if isinstance(value, np.generic):
        return canonicalize(value.item())
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return {"kind": "int", "decimal": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("G6_CANONICAL_NONFINITE_FLOAT")
        return {"kind": "float64", "hex": struct.pack(">d", value).hex()}
    if isinstance(value, Mapping):
        if any(not isinstance(key, (str, int)) for key in value):
            raise TypeError("G6_CANONICAL_MAPPING_KEY_INVALID")
        return {
            "kind": "mapping",
            "items": [
                [canonicalize(key), canonicalize(value[key])]
                for key in sorted(value, key=lambda item: (type(item).__name__, str(item)))
            ],
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [canonicalize(item) for item in value],
        }
    raise TypeError(f"G6_CANONICAL_TYPE_UNSUPPORTED:{type(value).__name__}")


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        canonicalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return _sha256(encoded)


def module_record(module_or_state: nn.Module | Mapping[str, Tensor]) -> dict:
    state = (
        module_or_state.state_dict()
        if isinstance(module_or_state, nn.Module)
        else module_or_state
    )
    tensors = {name: tensor_record(value) for name, value in sorted(state.items())}
    return {"tensors": tensors, "digest": canonical_digest(tensors)}


def optimizer_parameter_name_groups(
    optimizer: torch.optim.Optimizer,
    named_parameters: Mapping[str, nn.Parameter],
) -> list[list[str]]:
    by_id = {id(parameter): name for name, parameter in named_parameters.items()}
    groups = []
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            name = by_id.get(id(parameter))
            if name is None:
                raise RuntimeError("G6_OPTIMIZER_PARAMETER_NAME_NOT_FOUND")
            names.append(name)
        groups.append(names)
    flat = [name for group in groups for name in group]
    if len(flat) != len(set(flat)):
        raise RuntimeError("G6_OPTIMIZER_PARAMETER_NAME_DUPLICATE")
    return groups


def optimizer_record(
    optimizer_or_state: torch.optim.Optimizer | Mapping[str, Any],
    parameter_name_groups: Sequence[Sequence[str]],
) -> dict:
    state = (
        optimizer_or_state.state_dict()
        if isinstance(optimizer_or_state, torch.optim.Optimizer)
        else dict(optimizer_or_state)
    )
    groups = state.get("param_groups")
    states = state.get("state")
    if not isinstance(groups, list) or not isinstance(states, dict):
        raise ValueError("G6_OPTIMIZER_STATE_SCHEMA_INVALID")
    if len(groups) != len(parameter_name_groups):
        raise ValueError("G6_OPTIMIZER_GROUP_COUNT_MISMATCH")

    parameter_states = {}
    group_records = []
    seen_ids = set()
    for group_index, (group, names) in enumerate(zip(groups, parameter_name_groups, strict=True)):
        identifiers = group.get("params")
        if len(identifiers) != len(names):
            raise ValueError("G6_OPTIMIZER_GROUP_PARAMETER_COUNT_MISMATCH")
        hyperparameters = {key: value for key, value in group.items() if key != "params"}
        group_records.append({
            "group_index": group_index,
            "parameter_names": list(names),
            "hyperparameters": canonicalize(hyperparameters),
        })
        for identifier, name in zip(identifiers, names, strict=True):
            if identifier in seen_ids:
                raise ValueError("G6_OPTIMIZER_STATE_ID_REUSED")
            seen_ids.add(identifier)
            parameter_states[name] = (
                canonicalize(states[identifier])
                if identifier in states
                else {"kind": "missing_optimizer_state"}
            )
    if set(states) - seen_ids:
        raise ValueError("G6_OPTIMIZER_ORPHAN_STATE")
    record = {
        "groups": group_records,
        "parameter_states": dict(sorted(parameter_states.items())),
    }
    record["digest"] = canonical_digest(record)
    return record


def module_mode_and_grad_record(modules: Mapping[str, nn.Module]) -> dict:
    modes = {name: bool(module.training) for name, module in sorted(modules.items())}
    requires_grad = {
        f"{module_name}.{parameter_name}": bool(parameter.requires_grad)
        for module_name, module in sorted(modules.items())
        for parameter_name, parameter in module.named_parameters()
    }
    gradients = {
        f"{module_name}.{parameter_name}": parameter.grad is None
        for module_name, module in sorted(modules.items())
        for parameter_name, parameter in module.named_parameters()
    }
    if not all(gradients.values()):
        raise RuntimeError("G6_CYCLE_BOUNDARY_HAS_GRADIENT")
    return {
        "module_training_modes": modes,
        "requires_grad": requires_grad,
        "all_gradients_none": True,
    }


def training_state_payload(
    *,
    modules: Mapping[str, nn.Module | Mapping[str, Tensor]],
    actor_optimizer: torch.optim.Optimizer | Mapping[str, Any],
    critic_optimizer: torch.optim.Optimizer | Mapping[str, Any],
    actor_parameter_name_groups: Sequence[Sequence[str]],
    critic_parameter_name_groups: Sequence[Sequence[str]],
    actor_scheduler_state: Mapping[str, Any],
    critic_scheduler_state: Mapping[str, Any],
    sampler_states: Mapping[str, Any],
    rng_states: Mapping[str, Any],
    counters: Mapping[str, int],
    boundary_manifest: Mapping[str, Any],
    ownership_manifest: Mapping[str, Any],
) -> dict:
    sections = {
        "modules": {
            name: module_record(module) for name, module in sorted(modules.items())
        },
        "actor_optimizer": optimizer_record(
            actor_optimizer, actor_parameter_name_groups
        ),
        "critic_optimizer": optimizer_record(
            critic_optimizer, critic_parameter_name_groups
        ),
        "actor_scheduler": canonicalize(actor_scheduler_state),
        "critic_scheduler": canonicalize(critic_scheduler_state),
        "samplers": canonicalize(sampler_states),
        "rng": canonicalize(rng_states),
        "counters": canonicalize(dict(counters)),
        "boundary": canonicalize(dict(boundary_manifest)),
        "ownership": canonicalize(dict(ownership_manifest)),
    }
    section_digests = {
        name: canonical_digest(value) for name, value in sorted(sections.items())
    }
    return {
        "schema_version": "forcesmolvla_g6_canonical_training_state.v1",
        "comparison": {"rtol": 0.0, "atol": 0.0, "equal_nan": False},
        "sections": sections,
        "section_digests": section_digests,
        "training_state_digest": canonical_digest(section_digests),
    }


def assert_payload_exact(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> None:
    if left.get("training_state_digest") != right.get("training_state_digest"):
        left_sections = left.get("section_digests", {})
        right_sections = right.get("section_digests", {})
        mismatches = [
            name
            for name in sorted(set(left_sections) | set(right_sections))
            if left_sections.get(name) != right_sections.get(name)
        ]
        raise RuntimeError(f"G6_EXACT_PARITY_FAILED:{label}:sections={mismatches}")
    if left.get("sections") != right.get("sections"):
        raise RuntimeError(f"G6_DIGEST_COLLISION_OR_CANONICAL_DRIFT:{label}")
