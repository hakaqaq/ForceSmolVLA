"""Zero-update audit helpers for the Stage-2 gripper-domain boundary."""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import traceback
from typing import Any

import numpy as np
import torch

from forcesmolvla.action_delta import decode_binary_gripper_width


EXPECTED_PUBLIC_REJECTION = (
    "model gripper candidate is outside the frozen [-0.01,0.095] m tolerance"
)


def tensor_sha256(value: torch.Tensor | np.ndarray) -> str:
    array = (
        value.detach().cpu().contiguous().numpy()
        if isinstance(value, torch.Tensor)
        else np.ascontiguousarray(value)
    )
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def global_rng_digest() -> dict[str, Any]:
    result = {
        "python": hashlib.sha256(pickle.dumps(random.getstate(), protocol=5)).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=5)).hexdigest(),
        "torch_cpu": tensor_sha256(torch.get_rng_state()),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        result["torch_cuda"] = [tensor_sha256(state) for state in torch.cuda.get_rng_state_all()]
    else:
        result["torch_cuda"] = []
    return result


def gripper_domain_layers(
    flow_action7: torch.Tensor,
    *,
    delta_action_mean7: torch.Tensor,
    delta_action_std7: torch.Tensor,
) -> dict[str, Any]:
    """Describe one normalized Flow action without constructing a replacement."""
    if tuple(flow_action7.shape) != (7,) or flow_action7.dtype != torch.float32:
        raise ValueError("AUDIT_FLOW_ACTION_MUST_BE_FLOAT32_7")
    if tuple(delta_action_mean7.shape) != (7,) or tuple(delta_action_std7.shape) != (7,):
        raise ValueError("AUDIT_NORMALIZER_MUST_BE_7D")
    if not torch.all(torch.isfinite(flow_action7)) or not torch.all(torch.isfinite(delta_action_std7)):
        raise ValueError("AUDIT_NONFINITE_ACTION_OR_NORMALIZER")
    continuous = (
        flow_action7 * delta_action_std7 + delta_action_mean7
    ).detach().cpu().numpy().astype(np.float64)
    before = global_rng_digest()
    valid = True
    failure_code = None
    decoded = None
    caught_traceback = None
    try:
        decoded = decode_binary_gripper_width(continuous[None, :])[0]
    except ValueError as error:
        if str(error) != EXPECTED_PUBLIC_REJECTION:
            raise
        valid = False
        failure_code = "PUBLIC_GRIPPER_CANDIDATE_OUTSIDE_FROZEN_TOLERANCE"
        caught_traceback = traceback.format_exc()
    after = global_rng_digest()
    if before != after:
        raise RuntimeError("DETACHED_PUBLIC_VALIDITY_AUDIT_CONSUMED_GLOBAL_RNG")
    normalized_endpoint = None
    if decoded is not None:
        normalized_endpoint = float(
            (decoded[6] - float(delta_action_mean7[6].cpu()))
            / float(delta_action_std7[6].cpu())
        )
    return {
        "valid": valid,
        "failure_code": failure_code,
        "g_flow_normalized": float(flow_action7[6].detach().cpu()),
        "g_unnormalized_continuous_width_m": float(continuous[6]),
        "g_public_decoded_endpoint_m": None if decoded is None else float(decoded[6]),
        "g_critic_normalized": normalized_endpoint,
        "public_input_action7_sha256": tensor_sha256(continuous),
        "public_output_action7_sha256": None if decoded is None else tensor_sha256(decoded),
        "public_exception_traceback": caught_traceback,
        "global_rng_before": before,
        "global_rng_after": after,
        "global_rng_unchanged": True,
        "replacement_action_created": False,
        "clipping_or_resampling": False,
    }


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
