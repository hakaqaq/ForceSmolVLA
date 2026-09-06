"""Differentiable mirror of the deployed HIL-SERL command acceptance path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


HILSERL_ACCEPTANCE_MAPPING_KIND = "hilserl_absolute_adapter_filter_leash"


def merge_robot_acceptance_context(
    local: Mapping[str, Any], pose_ack: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(local)
    robot = pose_ack.get("candidate_acceptance_mapping")
    if isinstance(robot, Mapping) and robot.get("mapping_kind") == ("hilserl_filter_leash"):
        result.update(robot)
        result["mapping_kind"] = HILSERL_ACCEPTANCE_MAPPING_KIND
        result["unavailable_reason"] = None
    return result


@dataclass(frozen=True)
class ControllerAcceptanceBatch:
    valid: Tensor
    control_source: Tensor
    decision_state7: Tensor
    upper_position3: Tensor
    upper_quaternion4: Tensor
    adapter_position3: Tensor
    adapter_quaternion4: Tensor
    translation_scale3: Tensor
    rotation_scale3: Tensor
    workspace_min3: Tensor
    workspace_max3: Tensor
    filter_position_before3: Tensor
    filter_quaternion_before4: Tensor
    actual_position3: Tensor
    actual_quaternion4: Tensor
    step_dt_s: Tensor
    filter_time_constant_s: Tensor
    translation_clip_positive3: Tensor
    translation_clip_negative3: Tensor
    rotation_clip_positive3: Tensor
    rotation_clip_negative3: Tensor
    delta_action_mean6: Tensor
    delta_action_std6: Tensor

    def select(self, indices: Tensor) -> "ControllerAcceptanceBatch":
        return ControllerAcceptanceBatch(
            **{name: getattr(self, name)[indices] for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True)
class AcceptedResidual:
    residual_k6: Tensor
    valid: Tensor


def _normalize_quaternion(value: Tensor) -> Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    lx, ly, lz, lw = left.unbind(-1)
    rx, ry, rz, rw = right.unbind(-1)
    return _normalize_quaternion(
        torch.stack(
            (
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ),
            dim=-1,
        )
    )


def _quaternion_inverse(value: Tensor) -> Tensor:
    value = _normalize_quaternion(value)
    return torch.cat((-value[..., :3], value[..., 3:]), dim=-1)


def _rpy_to_quaternion(rpy: Tensor) -> Tensor:
    roll, pitch, yaw = (rpy * 0.5).unbind(-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return _normalize_quaternion(
        torch.stack(
            (
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ),
            dim=-1,
        )
    )


def _quaternion_to_rpy(value: Tensor) -> Tensor:
    x, y, z, w = _normalize_quaternion(value).unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)


def _quaternion_matrix(value: Tensor) -> Tensor:
    x, y, z, w = _normalize_quaternion(value).unbind(-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(value.shape[:-1] + (3, 3))


def _quaternion_slerp(start: Tensor, target: Tensor, fraction: Tensor) -> Tensor:
    start = _normalize_quaternion(start)
    target = _normalize_quaternion(target)
    dot = (start * target).sum(-1, keepdim=True)
    target = torch.where(dot < 0.0, -target, target)
    dot = dot.abs().clamp(0.0, 1.0)
    # Keep the unused general branch away from acos(1), whose infinite
    # derivative can otherwise leak NaNs through torch.where at zero rotation.
    general_dot = dot.clamp(max=0.9995)
    angle = torch.acos(general_dot)
    denominator = torch.sin(angle).clamp_min(1.0e-12)
    general = (
        torch.sin((1.0 - fraction) * angle) / denominator * start
        + torch.sin(fraction * angle) / denominator * target
    )
    linear = start + fraction * (target - start)
    return _normalize_quaternion(torch.where(dot > 0.9995, linear, general))


def _quaternion_to_rotvec(value: Tensor) -> Tensor:
    quaternion = _normalize_quaternion(value)
    quaternion = torch.where(quaternion[..., 3:] < 0.0, -quaternion, quaternion)
    vector = quaternion[..., :3]
    squared_norm = vector.square().sum(dim=-1, keepdim=True)
    safe_norm = torch.sqrt(squared_norm.clamp_min(1.0e-16))
    scalar = quaternion[..., 3:].clamp_min(1.0e-12)
    general_scale = 2.0 * torch.atan2(safe_norm, scalar) / safe_norm
    small_scale = 2.0 / scalar
    scale = torch.where(squared_norm > 1.0e-16, general_scale, small_scale)
    return vector * scale


def _rotvec_to_quaternion(value: Tensor) -> Tensor:
    squared_angle = value.square().sum(dim=-1, keepdim=True)
    safe_angle = torch.sqrt(squared_angle.clamp_min(1.0e-16))
    half = safe_angle * 0.5
    general_scale = torch.sin(half) / safe_angle
    small_scale = 0.5 - squared_angle / 48.0
    scale = torch.where(squared_angle > 1.0e-16, general_scale, small_scale)
    general_scalar = torch.cos(half)
    small_scalar = 1.0 - squared_angle / 8.0
    scalar = torch.where(squared_angle > 1.0e-16, general_scalar, small_scalar)
    return _normalize_quaternion(torch.cat((value * scale, scalar), dim=-1))


def _wrap_to_pi(value: Tensor) -> Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def map_residual_to_controller_ack(
    candidate_residual6: Tensor,
    base_normalized_action6: Tensor,
    context: ControllerAcceptanceBatch,
) -> AcceptedResidual:
    """Apply the real upper adapter, workspace, filter and leash equations."""

    batch = int(candidate_residual6.shape[0])
    if candidate_residual6.shape != (batch, 6) or base_normalized_action6.shape != (
        batch,
        6,
    ):
        raise ValueError("FORCERFT_ACCEPTANCE_ACTION_SHAPE_INVALID")
    if context.valid.dtype != torch.bool or context.valid.shape != (batch,):
        raise ValueError("FORCERFT_ACCEPTANCE_CONTEXT_MASK_INVALID")
    if not bool(context.valid.any()):
        return AcceptedResidual(
            candidate_residual6[:, None, :].expand(-1, 3, -1),
            context.valid,
        )

    normalized_action = base_normalized_action6 + candidate_residual6
    delta = normalized_action * context.delta_action_std6 + context.delta_action_mean6
    target_position = context.decision_state7[:, :3] + delta[:, :3]
    target_rpy = context.decision_state7[:, 3:6] + delta[:, 3:6]
    target_quaternion = _rpy_to_quaternion(target_rpy)

    upper_rotation_delta = _quaternion_to_rotvec(
        _quaternion_multiply(_quaternion_inverse(context.upper_quaternion4), target_quaternion)
    )
    upper_translation = torch.clamp(
        (target_position - context.upper_position3) / context.translation_scale3,
        -1.0,
        1.0,
    )
    upper_rotation = torch.clamp(upper_rotation_delta / context.rotation_scale3, -1.0, 1.0)
    requested_position = torch.clamp(
        context.adapter_position3 + upper_translation * context.translation_scale3,
        context.workspace_min3,
        context.workspace_max3,
    )
    requested_quaternion = _quaternion_multiply(
        context.adapter_quaternion4,
        _rotvec_to_quaternion(upper_rotation * context.rotation_scale3),
    )

    alpha = 1.0 - torch.exp(-context.step_dt_s / context.filter_time_constant_s)
    filtered_position = context.filter_position_before3 + alpha * (
        requested_position - context.filter_position_before3
    )
    filtered_quaternion = _quaternion_slerp(
        context.filter_quaternion_before4, requested_quaternion, alpha
    )
    translation_error = context.actual_position3 - filtered_position
    limited_translation_error = torch.maximum(
        torch.minimum(translation_error, context.translation_clip_positive3),
        -context.translation_clip_negative3,
    )
    accepted_position = context.actual_position3 - limited_translation_error

    actual_quaternion = _normalize_quaternion(context.actual_quaternion4)
    desired_quaternion = torch.where(
        (actual_quaternion * filtered_quaternion).sum(-1, keepdim=True) < 0.0,
        -filtered_quaternion,
        filtered_quaternion,
    )
    error_quaternion = _quaternion_multiply(
        _quaternion_inverse(actual_quaternion), desired_quaternion
    )
    actual_rotation = _quaternion_matrix(actual_quaternion)
    rotation_error = -torch.bmm(actual_rotation, error_quaternion[:, :3, None]).squeeze(-1)
    limited_rotation_error = torch.maximum(
        torch.minimum(rotation_error, context.rotation_clip_positive3),
        -context.rotation_clip_negative3,
    )
    local_vector = -torch.bmm(
        actual_rotation.transpose(1, 2), limited_rotation_error[:, :, None]
    ).squeeze(-1)
    local_norm = local_vector.norm(dim=-1, keepdim=True)
    local_vector = local_vector / torch.maximum(local_norm, torch.ones_like(local_norm))
    local_scalar = torch.sqrt((1.0 - local_vector.square().sum(-1, keepdim=True)).clamp_min(0.0))
    accepted_quaternion = _quaternion_multiply(
        actual_quaternion, torch.cat((local_vector, local_scalar), dim=-1)
    )

    accepted_delta = torch.cat(
        (
            accepted_position - context.decision_state7[:, :3],
            _wrap_to_pi(_quaternion_to_rpy(accepted_quaternion) - context.decision_state7[:, 3:6]),
        ),
        dim=-1,
    )
    accepted_normalized = (accepted_delta - context.delta_action_mean6) / context.delta_action_std6
    accepted_residual = accepted_normalized - base_normalized_action6
    finite = torch.isfinite(accepted_residual).all(dim=1)

    target_distance = (target_position - context.upper_position3).norm(dim=1)
    target_angle = _quaternion_to_rotvec(
        _quaternion_multiply(_quaternion_inverse(context.upper_quaternion4), target_quaternion)
    ).norm(dim=1)
    policy_guard_valid = (target_distance <= 0.08) & (
        target_angle <= torch.deg2rad(target_angle.new_tensor(25.0))
    )
    human_source = context.control_source.squeeze(1) > 0.5
    accepted_valid = context.valid & finite & (human_source | policy_guard_valid)
    residual_k6 = accepted_residual[:, None, :].expand(-1, 3, -1)
    return AcceptedResidual(residual_k6, accepted_valid)
