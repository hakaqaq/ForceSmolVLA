"""Rotation representation helpers for online absolute actions."""

from __future__ import annotations

import numpy as np


ABSOLUTE_ACTION_ROTATION_REPRESENTATION = "rpy_xyz"


def quaternion_xyzw_to_rpy_xyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Convert finite xyzw quaternions to the model's ZYX RPY chart."""

    value = np.asarray(quaternion_xyzw, dtype=np.float64)
    if value.shape[-1:] != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must be finite and end in shape [4]")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("quaternion cannot be zero")
    x, y, z, w = np.moveaxis(value / norm, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def rotation_vector_to_rpy_xyz(rotation_vector: np.ndarray) -> np.ndarray:
    """Convert legacy online rotation vectors to the model's RPY chart."""

    value = np.asarray(rotation_vector, dtype=np.float64)
    if value.shape[-1:] != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("rotation vector must be finite and end in shape [3]")
    angle = np.linalg.norm(value, axis=-1, keepdims=True)
    scale = np.empty_like(angle)
    small = angle < 1.0e-12
    scale[small] = 0.5
    scale[~small] = np.sin(angle[~small] / 2.0) / angle[~small]
    quaternion = np.concatenate(
        (value * scale, np.cos(angle / 2.0)), axis=-1
    )
    return quaternion_xyzw_to_rpy_xyz(quaternion)


def legacy_absolute_action7_to_rpy_xyz(action7: np.ndarray) -> np.ndarray:
    """Convert only the rotation columns of a legacy absolute action."""

    result = np.asarray(action7, dtype=np.float64).copy()
    if result.shape[-1:] != (7,) or not np.all(np.isfinite(result)):
        raise ValueError("absolute action must be finite and end in shape [7]")
    result[..., 3:6] = rotation_vector_to_rpy_xyz(result[..., 3:6])
    return result
