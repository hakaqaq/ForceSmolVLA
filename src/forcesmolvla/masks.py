"""Active-feature and horizon masks for 7D-to-32D SmolVLA packing."""

from __future__ import annotations

import numpy as np


def pack_active_features(values: np.ndarray, *, maximum_dim: int = 32) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[-1] > maximum_dim:
        raise ValueError("active feature dimension exceeds maximum_dim")
    packed = np.zeros((*values.shape[:-1], maximum_dim), dtype=values.dtype)
    packed[..., : values.shape[-1]] = values
    return packed


def action_masks(action_valid_mask: np.ndarray, *, active_dim: int = 7, maximum_dim: int = 32):
    valid = np.asarray(action_valid_mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("action_valid_mask must have shape [B,H]")
    if not 0 < active_dim <= maximum_dim:
        raise ValueError("active_dim must be in (0, maximum_dim]")
    feature = np.zeros((*valid.shape, maximum_dim), dtype=bool)
    feature[..., :active_dim] = valid[..., None]
    return {
        "action_valid_mask": valid,
        "action_feature_mask": feature,
        "flow_valid_mask": feature.copy(),
        "suffix_valid_mask": valid.copy(),
    }
