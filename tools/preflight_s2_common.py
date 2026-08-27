"""Shared, mutation-free helpers for Stage-2 preflight tools."""

from __future__ import annotations

import hashlib


def module_state_dict_sha256(module) -> str:
    """Hash a module state_dict without converting BF16 values through NumPy."""

    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
