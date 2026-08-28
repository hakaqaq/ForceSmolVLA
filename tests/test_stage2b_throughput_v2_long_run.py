from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from forcesmolvla.rft.throughput_v2_long_run import (
    BoundedDecodedImageLRU,
    stable_draw_plan_sha256,
)


def _decode(payload: bytes) -> np.ndarray:
    value = int(payload.decode())
    return np.full((2, 2, 3), value, dtype=np.uint8)


def test_bounded_lru_evicts_and_never_exceeds_limit() -> None:
    cache = BoundedDecodedImageLRU(max_bytes=24, decode=_decode)
    first = cache.get(b"1")
    cache.get(b"2")
    assert cache.report()["decoded_cache_current_bytes"] == 24
    assert cache.get(b"1") is first
    cache.get(b"3")
    report = cache.report()
    assert report["decoded_cache_peak_bytes"] <= 24
    assert report["cache_evictions"] == 1
    assert report["resident_decoded_images"] == 2


def test_cache_rejects_nonfinite_size_contract() -> None:
    with pytest.raises(ValueError, match="THROUGHPUT_V2_CACHE_LIMIT"):
        BoundedDecodedImageLRU(max_bytes=0, decode=_decode)


def test_cache_reads_do_not_consume_any_rng() -> None:
    cache = BoundedDecodedImageLRU(max_bytes=24, decode=_decode)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    cache.get(b"1")
    cache.get(b"1")
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_draw_plan_digest_is_order_sensitive_and_stable() -> None:
    plan = [{"cycle": 1, "td": [1, 2], "actor": [3]}]
    assert stable_draw_plan_sha256(plan) == stable_draw_plan_sha256(plan)
    assert stable_draw_plan_sha256(plan) != stable_draw_plan_sha256(
        [{"cycle": 1, "td": [2, 1], "actor": [3]}]
    )
