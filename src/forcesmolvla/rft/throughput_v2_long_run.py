"""Bounded, rebuildable data cache for the Stage-2 throughput-v2 worker."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import threading
import time
import types
from typing import Any, Iterator

import numpy as np


def process_rss_bytes() -> int:
    """Read current Linux RSS without adding a runtime dependency."""

    status = Path("/proc/self/status").read_text()
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("THROUGHPUT_V2_RSS_UNAVAILABLE")


class BoundedDecodedImageLRU:
    """Thread-safe decoded-image LRU with an exact byte ceiling."""

    def __init__(self, max_bytes: int, decode) -> None:
        if max_bytes < 1:
            raise ValueError("THROUGHPUT_V2_CACHE_LIMIT")
        self.max_bytes = int(max_bytes)
        self.decode_source = decode
        self.values: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self.current_bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.decode_calls = 0
        self.unique_decoded_bytes = 0
        self._seen_sizes: dict[bytes, int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(payload: bytes) -> bytes:
        return hashlib.sha256(payload).digest()

    def get(self, payload: bytes) -> np.ndarray:
        key = self._key(payload)
        with self._lock:
            cached = self.values.get(key)
            if cached is not None:
                self.values.move_to_end(key)
                self.hits += 1
                return cached
            self.misses += 1
        decoded = self.decode_source(payload)
        if decoded.nbytes > self.max_bytes:
            raise MemoryError("THROUGHPUT_V2_SINGLE_IMAGE_EXCEEDS_CACHE")
        with self._lock:
            cached = self.values.get(key)
            if cached is not None:
                self.values.move_to_end(key)
                self.hits += 1
                return cached
            while self.values and self.current_bytes + decoded.nbytes > self.max_bytes:
                _old_key, old = self.values.popitem(last=False)
                self.current_bytes -= old.nbytes
                self.evictions += 1
            self.values[key] = decoded
            self.current_bytes += decoded.nbytes
            if key not in self._seen_sizes:
                self._seen_sizes[key] = int(decoded.nbytes)
                self.unique_decoded_bytes += int(decoded.nbytes)
            self.peak_bytes = max(self.peak_bytes, self.current_bytes)
            self.decode_calls += 1
            return decoded

    def report(self) -> dict[str, Any]:
        requests = self.hits + self.misses
        return {
            "decoded_cache_max_bytes": self.max_bytes,
            "decoded_cache_current_bytes": self.current_bytes,
            "decoded_cache_peak_bytes": self.peak_bytes,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_hit_rate": self.hits / requests if requests else 0.0,
            "cache_evictions": self.evictions,
            "decode_calls": self.decode_calls,
            "total_unique_images": len(self._seen_sizes),
            "total_unique_decoded_image_bytes": self.unique_decoded_bytes,
            "resident_decoded_images": len(self.values),
        }


class BoundedTrainingDataCache:
    """Read Parquet once; predecode only current/next observations per batch."""

    def __init__(self, data, *, max_bytes: int, prefetch_workers: int) -> None:
        from forcesmolvla.rft import training_cycle as g5

        if prefetch_workers < 1:
            raise ValueError("THROUGHPUT_V2_PREFETCH_WORKERS")
        self.data = data
        self.g5 = g5
        self.prefetch_workers = int(prefetch_workers)
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.table_lock = threading.Lock()
        self.images = BoundedDecodedImageLRU(max_bytes, g5.decode_rgb)
        self.parquet_read_calls = 0
        self.parquet_rows_materialized = 0
        self.prefetch_calls = 0
        self.prefetch_images_requested = 0
        self.prefetch_window_peak_images = 0
        self.prefetch_seconds = 0.0
        self.unique_observation_row_references: set[tuple[str, int]] = set()
        self.rss_start_bytes = process_rss_bytes()
        self.rss_peak_bytes = self.rss_start_bytes

    def _load_table(self, relative: str) -> list[dict[str, Any]]:
        import pyarrow.parquet as pq
        from forcesmolvla.rft.offline_transitions import PROVENANCE_KEYS
        from forcesmolvla.rft.training_cycle_runtime import DATASET

        with self.table_lock:
            cached = self.tables.get(relative)
            if cached is not None:
                return cached
            columns = [
                "observation.images.camera1", "observation.images.camera2",
                "observation.state", "observation.wrench", "frame_index",
                "episode_index", "index", "action", *PROVENANCE_KEYS,
            ]
            rows = pq.read_table(DATASET / relative, columns=columns).to_pylist()
            self.tables[relative] = rows
            self.parquet_read_calls += 1
            self.parquet_rows_materialized += len(rows)
            self._sample_rss()
            return rows

    def raw_rows(
        self, requested: dict[str, set[int]], *, include_actions: bool
    ) -> dict[tuple[str, int], dict[str, Any]]:
        del include_actions
        return {
            (relative, index): self._load_table(relative)[index]
            for relative, indices in requested.items()
            for index in sorted(indices)
        }

    def observation_payloads(self, indices: list[int]) -> list[bytes]:
        payloads: list[bytes] = []
        for index in indices:
            transition = self.data.rows[index]
            for name in ("observation_row_reference", "next_observation_row_reference"):
                reference = transition[name]
                self.unique_observation_row_references.add(
                    (reference["data_relative_path"], int(reference["row_index"]))
                )
                row = self._load_table(reference["data_relative_path"])[reference["row_index"]]
                payloads.extend((
                    row["observation.images.camera1"]["bytes"],
                    row["observation.images.camera2"]["bytes"],
                ))
        return list(dict.fromkeys(payloads))

    def prefetch_indices(self, indices: list[int]) -> None:
        started = time.perf_counter()
        payloads = self.observation_payloads(indices)
        self.prefetch_calls += 1
        self.prefetch_images_requested += len(payloads)
        self.prefetch_window_peak_images = max(
            self.prefetch_window_peak_images, len(payloads)
        )
        with ThreadPoolExecutor(
            max_workers=self.prefetch_workers,
            thread_name_prefix="s2-throughput-v2-prefetch",
        ) as pool:
            list(pool.map(self.images.get, payloads))
        self.prefetch_seconds += time.perf_counter() - started
        self._sample_rss()

    def _sample_rss(self) -> None:
        self.rss_peak_bytes = max(self.rss_peak_bytes, process_rss_bytes())

    def report(self) -> dict[str, Any]:
        self._sample_rss()
        return {
            "parquet_read_calls": self.parquet_read_calls,
            "parquet_rows_materialized": self.parquet_rows_materialized,
            "prefetch_workers": self.prefetch_workers,
            "prefetch_window": "current_build_batch_only",
            "prefetch_window_peak_images": self.prefetch_window_peak_images,
            "prefetch_calls": self.prefetch_calls,
            "prefetch_images_requested": self.prefetch_images_requested,
            "prefetch_seconds": self.prefetch_seconds,
            "total_unique_observation_row_references": len(
                self.unique_observation_row_references
            ),
            "process_rss_start_bytes": self.rss_start_bytes,
            "process_rss_peak_bytes": self.rss_peak_bytes,
            "cache_reconstructed_on_resume": True,
            "sampler_or_training_rng_consumed_by_prefetch": False,
            **self.images.report(),
        }


@contextmanager
def install_bounded_training_cache(
    data, *, max_bytes: int, prefetch_workers: int
) -> Iterator[BoundedTrainingDataCache]:
    """Install process-local cache without changing sampler or RNG state."""

    from forcesmolvla.rft import training_cycle as g5

    cache = BoundedTrainingDataCache(
        data, max_bytes=max_bytes, prefetch_workers=prefetch_workers
    )
    original_raw = data._raw_rows
    original_build = data.build_batch
    original_decode = g5.decode_rgb

    def raw(_data, requested, *, include_actions):
        return cache.raw_rows(requested, include_actions=include_actions)

    def build(_data, indices, policy, device, **kwargs):
        cache.prefetch_indices(list(indices))
        return original_build(indices, policy, device, **kwargs)

    data._raw_rows = types.MethodType(raw, data)
    data.build_batch = types.MethodType(build, data)
    g5.decode_rgb = cache.images.get
    try:
        yield cache
    finally:
        data._raw_rows = original_raw
        data.build_batch = original_build
        g5.decode_rgb = original_decode


def stable_draw_plan_sha256(plan: list[dict[str, Any]]) -> str:
    import json

    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
