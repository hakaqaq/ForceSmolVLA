"""Deterministic episode-disjoint split and train-only normalizer helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def assert_disjoint(self) -> None:
        groups = [set(self.train), set(self.val), set(self.test)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise AssertionError("episode split is not disjoint")


def split_episodes(
    episode_ids: Iterable[str],
    *,
    ratios: tuple[float, float, float] | None,
    seed: str | int | None,
) -> EpisodeSplit:
    if ratios is None or seed is None:
        raise ValueError("episode split ratios and seed are required; there are no defaults")
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("ratios must contain three positive values summing to one")
    ids = tuple(sorted(set(episode_ids)))
    if len(ids) < 3:
        raise ValueError("at least three eligible episodes are required")
    ranked = sorted(
        ids,
        key=lambda episode: hashlib.sha256(f"{seed}:{episode}".encode()).digest(),
    )
    n = len(ranked)
    train_end = max(1, min(n - 2, round(n * ratios[0])))
    val_count = max(1, min(n - train_end - 1, round(n * ratios[1])))
    split = EpisodeSplit(
        tuple(sorted(ranked[:train_end])),
        tuple(sorted(ranked[train_end : train_end + val_count])),
        tuple(sorted(ranked[train_end + val_count :])),
    )
    split.assert_disjoint()
    return split


@dataclass(frozen=True)
class NormalizerStats:
    mean: np.ndarray
    std: np.ndarray
    fit_episode_ids: tuple[str, ...]


def fit_train_only_normalizer(
    values: np.ndarray,
    sample_episode_ids: Iterable[str],
    *,
    split: EpisodeSplit,
) -> NormalizerStats:
    array = np.asarray(values, dtype=np.float64)
    episode_ids = np.asarray(tuple(sample_episode_ids), dtype=object)
    if array.ndim != 2 or len(array) != len(episode_ids):
        raise ValueError("values must be [N,D] and align with sample_episode_ids")
    allowed = set(split.train)
    if any(episode not in allowed for episode in episode_ids):
        raise ValueError("normalizer input contains validation/test or unknown episodes")
    if not np.all(np.isfinite(array)):
        raise ValueError("normalizer input must be finite")
    std = array.std(axis=0)
    if np.any(std <= 0):
        raise ValueError("normalizer encountered a constant feature")
    return NormalizerStats(array.mean(axis=0), std, tuple(sorted(set(episode_ids))))
