"""Causal timestamp selection for the v4.1 available-sensor profile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalPoseMatches:
    pose_indices: np.ndarray
    pose_age_ms: np.ndarray
    valid: np.ndarray
    failure_codes: tuple[str | None, ...]


@dataclass(frozen=True)
class CausalSelections:
    source_indices: np.ndarray
    age_ms: np.ndarray
    valid: np.ndarray


def _strictly_increasing(stamps_ns: np.ndarray, name: str) -> None:
    if stamps_ns.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(stamps_ns.dtype, np.integer):
        raise TypeError(f"{name} must use integer nanoseconds")
    if len(stamps_ns) > 1 and np.any(np.diff(stamps_ns) <= 0):
        raise ValueError(f"{name} must be strictly increasing")


def match_measured_tcp_pose_causal_zoh(
    pose_stamps_ns: np.ndarray,
    wrench_stamps_ns: np.ndarray,
    *,
    max_pose_age_ms: float | None,
) -> CausalPoseMatches:
    """Match each wrench to the latest measured TCP pose at or before it.

    ``max_pose_age_ms`` deliberately has no default.  A missing or invalid value
    is a contract error rather than a permissive fallback.
    """

    poses = np.asarray(pose_stamps_ns)
    wrenches = np.asarray(wrench_stamps_ns)
    _strictly_increasing(poses, "pose_stamps_ns")
    _strictly_increasing(wrenches, "wrench_stamps_ns")
    if max_pose_age_ms is None or not np.isfinite(max_pose_age_ms) or max_pose_age_ms < 0:
        raise ValueError("max_pose_age_ms is required and must be finite and non-negative")

    indices = np.searchsorted(poses, wrenches, side="right") - 1
    has_prior_pose = indices >= 0
    safe_indices = np.maximum(indices, 0)
    ages_ms = np.full(wrenches.shape, np.nan, dtype=np.float64)
    ages_ms[has_prior_pose] = (
        wrenches[has_prior_pose] - poses[safe_indices[has_prior_pose]]
    ) / 1_000_000.0
    if np.any(ages_ms[has_prior_pose] < 0):
        raise AssertionError("causal pose lookup selected a future pose")

    valid = has_prior_pose & (ages_ms <= max_pose_age_ms)
    failures: list[str | None] = []
    for has_pose, is_valid in zip(has_prior_pose, valid, strict=True):
        failures.append(None if is_valid else "WRENCH_POSE_MISSING_OR_STALE")
    return CausalPoseMatches(indices, ages_ms, valid, tuple(failures))


def controller_reference_grid(
    *, session_start_ack_ns: int, episode_end_ns: int, fps: int = 30
) -> np.ndarray:
    """Return fixed-phase controller ticks without compression or renumbering."""
    if session_start_ack_ns <= 0 or episode_end_ns < session_start_ack_ns:
        raise ValueError("invalid controller-clock interval")
    if fps <= 0:
        raise ValueError("fps must be positive")
    nanoseconds_per_second = 1_000_000_000
    first_index = (
        session_start_ack_ns * fps + nanoseconds_per_second - 1
    ) // nanoseconds_per_second
    last_index = (episode_end_ns * fps) // nanoseconds_per_second
    if last_index < first_index:
        return np.empty(0, dtype=np.int64)
    indices = np.arange(first_index, last_index + 1, dtype=np.int64)
    return ((indices * nanoseconds_per_second + fps // 2) // fps).astype(np.int64)


def select_latest_causal(
    source_stamps_ns: np.ndarray,
    target_stamps_ns: np.ndarray,
    *,
    max_age_ms: float | None,
) -> CausalSelections:
    sources = np.asarray(source_stamps_ns)
    targets = np.asarray(target_stamps_ns)
    _strictly_increasing(sources, "source_stamps_ns")
    _strictly_increasing(targets, "target_stamps_ns")
    if max_age_ms is None or not np.isfinite(max_age_ms) or max_age_ms < 0:
        raise ValueError("max_age_ms is required and must be finite and non-negative")
    indices = np.searchsorted(sources, targets, side="right") - 1
    has_source = indices >= 0
    safe = np.maximum(indices, 0)
    ages = np.full(targets.shape, np.nan, dtype=np.float64)
    ages[has_source] = (targets[has_source] - sources[safe[has_source]]) / 1_000_000.0
    if np.any(ages[has_source] < 0):
        raise AssertionError("latest-causal selector returned a future sample")
    return CausalSelections(indices, ages, has_source & (ages <= max_age_ms))


def action_chunk_zoh_indices(
    acknowledgement_stamps_ns: np.ndarray,
    *,
    tau0_ns: int,
    horizon: int,
    action_period_ns: int,
    episode_end_ns: int,
    minimum_valid: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Select acknowledgement-proven ZOH targets at future label times tau_k."""
    acknowledgements = np.asarray(acknowledgement_stamps_ns)
    _strictly_increasing(acknowledgements, "acknowledgement_stamps_ns")
    if tau0_ns <= 0 or horizon <= 0 or action_period_ns <= 0 or minimum_valid <= 0:
        raise ValueError("invalid action chunk parameters")
    tau = tau0_ns + np.arange(horizon, dtype=np.int64) * action_period_ns
    indices = np.searchsorted(acknowledgements, tau, side="right") - 1
    valid = (indices >= 0) & (tau <= episode_end_ns)
    if int(valid.sum()) < minimum_valid:
        raise ValueError("ACTION_TAIL_TOO_SHORT")
    return indices, valid
