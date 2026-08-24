"""Train-episode-only feature normalizers with exactly-once application sentinels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
import struct
from typing import Iterable

import numpy as np

from .action_delta import ActionDeltaProcessor
from .split import EpisodeSplit, fit_train_only_normalizer


NORMALIZER_SCHEMA_VERSION = "2.0"
DELTA_ACTION_FIT_CONTRACT = {
    "semantic_name": "action_target7",
    "source": "train episodes only",
    "horizon": 50,
    "reference_state": "raw measured state7 at each t_ref",
    "targets": "all valid future absolute action7[t_ref:t_ref+H] within the same episode",
    "transform": "ActionDeltaProcessor.to_delta(action_chunk, state7_at_t_ref)",
    "invalid_right_padded_tail": "excluded",
    "gripper": "absolute target_gripper_width_m unchanged",
}
ACTION_TARGET_QUANTILES = (0.01, 0.1, 0.5, 0.9, 0.99)


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _population_statistics(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, ACTION_TARGET_QUANTILES, axis=0)
    return {
        "count": len(array),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0, ddof=0).tolist(),
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "quantiles": {
            str(value): row.tolist()
            for value, row in zip(ACTION_TARGET_QUANTILES, quantiles, strict=True)
        },
    }


@dataclass(frozen=True)
class ActionTargetPopulation:
    action_target7: np.ndarray
    episode_ids: tuple[str, ...]
    anchor_t: np.ndarray
    horizon_k: np.ndarray
    horizon: int = 50

    def __post_init__(self) -> None:
        targets = np.array(self.action_target7, dtype=np.float64, copy=True)
        anchors = np.array(self.anchor_t, dtype=np.int64, copy=True)
        horizons = np.array(self.horizon_k, dtype=np.int64, copy=True)
        count = len(targets)
        if (
            targets.shape != (count, 7)
            or len(self.episode_ids) != count
            or anchors.shape != (count,)
            or horizons.shape != (count,)
            or self.horizon != 50
            or np.any((horizons < 0) | (horizons >= self.horizon))
        ):
            raise ValueError("invalid action-target population")
        for value in (targets, anchors, horizons):
            value.setflags(write=False)
        object.__setattr__(self, "action_target7", targets)
        object.__setattr__(self, "anchor_t", anchors)
        object.__setattr__(self, "horizon_k", horizons)

    def identity_sha256(self) -> str:
        digest = hashlib.sha256()
        for episode_id, anchor_t, horizon_k in zip(
            self.episode_ids, self.anchor_t, self.horizon_k, strict=True
        ):
            encoded = episode_id.encode("utf-8")
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
            digest.update(struct.pack("<qB", int(anchor_t), int(horizon_k)))
        return digest.hexdigest()

    def tensor_sha256(self) -> str:
        tensor = np.ascontiguousarray(self.action_target7, dtype="<f8")
        digest = hashlib.sha256()
        digest.update(struct.pack("<QQ", *tensor.shape))
        digest.update(tensor.view(np.uint8))
        return digest.hexdigest()

    def statistics(self) -> dict:
        per_horizon = []
        for horizon_k in range(self.horizon):
            per_horizon.append(
                {
                    "horizon_k": horizon_k,
                    **_population_statistics(
                        self.action_target7[self.horizon_k == horizon_k]
                    ),
                }
            )
        return {
            "method": "numpy population moments ddof=0; linear quantiles [0.01,0.1,0.5,0.9,0.99]",
            "global": _population_statistics(self.action_target7),
            "per_horizon": per_horizon,
        }

    def manifest(self, *, split_sha256: str, builder_source_sha256: str) -> dict:
        statistics = self.statistics()
        payload = {
            "status": "pass",
            "semantic_name": "action_target7",
            "builder": "forcesmolvla.normalizer.build_action_target_population",
            "builder_source_sha256": builder_source_sha256,
            "horizon": self.horizon,
            "mask": "action_valid_mask=True only; right-padded tail excluded",
            "split_sha256": split_sha256,
            "valid_pair_count": len(self.action_target7),
            "ordered_pair_identity_sha256": self.identity_sha256(),
            "action_target7_float64_tensor_sha256": self.tensor_sha256(),
            "statistics": statistics,
            "statistics_sha256": _canonical_sha256(statistics),
        }
        payload["population_manifest_sha256"] = _canonical_sha256(payload)
        return payload


def build_action_target_population(
    episodes: Iterable[tuple[str, np.ndarray, np.ndarray]],
    *,
    horizon: int = 50,
) -> ActionTargetPopulation:
    """Build the ordered train anchor/horizon target population used for fitting."""

    if horizon != DELTA_ACTION_FIT_CONTRACT["horizon"]:
        raise ValueError("action-target normalizer horizon must be exactly 50")
    rows: list[np.ndarray] = []
    episode_ids: list[str] = []
    anchor_indices: list[int] = []
    horizon_indices: list[int] = []
    for episode_id, state7, absolute_action7 in episodes:
        states = np.asarray(state7, dtype=np.float64)
        actions = np.asarray(absolute_action7, dtype=np.float64)
        if not episode_id or states.ndim != 2 or states.shape[1:] != (7,):
            raise ValueError("normalizer episode state must be nonempty [N,7]")
        if actions.shape != states.shape or len(states) == 0:
            raise ValueError("normalizer episode action must align with nonempty state [N,7]")
        for t_ref in range(len(states)):
            valid_actions = actions[t_ref : min(t_ref + horizon, len(actions))]
            rows.append(ActionDeltaProcessor.to_delta(valid_actions, states[t_ref]))
            count = len(valid_actions)
            episode_ids.extend((episode_id,) * count)
            anchor_indices.extend((t_ref,) * count)
            horizon_indices.extend(range(count))
    if not rows:
        raise ValueError("action-target normalizer requires at least one train episode")
    return ActionTargetPopulation(
        np.concatenate(rows, axis=0),
        tuple(episode_ids),
        np.asarray(anchor_indices, dtype=np.int64),
        np.asarray(horizon_indices, dtype=np.int64),
        horizon,
    )


def chunk_relative_delta_fit_rows(
    episodes: Iterable[tuple[str, np.ndarray, np.ndarray]],
    *,
    horizon: int = 50,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build exactly the valid delta-action rows seen by H-step training chunks."""

    population = build_action_target_population(episodes, horizon=horizon)
    return population.action_target7, population.episode_ids


@dataclass(frozen=True)
class FrozenFeatureNormalizer:
    name: str
    mean: np.ndarray
    std: np.ndarray
    fit_episode_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        std = np.array(self.std, dtype=np.float64, copy=True)
        if mean.ndim != 1 or std.shape != mean.shape:
            raise ValueError("normalizer mean/std must be matching 1D arrays")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("normalizer stats must be finite with positive std")
        mean.setflags(write=False)
        std.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.mean) or not np.all(np.isfinite(array)):
            raise ValueError(f"{self.name} values have invalid shape or nonfinite data")
        return (array - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.mean) or not np.all(np.isfinite(array)):
            raise ValueError(f"{self.name} values have invalid shape or nonfinite data")
        return array * self.std + self.mean

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "fit_episode_ids": list(self.fit_episode_ids),
        }


@dataclass
class NormalizationLedger:
    _claims: set[tuple[str, str]] = field(default_factory=set)
    counts: dict[str, int] = field(default_factory=dict)

    def claim(self, batch_id: str, feature_name: str) -> None:
        key = (batch_id, feature_name)
        if key in self._claims:
            raise RuntimeError(f"NORMALIZER_APPLIED_MORE_THAN_ONCE: {batch_id}/{feature_name}")
        self._claims.add(key)
        self.counts[feature_name] = self.counts.get(feature_name, 0) + 1


@dataclass(frozen=True)
class CartesianNormalizerBundle:
    state7: FrozenFeatureNormalizer
    wrench6: FrozenFeatureNormalizer
    delta_action7: FrozenFeatureNormalizer
    split_sha256: str
    calibration_bundle_sha256: str
    wrench_geometry_spec_sha256: str
    action_target_population: dict | None = None

    @classmethod
    def fit(
        cls,
        *,
        state7: np.ndarray,
        wrench6: np.ndarray,
        delta_action7: np.ndarray,
        sample_episode_ids,
        delta_action_episode_ids=None,
        split: EpisodeSplit,
        split_sha256: str,
        calibration_bundle_sha256: str,
        wrench_geometry_spec_sha256: str,
        action_target_population: dict | None = None,
    ) -> "CartesianNormalizerBundle":
        episode_ids = tuple(sample_episode_ids)
        action_episode_ids = (
            episode_ids
            if delta_action_episode_ids is None
            else tuple(delta_action_episode_ids)
        )
        expected = {"state7": 7, "wrench6": 6, "delta_action7": 7}
        values = {"state7": state7, "wrench6": wrench6, "delta_action7": delta_action7}
        feature_episode_ids = {
            "state7": episode_ids,
            "wrench6": episode_ids,
            "delta_action7": action_episode_ids,
        }
        normalizers = {}
        for name, dimension in expected.items():
            array = np.asarray(values[name], dtype=np.float64)
            ids = feature_episode_ids[name]
            if array.ndim != 2 or array.shape != (len(ids), dimension):
                raise ValueError(f"{name} must have shape [N,{dimension}]")
            stats = fit_train_only_normalizer(array, ids, split=split)
            normalizers[name] = FrozenFeatureNormalizer(
                name, stats.mean, stats.std, stats.fit_episode_ids
            )
        for digest in (split_sha256, calibration_bundle_sha256, wrench_geometry_spec_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("compatibility hashes must be lowercase SHA256")
        return cls(
            **normalizers,
            split_sha256=split_sha256,
            calibration_bundle_sha256=calibration_bundle_sha256,
            wrench_geometry_spec_sha256=wrench_geometry_spec_sha256,
            action_target_population=action_target_population,
        )

    def normalize_once(
        self,
        *,
        batch_id: str,
        state7: np.ndarray,
        wrench6: np.ndarray,
        delta_action7: np.ndarray,
        ledger: NormalizationLedger,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = []
        for normalizer, values in (
            (self.state7, state7),
            (self.wrench6, wrench6),
            (self.delta_action7, delta_action7),
        ):
            ledger.claim(batch_id, normalizer.name)
            outputs.append(normalizer.apply(values))
        return tuple(outputs)

    def manifest(self) -> dict:
        payload = {
            "schema_version": NORMALIZER_SCHEMA_VERSION,
            "owner": "forcesmolvla.CartesianNormalizerBundle",
            "inherited_lerobot_normalizers": "Identity/disconnected",
            "fit_contract": {
                "state7": "one raw measured state7 row per eligible train tuple",
                "wrench6": "one calibrated wrench6 row per eligible train tuple",
                "delta_action7": DELTA_ACTION_FIT_CONTRACT,
            },
            "action_target_population": (
                json.loads(json.dumps(self.action_target_population, sort_keys=True))
                if self.action_target_population is not None
                else {"status": "unbound_synthetic"}
            ),
            "features": {
                "state7": self.state7.manifest(),
                "wrench6": self.wrench6.manifest(),
                "delta_action7": self.delta_action7.manifest(),
            },
            "split_sha256": self.split_sha256,
            "calibration_bundle_sha256": self.calibration_bundle_sha256,
            "wrench_geometry_spec_sha256": self.wrench_geometry_spec_sha256,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["normalizer_stats_sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload
