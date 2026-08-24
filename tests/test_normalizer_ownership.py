import numpy as np
import pytest

from forcesmolvla.normalizer import (
    CartesianNormalizerBundle,
    NormalizationLedger,
    build_action_target_population,
    chunk_relative_delta_fit_rows,
)
from forcesmolvla.split import EpisodeSplit


def test_normalizer_fits_train_only_and_applies_exactly_once():
    split = EpisodeSplit(("e0", "e1"), ("e2",), ("e3",))
    ids = ("e0", "e0", "e1", "e1")
    base = np.arange(4, dtype=np.float64)[:, None]
    bundle = CartesianNormalizerBundle.fit(
        state7=base + np.arange(7)[None, :],
        wrench6=base + np.arange(6)[None, :],
        delta_action7=base * 2 + np.arange(7)[None, :],
        sample_episode_ids=ids,
        split=split,
        split_sha256="1" * 64,
        calibration_bundle_sha256="2" * 64,
        wrench_geometry_spec_sha256="3" * 64,
    )
    assert bundle.state7.fit_episode_ids == ("e0", "e1")
    ledger = NormalizationLedger()
    args = dict(
        batch_id="batch-0",
        state7=np.ones((2, 7)),
        wrench6=np.ones((2, 6)),
        delta_action7=np.ones((2, 7)),
        ledger=ledger,
    )
    outputs = bundle.normalize_once(**args)
    assert [output.shape[-1] for output in outputs] == [7, 6, 7]
    assert ledger.counts == {"state7": 1, "wrench6": 1, "delta_action7": 1}
    with pytest.raises(RuntimeError, match="MORE_THAN_ONCE"):
        bundle.normalize_once(**args)


def test_normalizer_rejects_nontrain_episode_fit_input():
    split = EpisodeSplit(("e0",), ("e1",), ("e2",))
    with pytest.raises(ValueError, match="validation/test"):
        CartesianNormalizerBundle.fit(
            state7=np.ones((2, 7)),
            wrench6=np.ones((2, 6)),
            delta_action7=np.ones((2, 7)),
            sample_episode_ids=("e0", "e1"),
            split=split,
            split_sha256="1" * 64,
            calibration_bundle_sha256="2" * 64,
            wrench_geometry_spec_sha256="3" * 64,
        )


def test_chunk_relative_delta_fit_matches_training_windows_and_excludes_padding():
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03],
        ]
    )
    actions = states.copy()
    actions[:, 0] += np.array([0.1, 0.2, 0.3])
    rows, ids = chunk_relative_delta_fit_rows((("e0", states, actions),), horizon=50)
    # Valid targets are 3 + 2 + 1. No repeated right-padding rows are fitted.
    assert rows.shape == (6, 7)
    assert ids == ("e0",) * 6
    np.testing.assert_allclose(rows[:3, 0], [0.1, 1.2, 2.3])
    np.testing.assert_allclose(rows[:, 6], [0.01, 0.02, 0.03, 0.02, 0.03, 0.03])


def test_action_normalizer_accepts_more_valid_chunk_targets_than_state_rows():
    split = EpisodeSplit(("e0", "e1"), ("e2",), ("e3",))
    base = np.arange(4, dtype=np.float64)[:, None]
    action_rows = np.arange(42, dtype=np.float64).reshape(6, 7)
    bundle = CartesianNormalizerBundle.fit(
        state7=base + np.arange(7)[None, :],
        wrench6=base + np.arange(6)[None, :],
        delta_action7=action_rows,
        sample_episode_ids=("e0", "e0", "e1", "e1"),
        delta_action_episode_ids=("e0", "e0", "e0", "e1", "e1", "e1"),
        split=split,
        split_sha256="1" * 64,
        calibration_bundle_sha256="2" * 64,
        wrench_geometry_spec_sha256="3" * 64,
    )
    np.testing.assert_allclose(bundle.delta_action7.mean, action_rows.mean(axis=0))
    assert bundle.manifest()["fit_contract"]["delta_action7"]["horizon"] == 50


def test_future_target_uses_anchor_state_not_future_same_frame_state():
    states = np.zeros((3, 7), dtype=np.float64)
    states[:, 0] = [0.0, 10.0, 20.0]
    states[:, 6] = 0.04
    actions = states.copy()
    actions[:, 0] += 1.0
    population = build_action_target_population((("e0", states, actions),))
    anchor_zero_k1 = (population.anchor_t == 0) & (population.horizon_k == 1)
    assert population.action_target7[anchor_zero_k1, 0].item() == 11.0
    assert population.action_target7[anchor_zero_k1, 0].item() != 1.0
    assert population.action_target7[anchor_zero_k1, 6].item() == 0.04
