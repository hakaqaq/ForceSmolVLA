import numpy as np
import pytest

from forcesmolvla.split import EpisodeSplit, fit_train_only_normalizer, split_episodes


def test_episode_split_is_deterministic_and_disjoint():
    episodes = [f"episode_{index:06d}" for index in range(50)]
    first = split_episodes(episodes, ratios=(0.8, 0.1, 0.1), seed="fixture")
    second = split_episodes(reversed(episodes), ratios=(0.8, 0.1, 0.1), seed="fixture")
    assert first == second
    first.assert_disjoint()
    assert set(first.train + first.val + first.test) == set(episodes)


def test_split_semantics_have_no_default():
    with pytest.raises(ValueError, match="required"):
        split_episodes(["a", "b", "c"], ratios=None, seed=None)


def test_normalizer_rejects_validation_episode():
    split = EpisodeSplit(("train",), ("val",), ("test",))
    with pytest.raises(ValueError, match="validation/test"):
        fit_train_only_normalizer(
            np.array([[1.0, 2.0], [2.0, 4.0]]),
            ["train", "val"],
            split=split,
        )


def test_normalizer_records_train_episodes_only():
    split = EpisodeSplit(("train_a", "train_b"), ("val",), ("test",))
    stats = fit_train_only_normalizer(
        np.array([[1.0, 2.0], [3.0, 6.0]]),
        ["train_a", "train_b"],
        split=split,
    )
    assert stats.fit_episode_ids == ("train_a", "train_b")
