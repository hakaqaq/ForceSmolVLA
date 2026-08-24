import json
from pathlib import Path

import numpy as np
import pytest

from forcesmolvla.temporal import (
    action_chunk_zoh_indices,
    controller_reference_grid,
    match_measured_tcp_pose_causal_zoh,
    select_latest_causal,
)


FIXTURE = Path(__file__).parents[1] / "golden_fixtures" / "causal_pose_zoh.json"


def test_causal_pose_zoh_golden_fixture():
    fixture = json.loads(FIXTURE.read_text())
    poses = np.asarray(fixture["pose_stamps_ns"], dtype=np.int64)
    wrenches = np.asarray([case["stamp_ns"] for case in fixture["wrench_cases"]], dtype=np.int64)
    matches = match_measured_tcp_pose_causal_zoh(
        poses, wrenches, max_pose_age_ms=fixture["max_pose_age_ms_candidate"]
    )
    for case, index, age, valid in zip(
        fixture["wrench_cases"],
        matches.pose_indices,
        matches.pose_age_ms,
        matches.valid,
        strict=True,
    ):
        expected_index = case["expected_pose_index"]
        assert int(index) == (-1 if expected_index is None else expected_index)
        assert bool(valid) is case["expected_valid"]
        if "expected_pose_age_ms" in case:
            assert age == pytest.approx(case["expected_pose_age_ms"])


def test_max_pose_age_has_no_default():
    with pytest.raises(ValueError, match="required"):
        match_measured_tcp_pose_causal_zoh(
            np.array([1], dtype=np.int64),
            np.array([1], dtype=np.int64),
            max_pose_age_ms=None,
        )


def test_nonmonotonic_source_timestamp_fails():
    with pytest.raises(ValueError, match="strictly increasing"):
        match_measured_tcp_pose_causal_zoh(
            np.array([1, 1], dtype=np.int64),
            np.array([2], dtype=np.int64),
            max_pose_age_ms=12.0,
        )


def test_reference_grid_has_fixed_global_phase_and_no_compression():
    grid = controller_reference_grid(
        session_start_ack_ns=100_000_001,
        episode_end_ns=210_000_000,
        fps=30,
    )
    np.testing.assert_array_equal(grid, [133_333_333, 166_666_667, 200_000_000])


def test_latest_causal_never_reads_future():
    selected = select_latest_causal(
        np.array([90, 110, 130], dtype=np.int64) * 1_000_000,
        np.array([100, 120, 140], dtype=np.int64) * 1_000_000,
        max_age_ms=15,
    )
    np.testing.assert_array_equal(selected.source_indices, [0, 1, 2])
    np.testing.assert_allclose(selected.age_ms, [10, 10, 10])
    assert selected.valid.all()


def test_action_tail_requires_three_acknowledged_zoh_labels():
    stamps = np.array([100, 200, 300], dtype=np.int64)
    indices, valid = action_chunk_zoh_indices(
        stamps,
        tau0_ns=100,
        horizon=5,
        action_period_ns=100,
        episode_end_ns=300,
    )
    np.testing.assert_array_equal(indices, [0, 1, 2, 2, 2])
    np.testing.assert_array_equal(valid, [True, True, True, False, False])
    with pytest.raises(ValueError, match="TAIL_TOO_SHORT"):
        action_chunk_zoh_indices(
            stamps,
            tau0_ns=200,
            horizon=5,
            action_period_ns=100,
            episode_end_ns=300,
        )
