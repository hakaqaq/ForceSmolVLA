from pathlib import Path

import numpy as np
import pytest

from forcesmolvla.raw_to_lerobot_v3 import (
    SUPPORTED_ACTION_ASSOCIATION,
    SUPPORTED_CLOCK_MAP,
    SUPPORTED_FILTER,
    SUPPORTED_GRID_ANCHOR,
    RuntimeContract,
    _associate_acknowledged_actions,
    estimate_clock_map,
    parse_args,
    source_tree_manifest,
)


def contract(**overrides) -> RuntimeContract:
    values = {
        "max_pose_age_ms": 12.0,
        "camera_max_age_ms": 34.0,
        "max_intercamera_skew_ms": 33.0,
        "clock_map_method": SUPPORTED_CLOCK_MAP,
        "clock_map_lower_fraction": 0.01,
        "clock_map_min_lower_samples": 2,
        "clock_map_max_callback_delay_p99_ms": 2.0,
        "controller_grid_anchor": SUPPORTED_GRID_ANCHOR,
        "action_association": SUPPORTED_ACTION_ASSOCIATION,
        "action_pose_tolerance_m": 1e-12,
        "action_quaternion_tolerance_rad": 1e-7,
        "filter_implementation": SUPPORTED_FILTER,
        "filter_sos": np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
        "filter_warmup_samples": 2,
        "max_wrench_source_gap_ms": 9.0,
        "split_ratios": (0.8, 0.1, 0.1),
        "split_seed": "fixture",
    }
    values.update(overrides)
    return RuntimeContract(**values)


def test_cli_defaults_are_direct_raw_to_separate_v3_output():
    args = parse_args([])
    assert args.raw_root == Path("/home/rlc123/fr3_client_ws/datasets/task1")
    assert args.output_root == Path(
        "/home/rlc123/ForceSmolVLA/datasets/task1_forcesmolvla_v4_1"
    )
    assert args.project_root == Path("/home/rlc123/ForceSmolVLA")


def test_cli_accepts_explicit_task_specific_development_runtime_spec(tmp_path):
    runtime_spec = tmp_path / "task2.json"
    args = parse_args(["--development-only", "--runtime-spec", str(runtime_spec)])
    assert args.runtime_spec == runtime_spec


def test_clock_map_uses_explicit_supported_contract_and_is_deterministic():
    records = [
        {"source_stamp_ns": 100, "receive_monotonic_ns": 110},
        {"source_stamp_ns": 200, "receive_monotonic_ns": 211},
        {"source_stamp_ns": 300, "receive_monotonic_ns": 312},
    ]
    result = estimate_clock_map((("pose", records),), contract())
    assert result.offset_ns == 10
    np.testing.assert_array_equal(result.source_to_host(np.array([100, 200])), [110, 210])
    assert len(result.sha256) == 64


def test_runtime_contract_rejects_unfrozen_semantics():
    with pytest.raises(ValueError, match="unsupported clock-map"):
        contract(clock_map_method="guess-latest")


def test_ack_action_uses_latest_causal_reference_and_preserves_absolute_width():
    references = [
        {
            "accepted_receive_monotonic_ns": 100,
            "pose": {"position_m": [1, 2, 3], "quaternion_xyzw": [0, 0, 0, 1]},
            "target_gripper_width_m": 0.085,
        },
        {
            "accepted_receive_monotonic_ns": 200,
            "pose": {"position_m": [4, 5, 6], "quaternion_xyzw": [0, 0, 0, 1]},
            "target_gripper_width_m": 0.02,
        },
    ]
    acknowledgements = [
        {
            "receive_monotonic_ns": 101,
            "payload": {
                "accepted": True,
                "accepted_pose": references[0]["pose"],
            },
        },
        {
            "receive_monotonic_ns": 201,
            "payload": {
                "accepted": True,
                "accepted_pose": references[1]["pose"],
            },
        },
    ]
    times, actions = _associate_acknowledged_actions(references, acknowledgements, contract())
    np.testing.assert_array_equal(times, [101, 201])
    np.testing.assert_allclose(actions[:, :3], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(actions[:, 7], [0.085, 0.02])


def test_source_tree_manifest_is_content_bound_and_sorted(tmp_path):
    (tmp_path / "b.txt").write_text("second")
    (tmp_path / "a.txt").write_text("first")
    entries, digest = source_tree_manifest(tmp_path)
    assert [entry["path"] for entry in entries] == ["a.txt", "b.txt"]
    assert all(len(entry["sha256"]) == 64 for entry in entries)
    assert len(digest) == 64
