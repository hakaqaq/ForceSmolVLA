from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from forcesmolvla.rft.online import temporal_parity as temporal_parity_module
from forcesmolvla.rft.critic import (
    ForceAwareMacroCritic,
    FrozenConRFTResNet10,
    frozen_task_feature,
)
from forcesmolvla.rft.losses import CriticObservation
from forcesmolvla.rft.frozen_vlm_trainability import frozen_prefix_flow_matching_terms
from forcesmolvla.rft.online.training_losses import compute_online_twin_q_td_loss
from forcesmolvla.rft.online.temporal_parity import (
    ROOT,
    TemporalParityError,
    blocked_temporal_parity_report,
    directory_tree_sha256,
    run_recorded_ack_parity,
    sha256_file,
    validate_recorded_ack_fixture,
    validate_temporal_parity_report,
)
from forcesmolvla.rft.batch import build_actor_batch


def _binding(relative: str) -> dict:
    path = ROOT / relative
    return {"path": relative, "sha256": sha256_file(path)}


def _path_binding(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _synthetic_fixture(tmp_path: Path) -> dict:
    raw = tmp_path / "raw_session"
    episode = raw / "episodes/episode_000000"
    streams = episode / "streams"
    images = episode / "images"
    streams.mkdir(parents=True)
    images.mkdir()
    (images / "camera1.rgb").write_bytes(b"synthetic camera1")
    (images / "camera2.rgb").write_bytes(b"synthetic camera2")
    capture = tmp_path / "capture_manifest.json"
    _write_json(capture, {"synthetic": True})

    grid = [
        1_000_000_000, 1_033_333_333, 1_066_666_667,
        1_100_000_000, 1_133_333_333, 1_166_666_667,
        1_200_000_000, 1_233_333_333, 1_266_666_667,
    ]
    state_position = [float(np.float32(0.39)), 0.0, float(np.float32(0.29))]
    state_pose = {
        "position_m": state_position,
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    poses = [
        {
            "source_stamp_ns": timestamp,
            "receive_monotonic_ns": timestamp,
            "pose": state_pose,
        }
        for timestamp in grid
    ]
    wrenches = [
        {
            "source_stamp_ns": timestamp,
            "receive_monotonic_ns": timestamp,
            "force_xyz_n_torque_xyz_nm": [0.0] * 6,
        }
        for timestamp in grid
    ]
    grippers = [
        {
            "source_stamp_ns": timestamp,
            "receive_monotonic_ns": timestamp,
            "width_m": 0.0,
        }
        for timestamp in grid
    ]
    camera1 = [
        {
            "receive_monotonic_ns": timestamp,
            "rgb_path": "images/camera1.rgb",
            "role": "external",
            "model": "D435",
        }
        for timestamp in grid
    ]
    camera2 = [
        {
            "receive_monotonic_ns": timestamp,
            "rgb_path": "images/camera2.rgb",
            "role": "wrist",
            "model": "D405",
        }
        for timestamp in grid
    ]

    ack_times = [
        995_000_000, 1_030_000_000, 1_097_000_000,
        1_164_000_000, 1_231_000_000, 1_270_000_000,
    ]
    references = []
    acknowledgements = []
    for index, ack_time in enumerate(ack_times):
        pose = {
            "position_m": [
                float(np.float32(0.40 + index * 0.001)),
                0.0,
                float(np.float32(0.30)),
            ],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        command = f"gripper-{index}"
        references.append({
            "accepted_receive_monotonic_ns": ack_time - 1_000_000,
            "source_stamp_ns": ack_time - 1_000_000,
            "frame_id": "fr3_link0",
            "pose": pose,
            "target_gripper_width_m": 0.085 if index % 2 else 0.0,
            "gripper_command_id": command,
            "gripper_command_id_origin": "synthetic_unit_test",
        })
        acknowledgements.append({
            "ack_id": f"ack-{index}",
            "ack_id_origin": "synthetic_unit_test",
            "receive_monotonic_ns": ack_time,
            "request_sequence": index,
            "request_stamp_ns": ack_time - 2_000_000,
            "controller_ack_monotonic_ns": ack_time - 500_000,
            "action_decision_id": index,
            "action_source_receive_monotonic_ns": ack_time - 2_000_000,
            "gripper_command_id": command,
            "gripper_ack_command_id": command,
            "slot_owner": "policy",
            "accepted_action_source": "policy",
            "intervention": False,
            "workspace_clipped": False,
            "payload": {"accepted": True, "accepted_pose": pose},
        })

    source_streams = {
        "measured_tcp_pose": poses,
        "wrench_notch_sensor": wrenches,
        "gripper_state": grippers,
        "external_camera": camera1,
        "wrist_camera": camera2,
        "accepted_reference": references,
        "reference_ack": acknowledgements,
    }
    for name, records in source_streams.items():
        _write_jsonl(streams / f"{name}.jsonl", records)
    _write_json(episode / "episode_result.json", {
        "saved": True,
        "fatal_reason": None,
        "task": "Pick up the purple ring and place it onto the red peg.",
        "started_monotonic_ns": grid[0],
        "finished_monotonic_ns": grid[-1],
        "stream_counts": {name: len(records) for name, records in source_streams.items()},
    })
    _write_json(raw / "session.json", {
        "raw_format_version": "fr3-hilserl-impedance-native-raw-v5",
        "canonical_fps": 30,
        "cameras": {
            "observation.image": {"role": "external", "model": "D435"},
            "observation.wrist_image": {"role": "wrist", "model": "D405"},
        },
    })

    runtime = tmp_path / "converter_runtime.development.json"
    _write_json(runtime, {
        "artifact_status": "development_only",
        "formal_ready": False,
        "pose": {"max_age_ms": 40.0},
        "cameras": {"max_age_ms": 40.0, "max_intercamera_skew_ms": 40.0},
        "clock_map": {
            "method": "shared-lower-tail-median-v1",
            "lower_fraction": 0.01,
            "min_lower_samples": 1,
            "max_callback_delay_p99_ms": None,
        },
        "controller_grid": {
            "anchor": "first-reference-ack-global-zero-phase-rational-30hz-v1",
        },
        "action": {
            "association": "latest-causal-accepted-reference-with-pose-check-v1",
            "pose_tolerance_m": 1e-12,
            "quaternion_tolerance_rad": 1e-7,
        },
        "wrench_filter": {
            "implementation": "scipy-sosfilt-fixed-500hz-per-valid-source-sample-v1",
            "sos": [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            "warmup_samples": 0,
            "max_source_gap_ms": 40.0,
        },
        "split": {"ratios": [0.8, 0.1, 0.1], "seed": "synthetic-parity"},
    })
    calibration = tmp_path / "calibration_bundle.development.json"
    _write_json(calibration, {
        "artifact_status": "development_only",
        "formal_ready": False,
        "calibration_id": "synthetic-zero-wrench",
        "sensor_bias6": [0.0] * 6,
        "wrench_sign6": [1.0] * 6,
        "downstream_mass_kg": 0.0,
        "downstream_com_sensor_m": [0.0] * 3,
        "gravity_base_m_s2": [0.0, 0.0, -9.80665],
        "static_transform_tcp_sensor": {
            "translation_m": [0.0] * 3,
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    })

    return {
        "schema_version": "forcesmolvla_stage3_recorded_ack_fixture.v1",
        "fixture_id": "synthetic-tool-test",
        "fixture_kind": "synthetic_unit_test",
        "synthetic": True,
        "action_source": "policy",
        "capture_origin": "synthetic_unit_test",
        "provenance": {
            "recorded_live_evidence": False,
            "raw_session_path": str(raw),
            "raw_episode_path": str(episode),
            "raw_session_tree_sha256": directory_tree_sha256(raw),
            "capture_manifest_path": str(capture),
            "capture_manifest_sha256": sha256_file(capture),
        },
        "bindings": {
            "stage2_ack_converter": _binding("src/forcesmolvla/raw_to_lerobot_v3.py"),
            "stage2_temporal": _binding("src/forcesmolvla/temporal.py"),
            "action_delta": _binding("src/forcesmolvla/action_delta.py"),
            "normalizer_source": _binding("src/forcesmolvla/normalizer.py"),
            "normalizer_manifest": _binding(
                "artifacts/development/offline/offline_actor_critic_cycle_000210_actor_export.v1/manifests/normalizer_manifest.json"
            ),
            "action_contract_v2": _binding("configs/stage2_action_contract.v2.development.json"),
            "stage2_runtime_contract": _path_binding(runtime),
            "calibration_bundle": _path_binding(calibration),
            "terminal_transition_index": _path_binding(capture),
        },
        "selection": {
            "prepared_grid_start_index": 0,
            "prepared_grid_stop_index_exclusive": 9,
            "current_observation_grid_index": 0,
            "next_observation_grid_index": 3,
            "terminal_observation_grid_index": 8,
            "last_executable_grid_index": 7,
            "full_macro_transition_index": 0,
            "terminal_transition_index": 1,
            "observation_provenance": [
                {
                    "role": role,
                    "local_grid_index": index,
                    "global_grid_index": index,
                    "grid_monotonic_ns": grid[index],
                    "state7": state_position + [0.0, 0.0, 0.0, 0.0],
                    "wrench6": [0.0] * 6,
                    "external_camera_relative_path": "images/camera1.rgb",
                    "wrist_camera_relative_path": "images/camera2.rgb",
                    "state_pose_source_stamp_ns": grid[index],
                    "camera1_receive_monotonic_ns": grid[index],
                    "camera2_receive_monotonic_ns": grid[index],
                    "gripper_source_stamp_ns": grid[index],
                    "wrench_filter_output_stamp_ns": grid[index],
                    "action_ack_receive_monotonic_ns": {
                        0: ack_times[0], 3: ack_times[2], 8: ack_times[4],
                    }[index],
                    "validity_bits": 255,
                }
                for role, index in (("current", 0), ("next", 3), ("terminal", 8))
            ],
            "gripper_authority": {
                "action_goal_id": "synthetic-gripper-authority",
                "local_goal_sequence": 1,
                "accepted_monotonic_ns": 900_000_000,
                "target_receive_monotonic_ns": 901_000_000,
                "finished_monotonic_ns": 950_000_000,
                "status_receive_monotonic_ns": 951_000_000,
                "requested_state": "OPEN",
                "target_width_m": 0.085,
                "outcome": "reached",
            },
        },
        "temporal": {
            "session_start_ack_ns": grid[0],
            "episode_end_ns": grid[-1],
            "terminal_boundary_ns": 1_233_333_333,
            "data_grid_hz": 30,
            "policy_hz": 10,
            "K": 3,
            "macro_duration_ms": 100,
            "policy_anchor_phase_on_30hz_grid": 0,
            "max_ack_age_ms": 50.0,
        },
        "accepted_references": references,
        "reference_acks": acknowledgements,
        "anchor_states": [
            {"grid_index": index, "state7": state_position + [0.0, 0.0, 0.0, 0.0]}
            for index in (0, 3, 6)
        ],
    }


def test_synthetic_fixture_exercises_both_paths_but_cannot_open_formal_gate(tmp_path: Path) -> None:
    fixture = validate_recorded_ack_fixture(_synthetic_fixture(tmp_path))
    report = run_recorded_ack_parity(fixture)
    assert report["tool_status"] == "synthetic_tool_test_pass"
    assert all(report["bindings"].values())
    assert all(report["comparisons"].values())
    assert report["stage2"]["converter"] == {
        "module": "forcesmolvla.raw_to_lerobot_v3",
        "symbol": "prepare_episode",
        "call_count": 1,
        "numeric_output_source": "PreparedEpisode",
    }
    assert "not PreparedEpisode" in report["stage2"]["raw_identity_provenance"]["source"]
    for recorded_macro, online_macro in zip(
        report["stage2"]["macros"], report["stage3"]["macros"], strict=True,
    ):
        for field in (
            "grid_monotonic_ns", "anchor_state7", "accepted_absolute_action_k7",
            "anchor_relative_delta_k7", "normalized_delta_action_k7",
        ):
            np.testing.assert_array_equal(recorded_macro[field], online_macro[field])
        assert recorded_macro["normalizer_application_count"] == 1
        assert online_macro["normalizer_application_count"] == 1
    assert report["stage3"]["partial_macro_quarantine"] == [{
        "anchor_grid_index": 6,
        "available_slots": 2,
        "reason": "partial_macro_crosses_terminal_or_episode_end",
    }]
    assert report["G1_TEMPORAL_PARITY_GATE"] == "BLOCKED"
    assert report["RECORDED_FIXTURE_CAPTURE_REQUIRED"] is True
    assert report["G1_GATE_PASSED"] is False
    assert report["G2_FORMAL_GATE"] == "BLOCKED_ON_G1"


def test_stage2_parity_path_calls_production_prepare_episode_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _synthetic_fixture(tmp_path)
    production = temporal_parity_module.prepare_episode
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return production(*args, **kwargs)

    monkeypatch.setattr(temporal_parity_module, "prepare_episode", spy)
    report = run_recorded_ack_parity(fixture)
    assert len(calls) == 1
    assert report["comparisons"]["phase2_prepare_episode_called_once"] is True
    assert report["stage2"]["converter"]["call_count"] == 1


def test_missing_raw_episode_fails_closed(tmp_path: Path) -> None:
    fixture = _synthetic_fixture(tmp_path)
    fixture["provenance"]["raw_episode_path"] = str(tmp_path / "missing_episode")
    with pytest.raises(TemporalParityError, match="RAW_EPISODE_MISSING"):
        run_recorded_ack_parity(fixture)


def test_missing_ack_id_fails_closed(tmp_path: Path) -> None:
    fixture = _synthetic_fixture(tmp_path)
    del fixture["reference_acks"][0]["ack_id"]
    with pytest.raises(TemporalParityError, match="RECORDED_ACK_FIXTURE_SCHEMA"):
        run_recorded_ack_parity(fixture)


def test_missing_gripper_identity_fails_closed(tmp_path: Path) -> None:
    fixture = _synthetic_fixture(tmp_path)
    del fixture["reference_acks"][0]["gripper_ack_command_id"]
    with pytest.raises(TemporalParityError, match="RECORDED_ACK_FIXTURE_SCHEMA"):
        run_recorded_ack_parity(fixture)


def test_missing_recorded_fixture_is_schema_valid_blocked_report(tmp_path: Path) -> None:
    missing = tmp_path / "recorded_ack_fixture.json"
    report = blocked_temporal_parity_report(missing)
    assert validate_temporal_parity_report(report) == report
    assert report["tool_status"] == "blocked"
    assert "recorded positive reference_ack stream with ACK and gripper ACK IDs" in report["missing_required_fields"]
    assert report["ROBOT_COMMAND_COUNT"] == 0


def test_recorded_gripper_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture = _synthetic_fixture(tmp_path)
    fixture["reference_acks"][1]["gripper_ack_command_id"] = "wrong-command"
    with pytest.raises(TemporalParityError, match="GRIPPER_COMMAND_ID_MISMATCH"):
        run_recorded_ack_parity(fixture)


class _NeverCalledTarget(nn.Module):
    def forward(self, *_args):
        raise AssertionError("terminal row called target critic")


def test_online_td_calls_real_force_aware_macro_critic_interface() -> None:
    feature = frozen_task_feature()
    critic = ForceAwareMacroCritic(
        FrozenConRFTResNet10(), FrozenConRFTResNet10(), task_feature=feature,
    ).eval()
    camera = torch.zeros(1, 3, 32, 32, dtype=torch.uint8)
    observation = CriticObservation(
        camera1=camera,
        camera2=camera.clone(),
        task_feature=torch.from_numpy(feature).unsqueeze(0),
        normalized_state7=torch.zeros(1, 7),
        normalized_wrench6=torch.zeros(1, 6),
    )
    result = compute_online_twin_q_td_loss(
        q1=critic,
        q2=critic,
        q1_target=_NeverCalledTarget(),
        q2_target=_NeverCalledTarget(),
        observation=observation,
        next_observation=observation,
        ack_behavior_action_k7=torch.zeros(1, 3, 7),
        behavior_mask=torch.ones(1, 3, dtype=torch.bool),
        reward=torch.ones(1),
        discount=torch.zeros(1),
        terminated=torch.ones(1, dtype=torch.bool),
        truncated=torch.zeros(1, dtype=torch.bool),
        bootstrap_mask=torch.zeros(1, dtype=torch.bool),
        next_policy_action_fn=lambda _observation: (_ for _ in ()).throw(AssertionError()),
    )
    assert result.target.tolist() == [1.0]
    assert result.q1_value.shape == result.q2_value.shape == (1,)
    assert torch.isfinite(result.total)


class _Tokenizer:
    padding_side = ""
    truncation_side = ""

    def __init__(self) -> None:
        self.prompts = None

    def __call__(self, prompts, **_kwargs):
        self.prompts = prompts
        return {
            "input_ids": torch.ones(len(prompts), 48, dtype=torch.long),
            "attention_mask": torch.ones(len(prompts), 48, dtype=torch.long),
        }


def test_real_phase2_actor_critic_image_range_and_task_feature_contract() -> None:
    tokenizer = _Tokenizer()
    policy = type("Policy", (), {})()
    policy.model = type("Model", (), {})()
    policy.model.vlm_with_expert = type("VLM", (), {})()
    policy.model.vlm_with_expert.processor = type("Processor", (), {"tokenizer": tokenizer})()
    image = np.full((3, 8, 8), 255, dtype=np.uint8)
    task = "Pick up the purple ring and place it onto the red peg."
    batch = build_actor_batch(policy, [{
        "camera1": image,
        "camera2": image,
        "state7": np.zeros(7, dtype=np.float32),
        "wrench6": np.zeros(6, dtype=np.float32),
        "task": task,
        "sample_identity": "fixture/frame=0",
    }], torch.device("cpu"), include_action=False)
    assert batch["observation.images.camera1"].dtype == torch.float32
    assert batch["observation.images.camera1"].min().item() == 1.0
    assert batch["observation.images.camera1"].max().item() == 1.0
    assert tokenizer.prompts == [task + "\n"]
    critic_feature = torch.from_numpy(frozen_task_feature(task))
    assert critic_feature.dtype == torch.float32 and critic_feature.shape == (256,)


def test_real_phase2_frozen_prefix_path_is_no_grad_detached_and_force_kv_once() -> None:
    source = inspect.getsource(frozen_prefix_flow_matching_terms)
    assert "with torch.no_grad():" in source
    assert "context.prefix_out.detach()" in source
    assert '"prefix_prefill_count": 1' in source
    assert '"prefix_grad_enabled": False' in source
    assert '"prefix_representation_detached": True' in source
    assert '"prefix_cache_detached": True' in source
    assert '"force_kv_projection_count": 1' in source
