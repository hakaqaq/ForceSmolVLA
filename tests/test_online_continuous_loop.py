from __future__ import annotations

from pathlib import Path
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_forcerft_online_loop as loop  # noqa: E402
import run_forcerft_integrated_capture as capture  # noqa: E402
import run_forcerft_production_bridge as bridge_tool  # noqa: E402


def test_reward_worker_request_passes_image_paths_without_temporary_frame_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera1 = [tmp_path / f"external-{index}.jpg" for index in range(3)]
    camera2 = [tmp_path / f"wrist-{index}.jpg" for index in range(3)]
    detector = bridge_tool.OneShotFrozenRewardDetector(
        tmp_path / "reward.msgpack",
        "task-reward-v1",
        120,
        worker_socket=tmp_path / "reward.sock",
    )

    def fake_request(_socket: Path, request: Path, output: Path) -> None:
        payload = bridge_tool.json.loads(request.read_text(encoding="utf-8"))
        assert payload["batches"] == [{
            "count": 3,
            "camera1_paths": [str(path) for path in camera1],
            "camera2_paths": [str(path) for path in camera2],
        }]
        assert not list(request.parent.glob("camera*.npy"))
        np.save(output, np.asarray([0.1, 0.2, 0.3]), allow_pickle=False)

    monkeypatch.setattr(bridge_tool, "_request_detector_worker", fake_request)
    scores = detector(SimpleNamespace(
        camera1_paths=camera1,
        camera2_paths=camera2,
    ))

    assert scores.probabilities == pytest.approx((0.1, 0.2, 0.3))


def test_task_output_root_and_replay_default_are_task_scoped(tmp_path: Path) -> None:
    args = loop.parse_args([
        "--task-id", "task2", "--output-root", str(tmp_path / "outputs/task2"),
        "--max-episodes", "1", "--root-prefix", str(tmp_path / "capture"),
        "--task", "ring",
    ])

    assert args.output_root == (tmp_path / "outputs/task2").resolve()
    assert args.dataset_root == (ROOT / "datasets/task2_lerobotv3").resolve()
    assert not hasattr(args, "reward_transition_root")
    assert args.formal_r_root == (tmp_path / "outputs/task2/online").resolve()
    assert args.allow_development_policy_execution_smoke is False
    assert not hasattr(args, "deployment_profile")
    assert not hasattr(args, "deployment_binding")


def test_online_capture_defaults_to_repository_dataset_root() -> None:
    args = loop.parse_args([
        "--task-id", "task2", "--max-episodes", "1", "--task", "ring",
    ])

    assert args.root_prefix == (ROOT / "datasets/task2_forcerft_online").resolve()


@pytest.mark.parametrize("reason", [
    "POLICY_EXECUTE_DECISION_TIMEOUT:7",
    "POLICY_EXECUTE_GRIPPER_ACK_TIMEOUT:policy-gripper:1",
    "POLICY_EXECUTE_OBSERVATION_READY_TIMEOUT",
    "POLICY_EXECUTE_POSE_ACK_TIMEOUT:123",
    (
        "SHADOW_BACKEND_FAILED:RuntimeError:"
        "CONTROLLER_ACK_POSITION_MISMATCH: error=0.018566m limit=0.017321m"
    ),
    (
        "POLICY_EXECUTE_GRIPPER_TERMINAL_INVALID:"
        "{'command_id': 'policy-gripper:1000001', 'outcome': 'not_reached'}"
    ),
])
def test_episode_local_timeout_uses_transient_exit(reason: str) -> None:
    assert capture._blocked_exit_code(
        capture.IntegratedCaptureError(reason)
    ) == os.EX_TEMPFAIL


def test_safety_failure_does_not_use_transient_exit() -> None:
    assert capture._blocked_exit_code(
        capture.IntegratedCaptureError("POLICY_EXECUTE_WRENCH_GUARD")
    ) == 2
    assert capture._blocked_exit_code(capture.IntegratedCaptureError(
        "POLICY_EXECUTE_GRIPPER_TERMINAL_INVALID:"
        "{'command_id': 'policy-gripper:1000001', 'outcome': 'result_error'}"
    )) == 2


def test_transient_capture_exit_is_typed_for_the_outer_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        loop.subprocess,
        "run",
        lambda *_args, **_kwargs: loop.subprocess.CompletedProcess(
            ["python", "capture.py"], os.EX_TEMPFAIL, "", ""
        ),
    )

    with pytest.raises(loop.EpisodeLocalTransientError):
        loop._run(["python", "capture.py"])


def test_command_report_uses_final_json_after_child_process_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (
        'Loaded reward model\n'
        '{"backend": "gpu", "frames": 1030}\n'
        'Finished detector\n'
        '{\n  "status": "FORMAL_ONLINE_R_ADMITTED",\n  "accepted": 1\n}\n'
    )
    monkeypatch.setattr(
        loop,
        "_run",
        lambda *_args, **_kwargs: loop.subprocess.CompletedProcess(
            ["python", "bridge.py"], 0, output, ""
        ),
    )

    assert loop._report(["python", "bridge.py"]) == {
        "status": "FORMAL_ONLINE_R_ADMITTED",
        "accepted": 1,
    }


def test_online_capture_restart_uses_next_session_index(tmp_path: Path) -> None:
    prefix = tmp_path / "task2_forcerft_online"
    prefix.mkdir()
    (prefix / "000").mkdir()
    (prefix / "002").mkdir()

    assert loop._next_capture_index(prefix) == 3


@pytest.mark.parametrize("sealed", [False, True])
def test_failed_capture_discards_only_unsealed_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sealed: bool,
) -> None:
    root_prefix = tmp_path / "capture"
    root_prefix.mkdir()
    root = root_prefix / "001"
    monkeypatch.setattr(loop, "_post_json", lambda *_args, **_kwargs: {
        "runtime_session_id": "capture_001",
        "runtime_episode_id": loop.EPISODE_ID,
        "server_persistent": True,
    })

    def fail_capture(_command, **_kwargs):
        assert _command[_command.index("--root") + 1] == str(root)
        root.mkdir()
        if sealed:
            seal = (
                root / "integrated_capture" / loop.EPISODE_ID
                / "streams" / "policy_execute_episode_seal.json"
            )
            seal.parent.mkdir(parents=True)
            seal.write_text("{}\n", encoding="utf-8")
        else:
            (root / ".episode_000000.inprogress").mkdir()
        raise loop.ContinuousLoopError("capture failed")

    monkeypatch.setattr(loop, "_run", fail_capture)
    args = type("Args", (), {
        "root_prefix": root_prefix, "policy_port": 8000,
        "robot_python": Path("python"), "task": "ring",
        "episode_time": 10.0, "tool_profile": "tool",
        "policy_replan_steps": 8, "policy_queue_low_watermark": 7,
        "max_force_n": 25.0, "max_torque_nm": 2.0,
    })()

    with pytest.raises(loop.ContinuousLoopError, match="capture failed"):
        loop._run_episode(
            args, 1, server=object(), model_revision="model", policy_epoch=0,
        )
    assert root.exists() is sealed


def test_failed_capture_preserves_recorder_rejected_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_prefix = tmp_path / "capture"
    root_prefix.mkdir()
    root = root_prefix / "001"
    rejected = root / "rejected_episodes" / "episode_rejected" / "raw.jsonl"
    monkeypatch.setattr(loop, "_post_json", lambda *_args, **_kwargs: {
        "runtime_session_id": "capture_001",
        "runtime_episode_id": loop.EPISODE_ID,
        "server_persistent": True,
    })

    def fail_capture(_command, **_kwargs):
        rejected.parent.mkdir(parents=True)
        rejected.write_text("{}\n", encoding="utf-8")
        raise loop.ContinuousLoopError("capture failed")

    monkeypatch.setattr(loop, "_run", fail_capture)
    args = type("Args", (), {
        "root_prefix": root_prefix, "policy_port": 8000,
        "robot_python": Path("python"), "task": "ring",
        "episode_time": 10.0, "tool_profile": "tool",
        "policy_replan_steps": 8, "policy_queue_low_watermark": 7,
        "max_force_n": 25.0, "max_torque_nm": 2.0,
    })()

    with pytest.raises(loop.ContinuousLoopError, match="capture failed"):
        loop._run_episode(
            args, 1, server=object(), model_revision="model", policy_epoch=0,
        )
    assert rejected.read_text(encoding="utf-8") == "{}\n"


def test_recorder_integrity_rejection_skips_episode_and_keeps_learner_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    root_prefix = tmp_path / "capture"
    root_prefix.mkdir()
    root = root_prefix / "001"
    rejected = root / "rejected_episodes" / "episode_rejected"
    monkeypatch.setattr(loop, "_post_json", lambda *_args, **_kwargs: {
        "runtime_session_id": "capture_001",
        "runtime_episode_id": loop.EPISODE_ID,
        "server_persistent": True,
    })

    def reject_capture(_command, **_kwargs):
        rejected.mkdir(parents=True)
        (rejected / "episode_result.json").write_text(
            '{"saved": false, "fatal_reason": '
            '"measured_tcp_pose native gap 109.3 ms exceeds 50.0 ms"}\n',
            encoding="utf-8",
        )
        raise loop.ContinuousLoopError("capture failed")

    monkeypatch.setattr(loop, "_run", reject_capture)
    monkeypatch.setattr(loop, "_wait_json", lambda *_args, **_kwargs: {
        "runtime_session_id": "capture_001",
        "runtime_episode_id": loop.EPISODE_ID,
        "episode_active": False,
        "learner_state": "running",
        "current_episode_sampled": False,
        "server_persistent": True,
    })
    args = type("Args", (), {
        "root_prefix": root_prefix, "policy_port": 8000,
        "robot_python": Path("python"), "task": "ring",
        "episode_time": 10.0, "tool_profile": "tool",
        "policy_replan_steps": 8, "policy_queue_low_watermark": 7,
        "max_force_n": 25.0, "max_torque_nm": 2.0,
    })()

    assert loop._run_episode(
        args, 1, server=object(), model_revision="model", policy_epoch=0,
    ) is None
    assert (rejected / "episode_result.json").is_file()
    output = capsys.readouterr().out
    assert "capture rejected" in output
    assert "learner continues" in output


def test_pose_ack_timeout_skips_episode_and_keeps_learner_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    root_prefix = tmp_path / "capture"
    root_prefix.mkdir()
    root = root_prefix / "001"
    monkeypatch.setattr(loop, "_post_json", lambda *_args, **_kwargs: {
        "runtime_session_id": "capture_001",
        "runtime_episode_id": loop.EPISODE_ID,
        "server_persistent": True,
    })

    def timeout_capture(_command, **_kwargs):
        root.mkdir()
        (root / ".episode_000000.inprogress").mkdir()
        raise loop.EpisodeLocalTransientError("integrated capture tempfail")

    monkeypatch.setattr(loop, "_run", timeout_capture)
    monkeypatch.setattr(loop, "_wait_json", lambda *_args, **_kwargs: {
        "runtime_session_id": "capture_001",
        "runtime_episode_id": loop.EPISODE_ID,
        "episode_active": False,
        "learner_state": "running",
        "current_episode_sampled": False,
        "server_persistent": True,
    })
    args = type("Args", (), {
        "root_prefix": root_prefix, "policy_port": 8000,
        "robot_python": Path("python"), "task": "ring",
        "episode_time": 10.0, "tool_profile": "tool",
        "policy_replan_steps": 8, "policy_queue_low_watermark": 7,
        "max_force_n": 25.0, "max_torque_nm": 2.0,
    })()

    assert loop._run_episode(
        args, 1, server=object(), model_revision="model", policy_epoch=0,
    ) is None
    assert not root.exists()
    output = capsys.readouterr().out
    assert "episode-local transient capture failure" in output
    assert "learner continues" in output


def test_capture_and_admission_output_is_compact(capsys, monkeypatch) -> None:
    admission_commands: list[list[str]] = []
    capture._print_payload({
        "status": "CAPTURE_SEALED",
        "episode_seal": {
            "episode_id": "episode_000000",
            "observation_count": 370,
            "policy_action_ack_count": 364,
            "human_action_ack_count": 2,
            "intervention_count": 69,
            "critic_updates": 2,
            "actor_updates": 1,
            "actor_parameter_broadcast_count": 0,
            "online_checkpoint_path": None,
            "current_episode_sampled_by_learner": False,
            "native_episode_result": {"stream_counts": {"camera": 99999}},
        },
    }, compact=True)
    output = capsys.readouterr().out
    assert output.count("\n") == 2
    assert "observations=370" in output
    assert "stream_counts" not in output

    monkeypatch.setattr(loop, "_report", lambda command: admission_commands.append(command) or {
        "status": "FORMAL_ONLINE_R_ADMITTED",
        "accepted_unique_r_transition_count": 364,
        "human_override_replay_count": 2,
        "total_unique_r_transition_count": 748,
        "training_starts_reached": True,
    })
    assert loop._admit(
        type("Args", (), {
            "model_python": Path("python"), "formal_r_root": Path("r"),
            "task_id": "task3", "output_root": Path("outputs/task3"),
            "deployed_actor_checkpoint": Path("checkpoints/task3/actor"),
            "detector_worker_socket": Path("/tmp/task3-detector.sock"),
        })(),
        Path("episode"),
        outcome="success",
    ) is True
    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert "human_expert=2" in output
    assert admission_commands[0][2:6] == [
        "--task-id", "task3", "--output-root", "outputs/task3",
    ]
    assert admission_commands[0][
        admission_commands[0].index("--deployed-actor-checkpoint") + 1
    ] == "checkpoints/task3/actor"
    assert admission_commands[0][
        admission_commands[0].index("--detector-worker-socket") + 1
    ] == "/tmp/task3-detector.sock"


def test_wrench_gap_rejects_only_episode_and_writes_no_replay(capsys, monkeypatch) -> None:
    monkeypatch.setattr(loop, "_report", lambda _command: {
        "status": "FORMAL_ONLINE_R_REJECTED",
        "reason": (
            "BRIDGE_POLICY_EXECUTION_OBSERVATION_MATERIALIZATION_FAILED:"
            "ValueError:WRENCH_SOURCE_GAP_EXCEEDED"
        ),
    })

    admitted = loop._admit(
        type("Args", (), {
            "model_python": Path("python"), "formal_r_root": Path("r"),
            "task_id": "task3", "output_root": Path("outputs/task3"),
            "deployed_actor_checkpoint": Path("checkpoints/task3/actor"),
        })(),
        Path("episode"),
        outcome="failure",
    )

    assert admitted is False
    assert "FORMAL_ONLINE_R_REJECTED" in capsys.readouterr().out


def test_intercamera_skew_is_an_episode_quality_rejection() -> None:
    assert bridge_tool._is_episode_quality_rejection(
        "BRIDGE_POLICY_EXECUTION_OBSERVATION_MATERIALIZATION_FAILED:"
        "ValueError:INTERCAMERA_SKEW_EXCEEDED"
    )
    assert not bridge_tool._is_episode_quality_rejection(
        "BRIDGE_POLICY_EXECUTION_GENERATION_INVALID"
    )


def test_incomplete_action7_ack_coverage_is_episode_quality_rejection() -> None:
    assert bridge_tool._is_episode_quality_rejection(
        "BRIDGE_POLICY_EXECUTION_ACTION_ACK_COVERAGE_MISMATCH"
    )
    assert not bridge_tool._is_episode_quality_rejection(
        "BRIDGE_POLICY_EXECUTION_ACTION_ACK_INVALID"
    )


def test_operator_success_detector_miss_is_episode_quality_rejection() -> None:
    assert bridge_tool._is_episode_quality_rejection(
        "BRIDGE_FORMAL_R_OPERATOR_SUCCESS_DETECTOR_MISS"
    )


def test_formal_admission_requires_explicit_deployed_actor_checkpoint(
    tmp_path: Path,
) -> None:
    args = type("Args", (), {
        "deployed_actor_checkpoint": None,
        "admit_formal_online_r": True,
    })()
    with pytest.raises(SystemExit, match="deployed-actor-checkpoint is required"):
        bridge_tool._resolve_actor_checkpoint(args, output_root=tmp_path)


def test_episode_admission_passes_success_and_failure_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        loop,
        "_admit",
        lambda _args, episode, *, outcome: calls.append((episode, outcome)) or True,
    )

    episode = tmp_path / "episode"
    loop._finish_episode(object(), episode=episode, outcome="success")
    loop._finish_episode(object(), episode=episode, outcome="failure")
    assert calls == [(episode, "success"), (episode, "failure")]


def test_loop_continues_after_rejected_episode_until_one_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume = tmp_path / "resume"
    (resume / "actor").mkdir(parents=True)
    results = iter((None, True, False))
    indices: list[int] = []

    class Process:
        pass

    monkeypatch.setattr(
        loop,
        "select_resume_or_seed_checkpoint",
        lambda *_args, **_kwargs: type("Selected", (), {"path": resume})(),
    )
    monkeypatch.setattr(loop.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(
        loop, "_start_detector_worker",
        lambda _args: (Process(), None, tmp_path / "detector.sock"),
    )
    monkeypatch.setattr(loop, "_stop_detector_worker", lambda *_args: None)
    monkeypatch.setattr(loop, "_wait_json", lambda *_args, **_kwargs: {
        "server_persistent": True,
        "learner_resume_checkpoint": str(resume.resolve()),
        "active_actor_model_revision": "model",
        "active_actor_checkpoint": str((resume / "actor").resolve()),
        "policy_epoch": 0,
    })
    monkeypatch.setattr(
        loop,
        "_run_episode",
        lambda _args, index, **_kwargs: indices.append(index) or next(results),
    )
    monkeypatch.setattr(loop, "_stop_server", lambda *_args, **_kwargs: None)
    args = type("Args", (), {
        "allow_development_policy_execution_smoke": True,
        "output_root": tmp_path,
            "model_python": Path("python"),
            "task_id": "task2",
            "task": "ring",
            "dataset_root": tmp_path / "datasets/task2_lerobotv3",
            "safety_config": None,
            "policy_port": 8000,
        "server_start_timeout": 1.0,
        "max_episodes": 2,
        "root_prefix": tmp_path / "capture",
    })()

    assert loop.run_loop(args) == 1
    assert indices == [0, 1, 2]


def test_loop_passes_selected_exact_resume_directly_to_unified_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume = tmp_path / "online/checkpoints/online_actor_critic_cycle_000100"
    (resume / "actor").mkdir(parents=True)
    commands: list[list[str]] = []

    class Process:
        pass

    monkeypatch.setattr(
        loop,
        "select_resume_or_seed_checkpoint",
        lambda *_args, **_kwargs: type("Selected", (), {"path": resume})(),
    )
    monkeypatch.setattr(
        loop.subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(command) or Process(),
    )
    monkeypatch.setattr(
        loop, "_start_detector_worker",
        lambda _args: (Process(), None, tmp_path / "detector.sock"),
    )
    monkeypatch.setattr(loop, "_stop_detector_worker", lambda *_args: None)
    monkeypatch.setattr(loop, "_wait_json", lambda *_args, **_kwargs: {
        "server_persistent": True,
        "learner_resume_checkpoint": str(resume.resolve()),
        "active_actor_model_revision": "model",
        "active_actor_checkpoint": str((resume / "actor").resolve()),
        "policy_epoch": 0,
    })
    monkeypatch.setattr(loop, "_run_episode", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(loop, "_stop_server", lambda *_args, **_kwargs: None)
    args = type("Args", (), {
        "allow_development_policy_execution_smoke": True,
        "output_root": tmp_path,
            "model_python": Path("python"),
            "task_id": "task2",
            "task": "ring",
            "dataset_root": tmp_path / "datasets/task2_lerobotv3",
            "safety_config": None,
            "policy_port": 8000,
        "server_start_timeout": 1.0,
        "max_episodes": 1,
        "root_prefix": tmp_path / "capture",
    })()

    assert loop.run_loop(args) == 0
    assert args.deployed_actor_checkpoint == (resume / "actor").resolve()
    command = commands[0]
    assert command[command.index("--learner-resume-checkpoint") + 1] == str(resume)
    assert "--allow-development-policy-execution-smoke" in command
    assert "--deployment-profile" not in command
    assert "--deployment-binding" not in command


def test_q_stops_before_admission_and_server_gets_graceful_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    commands: list[list[str]] = []
    posts: list[tuple[str, dict]] = []

    def post(url, payload):
        posts.append((url, payload))
        return {
            "runtime_session_id": "capture_001",
            "runtime_episode_id": loop.EPISODE_ID,
            "server_persistent": True,
            "operator_q_checkpoint_path": "/tmp/checkpoint",
        }

    monkeypatch.setattr(loop, "_post_json", post)
    monkeypatch.setattr(
        loop, "_run", lambda command, **_kwargs: commands.append(command)
    )
    monkeypatch.setattr(loop, "_wait_json", lambda *_args, **_kwargs: {
        "learner_state": "waiting_for_replay",
        "current_episode_sampled": False,
        "server_persistent": True,
    })
    monkeypatch.setattr("builtins.input", lambda _prompt: "q")
    monkeypatch.setattr(loop, "_admit", lambda *_args: calls.append("admit"))

    args = type("Args", (), {
        "root_prefix": tmp_path / "capture",
        "policy_port": 8000,
        "robot_python": Path("python"),
        "task": "ring", "episode_time": 10.0, "tool_profile": "tool",
        "policy_replan_steps": 8, "policy_queue_low_watermark": 7,
        "max_force_n": 25.0, "max_torque_nm": 2.0,
    })()
    assert loop._run_episode(
        args, 1, server=object(),
        model_revision="model", policy_epoch=0,
    ) is False
    assert calls == []
    assert posts[-1][0].endswith("/runtime/operator-q-checkpoint")
    assert posts[-1][1]["session_id"] == "capture_001"
    assert "--allow-development-policy-execution-smoke" in commands[0]
    assert "--compact-output" in commands[0]
    assert "--deployment-profile" not in commands[0]
    assert "--deployment-binding" not in commands[0]

    class Process:
        def __init__(self) -> None:
            self.signal = None
            self.killed = False

        def poll(self):
            return None

        def send_signal(self, signal):
            self.signal = signal

        def wait(self, timeout):
            assert timeout == 300
            return 0

        def kill(self):
            self.killed = True

    process = Process()
    loop._stop_server(process)
    assert process.signal == loop.signal.SIGINT
    assert process.killed is False
