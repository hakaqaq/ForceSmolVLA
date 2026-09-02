from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_forcerft_online_loop as loop  # noqa: E402
import run_forcerft_integrated_capture as capture  # noqa: E402


def test_task_output_root_and_replay_default_are_task_scoped(tmp_path: Path) -> None:
    args = loop.parse_args([
        "--task-id", "task2", "--output-root", str(tmp_path / "outputs/task2"),
        "--max-episodes", "1", "--root-prefix", str(tmp_path / "capture"),
        "--task", "ring",
    ])

    assert args.output_root == (tmp_path / "outputs/task2").resolve()
    assert args.formal_r_root == (tmp_path / "outputs/task2/online").resolve()
    assert args.allow_development_policy_execution_smoke is False
    assert not hasattr(args, "deployment_profile")
    assert not hasattr(args, "deployment_binding")


def test_online_capture_defaults_to_repository_dataset_root() -> None:
    args = loop.parse_args([
        "--task-id", "task2", "--max-episodes", "1", "--task", "ring",
    ])

    assert args.root_prefix == (ROOT / "datasets/task2_forcerft_online").resolve()


def test_online_capture_restart_uses_next_session_index(tmp_path: Path) -> None:
    prefix = tmp_path / "task2_forcerft_online"
    (tmp_path / "task2_forcerft_online_001").mkdir()
    (tmp_path / "task2_forcerft_online_002").mkdir()

    assert loop._next_capture_index(prefix) == 3


@pytest.mark.parametrize("sealed", [False, True])
def test_failed_capture_discards_only_unsealed_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sealed: bool,
) -> None:
    root_prefix = tmp_path / "capture"
    root = tmp_path / "capture_001"
    monkeypatch.setattr(loop, "_post_json", lambda *_args, **_kwargs: {
        "runtime_session_id": root.name,
        "runtime_episode_id": loop.EPISODE_ID,
        "server_persistent": True,
    })

    def fail_capture(_command, **_kwargs):
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


def test_capture_and_admission_output_is_compact(capsys, monkeypatch) -> None:
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

    monkeypatch.setattr(loop, "_report", lambda _command: {
        "status": "FORMAL_ONLINE_R_ADMITTED",
        "accepted_unique_r_transition_count": 364,
        "human_override_replay_count": 2,
        "total_unique_r_transition_count": 748,
        "training_starts_reached": True,
    })
    assert loop._admit(
        type("Args", (), {
            "model_python": Path("python"), "formal_r_root": Path("r"),
        })(),
        Path("episode"),
    ) is True
    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert "human_expert=2" in output


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
        })(),
        Path("episode"),
    )

    assert admitted is False
    assert "FORMAL_ONLINE_R_REJECTED" in capsys.readouterr().out


def test_episode_admission_is_called_once_only_for_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(
        loop, "_admit", lambda _args, episode: calls.append(episode) or True,
    )

    episode = tmp_path / "episode"
    loop._finish_episode(object(), episode=episode, outcome="success")
    assert calls == [episode]
    with pytest.raises(loop.ContinuousLoopError, match="OPERATOR_OUTCOME_NOT_SUCCESS"):
        loop._finish_episode(object(), episode=episode, outcome="failure")
    assert calls == [episode]


def test_loop_continues_after_rejected_episode_until_one_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume = tmp_path / "resume"
    results = iter((None, True, False))
    indices: list[int] = []

    class Process:
        pass

    monkeypatch.setattr(loop, "select_exact_resume_checkpoint", lambda _root: resume)
    monkeypatch.setattr(loop.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(loop, "_wait_json", lambda *_args, **_kwargs: {
        "server_persistent": True,
        "learner_resume_checkpoint": str(resume.resolve()),
        "active_actor_model_revision": "model",
        "policy_epoch": 0,
    })
    monkeypatch.setattr(
        loop,
        "_run_episode",
        lambda _args, index, **_kwargs: indices.append(index) or next(results),
    )
    monkeypatch.setattr(loop, "_stop_server", lambda _process: None)
    args = type("Args", (), {
        "allow_development_policy_execution_smoke": True,
        "output_root": tmp_path,
        "model_python": Path("python"),
        "task_id": "task2",
        "policy_port": 8000,
        "server_start_timeout": 1.0,
        "max_episodes": 2,
        "root_prefix": tmp_path / "capture",
    })()

    assert loop.run_loop(args) == 1
    assert indices == [1, 2, 3]


def test_loop_passes_selected_exact_resume_directly_to_unified_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume = tmp_path / "online/checkpoints/online_actor_critic_cycle_000100"
    commands: list[list[str]] = []

    class Process:
        pass

    monkeypatch.setattr(loop, "select_exact_resume_checkpoint", lambda _root: resume)
    monkeypatch.setattr(
        loop.subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(command) or Process(),
    )
    monkeypatch.setattr(loop, "_wait_json", lambda *_args, **_kwargs: {
        "server_persistent": True,
        "learner_resume_checkpoint": str(resume.resolve()),
        "active_actor_model_revision": "model",
        "policy_epoch": 0,
    })
    monkeypatch.setattr(loop, "_run_episode", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(loop, "_stop_server", lambda _process: None)
    args = type("Args", (), {
        "allow_development_policy_execution_smoke": True,
        "output_root": tmp_path,
        "model_python": Path("python"),
        "task_id": "task2",
        "policy_port": 8000,
        "server_start_timeout": 1.0,
        "max_episodes": 1,
        "root_prefix": tmp_path / "capture",
    })()

    assert loop.run_loop(args) == 0
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
            assert timeout == 60
            return 0

        def kill(self):
            self.killed = True

    process = Process()
    loop._stop_server(process)
    assert process.signal == loop.signal.SIGINT
    assert process.killed is False
