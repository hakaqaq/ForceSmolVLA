from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_forcerft_online_loop as loop  # noqa: E402


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


def test_episode_admission_is_called_once_only_for_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(loop, "_admit", lambda _args, episode: calls.append(episode))

    episode = tmp_path / "episode"
    loop._finish_episode(object(), episode=episode, outcome="success")
    assert calls == [episode]
    with pytest.raises(loop.ContinuousLoopError, match="OPERATOR_OUTCOME_NOT_SUCCESS"):
        loop._finish_episode(object(), episode=episode, outcome="failure")
    assert calls == [episode]


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
    monkeypatch.setattr(loop, "_post_json", lambda *_args, **_kwargs: {
        "runtime_session_id": "capture_001",
        "runtime_episode_id": loop.EPISODE_ID,
        "server_persistent": True,
    })
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
    assert "--allow-development-policy-execution-smoke" in commands[0]
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
