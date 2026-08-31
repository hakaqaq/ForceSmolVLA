from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_forcerft_online_loop as loop  # noqa: E402


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_task_output_root_and_replay_default_are_task_scoped(tmp_path: Path) -> None:
    profile = tmp_path / "deployment.json"
    _write(profile, {})
    args = loop.parse_args([
        "--task-id", "task2", "--output-root", str(tmp_path / "outputs/task2"),
        "--max-episodes", "1", "--root-prefix", str(tmp_path / "capture"),
        "--task", "ring", "--deployment-profile", str(profile),
    ])

    assert args.output_root == (tmp_path / "outputs/task2").resolve()
    assert args.formal_r_root == (tmp_path / "outputs/task2/online").resolve()


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


def test_q_stops_before_admission_and_server_gets_graceful_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(loop, "_post_json", lambda *_args, **_kwargs: {
        "runtime_session_id": "capture_001",
        "runtime_episode_id": loop.EPISODE_ID,
        "server_persistent": True,
    })
    monkeypatch.setattr(loop, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loop, "_wait_json", lambda *_args, **_kwargs: {
        "learner_state": "waiting_for_replay",
        "current_episode_sampled": False,
        "server_persistent": True,
    })
    monkeypatch.setattr("builtins.input", lambda _prompt: "q")
    monkeypatch.setattr(loop, "_admit", lambda *_args: calls.append("admit"))

    deployment = loop.Deployment(tmp_path / "profile", tmp_path / "binding", "a" * 64)
    args = type("Args", (), {
        "root_prefix": tmp_path / "capture",
        "policy_port": 8000,
        "robot_python": Path("python"),
        "task": "ring", "episode_time": 10.0, "tool_profile": "tool",
        "policy_replan_steps": 8, "policy_queue_low_watermark": 7,
        "max_force_n": 25.0, "max_torque_nm": 2.0,
    })()
    assert loop._run_episode(
        args, 1, server=object(), deployment=deployment,
        model_revision="model", policy_epoch=0,
    ) is False
    assert calls == []

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
