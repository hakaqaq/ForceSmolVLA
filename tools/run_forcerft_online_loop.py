#!/usr/bin/env python3
"""Coordinate persistent ForceRFT HIL capture and online Actor/Learner training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODEL_PYTHON = Path("/home/rlc123/anaconda3/envs/forcesmolvla/bin/python")
ROBOT_PYTHON = Path("/home/rlc123/fr3_client_ws/.venv/bin/python")
EPISODE_ID = "episode_000000"
class ContinuousLoopError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuousLoopError(message)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContinuousLoopError(f"FORCERFT_ONLINE_JSON_INVALID:{path}") from error
    require(isinstance(value, dict), f"FORCERFT_ONLINE_JSON_OBJECT_REQUIRED:{path}")
    return value


@dataclass(frozen=True)
class Deployment:
    profile: Path
    binding: Path
    trusted_binding: str


def _deployment(args: argparse.Namespace) -> Deployment:
    profile = _json(args.deployment_profile)
    require(
        profile.get("artifact_status") == "development_only",
        "FORCERFT_ONLINE_DEPLOYMENT_PROFILE_INVALID",
    )
    binding = args.deployment_binding or Path(str(profile["deployment_binding"]))
    if not binding.is_absolute():
        binding = ROOT / binding
    trusted = args.trusted_deployment_binding_sha256 or str(
        profile["deployment_binding_sha256"]
    )
    require(binding.is_file() and len(trusted) == 64, "FORCERFT_ONLINE_BINDING_INVALID")
    return Deployment(args.deployment_profile.resolve(), binding.resolve(), trusted)


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=capture,
        check=False,
    )
    if capture:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
    require(result.returncode == 0, f"FORCERFT_ONLINE_COMMAND_FAILED:{command[1]}")
    return result


def _report(command: list[str]) -> dict[str, Any]:
    output = _run(command, capture=True).stdout
    start = output.find("{")
    require(start >= 0, "FORCERFT_ONLINE_COMMAND_REPORT_MISSING")
    try:
        value = json.loads(output[start:])
    except json.JSONDecodeError as error:
        raise ContinuousLoopError("FORCERFT_ONLINE_COMMAND_REPORT_INVALID") from error
    require(isinstance(value, dict), "FORCERFT_ONLINE_COMMAND_REPORT_INVALID")
    return value


def _admit(args: argparse.Namespace, episode: Path) -> None:
    """Materialize the sealed episode and append it exactly once to Online-R."""

    report = _report([
        str(args.model_python), str(ROOT / "tools/run_forcerft_production_bridge.py"),
        "--episode", str(episode), "--state-root", str(args.formal_r_root),
        "--operator-task-outcome", "success", "--admit-formal-online-r",
    ])
    require(
        report.get("status") == "FORMAL_ONLINE_R_ADMITTED",
        "FORCERFT_ONLINE_ADMISSION_FAILED",
    )


def _finish_episode(
    args: argparse.Namespace,
    *,
    episode: Path,
    outcome: str,
) -> None:
    require(outcome == "success", "FORCERFT_ONLINE_OPERATOR_OUTCOME_NOT_SUCCESS")
    _admit(args, episode)


def _post_json(url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10.0) as response:
        value = json.loads(response.read())
    require(isinstance(value, dict), "FORCERFT_ONLINE_SERVER_RESPONSE_INVALID")
    return value


def _wait_json(
    url: str,
    *,
    process: subprocess.Popen[Any],
    timeout: float,
    ready: Callable[[Mapping[str, Any]], bool] = lambda _value: True,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        require(process.poll() is None, "FORCERFT_ONLINE_SERVER_EXITED")
        try:
            with urlopen(url, timeout=2.0) as response:
                value = json.loads(response.read())
            if isinstance(value, dict) and ready(value):
                return value
        except (OSError, URLError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(0.25)
    raise ContinuousLoopError(f"FORCERFT_ONLINE_SERVER_TIMEOUT:{last_error}")


def _stop_server(process: subprocess.Popen[Any]) -> None:
    """Let the server finish its current optimizer step and exact-resume save."""

    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run_episode(
    args: argparse.Namespace,
    index: int,
    *,
    server: subprocess.Popen[Any],
    deployment: Deployment,
    model_revision: str,
    policy_epoch: int,
) -> bool:
    root = Path(f"{args.root_prefix}_{index:03d}").resolve()
    session_id = root.name
    require(not root.exists(), "FORCERFT_ONLINE_CAPTURE_ROOT_EXISTS")
    identity = {
        "session_id": session_id,
        "episode_id": EPISODE_ID,
        "policy_revision": model_revision,
    }
    metadata = _post_json(
        f"http://127.0.0.1:{args.policy_port}/runtime/prepare-episode", identity,
    )
    require(
        metadata.get("runtime_session_id") == session_id
        and metadata.get("runtime_episode_id") == EPISODE_ID
        and metadata.get("server_persistent") is True,
        "FORCERFT_ONLINE_SERVER_IDENTITY_MISMATCH",
    )
    _run([
        str(args.robot_python), str(ROOT / "tools/run_forcerft_integrated_capture.py"),
        "--mode", "policy-execute", "--allow-development-policy-execution-smoke",
        "--async-learner", "--root", str(root), "--task", args.task,
        "--episodes", "1", "--episode-time", str(args.episode_time),
        "--tool-profile", args.tool_profile, "--session-id", session_id,
        "--episode-id", EPISODE_ID, "--policy-revision", model_revision,
        "--policy-epoch", str(policy_epoch), "--takeover-generation", "0",
        "--deployment-profile", str(deployment.profile),
        "--deployment-binding", str(deployment.binding),
        "--policy-host", "127.0.0.1", "--policy-port", str(args.policy_port),
        "--policy-replan-steps", str(args.policy_replan_steps),
        "--policy-queue-low-watermark", str(args.policy_queue_low_watermark),
        "--max-force-n", str(args.max_force_n),
        "--max-torque-nm", str(args.max_torque_nm), "--launch",
    ])
    status = _wait_json(
        f"http://127.0.0.1:{args.policy_port}/runtime/status",
        process=server,
        timeout=10.0,
    )
    require(
        status.get("learner_state") != "failed"
        and status.get("current_episode_sampled") is False
        and status.get("server_persistent") is True,
        "FORCERFT_ONLINE_LEARNER_INVALID",
    )
    outcome = input("operator_task_outcome [success/failure/q]: ").strip().lower()
    require(outcome in {"success", "failure", "q"}, "FORCERFT_ONLINE_OPERATOR_OUTCOME_INVALID")
    if outcome == "q":
        return False
    _finish_episode(args, episode=root / "episodes" / EPISODE_ID, outcome=outcome)
    return True


def run_loop(args: argparse.Namespace) -> int:
    deployment = _deployment(args)
    resume = args.output_root / "offline/checkpoints/offline_actor_critic_cycle_000210"
    require(resume.is_dir(), "FORCERFT_OFFLINE_EXACT_RESUME_MISSING")
    server_command = [
        str(args.model_python), str(ROOT / "tools/serve_forcerft_actor_learner.py"),
        "--task-id", args.task_id, "--output-root", str(args.output_root),
        "--session-id", "waiting-for-episode", "--episode-id", EPISODE_ID,
        "--learner-resume-checkpoint", str(resume),
        "--host", "127.0.0.1", "--port", str(args.policy_port),
    ]
    server = subprocess.Popen(server_command, cwd=ROOT, env=os.environ.copy())
    completed = 0
    try:
        metadata = _wait_json(
            f"http://127.0.0.1:{args.policy_port}/metadata",
            process=server,
            timeout=args.server_start_timeout,
            ready=lambda value: value.get("server_persistent") is True,
        )
        require(
            metadata.get("learner_resume_checkpoint") == str(resume.resolve()),
            "FORCERFT_ONLINE_RESUME_CHECKPOINT_MISMATCH",
        )
        model_revision = str(metadata.get("active_actor_model_revision", ""))
        policy_epoch = int(metadata.get("policy_epoch", -1))
        require(model_revision and policy_epoch >= 0, "FORCERFT_ONLINE_SERVER_METADATA_INVALID")
        for index in range(1, args.max_episodes + 1):
            if not _run_episode(
                args,
                index,
                server=server,
                deployment=deployment,
                model_revision=model_revision,
                policy_epoch=policy_epoch,
            ):
                break
            completed += 1
    finally:
        _stop_server(server)
    return completed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="task2")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-episodes", type=int, required=True)
    parser.add_argument("--root-prefix", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-time", type=float, default=60.0)
    parser.add_argument("--tool-profile", default="onrobot_robotiq")
    parser.add_argument("--policy-replan-steps", type=int, default=8)
    parser.add_argument("--policy-queue-low-watermark", type=int, default=7)
    parser.add_argument("--max-force-n", type=float, default=25.0)
    parser.add_argument("--max-torque-nm", type=float, default=2.0)
    parser.add_argument("--formal-r-root", type=Path)
    parser.add_argument("--deployment-profile", type=Path, required=True)
    parser.add_argument("--deployment-binding", type=Path)
    parser.add_argument("--trusted-deployment-binding-sha256")
    parser.add_argument("--model-python", type=Path, default=MODEL_PYTHON)
    parser.add_argument("--robot-python", type=Path, default=ROBOT_PYTHON)
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--server-start-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    if (
        args.max_episodes <= 0
        or args.episode_time <= 0
        or not 0 < args.policy_queue_low_watermark < args.policy_replan_steps <= 50
        or args.max_force_n <= 0
        or args.max_torque_nm <= 0
        or args.policy_port <= 0
    ):
        parser.error("invalid continuous-loop limits")
    from forcesmolvla.training_runtime import resolve_task_output_root

    args.root_prefix = args.root_prefix.resolve()
    args.output_root = resolve_task_output_root(
        ROOT, task_id=args.task_id, output_root=args.output_root
    )
    args.formal_r_root = (
        args.output_root / "online"
        if args.formal_r_root is None else args.formal_r_root.resolve()
    )
    args.deployment_profile = args.deployment_profile.resolve()
    if args.deployment_binding is not None:
        args.deployment_binding = args.deployment_binding.resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        completed = run_loop(args)
    except (ContinuousLoopError, OSError) as error:
        print(f"[online] STOP:{error}", file=sys.stderr)
        return 2
    print(f"[online] complete episodes={completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
