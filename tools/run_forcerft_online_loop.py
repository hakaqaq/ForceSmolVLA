#!/usr/bin/env python3
"""Coordinate persistent ForceRFT HIL capture and online Actor/Learner training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forcesmolvla.rft.online.actor_learner_runtime import (  # noqa: E402
    select_resume_or_seed_checkpoint,
)

MODEL_PYTHON = Path("/home/rlc123/anaconda3/envs/forcesmolvla/bin/python")
ROBOT_PYTHON = Path("/home/rlc123/fr3_client_ws/.venv/bin/python")
EPISODE_ID = "episode_000000"


class ContinuousLoopError(RuntimeError):
    pass


class EpisodeLocalTransientError(ContinuousLoopError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuousLoopError(message)


def _run(
    command: list[str], *, capture: bool = False, echo_captured: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=capture,
        check=False,
    )
    if capture and (echo_captured or result.returncode != 0):
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
    if result.returncode == os.EX_TEMPFAIL:
        raise EpisodeLocalTransientError(
            f"FORCERFT_ONLINE_EPISODE_LOCAL_TRANSIENT:{command[1]}"
        )
    require(result.returncode == 0, f"FORCERFT_ONLINE_COMMAND_FAILED:{command[1]}")
    return result


def _report(command: list[str]) -> dict[str, Any]:
    output = _run(command, capture=True, echo_captured=False).stdout
    decoder = json.JSONDecoder()
    for start in range(len(output) - 1, -1, -1):
        if output[start] != "{":
            continue
        try:
            value, end = decoder.raw_decode(output, start)
        except json.JSONDecodeError:
            continue
        if not output[end:].strip() and isinstance(value, dict):
            return value
    require("{" in output, "FORCERFT_ONLINE_COMMAND_REPORT_MISSING")
    raise ContinuousLoopError("FORCERFT_ONLINE_COMMAND_REPORT_INVALID")


def _admit(
    args: argparse.Namespace,
    episode: Path,
    *,
    outcome: str,
    actor_checkpoint: Path | None = None,
) -> bool:
    """Materialize the sealed episode and append it exactly once to Online-R."""

    command = [
        str(args.model_python), str(ROOT / "tools/run_forcerft_production_bridge.py"),
        "--task-id", args.task_id, "--output-root", str(args.output_root),
        "--episode", str(episode), "--state-root", str(args.formal_r_root),
        "--deployed-actor-checkpoint", str(
            actor_checkpoint or args.deployed_actor_checkpoint
        ),
        "--operator-task-outcome", outcome, "--admit-formal-online-r",
    ]
    detector_socket = getattr(args, "detector_worker_socket", None)
    if detector_socket is not None:
        command.extend(["--detector-worker-socket", str(detector_socket)])
    started = time.monotonic()
    report = _report(command)
    if report.get("status") == "FORMAL_ONLINE_R_REJECTED":
        print(
            f"[admission] status={report['status']} "
            f"reason={report.get('reason')} replay_written=0"
        )
        return False
    require(
        report.get("status") == "FORMAL_ONLINE_R_ADMITTED",
        "FORCERFT_ONLINE_ADMISSION_FAILED",
    )
    print(
        f"[admission] status={report['status']} "
        f"accepted={report.get('accepted_unique_r_transition_count')} "
        f"human_expert={report.get('human_override_replay_count')} "
        f"total={report.get('total_unique_r_transition_count')} "
        f"training_started={str(bool(report.get('training_starts_reached'))).lower()} "
        f"elapsed={time.monotonic() - started:.1f}s"
    )
    return True


def _finish_episode(
    args: argparse.Namespace,
    *,
    episode: Path,
    outcome: str,
    actor_checkpoint: Path | None = None,
) -> bool:
    require(
        outcome in {"success", "failure"},
        "FORCERFT_ONLINE_OPERATOR_OUTCOME_INVALID",
    )
    if actor_checkpoint is None:
        return _admit(args, episode, outcome=outcome)
    return _admit(
        args, episode, outcome=outcome, actor_checkpoint=actor_checkpoint
    )


def _post_json(
    url: str, payload: Mapping[str, Any], *, timeout: float = 10.0
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
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


def _stop_server(
    process: subprocess.Popen[Any], *, policy_port: int | None = None
) -> None:
    """Quiesce and save before asking the persistent server to exit."""

    if process.poll() is not None:
        return
    if policy_port is not None:
        try:
            report = _post_json(
                f"http://127.0.0.1:{policy_port}/runtime/quiesce-and-save",
                {"reason": "online_loop_shutdown"},
                timeout=300.0,
            )
            require(
                report.get("quiesced") is True,
                "FORCERFT_ONLINE_SERVER_QUIESCE_FAILED",
            )
        except (OSError, URLError, ContinuousLoopError, ValueError):
            # SIGINT remains the recovery path if the local HTTP server is gone.
            pass
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _start_detector_worker(
    args: argparse.Namespace,
) -> tuple[subprocess.Popen[Any], tempfile.TemporaryDirectory[str], Path]:
    directory = tempfile.TemporaryDirectory(prefix="forcerft-reward-worker-")
    socket_path = Path(directory.name) / "detector.sock"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    process = subprocess.Popen([
        shutil.which("conda") or "conda", "run", "--no-capture-output",
        "-n", "conrft_reward", "python",
        str(ROOT / "tools/run_forcerft_production_bridge.py"),
        "--task-id", args.task_id, "--output-root", str(args.output_root),
        "--detector-worker-socket", str(socket_path), "--serve-detector-worker",
    ], cwd=ROOT, env=environment)
    deadline = time.monotonic() + args.server_start_timeout
    while not socket_path.exists():
        if process.poll() is not None:
            directory.cleanup()
            raise ContinuousLoopError("FORCERFT_REWARD_DETECTOR_WORKER_EXITED")
        if time.monotonic() >= deadline:
            process.terminate()
            process.wait(timeout=10)
            directory.cleanup()
            raise ContinuousLoopError("FORCERFT_REWARD_DETECTOR_WORKER_TIMEOUT")
        time.sleep(0.1)
    print(f"[reward] persistent detector ready socket={socket_path}")
    return process, directory, socket_path


def _stop_detector_worker(
    process: subprocess.Popen[Any], socket_path: Path
) -> None:
    if process.poll() is not None:
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(5.0)
            connection.connect(str(socket_path))
            connection.sendall(b'{"shutdown":true}\n')
            connection.recv(1024)
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _next_capture_index(root_prefix: Path) -> int:
    indices = [
        int(path.name)
        for path in root_prefix.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    return max(indices, default=-1) + 1


def _discard_unsealed_capture(root: Path) -> None:
    seal = (
        root / "integrated_capture" / EPISODE_ID
        / "streams" / "policy_execute_episode_seal.json"
    )
    rejected = root / "rejected_episodes"
    if root.is_dir() and not seal.is_file() and not rejected.is_dir():
        shutil.rmtree(root)


def _recorder_rejection_reason(root: Path) -> str | None:
    for path in sorted((root / "rejected_episodes").glob("*/episode_result.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        reason = result.get("fatal_reason") if isinstance(result, dict) else None
        if (
            isinstance(result, dict)
            and result.get("saved") is False
            and isinstance(reason, str)
            and reason
        ):
            return reason
    return None


def _run_episode(
    args: argparse.Namespace,
    index: int,
    *,
    server: subprocess.Popen[Any],
    model_revision: str | None = None,
    policy_epoch: int | None = None,
) -> bool | None:
    root = (args.root_prefix / f"{index:03d}").resolve()
    session_id = f"{args.root_prefix.name}_{index:03d}"
    require(not root.exists(), "FORCERFT_ONLINE_CAPTURE_ROOT_EXISTS")
    prepare_identity = {
        "session_id": session_id,
        "episode_id": EPISODE_ID,
    }
    metadata = _post_json(
        f"http://127.0.0.1:{args.policy_port}/runtime/prepare-episode",
        prepare_identity,
    )
    compatibility_identity = model_revision is not None and policy_epoch is not None
    model_revision = str(
        metadata.get("active_actor_model_revision", model_revision or "")
    )
    policy_epoch = int(metadata.get("policy_epoch", policy_epoch or 0))
    actor_checkpoint_value = str(
        metadata.get(
            "base_actor_checkpoint", metadata.get("active_actor_checkpoint", "")
        )
    )
    actor_checkpoint = (
        Path(actor_checkpoint_value).resolve() if actor_checkpoint_value else None
    )
    require(
        metadata.get("runtime_session_id") == session_id
        and metadata.get("runtime_episode_id") == EPISODE_ID
        and metadata.get("server_persistent") is True,
        "FORCERFT_ONLINE_SERVER_IDENTITY_MISMATCH",
    )
    require(
        model_revision
        and policy_epoch >= 0
        and (
            compatibility_identity
            or actor_checkpoint is not None and actor_checkpoint.is_dir()
        ),
        "FORCERFT_ONLINE_SERVER_IDENTITY_MISMATCH",
    )
    identity = {
        **prepare_identity,
        "policy_revision": model_revision,
    }
    try:
        _run([
            str(args.robot_python), str(ROOT / "tools/run_forcerft_integrated_capture.py"),
            "--mode", "policy-execute", "--allow-development-policy-execution-smoke",
            "--async-learner", "--root", str(root), "--task", args.task,
            "--episodes", "1", "--episode-time", str(args.episode_time),
            "--tool-profile", args.tool_profile, "--session-id", session_id,
            "--episode-id", EPISODE_ID, "--policy-revision", model_revision,
            "--policy-epoch", str(policy_epoch), "--takeover-generation", "0",
            "--policy-host", "127.0.0.1", "--policy-port", str(args.policy_port),
            "--policy-replan-steps", str(args.policy_replan_steps),
            "--policy-queue-low-watermark", str(args.policy_queue_low_watermark),
            "--max-force-n", str(args.max_force_n),
            "--max-torque-nm", str(args.max_torque_nm), "--launch", "--compact-output",
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
    except ContinuousLoopError as error:
        _discard_unsealed_capture(root)
        rejection_reason = _recorder_rejection_reason(root)
        episode_local_transient = isinstance(error, EpisodeLocalTransientError)
        if rejection_reason is None and not episode_local_transient:
            raise
        status = _wait_json(
            f"http://127.0.0.1:{args.policy_port}/runtime/status",
            process=server,
            timeout=10.0,
        )
        require(
            status.get("runtime_session_id") == session_id
            and status.get("runtime_episode_id") == EPISODE_ID
            and status.get("episode_active") is False
            and status.get("learner_state") != "failed"
            and status.get("current_episode_sampled") is False
            and status.get("server_persistent") is True,
            "FORCERFT_ONLINE_REJECTED_CAPTURE_RUNTIME_INVALID",
        )
        print(
            f"[episode] capture rejected session={session_id}; "
            f"reason={rejection_reason or 'episode-local transient capture failure'}; "
            "replay_written=0; learner continues"
        )
        return None
    except (OSError, KeyboardInterrupt):
        _discard_unsealed_capture(root)
        raise
    outcome = input("operator_task_outcome [success/failure/q]: ").strip().lower()
    require(outcome in {"success", "failure", "q"}, "FORCERFT_ONLINE_OPERATOR_OUTCOME_INVALID")
    if outcome == "q":
        checkpoint = _post_json(
            f"http://127.0.0.1:{args.policy_port}/runtime/operator-q-checkpoint",
            identity,
        ).get("operator_q_checkpoint_path")
        print(f"[learner] operator-q checkpoint={checkpoint or 'none'}")
        return False
    if not _finish_episode(
        args,
        episode=root / "episodes" / EPISODE_ID,
        outcome=outcome,
        actor_checkpoint=actor_checkpoint,
    ):
        print(f"[episode] rejected session={session_id}; continuing with next capture")
        return None
    return True


def run_loop(args: argparse.Namespace) -> int:
    require(
        args.allow_development_policy_execution_smoke,
        "FORCERFT_ONLINE_ROBOT_EXECUTION_FLAG_REQUIRED",
    )
    resume = select_resume_or_seed_checkpoint(
        args.output_root,
        configured_seed_bundle=getattr(args, "stage3_seed_bundle", None),
    ).path
    server_command = [
        str(args.model_python), str(ROOT / "tools/serve_forcerft_actor_learner.py"),
        "--task-id", args.task_id, "--output-root", str(args.output_root),
        "--task", args.task,
        "--dataset-root", str(args.dataset_root),
        "--session-id", "waiting-for-episode", "--episode-id", EPISODE_ID,
        "--learner-resume-checkpoint", str(resume),
        "--allow-development-policy-execution-smoke",
        "--host", "127.0.0.1", "--port", str(args.policy_port),
    ]
    if args.safety_config is not None:
        server_command.extend(["--safety-config", str(args.safety_config)])
    if getattr(args, "stage3_seed_bundle", None) is not None:
        server_command.extend(
            ["--stage3-seed-bundle", str(args.stage3_seed_bundle)]
        )
    server = subprocess.Popen(server_command, cwd=ROOT, env=os.environ.copy())
    detector_process = detector_directory = detector_socket = None
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
        base_actor_checkpoint = str(
            metadata.get(
                "base_actor_checkpoint",
                metadata.get("active_actor_checkpoint", ""),
            )
        )
        args.deployed_actor_checkpoint = Path(base_actor_checkpoint).resolve()
        require(
            str(metadata.get("active_actor_model_revision", ""))
            and int(metadata.get("policy_epoch", -1)) >= 0
            and base_actor_checkpoint
            and args.deployed_actor_checkpoint.is_dir(),
            "FORCERFT_ONLINE_SERVER_METADATA_INVALID",
        )
        detector_process, detector_directory, detector_socket = (
            _start_detector_worker(args)
        )
        args.detector_worker_socket = detector_socket
        args.root_prefix.mkdir(parents=True, exist_ok=True)
        index = _next_capture_index(args.root_prefix)
        while completed < args.max_episodes:
            result = _run_episode(
                args,
                index,
                server=server,
            )
            index += 1
            if result is False:
                break
            if result is True:
                completed += 1
    finally:
        if detector_process is not None and detector_socket is not None:
            _stop_detector_worker(detector_process, detector_socket)
        if detector_directory is not None:
            detector_directory.cleanup()
        _stop_server(server, policy_port=args.policy_port)
    return completed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="task2")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--safety-config", type=Path)
    parser.add_argument("--stage3-seed-bundle", type=Path)
    parser.add_argument("--max-episodes", type=int, required=True)
    parser.add_argument("--root-prefix", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-time", type=float, default=60.0)
    parser.add_argument("--tool-profile", default="onrobot_robotiq")
    parser.add_argument("--policy-replan-steps", type=int, default=8)
    parser.add_argument("--policy-queue-low-watermark", type=int, default=7)
    parser.add_argument("--max-force-n", type=float, default=25.0)
    parser.add_argument("--max-torque-nm", type=float, default=2.0)
    parser.add_argument("--formal-r-root", type=Path)
    parser.add_argument(
        "--allow-development-policy-execution-smoke",
        action="store_true",
        help="explicitly enable the existing supervised HIL robot-execution path",
    )
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
    from forcesmolvla.training_runtime import (
        resolve_task_dataset_root,
        resolve_task_output_root,
    )

    args.root_prefix = (
        ROOT / "datasets" / f"{args.task_id}_forcerft_online"
        if args.root_prefix is None
        else args.root_prefix
    ).resolve()
    args.output_root = resolve_task_output_root(
        ROOT, task_id=args.task_id, output_root=args.output_root
    )
    args.dataset_root = resolve_task_dataset_root(
        ROOT, task_id=args.task_id, dataset_root=args.dataset_root
    )
    if args.safety_config is not None:
        args.safety_config = args.safety_config.resolve()
    args.formal_r_root = (
        args.output_root / "online"
        if args.formal_r_root is None else args.formal_r_root.resolve()
    )
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
