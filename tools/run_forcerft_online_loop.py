#!/usr/bin/env python3
"""Coordinate persistent ForceRFT HIL capture and online Actor/Learner training."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forcesmolvla.rft.online.integrated_capture_backend import (  # noqa: E402
    CAPTURE_DISCARDED_EXIT_CODE,
    CAPTURE_EXITED_EXIT_CODE,
    CAPTURE_TIMED_OUT_EXIT_CODE,
)
from forcesmolvla.rft.online.replay_training import (  # noqa: E402
    load_common_actor_critic_config,
)
from forcesmolvla.rft.online.residual_actor_critic_runtime import (  # noqa: E402
    ONLINE_ADAPTATION_DIRECTORY_NAME,
    load_checkpoint_training_config,
    select_resume_or_bootstrap_checkpoint,
)
from forcesmolvla.rft.online.schedule_migration import (  # noqa: E402
    schedule_migration_required,
)

MODEL_PYTHON = Path("/home/rlc123/anaconda3/envs/forcesmolvla/bin/python")
ROBOT_PYTHON = Path("/home/rlc123/fr3_client_ws/.venv/bin/python")
EPISODE_ID = "episode_000000"
SERVER_SUMMARY_INTERVAL_SECONDS = 5.0


class ContinuousLoopError(RuntimeError):
    pass


class EpisodeLocalTransientError(ContinuousLoopError):
    pass


class CaptureDiscardedError(EpisodeLocalTransientError):
    pass


class CaptureTimedOutError(EpisodeLocalTransientError):
    pass


class CaptureOperatorExit(ContinuousLoopError):
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
    integrated_capture = (
        len(command) > 1
        and Path(command[1]).name == "run_forcerft_integrated_capture.py"
    )
    if integrated_capture and result.returncode == CAPTURE_DISCARDED_EXIT_CODE:
        raise CaptureDiscardedError("FORCERFT_ONLINE_CAPTURE_DISCARDED")
    if integrated_capture and result.returncode == CAPTURE_TIMED_OUT_EXIT_CODE:
        raise CaptureTimedOutError("FORCERFT_ONLINE_CAPTURE_TIMED_OUT")
    if integrated_capture and result.returncode == CAPTURE_EXITED_EXIT_CODE:
        raise CaptureOperatorExit("FORCERFT_ONLINE_CAPTURE_EXITED")
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
) -> dict[str, Any] | None:
    """Materialize the sealed episode and append it exactly once to Online-R."""

    command = [
        str(args.model_python), str(ROOT / "tools/run_forcerft_production_bridge.py"),
        "--task-id", args.task_id, "--output-root", str(args.output_root),
        "--episode", str(episode), "--state-root", str(args.ack_replay_root),
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
        return None
    require(
        report.get("status") == "FORMAL_ONLINE_R_ADMITTED",
        "FORCERFT_ONLINE_ADMISSION_FAILED",
    )
    timings = report.get("admission_timing_seconds", {})
    timing_text = ""
    if isinstance(timings, Mapping):
        timing_text = (
            f" prepare={float(timings.get('data_preparation', 0.0)):.3f}s"
            f" detector={float(timings.get('reward_detection', 0.0)):.3f}s"
            f" transitions={float(timings.get('transition_build', 0.0)):.3f}s"
            f" persistence={float(timings.get('persistence', 0.0)):.3f}s"
        )
    print(
        f"[admission] status={report['status']} "
        f"accepted={report.get('accepted_unique_r_transition_count')} "
        f"human_expert={report.get('human_override_replay_count')} "
        f"total={report.get('total_unique_r_transition_count')} "
        f"task_success={str(bool(report.get('task_success'))).lower()} "
        f"autonomous_success={str(bool(report.get('autonomous_success'))).lower()} "
        f"assisted_success={str(bool(report.get('assisted_success'))).lower()} "
        f"takeovers={int(report.get('takeover_count', 0))} "
        f"human_control_s={float(report.get('human_control_duration_s', 0.0)):.2f} "
        f"elapsed={time.monotonic() - started:.1f}s"
        f"{timing_text}"
    )
    return report


def _finish_episode(
    args: argparse.Namespace,
    *,
    episode: Path,
    outcome: str,
    actor_checkpoint: Path | None = None,
) -> dict[str, Any] | None:
    require(
        outcome in {"success", "failure"},
        "FORCERFT_ONLINE_OPERATOR_OUTCOME_INVALID",
    )
    if actor_checkpoint is None:
        return _admit(args, episode, outcome=outcome)
    return _admit(
        args, episode, outcome=outcome, actor_checkpoint=actor_checkpoint
    )


def _notify_admission_committed(
    args: argparse.Namespace,
    *,
    identity: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    admission_id = str(admission.get("admission_id", ""))
    require(admission_id, "FORCERFT_ONLINE_ADMISSION_ID_MISSING")
    result = _post_json(
        f"http://127.0.0.1:{args.policy_port}/runtime/notify-admission-committed",
        {
            "session_id": identity["session_id"],
            "episode_id": identity["episode_id"],
            "admission_id": admission_id,
        },
    )
    require(
        result.get("status")
        in {"FORMAL_ADMISSION_REGISTERED", "FORMAL_ADMISSION_ALREADY_REGISTERED"}
        and result.get("admission_id") == admission_id,
        "FORCERFT_ONLINE_ADMISSION_NOTIFICATION_FAILED",
    )
    print(
        "[admission-notify] "
        f"admission={admission_id} "
        "status=registered learner_woken=true "
        f"learner_state={result.get('learner_state')} "
        f"training_started={str(result.get('learner_state') in {'ack_critic_warmup', 'residual_actor_critic_training'}).lower()}"
    )
    return result


def _post_json(
    url: str, payload: Mapping[str, Any], *, timeout: float = 10.0
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except HTTPError as error:
        raw = error.read(2048).decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = None
        if isinstance(body, Mapping):
            parts = [
                str(body[name])
                for name in ("error", "detail")
                if body.get(name) not in (None, "")
            ]
            detail = ": ".join(parts) or raw
        else:
            detail = raw
        raise ContinuousLoopError(
            f"FORCERFT_ONLINE_HTTP_ERROR endpoint={url} "
            f"status={error.code} detail={detail}"
        ) from error
    require(isinstance(value, dict), "FORCERFT_ONLINE_SERVER_RESPONSE_INVALID")
    return value


def _wait_json(
    url: str,
    *,
    process: subprocess.Popen[Any],
    timeout: float,
    log_path: Path | None = None,
    ready: Callable[[Mapping[str, Any]], bool] = lambda _value: True,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ContinuousLoopError(_server_exit_message(process, log_path))
        try:
            with urlopen(url, timeout=2.0) as response:
                value = json.loads(response.read())
            if isinstance(value, dict) and ready(value):
                return value
        except (OSError, URLError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(0.25)
    raise ContinuousLoopError(
        f"FORCERFT_ONLINE_SERVER_TIMEOUT:{last_error}:"
        f"log={log_path or 'unavailable'}"
    )


def _server_exit_message(
    process: subprocess.Popen[Any], log_path: Path | None
) -> str:
    tail: deque[str] = deque(maxlen=20)
    if log_path is not None:
        try:
            with log_path.open(encoding="utf-8", errors="replace") as stream:
                tail.extend(line.strip() for line in stream if line.strip())
        except OSError:
            pass
    reason = tail[-1][:500] if tail else "no server log output"
    return (
        f"FORCERFT_ONLINE_SERVER_EXITED:exit_code={process.poll()}:"
        f"reason={reason}:log={log_path or 'unavailable'}"
    )


def _relay_server_log(
    log_path: Path,
    process: subprocess.Popen[Any],
    pause_summaries: threading.Event,
    stop: threading.Event,
    failure_reported: threading.Event | None = None,
) -> None:
    last_summary_at: float | None = None
    poll = getattr(process, "poll", lambda: None)
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        while True:
            line = stream.readline()
            if not line:
                if poll() is not None:
                    if not stop.is_set() and failure_reported is not None:
                        print(
                            f"[online] STOP:{_server_exit_message(process, log_path)}",
                            file=sys.stderr,
                            flush=True,
                        )
                        failure_reported.set()
                    return
                if stop.wait(0.1):
                    return
                continue
            if line.startswith(("[residual-activation]", "[training-checkpoint]")):
                print(line, end="", flush=True)
                continue
            if not line.startswith(("[residual-training]", "[critic-warmup]")):
                continue
            now = time.monotonic()
            if (
                not pause_summaries.is_set()
                and (
                    last_summary_at is None
                    or now - last_summary_at >= SERVER_SUMMARY_INTERVAL_SECONDS
                )
            ):
                print(line, end="", flush=True)
                last_summary_at = now


def _stop_server(
    process: subprocess.Popen[Any], *, policy_port: int | None = None
) -> None:
    """Quiesce and save before asking the persistent server to exit."""

    if process.poll() is not None:
        return
    checkpoint = None
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
            checkpoint = report.get("quiesced_checkpoint_path")
        except (OSError, URLError, ContinuousLoopError, ValueError):
            # SIGINT remains the recovery path if the local HTTP server is gone.
            pass
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    if checkpoint is not None:
        print(f"[training-checkpoint] graceful-exit={checkpoint}")


def _start_detector_worker(
    args: argparse.Namespace,
) -> tuple[subprocess.Popen[Any], tempfile.TemporaryDirectory[str], Path]:
    directory = tempfile.TemporaryDirectory(prefix="forcerft-reward-worker-")
    socket_path = Path(directory.name) / "detector.sock"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    with args.detector_log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([
            shutil.which("conda") or "conda", "run", "--no-capture-output",
            "-n", "conrft_reward", "python",
            str(ROOT / "tools/run_forcerft_production_bridge.py"),
            "--task-id", args.task_id, "--output-root", str(args.output_root),
            "--detector-worker-socket", str(socket_path), "--serve-detector-worker",
        ], cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + args.server_start_timeout
    while not socket_path.exists():
        if process.poll() is not None:
            directory.cleanup()
            raise ContinuousLoopError(
                "FORCERFT_REWARD_DETECTOR_WORKER_EXITED:"
                f"log={args.detector_log_path}"
            )
        if time.monotonic() >= deadline:
            process.terminate()
            process.wait(timeout=10)
            directory.cleanup()
            raise ContinuousLoopError(
                "FORCERFT_REWARD_DETECTOR_WORKER_TIMEOUT:"
                f"log={args.detector_log_path}"
            )
        time.sleep(0.1)
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


def _next_capture_index(capture_output_root: Path) -> int:
    indices = [
        int(path.name)
        for path in capture_output_root.iterdir()
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


def _run_episode(
    args: argparse.Namespace,
    index: int,
    *,
    server: subprocess.Popen[Any],
    model_revision: str | None = None,
    policy_epoch: int | None = None,
    pause_summaries: threading.Event | None = None,
) -> bool | None:
    root = (args.capture_output_root / f"{index:03d}").resolve()
    session_id = f"{args.capture_output_root.name}_{index:03d}"
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
            "frozen_base_policy_checkpoint", metadata.get("active_actor_checkpoint", "")
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
            log_path=getattr(args, "server_log_path", None),
        )
        require(
            status.get("learner_worker_state") != "failed"
            and status.get("current_episode_sampled") is False
            and status.get("server_persistent") is True,
            "FORCERFT_ONLINE_LEARNER_INVALID",
        )
        probe = status.get("newest_candidate_probe", {})
        proposal_translation = probe.get(
            "proposal_translation_mm_mean_p95_max", [None, None, None]
        )
        proposal_rpy = probe.get(
            "proposal_rpy_deg_mean_p95_max", [None, None, None]
        )
        base_age = status.get(
            "pre_takeover_base_age_s_p50_p95_max", [0.0, 0.0, 0.0]
        )
        print(
            "[learner] "
            f"td={status.get('unique_td_rows', 0)} "
            f"episodes={status.get('distinct_td_episodes', 0)} "
            f"cycles={status.get('completed_cycles', 0)}/"
            f"{status.get('allowed_cycles', 0)} "
            f"in_flight={status.get('in_flight_cycle')} "
            f"available={status.get('available_cycles', 0)} "
            f"warmup_q={status.get('warmup_twin_q_optimizer_steps', 0)} "
            f"joint_q={status.get('joint_twin_q_optimizer_steps', 0)} "
            f"actor={status.get('residual_actor_update_attempts', 0)}/"
            f"{status.get('residual_actor_optimizer_steps', 0)}/"
            f"{status.get('residual_actor_updates_skipped_no_gradient', 0)} "
            f"draws={status.get('actual_q_sample_draws', 0)}/"
            f"{status.get('policy_sample_draws', 0)}/"
            f"{status.get('human_sample_draws', 0)} "
            f"pinned={status.get('active_actor_revision')}/"
            f"{status.get('active_actor_online_cycle')} "
            f"newest_candidate={status.get('newest_candidate_cycle')}/"
            f"{status.get('newest_candidate_activation_eligible')} "
            f"proposal_mm_p95={proposal_translation[1]} "
            f"proposal_deg_p95={proposal_rpy[1]} "
            f"human_projection={status.get('human_projection_row_fraction', 0.0):.3f}/"
            f"{status.get('human_projection_axis_fraction', 0.0):.3f} "
            f"base_age_s_p95={base_age[1]}"
        )
    except (EpisodeLocalTransientError, CaptureOperatorExit) as error:
        _discard_unsealed_capture(root)
        status = _wait_json(
            f"http://127.0.0.1:{args.policy_port}/runtime/status",
            process=server,
            timeout=10.0,
            log_path=getattr(args, "server_log_path", None),
        )
        require(
            status.get("runtime_session_id") == session_id
            and status.get("runtime_episode_id") == EPISODE_ID
            and status.get("episode_active") is False
            and status.get("learner_worker_state") != "failed"
            and status.get("current_episode_sampled") is False
            and status.get("server_persistent") is True,
            "FORCERFT_ONLINE_REJECTED_CAPTURE_RUNTIME_INVALID",
        )
        if isinstance(error, CaptureOperatorExit):
            print(
                f"[episode] capture exited session={session_id}; "
                "replay_written=0"
            )
            return False
        reason = (
            "operator discard"
            if isinstance(error, CaptureDiscardedError)
            else "episode timeout"
            if isinstance(error, CaptureTimedOutError)
            else "episode-local transient capture failure"
        )
        print(
            f"[episode] capture rejected session={session_id}; "
            f"reason={reason}; "
            "replay_written=0; learner continues"
        )
        return None
    except ContinuousLoopError:
        _discard_unsealed_capture(root)
        raise
    except (OSError, KeyboardInterrupt):
        _discard_unsealed_capture(root)
        raise
    if pause_summaries is not None:
        pause_summaries.set()
    try:
        outcome = input("operator_task_outcome [success/failure/q]: ").strip().lower()
    finally:
        if pause_summaries is not None:
            pause_summaries.clear()
    require(outcome in {"success", "failure", "q"}, "FORCERFT_ONLINE_OPERATOR_OUTCOME_INVALID")
    if outcome == "q":
        checkpoint = _post_json(
            f"http://127.0.0.1:{args.policy_port}/runtime/operator-q-checkpoint",
            identity,
        ).get("operator_q_checkpoint_path")
        print(f"[training-checkpoint] operator-q={checkpoint or 'none'}")
        return False
    admission = _finish_episode(
        args,
        episode=root / "episodes" / EPISODE_ID,
        outcome=outcome,
        actor_checkpoint=actor_checkpoint,
    )
    if admission is None:
        _post_json(
            f"http://127.0.0.1:{args.policy_port}"
            "/runtime/resolve-rejected-admission",
            {
                "session_id": identity["session_id"],
                "episode_id": identity["episode_id"],
            },
        )
        print(f"[episode] rejected session={session_id}; continuing with next capture")
        return None
    _notify_admission_committed(
        args,
        identity=identity,
        admission=admission,
    )
    return True


def run_loop(args: argparse.Namespace) -> int:
    require(
        args.allow_development_policy_execution_smoke,
        "FORCERFT_ONLINE_ROBOT_EXECUTION_FLAG_REQUIRED",
    )
    ack_replay_root = getattr(
        args,
        "ack_replay_root",
        args.output_root
        / ONLINE_ADAPTATION_DIRECTORY_NAME
        / "formal_replay",
    )
    resume = (
        getattr(args, "learner_resume_checkpoint", None).resolve()
        if getattr(args, "learner_resume_checkpoint", None) is not None
        else select_resume_or_bootstrap_checkpoint(
            args.output_root,
            configured_bootstrap_checkpoint=getattr(
                args, "online_residual_bootstrap_checkpoint", None
            ),
        ).path
    )
    if schedule_migration_required(
        load_checkpoint_training_config(resume),
        load_common_actor_critic_config(args.task_id),
    ):
        raise ContinuousLoopError(
            f"FORCERFT_SCHEDULE_MIGRATION_REQUIRED:{resume}"
        )
    server_command = [
        str(args.model_python), str(ROOT / "tools/serve_forcerft_residual_actor_critic.py"),
        "--task-id", args.task_id, "--output-root", str(args.output_root),
        "--task", args.task,
        "--dataset-root", str(args.dataset_root),
        "--ack-replay-root", str(ack_replay_root),
        "--session-id", "waiting-for-episode", "--episode-id", EPISODE_ID,
        "--learner-resume-checkpoint", str(resume),
        "--allow-development-policy-execution-smoke",
        "--host", "127.0.0.1", "--port", str(args.policy_port),
    ]
    if args.safety_config is not None:
        server_command.extend(["--safety-config", str(args.safety_config)])
    log_root = (
        args.output_root / ONLINE_ADAPTATION_DIRECTORY_NAME / "runtime_logs"
    )
    log_root.mkdir(parents=True, exist_ok=True)
    run_id = str(time.time_ns())
    args.server_log_path = log_root / f"server_{run_id}.log"
    args.detector_log_path = log_root / f"detector_{run_id}.log"
    server_environment = os.environ.copy()
    server_environment["PYTHONUNBUFFERED"] = "1"
    with args.server_log_path.open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            server_command,
            cwd=ROOT,
            env=server_environment,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
    pause_summaries = threading.Event()
    relay_stop = threading.Event()
    args.server_failure_reported = threading.Event()
    relay = threading.Thread(
        target=_relay_server_log,
        args=(
            args.server_log_path,
            server,
            pause_summaries,
            relay_stop,
            args.server_failure_reported,
        ),
        daemon=True,
    )
    relay.start()
    detector_process = detector_directory = detector_socket = None
    completed = 0
    try:
        metadata = _wait_json(
            f"http://127.0.0.1:{args.policy_port}/metadata",
            process=server,
            timeout=args.server_start_timeout,
            log_path=args.server_log_path,
            ready=lambda value: value.get("server_persistent") is True,
        )
        require(
            metadata.get("learner_resume_checkpoint") == str(resume.resolve()),
            "FORCERFT_ONLINE_RESUME_CHECKPOINT_MISMATCH",
        )
        frozen_base_policy_checkpoint = str(
            metadata.get(
                "frozen_base_policy_checkpoint",
                metadata.get("active_actor_checkpoint", ""),
            )
        )
        args.deployed_actor_checkpoint = Path(frozen_base_policy_checkpoint).resolve()
        require(
            str(metadata.get("active_actor_model_revision", ""))
            and int(metadata.get("policy_epoch", -1)) >= 0
            and frozen_base_policy_checkpoint
            and args.deployed_actor_checkpoint.is_dir(),
            "FORCERFT_ONLINE_SERVER_METADATA_INVALID",
        )
        detector_process, detector_directory, detector_socket = (
            _start_detector_worker(args)
        )
        args.detector_worker_socket = detector_socket
        args.capture_output_root.mkdir(parents=True, exist_ok=True)
        index = _next_capture_index(args.capture_output_root)
        while completed < args.max_episodes:
            result = _run_episode(
                args,
                index,
                server=server,
                pause_summaries=pause_summaries,
            )
            index += 1
            if result is False:
                break
            if result is True:
                completed += 1
    except Exception as error:
        if getattr(server, "poll", lambda: None)() is not None:
            raise ContinuousLoopError(
                _server_exit_message(server, args.server_log_path)
            ) from error
        raise
    finally:
        if detector_process is not None and detector_socket is not None:
            _stop_detector_worker(detector_process, detector_socket)
        if detector_directory is not None:
            detector_directory.cleanup()
        relay_stop.set()
        relay.join(timeout=2.0)
        _stop_server(server, policy_port=args.policy_port)
    return completed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="task2")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--safety-config", type=Path)
    parser.add_argument("--online-residual-bootstrap-checkpoint", type=Path)
    parser.add_argument(
        "--learner-resume-checkpoint",
        type=Path,
        help="explicit full checkpoint, including a reviewed schedule migration result",
    )
    parser.add_argument("--max-episodes", type=int, required=True)
    parser.add_argument("--capture-output-root", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-time", type=float, default=60.0)
    parser.add_argument("--tool-profile", default="onrobot_robotiq")
    parser.add_argument("--policy-replan-steps", type=int, default=8)
    parser.add_argument("--policy-queue-low-watermark", type=int, default=7)
    parser.add_argument("--max-force-n", type=float, default=25.0)
    parser.add_argument("--max-torque-nm", type=float, default=2.0)
    parser.add_argument("--ack-replay-root", type=Path)
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

    args.capture_output_root = (
        ROOT / "datasets" / f"{args.task_id}_forcerft_online"
        if args.capture_output_root is None
        else args.capture_output_root
    ).resolve()
    args.output_root = resolve_task_output_root(
        ROOT, task_id=args.task_id, output_root=args.output_root
    )
    args.dataset_root = resolve_task_dataset_root(
        ROOT, task_id=args.task_id, dataset_root=args.dataset_root
    )
    if args.safety_config is not None:
        args.safety_config = args.safety_config.resolve()
    args.ack_replay_root = (
        args.output_root
        / ONLINE_ADAPTATION_DIRECTORY_NAME
        / "formal_replay"
        if args.ack_replay_root is None else args.ack_replay_root.resolve()
    )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        completed = run_loop(args)
    except (ContinuousLoopError, OSError) as error:
        failure_reported = getattr(args, "server_failure_reported", None)
        if failure_reported is not None and failure_reported.is_set():
            return 2
        log_path = getattr(args, "server_log_path", None)
        suffix = (
            ""
            if log_path is None or f"log={log_path}" in str(error)
            else f":log={log_path}"
        )
        print(f"[online] STOP:{error}{suffix}", file=sys.stderr)
        return 2
    print(f"[online] complete episodes={completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
