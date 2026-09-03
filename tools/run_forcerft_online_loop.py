#!/usr/bin/env python3
"""Coordinate persistent ForceRFT HIL capture and online Actor/Learner training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
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
    start = output.find("{")
    require(start >= 0, "FORCERFT_ONLINE_COMMAND_REPORT_MISSING")
    try:
        value = json.loads(output[start:])
    except json.JSONDecodeError as error:
        raise ContinuousLoopError("FORCERFT_ONLINE_COMMAND_REPORT_INVALID") from error
    require(isinstance(value, dict), "FORCERFT_ONLINE_COMMAND_REPORT_INVALID")
    return value


def _admit(args: argparse.Namespace, episode: Path, *, outcome: str) -> bool:
    """Materialize the sealed episode and append it exactly once to Online-R."""

    report = _report([
        str(args.model_python), str(ROOT / "tools/run_forcerft_production_bridge.py"),
        "--task-id", args.task_id, "--output-root", str(args.output_root),
        "--episode", str(episode), "--state-root", str(args.formal_r_root),
        "--deployed-actor-checkpoint", str(args.deployed_actor_checkpoint),
        "--operator-task-outcome", outcome, "--admit-formal-online-r",
    ])
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
        f"training_started={str(bool(report.get('training_starts_reached'))).lower()}"
    )
    return True


def _finish_episode(
    args: argparse.Namespace,
    *,
    episode: Path,
    outcome: str,
) -> bool:
    require(
        outcome in {"success", "failure"},
        "FORCERFT_ONLINE_OPERATOR_OUTCOME_INVALID",
    )
    return _admit(args, episode, outcome=outcome)


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
    """Let the server finish its current optimizer step without saving."""

    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=60)
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
    model_revision: str,
    policy_epoch: int,
) -> bool | None:
    root = (args.root_prefix / f"{index:03d}").resolve()
    session_id = f"{args.root_prefix.name}_{index:03d}"
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
        args, episode=root / "episodes" / EPISODE_ID, outcome=outcome,
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
        allow_legacy_offline_fallback=getattr(
            args, "allow_legacy_offline_fallback", False
        ),
    ).path
    args.deployed_actor_checkpoint = (resume / "actor").resolve()
    server_command = [
        str(args.model_python), str(ROOT / "tools/serve_forcerft_actor_learner.py"),
        "--task-id", args.task_id, "--output-root", str(args.output_root),
        "--task", args.task,
        "--dataset-root", str(args.dataset_root),
        "--reward-transition-root", str(args.reward_transition_root),
        "--session-id", "waiting-for-episode", "--episode-id", EPISODE_ID,
        "--learner-resume-checkpoint", str(resume),
        "--allow-development-policy-execution-smoke",
        "--host", "127.0.0.1", "--port", str(args.policy_port),
    ]
    if args.safety_config is not None:
        server_command.extend(["--safety-config", str(args.safety_config)])
    if getattr(args, "sft_reference_checkpoint", None) is not None:
        server_command.extend(
            [
                "--sft-reference-checkpoint",
                str(args.sft_reference_checkpoint),
            ]
        )
    if getattr(args, "actor_readiness_manifest", None) is not None:
        server_command.extend(
            [
                "--actor-readiness-manifest",
                str(args.actor_readiness_manifest),
            ]
        )
    if getattr(args, "actor_readiness_mode", None) is not None:
        server_command.extend(
            ["--actor-readiness-mode", args.actor_readiness_mode]
        )
    if getattr(args, "allow_legacy_offline_fallback", False):
        server_command.append("--allow-legacy-offline-fallback")
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
        args.root_prefix.mkdir(parents=True, exist_ok=True)
        index = _next_capture_index(args.root_prefix)
        while completed < args.max_episodes:
            result = _run_episode(
                args,
                index,
                server=server,
                model_revision=model_revision,
                policy_epoch=policy_epoch,
            )
            index += 1
            if result is False:
                break
            if result is True:
                completed += 1
    finally:
        _stop_server(server)
    return completed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="task2")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--reward-transition-root", type=Path)
    parser.add_argument("--safety-config", type=Path)
    parser.add_argument("--stage3-seed-bundle", type=Path)
    parser.add_argument("--sft-reference-checkpoint", type=Path)
    parser.add_argument("--actor-readiness-manifest", type=Path)
    parser.add_argument(
        "--actor-readiness-mode",
        choices=("manual_approval", "automatic_readiness"),
    )
    parser.add_argument("--allow-legacy-offline-fallback", action="store_true")
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
        resolve_task_reward_transition_root,
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
    args.reward_transition_root = resolve_task_reward_transition_root(
        ROOT,
        task_id=args.task_id,
        reward_transition_root=args.reward_transition_root,
    )
    if args.safety_config is not None:
        args.safety_config = args.safety_config.resolve()
    if args.sft_reference_checkpoint is not None:
        args.sft_reference_checkpoint = args.sft_reference_checkpoint.resolve()
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
