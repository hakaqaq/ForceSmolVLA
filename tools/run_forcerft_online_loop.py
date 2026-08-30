#!/usr/bin/env python3
"""Coordinate the existing Stage-3 Actor/Learner episode-boundary loop."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
FORMAL_R_ROOT = (
    ROOT
    / "artifacts/development/stage3/formal_online_r"
    / "task2_policy_execute_stage3_cycle210_smoke_20260829_001"
)
REGISTRY = (
    ROOT
    / "artifacts/development/stage3/runtime/stage3_policy_revision_registry.json"
)
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
        raise ContinuousLoopError(f"STAGE3_CONTINUOUS_JSON_INVALID:{path}") from error
    require(isinstance(value, dict), f"STAGE3_CONTINUOUS_JSON_OBJECT_REQUIRED:{path}")
    return value


@dataclass(frozen=True)
class ActiveRevision:
    revision_id: str
    model_revision: str
    policy_epoch: int


@dataclass(frozen=True)
class Deployment:
    profile: Path
    binding: Path
    trusted_binding: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    model_revision: str
    checkpoint: Path
    package: Path
    profile: Path
    binding: Path


def read_active_revision(registry: Path) -> ActiveRevision:
    state = _json(registry).get("state")
    require(isinstance(state, Mapping), "STAGE3_CONTINUOUS_REGISTRY_STATE_INVALID")
    active_id = str(state.get("active_revision_id", ""))
    records = state.get("records")
    require(active_id and isinstance(records, list), "STAGE3_CONTINUOUS_ACTIVE_REVISION_MISSING")
    active = next(
        (
            item for item in records
            if isinstance(item, Mapping)
            and item.get("revision_id") == active_id
            and item.get("state") == "active"
        ),
        None,
    )
    require(active is not None, "STAGE3_CONTINUOUS_ACTIVE_RECORD_MISSING")
    model_revision = str(active.get("model_sha256", ""))
    epoch = state.get("policy_epoch")
    require(model_revision and isinstance(epoch, int), "STAGE3_CONTINUOUS_ACTIVE_IDENTITY_INVALID")
    return ActiveRevision(active_id, model_revision, epoch)


def discover_active_deployment(active: ActiveRevision) -> Deployment:
    matches: list[Deployment] = []
    for profile_path in (ROOT / "configs").glob("deployment.*.development.json"):
        try:
            profile = _json(profile_path)
            checkpoint = Path(str(profile["checkpoint"]))
            binding = Path(str(profile["deployment_binding"]))
            if not checkpoint.is_absolute():
                checkpoint = ROOT / checkpoint
            if not binding.is_absolute():
                binding = ROOT / binding
            candidate = _json(checkpoint / "candidate.json")
        except (ContinuousLoopError, KeyError):
            continue
        if (
            profile.get("artifact_status") == "development_only"
            and candidate.get("state") == "published"
            and candidate.get("published") is True
            and candidate.get("revision_id") == active.revision_id
            and candidate.get("model_revision") == active.model_revision
            and binding.is_file()
        ):
            matches.append(
                Deployment(
                    profile=profile_path.resolve(),
                    binding=binding.resolve(),
                    trusted_binding=str(profile.get("deployment_binding_sha256", "")),
                )
            )
    require(len(matches) == 1, "STAGE3_CONTINUOUS_ACTIVE_DEPLOYMENT_NOT_UNIQUE")
    require(matches[0].trusted_binding, "STAGE3_CONTINUOUS_TRUSTED_BINDING_MISSING")
    return matches[0]


def discover_checkpoint_for_revision(
    formal_r_root: Path, revision_id: str
) -> Path:
    matches: list[Path] = []
    for metadata_path in (formal_r_root / "checkpoints").glob("*/metadata.json"):
        metadata = _json(metadata_path)
        revision = metadata.get("candidate_policy_revision")
        if (
            metadata.get("complete") is True
            and isinstance(revision, Mapping)
            and revision.get("revision_id") == revision_id
        ):
            matches.append(metadata_path.parent.resolve())
    require(len(matches) == 1, "STAGE3_CONTINUOUS_ACTIVE_CHECKPOINT_NOT_UNIQUE")
    return matches[0]


def _checkpoint_identity(checkpoint: Path) -> tuple[str, int]:
    metadata = _json(checkpoint / "metadata.json")
    revision = metadata.get("candidate_policy_revision")
    cycle = metadata.get("joint_cycles")
    require(
        metadata.get("complete") is True
        and isinstance(revision, Mapping)
        and isinstance(revision.get("revision_id"), str)
        and isinstance(cycle, int),
        "STAGE3_CONTINUOUS_PENDING_CHECKPOINT_INVALID",
    )
    return str(revision["revision_id"]), cycle


def _slug(value: str) -> str:
    result = "".join(character if character.isalnum() else "_" for character in value)
    return result.strip("_")


def candidate_artifacts(checkpoint: Path) -> Candidate:
    candidate_id, _cycle = _checkpoint_identity(checkpoint)
    slug = _slug(candidate_id)
    return Candidate(
        candidate_id=candidate_id,
        model_revision="",
        checkpoint=checkpoint.resolve(),
        package=(ROOT / "artifacts/development/stage3/published" / f"{slug}.v1"),
        profile=(ROOT / "configs" / f"deployment.{slug}.development.json"),
        binding=(ROOT / "artifacts/development/live" / f"task2_{slug}_deployment_binding.v1.json"),
    )


def _episode_seal(root: Path) -> Path:
    return (
        root / "integrated_capture" / EPISODE_ID
        / "streams/policy_execute_episode_seal.json"
    )


def _episode_dir(root: Path) -> Path:
    return root / "episodes" / EPISODE_ID


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
    require(result.returncode == 0, f"STAGE3_CONTINUOUS_COMMAND_FAILED:{command[1]}")
    return result


def _report(command: list[str]) -> dict[str, Any]:
    output = _run(command, capture=True).stdout
    start = output.find("{")
    require(start >= 0, "STAGE3_CONTINUOUS_COMMAND_REPORT_MISSING")
    try:
        value = json.loads(output[start:])
    except json.JSONDecodeError as error:
        raise ContinuousLoopError("STAGE3_CONTINUOUS_COMMAND_REPORT_INVALID") from error
    require(isinstance(value, dict), "STAGE3_CONTINUOUS_COMMAND_REPORT_INVALID")
    return value


def _bridge(args: argparse.Namespace, episode: Path, outcome: str) -> None:
    report = _report([
        str(args.model_python), str(ROOT / "tools/run_forcerft_production_bridge.py"),
        "--episode", str(episode), "--operator-task-outcome", outcome, "--dry-run",
    ])
    require(
        outcome == "success"
        and report.get("status") == "DRY_RUN_READY"
        and int(report.get("quarantined_count", -1)) == 0
        and report.get("detector_outcome") == "success",
        "STAGE3_CONTINUOUS_BRIDGE_NOT_PASS",
    )


def _admit(args: argparse.Namespace, episode: Path) -> None:
    report = _report([
        str(args.model_python), str(ROOT / "tools/run_forcerft_production_bridge.py"),
        "--episode", str(episode), "--state-root", str(args.formal_r_root),
        "--operator-task-outcome", "success", "--admit-formal-online-r",
    ])
    require(
        report.get("status") == "FORMAL_ONLINE_R_ADMITTED",
        "STAGE3_CONTINUOUS_ADMISSION_FAILED",
    )


def _publish(args: argparse.Namespace, checkpoint: Path) -> Candidate:
    candidate = candidate_artifacts(checkpoint)
    require(
        not any(path.exists() for path in (candidate.package, candidate.profile, candidate.binding)),
        "STAGE3_CONTINUOUS_CANDIDATE_OUTPUT_EXISTS",
    )
    _run([
        str(args.model_python), str(ROOT / "tools/export_forcerft_candidate.py"),
        "--joint-checkpoint", str(checkpoint),
        "--destination", str(candidate.package),
        "--deployment-profile", str(candidate.profile),
        "--deployment-binding", str(candidate.binding),
        "--candidate-revision-id", candidate.candidate_id,
        "--deployment-id", f"task2-{candidate.candidate_id}",
        "--approval-id", f"forcesmolvla-{candidate.candidate_id}",
    ])
    published = _json(candidate.package / "candidate.json")
    require(
        published.get("revision_id") == candidate.candidate_id
        and published.get("state") == "published"
        and published.get("published") is True
        and published.get("activated") is False
        and isinstance(published.get("model_revision"), str)
        and candidate.profile.is_file()
        and candidate.binding.is_file(),
        "STAGE3_CONTINUOUS_CANDIDATE_PUBLICATION_INVALID",
    )
    return replace(candidate, model_revision=str(published["model_revision"]))


def _activate(
    args: argparse.Namespace,
    *,
    candidate: Candidate,
    previous_episode_seal: Path,
) -> ActiveRevision:
    before = read_active_revision(args.registry)
    witness = (
        args.registry.parent
        / f"{_slug(candidate.candidate_id)}_reset_home_quiescent.json"
    )
    require(not witness.exists(), "STAGE3_CONTINUOUS_HOME_WITNESS_EXISTS")
    _run([
        str(args.robot_python), str(ROOT / "robot/deployment/reset_home_witness.py"),
        "--output", str(witness), "--previous-episode-seal", str(previous_episode_seal),
        "--interface-timeout", "10", "--home-timeout", "30",
    ])
    _run([
        str(args.model_python), str(ROOT / "tools/activate_forcerft_policy_revision.py"),
        "activate", "--registry", str(args.registry), "--home-witness", str(witness),
        "--candidate-package", str(candidate.package),
        "--candidate-id", candidate.candidate_id,
        "--candidate-revision", candidate.model_revision,
        "--current-active-revision", before.revision_id,
    ])
    after = read_active_revision(args.registry)
    require(
        after.revision_id == candidate.candidate_id
        and after.model_revision == candidate.model_revision
        and after.policy_epoch == before.policy_epoch + 1,
        "STAGE3_CONTINUOUS_BOUNDARY_ACTIVATION_FAILED",
    )
    return after


def _bootstrap(args: argparse.Namespace) -> None:
    if args.bootstrap_episode is None and args.bootstrap_checkpoint is None:
        return
    require(
        args.bootstrap_episode is not None and args.bootstrap_checkpoint is not None,
        "STAGE3_CONTINUOUS_BOOTSTRAP_ARGUMENTS_INCOMPLETE",
    )
    episode = args.bootstrap_episode.resolve()
    checkpoint = args.bootstrap_checkpoint.resolve()
    require(episode.is_dir(), "STAGE3_CONTINUOUS_BOOTSTRAP_EPISODE_MISSING")
    _admit(args, episode)
    candidate = _publish(args, checkpoint)
    _activate(
        args,
        candidate=candidate,
        previous_episode_seal=_episode_seal(episode.parent.parent),
    )


def _finish_episode(
    args: argparse.Namespace,
    *,
    episode: Path,
    episode_seal: Path,
    pending: Path,
    outcome: str,
) -> None:
    _bridge(args, episode, outcome)
    _admit(args, episode)
    candidate = _publish(args, pending)
    _activate(args, candidate=candidate, previous_episode_seal=episode_seal)


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
        require(process.poll() is None, "STAGE3_CONTINUOUS_SERVER_EXITED")
        try:
            with urlopen(url, timeout=2.0) as response:
                value = json.loads(response.read())
            if isinstance(value, dict) and ready(value):
                return value
        except (OSError, URLError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(0.25)
    raise ContinuousLoopError(f"STAGE3_CONTINUOUS_SERVER_TIMEOUT:{last_error}")


def _stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _episode_plan(
    args: argparse.Namespace, index: int
) -> tuple[Path, str, ActiveRevision, Deployment, Path, Path, str]:
    active = read_active_revision(args.registry)
    deployment = discover_active_deployment(active)
    resume = discover_checkpoint_for_revision(args.formal_r_root, active.revision_id)
    _current_id, cycle = _checkpoint_identity(resume)
    root = Path(f"{args.root_prefix}_{index:03d}").resolve()
    session_id = root.name
    pending_id = (
        f"stage3-online-r-real-async-joint-cycle-{cycle + 1:06d}"
        f"-pending-{session_id}"
    )
    pending = (
        args.formal_r_root / "checkpoints"
        / f"stage3_real_async_joint_cycle_{cycle + 1:06d}_pending_{session_id}"
    ).resolve()
    require(not root.exists(), "STAGE3_CONTINUOUS_CAPTURE_ROOT_EXISTS")
    require(not pending.exists(), "STAGE3_CONTINUOUS_PENDING_CHECKPOINT_EXISTS")
    return root, session_id, active, deployment, resume, pending, pending_id


def _validate_capture(
    *,
    root: Path,
    active: ActiveRevision,
    resume: Path,
    pending: Path,
    pending_id: str,
) -> None:
    seal = _json(_episode_seal(root))
    pending_revision, _cycle = _checkpoint_identity(pending)
    require(
        seal.get("technical_seal") == "complete"
        and seal.get("active_actor_revision") == active.revision_id
        and seal.get("active_actor_model_revision") == active.model_revision
        and seal.get("learner_resume_checkpoint") == str(resume)
        and seal.get("current_episode_sampled_by_learner") is False
        and int(seal.get("learner_critic_steps", -1)) == 2
        and int(seal.get("learner_actor_steps", -1)) == 1
        and seal.get("pending_checkpoint_path") == str(pending)
        and seal.get("pending_candidate_id") == pending_id
        and seal.get("pending_candidate_published") is False
        and seal.get("pending_candidate_activated") is False
        and pending_revision == pending_id,
        "STAGE3_CONTINUOUS_CAPTURE_SEAL_INVALID",
    )


def _run_episode(args: argparse.Namespace, index: int) -> None:
    root, session_id, active, deployment, resume, pending, pending_id = _episode_plan(
        args, index
    )
    server_command = [
        str(args.model_python), str(ROOT / "tools/serve_forcerft_actor_learner.py"),
        "--deployment-profile", str(deployment.profile),
        "--deployment-binding", str(deployment.binding),
        "--trusted-deployment-binding-sha256", deployment.trusted_binding,
        "--session-id", session_id, "--episode-id", EPISODE_ID,
        "--learner-resume-checkpoint", str(resume),
        "--pending-checkpoint", str(pending),
        "--pending-candidate-id", pending_id,
        "--host", "127.0.0.1", "--port", str(args.policy_port),
        "--allow-development-robot-execution",
    ]
    server = subprocess.Popen(server_command, cwd=ROOT, env=os.environ.copy())
    try:
        metadata = _wait_json(
            f"http://127.0.0.1:{args.policy_port}/metadata",
            process=server,
            timeout=args.server_start_timeout,
        )
        require(
            metadata.get("runtime_session_id") == session_id
            and metadata.get("runtime_episode_id") == EPISODE_ID
            and metadata.get("active_actor_revision") == active.revision_id
            and metadata.get("active_actor_model_revision") == active.model_revision
            and metadata.get("learner_resume_checkpoint") == str(resume)
            and metadata.get("pending_candidate_id") == pending_id,
            "STAGE3_CONTINUOUS_SERVER_IDENTITY_MISMATCH",
        )
        _run([
            str(args.robot_python), str(ROOT / "tools/run_forcerft_integrated_capture.py"),
            "--mode", "policy-execute", "--allow-development-policy-execution-smoke",
            "--async-learner", "--root", str(root), "--task", args.task,
            "--episodes", "1", "--episode-time", str(args.episode_time),
            "--tool-profile", args.tool_profile, "--session-id", session_id,
            "--episode-id", EPISODE_ID, "--policy-revision", active.model_revision,
            "--policy-epoch", str(active.policy_epoch), "--takeover-generation", "0",
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
            timeout=args.learner_finish_timeout,
            ready=lambda value: value.get("learner_state") in {"complete", "failed"},
        )
        require(
            status.get("learner_state") == "complete"
            and status.get("current_episode_sampled") is False,
            "STAGE3_CONTINUOUS_LEARNER_NOT_COMPLETE",
        )
        _validate_capture(
            root=root,
            active=active,
            resume=resume,
            pending=pending,
            pending_id=pending_id,
        )
    finally:
        _stop_server(server)

    outcome = input("operator_task_outcome [success/failure]: ").strip().lower()
    require(outcome in {"success", "failure"}, "STAGE3_CONTINUOUS_OPERATOR_OUTCOME_INVALID")
    _finish_episode(
        args,
        episode=_episode_dir(root),
        episode_seal=_episode_seal(root),
        pending=pending,
        outcome=outcome,
    )


def run_loop(args: argparse.Namespace) -> None:
    _bootstrap(args)
    for index in range(1, args.max_episodes + 1):
        _run_episode(args, index)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-episodes", type=int, required=True)
    parser.add_argument("--root-prefix", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-time", type=float, default=60.0)
    parser.add_argument("--tool-profile", default="onrobot_robotiq")
    parser.add_argument("--policy-replan-steps", type=int, default=8)
    parser.add_argument("--policy-queue-low-watermark", type=int, default=7)
    parser.add_argument("--max-force-n", type=float, default=25.0)
    parser.add_argument("--max-torque-nm", type=float, default=2.0)
    parser.add_argument("--bootstrap-episode", type=Path)
    parser.add_argument("--bootstrap-checkpoint", type=Path)
    parser.add_argument("--formal-r-root", type=Path, default=FORMAL_R_ROOT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--model-python", type=Path, default=MODEL_PYTHON)
    parser.add_argument("--robot-python", type=Path, default=ROBOT_PYTHON)
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--server-start-timeout", type=float, default=300.0)
    parser.add_argument("--learner-finish-timeout", type=float, default=300.0)
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
    args.root_prefix = args.root_prefix.resolve()
    args.formal_r_root = args.formal_r_root.resolve()
    args.registry = args.registry.resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_loop(args)
    except (ContinuousLoopError, OSError) as error:
        print(f"[continuous] STOP:{error}", file=sys.stderr)
        return 2
    active = read_active_revision(args.registry)
    print(
        f"[continuous] complete episodes={args.max_episodes} "
        f"active={active.revision_id} policy_epoch={active.policy_epoch}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
